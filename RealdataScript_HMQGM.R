###############################################################################
# RealdataScript_HMQGM.R
#
# Empirical application: PM2.5 in Northern Italy (2019–2022)
#
# What this script does
# ---------------------
# Goal
#   Fit the Hidden Markov Quantile Graphical Model (HMQGM) to analyze the
#   time-varying conditional dependence structure of PM2.5 concentrations
#   across 14 cities in Northern Italy, leveraging quantiles to capture
#   tail behavior and seasonal heterogeneity.
#
# Data
#   - Daily PM2.5 (µg/m^3) for 14 cities (Milan, Cremona, Mantova, Bergamo,
#     Brescia, Parma, Modena, Bologna, Padova, Venezia, Alessandria, Torino,
#     Novara, Trento).
#   - Period: 2019-01-01 to 2022-12-31 (N = 1439 observations).
#   - Loaded from: pm2.5_northIta1922.RData (column 1 = date/ID; columns 2:15 = cities).
#
# Workflow
#   1) Clean session and set options.
#   2) (Optional) Auto-install required CRAN packages; install rqPen and LQGM
#      from local tar.gz if missing.
#   3) Load all libraries and source required files:
#        - EM_lqgm.Mix_c.R, MainFunctions.R, RealdataFun.R.
#   4) Load the PM2.5 dataset and build Y (N × d) with d = 14 cities.
#   5) Define tuning/fit settings:
#        - R = 10 restarts, nlambda = 100, K ∈ {1,2,3,4},
#          EM tolerance err = 1e-04, max iterations = 2e2.
#   6) For each K in {1,2,3,4}:
#        - Run realdata() to fit HMQGM over the lambda grid with R restarts.
#        - The function selects models by AIC/BIC/ICL and returns estimates
#          (betas/adjacencies), selected lambdas, and timing.
#        - Save results to: HMQGMrealdata_<K>K.RData.
#
# Output
#   - One .RData file per K containing selected models (AIC/BIC/ICL), adjacency
#     structures, selected lambdas, and runtime information for downstream
#     tables/figures.
#
# Notes
#   - Parallelization is enabled by default (use_parallel = TRUE).
#   - rqPen (>= 3.2.1) and LQGM are installed from local source tarballs if needed.
#   - The script expects the three source files and the PM2.5 .RData to be present
#     in the working directory; it fails fast if any is missing.
###############################################################################



## --------------------------- CLEAN SETTINGS -------------------------------- ##
rm(list = ls(all.names = TRUE))
invisible(gc())
options(stringsAsFactors = FALSE, warn = 1)

## --------------------------- USER SETTINGS -------------------------------- ##
use_parallel <- TRUE  # enable/disable parallel backend
AUTO_INSTALL_PKGS <- TRUE #auto-install missing deps in clean envs
## --------------------------- ENV & LIBRARIES ------------------------------ ##

# Packages used:
# - foreach, doParallel, parallel: parallel loop primitives
# - ald, quantreg, lqmm, rqPen, LQGM: quantile/ALD-related
# - mclust: adjustedRandIndex
# - markovchain: HMM chain init/fit
# - MASS, mvtnorm: distributions (Gaussian draws)
# - pracma: small helpers
# - Rcpp, RcppArmadillo: backend for LQGM
# - car, glasso, skewt, cluster, copula
pkgs <- c(
  "parallel","doParallel","foreach",
  "ald","quantreg","lqmm","rqPen",
  "mclust","markovchain",
  "MASS","mvtnorm",
  "pracma",
  "Rcpp","RcppArmadillo",
  "LQGM","car","glasso","skewt","cluster","copula"
)

if (AUTO_INSTALL_PKGS) {
  
  ## 1) Install every CRAN package, exluding LQGM and rqPen (source installation)
  cran_pkgs <- setdiff(pkgs, c("LQGM", "rqPen"))
  miss <- cran_pkgs[!cran_pkgs %in% rownames(installed.packages())]
  if (length(miss)) {
    message("Installing missing CRAN packages: ", paste(miss, collapse = ", "))
    install.packages(miss, repos = "https://cloud.r-project.org")
  }
  
  ## 2) rqPen: install version 3.2.1 from local tar.gz
  if (!("rqPen" %in% rownames(installed.packages()))) {
    tarball <- "rqPen_3.2.1.tar.gz"   # assicurati che sia nella working directory
    if (!file.exists(tarball)) {
      stop("rqPen not installed and tarball ", tarball, " not found in working directory.")
    }
    message("Installing rqPen from local tarball: ", tarball)
    install.packages(tarball, repos = NULL, type = "source")
  }
  
  ## 3) LQGM: install from local tar.gz
  if (!("LQGM" %in% rownames(installed.packages()))) {
    tarball <- "LQGM_1.0.tar.gz"  
    if (!file.exists(tarball)) {
      stop("LQGM not installed and tarball ", tarball, " not found in working directory.")
    }
    message("Installing LQGM from local tarball: ", tarball)
    install.packages(tarball, repos = NULL, type = "source")
  }}

library(foreach)
library(parallel)
library(doParallel)
library(ald)
library(quantreg)
library(mclust)
library(skewt)
library(cluster)
library(markovchain)
library(rqPen)
library(lqmm)
library(MASS)
library(mvtnorm)
library(pracma)
library(Rcpp)
library(RcppArmadillo)
library(LQGM)
library(copula)
library(car)
library(glasso)

## --------------------------- SOURCE FILES --------------------------------------- ##
# Required source files (fail fast if missing)
if (!file.exists("EM_lqgm.Mix_c.R"))   stop("Required file not found: ", "EM_lqgm.Mix_c.R")
if (!file.exists("MainFunctions.R")) stop("Required file not found: ", "MainFunctions.R")
if (!file.exists("RealdataFun.R")) stop("Required file not found: ", "RealdataFun.R")
source("EM_lqgm.Mix_c.R")
source("MainFunctions.R")
source("RealdataFun.R")

## --------------------------- LOAD DATA --------------------------------------- ##
# Load PM2.5 data (N = 1439 days, d = 14 cities)
if (!file.exists("pm2.5_northIta1922.RData")) stop("Data file not found: pm2.5_northIta1922.RData")
load("pm2.5_northIta1922.RData")
data <- pm2.5_northIta1922

## --------------------------- SETUP ------------------------------- ##
# Data matrix Y (N x d)
N <- dim(pm2.5_northIta1922)[1] #sample size
d <- dim(pm2.5_northIta1922)[2] - 1 #dimension
R <- 10 #restart
K <- c(1,2,3,4)
nlambda <- 100
Y <- as.matrix(data[,2:15])
err <- 1e-04
iterMax <- 2e2

## --------------------------- FIT HMQGM FOR K = 1,2,3,4 ------------------------------- ##

k = K[1] #number of states
out_K1 <- realdata(Y, N, d, R, k, nlambda, iterMax, err, use_parallel)
save(out_K1, file = paste0("HMQGMrealdata_", k, "K.RData"))

k = K[2] #number of states
out_K2 <- realdata(Y, N, d, R, k, nlambda, iterMax, err, use_parallel)
save(out_K2, file = paste0("HMQGMrealdata_", k, "K.RData"))


k = K[3] #number of states
out_K3 <- realdata(Y, N, d, R, k, nlambda, iterMax, err, use_parallel)
save(out_K3, file = paste0("HMQGMrealdata_", k, "K.RData"))

k = K[4] #number of states
out_K4 <- realdata(Y, N, d, R, k, nlambda, iterMax, err, use_parallel)
save(out_K4, file = paste0("HMQGMrealdata_", k, "K.RData"))






