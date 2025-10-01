###############################################################################
# simulations_4.3_HMQGM.R
# Simulation study for manuscript §§4.3 (HMQGM version):
# - Assess the performance of three widely employed penalized likelihood criteria (AIB, BIC, ICL) 
#   for selecting the true number of hidden states K:
#     - We generate data from conditional quantile models (via Ygen1_sep) for K=1,...,4
#
# What this script does:
#   1) Ensures dependencies (optional auto-install), clean sourcing of required functions
#   2) Configures parallel backend (optional)
#   3) Generates synthetic data per chosen scenario
#   4) Runs EM for HMQGM over a lambda grid with multi-start
#   5) Selects best start by log-likelihood; 
#   6) Computes AIC/BIC/ICL and saves lower values for each simulation;
#   7) Saves a single list object `results` containing everything needed for §§4.3
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
  }
}


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
source("EM_lqgm.Mix_c.R")
source("MainFunctions.R")


## --------------------------- PARALLEL BACKEND ----------------------------- ##
cl <- NULL
if (use_parallel) {
  ncores <- parallel::detectCores()
  cl <- parallel::makeCluster(max(1, ncores - 2))
  doParallel::registerDoParallel(cl)
  invisible(parallel::clusterEvalQ(
    cl, c(source("EM_lqgm.Mix_c.R"),
          source("MainFunctions.R"),
          library(LQGM), library(ald), library(quantreg), library(mclust),
          library(skewt), library(cluster), library(markovchain),
          library(rqPen), library(car), library(glasso),
          library(lqmm), library(stats),
          library(mvtnorm), library(MASS))
  ))
  parallel::clusterSetRNGStream(cl, 8693)
}


## --------------------------- DESIGN CHOICES ------------------------------- ##
N            <- c(1000)
K            <- 4     # max number of states
R            <- 5           # restarts per lambda
MM           <- 100         # Monte Carlo simulations
d            <- c(10)
nlambda      <- 50
distribution <- c("n")
settings     <- c("scenario1", "scenario2")


## --------------------------- SETUP ----------------------------- ##
n    <- N[1]
k    <- 2 #true number of states
p    <- d[1]
dist <- distribution[1]
setting <- settings[2]


## Quantile grid (you can toggle alternatives below; default = 7 octiles)
tau_mode <- "octiles"  # choose between "median" and "octiles"
if (tau_mode == "median") {
  tauvec <- c(0.5)
} else if (tau_mode == "octiles") {
  tauvec <- seq(from = 1/8, to = 7/8, length.out = 7)
} else stop("tau_mode must be 'median' or 'octiles'")
l.tau <- length(tauvec)


## Mixture weights over quantiles (uniform by default; non-uniform option for L=7)
# if (l.tau == 7) {
#   nu_vec <- c(1,2,3,4,3,2,1); nu_vec <- nu_vec/sum(nu_vec)
# } else {
#   nu_vec <- rep(1/l.tau, l.tau)
# }
nu_vec <- rep(1/l.tau, l.tau)

## --------------------------- Sanity check ------------------------------ ##
# Sanity check for Scenario 2 generator
if (setting == "scenario2" && !exists("Ygen1_sep")) {
  stop("Scenario 'scenario2' selected but function 'Ygen1_sep' is not available in ", "MainFunctions.R")
}

## --------------------------- TRUE STRUCTURES ------------------------------ ##
mu     <- matrix(NA, k, p, byrow = TRUE)
Omega  <- replicate(k, matrix(NA, p, p))
Sigma  <- A <- replicate(k, matrix(NA, p, p), simplify = FALSE)
MC.sim <- list()

if (setting == "scenario1") {
  # Generate sparse precision matrices and their adjacencies
  Omega[,,1] <- Theta_gen(p = p, k = k)$Omega1
  Omega[,,2] <- Theta_gen(p = p, k = k)$Omega2
  if (k == 3) Omega[,,3] <- Theta_gen(p = p, k = k)$Omega3
  A <- lapply(1:k, function(j) ifelse(Omega[,,j] != 0 & row(Omega[,,j]) != col(Omega[,,j]), 1, 0))
  for (j in 1:k) Sigma[[j]] <- solve(Omega[,,j])
}


## --------------------------- LAMBDA GRIDS -------------------------------- ##
lambdaseq <- exp(seq(log(1e-05), log(1), length.out = nlambda))


## --------------------------- HMM INIT ------------------------------------- ##
if (k == 2) {
  delta <- c(1, 0)
  gamma <- matrix(c(0.9, 0.2, 0.1, 0.8), k, k)
} else if (k == 3) {
  delta <- c(0.4, 0.3, 0.3)
  gamma <- matrix(c(0.8, 0.1, 0.1,
                    0.1, 0.8, 0.1,
                    0.1, 0.1, 0.8), k, k, byrow = TRUE)
}

## --------------------------- OUTPUT HOLDERS ------------------------------- ##

bic_vec <- aic_vec <- icl_vec <- c()
aic_fin <- bic_fin <- icl_fin <- c()
# aRI_model <- matrix(NA, MM, p)
AICs <- BICs <- ICLs <- matrix(NA, nlambda, K)
Y <-  matrix(NA, n, p)



## --------------------------- MONTE CARLO LOOP ----------------------------- ##
t0 <- Sys.time()


for (i in 1:MM) {
  cat("Simulation =", i, "\n")
  set.seed(i)
  
  tmp <- list()
  model  <- replicate(R, list(), simplify = FALSE)
  comparison_lqgmHM <- replicate(k, list(), simplify = FALSE)
  model_def <- replicate(nlambda, list(), simplify = FALSE)
  
  
  if( k != 1){
    States <- paste("k", 1:k, sep = "")
    MC <- methods::new("markovchain", states = States, byrow = T, transitionMatrix = gamma)
    MC.sim <- markovchain::rmarkovchain(n = n-1, object = MC, t0 = sample(States, size = 1, prob = delta), include.t0 = T, what = data.frame)
    for(j in 1:k) {
      MC.sim[MC.sim==paste("k", j, sep = "")]=j
    }
    MC.sim = as.numeric(MC.sim)
  } else {
    MC.sim <- rep(1, n)
  }
  
  
  ## Generate Y under chosen scenario
  if(setting == "scenario1"){
    if(dist == "n"){
      mu[1,] <- rep(-.45, p)
      mu[2,] <- rep(.45, p)
      for (j in 1:k) {
        Y[MC.sim == j,] = mvtnorm::rmvnorm(n = sum(MC.sim == j), mean = mu[j,], sigma = Sigma[[j]])
      }}
  } else if(setting=="scenario2"){
    Y <- Ygen1_sep(n, k, p, MC.sim)$Y
    A <- Ygen1_sep(n, k, p, MC.sim)$A
  }

  for(kk in 1:K){
    cat("Fitting model with K =", kk, "\n")
    
    
    if (use_parallel) clusterExport(cl, varlist = ls(envir = globalenv()))
    
    ## Fit over lambda grid with R restarts; keep best-LL per lambda
    
    model_def <- parLapply(cl = if (use_parallel) cl else NULL, 1:nlambda, function(l) {
      nas <- 0
      for (r in 1:R) {
        set.seed(r)
        ## EM initialization (K-means + simple regressions)
        delta.s <- runif(kk); delta.s <- delta.s / sum(delta.s)
        beta.init <- replicate(kk, array(NA, dim = c(p, p, l.tau)), simplify = FALSE)
        sigma.s   <- replicate(kk, matrix(1, l.tau, p), simplify = FALSE)
        lk <- kmeans(Y, kk)
        hmm_init <- markovchainFit(lk$cluster)
        delta.s <- rep(0, kk); delta.s[lk$cluster[1]] <- 1
        gamma.s <- hmm_init$estimate@transitionMatrix
        if (sum(lk$cluster == 1) < 10 || sum(lk$cluster == 2) < 10) {
          lk <- kmeans(Y, kk)
          hmm_init <- markovchainFit(lk$cluster)
          delta.s <- rep(0, kk); delta.s[lk$cluster[1]] <- 1
          gamma.s <- hmm_init$estimate@transitionMatrix
        }
        lm.init <- vector("list", kk)
        for (jj in 1:kk) {
          for (pp in 1:p) {
            for (tq in 1:l.tau) {
              lm.init[[jj]] <- lm(Y[lk$cluster == jj, pp] ~ Y[lk$cluster == jj, -pp]) # with intercept
              beta.init[[jj]][pp, , tq] <- lm.init[[jj]]$coefficients + runif(1, -.5, .5)
              sigma.s[[jj]][tq, pp] <- 1
            }
          }
        }
        model[[r]] <- tryCatch(
          em.hmm.pen.lqgm(
            y = Y, K = kk, delta = delta.s, unilambda = lambdaseq[l],
            gamma = gamma.s, beta = beta.init, nu_vec = nu_vec,
            sigma = sigma.s, tauvec = tauvec, tol = 1e-04, maxiter = 2e2, traceEM = TRUE
          ),
          error = function(e) NA, silent = TRUE
        )
      } # end restarts
      
      tmp.llk <- sapply(model, function(x) as.numeric(x[5]))
      if (sum(is.na(tmp.llk)) > 0 & sum(is.na(tmp.llk)) < R) tmp.llk[which(is.na(tmp.llk))] <- +Inf
      if (sum(is.na(tmp.llk)) == R) return(NA)
      best_model_index <- which.max(tmp.llk)
      model[[best_model_index]]
    })
    
    for (l in 1:nlambda) {
      if (is.null(model_def[[l]]) || any(is.na(model_def[[l]]))) next
      
      AICs[l,kk] <- model_def[[l]]$crit$AIC
      BICs[l,kk] <- model_def[[l]]$crit$BIC
      ICLs[l,kk] <- model_def[[l]]$crit$ICL
      
    } # end lambda loop
    
  } # end K loop
  
  aic_vec <- c(AICs[which.min(AICs[,1]),1], AICs[which.min(AICs[,2]),2], AICs[which.min(AICs[,3]),3], AICs[which.min(AICs[,4]),4])
  bic_vec <- c(BICs[which.min(BICs[,1]),1], BICs[which.min(BICs[,2]),2], BICs[which.min(BICs[,3]),3], BICs[which.min(BICs[,4]),4])
  icl_vec <- c(ICLs[which.min(ICLs[,1]),1], ICLs[which.min(ICLs[,2]),2], ICLs[which.min(ICLs[,3]),3], ICLs[which.min(ICLs[,4]),4])
  
  aic_fin[i] <- which.min(aic_vec)
  bic_fin[i] <- which.min(bic_vec)
  icl_fin[i] <- which.min(icl_vec)
  
} # end MC loop

if (use_parallel && !is.null(cl)) stopCluster(cl)
t1 <- Sys.time()
time <- t1 - t0

results <- list(AIC = aic_fin, BIC = bic_fin, ICL = icl_fin, time = time)

## --------------------------- SAVE OUTPUT ---------------------------------- ##
tau_tag <- if (l.tau == 1) "tau1" else paste0("tau", l.tau)
ts_tag  <- format(Sys.time(), "%Y%m%d")
fname <- sprintf("CriteriaHMqgm_%s_%s_N%d_p%d_K1:%d_MM%d_%s.RData",
                 tau_tag, setting, n, p, K, MM, ts_tag)
save(results, file = fname)
message("Saved: ", fname)
    
    
    
    
    
    