###############################################################################
# simulations_S2.1_HMGlasso.R
# Simulation study for SM §§S2.1 (HMGlasso version):
# - Ability to correctly retrieve the edge set associated to each hidden state for the HMGlasso by generating data as in 
#   Chun et al. (2016) with K=1 hidden state

#
# What this script does:
#   1) Ensures dependencies (optional auto-install) + clean sourcing
#   2) Configures optional parallel backend
#   3) Generates synthetic data as Chun via Ygen_Chun (in MainFunctions)
#      (time-varying graphs controlled by alpha) per scenario
#   4) Runs EM for HMGlasso over a lambda grid with multi-start
#   5) Picks best start by log-likelihood;
#   6) Builds ROC summaries (TPR/FPR per state; means across states)
#   7) Saves a single list object `out_sim` containing everything needed for §§S2.1
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



## --------------------------- SOURCE FILES --------------------------------- ##
if (!file.exists("em_glasso.R"))    stop("Required file not found: em_glasso.R")
if (!file.exists("MainFunctions.R")) stop("Required file not found: MainFunctions.R")
source("em_glasso.R")
source("MainFunctions.R")

## --------------------------- PARALLEL BACKEND ----------------------------- ##
cl <- NULL
if (use_parallel) {
  ncores <- parallel::detectCores()
  cl <- parallel::makeCluster(max(1, ncores - 2))
  doParallel::registerDoParallel(cl)
  invisible(parallel::clusterEvalQ(
    cl, c(source("em_glasso.R"),
          source("MainFunctions.R"),
          library(LQGM), library(ald), library(quantreg), library(mclust),
          library(skewt), library(cluster), library(markovchain),
          library(rqPen), library(glasso),
          library(lqmm), library(stats), library(car),
          library(mvtnorm), library(MASS))
  ))
  parallel::clusterSetRNGStream(cl, 8693)
}


## --------------------------- DESIGN CHOICES ------------------------------- ##
N            <- c(300)
K            <- c(1, 2, 3)     # number of states
R            <- 1           # restarts per lambda
MM           <- 100         # Monte Carlo simulations
d            <- c(10)
nlambda      <- 50
distribution <- c("n")
settings     <- c("chun")


## --------------------------- SETUP ----------------------------- ##
n    <- N[1]
k    <- K[1]
p    <- d[1]
dist <- distribution[1]
setting <- settings[1]


## --------------------------- Sanity check ------------------------------ ##
# Sanity check for Scenario 2 generator
if (setting == "chun" && !exists("Ygen_Chun")) {
  stop("Scenario 'chun' selected but function 'Ygen_Chun' is not available in ", "MainFunctions.R")
}

## --------------------------- TRUE STRUCTURES ------------------------------ ##
mu     <- matrix(NA, k, p, byrow = TRUE)
Omega  <- replicate(k, matrix(NA, p, p))
Sigma  <- A <- replicate(k, matrix(NA, p, p), simplify = FALSE)
MC.sim <- list()


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

emissionprobs <- replicate(MM, list())
Graph.grid.info <- replicate(k, array(NA, dim = c(MM, nlambda, 8)), simplify = FALSE)
lambda.opt.index <- matrix(NA, MM, 3)

Y <- matrix(NA, n, p)
TPR_list <- FPR_list <- replicate(k, matrix(NA, MM, nlambda), simplify = FALSE)
names(TPR_list) <- names(FPR_list) <- paste("state", 1:k)

Theta.f <- A.est <- replicate(k, array(NA, dim = c(p, p, nlambda)), simplify = FALSE)
deltas <- replicate(MM, list())
iterations <- logliks <- matrix(NA, MM, nlambda)
gammas <- replicate(MM, array(NA, dim = c(k, k, nlambda)), simplify = FALSE)


## --------------------------- MONTE CARLO LOOP ----------------------------- ##
t0 <- Sys.time()

for (i in 1:MM) {
  cat("Simulation =", i, "\n")
  set.seed(i)
  
  tmp <- list()
  model  <- replicate(R, list(), simplify = FALSE)
  comparison_lqgmHM <- replicate(k, list(), simplify = FALSE)
  model_def <- replicate(nlambda, list(), simplify = FALSE)
  
  if(k != 1){
    States <- paste("k", 1:k, sep = "")
    MC <- methods::new("markovchain", states = States, byrow = TRUE, transitionMatrix = gamma)
    MC.sim <- markovchain::rmarkovchain(n = n-1, object = MC,
                                        t0 = sample(States, size = 1, prob = delta),
                                        include.t0 = TRUE, what = data.frame)
    for(j in 1:k) {
      MC.sim[MC.sim==paste("k", j, sep = "")]=j
    }
    MC.sim = as.numeric(MC.sim)
  } else {
    MC.sim = rep(1, n)
  }
  
  ## Generate Y under chosen scenario
  if(setting == "scenario1"){
    if(dist == "n"){
      mu[1,] <- rep(-.45, p) #-> overlap = 0.10
      mu[2,] <- rep(.45, p)
      for (j in 1:k) {
        Y[MC.sim == j,] = mvtnorm::rmvnorm(n = sum(MC.sim == j), mean = mu[j,], sigma = Sigma[[j]])
      }}
  } else if(setting=="chun"){
    Y <- Ygen_Chun(n, p)$Y
    A <- Ygen_Chun(n, p)$A
  }
  
  
  if (use_parallel) clusterExport(cl, varlist = ls(envir = globalenv()))
  
  
  ## Fit over lambda grid with R restarts; keep best-LL per lambda
  model_def <- parLapply(cl = if (use_parallel) cl else NULL, 1:nlambda, function(l) {
    nas <- 0
    for (r in 1:R) {
      set.seed(r)
      ## EM initialization (K-means; enforce min size of clusters)
      delta.s <- runif(k); delta.s <- delta.s / sum(delta.s)
      Sigma.s <- replicate(k, matrix(0, p, p), simplify = FALSE)
      mu.s    <- matrix(0, k, p)
      max_attempts <- 10
      attempts <- 0
      repeat {
        lk <- kmeans(Y, k)
        hmm_init <- markovchainFit(lk$cluster)
        delta.s <- rep(0, k); delta.s[lk$cluster[1]] <- 1
        gamma.s <- hmm_init$estimate@transitionMatrix
        valid_clusters <- all(sapply(1:k, function(ii) sum(lk$cluster == ii) >= 10))
        if (valid_clusters || attempts >= max_attempts) break
        attempts <- attempts + 1
      }
      for (jj in 1:k) {
        mu.s[jj, ]     <- colMeans(Y[lk$cluster == jj, ])
        Sigma.s[[jj]]  <- var(Y[lk$cluster == jj, ])
      }
      model[[r]] <- tryCatch(
        em.glasso(Y = Y, K = k, delta = delta.s, rho = lambdaseq[l],
                  gamma = gamma.s, mu = mu.s, Sigma = Sigma.s,
                  iterMax = 3e2, err = 1e-04, traceEM = TRUE),
        error = function(e) NA, silent = TRUE
      )
    } # end restarts
    
    tmp.llk <- sapply(model, function(x) as.numeric(x[4]))
    if (sum(is.na(tmp.llk)) > 0 & sum(is.na(tmp.llk)) < R) tmp.llk[which(is.na(tmp.llk))] <- +Inf
    if (sum(is.na(tmp.llk)) == R) return(NA)
    model[[which.max(tmp.llk)]]
  })
  
  
  ## Post-processing per lambda
  for (l in 1:nlambda) {
    if (is.null(model_def[[l]]) || any(is.na(model_def[[l]]))) next
    
    if (k == 2) {
      model_def[[l]]$Theta <- model_def[[l]]$Theta[order(diag(model_def[[l]]$gamma), decreasing = TRUE)]
    } else if (k == 3) {
      model_def[[l]]$Theta <- model_def[[l]]$Theta[order(rowMeans(model_def[[l]]$mu), decreasing = FALSE)]
    }
    
    ## Store outputs
    deltas[[i]][[l]]   <- model_def[[l]]$delta
    gammas[[i]][,, l]  <- model_def[[l]]$gamma
    iterations[i, l]   <- model_def[[l]]$iter
    logliks[i, l]      <- model_def[[l]]$loglik
    
    for (j in 1:k) {
      Theta.f[[j]][,, l] <- model_def[[l]]$Theta[[j]]
      A.est[[j]][,, l]   <- (Theta.f[[j]][,, l] != 0) * 1
      
      Graph.grid.info[[j]][i, l, ] <- Graph.performance(
        est.A = A.est[[j]][,, l],
        true.adj = A[[j]]
      )
      
      # Per-state comparison row for ROC summaries
      comparison <- data.frame(
        rho            = lambdaseq[l],
        falsePositives = sum(A[[j]][upper.tri(A[[j]])] == 0 &
                               Theta.f[[j]][,, l][upper.tri(Theta.f[[j]][,, l])] != 0) /
          sum(A[[j]][upper.tri(A[[j]])] == 0),
        falseNegatives = sum(A[[j]][upper.tri(A[[j]])] != 0 &
                               Theta.f[[j]][,, l][upper.tri(Theta.f[[j]][,, l])] == 0) /
          sum(A[[j]][upper.tri(A[[j]])] != 0),
        truePositives  = sum(A[[j]][upper.tri(A[[j]])] != 0 &
                               Theta.f[[j]][,, l][upper.tri(Theta.f[[j]][,, l])] != 0) /
          sum(A[[j]][upper.tri(A[[j]])] != 0)
      )
      
      # Store TPR/FPR for global summaries
      TPR_list[[j]][i, l] <- comparison$truePositives
      FPR_list[[j]][i, l] <- comparison$falsePositives
    } # end state loop
  }   # end lambda loop
  
} # end MC loop

if (use_parallel && !is.null(cl)) stopCluster(cl)
t1 <- Sys.time()
time <- t1 - t0


## --------------------------- ROC SUMMARIES -------------------------------- ##
TPR <- FPR <- matrix(NA, nrow = k, ncol = length(lambdaseq),
                     dimnames = list(paste("state", 1:k), as.character(lambdaseq)))
for (j in 1:k) {
  TPR[j, ] <- colMeans(as.matrix(TPR_list[[j]]), na.rm = TRUE)
  FPR[j, ] <- colMeans(as.matrix(FPR_list[[j]]), na.rm = TRUE)
}
TPR_final <- colMeans(TPR)
FPR_final <- colMeans(FPR)

# Optional: quick AUC check (only if bayestestR installed)
# plot(FPR_final, TPR_final, type = "l")
# bayestestR::area_under_curve(FPR_final, TPR_final)

model_out <- list(Theta.f = Theta.f, deltas = deltas, gammas = gammas,
                  iterations = iterations, logliks = logliks)

out_glasso <- list(model_out = model_out, Theta.f = Theta.f, Y = Y, MC.sim = MC.sim, A = A,
                     lambdaseq = lambdaseq, TPR = TPR_list, FPR = FPR_list,
                     TPR_final = TPR_final, FPR_final = FPR_final,
                     n = n, k = k, MM = MM, p = p, time = time, setting = setting,
                     Graph.grid.info = Graph.grid.info)

## --------------------------- SAVE OUTPUTS ---------------------------------- ##

ts_tag  <- format(Sys.time(), "%Y%m%d")
fname <- sprintf("Sim2.1_glasso_%s_N%d_p%d_K%d_MM%d_%s.RData",
                  setting, n, p, k, MM, ts_tag)
save(out_glasso, file = fname)
message("Saved: ", fname)


