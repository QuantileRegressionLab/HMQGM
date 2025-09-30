###############################################################################
# simulations_S2.1_HMQGM.R
# Simulation study for SM §§S2.1 (HMQGM version):
# - Ability to correctly retrieve the edge set associated to each hidden state for the HMQGM by generating data as in 
#   Chun et al. (2016) with K=1 hidden state

#
# What this script does:
#   1) Ensures dependencies (optional auto-install) + clean sourcing
#   2) Configures optional parallel backend
#   3) Generates synthetic data as Chun via Ygen_Chun (in MainFunctions)
#      (time-varying graphs controlled by alpha) per scenario
#   4) Runs EM for HMQGM over a lambda grid with multi-start
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

## Quantile grid (you can toggle alternatives below; default = 7 octiles)
tau_mode <- "octiles"  # choose between "median" and "octiles"
if (tau_mode == "median") {
  tauvec <- c(0.5)
} else if (tau_mode == "octiles") {
  tauvec <- seq(from = 1/8, to = 7/8, length.out = 7)
} else stop("tau_mode must be 'median' or 'octiles'")
l.tau <- length(tauvec)
nu_vec <- rep(1/l.tau, l.tau)

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

lambda.opt.index <- matrix(NA, MM, 3)

betas_TPR <- replicate(k, replicate(nlambda, matrix(NA, p, p)), simplify = FALSE)
res <- array(NA, dim = c(p, p, l.tau))
Y <- matrix(NA, n, p)


TPR_list <- FPR_list <- replicate(k, matrix(NA, MM, nlambda), simplify = FALSE)
names(TPR_list) <- names(FPR_list) <- paste("state", 1:k)

betas <- deltas <- sigmas <- emissionprobs <- replicate(MM, list())
iterations <- logliks <- matrix(NA, MM, nlambda)
gammas <- replicate(MM, array(NA, dim = c(k, k, nlambda)), simplify = FALSE)
emission_mix <- matrix(NA, nrow = n, ncol = k)


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
      ## EM initialization (K-means + simple regressions)
      delta.s <- runif(k); delta.s <- delta.s / sum(delta.s)
      beta.init <- replicate(k, array(NA, dim = c(p, p, l.tau)), simplify = FALSE)
      sigma.s   <- replicate(k, matrix(1, l.tau, p), simplify = FALSE)
      lk <- kmeans(Y, k)
      hmm_init <- markovchainFit(lk$cluster)
      delta.s <- rep(0, k); delta.s[lk$cluster[1]] <- 1
      gamma.s <- hmm_init$estimate@transitionMatrix
      if (sum(lk$cluster == 1) < 10 || sum(lk$cluster == 2) < 10) {
        lk <- kmeans(Y, k)
        hmm_init <- markovchainFit(lk$cluster)
        delta.s <- rep(0, k); delta.s[lk$cluster[1]] <- 1
        gamma.s <- hmm_init$estimate@transitionMatrix
      }
      lm.init <- vector("list", k)
      for (jj in 1:k) {
        for (pp in 1:p) {
          for (tq in 1:l.tau) {
            lm.init[[jj]] <- lm(Y[lk$cluster == jj, pp] ~ Y[lk$cluster == jj, -pp]) # with intercept
            beta.init[[jj]][pp, , tq] <- lm.init[[jj]]$coefficients
            sigma.s[[jj]][tq, pp] <- 1
          }
        }
      }
      model[[r]] <- tryCatch(
        em.hmm.pen.lqgm(
          y = Y, K = k, delta = delta.s, unilambda = lambdaseq[l],
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
  
  
  ## Post-processing per lambda
  for (l in 1:nlambda) {
    if (is.null(model_def[[l]]) || any(is.na(model_def[[l]]))) next
    
    ## Order states by intercept at median tau (or single tau)
    if (l.tau == 1) {
      betas_int_ord <- sapply(1:k, function(j) mean(model_def[[l]]$betas[[j]][, 1, 1]))
    } else {
      betas_int_ord <- sapply(1:k, function(j) mean(model_def[[l]]$betas[[j]][, 1, (l.tau + 1) / 2]))
    }
    model_def[[l]]$betas <- model_def[[l]]$betas[order(betas_int_ord, decreasing = FALSE)]
    
    ## Store outputs
    betas[[i]][[l]]   <- model_def[[l]]$betas
    sigmas[[i]][[l]]  <- model_def[[l]]$sigma
    deltas[[i]][[l]]  <- model_def[[l]]$delta
    gammas[[i]][,,l]  <- model_def[[l]]$gamma
    iterations[i, l]  <- model_def[[l]]$iter
    logliks[i, l]     <- model_def[[l]]$loglik
    
    ## Symmetrize beta arrays and compute TPR/FPR against true A
    for (j in 1:k) {
      for (tq in 1:l.tau) {
        v <- betas[[i]][[l]][[j]][, 1, tq]
        for (ii in 2:p) {
          v2 <- betas[[i]][[l]][[j]][ii, 2:ii, tq]
          betas[[i]][[l]][[j]][ii, 1:(ii - 1), tq] <- v2
        }
        diag(betas[[i]][[l]][[j]][,, tq]) <- v
      }
    }
    for (j in 1:k) {
      for (tq in 1:l.tau) {
        for (a in 1:p) {
          for (b in 1:p) {
            res[a, b, tq] <- max(abs(betas[[i]][[l]][[j]][a, b, tq]),
                                 abs(betas[[i]][[l]][[j]][b, a, tq]))
          }
        }
      }
      # Combine over quantiles (max abs value)
      betas_TPR[[j]][,, l] <- apply(res, c(1, 2), max)
      
      comparison_lqgmHM[[j]][[l]] <- data.frame(
        lambda         = lambdaseq[l],
        falsePositives = sum(A[[j]][upper.tri(A[[j]])] == 0 &
                               betas_TPR[[j]][,, l][upper.tri(betas_TPR[[j]][,, l])] != 0) /
          sum(A[[j]][upper.tri(A[[j]])] == 0),
        falseNegatives = sum(A[[j]][upper.tri(A[[j]])] != 0 &
                               betas_TPR[[j]][,, l][upper.tri(betas_TPR[[j]][,, l])] == 0) /
          sum(A[[j]][upper.tri(A[[j]])] != 0),
        truePositives  = sum(A[[j]][upper.tri(A[[j]])] != 0 &
                               betas_TPR[[j]][,, l][upper.tri(betas_TPR[[j]][,, l])] != 0) /
          sum(A[[j]][upper.tri(A[[j]])] != 0)
      )
      colnames(comparison_lqgmHM[[j]][[l]]) <-
        c("lambda", "falsePositives", "falseNegatives", "truePositives")
      
      TPR_list[[j]][i, l] <- comparison_lqgmHM[[j]][[l]]$truePositives
      FPR_list[[j]][i, l] <- comparison_lqgmHM[[j]][[l]]$falsePositives
    }
  } # end lambda loop
  
} # end MC loop

if (use_parallel && !is.null(cl)) stopCluster(cl)
t1 <- Sys.time()
time <- t1 - t0


## --------------------------- ROC SUMMARIES -------------------------------- ##
TPR <- FPR <- matrix(NA, nrow = k, ncol = length(lambdaseq),
                     dimnames = list(paste("state", 1:k), as.character(lambdaseq)))
for (j in 1:k) {
  TPR[j, ] <- apply(TPR_list[[j]], 2, mean, na.rm = TRUE)
  FPR[j, ] <- apply(FPR_list[[j]], 2, mean, na.rm = TRUE)
}
TPR_final <- colMeans(TPR)
FPR_final <- colMeans(FPR)


model_out <- list(betas = betas, sigmas = sigmas, deltas = deltas, gammas = gammas, iterations = iterations, logliks = logliks)

out_sim <- list(model_out = model_out, Y = Y, MC.sim = MC.sim, mu = mu, Sigma = Sigma, Omega = Omega, A = A,
                tauvec = tauvec, lambdaseq = lambdaseq, TPR = TPR_list, FPR = FPR_list, betas_TPR = betas_TPR,
                n = n, k = k, MM = MM, p = p, time = time, setting = setting, TPR_final = TPR_final, FPR_final = FPR_final)

## --------------------------- SAVE OUTPUTS ---------------------------------- ##

tau_tag <- if (l.tau == 1) "tau1" else paste0("tau", l.tau)
ts_tag  <- format(Sys.time(), "%Y%m%d")
fname <- sprintf("Sim2.1_%s_%s_N%d_p%d_K%d_MM%d_%s.RData",
                 tau_tag, setting, n, p, k, MM, ts_tag)
save(out_sim, file = fname)
message("Saved: ", fname)


