###############################################################################
# simulations_4.1-2_HMglasso.R
# Simulation study for manuscript §§4.1–4.2 (HMGlasso version):
# - Ability to correctly retrieve the edge set per hidden state under:
#     - Scenario 1 ("scenario1"): multivariate Gaussian (K=2; K=3 in SM)
#     - Scenario 2 ("scenario2"): conditional quantile generator via Ygen1_sep (K=2; K=3 in SM)
# - Clustering performance via ARI (posterior and Viterbi) for varying K
#
# What this script does:
#   1) Ensures dependencies (optional auto-install) + clean sourcing
#   2) Configures optional parallel backend
#   3) Generates synthetic data per scenario
#   4) Runs EM for HMGlasso over a lambda grid with multi-start
#   5) Picks best start by log-likelihood; computes AIC/BIC/ICL; ARI (post) + Viterbi ARI
#   6) Builds ROC summaries (TPR/FPR per state; macro averages)
#   7) Saves a single list object `out_glassoHM` for §§4.1–4.2
###############################################################################

## --------------------------- CLEAN SETTINGS -------------------------------- ##
rm(list = ls(all.names = TRUE))
invisible(gc())
options(stringsAsFactors = FALSE, warn = 1)

## --------------------------- USER SETTINGS -------------------------------- ##
use_parallel     <- TRUE     # enable/disable parallel backend
AUTO_INSTALL_PKGS <- TRUE    # auto-install missing deps in clean envs

## --------------------------- ENV & LIBRARIES ------------------------------ ##
pkgs <- c(
  "parallel","doParallel","foreach",
  "ald","quantreg","lqmm","rqPen",
  "mclust","markovchain",
  "MASS","mvtnorm",
  "pracma","Rcpp","RcppArmadillo",
  "LQGM","car","glasso","skewt","cluster","copula"
  # bayestestR used via ::; optional auto-install below
)

if (AUTO_INSTALL_PKGS) {
  cran_pkgs <- setdiff(pkgs, c("LQGM","rqPen"))
  miss <- cran_pkgs[!cran_pkgs %in% rownames(installed.packages())]
  if (length(miss)) {
    message("Installing missing CRAN packages: ", paste(miss, collapse = ", "))
    install.packages(miss, repos = "https://cloud.r-project.org")
  }
  # rqPen from local tarball (as in your other script)
  if (!("rqPen" %in% rownames(installed.packages()))) {
    tarball <- "rqPen_3.2.1.tar.gz"
    if (!file.exists(tarball)) stop("Missing local tarball: ", tarball)
    install.packages(tarball, repos = NULL, type = "source")
  }
  # LQGM from local tarball
  if (!("LQGM" %in% rownames(installed.packages()))) {
    tarball <- "LQGM_1.0.tar.gz"
    if (!file.exists(tarball)) stop("Missing local tarball: ", tarball)
    install.packages(tarball, repos = NULL, type = "source")
  }
  # optional: bayestestR for AUC computation (used via bayestestR::)
  if (!("bayestestR" %in% rownames(installed.packages()))) {
    install.packages("bayestestR", repos = "https://cloud.r-project.org")
  }
}


  library(foreach); library(parallel); library(doParallel)
  library(ald); library(quantreg); library(lqmm); library(rqPen)
  library(mclust); library(markovchain)
  library(MASS); library(mvtnorm)
  library(pracma); library(Rcpp); library(RcppArmadillo)
  library(LQGM); library(car); library(glasso)
  library(skewt); library(cluster); library(copula)


## --------------------------- SOURCE FILES --------------------------------- ##
if (!file.exists("em_glasso.R"))    stop("Required file not found: em_glasso.R")
if (!file.exists("MainFunctions.R")) stop("Required file not found: MainFunctions.R")
source("em_glasso.R")
source("MainFunctions.R")

## --------------------------- DESIGN CHOICES ------------------------------- ##
N            <- c(1000)
K            <- c(1, 2, 3)      # number of states
R            <- 5               # restarts per lambda
MM           <- 100              # Monte Carlo simulations
d            <- c(10)
nlambda      <- 50
distribution <- c("n")
settings     <- c("scenario1", "scenario2")

## --------------------------- SETUP ----------------------------- ##
n       <- N[1]
k       <- K[2]
p       <- d[1]
dist    <- distribution[1]
setting <- settings[2]

## --------------------------- TRUE STRUCTURES ------------------------------ ##
mu     <- matrix(NA, k, p, byrow = TRUE)
Omega  <- replicate(k, matrix(NA, p, p))
Sigma  <- A <- replicate(k, matrix(NA, p, p), simplify = FALSE)

if (setting == "scenario1") {
  Omega[,,1] <- Theta_gen(p = p, k = k)$Omega1
  Omega[,,2] <- Theta_gen(p = p, k = k)$Omega2
  if (k == 3) Omega[,,3] <- Theta_gen(p = p, k = k)$Omega3
  A <- lapply(1:k, function(j) ifelse(Omega[,,j] != 0 & row(Omega[,,j]) != col(Omega[,,j]), 1, 0))
  for (j in 1:k) Sigma[[j]] <- solve(Omega[,,j])
}

## --------------------------- LAMBDA GRID ---------------------------------- ##
# IMPORTANT:
# - For ARI results use lambda_mode = "ari" (default).
# - For ROC curves use lambda_mode = "roc".
# Grids depend on K as in the manuscript/SM; fallback mirrors ARI-like grid.
lambda_mode <- "ari"    # choose between "ari" and "roc" (affects the lambda grid)

lambdaseq <- switch(
  as.character(k),
  "2" = if (tolower(lambda_mode) == "roc")
    exp(seq(log(1e-03), log(1), length.out = nlambda))   # K=2 ROC
  else
    exp(seq(log(1e-02), log(1), length.out = nlambda)),   # K=2 ARI (first submission)
  "3" = if (tolower(lambda_mode) == "roc")
    exp(seq(log(1e-05), log(7), length.out = nlambda))    # K=3 ROC (revision)
  else
    exp(seq(log(1e-05), log(3), length.out = nlambda)),   # K=3 ARI (revision)
  { # K=1 or other values: ARI-like default
    exp(seq(log(1e-02), log(1), length.out = nlambda))
  }
)

## --------------------------- HMM INIT ------------------------------------- ##
if (k == 2) {
  delta <- c(1, 0)
  gamma <- matrix(c(0.9, 0.2,
                    0.1, 0.8), k, k)
} else if (k == 3) {
  delta <- c(0.4, 0.3, 0.3)
  gamma <- matrix(c(0.8, 0.1, 0.1,
                    0.1, 0.8, 0.1,
                    0.1, 0.1, 0.8), k, k, byrow = TRUE)
} else { # k == 1
  delta <- 1
  gamma <- matrix(1, 1, 1)
}

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

## --------------------------- OUTPUT HOLDERS ------------------------------- ##
emissionprobs <- replicate(MM, list())
aRI_aic <- aRI_bic <- aRI_icl <- aRI_Vit_aic <- aRI_Vit_bic <- aRI_Vit_icl <- c()
bic_vec <- aic_vec <- icl_vec <- c()
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
  
  ## Hidden state sequence
  if (k != 1) {
    States <- paste0("k", 1:k)
    MC <- methods::new("markovchain", states = States, byrow = TRUE, transitionMatrix = gamma)
    MC.sim <- markovchain::rmarkovchain(n = n - 1, object = MC,
                                        t0 = sample(States, size = 1, prob = delta),
                                        include.t0 = TRUE, what = data.frame)
    for (j in 1:k) MC.sim[MC.sim == paste0("k", j)] <- j
    MC.sim <- as.numeric(MC.sim)
  } else {
    MC.sim <- rep(1, n)
  }
  
  ## Generate Y under chosen scenario
  if (setting == "scenario2") {
    tmpY <- Ygen1_sep(n, k, p, MC.sim)
    Y <- tmpY$Y
    A <- tmpY$A
  } else if (setting == "scenario1" && dist == "n") {
    if (k == 2) {
      mu[1, ] <- rep(-.45, p); mu[2, ] <- rep(.45, p)
    } else if (k == 3) {
      mu[1, ] <- rep(-1, p); mu[2, ] <- rep(0, p); mu[3, ] <- rep(1, p)
    } else { # k == 1
      mu[1, ] <- rep(0, p)
    }
    for (j in 1:k) {
      Y[MC.sim == j, ] <- mvtnorm::rmvnorm(n = sum(MC.sim == j), mean = mu[j, ], sigma = Sigma[[j]])
    }
  }
  
  ## Fit over lambda grid with R restarts; keep best-LL per lambda
  model <- replicate(R, list(), simplify = FALSE)
  
  if (use_parallel) clusterExport(cl, varlist = ls(envir = globalenv()))

  model_def <- parLapply(cl = if (use_parallel) cl else NULL, 1:nlambda, function(l) {
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
    } # restarts
    tmp.llk <- sapply(model, function(x) as.numeric(x[4]))
    if (sum(is.na(tmp.llk)) > 0 & sum(is.na(tmp.llk)) < R) tmp.llk[which(is.na(tmp.llk))] <- +Inf
    if (sum(is.na(tmp.llk)) == R) return(NA)
    model[[which.max(tmp.llk)]]
  })
  
  ## Post-processing per lambda
  for (l in 1:nlambda) {
    if (is.null(model_def[[l]]) || any(is.na(model_def[[l]]))) next
    
    bic_vec[l] <- model_def[[l]]$crit$BIC
    aic_vec[l] <- model_def[[l]]$crit$AIC
    icl_vec[l] <- model_def[[l]]$crit$ICL
    
    if (k == 2) {
      model_def[[l]]$Theta <- model_def[[l]]$Theta[order(diag(model_def[[l]]$gamma), decreasing = TRUE)]
    } else if (k == 3) {
      model_def[[l]]$Theta <- model_def[[l]]$Theta[order(rowMeans(model_def[[l]]$mu), decreasing = FALSE)]
    }
    
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
  
  ## Select optimal lambda by info criteria (AIC/BIC/ICL)
  lambda.opt.index[i, 1] <- which.min(aic_vec)
  lambda.opt.index[i, 2] <- which.min(bic_vec)
  lambda.opt.index[i, 3] <- which.min(icl_vec)
  
  aRI_aic[i] <- mclust::adjustedRandIndex(
    x = MC.sim, y = apply(model_def[[lambda.opt.index[i, 1]]]$post, 2, which.max)
  )
  aRI_bic[i] <- mclust::adjustedRandIndex(
    x = MC.sim, y = apply(model_def[[lambda.opt.index[i, 2]]]$post, 2, which.max)
  )
  aRI_icl[i] <- mclust::adjustedRandIndex(
    x = MC.sim, y = apply(model_def[[lambda.opt.index[i, 3]]]$post, 2, which.max)
  )
  
  emissionprobs[[i]] <- model_def[[lambda.opt.index[i, 3]]]$emission_probs
  
  seq_Viterbi_aic <- Viterbi(Y[, 1],
                             transProbs = gammas[[i]][,, lambda.opt.index[i, 1]],
                             emissionProbs = emissionprobs[[i]],
                             initial_distribution = deltas[[i]][[lambda.opt.index[i, 1]]])
  seq_Viterbi_bic <- Viterbi(Y[, 1],
                             transProbs = gammas[[i]][,, lambda.opt.index[i, 2]],
                             emissionProbs = emissionprobs[[i]],
                             initial_distribution = deltas[[i]][[lambda.opt.index[i, 2]]])
  seq_Viterbi_icl <- Viterbi(Y[, 1],
                             transProbs = gammas[[i]][,, lambda.opt.index[i, 3]],
                             emissionProbs = emissionprobs[[i]],
                             initial_distribution = deltas[[i]][[lambda.opt.index[i, 3]]])
  
  aRI_Vit_aic[i] <- mclust::adjustedRandIndex(x = MC.sim, y = seq_Viterbi_aic)
  aRI_Vit_bic[i] <- mclust::adjustedRandIndex(x = MC.sim, y = seq_Viterbi_bic)
  aRI_Vit_icl[i] <- mclust::adjustedRandIndex(x = MC.sim, y = seq_Viterbi_icl)
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

out_glassoHM <- list(model_out = model_out, Theta.f = Theta.f, Y = Y, MC.sim = MC.sim, A = A,
                     lambdaseq = lambdaseq, TPR = TPR_list, FPR = FPR_list,
                     TPR_final = TPR_final, FPR_final = FPR_final,
                     n = n, k = k, MM = MM, p = p, time = time, setting = setting,
                     aRI_aic = aRI_aic, aRI_bic = aRI_bic, aRI_icl = aRI_icl,
                     aRI_Vit_aic = aRI_Vit_aic, aRI_Vit_bic = aRI_Vit_bic, aRI_Vit_icl = aRI_Vit_icl,
                     lambda.opt.index = lambda.opt.index,
                     Graph.grid.info = Graph.grid.info)

## --------------------------- SAVE OUTPUT ---------------------------------- ##
ts_tag <- format(Sys.time(), "%Y%m%d")
fname  <- sprintf("out_glassoHM_%s_%s_N%d_p%d_K%d_MM%d_%s.RData",
                  setting, lambda_mode, n, p, k, MM, ts_tag)
save(out_glassoHM, file = fname)
message("Saved: ", fname)
