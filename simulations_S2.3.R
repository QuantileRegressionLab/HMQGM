###############################################################################
# simulations_S2.3.R
# Simulation study for SM §§S2.3:
# - Sensitivity analysis in which the true adjacency matrix varies smoothly over time:
#     - We fit and compare our model with K=1,...,K=5 latent states over a grid of L = 7 quantile levels ("octile model")
# - Clustering performance via ARI (posterior and Viterbi) for varying K
#
# What this script does:
#   1) Ensures dependencies (optional auto-install) + clean sourcing
#   2) Configures optional parallel backend
#   3) Generates synthetic dynamic data via Ygen1_dynamic
#      (time-varying graphs controlled by alpha) per scenario
#   4) Runs EM for HMQGM over a lambda grid with multi-start
#   5) Picks best start by log-likelihood; computes AIC/BIC/ICL
#   6) Builds summaries (AUC per state; macro averages)
#   7) Saves a single list object `out_hmqgm_dyn` for §§S2.3
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
# - bayestestR: AUC
pkgs <- c(
  "parallel","doParallel","foreach",
  "ald","quantreg","lqmm","rqPen",
  "mclust","markovchain",
  "MASS","mvtnorm",
  "pracma", "bayestestR",
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
library(bayestestR)

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
          library(lqmm), library(stats), library(bayestestR),
          library(mvtnorm), library(MASS))
  ))
  parallel::clusterSetRNGStream(cl, 8693)
}


## --------------------------- DESIGN CHOICES ------------------------------- ##
N            <- c(1000)
K            <- c(5)     # max number of states
R            <- 5           # restarts per lambda
MM           <- 50         # Monte Carlo simulations
d            <- c(10)
nlambda      <- 50
distribution <- c("n")
settings     <- c("scenario3_smooth")
alpha = 0.005  # smoothness parameter for scenario 3

## --------------------------- SETUP ----------------------------- ##
n    <- N[1]
p    <- d[1]
dist <- distribution[1]
setting <- settings[1]

##States loop
for(k in 1:K){ 
  
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
  if (setting == "scenario3_smooth" && !exists("Ygen1_dynamic")) {
    stop("Scenario 'scenario3_smooth' selected but function 'Ygen1_dynamic' is not available in ", "MainFunctions.R")
  }
  
  ## --------------------------- TRUE STRUCTURES ------------------------------ ##
  mu     <- matrix(NA, k, p, byrow = TRUE)
  Omega  <- replicate(k, matrix(NA, p, p))
  Sigma  <- A <- replicate(k, matrix(NA, p, p), simplify = FALSE)
  MC.sim <- list()
  
  ## --------------------------- LAMBDA GRIDS -------------------------------- ##
  lambdaseq <- exp(seq(log(1e-03), log(3), length.out = nlambda))
  
  
  ## --------------------------- OUTPUT HOLDERS ------------------------------- ##
  Graph.grid.info_l <- array(data = NA, dim = c(n, 8, nlambda),
                             dimnames = list(NULL, c("Precision", "TPR", "FPR", 
                                                     "F1-score", "Matthews CC", 
                                                     "Accuracy",
                                                     "Un_edges", "Tot_edges"),
                                             paste0("lambda",rep(1:nlambda)) ))
  Graph.grid.info <- replicate(MM,matrix(NA, nrow = nlambda, ncol = 8, 
                                         dimnames = list(NULL, c("Precision", 
                                                                 "TPR", "FPR", 
                                                                 "F1-score", "Matthews CC",
                                                                 "Accuracy",
                                                                 "Un_edges", "Tot_edges"))), 
                               simplify = F)
  
  AUC_t <- matrix(NA, nrow = n, ncol = MM)
  AUC <- c()
  betas_TPR <- replicate(k, replicate(nlambda, matrix(NA, p, p)), simplify = FALSE)
  res <- array(NA, dim = c(p, p, l.tau))
  betas <- deltas <- sigmas <- emissionprobs <- replicate(MM, list())
  iterations <- logliks <- matrix(NA, MM, nlambda)
  gammas <- replicate(MM, array(NA, dim = c(k, k, nlambda)), simplify = FALSE)
  Theta_t <- A.est <- vector("list", n)
  
  
  ## --------------------------- MONTE CARLO LOOP ----------------------------- ##

  for (i in 1:MM) {
    cat("Simulation =", i, "\n")
    set.seed(i)
    
    ## Generate dynamic data (time-varying graphs controlled by alpha)
    out_Ygen   <- Ygen1_dynamic(N = n, D = p, seed = i, alpha = alpha)
    Y          <- out_Ygen$Y
    A_list     <- out_Ygen$adj_matrices
    Sigma_list <- out_Ygen$Sigma_list
    Omega_list <- out_Ygen$Omega_list
    
    tmp   <- list()
    model <- list()  # multistart container
    
    ## Export necessary variables/functions to the cluster
    if (use_parallel== TRUE) {
      clusterExport(cl, varlist = ls(envir = globalenv()))
    }
    
    ## Fit over lambda grid with R restarts; keep best-LL per lambda
    model_def <- parLapply(cl = cl, 1:nlambda, function(l) {
      nas <- 0
      for (r in 1:R) {
        set.seed(r)
        
        ## ----- EM initialization -----
        delta.s <- runif(k); delta.s <- delta.s / sum(delta.s)    # simplex
        beta.init <- replicate(k, array(NA, dim = c(p, p, l.tau)), simplify = FALSE)
        sigma.s   <- replicate(k, matrix(1, l.tau, p), simplify = FALSE)
        lm.init   <- list()
        
        ## K-means-based starting values (clusters are randomized to avoid empties)
        lk <- kmeans(Y, k)
        lk$cluster <- sample(1:k, n, replace = TRUE)
        lc        <- lk$centers
        hmm_init  <- markovchainFit(lk$cluster)
        delta.s   <- rep(0, k); delta.s[lk$cluster[1]] <- 1
        gamma.s   <- hmm_init$estimate@transitionMatrix
        
        ## Re-try if a cluster is too small (<10 obs in either of the first two states)
        if (sum(lk$cluster == 1) < 10 || sum(lk$cluster == 2) < 10) {
          lk <- kmeans(Y, k)
          lk$cluster <- sample(1:k, n, replace = TRUE)
          lc        <- lk$centers
          hmm_init  <- markovchainFit(lk$cluster)
          delta.s   <- rep(0, k); delta.s[lk$cluster[1]] <- 1
          gamma.s   <- hmm_init$estimate@transitionMatrix
        }
        
        ## Nodewise regressions with intercept for each state/quantile
        for (j in 1:k) {
          for (pp in 1:p) {
            for (t in 1:l.tau) {
              lm.init[[j]] <- lm(Y[lk$cluster == j, pp] ~ Y[lk$cluster == j, -pp])  # with intercept
              beta.init[[j]][pp, , t] <- lm.init[[j]]$coefficients                  # store coefficients
              sigma.s[[j]][t, pp]      <- 1
            }
          }
        }
        
        ## EM fit for the given lambda and restart
        model[[r]] <- tryCatch(
          em.hmm.pen.lqgm(
            y = Y, K = k, delta = delta.s, unilambda = lambdaseq[l],
            gamma = gamma.s, beta = beta.init, nu_vec = nu_vec,
            sigma = sigma.s, tauvec = tauvec, tol = 1e-04,
            maxiter = 1e2, traceEM = TRUE
          ),
          error = function(e) NA, silent = TRUE
        )
      } # end restarts
      
      ## Select best restart by log-likelihood
      tmp.llk <- sapply(model, function(x) as.numeric(x[5]))
      if (sum(is.na(tmp.llk)) > 0 & sum(is.na(tmp.llk)) < R) tmp.llk[which(is.na(tmp.llk))] <- +Inf
      if (sum(is.na(tmp.llk)) == R) {
        nas <- nas + 1
        l_na[nas] <- l
        next
      }
      best_model_index <- which.max(tmp.llk)
      model_def <- model[[best_model_index]]
      return(model_def)
    })  # end parLapply over lambda
    
    ## ----- Post-processing over the lambda grid -----
    for (l in 1:nlambda) {
      if (is.null(model_def[[l]]) || any(is.na(model_def[[l]]))) next
      
      ## Store per-lambda outputs
      betas[[i]][[l]]   <- model_def[[l]]$betas
      sigmas[[i]][[l]]  <- model_def[[l]]$sigma
      deltas[[i]][[l]]  <- model_def[[l]]$delta
      gammas[[i]][,, l] <- model_def[[l]]$gamma
      iterations[i, l]  <- model_def[[l]]$iter
      logliks[i, l]     <- model_def[[l]]$loglik
      post_max_index    <- apply(model_def[[l]]$post, 1, which.max)
      
      ## Build symmetric beta matrices (max-abs symmetrization across (i,j))
      for (j in 1:k) {
        for (tt in 1:length(tauvec)) {
          v <- betas[[i]][[l]][[j]][, 1, tt]              # diagonal (intercepts)
          for (ii in 2:p) {
            v2 <- betas[[i]][[l]][[j]][ii, 2:ii, tt]
            betas[[i]][[l]][[j]][ii, 1:(ii - 1), tt] <- v2
          }
          diag(betas[[i]][[l]][[j]][,, tt]) <- v
        }
      }
      for (j in 1:k) {
        for (tt in 1:l.tau) {
          for (a in 1:p) {
            for (b in 1:p) {
              res[a, b, tt] <- max(
                abs(betas[[i]][[l]][[j]][a, b, tt]),
                abs(betas[[i]][[l]][[j]][b, a, tt])
              )
            }
          }
        }
        betas_TPR[[j]][,, l] <- apply(res, c(1, 2), max)  # collapse across tau via max
      }
      
      ## Per-time adjacency from the most likely state; graph metrics per time & lambda
      for (t in 1:n) {
        Theta_t[[t]] <- betas_TPR[[post_max_index[t]]][,, l]
        A.est[[t]]   <- (Theta_t[[t]] != 0) * 1
        Graph.grid.info_l[t,, l] <- Graph.performance(est.A = A.est[[t]], true.adj = A_list[[t]])
      }
      
      ## Average graph metrics over time for this simulation & lambda
      Graph.grid.info[[i]][l, ] <- colMeans(Graph.grid.info_l[,, l])
    }  # end lambda loop
    
    ## Per-time AUCs for this simulation; then mean AUC across times
    AUC_t[, i] <- sapply(1:n, function(t) {
      bayestestR::auc(Graph.grid.info_l[t, 3, ], Graph.grid.info_l[t, 2, ])
    })
    AUC[i] <- mean(AUC_t[, i], na.rm = TRUE)
  }  # end MC loop
  
  
  
  Graph.grid.info.3d <- simplify2array(Graph.grid.info)
  Graph.grid.info.out <- apply(Graph.grid.info.3d, c(1,2), median, na.rm = T)
  
  
  
  out_hmqgm_dyn <- list(Graph.grid.info = Graph.grid.info,
                        Graph.grid.median = Graph.grid.info.out,
                        K = K, n = n, p = p, setting = setting,
                        lambdaseq = lambdaseq, AUC_t = AUC_t, AUC = AUC,
                        A_list = A_list, Sigma_list = Sigma_list,
                        Omega_list = Omega_list, Theta_t = Theta_t,
                        model = model_def, R = R,
                        Graph.grid.info_l = Graph.grid.info_l,
                        Y = Y)
  
  
  ## --------------------------- SAVE OUTPUT ---------------------------------- ##
  tau_tag <- if (l.tau == 1) "tau1" else paste0("tau", l.tau)
  ts_tag  <- format(Sys.time(), "%Y%m%d")
  fname <- sprintf("dynsmooth_hmqgm_%s_N%d_p%d_%s_K%d_MM%d_alpha%s_%s.RData",
                   tau_tag, n, p, setting, k, MM, alpha, ts_tag)
  
  save(out_hmqgm_dyn, file = fname)
  message("Saved: ", fname)
  

} #K loop


if (use_parallel && !is.null(cl)) stopCluster(cl)
t1 <- Sys.time()
time <- t1 - t0
