realdata <- function(Y, N, d, R, k, nlambda, iterMax, err, use_parallel){
  #' @description Fit HMQGM on real data with multi-start and lambda grid
  #'
  #' @param Y         data matrix (N x d)
  #' @param N         number of observations
  #' @param d         number of variables
  #' @param R         number of restarts per lambda
  #' @param k         number of states
  #' @param nlambda   number of lambda values
  #' @param iterMax   maximum number of EM iterations
  #' @param err       EM tolerance (stopping criterion)
  #' @param use_parallel  whether to use a parallel backend (T/F)
  #'
  #' @return A named list with selected models, criteria, selected lambdas,
  #'         estimated adjacency/beta matrices for AIC/BIC/ICL, and timing.
  
  
  ## --------------------------- SETUP ------------------------------------- ##
  t0 <- Sys.time()
  # S <- cov(Y)  # (not used downstream, kept for completeness)
  lambdaseq <- exp(seq(log(1e-02), log(1), length.out = nlambda))
  bic_vec <- aic_vec <- icl_vec <- c()
  
  ## Quantile grid and weights (L = 7 octiles)
  tauvec <- seq(from = 1/8, to = 7/8, length.out = 7)
  l.tau  <- length(tauvec)
  
  ## Mixture weights over quantiles (non-uniform option for L=7)
  if (l.tau == 7) {
    nu_vec <- c(1,2,3,4,3,2,1); nu_vec <- nu_vec/sum(nu_vec)
  } else {
    nu_vec <- rep(1/l.tau, l.tau)
  }
  
  model <- tmp <- list()
  
  ## --------------------------- LAMBDA LOOP ------------------------------- ##
  for (l in 1:nlambda) {
    cat("lambda ", l, "\n")
    set.seed(l)
    
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
              library(rqPen), library(lqmm), library(stats),
              library(mvtnorm), library(MASS))
      ))
      parallel::clusterSetRNGStream(cl, 8693)
    }
    
    if (use_parallel) clusterExport(cl, varlist = ls(envir = globalenv()))
    
    ## ----- RESTARTS (parallelized across r) ------------------------------ ##  
    
    tmp <- parLapply(cl = if (use_parallel) cl else NULL, 1:R, function(r){
      cat("Restart ", r, "\n")
      set.seed(r)
      
      ## EM initialization
      beta.init <- replicate(k, array(NA, dim = c(d,d,l.tau)), simplify = F)
      sigma.s <- replicate(k, matrix(1, l.tau, d), simplify = F)
      lm.init <- list()
      
      ## K-means-based starting values
      lk=kmeans(Y,k)
      # lc=lk$centers
      hmm_init = markovchainFit(lk$cluster)
      delta.em = rep(0, k); delta.em[lk$cluster[1]] = 1
      gamma.em = hmm_init$estimate@transitionMatrix
      
      ## Nodewise regressions with intercept (per state and tau)
      for(j in 1:k){
        for (pp in 1:d) {
          for (t in 1:l.tau) {
            lm.init[[j]] <- lm(Y[lk$cluster==j,pp] ~ Y[lk$cluster==j,-pp]) # with intercept
            beta.init[[j]][pp,,t] <- lm.init[[j]]$coefficients  #with intercept
            sigma.s[[j]][t,pp] <- 1
          } #loop in t
        } #loop in p
      } #loop in j

      ## EM fit for this restart
      em.hmm.pen.lqgm(y = Y, K = k, delta=delta.em, unilambda = lambdaseq[l],
                      gamma=gamma.em,
                      beta=beta.init, nu_vec = nu_vec,
                      sigma=sigma.s, tauvec = tauvec,
                      tol=err, maxiter=iterMax, traceEM=T)
      
      
    }) # end parLapply over restarts

    
    ## ----- Stop cluster for this lambda ---------------------------------- ##
    if (use_parallel && !is.null(cl)) stopCluster(cl)
    
    ## ----- Select best restart by log-likelihood ------------------------- ##
    if (R == 1) {
      model[[l]] <- tmp[[1]]
    } else if (R > 1) {
      tmp.llk <- sapply(tmp, function(x) as.numeric(x[5]))
      if (sum(is.na(tmp.llk)) > 0 & sum(is.na(tmp.llk)) < R) tmp.llk[which(is.na(tmp.llk))] <- +Inf
      if (sum(is.na(tmp.llk)) == R) {
        nas <- nas + 1
        l_na[nas] <- l
        next
      }
      best_model_index <- which.max(tmp.llk)
      model[[l]] <- tmp[[best_model_index]]
    }
    
    ## ----- Criteria ------------------------------------------------------- ##
    bic_vec[l] <- model[[l]]$crit$BIC
    aic_vec[l] <- model[[l]]$crit$AIC
    icl_vec[l] <- model[[l]]$crit$ICL
    
  } # end lambda loop
  
  
  ## --------------------------- MODEL SELECTION --------------------------- ##
  lambda_aic <- lambdaseq[which.min(aic_vec)]
  lambda_bic <- lambdaseq[which.min(bic_vec)]
  lambda_icl <- lambdaseq[which.min(icl_vec)]
  
  ## Order states (ascending) for the selected lambdas
  Ahat_bic <- Ahat_aic <- Ahat_icl <- betahat_bic <- betahat_aic <- betahat_icl <- betahat_final <- replicate(k, matrix(0, d, d), simplify = F)
  res <- array(NA, dim = c(d,d,l.tau))
  betahat_bic = model[[which.min(bic_vec)]]$betas[order(diag(model[[which.min(bic_vec)]]$gamma), decreasing = T)]
  betahat_aic = model[[which.min(aic_vec)]]$betas[order(diag(model[[which.min(aic_vec)]]$gamma), decreasing = T)]
  betahat_icl = model[[which.min(icl_vec)]]$betas[order(diag(model[[which.min(icl_vec)]]$gamma), decreasing = T)]
  betahats <- list(betahat_bic, betahat_aic, betahat_icl)
  Ahats <- list(Ahat_bic = Ahat_bic, Ahat_aic = Ahat_aic, Ahat_icl = Ahat_icl)
  
  ## --------------------------- BUILD A & BETA ---------------------------- ##
  
  ## Symmetrize beta and derive adjacency for each selected criterion
  for(c in 1:3){
    ## Fill lower triangles and set diagonals from the stored intercepts
    for (j in 1:k) {
      for (t in 1:l.tau) {
        v <- betahats[[c]][[j]][,1,t] #save the first column of the matrix for each l,j,t
        for (ii in 2:d) {
          v2 <- numeric(ii-1)
          v2 <- betahats[[c]][[j]][ii,2:ii,t]
          betahats[[c]][[j]][ii,1:(ii-1),t] <- v2
        }
        diag(betahats[[c]][[j]][,,t]) <- v
      }
    }
    
    ## Max-abs symmetrization over (i,j) and collapse over tau by max
    for (j in 1:k) {
      for (t in 1:l.tau) {
        for (a in 1:d) {
          for (b in 1:d) {
            res[a,b,t] <- max(abs(betahats[[c]][[j]][a,b,t]), abs(betahats[[c]][[j]][b,a,t]))
          }}
      }
      
    # Putting togheter the matrices taking the max
      betahat_final[[j]] <- apply(res, c(1,2), max) 
      
      
      ## Adjacency (remove diagonal)
      Ahats[[c]][[j]] <- ifelse(
        as.matrix(betahat_final[[j]]) != 0 &
          row(as.matrix(betahat_final[[j]])) != col(as.matrix(betahat_final[[j]])),
        1, 0
      )
    }
  } # end c loop
  
  
  ## --------------------------- WRAP-UP ----------------------------------- ##
  
  sel_models <- list(
    bic_sel = model[[which.min(bic_vec)]],
    aic_sel = model[[which.min(aic_vec)]],
    icl_sel = model[[which.min(icl_vec)]]
  )
  t1 <- Sys.time()
  time_tot <- t1 - t0
  
  out <- list(
    lambdaseq      = lambdaseq,
    selected_models = sel_models,
    lambdas_sel    = c(lambda_aic, lambda_bic, lambda_icl),
    K              = k,
    Ahat_bic       = Ahats[[1]],
    Ahat_aic       = Ahats[[2]],
    Ahat_icl       = Ahats[[3]],
    betahat_bic    = betahats[[1]],
    betahat_aic    = betahats[[2]],
    betahat_icl    = betahats[[3]],
    time           = time_tot,
    Crit           = list(aic_vec, bic_vec, icl_vec),
    BIC            = bic_vec[which.min(bic_vec)],
    AIC            = aic_vec[which.min(aic_vec)],
    ICL            = icl_vec[which.min(icl_vec)]
  )
  
  return(out)
}
