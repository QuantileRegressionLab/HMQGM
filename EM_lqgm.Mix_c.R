check <- function(x,tau=.5){
  x*(tau - (x<0))
}

wlogSumExp = function(x, w = NULL) {
  a = max(x)
  a + log(sum(w * exp(x - a)))
}

#########################################.############################################
####################### EM for penalized HM quantile regression ####################
#########################################.############################################


em.hmm.pen.lqgm = function(y, tauvec, K, delta, unilambda, gamma, beta, sigma, nu_vec, maxiter = 1e2, tol = 1e-04, traceEM){
  #' @param y: n x p data matrix
  #' @param tauvec: vector of quantile levels
  #' @param K: number of hidden states
  #' @param delta: initial distribution of the hidden states
  #' @param unilambda: tuning parameter for the LASSO penalty
  #' @param gamma: K x K transition matrix
  #' @param beta: list of K elements, each element is a p x (p+1) x L matrix of regression coefficients
  #' @param sigma: list of K elements, each element is a L x p matrix of scale parameters
  #' @param nu_vec: vector of length L, weights for the mixture representation of the ALD
  #' @param maxiter: maximum number of iterations
  #' @param tol: tolerance for the convergence criterion
  #' @param traceEM: if TRUE, print the log-likelihood at each iteration
  
  
  start_time = Sys.time()
  n = dim(y)[1]
  p = dim(y)[2]
  n.tau = length(tauvec)
  # nu_vec = rep(1/L, L)
  
  dif <- Inf
  t.iter <- 0
  llk.pred = -1e250
  em_break <- 0
  
  while (dif > tol & t.iter < maxiter) {
    # print(paste("iteration", t.iter))
    set.seed(1)
    t.iter <- t.iter + 1
    # lallprobs = matrix(NA, nrow = n,ncol = K)
    lallprobs = array(NA, dim = c(n,n.tau,K))
    
    t0 <- Sys.time()
    for (j in 1:K){
      tmp = core(Y = y, beta = beta[[j]], sigma = sigma[[j]], ns = n, L = n.tau, 
                 P = p, K = K, tauvec = tauvec)
      lallprobs[,,j] = tmp$fden
    }
    t1 <- Sys.time()
    # print(paste("time for lallprobs", t1-t0))
    
    tmp = forward(delta = delta, gamma = gamma, K = K, fden = lallprobs, L = n.tau, ns = n)
    a = tmp$a
    c_scale = tmp$c_scale
    b = backward(delta = delta, gamma = gamma, K = K, fden = lallprobs, L = n.tau, ns = n)
    
    ### Define z, post, gamma
    llk_v <- c()
    for(tt in 1:n.tau) {
      llk_v[tt] = -sum(log(c_scale[,tt])) # log(1/L)
    }
    # llk = matrixStats::logSumExp(llk_v)
    llk = wlogSumExp(llk_v, w = nu_vec)
    # if(is.nan(llk)){
    #   print(paste("llk=",llk)); llk = c(.Machine$double.eps)}
    
    post.hmm = matrix(NA , ncol = n, nrow = K)
    
    z = a * b 
    for (t in 1:n) {
      z[t,,] = diag(nu_vec)%*%z[t,,]
    }
    z = z / (apply(z, 1, sum) %o% rep(1, n.tau) %o% rep(1, K))
    z[is.na(z)]=.Machine$double.eps
    z[z < .Machine$double.eps]=.Machine$double.eps
    post.hmm = apply(z, c(1, 3), sum)
    
    # Check if any state is not visited
    if(K == 2){
    if(sum(post.hmm[,1]) < p/n || sum(post.hmm[,2]) < p/n) {
      cat("Warning: one of the two states is not visited in the E-step\n")
      em_break <- 1
      break
    }} else if(K == 3){
      if(sum(post.hmm[,1]) < p/n || sum(post.hmm[,2]) < p/n || sum(post.hmm[,3]) < p/n) {
        cat("Warning: one of the three states is not visited in the E-step\n")
        em_break <- 1
        break
      }
    } else if(K == 4){
      if(sum(post.hmm[,1]) < p/n || sum(post.hmm[,2]) < p/n || sum(post.hmm[,3]) < p/n || sum(post.hmm[,4]) < p/n) {
        cat("Warning: one of the four states is not visited in the E-step\n")
        em_break <- 1
        break
      }
    }
    
    # M-step
    #Delta estimate
    delta.next = post.hmm[1,]/sum(post.hmm[1,])
    
    #Gamma estimate
    gamma.next = matrix(NA,nrow = K, ncol = K)
    
    v = vnum(a = a, b = b, fden = lallprobs, gamma = gamma, ns = n, K = K, 
             L = n.tau, nu_vec = array(nu_vec, dim = c(1, n.tau, 1)))
    vden = apply(v, 1, sum, na.rm = T) %o% rep(1, K) %o% rep(1, K)
    v = v / vden
    v[is.na(v)]=0
    
    for(j in 1:K){
      for(k in 1:K){
        gamma.next[j,k]=sum(v[,j,k])/sum(v[,j,])
      }
    }
    gamma.next

    
    ######## Beta and Sigma estimate ############################
    
    beta.next = replicate(K, array(NA, dim = c(p, p, n.tau)), simplify = F) #no intercept
    aux = replicate(K, array(NA, dim = c(n,p,n.tau)), simplify = F)
    sigma.next = replicate(K, matrix(NA, n.tau, p), simplify = F)
    
    # lambda <- c()
    lambda <- matrix(NA, nrow = K, ncol = n.tau)
    fit <- list()

    
    for (j in 1:K){
    
      if(sum(is.infinite(post.hmm)) >0){
        post.hmm[post.hmm==Inf]<-1
        post.hmm[post.hmm== -Inf]<-0
      }
      ### ****************
      ## Penalized estimation ############################
      
      for (tt in 1:n.tau) {
        lambda[j,tt] = unilambda*sqrt(sum(z[,tt,j])/n)
        
        for (pp in 1:p) {
          # cat("j=",j," tt=",tt," pp=",pp,"\n")
          if(sum(z[,tt,j]) < n*0.01){coef = c(1,rep(0,p-1))
          }else{
            #rqPen3.2.1
          coef = rqPen:::LASSO.fit(y = y[,pp], x = y[,-pp], tau = tauvec[tt], 
                                   lambda = lambda[j,tt],
                                   weights = z[,tt,j],
                                   coef.cutoff=1e-6, intercept = T)
          }

          
          fit$coefficients = coef
          beta.next[[j]][pp,,tt] = fit$coefficients
          fit$residuals = y[,pp] - cbind(1, y[,-pp]) %*% beta.next[[j]][pp,,tt]
          sigma.next[[j]][tt,pp] = sum(z[,tt,j]*check(as.numeric(fit$residuals), 
                                                      tau=tauvec[tt]))/sum(z[,tt,j])
        } #loop in pp
      } #loop in tt
    } #loop in j
    
    #Set Convergence Criterion
    penalty = 0
    for (j in 1:K){
      for (tt in 1:n.tau) {
        penalty = penalty + lambda[j,tt] * sum(abs(beta.next[[j]][,-1,tt]))
      }
    }
    llk_pen = llk - penalty
    
    dif = abs(llk_pen-llk.pred)
    
    
    
    if(traceEM) {cat("iteration: ",t.iter," loglik = ", llk_pen, " dif = ", dif, "\n")}
    
    delta = delta.next
    beta = beta.next
    gamma = gamma.next
    sigma = sigma.next
    # llk.pred = llk
    llk.pred = llk_pen
    
    
  } # while loop
  
  end_time = Sys.time()
  timetot = end_time - start_time
  #info.criteria
  n.par = K*(K-1) + (K-1)  +  K*n.tau*p + sum(unlist(beta) != 0)
  aic.crit = -2*llk + 2*n.par
  bic.crit = -2*llk + log(n)*n.par
  icl.crit = bic.crit - 2 * sum(post.hmm * ifelse(post.hmm > 0, log(post.hmm), 0))
  crit = list(AIC = aic.crit, BIC = bic.crit, ICL = icl.crit)
  no_convergence = 0
  
  if(t.iter == maxiter){
    cat(paste("\nNo convergence after",maxiter,"iterations\n\n"))  
    no_convergence = 1
  }                     
  
  
  
  output <- list(K=K, betas = beta, delta = delta, gamma = gamma, 
                 loglik = llk, emissionprobs = lallprobs,
                 iteration = t.iter, no_convergence = no_convergence, 
                 sigma = sigma, post = post.hmm, 
                 crit = crit, dif = dif, time = timetot, em_break = em_break, z = z)
  
  #Output
  return(output)
}
