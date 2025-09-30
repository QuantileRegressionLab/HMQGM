sumdiffabs <- function(x, y){
  out <- sum(abs(x - y)^2)
  return(out)
}

# log-forward
l.forward = function(delta, gamma, K, emission_probs){
  #' @param delta initial distribution
  #' @param gamma transition matrix
  #' @param K number of states
  #' @param emission_probs matrix of size ns x K of emission probabilities
  
  ns = nrow(emission_probs)
  l.alpha = matrix(NA,ns,K)
  foo = delta*emission_probs[1,]
  sumfoo = sum(foo)
  lscale = log(sumfoo)
  foo = foo/sumfoo
  l.alpha[1,] = log(foo)+lscale
  for(t in 2:ns)
  {
    foo = foo%*%gamma*emission_probs[t,]
    sumfoo = sum(foo)
    lscale = lscale+log(sumfoo)
    foo = foo/sumfoo
    l.alpha[t,] = log(foo)+lscale
  }
  l.alpha = t(l.alpha)
  return(l.alpha)
}
#############
# log-backward
l.backward = function(delta, gamma, K, emission_probs){
  #' @param delta initial distribution
  #' @param gamma transition matrix
  #' @param K number of states
  #' @param emission_probs matrix of size ns x K of emission probabilities
  
  ns = nrow(emission_probs)
  l.beta = matrix(NA,ns,K)
  l.beta[ns,] = rep(0,K)
  foo = rep(1/K,K)
  lscale = log(K)
  for(t in (ns-1):1)
  {
    foo = gamma%*%(emission_probs[t+1,]*foo)
    l.beta[t,] = log(foo)+lscale
    sumfoo = sum(foo)
    foo = foo/sumfoo
    lscale = lscale+log(sumfoo)
  }
  l.beta = t(l.beta)
  return(l.beta)
}



em.glasso <- function(Y, K, delta, gamma, mu, Sigma, rho, iterMax = 3e2, err = 1e-04, traceEM){
  #' @param Y: n x p data matrix
  #' @param K: number of hidden states
  #' @param delta: initial distribution of the hidden states
  #' @param gamma: K x K transition matrix
  #' @param mu: K x p matrix of means
  #' @param Sigma: list of K elements, each element is a p x p variance-covariance matrix
  #' @param rho: tuning parameter for the graphical LASSO penalty
  #' @param iterMax: maximum number of iterations
  #' @param err: tolerance for the convergence criterion
  #' @param traceEM: if TRUE, print the log-likelihood at each iteration
  
  start_time = Sys.time()
  N <- nrow(Y)
  d <- ncol(Y)
  
  dif <- Inf
  t.iter <- 0
  llkold_pen <- -10^250
  
  while (dif > err & t.iter < iterMax) {
  
    t.iter <- t.iter + 1
    emission_probs = sapply(1:K, function(j) dmvnorm(x = Y, mean = mu[j,], sigma = Sigma[[j]]))
    
    #forward probs
    la = l.forward(delta = delta, gamma = gamma, K = K, emission_probs = emission_probs)
    #backward probs
    lb = l.backward(delta = delta, gamma = gamma, K = K, emission_probs = emission_probs)
    
    c = max(la[,N])                                      
    llk = c+log(sum(exp(la[,N]-c)))
    if(is.nan(llk)){
      print(paste("llk=",llk)); llk = c(.Machine$double.eps)}
    #print(paste("llk=",llk,";c=",c))
    post.hmm = matrix(NA, nrow = K, ncol = N)
    for (j in 1:K){                                           
      post.hmm[j,] = exp(la[j,]+lb[j,]-llk)
    }
    
    #Delta estimate
    delta.next = exp(la[,1]+lb[,1]-llk)
    delta.next = delta.next/sum(delta.next)
    
    #Gamma estimate
    gamma.next = matrix(NA, nrow = K, ncol = K)
    for (j in 1:K){
      for (k in 1:K){                                                      
        tmp = exp(la[j,1:(N-1)]+   
                    log(emission_probs[2:N,k])+lb[k,2:N]-llk)
        tmp[tmp==Inf] <- 1
        tmp[tmp==-Inf] <- 0
        gamma.next[j,k] = gamma[j,k]*sum(tmp)  
      }   
    }                                                       
    gamma.next = gamma.next/apply(gamma.next,1,sum)
    
    # Mu, Sigma estimates
    mu.next = matrix(NA, nrow = K, ncol = d)
    Sigma.next = Theta.next <- replicate(K, matrix(NA, d, d), simplify = F)
    
    for (j in 1:K) {
      mu.next[j,] = colSums(post.hmm[j,]*Y)/sum(post.hmm[j,])
      Sigma.next[[j]] = wcrossprod(x = Y - matrix(mu.next[j,], nrow = N, ncol = d, byrow = TRUE), 
                                   y = Y - matrix(mu.next[j,], nrow = N, ncol = d, byrow = TRUE), 
                                   w = post.hmm[j,]) / sum(post.hmm[j,])
      foo <- glasso(s = Sigma.next[[j]], rho = rho*sqrt(sum(post.hmm[j,])/N), penalize.diagonal = F)
      Sigma.next[[j]] <- foo$w
      Theta.next[[j]] <- foo$wi # inverse of var-cov matrix
    }
    
    llk_pen = llk - rho * sum(abs(unlist(Theta.next)))
    
    #Set Convergence Criterion
    for (j in 1:K) {
      crit_mu <- sumdiffabs(mu[j,], mu.next[j,])
      crit_Sigma <- sumdiffabs(Sigma[[j]], Sigma.next[[j]])
    }
    dif <- sum(sum(abs(delta-delta.next)) + sum(abs(gamma-gamma.next)) + 
                 sum(abs(crit_Sigma)) + sum(abs(crit_mu)))
    
    dif = abs(llk_pen - llkold_pen)
    dif = max(sapply(1:K, function(k) max(abs(Sigma.next[[k]] - Sigma[[k]]) / (1 + abs(Sigma.next[[k]])))))
    
if(traceEM){
    print(round(c(t.iter, dif, llk), digits = 3))}
    
    delta = delta.next
    gamma = gamma.next
    Sigma = Sigma.next
    Theta = Theta.next
    mu = mu.next
    
    llkold_pen = llk_pen
    
    
  } #while loop
  
  end_time = Sys.time()
  timetot = end_time - start_time
  
  #info.criteria
  Df <- sapply(1:K, function(j) d + sum(Theta[[j]][lower.tri(Theta[[j]], diag = F)] != 0))
  n.par = K*(K-1) + sum(Df)
  aic.crit = -2*llk + 2*n.par
  bic.crit = -2*llk + log(N)*n.par
  icl.crit = bic.crit - 2 * sum(post.hmm * ifelse(post.hmm > 0, log(post.hmm), 0))
  crit = list(AIC = aic.crit, BIC = bic.crit, ICL = icl.crit)
  
  if(t.iter == iterMax){
    cat(paste("\nNo convergence after",iterMax,"iterations\n\n"))  
  }
  
  outlist <- list(
    iterations = t.iter, K = K, dif = dif, loglik=llk, mu = mu, Sigma = Sigma,
    delta=delta, gamma=gamma, post = post.hmm, emission_probs = emission_probs,
    crit = crit, Theta = Theta, timetot = timetot)
  
  #Output
  return(outlist)
  
}