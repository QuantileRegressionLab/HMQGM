# Function to generate precision matrices for Scenario 1
Theta_gen <- function(p, k) {
  #' @param p: data matrix dimension
  #' @param k: number of states
  
  Omega <- array(0, c(p, p, k))
  for (i in 1:p) {
    for(j in 1:p) {
      if(abs(i-j) == 1) {
        Omega[i,j,1] = 0.2
      } else {
        Omega[i,j,1] = 0
      }
    }
  }
  
  for (i in 1:p) {
    for(j in 1:p) {
      if(abs(i-j) == 2) {
        Omega[i,j,2] = 0.2
      } else {
        Omega[i,j,2] = 0
      }
    }
  }
  
  if(k > 2) {
    for (i in 1:p) {
      for(j in 1:p) {
        if(abs(i-j) == 3) {
          Omega[i,j,3] = 0.6
        } else {
          Omega[i,j,3] = 0
        }
      }
    }
  }
  
  if(k == 2){
    diag(Omega[,,1]) = diag(Omega[,,2]) = 1
    out = list(Omega1 = Omega[,,1], Omega2 = Omega[,,2])
  } else {
    diag(Omega[,,1]) = diag(Omega[,,2]) = diag(Omega[,,3]) = 1
    out = list(Omega1 = Omega[,,1], Omega2 = Omega[,,2], Omega3 = Omega[,,3])
  }
  
  
  return(out)
  
}


# Function to generate Y data for SM §§S2.1 (as in Chun et al. 2016)
Ygen_Chun <- function(n,p){
  #' @param n: number of observations
  #' @param p: data matrix dimension
  
  Y <- matrix(NA, n, p)
  U <- matrix(NA, n, p)
  
  for (d in 1:p) {
    U[,d] <- runif(n, 0, 1)
  }
  Y[,1] <- qnorm(U[,1], 0, 1)
  Y[,2] <- -.5*U[,2]^2*(Y[,1]+3)
  Y[,3] <- Y[,1] + qnorm(U[,3], 0, .5*abs(Y[,1]) + .1)
  Y[,4] <- 0.1*(Y[,3] + 5)^2*qnorm(U[,4], 0, sqrt(abs(Y[,3] + 5)))
  Y[,5] <- 2*(U[,5] - 0.5)*(Y[,1] + 2) + qnorm(U[,5], 0, 0.1 + 0.1*abs(Y[,1]))
  Y[,6] <- 3*U[,6]*cos(pi*Y[,1]/4) + .2*qnorm(U[,6], 0, sqrt(1 + abs(Y[,1])))
  Y[,7] <- 1/(U[,7] + .1)*(abs(Y[,3]+4))^(-1/2) + (1 + sqrt(U[,7]))*log(abs(Y[,5]) + 3) + .1*qgamma(U[,7], 1, 3)
  Y[,8] <- 0.1*abs(Y[,7] + 3)^(1.3) + 0.4*abs(Y[,2] + 4)^1.7 + 0.4*qt(U[,8], df = 5)*Y[,5]
  Y[,9] <- 0.3*(Y[,8] - 0.5)^2 + qnorm(U[,9], 0, sd = 0.5)
  Y[,10] <- exp(0.2*U[,10]*log(abs(Y[,9]) +0.1))
  
  Y[,1:p] = apply(Y[,1:p], 2, scale)
  
  A <- replicate(k, matrix(0, p, p), simplify = FALSE)
  
  
  
  A[[1]][1,2] <- A[[1]][1,3] <- A[[1]][3,4] <- A[[1]][1,5] <- A[[1]][1,6] <- A[[1]][3,7] <- A[[1]][5,7] <- 1
  A[[1]][7,8] <- A[[1]][2,8] <- A[[1]][5,8] <- 1
  A[[1]][8,9] <- 1
  A[[1]][9,10] <- 1
  A[[1]][lower.tri(A[[1]], diag = F)] <- t(A[[1]])[lower.tri(A[[1]], diag = F)]
  diag(A[[1]]) <- rep(1, p)
  
  out <- list(Y = Y, A = A)
  return(out)
}



# Function to generate Y data for Scenario 2 ( as in Section 4.1)
Ygen1_sep <- function(n, k, p, MC.sim) {
  #' @param n: number of observations
  #' @param k: number of states
  #' @param p: data matrix dimension
  #' @param MC.sim: vector of states
  
  Y <- matrix(NA, n, p)
  U <- matrix(NA, n, 6)
  mu <- c()
  
  mu58 <- matrix(NA, k, 4)
  if(k == 2) {mu <- c(-.3,.3)}
  if(k == 3) {mu <- c(-.8,0,.8)}
  mu58[1,] <- rep(1, 4)
  if(k == 2){
    mu58[2,] <- rep(-1, 4)
  } else if(k == 3){
    mu58[2,] <- rep(-1, 4)
    mu58[3,] <- rep(2, 4)
  }
  
  
  Sigma <- Omega <- replicate(k, matrix(NA, 4, 4))
  
  
  for (rr in 1:4) {
    for(cc in 1:4) {
      if(abs(rr-cc) == 1) {
        Omega[rr,cc,1] = 0.2
      } else {
        Omega[rr,cc,1] = 0
      }
    }
  }
  
  if(k == 2){
    for (rr in 1:4) {
      for(cc in 1:4) {
        if(abs(rr-cc) == 2) {
          Omega[rr,cc,2] = 0.2
        } else {
          Omega[rr,cc,2] = 0
        }
      }
    }
  }
  
  if(k == 3){
    for (rr in 1:4) {
      for(cc in 1:4) {
        if(abs(rr-cc) == 2) {
          Omega[rr,cc,2] = 0.2
        } else {
          Omega[rr,cc,2] = 0
        }
      }
    }
    for (rr in 1:4) {
      for(cc in 1:4) {
        if(abs(rr-cc) == 3) {
          Omega[rr,cc,3] = 0.2
        } else {
          Omega[rr,cc,3] = 0
        }
      }
    }
  }
  
  for(j in 1:k){
    diag(Omega[,,j]) = 1
  }
  
  for (d in 1:6) {
    U[,d] <- runif(n, 0, 1)
  }
  for (j in 1:k) {
    Y[MC.sim == j,1] <- qnorm(U[MC.sim == j,1], mu[j], 1)
    Y[MC.sim == j,2] <- -.5*U[MC.sim == j,2]^2*(Y[MC.sim == j,1]+3)
    Y[MC.sim == j,3] <- Y[MC.sim == j,1] + qnorm(U[MC.sim == j,3], mu[j], .5*abs(Y[MC.sim == j,1]) + .1)
    Y[MC.sim == j,4] <- 0.1*(Y[MC.sim == j,3] + 5)^2*qnorm(U[MC.sim == j,4], mu[j], sqrt(abs(Y[MC.sim == j,3] + 5)))
    Y[MC.sim == j,5] <- 2*(U[MC.sim == j,5] - 0.5)*(Y[MC.sim == j,1] + 2) + qnorm(U[MC.sim == j,5], mu[j], 0.1 + 0.1*abs(Y[MC.sim == j,1]))
    Y[MC.sim == j,6] <- 3*U[MC.sim == j,6]*cos(pi*Y[MC.sim == j,1]/4) + .2*qnorm(U[MC.sim == j,6], mu[j], sqrt(1 + abs(Y[MC.sim == j,1])))
  }
  
  for(j in 1:k) {
    Sigma[,,j] <- solve(Omega[,,j])
    Y[MC.sim == j, 7:10] <- mvrnorm(sum(MC.sim == j), mu58[j,], Sigma[,,j])
  } #j loop
  
  Y[,1:6] = apply(Y[,1:6], 2, scale)
  # Omega_58 <- round(Omega_58, 3)
  A <- replicate(k, matrix(0, p, p), simplify = FALSE)
  
  
  for(j in 1:k) {
    A[[j]][1,2] <- A[[j]][1,3] <- A[[j]][3,4] <- A[[j]][1,5] <- A[[j]][1,6] <- 1
    A[[j]][lower.tri(A[[j]], diag = F)] <- t(A[[j]])[lower.tri(A[[j]], diag = F)]
    A[[j]][7:10, 7:10] <- ifelse(Omega[,,j] != 0 & row(Omega[,,j]) != col(Omega[,,j]), 1, 0)
    diag(A[[j]]) <- rep(1, p)
  }
  
  
  out <- list(Y = Y, A = A, Sigma = Sigma, Omega = Omega)
  return(out)
}


# Function to generate dynamic gaussin data with smooth transitions
Ygen1_dynamic <- function(N = 1000, D = 10, seed = 123, alpha) {
  #' @param N: number of observations
  #' @param D: data matrix dimension
  #' @param seed: random seed
  #' @param alpha: parameter controlling the speed of change in sparsity (higher values correspond to faster changes)
  
  set.seed(seed)
  
  # Libraries
  library(MASS)
  library(markovchain)
  library(reshape2)
  
  # Dynamic sparsity
  n_t <- floor(D / (1 + exp(-alpha * (1:N))))
  
  # Function to generate list of possible edges
  all_possible_edges <- function(D) {
    which(upper.tri(matrix(0, D, D)), arr.ind = TRUE)
  }
  
  # Function to build matrices from list of edges
  edges_to_adjacency <- function(D, edges) {
    A <- matrix(0, D, D)
    for (i in 1:nrow(edges)) {
      A[edges[i, 1], edges[i, 2]] <- 1
      A[edges[i, 2], edges[i, 1]] <- 1
    }
    diag(A) <- 0
    return(A)
  }
  
  # from adjacency matrix to precision matrix
  adjacency_to_precision <- function(A, base_rho = 0.3) {
    D <- nrow(A)
    Omega <- diag(D)
    Omega[A == 1] <- base_rho
    diag(Omega) <- 1
    return(Omega)
  }
  
  make_positive_definite <- function(Omega) {
    eigenvalues <- eigen(Omega, only.values = TRUE)$values
    if (min(eigenvalues) <= 0) {
      Omega <- Omega + diag(abs(min(eigenvalues)) + 0.1, nrow(Omega))
    }
    return(Omega)
  }
  
 
  # Output
  Y <- matrix(NA, nrow = N, ncol = D)
  adj_matrices <- vector("list", N)
  Omega_list <- vector("list", N)
  Sigma_list <- vector("list", N)
  
  # List of possible edges
  edge_pool <- all_possible_edges(D)
  n_possible <- nrow(edge_pool)
  
  # Initialize network at time 1 with n_t[1] random edges
  A_edges_prev <- edge_pool[sample(1:n_possible, n_t[1]), , drop = FALSE]
  adj_matrices[[1]] <- edges_to_adjacency(D, A_edges_prev) + diag(D)
  
  # Uniform variables for generating data
  U <- matrix(runif(N * 6), nrow = N, ncol = 6)
  
  for (t in 2:N) {
    # Keep all previous edges
    A_edges_new <- A_edges_prev
    n_add <- n_t[t] - nrow(A_edges_new)
    
    # New edges
    remaining_edges <- edge_pool[!apply(edge_pool, 1, function(x) any(apply(A_edges_new, 1, function(e) all(e == x)))), , drop = FALSE]
    if (n_add > 0 && nrow(remaining_edges) > 0) {
      new_edges <- remaining_edges[sample(1:nrow(remaining_edges), min(n_add, nrow(remaining_edges))), , drop = FALSE]
      A_edges_new <- rbind(A_edges_new, new_edges)
    }
    
    
    A_edges_prev <- A_edges_new
    adj_matrices[[t]] <- edges_to_adjacency(D, A_edges_new) + diag(D)
  }
  
  
  # Build mu_list with blocks constant over n_t
  mu_list <- vector("list", length(n_t))
  unique_levels <- rle(n_t)
  alternating_means <- rep(c(1, -1, 2, -2, 3, -3), length.out = length(unique_levels$lengths))
  
  index <- 1
  for (i in seq_along(unique_levels$lengths)) {
    len <- unique_levels$lengths[i]
    val <- alternating_means[i]
    mu_block <- replicate(len, rep(val, D), simplify = FALSE)
    mu_list[index:(index + len - 1)] <- mu_block
    index <- index + len
  }
  
  for(t in 1:N){
    A_t <- adj_matrices[[t]]
    mu_t <- mu_list[[t]]
    Omega_t <- adjacency_to_precision(A_t)
    Omega_t <- make_positive_definite(Omega_t)
    Sigma_t <- solve(Omega_t)
    
    Omega_list[[t]] <- Omega_t
    Sigma_list[[t]] <- Sigma_t
    
    Y[t,] <- mvrnorm(1, mu = mu_t, Sigma = Sigma_t)
  }
  
  return(list(Y = Y, adj_matrices = adj_matrices, Omega_list = Omega_list, Sigma_list = Sigma_list, mu_list = mu_list))
}




# Graph performance
Graph.performance = function(est.A, true.adj) {
  #' @param est.A: estimated adjacency matrix
  #' @param true.adj: true adjacency matrix
  
  est.A = 1 * (est.A != 0)
  
  TP = sum(true.adj[upper.tri(true.adj)] != 0 & est.A[upper.tri(est.A)] != 0) 
  TN = sum(true.adj[upper.tri(true.adj)] == 0 & est.A[upper.tri(est.A)] == 0) 
  FP = sum(true.adj[upper.tri(true.adj)] == 0 & est.A[upper.tri(est.A)] != 0)
  FN = sum(true.adj[upper.tri(true.adj)] != 0 & est.A[upper.tri(est.A)] == 0)
  
  P = sum(true.adj[upper.tri(true.adj)] != 0)
  N = sum(true.adj[upper.tri(true.adj)] == 0)
  
  out = c( TP / (TP + FP), # PPV
           TP / (TP + FN), # TPR
           FP / (FP + TN), # FPR
           # FP / (FP + TP), # FDR
           # FN / (FN + TP), # FNR
           2 * TP / (2 * TP + FP + FN), 
           (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)),
           (TP + TN) / (TP + TN + FP + FN),
           FN,
           0.5 * sum(est.A))
  names(out) = c("Precision", "TPR", "FPR", "F1-score", "Matthews CC", "Accuracy",
                 "Un_edges", "Tot_edges")
  # names(out) = c("TPR", "FPR", "F1-score", "Accuracy",
  # "Un_edges", "Tot_edges")
  
  return(out)
}


Viterbi=function(y,transProbs,emissionProbs,initial_distribution) {
  #' @param y: observed sequence
  #' @param transProbs: transition probability matrix
  #' @param emissionProbs: emission probability matrix
  #' @param initial_distribution: initial state distribution
  
  T = length(y)
  M = nrow(transProbs)
  prev = matrix(0, T-1, M)
  omega = matrix(0, M, T)
  
  omega[, 1] = log(initial_distribution * emissionProbs[1, ])
  for(t in 2:T){
    for(s in 1:M) {
      probs = omega[, t - 1] + log(transProbs[, s]) + log(emissionProbs[t,s])
      prev[t - 1, s] = which.max(probs)
      omega[s, t] = max(probs)
    }
  }
  
  S = rep(0, T)
  last_state=which.max(omega[,ncol(omega)])
  S[1]=last_state
  
  j=2
  for(i in (T-1):1){
    S[j]=prev[i,last_state] 
    last_state=prev[i,last_state] 
    j=j+1
  }
  
  #S[which(S==1)]='1'
  #S[which(S==2)]='2'
  
  S=rev(S)
  
  return(S)
  
}
