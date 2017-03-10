gen.default.prior <- function(K) {
  default.prior <- NULL
  default.prior$c <- rep(1,K)
  default.prior$d <- rep(1,K)
  default.prior$a <- matrix(1,K,K)

  default.prior
}

pow <- function(X,n) {
  if (n==1) X else pow(X %*% X, n-1)
}

rdir <- function(A) { # A is a matrix of alphas
  N <- nrow(A)
  K <- ncol(A)
  R <- matrix(rgamma(N*K,A,1),N,K)
  R / rowSums(R)
}


# y is data
# K is number of states
# B is number of MCMC samples
hmm <- function(y, K=2, B=2000, burn=100, prior=gen.default.prior(K)) {

  N <- length(y)

  sample.lam <- function(param) {
    sapply(1:K, function(j) {
      idx <- which(param$z==j)
      rgamma(1, prior$c[j]+sum(y[idx]), prior$d[j]+length(idx))
    })
  }

  sample.Q <- function(param) {
    z <- param$z
    A <- matrix(NA,K,K)

    for (i in 1:K) for (j in 1:K) {
      A[i,j] <- prior$a[i,j]
      for (n in 2:N) {
        if (z[n-1]==i && z[n]==j) A[i,j] <- A[i,j] + 1
      }
    }

    rdir(A)
  }

  sample.z <- function(param) {
    
    Q <- param$Q
    lam <- param$lam

    # P[i,k] = p(z_i=k | Y_i)
    P <- matrix(NA, N, K)

    # For identifiability, set z_1 = 1
    #P[1,] <- c(1,rep(0,K-1))
    # Set the first state to have prob of the stationary distribution
    P[1,] <- pow(Q,30)[1,]
    for (i in 2:N) {
      # Prediction Step
      pred <- P[i-1,] %*% Q
      # Update Step
      updt <- dpois(y[i],lam)

      P[i,] <- pred * updt
    }

    # Smoothing
    z <- double(N)
    z[N] <- sample(1:K, 1, prob=P[N,])
    for (i in (N-1):1) {
      p <- P[i,] * Q[,z[i+1]]
      z[i] <- sample(1:K, 1, prob=p)
    }

    return(z)
  }

  init <- NULL
  init$z <- sample(1:K,N,replace=TRUE)
  init$lam <- sample.lam(init)
  init$Q <- sample.Q(init)

  params <- as.list(1:B)
  params[[1]] <- init
  for (i in 2:(B+burn)) {
    cat("\r",i)
    curr <- params[[i-1]]
    curr$lam <- sample.lam(curr)
    curr$Q <- sample.Q(curr)
    curr$z <- sample.z(curr)
    params[[i]] <- curr
  }

  return(tail(params,B))
}
