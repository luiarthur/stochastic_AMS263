gen.default.prior <- function(K) {
  default.prior <- NULL
  default.prior$c <- rep(1,K)
  default.prior$d <- rep(1,K)
  default.prior$a <- matrix(1,K,K)

  default.prior
}

rdir <- function(A) { # A is a matrix of alphas
  N <- nrow(A)
  K <- ncol(A)
  R <- matrix(rgamma(N*K,A,1),N,K)
  matrix(R,N,K) / rowSums(R)
}


# y is data
# K is number of states
# B is number of MCMC samples
hmm <- function(y, K=2, B=2000, prior=gen.default.prior(K)) {

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

    for (i in 1:K) {
      for (j in 1:K) {
        A[i,j] <- prior$a[i,j]
        for (n in 2:N) {
          if (z[n-1]==i && z[n]==j) A[i,j] <- A[i,j] + 1
        }
      }
    }

    rdir(A)
  }

  sample.z <- function(param) {
    
    z <- param$z
    Q <- param$Q
    lam <- param$lam

    d.forecast <- function(i) {
      if (i==1) {
        dpois(y[i],lam[1])
      } else {
        sum(dpois(y[i],lam) * Q[z[i-1],])
      }
    }

    # z_t | Y_{t-1}, etc \propto 
    # sum_{k=1}^m { p(z_t|z_{t-1}=k,etc) \times p(z_{t-1}=k|Y_{t-1},etc) }
    d.pred <- function(s,i) {
      sapply(1:K, function(k) {
        # FIXME
        #Q[k,] * if (i==1) 1 else d.z(i-1)
      })
    }

    # z_t | Y_t, etc \propto
    # p(z_t | Y_{t-1}, etc) \times f(y_t | Y_{t-1}, etc_{s_t})
    d.update <- function(s,i) {
      p.pred(s,i) * d.forecast(i)
    }

    z
  }

  init <- NULL
  init$z <- sample(1:K,N,replace=TRUE)
  init$lam <- sample.lam(init)
  init$Q <- sample.Q(init)

  params <- as.list(1:B)
  params[[1]] <- init
  for (i in 2:B) {
    curr <- params[[i-1]]
    curr$lam <- sample.lam(curr)
    curr$Q <- sample.Q(curr)
    curr$z <- sample.z(curr)
    params[[i]] <- curr
  }

  return(params)
}
