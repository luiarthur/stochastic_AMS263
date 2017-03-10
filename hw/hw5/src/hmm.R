gen.default.prior <- function(K) {
  default.prior <- NULL
  default.prior$c <- rep(1,K)
  default.prior$d <- rep(1,K)
  default.prior$a <- matrix(1,K,K)

  default.prior
}


# y is data
# K is number of states
# B is number of MCMC samples
hmm <- function(y, K=2, B=2000, prior=gen.default.prior(K)) {

  N <- length(y)

  update.lam <- function(param) {
    out <- sapply(1:K, function(j) {
      idx <- which(param$z==j)
      c(prior$c[j]+sum(y[idx]),
        prior$d[j]+length(idx))
    })

    list(c=out[1,], d=out[2,])
  }

  update.Q <- function(param) {
    z <- param$z
    Q <- matrix(NA,K,K)

    for (i in 1:K) {
      for (j in 1:K) {
        Q[i,j] <- prior$a[i,j]
        for (n in 2:N) {
          if (z[n-1]==i && z[n]==j) Q[i,j] <- Q[i,j] + 1
        }
      }
    }

    Q
  }

  update.z <- function(param) {
    param$z
  }

  init <- NULL
  init$z <- sample(1:K,N,replace=TRUE)
  init$lam <- update.lam(init)
  init$Q <- update.Q(init)

  params <- as.list(1:B)
  params[[1]] <- init
  for (i in 2:B) {
    curr <- params[[i-1]]
    curr$lam <- update.lam(curr)
    curr$Q <- update.Q(curr)
    curr$z <- update.z(curr)
    params[[i]] <- curr
  }

  return(params)
}
