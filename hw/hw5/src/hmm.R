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
hmm <- function(y, K, B=2000, prior=gen.default.prior(K)) {

  N <- length(y)
  params <- as.list(1:B)
  params[[1]]$c <- prior$c
  params[[1]]$d <- prior$d

  update.lam <- function(param) {
  }

  update.Q <- function(param) {
  }

  update.z <- function(param) {
  }
}
