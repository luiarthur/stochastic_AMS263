gen.default.prior <- function(K) {
  default.prior <- NULL
  default.prior$c <- rep(1,K)
  default.prior$d <- rep(1,K)
  default.prior$a <- matrix(1,K,K)

  default.prior
}


hmm <- function(y, K=2, prior=gen.default.prior(K)) {
  N <- length(y)
}
