plot.ibp <- function(Z,...) {
  N <- nrow(Z)
  K <- ncol(Z)
  #scaledZ <- (Z-min(Z))/ diff(range(Z))
  #image(1:K, 1:N, 1-t(scaledZ[N:1,]), fg='grey',...)
  image(1:K, 1:N, max(Z)-t(Z[N:1,]), fg='grey',...)
}

