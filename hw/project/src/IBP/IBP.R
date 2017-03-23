plot.ibp <- function(Z,col=grey(seq(0,1,len=12)),fg='grey',...) {
  N <- nrow(Z)
  K <- ncol(Z)
  COL <- col
  FG <- fg
  #scaledZ <- (Z-min(Z))/ diff(range(Z))
  #image(1:K, 1:N, 1-t(scaledZ[N:1,]), fg='grey',...)
  image(1:K, 1:N, max(Z)-t(Z[N:1,]), fg=FG,col=COL,...)
}

