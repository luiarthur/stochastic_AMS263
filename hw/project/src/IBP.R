plot.ibp <- function(Z,show_y_axis=TRUE, show_x_axis=TRUE) {
  N <- nrow(Z)
  K <- ncol(Z)
  image(t(Z), xaxt='n', yaxt='n',fg='grey')
  if (show_x_axis) axis(1,at=(0:(K-1))/(K-1),label=1:K,fg='grey')
  if (show_y_axis) axis(2,at=(0:(N-1))/(N-1),label=N:1,las=1,fg='grey')
}

