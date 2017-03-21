plot_res <- function (X,y,X_new,f_new,pred,u=NA,...) {
  ci <- apply(pred,1,quantile,c(.025,.975))
  ex <- apply(pred,1,mean)
  plot(X,y,pch=20,col='red',xlim=range(X_new), fg='grey',...)
  lines(X_new,ex,col='blue', pch=20, lwd=2)
  lines(X_new,f_new,col='grey', lty=2, lwd=2)
  color.btwn(X_new, ci[1,], ci[2,], from=-4, to=4, col=rgb(0,0,.5,.2))
  #if(!is.na(u[1])) abline(v=u,col='grey') 
}

mar.ts <- c(0, 5.1, 0, 2.1)
oma.ts <- c(6, 0, 5, 0)
mar.default <- c(5.1, 4.1, 4.1, 2.1)
oma.default <- rep(0,4)

