n <- 1:5 * 100
kc <- c(2.6, 5.1, 8.8, 16.5, 26)
gp <- c(4, 8.6, 19.9, 38, 64)

plot(n, gp, lwd=2, type='b', col='blue', 
     ylab='seconds')
lines(n, kc, lwd=2, type='b', col='red')
legend("topleft",bty='n',col=c('blue','red'),
       legend=c("GP","KC"),lwd=5, cex=3)

mod <- lm(gp/kc ~ n)
plot(n, gp/kc, lwd=2, type='b', col='blue', 
     ylab='seconds')
abline(mod$coef[1], mod$coef[2])
