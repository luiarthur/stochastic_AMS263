set.seed(2)
library(rcommon)
library(xtable)
source('hmm.R')
y <- read.table("../dat/lamb.dat",header=TRUE)$movements
N <- length(y)
#plot(y,type='l',main='Lamb Movements')

prior <- NULL
prior$c <- c(1,2)
prior$d <- c(2,1)
prior$a <- matrix(c(.75,.5,.25,.5),2,2)
system.time(out <- hmm(y,B=2000,burn=200,K=2,prior=prior))
#system.time(out <- hmm(y,B=2000,K=2))


# Z Posterior
Z <- sapply(out,function(o) o$z)# - 1
pz1 <- apply(Z,1,function(z) mean(z==1))
pdf('../tex/img/z.pdf')
par(mfrow=c(3,1))
plot(y,type='h',main='Lamb Movements',bty='n',fg='grey',xlab='time',
     ylab='No. of Movements',cex.main=2,col.main='grey30')
plot(pz1,type='h',xlab='',main=expression(P(z[t]==1~"|"~y)),
     bty='n',fg='grey',ylab='Probability',cex.main=2,col.main='grey30')
plot(1-pz1,type='h',xlab='',main=expression(P(z[t]==2~"|"~y)),
     bty='n',fg='grey',ylab='Probability',cex.main=2,col.main='grey30')
par(mfrow=c(1,1))
dev.off()

# Lambda Posterior
lam <- t(sapply(out, function(o) o$lam))
pdf('../tex/img/lam.pdf')
plotPosts(lam,cnames=c(expression(lambda[1]),expression(lambda[2])))
dev.off()

# Q Posterior
Q <- sapply(out,function(o) o$Q)
Q.mean <- matrix(apply(Q,1,mean),2,2)
Q.sd <- matrix(apply(Q,1,sd),2,2)


SUMMARY <- matrix(NA,4,4)
colnames(SUMMARY) <- c("Mean","SD","Lower","Upper")
rownames(SUMMARY) <- c("$\\lambda_1$","$\\lambda_2$","$Q_{11}$","$Q_{22}$")
SUMMARY[1:2,1] <- apply(lam,2,mean)
SUMMARY[1:2,2] <- apply(lam,2,sd)
SUMMARY[1:2,3:4] <- t(apply(lam,2,quantile,c(.025,.975)))
SUMMARY[1:4,1] <- diag(Q.mean)
SUMMARY[1:4,2] <- diag(Q.sd)
SUMMARY[3,3:4] <- quantile(Q[1,],c(.025,.975))
SUMMARY[4,3:4] <- quantile(Q[4,],c(.025,.975))

sink('../tex/img/SUMMARY.tex')
print(xtable(SUMMARY),sanitize.text.function=function(x) x)
sink()
