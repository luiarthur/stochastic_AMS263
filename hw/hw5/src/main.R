set.seed(2)
library(rcommon)
source('hmm.R')
y <- read.table("../dat/lamb.dat",header=TRUE)$movements
N <- length(y)
plot(y,type='l',main='Lamb Movements')

prior <- NULL
prior$c <- c(1,2)
prior$d <- c(2,1)
prior$a <- matrix(c(.75,.5,.25,.5),2,2)
system.time(out <- hmm(y,B=3000,burn=1000,K=2,prior=prior))
#system.time(out <- hmm(y,B=2000,K=2))


# Z Posterior
Z <- sapply(out,function(o) o$z)# - 1
pz1 <- apply(Z,1,function(z) mean(z==1))
par(mfrow=c(3,1))
plot(y,type='h',main='Lamb Movements',bty='n',fg='grey')
plot(pz1,type='h')
plot(1-pz1,type='h')
par(mfrow=c(1,1))

# Lambda Posterior
lam <- t(sapply(out, function(o) o$lam))
plotPosts(lam)

# Q Posterior
Q <- sapply(out,function(o) o$Q)
matrix(apply(Q,1,mean),2,2)
matrix(apply(Q,1,sd),2,2)
