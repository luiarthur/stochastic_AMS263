source('hmm.R')
y <- read.table("../dat/lamb.dat",header=TRUE)$movements
plot(y,type='l',main='Lamb Movements')

system.time(out <- hmm(y,B=1000,K=2))
