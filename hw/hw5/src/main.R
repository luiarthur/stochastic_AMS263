source('hmm.R')
y <- read.table("../dat/lamb.dat",header=TRUE)$movements
plot(y,type='l',main='Lamb Movements')

out <- hmm(y,B=10)
