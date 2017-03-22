# Just for reference, don't run it again!
#include("IBP.jl")
#using RCall
#srand(100);
#N=30; K=18; alpha=5.0; B = 1000
#
#X = [ IBP.randIBP(alpha, N, K) for i in 1:1000]
#
#R"pdf('../../tex/img/Z.pdf')"
#IBP.plot( IBP.lof(X[1]),yaxt="n",xaxt="n",ylab="customers",xlab="dishes" )
#R"axis(2,at=$N:1,lab=1:$N,las=1,fg='grey')"
#R"axis(1,at=1:$K,lab=1:$K,fg='grey')"
#R"title(main=paste0('Z ~ IBP(',$alpha,')'))"
#R"dev.off()"
#
#X_mean = reduce(+, IBP.lof.(X)) / B
#
#
#R"pdf('../../tex/img/EZ.pdf')"
#IBP.plot( sortcols(X_mean,rev=true),yaxt="n",xaxt="n",ylab="customers",xlab="dishes" )
#R"axis(2,at=$N:1,lab=1:$N,las=1,fg='grey')"
#R"axis(1,at=1:$K,lab=1:$K,fg='grey')"
#R"title(main=paste0('E[Z] ~ IBP(',$alpha,')'))"
#R"dev.off()"

