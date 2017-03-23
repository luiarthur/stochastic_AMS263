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
#
#R"pdf('../../tex/img/Z2.pdf')"
#IBP.plot( IBP.lof(X[2]),yaxt="n",xaxt="n",ylab="customers",xlab="dishes" )
#R"axis(2,at=$N:1,lab=1:$N,las=1,fg='grey')"
#R"axis(1,at=1:$K,lab=1:$K,fg='grey')"
#R"title(main=paste0('Z ~ IBP(',$alpha,')'))"
#R"dev.off()"
#
#R"pdf('../../tex/img/Z3.pdf')"
#IBP.plot( IBP.lof(X[3]),yaxt="n",xaxt="n",ylab="customers",xlab="dishes" )
#R"axis(2,at=$N:1,lab=1:$N,las=1,fg='grey')"
#R"axis(1,at=1:$K,lab=1:$K,fg='grey')"
#R"title(main=paste0('Z ~ IBP(',$alpha,')'))"
#R"dev.off()"
#
#
#X_mean = reduce(+, IBP.lof.(X)) / B
#
#R"pdf('../../tex/img/EZ.pdf')"
#IBP.plot( sortcols(X_mean,rev=true),yaxt="n",xaxt="n",ylab="customers",xlab="dishes" )
#R"axis(2,at=$N:1,lab=1:$N,las=1,fg='grey')"
#R"axis(1,at=1:$K,lab=1:$K,fg='grey')"
#R"title(main=paste0('E[Z] ~ IBP(',$alpha,')'))"
#R"dev.off()"
#

include("toy.jl")
R"par.mar <- par()$mar"
R"pdf('../../tex/img/A.pdf')"
R"par(mfrow=c(2,2),mar=c(2,2,2,2))"
IBP.plot(A1,yaxt="n",xaxt="n",ylab="",xlab="",fg="black",main="A1")
IBP.plot(A2,yaxt="n",xaxt="n",ylab="",xlab="",fg="black",main="A2")
IBP.plot(A3,yaxt="n",xaxt="n",ylab="",xlab="",fg="black",main="A3")
IBP.plot(A4,yaxt="n",xaxt="n",ylab="",xlab="",fg="black",main="A4")
R"par(mfrow=c(1,1),mar=par.mar)"
R"dev.off()"

