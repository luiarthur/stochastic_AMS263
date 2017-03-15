using RCall
R"library(rcommon)";
srand(263);

include("GPs/GP.jl")
include("GPs/KernelConvolution.jl")
include("GPs/PredictiveProcess.jl")

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)
toInt(x::Float64) = convert(Int, floor(x))
seq(a::Float64, b::Float64, n::Int) = reshape(Vector(linspace(a,b,n)),n,1)

n = 1000
m = toInt(sqrt(n))
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
u = reshape(seq(minimum(X), maximum(X), m), m, 1)
R"plot($X,$y,pch=20)";


# Running GPs
@time (gp,CS_gp) = GP.autofit(y,X,5000,printFreq=100);
@time (kc,CS_kc) = KernelConvolution.autofit(y,X,u,5000,printFreq=100);
@time (pp,CS_pp) = PredictiveProcess.autofit(y,X,u,5000,printFreq=100);

param_gp = hcat(gp...)'
param_kc = hcat(kc...)'
param_pp = hcat(pp...)'

# New Locations
n_new = 100;
X_new = reshape(Vector(linspace(-2,2,n_new)),n_new,1);
f_new = vec(g(X_new));

# Predict
@time pred_gp = GP.predict(gp, y, X, X_new, response="mean");
@time pred_kc = KernelConvolution.predict(kc, y, X, X_new, u, response="mean");
@time pred_pp = PredictiveProcess.predict(pp, y, X, X_new, u, response="mean");

# Plotter
R"""
plot_res <- function (X,y,X_new,f_new,pred,u=NA,...) {
  ci <- apply(pred,1,quantile,c(.025,.975))
  ex <- apply(pred,1,mean)
  plot(X,y,pch=20,col='red',xlim=range(X_new), bty='n',fg='grey',...)
  lines(X_new,ex,col='blue', pch=20, lwd=2)
  lines(X_new,f_new,col='grey', lty=2, lwd=2)
  color.btwn(X_new, ci[1,], ci[2,], from=-4, to=4, col=rgb(0,0,.5,.2))
  if(!is.na(u[1])) abline(v=$u,col='grey') 
}
"""

R"plot_res($X,$y,$X_new,$f_new,$pred_gp,ylim=c(-.2,2),main='GP Mean Function')"
R"plot_res($X,$y,$X_new,$f_new,$pred_kc,ylim=c(-.2,2),u=$u,main='KC Mean Function')"
R"plot_res($X,$y,$X_new,$f_new,$pred_pp,ylim=c(-.2,2),u=$u,main='PP Mean Function')"

#=
include("sim.jl")
=#
