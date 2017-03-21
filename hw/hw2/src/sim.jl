using RCall
R"library(rcommon)";
R"source('helper.R')";
srand(263);

include("GPs/GP.jl")
include("GPs/KernelConvolution.jl")
include("GPs/PredictiveProcess.jl")

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)
toInt(x::Float64) = convert(Int, floor(x))
seq(a::Float64, b::Float64, n::Int) = reshape(Vector(linspace(a,b,n)),n,1)

n = 30 # 1000
MCMC_its = 300 # 5000
m = toInt(sqrt(n))
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
u = reshape(seq(minimum(X), maximum(X), m), m, 1)
#R"plot($X,$y,pch=20)";


# Running GPs
@time (gp,CS_gp) = GP.autofit(y,X,MCMC_its,printFreq=100);
@time (kc,CS_kc) = KernelConvolution.autofit(y,X,u,MCMC_its,printFreq=100);
@time (pp,CS_pp) = PredictiveProcess.autofit(y,X,u,MCMC_its,printFreq=100);

param_gp = hcat(gp...)'
param_kc = hcat(kc...)'
param_pp = hcat(pp...)'

R"""
pdf('../img/paramGP.pdf')
plotPosts($param_gp,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'phi','alpha'))
dev.off()
pdf('../img/paramKC.pdf')
plotPosts($param_kc,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'tau2','phi'))
dev.off()
pdf('../img/paramPP.pdf')
plotPosts($param_pp,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'phi','alpha'))
dev.off()
"""

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
pdf('../img/pred.pdf')
par(mfrow=c(3,1), oma=oma.ts, mar=mar.ts)
plot_res($X,$y,$X_new,$f_new,$pred_gp,ylim=c(-.3,2.2),xaxt='n',ylab='GP')
plot_res($X,$y,$X_new,$f_new,$pred_kc,ylim=c(-.3,2.2),xaxt='n',u=$u,ylab='KC') 
plot_res($X,$y,$X_new,$f_new,$pred_pp,ylim=c(-.3,2.2),u=$u,ylab='PP')
par(mfrow=c(1,1), oma=oma.default, mar=mar.default)
dev.off()
"""

#=
include("sim.jl")
=#
