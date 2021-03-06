using RCall
R"library(rcommon)";
srand(263);

include("GPs/PredictiveProcess.jl")

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)

toInt(x::Float64) = convert(Int, floor(x))
seq(a::Float64, b::Float64, n::Int) = reshape(Vector(linspace(a,b,n)),n,1)

n = 100
m = toInt(sqrt(n))
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
u = reshape(seq(minimum(X), maximum(X), m), m, 1)
R"plot($X,$y,pch=20)";


#@time out = PredictiveProcess.fit(y, X, u, eye(3)*.1, 5000, 2000, printFreq=100);
@time (out,CS) = PredictiveProcess.autofit(y,X,u,5000,printFreq=100);

params = hcat(out...)';
R"plotPosts($params,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'phi','alpha'))";

n_new = 100;
X_new = reshape(Vector(linspace(-2,2,n_new)),n_new,1);
f_new = vec(g(X_new));

@time pred = PredictiveProcess.predict(out, y, X, X_new, u, response="mean")

ex = mean(pred,2)
ci = mapslices(p -> quantile(p,[.025,.975]),pred,2)

R"plot($X,$y,pch=20,col='red',ylim=range($ci),xlim=range($X_new),main='Mean Function',bty='n',fg='grey')";
R"lines($X_new,$ex,col='blue', pch=20, lwd=2)";
R"lines($X_new,$f_new,col='grey', lty=2, lwd=2)";
R"color.btwn($X_new, $ci[,1], $ci[,2], from=-4, to=4, col=rgb(0,0,.5,.2))";
R"abline(v=$u,col='grey')"

println()
#=
include("pp.jl")
=#
