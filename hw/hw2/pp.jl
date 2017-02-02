using RCall
R"library(rcommon)";
srand(263);

include("src/PredictiveProcess.jl")

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)

toInt(x::Float64) = convert(Int, floor(x))
seq(a::Float64, b::Float64, n::Int) = reshape(Vector(linspace(a,b,n)),n,1)

n = 500
m = toInt(sqrt(n))
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
u = reshape(seq(minimum(X), maximum(X), m), m, 1)
R"plot($X,$y,pch=20)";


@time out = PredictiveProcess.fit(y, X, u, [.1,.1,.1], 1000, 20000, printFreq=100);

params = hcat(out...)';
R"plotPosts($params,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'phi','alpha'))";

@time pred = PredictiveProcess.predict_mean_at_original_locs(out, y, X, u)

ex = mean(pred,2)
ci = mapslices(p -> quantile(p,[.025,.975]),pred,2)
gx = g(X)

R"plot($X,$gx,pch=20,col='red',ylim=range($ci),xlim=range($X))";
R"lines($X,$ex,col='blue', pch=20, lwd=2)";
R"color.btwn($X, $ci[,1], $ci[,2], from=-4, to=4, col=rgb(0,0,.5,.2))";

println()
#=
include("pp.jl")
=#
