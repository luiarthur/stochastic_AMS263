using RCall
R"library(rcommon)";
srand(263);

include("src/GP.jl")

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)

n = 100
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
R"plot($X,$y,pch=20)";

@time out = GP.fit(y, X, [.1,.1,.1], 2000, 20000, printFreq=100)

params = hcat(out...)';
R"plotPosts($params,cnames=c(paste0('sig2 (truth=',$sig2_truth,')'),'phi','alpha'))";

n_new = 100;
X_new = reshape(Vector(linspace(-2,2,n_new)),n_new,1);
f = vec(g(X_new));
@time pred = GP.predict(out, y, X, X_new);

ex = mean(pred,2)
ci = mapslices(p -> quantile(p,[.025,.975]),pred,2)

R"plot($X,$y,pch=20,col='red',ylim=range($ci),xlim=range($X_new))";
R"lines($X_new,$ex,col='blue',pch=20, lwd=2)";
R"lines($X_new,$f,col='grey',pch=20, lwd=2, lty=2)";
R"color.btwn($X_new, $ci[,1], $ci[,2], from=-4, to=4, col=rgb(0,0,.5,.2))";

#=
include("1.jl")
=#
