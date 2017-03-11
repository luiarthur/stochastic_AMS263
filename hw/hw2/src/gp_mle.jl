using RCall, Distributions, Distances, Optim
include("GPs/GP.jl")
R"library(rcommon)";
srand(263);

g(X::Matrix{Float64}) = .3 + .4*X + .5*sin(2.7*X) + 1.1 ./ (1+X.^2)

n = 300
X = sort(randn(n,1),1)
sig2_truth = .01
y = vec(g(X) + randn(n)*sqrt(sig2_truth) )
R"plot($X,$y,pch=20)";
D = pairwise(Euclidean(),X')



function ll(param::Vector{Float64})

  const out = if any(param .<= 0) 
    -Inf
  else
    const sig2 = param[1]
    const phi = param[2]
    const alpha = param[3]

    const N = length(y)
    const I = eye(N)
    const K = GP.exp_cov(D, phi, alpha)

    logpdf(MvNormal(sig2*I + K), y)
  end

  return out
end

neg_ll(x::Vector{Float64}) = -ll(x)

@time opt = optimize(neg_ll,ones(3)*100)
mle = opt.minimizer
println(mle)

n_new = 100;
X_new = reshape(Vector(linspace(-2,2,n_new)),n_new,1);
f_new = vec(g(X_new))
@time pred = GP.predict([mle for i in 1:1000], y, X, X_new)
ex = mean(pred,2)
ci = mapslices(p -> quantile(p,[.025,.975]),pred,2)

R"plot($X,$y,pch=20,col='red',ylim=range($ci),xlim=range($X_new))";
R"lines($X_new,$ex,col='blue',pch=20, lwd=2)";
R"lines($X_new,$f_new,col='grey',pch=20, lwd=2, lty=2)";
R"color.btwn($X_new, $ci[,1], $ci[,2], from=-4, to=4, col=rgb(0,0,.5,.2))";

#=
include("gp_mle.jl")
=#
