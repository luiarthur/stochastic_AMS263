using RCall
using Distributions
srand(1);

include("LFM.jl")
include("IBP.jl")
include("toy.jl")

# Plot toy data
IBP.plot(A1,xlab="",ylab="");
IBP.plot(A2,xlab="",ylab="");
IBP.plot(A3,xlab="",ylab="");
IBP.plot(A4,xlab="",ylab="");

IBP.plot(A1 + A2 + A3 + A4)

# gen samples
K = 10 
N = 100
sig2_true = .04
(K_true, D) = size(A)
Z_true = rand(Multinomial(2, [.25,.25,.25,.25]),N)'
X = Z_true * A + randn(N,D)*sqrt(sig2_true)
IBP.plot(Z_true)

# Plot data
IBP.plot(X)
IBP.plot(reshape(X[2,:],6,6));


println("Start MCMC...")

# BURN-IN
@time tmp = LFM.fit(X,K=K,B=100,burn=0,printFreq=1,
                    sig2_true=sig2_true,
                    cs_v=eye(K))
cs_v = cov(hcat(unique(map(o->o.v, tmp))...)') + eye(K)*.1

@time out = LFM.fit(X,K=K,B=500,burn=1000,printFreq=1,
                    sig2_true=sig2_true,
                    init=tmp[end],cs_v=cs_v);

sig2_out = map(o -> o.sig2, out)
R"plotPost($sig2_out)";
alpha_out = map(o -> o.alpha, out)
length(unique(alpha_out)) / length(alpha_out)
R"plotPost($alpha_out)";
v_out = hcat(map(o -> o.v, out)...)'
length(unique(v_out)) / length(v_out)
R"plotPosts($v_out[,1:3])";
R"plotPosts($v_out[,4:6])";
R"plotPosts($v_out[,7:10])";
Z_out = map(o -> o.Z, out)
Z_mean = reduce(+, Z_out) / length(Z_out)
Z_true_ext = hcat(Z_true, zeros(N,K-size(Z_true,2)))
IBP.plot(Z_mean)
IBP.plot(Z_mean[:,1:4])
IBP.plot(Z_mean[:,1:4]-Z_true)
IBP.plot(Z_true)
IBP.plot(Z_mean-Z_true_ext)
#A_out = map( o -> o.A, out)
#A_mean = reduce(+, A_out) / length(A_out)
#IBP.plot(A_mean)
#IBP.plot(A)

function estA(Z::Matrix{Int64})
  const positiveCol = find(sum(Z,1) .> 0)
  const Z1 = Z[:,positiveCol]
  inv(Z1'Z1)*Z1'*X
end

A_est = map(estA, Z_out)
A_mean = reduce(+, A_est) / length(A_est)

IBP.plot(reshape(A_mean[1,:],6,6))
IBP.plot(reshape(A_mean[2,:],6,6))
IBP.plot(reshape(A_mean[3,:],6,6))
IBP.plot(reshape(A_mean[4,:],6,6))
IBP.plot(reshape(A_mean[5,:],6,6))
IBP.plot(reshape(A_mean[6,:],6,6))
IBP.plot(reshape(A_mean[7,:],6,6))
IBP.plot(reshape(A_mean[8,:],6,6))
IBP.plot(reshape(A_mean[9,:],6,6))

#=
include("test.jl")
=#
