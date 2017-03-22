using RCall
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
Z_true = rand(0:1, N, K_true)
X = Z_true * A + randn(N,D)*sqrt(sig2_true)
IBP.plot(Z_true)

# Plot data
IBP.plot(X)
IBP.plot(reshape(X[2,:],6,6));


println("Start MCMC...")

# BURN-IN
@time tmp = LFM.fit(X,K=K,B=100,burn=0,printFreq=1,
                    cs_v=eye(K), A_true=A*1.)
cs_v = cov(hcat(unique(map(o->o.v, tmp))...)') + eye(K)*.1

@time out = LFM.fit(X,K=K,B=200,burn=200,printFreq=1,A_true=A*1.,
                    init=tmp[end], cs_v=cs_v*1000)

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
IBP.plot(Z_mean-Z_true_ext)
A_out = map( o -> o.A, out)
A_mean = reduce(+, A_out) / length(A_out)
IBP.plot(A_mean)
IBP.plot(A)


#=
include("test.jl")
=#
