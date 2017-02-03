module KernelConvolution

using Distributions, Distances

export fit
include("../../../MCMC/MCMC.jl")

gaussian_kernel(d::Float64) = pdf(Normal(), d)

function fit(y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64},
             cs::Vector{Float64}, # [cs_σ², cs_τ²]
             B::Int, burn::Int; printFreq=0,
             a_σ::Float64=2.0, b_σ::Float64=1.0,
             a_τ::Float64=2.0, b_τ::Float64=1.0,
             kernel=gaussian_kernel,
             dist=euclidean)

  assert(length(y) == size(X,1))

  const n = size(X,1)
  const m = size(u,1)
  const cs_matrix = Matrix(Diagonal(cs))
  const Iₙ = eye(n)
  const Iₘ = eye(m)
  const M = [ kernel(dist(X[i,:],u[j,:])) for i in 1:n, j in 1:m ]
  const MT = M'
  const MTM = MT * M
  const MMT = M * MT
  const yT = y'

  # v: [σ², τ²]
  function update(param::Vector{Float64})
    function ll(log_v::Vector{Float64})
      const sig2 = exp(log_v[1])
      const tau2 = exp(log_v[2])
      const K = Iₘ/tau2 + MTM/sig2

      const ld_Cov = -.5 * ( n*log_v[1] + m*log_v[2] + logdet(K) )
      const log_exp = -.5 * yT*(Iₙ/sig2 - M/sig2 * inv(K) * MT/sig2)*y

      return ld_Cov + log_exp[1]
      #return logpdf(MvNormal(MMT*tau2 + sig2*Iₙ), y)
    end

    function lp(log_v::Vector{Float64})
      return MCMC.lp_log_invgamma(log_v[1], a_σ, b_σ) + 
             MCMC.lp_log_invgamma(log_v[2], a_τ, b_τ)
    end

    return exp(MCMC.metropolis(log(param), cs_matrix, ll, lp))
  end

  const init = [b_σ/(a_σ-1), b_τ/(a_τ-1)]
  const out = MCMC.gibbs(init, update, B, burn, printFreq=printFreq)
  println("\tAcceptance Rate: ", length(unique(out))/length(out))

  return out
end # END OF FIT FUNCTION

#function predict(post::Vector{Vector{Float64}},
#                 y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64};
#                 kernel=gaussian_kernel,
#                 dist=euclidean)
#
#  const (n,K) = size(X)
#  const Iₙ = eye(n)
#  const m = size(u,1)
#  const M = [ kernel(euclidean(X[i,:],u[j,:])) for i in 1:n, j in 1:m ]
#  const MMT = M * M'
#  return map(p -> rand(MvNormal(MMT*p[2] + p[1]*Iₙ)), post)
#end

end # END OF MODULE
