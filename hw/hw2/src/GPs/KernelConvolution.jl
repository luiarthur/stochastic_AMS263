module KernelConvolution

using Distributions, Distances

export fit
include("../../../MCMC/MCMC.jl")
include("autofit.jl")

gaussian_kernel(d::Float64) = pdf(Normal(0,.1), d)

function fit(y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64},
             cs::Matrix{Float64}, # [cs_σ², cs_τ²]
             B::Int, burn::Int; printFreq=0,
             init::Vector{Float64}=zeros(2),
             a_σ::Float64=2.0, b_σ::Float64=1.0,
             a_τ::Float64=2.0, b_τ::Float64=1.0,
             kernel=gaussian_kernel,
             dist=euclidean)

  assert(length(y) == size(X,1))

  const n = size(X,1)
  const m = size(u,1)
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
      #const K = Iₘ/tau2 + MTM/sig2

      #const ld_Cov = -.5 * ( n*log_v[1] + m*log_v[2] + logdet(K) )
      #const log_exp = -.5 * yT*(Iₙ/sig2 - M/sig2 * inv(K) * MT/sig2)*y

      #return ld_Cov + log_exp[1]
      return logpdf(MvNormal(MMT*tau2 + sig2*Iₙ), y)
    end

    function lp(log_v::Vector{Float64})
      return MCMC.lp_log_invgamma(log_v[1], a_σ, b_σ) + 
             MCMC.lp_log_invgamma(log_v[2], a_τ, b_τ)
    end

    return exp(MCMC.metropolis(log(param), cs, ll, lp))
  end

  const INIT = if init==zeros(2)
    [b_σ/(a_σ-1), b_τ/(a_τ-1)]
  else
    init
  end

  const out = MCMC.gibbs(INIT, update, B, burn, printFreq=printFreq)
  println("\tAcceptance Rate: ", length(unique(out))/length(out))

  return out
end # END OF FIT FUNCTION

function autofit(y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64},B::Int; 
                 burn::Int=0, cs::Matrix{Float64}=eye(2),
                 kernel=gaussian_kernel,
                 dist=euclidean,
                 printFreq::Int=0, init::Vector{Float64}=zeros(2),
                 a_σ::Float64=2.0, b_σ::Float64=1.0,
                 a_τ::Float64=2.0, b_τ::Float64=1.0,
                 max_autotune::Int=20, 
                 window::Int=500,
                 target::Float64=.25,
                 target_lower::Float64=.25,
                 target_upper::Float64=.40,
                 k::Float64=2.5)

  function f(cs::Matrix{Float64}, B::Int, burn::Int, 
             init::Vector{Float64}, printFreq::Int)
    return fit(y,X,u,cs,B,burn,printFreq=printFreq,init=init,
               a_σ=a_σ, b_σ=b_σ, a_τ=a_τ, b_τ=b_τ,kernel=kernel,dist=dist)
  end

  return AUTOFIT(f,B,2,
                 burn=burn, printFreq=printFreq, max_autotune=max_autotune,
                 window=window, target=target, target_lower=target_lower,
                 target_upper=target_upper, k=k)
end # end of autofit



"""
Goal: Sample y*|z ~ N(Kz, σ²Iₙ), where K is a kernel and z ~ GP(0, τ²Iₘ)
This means, y* ~ N(0, σ²Iₙ + KK'τ²)
"""
# FIXME: NOT WORKING!!!
function predict(post::Vector{Vector{Float64}},
                 y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64};
                 kernel=gaussian_kernel,
                 dist=euclidean)
  println("KernelConvolution.predict IS BROKEN & STILL NEEDS FIXING!!!")

  const n = size(X,1)
  const m = size(u,1)
  const Iₙ = eye(n)
  const Iₘ = eye(m)

  const K = [ kernel(dist(X[i,:],u[j,:])) for i in 1:n, j in 1:m ]
  const KKT = K*K'

  function pred(sig2::Float64, tau2::Float64)
    return rand( MvNormal(sig2*Iₙ + KKT*tau2) )
  end

  return map(p -> pred(p[1],p[2]), post)
end

end # END OF MODULE
