module PredictiveProcess

using Distributions, Distances

export fit

include("../../../MCMC/MCMC.jl")
include("autofit.jl")

sym(M::Matrix{Float64}) = (M + M') / 2

function exp_cov(D::Matrix{Float64}, ϕ::Float64, α::Float64)
  return α * exp(-ϕ * D)
end

function autofit(y::Vector{Float64},X::Matrix{Float64},u::Matrix{Float64},B::Int;
                 burn::Int=0, cs::Matrix{Float64}=eye(3),
                 printFreq::Int=0, init::Vector{Float64}=zeros(3),
                 a_σ::Float64=2.0, b_σ::Float64=1.0,
                 a_ϕ::Float64=0.0, b_ϕ::Float64=10.0,
                 a_a::Float64=2.0, b_a::Float64=1.0,
                 dist=Euclidean,
                 max_autotune::Int=20, 
                 window::Int=500,
                 target::Float64=.25,
                 target_lower::Float64=.25,
                 target_upper::Float64=.40,
                 k::Float64=2.5)

  function f(cs::Matrix{Float64}, B::Int, burn::Int, 
             init::Vector{Float64}, printFreq::Int)
    return fit(y,X,u,cs,B,burn,printFreq=printFreq,init=init,dist=dist,
               a_σ=a_σ, b_σ=b_σ, a_ϕ=a_ϕ, b_ϕ=b_ϕ, a_a=a_a, b_a=b_a)
  end

  return AUTOFIT(f,B,3,
                 burn=burn, printFreq=printFreq, max_autotune=max_autotune,
                 window=window, target=target, target_lower=target_lower,
                 target_upper=target_upper, k=k)
end


function fit(y::Vector{Float64}, X::Matrix{Float64}, u::Matrix{Float64},
             cs::Matrix{Float64}, # [cs_σ², cs_ϕ, cs_α]
             B::Int, burn::Int; printFreq=0,
             init::Vector{Float64}=zeros(3),
             a_σ::Float64=2.0, b_σ::Float64=1.0,
             a_ϕ::Float64=0.0, b_ϕ::Float64=10.0,
             a_a::Float64=2.0, b_a::Float64=1.0,
             dist=Euclidean)

  assert(length(y) == size(X,1))

  const yT = y'
  const n = size(X,1)
  const m = size(u,1)
  const Iₙ = eye(n)
  const Iₘ = eye(m)

  const D_XU = pairwise(dist(), [X;u]')
  const D_C = D_XU[1:n, (n+1):end]
  const D_K = D_XU[(n+1):end, (n+1):end]

  # param: [σ², ϕ, α]
  function trans_param(v::Vector{Float64})
    return [log(v[1]), MCMC.logit(v[2],a_ϕ,b_ϕ), log(v[3])]
  end

  function inv_trans_param(x::Vector{Float64})
    return [exp(x[1]), MCMC.inv_logit(x[2],a_ϕ,b_ϕ), exp(x[3])]
  end

  # param: [σ², ϕ, α]
  function update(param::Vector{Float64})

    function ll(t_v::Vector{Float64})
      const sig2 = exp(t_v[1])
      const phi = MCMC.inv_logit(t_v[2],a_ϕ,b_ϕ)
      const alpha = exp(t_v[3])
      const K = exp_cov(D_K, phi, alpha)
      const C = exp_cov(D_C, phi, alpha)
      const CT = C'
      const CTC = CT * C
      const H = K + CTC / sig2
      const Hi = inv(H)
      
      #const ld_cov = -.5 * (n*log(sig2) + logdet(H) - logdet(K))
      const ld_cov = -.5 * (n*log(sig2) - logdet(Hi) - logdet(K))
      const log_exp = -.5 * yT*(Iₙ/sig2 - C/sig2 * Hi * CT/sig2)*y
      
      return log_exp[1] + ld_cov
    end

    function lp(t_v::Vector{Float64})
      return MCMC.lp_log_invgamma(t_v[1],a_σ,b_σ) +
             MCMC.lp_logit_unif(t_v[2]) +
             MCMC.lp_log_invgamma(t_v[3],a_a,b_a)
    end

    return inv_trans_param(MCMC.metropolis(trans_param(param), cs, ll, lp))
  end

  # param: [σ², ϕ, α]
  const INIT = if init==zeros(3)
    [b_σ/(a_σ-1.0), (a_ϕ + b_ϕ)/2.0, b_a/(a_a-1.0)]
  else
    init
  end

  const out = MCMC.gibbs(INIT, update, B, burn, printFreq=printFreq)
  println("\tAcceptance Rate: ", length(unique(out))/length(out))
  return out
end
 
function predict_mean_at_original_locs(post::Vector{Vector{Float64}},
                                       y::Vector{Float64}, X::Matrix{Float64},
                                       u::Matrix{Float64}; dist=Euclidean)
  const n = size(X,1)
  const D_XU = pairwise(dist(), [X;u]')
  const D_C = D_XU[1:n, (n+1):end]
  const D_K = D_XU[(n+1):end, (n+1):end]

  function pred(p::Vector{Float64})
    const sig2 = p[1]
    const phi = p[2]
    const a = p[3]

    const K = exp_cov(D_K, phi, a)
    const Ki = inv(K)
    const C = exp_cov(D_C, phi, a)
    const H = C * Ki
    const S = Ki + H'H / sig2
    const Si = sym(inv(S))

    const mu_u = rand( MvNormal(Si*H'*y /sig2, Si) )
    const mu_x = C * Ki * mu_u
    return mu_x
  end

  return hcat(map(p -> pred(p), post)...)
end

end # END OF MODULE

#=
Need to make corrections on website: http://luiarthur.github.io/ucsc_notes/advBayesComputing/17/

1. Under Kernel Convolution, the distribution for $z$  should be
    - $z \sim \N(0,\sigma^2_v I_m)$
2. The posterior for mean function at the knot points is 
    - $\mu^* \v y,\sigma^2,\tau^2,\phi \sim \text{MVN}(S^{-1}H'y/\sigma^2,S^{-1})$
3. At the end of the Predictive Process section, it should be: 
    - $S =K^{*-1}+\frac{H’H}{\sigma^2}$
=#
