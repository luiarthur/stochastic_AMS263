module GP

using Distributions
using Distances

export fit, autofit

include("../../../MCMC/MCMC.jl")
include("autofit.jl")

function exp_cov(D::Matrix{Float64}, ϕ::Float64, α::Float64)
  return α * exp(-ϕ * D)
  #return α * exp(-ϕ * D .^ (1.5))
end

const dist = Euclidean()

# Assumes y,X are centered and scaled


function fit(y::Vector{Float64}, X::Matrix{Float64}, cs::Matrix{Float64},
             B::Int, burn::Int; printFreq=0,
             init::Vector{Float64}=zeros(3),
             a_σ::Float64=2.0, b_σ::Float64=1.0,
             a_ϕ::Float64=0.0, b_ϕ::Float64=10.0,
             a_a::Float64=2.0, b_a::Float64=1.0)

  assert(length(y) == size(X,1))

  const n = size(X,1)
  const D = pairwise(dist, X')
  const Iₙ = eye(n)

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
      const v = inv_trans_param(t_v)
      const K = exp_cov(D, v[2], v[3])
      return logpdf(MvNormal(K + Iₙ*v[1]), y)
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

function autofit(y::Vector{Float64}, X::Matrix{Float64}, B::Int; 
                 burn::Int=0, cs::Matrix{Float64}=eye(3),
                 printFreq::Int=0, init::Vector{Float64}=zeros(3),
                 a_σ::Float64=2.0, b_σ::Float64=1.0,
                 a_ϕ::Float64=0.0, b_ϕ::Float64=10.0,
                 a_a::Float64=2.0, b_a::Float64=1.0,
                 max_autotune::Int=20, 
                 window::Int=500,
                 target::Float64=.25,
                 target_lower::Float64=.25,
                 target_upper::Float64=.40,
                 k::Float64=2.5)

  function f(cs::Matrix{Float64}, B::Int, burn::Int, 
             init::Vector{Float64}, printFreq::Int)
    return fit(y,X,cs,B,burn,printFreq=printFreq,init=init,
               a_σ=a_σ, b_σ=b_σ, a_ϕ=a_ϕ, b_ϕ=b_ϕ, a_a=a_a, b_a=b_a)
  end

  return AUTOFIT(f,B,3,
                 burn=burn, printFreq=printFreq, max_autotune=max_autotune,
                 window=window, target=target, target_lower=target_lower,
                 target_upper=target_upper, k=k)
end



# predict function at new location (f)
# Write another pred function for observations (y)
function predict(post::Vector{Vector{Float64}},
                 y::Vector{Float64}, X_old::Matrix{Float64},
                 X_new::Matrix{Float64};response::String="mean")

  const n_new = size(X_new,1)
  const n_old = size(X_old,1)
  const D = pairwise(dist, [X_new; X_old]')
  const D_new = D[1:n_new,1:n_new]
  const D_old = D[n_new+1:end,n_new+1:end]
  const D_new_old = D[1:n_new,n_new+1:end]
  const Iₙ = eye(n_old)
  const Iₘ = eye(n_new)

  function pred(param::Vector{Float64})
    const sig2 = param[1]
    const phi = param[2]
    const a = param[3]

    const K_new = sym(exp_cov(D_new, phi, a))
    const K_old = sym(exp_cov(D_old, phi, a))
    # Note that Cov(y*,y) = Cov(f*,f) = Cov(f*,y) because y=f+ϵ and Cov(ϵᵢ,ϵⱼ)=0
    const C = exp_cov(D_new_old, phi, a)
    const MVN = if response == "obs"
      # y* | y
      CondNormal(y, zeros(n_old), zeros(n_new), 
                 sig2*Iₙ+K_old, sig2*Iₘ+K_new, C)
    else # default response is mean function f
      # f* | y
      # Note: (f*,y) ~ N(0, G)
      # where G11= K*, G12=C, G22=σ²Iₙ+K, and C is written above in code.
      CondNormal(y, zeros(n_old), zeros(n_new), 
                 sig2*Iₙ+K_old, K_new, C)
    end

    return rand( MVN )
  end

  return hcat(map(p->pred(p), post)...)
end

end #module
