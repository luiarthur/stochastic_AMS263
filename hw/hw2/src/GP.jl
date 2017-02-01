module GP

using Distributions
using Distances

export fit

include("../../MCMC/MCMC.jl")

function exp_cov(D::Matrix{Float64}, ϕ::Float64, α::Float64)
  return α * exp(-ϕ * D)
  #return α * exp(-ϕ * D .^ (1.5))
end

const dist = Euclidean()

# Assumes y,X are centered and scaled


function fit(y::Vector{Float64}, X::Matrix{Float64}, cs::Vector{Float64},
             B::Int, burn::Int; printFreq=0,
             a_σ::Float64=2.0, b_σ::Float64=1.0,
             a_ϕ::Float64=0.0, b_ϕ::Float64=10.0,
             a_a::Float64=2.0, b_a::Float64=1.0)

  assert(length(y) == size(X,1))

  const n = size(X,1)
  const D = pairwise(dist, X')
  const cs_matrix = Matrix(Diagonal(cs))
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

    return inv_trans_param(MCMC.metropolis(trans_param(param), cs_matrix, ll, lp))
  end

  # param: [σ², ϕ, α]
  const init = [b_σ/(a_σ-1.0), (a_ϕ + b_ϕ)/2.0, b_a/(a_a-1.0)]

  const out = MCMC.gibbs(init, update, B, burn, printFreq=printFreq)
  println("\tAcceptance Rate: ", length(unique(out))/length(out))
  return out
end

sym(M::Matrix{Float64}) = (M + M') / 2

function predict(post::Vector{Vector{Float64}},
                 y::Vector{Float64}, X_old::Matrix{Float64},
                 X_new::Matrix{Float64})

  const n_new = size(X_new,1)
  const n_old = size(X_old,1)
  const D = pairwise(dist, [X_new; X_old]')
  const D_new = D[1:n_new,1:n_new]
  const D_old = D[n_new+1:end,n_new+1:end]
  const D_new_old = D[1:n_new,n_new+1:end]
  const Iₙ = eye(n_old)

  function pred(param::Vector{Float64})
    const sig2 = param[1]
    const phi = param[2]
    const a = param[3]

    const K_new = exp_cov(D_new, phi, a)
    const K_old = exp_cov(D_old, phi, a)
    const M = inv(sig2*Iₙ + K_old)
    const C = exp_cov(D_new_old, phi, a)
    return rand( MvNormal(C*M*y, sym(K_new-C*M*C')) )
  end

  return hcat(map(p->pred(p), post)...)
end

function predict_mean(post::Vector{Vector{Float64}},
                 y::Vector{Float64}, X::Matrix{Float64})

  const n= size(X,1)
  const D = pairwise(dist, X')
  const Iₙ = eye(n)

  function pred(param::Vector{Float64})
    const sig2 = param[1]
    const phi = param[2]
    const a = param[3]

    const K= exp_cov(D, phi, a)
    const S = sym(inv(Iₙ/sig2 + inv(K)))
    return rand( MvNormal(S*y/sig2, S) )
  end

  return hcat(map(p->pred(p), post)...)
end

end #module
