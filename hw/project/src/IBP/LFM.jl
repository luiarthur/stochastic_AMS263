"""
# LFM (Latent Feature Model)

Model: X = ZA + ϵ
"""
module LFM # Latent Feature Model

using Distributions
include("../../../MCMC/MCMC.jl")
include("IBP.jl")

immutable State
  A::Matrix{Float64}
  sig2::Float64
  alpha::Float64
  v::Vector{Float64}
  Z::Matrix{Int64}
end

const nullState = State(zeros(0,0), 0.0, 0.0, ones(0), ones(Int64,0,0))
const nullA = zeros(0,0)

function fit(X::Matrix{Float64}; K::Int=100, B::Int=1000, burn::Int=100,
             init::State=nullState, printFreq::Int=0, 
             a_sig::Float64=2.0, b_sig::Float64=1.0,
             a_a::Float64=1.0, b_a::Float64=.1, cs_a::Float64=1.0,
             cs_v::Matrix{Float64}=zeros(0,0),
             A_true::Matrix{Float64}=nullA)

  const (N,D) = size(X)
  const x = vec(X)
  const I_D = eye(D)
  const I_ND = eye(N*D)

  # prior β
  const zero_KD = zeros(K*D)
  const V_A = eye(K*D)*1E10

  # Candidate Sigma for v (vector)
  const CS_V = if cs_v == zeros(0,0)
    eye(K)
  else
    cs_v
  end

  function update(state::State)
    # A: Gibbs
    const A = if A_true != nullA
      vcat(A_true,zeros(K-size(A_true,1),D))
    else
      reshape(rand(MCMC.β_post(x, kron(I_D,state.Z), I_ND*state.sig2, 
                               zero_KD, V_A)), K,D)
    end

    # σ²: Gibbs
    const sig2 = rand(InverseGamma(a_sig + N*D/2, 
                                   b_sig + (vec(X-state.Z*A)'vec(X-state.Z*A)/2)[1]))

    # α: Metropolis
    const alpha = let
      lp(loga::Float64) = MCMC.lp_log_gamma(loga, a_a, b_a)
      ll(loga::Float64) = K*loga + exp(loga)*sum(log(state.v))

      exp(MCMC.metropolis(log(state.alpha), cs_a, ll, lp))
    end
    
    # v: Metropolis
    const v = let
      b(v::Vector{Float64}) = cumprod(v)
      const m = vec(sum(state.Z,1))

      function ll(logit_v::Vector{Float64}) 
        const v = MCMC.inv_logit.(logit_v)
        sum(m.*log(b(v)) + (N-m).*log(1-b(v)))
      end

      function lp(logit_v::Vector{Float64})
        sum(MCMC.lp_logit_unif.(logit_v) + MCMC.inv_logit.(logit_v).^alpha)
      end
      
      MCMC.inv_logit.(MCMC.metropolis(MCMC.logit.(state.v), CS_V, ll, lp))
    end

    # Z: Gibbs
    const Z = let
      function log_kern(j::Int, i::Int, k::Int, Z::Matrix{Int64})
        Z[i,k] = j
        const y = x - vec(Z*A)
        (-y'y/2sig2)[1]
      end

      const b = cumprod(v)
      const Z = state.Z

      for k in 1:K
        for i in 1:N
          const lp1 = log(b[k]) + log_kern(1,i,k,Z)
          const lp0 = log(1-b[k]) + log_kern(0,i,k,Z)
          const maxp = max(lp1,lp0)
          const p = exp(lp1-maxp) / (exp(lp1-maxp) + exp(lp0-maxp))

          Z[i,k] = p > rand() ? 1 : 0
        end
      end

      #Z
      IBP.lof(Z) # necessary?
    end

    return State(A, sig2, alpha, v, Z)
  end

  const INIT = if init == nullState
    State(zeros(K,D), 1.0, 1.0, rand(K), rand(0:1,N,K))
  else
    init
  end

  return MCMC.gibbs(INIT, update, B, burn, printFreq=printFreq)
end # end of fit function

# Note that julia matrices are column major

end # end of module LFM
