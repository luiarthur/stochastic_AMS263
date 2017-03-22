"""
My MCMC stuff
"""
module MCMC

import Distributions: logpdf, Normal, MvNormal, InverseGamma, Gamma, Logistic
export gibbs, metropolis, autotune


function gibbs{T}(init::T, update, B::Int, burn::Int; 
                  printFreq::Int=0)

  const out = Array{T,1}( (B+burn) )
  out[1] = init
  for i in 2:(B+burn)
    out[i] = update(out[i-1])
    if printFreq > 0 && i % printFreq == 0
      print("\rProgress: ",i,"/",B+burn)
    end
  end

  return out[ (burn+1):end ]
end



"""
univariate metropolis step
"""
function metropolis(curr::Float64, cs::Float64, 
                    logLike, logPrior)

  g(p::Float64) = logLike(p) + logPrior(p)

  const cand = rand( Normal(curr,cs) )

  const new_state = if g(cand) - g(curr) > log(rand())
    cand
  else
    curr
  end

  return new_state
end



"""
multivariate metropolis step
"""
function metropolis(curr::Vector{Float64}, candΣ::Matrix{Float64},
                    logLike, logPrior)

  g(p::Vector{Float64}) = logLike(p) + logPrior(p)

  const cand = rand( MvNormal(curr,candΣ) )

  new_state = if g(cand) - g(curr) > log(rand())
    cand
  else
    copy(curr)
  end

  return new_state
end

# For transforming bounded parameters to have infinite support
inv_logit(x::Float64,a::Float64=0.0,b::Float64=1.0) = (b*exp(x)+a) / (1+exp(x))

logit(p::Float64,a::Float64=0.0,b::Float64=1.0) = log( (p-a)/ (b-p) )

function lp_log_gamma(log_x::Float64, shape::Float64, rate::Float64)
  #shape*log(rate) - lgamma(shape) + shape*log_x - rate*exp(log_x)
  const log_abs_J = log_x
  return logpdf(Gamma(shape,1/rate), exp(log_x)) + log_abs_J
end

function lp_log_invgamma(log_x::Float64, a::Float64, b_numer::Float64)
  #a*log(b_numer) - lgamma(a) - a*log_x - b_numer*exp(-log_x)
  const log_abs_J = log_x
  return logpdf(InverseGamma(a,b_numer), exp(log_x)) + log_abs_J
end

#lp_logit_unif(logit_u::Float64) = logit_u - 2*log(1+exp(logit_u)) 
lp_logit_unif(logit_u::Float64) = logpdf(Logistic(), logit_u)


sym(M::Matrix{Float64}) = (M + M') / 2

# Distribution of y|x
function CondNormal(x::Vector{Float64}, μx::Vector{Float64}, μy::Vector{Float64}, 
                    Σx::Matrix{Float64}, Σy::Matrix{Float64}, Σyx::Matrix{Float64})

  const S = Σyx * inv(sym(Σx))

  return MvNormal(μy + S*(x-μx), sym(Σy-S*Σyx'))
end

"""
Model:
     y | β,X ~ Nₙ(Xβ, Σ)
         β   ~ Nₚ(β₀, V)

=>  y* | β,X ~ Nₙ₊ₚ(X* β, Σ*)
       p(β)  ∝ 1

where X* = X  , Σ* = Σ  0, and y* = y
           Iₚ        0  V           β₀

∴     β | y,X ~ N( (X*Σ*⁻¹X*)⁻¹X*'y* , (X*Σ*⁻¹X*)⁻¹)
"""
function β_post(y::Vector{Float64},  X::Matrix{Float64}, Σ::Matrix{Float64},
                β₀::Vector{Float64}, V::Matrix{Float64})

  const (N,P) = size(X)
  const zeroNP = zeros(N,P)

  const y1 = [y; β₀]
  const X1 = vcat(X,eye(P))
  const Σ1i = inv(vcat(hcat(Σ, zeroNP), hcat(zeroNP', V)))
  const Vi = sym(inv(X1' * Σ1i * X1))

  return MvNormal(Vi*X1'*y1, Vi)
end


"""
autotune(accept::Float64, target::Float64=0.25, k::Float64=2.5)

`accept`: current accpetance rate
`target`: target accpetance rate
`k`: some tuning parameter...

Returns a factor to multiply the current proposal step size by.
"""
function autotune(accept::Float64; target::Float64=0.25, k::Float64=2.5)
  return (1+(cosh(accept-target)-1)*(k-1) / 
          (cosh(target-ceil(accept-target))-1))^sign(accept-target)
end

end # module
