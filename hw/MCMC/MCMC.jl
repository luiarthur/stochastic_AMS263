"""
My MCMC stuff
"""
module MCMC

import Distributions: logpdf, Normal, MvNormal, InverseGamma, Gamma, Logistic
export gibbs, metropolis


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


end # module
