"""
My MCMC stuff
"""
module MCMC

import Distributions
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

  const cand = rand( Distributions.Normal(curr,cs) )

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

  const cand = rand( Distributions.MvNormal(curr,candΣ) )

  new_state = if g(cand) - g(curr) > log(rand())
    cand
  else
    copy(curr)
  end

  return new_state
end


end # module

