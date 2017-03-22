sym(M::Matrix{Float64}) = (M + M') / 2


# Distribution of y|x
function CondNormal(x::Vector{Float64}, μx::Vector{Float64}, μy::Vector{Float64}, 
                    Σx::Matrix{Float64}, Σy::Matrix{Float64}, Σyx::Matrix{Float64})

  const S = Σyx * inv(sym(Σx))

  return MvNormal(μy + S*(x-μx), sym(Σy-S*Σyx'))
end


function AUTOFIT(f, B::Int, dim::Int;
                 burn::Int=0, 
                 printFreq::Int=0, 
                 max_autotune::Int=20, 
                 window::Int=500,
                 target::Float64=.30,
                 target_lower::Float64=.25,
                 target_upper::Float64=.40,
                 k::Float64=2.5)

  function adjust(COR::Matrix{Float64},multiplier::Float64=1.0,it::Int=0,
                  INIT::Vector{Float64}=zeros(dim))

    const out = f(COR*multiplier,window,0,INIT,printFreq)
    const acc_rate = length(unique(out)) / length(out)

    if target_lower<acc_rate<target_upper || it==max_autotune
      const CS = COR*multiplier
      return (f(CS, B, burn, INIT, printFreq),CS)
    else
      const new_INIT = out[end]
      const params = hcat(out...)'
      const new_COR = cor(params)
      const new_cs_multiplier = MCMC.autotune(acc_rate,target=target,k=k)
      return adjust(new_COR, new_cs_multiplier*multiplier, it+1, new_INIT)
    end
  end # adjust

  return adjust(eye(dim))
end

"""
Sharman-Woodbury Inverse
see: http://luiarthur.github.io/ucsc_notes/advBayesComputing/17/
"""
function sw_inv(sig2::Float64, C::Matrix{Float64}, K::Matrix{Float64})
  const N = size(C,1)
  return eye(N)/sig2 - C/sig2 *inv(K+ C'C/sig2)* C'/sig2
end
function sw_logdet(sig2::Float64, C::Matrix{Float64}, K::Matrix{Float64})
  const N = size(C,1)
  return logdet(K + C'C/sig2) + N*log(sig2) - logdet(K)
end
