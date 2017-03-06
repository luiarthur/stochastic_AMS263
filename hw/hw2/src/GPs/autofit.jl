function AUTOFIT(f, B::Int, dim::Int;
                 burn::Int=0, 
                 printFreq::Int=0, 
                 max_autotune::Int=20, 
                 window::Int=500,
                 target::Float64=.25,
                 target_lower::Float64=.25,
                 target_upper::Float64=.40,
                 k::Float64=2.5)

  function adjust(COR::Matrix{Float64},multiplier::Float64=1.0,it::Int=0,
                  INIT::Vector{Float64}=zeros(dim))

    const out = f(COR*multiplier,window,0,INIT,printFreq)
    const acc_rate = length(unique(out)) / length(out)

    if target_lower<acc_rate<target_upper || it==max_autotune
      #return (COR*multiplier, INIT)
      return f(COR*multiplier, B, burn, INIT, printFreq)
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
