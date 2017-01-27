include("../../MCMC/MCMC.jl")


function exp_cov(i::Int, j::Int;
                 σ²::Float64=1., ϕ::Float64=1., α::Float64=1.)

  return σ² * exp(-ϕ * abs(i-j)^α)
end


# Assumes y,X are centered and scaled

immutable State
  σ²::Float64
  ϕ::Float64
  α::Float64
end

function gp_reg(y::Vector{Float64}, X::Matrix{Float64};
                a_σ::Float64=2.0, b_σ::Float64=1.0,
                a_ϕ::Float64=0.0, b_ϕ::Float64=1.0,
                a_a::Float64=2.0, b_a::Float64=1.0,
                cov_fn=exp_cov)

  assert(length(y) == size(X,1))

  const (n,K) = size(X)

  function ll(s::State)
    # update σ² 

    # update ϕ

    # update α
  end

end
