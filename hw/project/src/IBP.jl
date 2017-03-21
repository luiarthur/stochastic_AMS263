module IBP

using Distributions
using RCall
R"source('IBP.R')";
R"library(rcommon)";

export randIBP

function randIBP(alpha::Float64, N::Int, K::Int=20)

  const v = rand(Beta(alpha,1), K)
  const p = cumprod(v,1)
  const Z = [ p[k] > rand() ? 1 : 0 for i in 1:N, k in 1:K ]

  return Z
end

#=
randIBP(1.0, 30, 10)
=#

plot = R"plot.ibp";

@time Zs = [randIBP(5.4,100,30) for i in 1:10000]

stat = [mean(sum(Z,2)) for Z in Zs]
R"plotPost($stat)"

#for i in 1:1000
#  plot(Zs[i],show_y_axis=false, show_x_axis=false)
#end

end
