#=
learning.jl; Provides the basic functions to do learning.
=#

train(y, hdvs) = Dict(c=>aggregate(hdvs[y.==c]) for c in unique(y)) 

predict(v::AbstractHDV, centers) = maximum((similarity(v, xcᵢ), yᵢ) for (yᵢ, xcᵢ) in centers)[2]

function retrain!(centers, y, hdvs; niters=100)
    nothing
end