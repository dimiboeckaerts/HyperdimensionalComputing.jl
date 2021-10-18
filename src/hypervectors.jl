#=
hypervectors.jl; This file implements different types of hypervectors and allows to play around with their parameters.
=#

const HYPERVECTOR_DIM = convert(Int16, 10000)


abstract type Hypervector end


struct BipolarVector <: Hypervector
    vector::Vector{Int8}
    BipolarVector() = new(rand([-1 1], HYPERVECTOR_DIM))
end


struct BinaryVector <: Hypervector
    vector::Vector{Bool}
    BinaryVector() = new(rand(Bool, HYPERVECTOR_DIM))
end

