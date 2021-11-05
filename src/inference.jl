#=
inference.jl; This file implements functions to compare two hyperdimensional vectors.
=#

using LinearAlgebra

LinearAlgebra.norm(hdv::AbstractHDV) = norm(hdv.v)

cos_sim(x::AbstractVector, y::AbstractVector) = dot(x, y) / (norm(x) * norm(y))

jacc_sim(x::AbstractVector, y::AbstractVector) = sum(maximum, zip(x, y)) / sum(minimum, zip(x, y))