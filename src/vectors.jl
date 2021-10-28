#=
vectors.jl; Implements the interface for HDV
=#

using LinearAlgebra

abstract type AbstractHDV{T} <: AbstractVector{T} end

Base.getindex(hdv::AbstractHDV, i) = hdv.v[(i+hdv.offset)%length(hdv)+1]
Base.size(hdv::AbstractHDV) = size(hdv.v)
Base.setindex!(hdv::AbstractHDV, val, i) = (hdv.v[(i+hdv.offset)%length(hdv)+1] = val)
normalize!(::AbstractHDV) = nothing  ## vectors have no normalization by default

getvector(hdv::AbstractHDV) = hdv.v

# We always provide a constructor with optinal dimensionality (n=10,000 by default) and
# a method `similar`.

mutable struct BipolarHDV <: AbstractHDV{Int}
    v::Vector{Int}
    offset::Int
    BipolarHDV(v::Vector, offset=0) = new(v, offset)
end

BipolarHDV(n::Int=10_000) = BipolarHDV(rand((-1, 1), n))
Base.similar(hdv::BipolarHDV) = BipolarHDV(similar(hdv.v))

normalize!(hdv::BipolarHDV) = (hdv.v .= sign.(hdv.v))

# `BinaryHDV` contain binary vectors.

mutable struct BinaryHDV <: AbstractHDV{Bool}
    v::BitVector
    offset::Int
    BinaryHDV(v::AbstractVector, offset=0) = new(v, offset)
end

BinaryHDV(n::Int=10_000) = BinaryHDV(rand(Bool, n))
Base.similar(hdv::BinaryHDV) = BinaryHDV(similar(hdv.v))

# GradedBipolarHDV are vectors in $[-1, 1]^n$, allowing for graded relations.


mutable struct GradedBipolarHDV{T<:Real} <: AbstractHDV{T}
    v::Vector{T}
    offset::Int
    GradedBipolarHDV(v::AbstractVector, offset=0) = new{eltype(v)}(v, offset)
end

GradedBipolarHDV(T::Type, n::Int=10_000) = GradedBipolarHDV(2rand(T, n) .- 1)
GradedBipolarHDV(n::Int=10_000) = GradedBipolarHDV(Float32, n)

Base.similar(hdv::GradedBipolarHDV) = GradedBipolarHDV(similar(hdv.v))

normalize!(hdv::GradedBipolarHDV)= (hdv.v .= camp.(hdv.v, -1, 1))

# Finally, `RealHDV` contain real values, drawn from a standard normal distribution
# by default.

mutable struct RealHDV{T<:Real} <: AbstractHDV{T}
    v::Vector{T}
    offset::Int
    RealHDV(v::Vector{T}, offset=0) where {T} = new{T}(v, offset)
end

RealHDV(n::Int=10_000) = RealHDV(randn(n), 1.0)

normalize!(hdv::RealHDV) = (hdv.v .*=  √(length(hdv) / sum(abs2, hdv.v)))

Base.similar(hdv::RealHDV) = RealHDV(similar(hdv.v))