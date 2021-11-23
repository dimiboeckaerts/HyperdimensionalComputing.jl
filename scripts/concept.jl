#=
# Hyperdimensional computing interface

date: 21 October 2021

author: Michiel Stock

In this document, I present a Julia interface for computing with hyperdimensional
vectors (HDV). These are vectors of large dimensionality (typically 10,000) that
allow for distributed and holographic representing, storing and computing with patterns.

## Hypervectors

We first define a root abstract `AbstractHDV` type as a subtype of `AbstractVector`,
meaning we inherit all behaviour expected from a vector.
=#

using LinearAlgebra

abstract type AbstractHDV{T} <: AbstractVector{T} end

Base.getindex(hdv::AbstractHDV, i) = hdv.v[(i+hdv.offset)%length(hdv)+1]
Base.size(hdv::AbstractHDV) = size(hdv.v)
Base.setindex!(hdv::AbstractHDV, val, i) = (hdv.v[(i+hdv.offset)%length(hdv)+1] = val)
normalize!(::AbstractHDV) = nothing  ## vectors have no normalization by default

# The first concrete type is `BipolarHDV`, which stores patterns with values $\{-1, 0, 1\}$.
# All concrete type store a vector with the values and an offset, allowing for inplace
# permutations.

mutable struct BipolarHDV <: AbstractHDV{Int}
    v::Vector{Int}
    offset::Int
    BipolarHDV(v::Vector, offset=0) = new(v, offset)
end

# We always provide a constructor with optinal dimensionality (n=10,000 by default) and
# a method `similar`.

BipolarHDV(n::Int=10_000) = BipolarHDV(rand((-1, 1), n))
Base.similar(hdv::BipolarHDV) = BipolarHDV(similar(hdv.v))

normalize!(hdv::BipolarHDV) = (hdv.v .= sign.(hdv.v))

# `BinaryHDV` contain binary vectors.

mutable struct BinaryHDV <: AbstractHDV{Bool}
    v::Vector{Bool}
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

GradedBipolarHDV(T::Type, n::Int=10_000) = GradedBipolarHDV(2rand(T, n).-1)
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

#=
## Computing with HDV

The following table contains the basic operations (two binary and one unary) that
are used to calculate with HDV. Important is that all these representations need to
be *reversible*.

| Operation            | symbol | remark                                                                                                          |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Bundling/aggregating | `+`    | combines the information of two vectors into a new vector similar to both                                       |
| Binding              | `*`    | mapping, combines the two vectors in something different from both, preserves distance, distributes of bundling |
| Shifting             | `Π`    | Permutation (in practice cyclic shifting), distributes over addition, conserves distance                        |

There are many options possible, so we use type-based dispatch to give the user control.

### Aggregation

Aggregation is at the highest level represented by `+`. (Question: should we introduce a specific symbol?)
=#

aggr(hdv1::AbstractHDV, hdv2::AbstractHDV) = aggr!(similar(hdv1), hdv1, hdv2)

Base.:+(hdv1::AbstractHDV, hdv2::AbstractHDV) = aggr(hdv1, hdv2)

aggr!(r::BipolarHDV, hdv1::BipolarHDV, hdv2::BipolarHDV) = (r .= hdv1 .+ hdv2)
aggr!(r::BinaryHDV, hdv1::BinaryHDV, hdv2::BinaryHDV) = (r .= hdv1 .& hdv2)  ## Conservative


function aggr!(r::RealHDV, hdv1::RealHDV, hdv2::RealHDV)
    r .=  hdv1 .+ hdv2
    r.norm = hdv1.norm + hdv2.norm
    return r
end

aggr!(r::GradedBipolarHDV, x::GradedBipolarHDV, y::GradedBipolarHDV) = (@. r = x * y / (x * y) + (1-x) * (1-y))

#= 

### Binding

=#

bind(hdv1::AbstractHDV, hdv2::AbstractHDV) = bind!(similar(hdv1), hdv1, hdv2)
Base.:*(hdv1::AbstractHDV, hdv2::AbstractHDV) = bind(hdv1, hdv2)

bind!(r::BipolarHDV, hdv1::BipolarHDV, hdv2::BipolarHDV) = (r .= hdv1 .* hdv2)
bind!(r::RealHDV, hdv1::RealHDV, hdv2::RealHDV) = (r .= hdv1 .* hdv2)
bind!(r::BinaryHDV, hdv1::BinaryHDV, hdv2::BinaryHDV) = (r .= hdv1 .⊻ hdv2)
bind!(r::GradedBipolarHDV, x::GradedBipolarHDV, y::GradedBipolarHDV) = (@. r = x + y - x * y)

#=
### Permutation

Vectors can be permuted by performing a cyclic shift.
=#

Base.circshift!(hdv::AbstractHDV, k) = (hdv.offset += k)

function Base.circshift(hdv::AbstractHDV, k)
    hdv = copy(hdv)
    hdv.offset += k
    return hdv
end

Π(hdv::AbstractHDV, k) = circshift(hdv, k)

#=

## Inference and comparision

To perform inference, one has to compute (dis)similarity between vectors.
=#


cos_sim(x::HDV, y::HDV) where {HDV<:AbstractHDV} = (x ⋅ y) / (norm(x.v) * norm(x.v))

jaccard(x::HDV, y::HDV) where {HDV<:BinaryHDV} = sum(xᵢ & yᵢ for (xᵢ, yᵢ) in zip(x, y)) / sum(xᵢ | yᵢ for (xᵢ, yᵢ) in zip(x, y))