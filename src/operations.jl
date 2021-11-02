#=
operations.jl; This file implements operations that can be done on hypervectors to enable them to encode text-based data.
=#

# Remark: use element-wise reduce, maybe using LazyArrays?


#=

| Operation            | symbol | remark                                                                                                          |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Bundling/aggregating | `+`    | combines the information of two vectors into a new vector similar to both                                       |
| Binding              | `*`    | mapping, combines the two vectors in something different from both, preserves distance, distributes of bundling |
| Shifting             | `Π`    | Permutation (in practice cyclic shifting), distributes over addition, conserves distance                        |
=#

"""
    grad2bipol(x::Number)

Maps a graded number in [0, 1] to the [-1, 1] interval.
"""
grad2bipol(x::Number) = 2x - one(x)


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Number) = (x + one(x)) / 2

three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x)-x) * y + x * (one(y)-y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))

aggfun(::AbstractHDV) = +
aggfun(::GradedHDV) = three_pi
aggfun(::GradedBipolarHDV) = three_pi_bipol

bindfun(::AbstractHDV) = *
bindfun(::BinaryHDV) = ⊻
bindfun(::GradedHDV) = fuzzy_xor
bindfun(::GradedBipolarHDV) = fuzzy_xor_bipol

neutralbind(hdv::AbstractHDV) = one(eltype(hdv))
neutralbind(hdv::BinaryHDV) = false
neutralbind(hdv::GradedHDV) = zero(eltype(hdv))
neutralbind(hdv::GradedBipolarHDV) = -one(eltype(hdv))

normalize!(::AbstractHDV, n) = nothing
normalize!(hdv::RealHDV, n) = (hdv.v ./= sqrt(n))
normalize!(hdv::BipolarHDV, n) = (hdv.v .= sign.(hdv.v))

function elementreduce!(f, itr, init)
    return foldl(itr; init) do acc, value
        acc .= f.(acc, value)
    end
end

# computes `r[i] = f(x[i], y[i+offset])`
# assumes postive offset (for now)
@inline function offsetcombine!(r, f, x, y, offset=0)
    @assert length(r) == length(x) == length(y)
    n = length(r)
    if offset==0
        r .= f.(x, y)
    else
        i′ = offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

# AGGREGATION
# -----------

# TODO: implement offsetadd/offsetprod

aggregate(hdvs::AbstractVector{<:AbstractHDV}) = aggregate!(similar(first(hdvs)), hdvs)

#aggregate(hdvs::AbstractHDV...) = aggregate!(similar(first(hdvs)), )

Base.:+(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = aggregate!(similar(hdv1), (hdv1, hdv2))

function aggregate!(r::AbstractHDV, hdvs)
    fill!(r.v, zero(eltype(r)))
    aggr = aggfun(r)
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, aggr, acc, value.v, value.offset)
    end
    normalize!(r, length(hdvs))
    return r
end

function aggregate!(r::BinaryHDV, hdvs)
    counts = zeros(Int, length(r))
    aggr = aggfun(r)
    foldl(hdvs, init=counts) do acc, value
        offsetcombine!(acc, aggr, acc, value.v, value.offset)
    end
    # use majority rule
    r.v .= counts .> length(hdvs) / 2
    return r
end

function aggregate!(r::GradedHDV, hdvs)
    # neutral element is 0.5 for graded
    fill!(r.v, one(eltype(r))/2)
    aggr = aggfun(r)
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, aggr, acc, value.v, value.offset)
    end
    normalize!(r, length(hdvs))
    return r
end


# BINDING
# -------

Base.bind(hdvs::AbstractVector{<:AbstractHDV}) = bind!(similar(first(hdvs)), hdvs)
Base.:*(hdv1::HDV, hdv2::HDV) where {HDV<:AbstractHDV} = bind!(similar(hdv1), (hdv1, hdv2))

function bind!(r::AbstractHDV, hdvs)
    fill!(r.v, neutralbind(r))
    binder = bindfun(r)
    foldl(hdvs, init=r.v) do acc, value
        offsetcombine!(acc, binder, acc, value.v, value.offset)
    end
    return r
end

# SHIFTING
# --------

#=
Base.circshift!(hdv::AbstractHDV, k) = (hdv.offset += k)

function Base.circshift(hdv::AbstractHDV, k)
    hdv = copy(hdv)
    hdv.offset += k
    return hdv
end
=#

function Π(hdv::AbstractHDV, k)
    hdv = copy(hdv)
    hdv.offset += k
    return hdv
end

function Π(hdv::RealHDV, k)
    return RealHDV(hdv.v, hdv.offset+k)
end