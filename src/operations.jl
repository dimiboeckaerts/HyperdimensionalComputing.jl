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

# TODO: fix special case x=1, y=-1
three_pi(x, y) = abs(x-y)==1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = x + y - x * y

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))

function elementreduce!(f, itr, init)
    return foldl(itr; init) do acc, value
        acc .= f.(acc, value)
    end
end


# AGGREGATION
# -----------

aggregate(hdvs::AbstractVector{<:AbstractHDV}) = aggregate!(similar(first(hdvs)), hdvs)

#aggregate(hdvs::AbstractHDV...) = aggregate!(similar(first(hdvs)), )

Base.:+(hdv1::AbstractHDV, hdv2::AbstractHDV) = aggregate(similar(hdv1, hdv2))

# aggregation of bipolar vectors uses the majority rule
# TODO: unsafe for offsets!
function aggregate!(r::BipolarHDV, hdvs)
    fill!(r.v, zero(eltype(r)))
    foldl(hdvs, init=r.v) do acc, value
        if value.offset==0
            acc .= acc .+ value.v
        else
            acc .= acc .+ value
        end
    end
    r.v .= sign.(r.v)
    return r
end

function aggregate!(r::BinaryHDV, hdvs)
    counts = zeros(Int, length(r))
    foldl(hdvs, init=counts) do acc, value
        if value.offset==0
        acc .= acc .+ value.v
        else
            acc .= acc .+ value
        end
    end
    # use majority rule
    r.v .= counts .> length(hdvs) / 2
    return r
end

function aggregate!(r::GradedBipolarHDV, hdvs)
    fill!(r.v, zero(eltype(r)))
    foldl(hdvs, init=r.v) do acc, value
        if value.offset==0
            acc .= three_pi_bipol.(acc, value.v)
        else
            acc .= three_pi_bipol.(acc, value)
        end
    end
    return r
end

function aggregate!(r::RealHDV, hdvs; normalize=true)
    fill!(r.v, zero(eltype(r)))
    foldl(hdvs, init=r.v) do acc, value
        if value.offset==0
            acc .= acc .+ value.v
        else
            acc .= acc .+ value
        end
    end
    if normalize
        r.v ./= sqrt(length(hdvs))
    end
    return r
end

function aggregate!(r::RealHDV, hdvs, weights; normalize=false)
    fill!(r.v, zero(eltype(r)))
    foldl(zip(hdvs, weights), init=r.v) do acc, (value, weight)
        if value.offset==0
            acc .= acc .+ sqrt(weight) .* value.v
        else
            acc .= acc .+ sqrt(weight) .* value
        end
    end
    if normalize
        r.v ./= sqrt(sum(weights))
    end
    return r
end


# BINDING
# -------


# SHIFTING
# --------

Base.circshift!(hdv::AbstractHDV, k) = (hdv.offset += k)

function Base.circshift(hdv::AbstractHDV, k)
    hdv = copy(hdv)
    hdv.offset += k
    return hdv
end

Π(hdv::AbstractHDV, k) = circshift(hdv, k)
