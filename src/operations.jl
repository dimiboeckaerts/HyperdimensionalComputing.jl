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


three_pi(x, y) = x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = x + y - x * y

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))


# AGGREGATION
# -----------

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
