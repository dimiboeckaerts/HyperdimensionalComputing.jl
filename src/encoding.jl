#=
encoding.jl; This file implements functions to encode token-based data, be it alphabets, strings or full classes of text.
=#

# SEQUENCE EMBEDDING

# precomputing of the n-grams

ngrams_recursion(d::AbstractDict{K,V}, hdv::AbstractHDV) where {K,V<:AbstractHDV} = 
    Dict(k => hdv * Π(v, 1) for (k, v) in d)

ngrams_recursion(d::AbstractDict{K,V}, hdv::AbstractHDV) where {K,V<:AbstractDict} = 
    Dict(k => ngrams_recursion(v, hdv) for (k, v) in d)

function ngrams_recursion(d::AbstractDict{K,V}, hdvs) where {K,V}
    result = Dict{K,Dict{K,V}}()
    for k in keys(d)
        result[k] = ngrams_recursion(d, hdvs[k])
    end
    return result
end

compute_1_grams(hdvs, alphabet=1:length(hdvs)) = Dict(zip(alphabet, hdvs))
compute_2_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_1_grams(hdvs, alphabet), hdvs)
compute_3_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_2_grams(hdvs, alphabet), hdvs)
compute_4_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_3_grams(hdvs, alphabet), hdvs)
compute_5_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_4_grams(hdvs, alphabet), hdvs)
compute_6_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_5_grams(hdvs, alphabet), hdvs)
compute_7_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_6_grams(hdvs, alphabet), hdvs)
compute_8_grams(hdvs, alphabet=1:length(hdvs)) = ngrams_recursion(compute_7_grams(hdvs, alphabet), hdvs)

get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:Dict} =
            get_gram_embedding(sequence, i+1, embeddings[sequence[i]])
get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:AbstractHDV} = embeddings[sequence[i]]

order(embeddings::Dict{K,V}) where {K,V<:Dict} = order(first(embeddings::Dict{K,V})[2]) + 1
order(embeddings) = 1

function sequence_embedding!(result::AbstractHDV, sequence, token_vectors, w=3)
    fill!(result.v, zero(eltype(result)))
    tmp = similar(result)
    n = length(sequence)
    for i in 1:n-w
        fill!(tmp.v, neutralbind(tmp))
        for k in (w-1):-1:0
            v = token_vectors[sequence[i+k]]
            offsetcombine!(tmp.v, bindfun(tmp), tmp.v, v.v, v.offset + k)
        end
        offsetcombine!(result.v, aggfun(result), result.v, tmp.v, 0)
    end
    normalize!(result, length(sequence)-w)
    return result
end

# TODO change name?
function sequence_embedding_precomp!(result::AbstractHDV, sequence, ngrams_embedding)
    fill!(result.v, zero(eltype(result)))
    n = length(sequence)
    w = order(ngrams_embedding)
    for i in 1:n-w
        hdv = get_gram_embedding(sequence, i, ngrams_embedding)
        offsetcombine!(result.v, aggfun(result), result.v, hdv.v, 0)
    end
    normalize!(result, length(sequence)-w)
    return result
end

function sequence_embedding!(result::BinaryHDV, sequence, token_vectors, w=3)
    count = zeros(Int, length(result))
    fill!(result.v, zero(eltype(result)))
    tmp = similar(result)
    n = length(sequence)
    for i in 1:n-w
        fill!(tmp.v, neutralbind(tmp))
        for k in (w-1):-1:0
            v = token_vectors[sequence[i+k]]
            offsetcombine!(tmp.v, bindfun(tmp), tmp.v, v.v, v.offset + k)
        end
        offsetcombine!(count, +, count, tmp.v, 0)
    end
    result.v = count .> div(n-w, 2)
    return result
end



sequence_embedding(sequence, token_vectors::AbstractVector{V}, args...) where {V<:AbstractHDV} = 
                sequence_embedding!(similar(first(token_vectors)), sequence, token_vectors, args...)
    
sequence_embedding(sequence, token_vectors::Dict{T,V}, args...) where {T,V<:AbstractHDV} = 
                sequence_embedding!(similar(first(token_vectors)[2]), sequence, token_vectors, args...)

