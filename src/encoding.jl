#=
encoding.jl; This file implements functions to encode token-based data, be it alphabets, strings or full classes of text.
=#

# SEQUENCE EMBEDDING

# Maybe solve with @generated?
# could be using a nested dictionary using fold


function ngrams_recursion(d::Dict{K,V}) where {K,V<:Int}
    result = Dict{K,Dict{K,V}}()
    for c in keys(d)
        result[c] = Dict(k=>v+Int(c) for (k, v) in d)
    end
    return result
end

function ngrams_recursion(d::Dict{K,V}, hdvs) where {K,V<:AbstractHDV}
    result = Dict{K,Dict{K,V}}()
    for c in keys(d)
        hdv = hdvs[c]
        result[c] = Dict(k=> hdv * Π(v, 1) for (k, v) in d)
    end
    return result
end

function ngrams_recursion(d::Dict{K,V}, args...) where {K,V<:Dict}
    result = Dict{K,Dict{K,V}}()
    for (c, dc) in d
        result[c] = ngrams_recursion(dc, args...)
    end
    return result
end

get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:Dict} = get_gram_embedding(sequence, i+1, embeddings[sequence[i]])
get_gram_embedding(sequence, i, embeddings::Dict{K,V}) where {K,V<:AbstractHDV} = embeddings[sequence[i]]




ngrams(sequence, K) = ngrams(NTuple{K,eltype(sequence)}, sequence, K)
ngrams(T::Type, sequence, K) = [T(sequence[i+k] for k in 0:K-1) for i in 1:length(sequence)-K+1]

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

#=

"""
This function encodes a given alphabet (collection of tokens) as hypervectors of the chosen type.\n

Input: 
    - a vector of chars denoting the alphabet, e.g. ['A', 'C', 'T', 'G'] or 'a':'z'.\n
    - the type of vector, either bipolar or binary. By default this is a bipolar vector.\n
Output: a dictionary with vector encodings for each token in the alphabet.
"""
function encode_alphabet(alphabet::Vector{Char}; vectortype="bipolar")
    if vectortype=="bipolar"
        return Dict((token=>rand(Int64[-1 1], HYPERVECTOR_DIM) for token in alphabet))
    elseif vectortype=="binary"
        return Dict((token=>convert(BitVector, rand(Bool, HYPERVECTOR_DIM)) for token in alphabet))
    else
        error("Please choose a valid vectortype: either bipolar or binary")
    end
end


"""
This function creates a hyperdimensional vector for a given input sequence. It uses the encoded alphabet to go over the sequence elementwise or in k-mer blocks.\n

Input:
    - a string of tokens to encode. \n
    - a dictionary with vector encodings for each token in the alphabet.\n
    - an integer k that defines the size of k-mers to aggregate over. This value defaults to 1.\n
Output: a hypervector encoding for the sequence.\n

DISCLAIMER: Removed the 'dim' argument from this function, as it has to be equal to the dimensionality of the alphabet hypervectors.

Additional comments: \n
    - kmer_encoding with a sum instead of mutiplication?\n
    - circshift over entire sequence with k=1.
"""
function encode_sequence(sequence::String, encoded_alphabet::Dict; k=1::Int)
    # precompute the first kmer-encoding as the starting point as both ones() and zeros() vectors will cause issues with binary hypervectors
    encoding = encoded_alphabet[sequence[1:1+(k-1)][end]]
    for kmer_index in 1:k-1
        encoding = multiply(encoding, rotate(encoded_alphabet[sequence[1:1+(k-1)][end-kmer_index]], kmer_index))
    end
    
    # iterate over the others
    for sliding_index in 2:length(sequence)-(k-1)
        kmer_encoding = encoded_alphabet[sequence[sliding_index:sliding_index+(k-1)][end]]
        for kmer_index in 1:k-1
            kmer_encoding = multiply(kmer_encoding, rotate(encoded_alphabet[sequence[sliding_index:sliding_index+(k-1)][end-kmer_index]], kmer_index))
        end
        #encoding = add(encoding, kmer_encoding)
        encoding = add(encoding, kmer_encoding)
    end
    if typeof(encoding) == BitVector
        return encoding
    elseif eltype(encoding) == Int
        return sign.(encoding)
    end
end

#=
"""
This function loops over a matrix of hyperdimensional vectors and its associated\n
classes and constructs a profile for each class by summing the corresponding HVs.\n
\n
Compared to traditional machine learning, this encoding of classes constitutes the\n
'learning', as hyperdimensional vectors of all classes are combined via elementwise\n
addition to learn a representation of the class a a whole.\n
\n
Input:\n
- encoding_matrix: matrix with encodings (#encodings x dim)\n
- classes: corresponding class labels (# encodings)\n
- max_iterations: # of max iterations for retraining\n
Output: dictionary of HVs for each of the classes\n
\n
Addition: don't subtract from all classes, only wrong one?\n
Addition: rethink retraining for multiclass, more complex measure?
"""
function encode_classes(encoding_matrix::Array, classes; max_iterations=25::Int)
    # initial encodings
    class_encodings = Dict()
    for row in 1:size(encoding_matrix)[1]
        if classes[row] in keys(class_encodings)
            class_encodings[classes[row]] += encoding_matrix[row,:]
        else
            class_encodings[classes[row]] = encoding_matrix[row,:]
        end
    end

    # retraining
    count_wrong = 10000
    stop = 0
    iteration = 0
    while (iteration <= max_iterations) & (count_wrong > 0) & (stop == 0)
        # loop over matrix
        count_wrong_iter = 0
        for row in 1:size(encoding_matrix)[1]
            distances = Dict()
            actual_class = classes[row]
            for (class, class_vector) in class_encodings # compute distances
                distances[class] = cosine_dist(encoding_matrix[row,:], class_vector)
            end
            minimal_class = findmin(distances)[2]

            if minimal_class != actual_class # if wrong, adjust
                count_wrong_iter += 1
                for key in keys(class_encodings)
                    if key != actual_class
                        class_encodings[key] -= encoding_matrix[row,:]
                    else
                        class_encodings[key] += encoding_matrix[row,:]
                    end
                end
            end
        end
        #println("it: ", iteration, "wrong: ", count_wrong, "wrong it: ", count_wrong_iter)

        # check convergence
        if count_wrong_iter < count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
            stop = 0
        elseif count_wrong_iter > count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
            stop = 0
        else
            stop = 1
        end
    end

    println("number of iterations: ", iteration)
    println("number of wrong class assignments: ", count_wrong)
    return class_encodings
end
=#

=#