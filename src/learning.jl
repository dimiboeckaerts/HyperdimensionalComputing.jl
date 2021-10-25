#=
encoding.jl; This file implements functions to encode token-based data, be it alphabets, strings or full classes of text.
=#

const HYPERVECTOR_DIM = convert(Int16, 10_000)

"""
This function encodes a given set of tokens/items as hypervectors of the chosen type.
It creates the item memory needed to construct new hypervectors for sequences from.

Input: 
    - items: a vector of chars denoting the alphabet, e.g. ['A', 'C', 'T', 'G'] or 'a':'z'.
    - the type of vector, either bipolar or binary. By default this is a bipolar vector.
Output: the item memory as a dictionary of vector encodings for each item.
"""
function encode_items(items::Vector{Char}; vectortype="bipolar")
    if vectortype=="bipolar"
        return Dict((item=>rand((-1, 1), HYPERVECTOR_DIM) for item in items))
    elseif vectortype=="binary"
        return Dict((item=>convert(BitVector, rand(Bool, HYPERVECTOR_DIM)) for item in items))
    else
        error("Please choose a valid vectortype: either bipolar or binary")
    end
end


"""
This function creates a hyperdimensional vector for a given input sequence. 
It uses the encoded item memory to go over the sequence elementwise or in k-mer blocks.
Currently, only a k-mer/n-gram method is available to construct hypervectors from. The method
multiplies HDVs within each k-mer and adds HDVs across subsequent k-mers. Finally, the resulting
HDV is normalized.

Input:
    - sequence: a string of items to encode.
    - item_memory: item memory of vector encodings for each item.
    - k: integer that defines the size of k-mers to loop over (default = 1).
Output: a hypervector encoding for the sequence.

Additional comments:
    - circshift over entire sequence with k=1.
"""
function encode_sequence(sequence, item_memory::Dict; k::Int=1)
    # precompute the first kmer-encoding as the starting point as both ones() and zeros() vectors will cause issues with binary hypervectors
    encoding = item_memory[sequence[1:1+(k-1)][end]]
    for kmer_index in 1:k-1
        encoding = multiply(encoding, rotate(item_memory[sequence[1:1+(k-1)][end-kmer_index]], kmer_index))
    end
    
    # iterate over the others
    for sliding_index in 2:length(sequence)-(k-1)
        kmer_encoding = item_memory[sequence[sliding_index:sliding_index+(k-1)][end]]
        for kmer_index in 1:k-1
            kmer_encoding = multiply(kmer_encoding, rotate(item_memory[sequence[sliding_index:sliding_index+(k-1)][end-kmer_index]], kmer_index))
        end
        encoding = add(encoding, kmer_encoding)
    end
    if typeof(encoding) == BitVector
        return encoding
    elseif eltype(encoding) == Int
        return sign.(encoding)
    end
end


"""
This function loops over a matrix of hyperdimensional vectors and its associated
classes and constructs a profile for each class by summing the corresponding HVs.
In HDC-terminology, this function computes the associative memory of the different classes.

Compared to traditional machine learning, this encoding of classes constitutes the
'learning', as hyperdimensional vectors of all classes are combined via elementwise
addition to learn a representation of each class as a whole.

Input:
- encoding_matrix: matrix with encodings (#encodings x dim)
- classes: corresponding class labels (# encodings)
- retraining_iterations: max # of iterations for retraining
Output: dictionary of HVs for each of the classes

Addition: don't subtract from all classes, only wrong one?
Addition: rethink retraining for multiclass, more complex measure?
"""
function encode_associative_memory(encoding_matrix, classes; retraining_iterations::Int=25)
    # construct initial associative memory of classes
    associative_memory = Dict() # TO DO: to matrix
    for row in 1:size(encoding_matrix)[1]
        if classes[row] in keys(associative_memory)
            associative_memory[classes[row]] = add(associative_memory[classes[row]], encoding_matrix[row,:]) # NORMALIZE HERE?
        else
            associative_memory[classes[row]] = encoding_matrix[row,:]
        end
    end

    # retraining
    count_wrong = 10000
    iteration = 1
    while (iteration <= retraining_iterations) & (count_wrong > 0)
        # loop over matrix
        count_wrong_iter = 0
        for row in 1:size(encoding_matrix)[1]
            distances = Dict((class=>cosine_dist(encoding_matrix[row,:], class_vector) 
                            for (class, class_vector) in associative_memory))
            actual_class = classes[row]
            minimal_class = findmin(distances)[2]

            if minimal_class != actual_class # if wrong, adjust
                count_wrong_iter += 1
                for key in keys(associative_memory)
                    key != actual_class ? associative_memory[key] = subtract(associative_memory[key], encoding_matrix[row,:]) : class_encodings[key] = add(associative_memory[key], encoding_matrix[row,:])
                end
            end
        end

        # check convergence
        if count_wrong_iter < count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
        elseif count_wrong_iter > count_wrong
            count_wrong = count_wrong_iter
            iteration += 1
        else
            iteration = retraining_iterations + 1 # break
        end
    end

    println("number of iterations: ", iteration)
    println("number of wrong class assignments: ", count_wrong)
    return associative_memory
end
