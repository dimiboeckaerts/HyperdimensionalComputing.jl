module HyperdimensionalComputing

"""
This function encodes an alphabet of characters into hyperdimensional\n
vectors (HVs) as a starting point to encode text or sequences into HVs.\n
The vectors are bipolar vectors.\n

Input: an alphabet as vector of strings; e.g. ["A", "B", "C"]\n
Output: a dictionary with HVs for each of the characters
"""
function encode_alphabet(alphabet::Vector{String}; dim=10000::Int)
    encodings = Dict()
    for character in alphabet
        encodings[character] = rand([-1 1], dim)
    end
    return encodings
end

"""
This function creates a hyperdimensional vector for a given input sequence.\n
It uses the encoded alphabet to go over the sequence elementwise or in k-mer blocks.\n
\n
Input:\n
- sequence to encode as string\n
- encoded_alphabet: dictionary of encoded characters\n
- k: k that defines the length of a k-mer\n
Output: an encoding for the sequence\n
\n
Addition: kmer_encoding with a sum instead of mutiplication?\n
Addition: circshift over entire sequence with k=1\n
Addition: bipolarize vector again at the end?
"""
function encode_sequence(sequence::String, encoded_alphabet::Dict; k=1::Int, dim=10000::Int)
    kmers = [sequence[i:i+k-1] for i in 1:length(sequence)-k+1]
    encoding = zeros(Int, dim)
    for kmer in kmers
        kmer_encoding = ones(Int, dim)
        for i in 1:k
            kmer_encoding .*= circshift(encoded_alphabet[string(kmer[i])], i-1)
        end
        encoding += kmer_encoding
    end
    return encoding
end

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

"""
This function makes predictions of the given encodings by comparing them to the\n
class encodings. The cosine distance is computed for each of the class encodings\n
and the class is returned as prediction for which the distance is the smallest.\n
\n
Input:\n
- encoding_matrix: HVs of 'testsamples' in a matrix to make predictions for\n
- class_encodings: HVs of the classes to compare with\n
Output: predictions for each of the rows of the matrix
"""
function make_predictions(encoding_matrix::Array, class_encodings::Dict)
    predictions = []
    for row in 1:size(encoding_matrix)[1]
        distances = Dict()
        test_sample = encoding_matrix[row, :]
        for (class, class_vector) in class_encodings # compute distances
            distances[class] = cosine_dist(test_sample, class_vector)
        end
        prediction = findmin(distances)[2]
        push!(predictions, prediction)
    end
    return predictions
end

end # module