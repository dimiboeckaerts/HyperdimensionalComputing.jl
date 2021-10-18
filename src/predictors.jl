#=
predictors.jl; This file implements methods to compare encoded strings to references and to make predictions from this comparison.
=#

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
function cosine_predict(encoding_matrix::Array, class_encodings::Dict)
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


### Add naive bayes predictor