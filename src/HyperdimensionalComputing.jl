module HyperdimensionalComputing

using Distances

export HYPERVECTOR_DIM, Hypervector, BipolarVector, BinaryVector
export multiply, aggregate, rotate
export encode_alphabet, encode_sequence
export cosine_predict

include("operations.jl")
include("encoding.jl")
include("predictors.jl")

end