module HyperdimensionalComputing

using Distances

export HYPERVECTOR_DIM, Hypervector, BipolarVector, BinaryVector
export multiply, rotate, add
export encode_alphabet, encode_sequence
export cosine_predict

include("operators.jl")
include("learning.jl")
include("predictors.jl")

end