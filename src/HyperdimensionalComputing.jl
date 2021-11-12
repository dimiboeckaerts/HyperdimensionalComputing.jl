module HyperdimensionalComputing

using Distances

export AbstractHDV, BinaryHDV, BipolarHDV,
    GradedBipolarHDV, RealHDV, GradedHDV
export offsetcombine, offsetcombine!
export aggregate, aggregate!, bind, bind!, Π, Π!, resetoffset!
export sequence_embedding, sequence_embedding!


include("vectors.jl")
include("operations.jl")
include("encoding.jl")
#include("predictors.jl")

end