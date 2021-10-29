module HyperdimensionalComputing

using Distances

export AbstractHDV, BinaryHDV, BipolarHDV, GradedBipolarHDV, RealHDV
export aggregate, aggregate!


include("vectors.jl")
include("operations.jl")
include("encoding.jl")
include("predictors.jl")

end