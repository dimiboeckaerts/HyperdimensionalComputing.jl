module HyperdimensionalComputing

using Distances

export AbstractHDV, BinaryHDV, BipolarHDV,
    GradedBipolarHDV, RealHDV, GradedHDV
export aggregate, aggregate!, bind, bind!


include("vectors.jl")
include("operations.jl")
include("encoding.jl")
include("predictors.jl")

end