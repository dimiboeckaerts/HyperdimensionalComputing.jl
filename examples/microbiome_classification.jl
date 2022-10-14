#=
microbiome_classification.jl; this script shows the application of hypervectors to a classification of microbiome disease profiles.
=#
using Pkg
Pkg.activate(".")
using CSV, DataFrames, HyperdimensionalComputing, MultivariateStats, Plots, Printf, StatsBase



# read in data
@info "Reading in & formatting data..."
meta = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.mf.csv", DataFrame)
sort!(meta, :"#SampleID")
counts = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.csv", DataFrame)
sort!(counts, :"#SampleID")
taxa = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.tax.csv", DataFrame, missingstring=["k__", ""])

# format data
@assert all(meta[!, "#SampleID"] == counts[!, "#SampleID"])
labels = map(x->isequal(x, "CD"), meta[!, "label"])
@printf("There are %d negative and %d positive samples.\n", values(countmap(labels))...)
relative_count_mat = Matrix(counts[!, 2:end])

"""
function to tranform relative (continuous between 0 and 1) abundances to integer counts. To this end,
the lowest non-zero abundance in each sample is set to a singleton. 
"""
function integer_transform_counts(relative_counts::Matrix)
    absolute_counts = zeros(Integer, size(relative_counts))
    for idx in 1:size(relative_counts, 1)
        smallest_non_zero = minimum(relative_counts[idx, iszero.(iszero.(relative_counts[idx, :]))])
        absolute_counts[idx, :] = round.((1 / smallest_non_zero) * relative_counts[idx, :])
    end
    return absolute_counts
end

absolute_count_mat = integer_transform_counts(relative_count_mat)



# encode OTUs
@info "Generating random hypervectors..."
encoded_taxa = Dict(x => BipolarHDV() for x in taxa[!, "otuID"])



# encode samples
@info "Encoding microbial counts..."
encoded_samples = []
for sample in 1:size(absolute_count_mat, 1)
    sample_encoding = similar(encoded_taxa[taxa[1, "otuID"]])
    for microbe in 1:size(absolute_count_mat, 2)
        count_int = absolute_count_mat[sample, microbe]
        microbe_encoding = encoded_taxa[taxa[microbe, "otuID"]]
        abundance_encoding = similar(microbe_encoding)
        for i in 1:count_int
            abundance_encoding =+ microbe_encoding
        end
        sample_encoding += abundance_encoding
    end
    push!(encoded_samples, sample_encoding)
end
encoded_samples = reduce(hcat, encoded_samples)



# encode samples
@info "Ordination..."
pca = fit(PCA, encoded_samples; pratio=1, maxoutdim=4)
encodings_transformed = projection(pca)' * (encoded_samples .- mean(pca))
h = plot(encodings_transformed[1,:], encodings_transformed[2,:], seriestype=:scatter, label="")
explained_var = principalvars(pca) ./ tvar(pca) * 100
xlab = string("PC1 [", round(explained_var[1]; digits=2), "%]")
ylab = string("PC2 [", round(explained_var[2]; digits=2), "%]")
plot!(xlabel=xlab, ylabel=ylab, framestyle=:box)
