#=
microbiome_classification.jl; this script shows the application of hypervectors to a classification of microbiome disease profiles.
=#
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using CSV, DataFrames, HyperdimensionalComputing, MLJ, MultivariateStats, NaiveBayes, Plots, Printf, StatsBase



### some useful functions
"""
Function to print out the accuracy and F-score based on given labels and predictions.
"""
function quickeval(pred_labels, true_labels)
    tp_ = [isequal(pred_labels[idx], 1) for idx in 1:length(true_labels) if isequal(true_labels[idx], 1)]
    tn_ = [isequal(pred_labels[idx], 0) for idx in 1:length(true_labels) if isequal(true_labels[idx], 0)]
    TP = sum(tp_);
    FP = length(tp_) - TP;
    TN = sum(tn_);
    FN = length(tn_) - TN;

    accuracy = round((TP + TN)/length(true_labels) * 100; digits=2)
    fscore = round(TP/(TP + (1/2)*(FP + FN)); digits=2)
    println("\tAccuracy: $accuracy%")
    println("\tF-score: $fscore")
    println(
            """
            \t  \t1\t \t0\t         
            \t  ------------------------------
            \t1|\t$TP\t|\t$FN\t|
            \t |--------------|---------------|
            \t0|\t$FP\t|\t$TN\t|
            \t  ------------------------------
            """
            )
end

"""
Function to transform relative or absolute abundances to a binary presence/absence.
"""
function binary_transform_counts(counts::Matrix)
    return relative_count_mat .> 0
end

"""
Function to tranform relative (continuous between 0 and 1) abundances to integer counts. To this end,
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

"""
Function to tranform absolute (integer) coutns to normalized, relative counts. Currently,
only a sum-normalisation is implemented, though ideally a VST should also be implemented.
"""
function normalize_counts(absolute_counts::Matrix; method="sum")
    @assert method ∈ ["sum", "VST"]
    normalized_counts = zeros(Integer, size(absolute_counts))
    if method == "sum"
        totals = sum(absolute_counts, dims=2)
        for idx in 1:size(absolute_counts, 1)
            normalized_counts[idx, :] = absolute_counts[idx,  :] ./ totals[idx]
        end
        return normalized_counts
    else
        @error "VST is not yet implemented"
    end
end

"""
Function to transform microbial abundances to the desired format. The grain argument 
refers to the resolution to which abundances are transformed: 
    - binary = micro-organisms will be represented as present or absent
    - relative = micro-organisms will be represented by their normalized proportion of 
                 the entire sample.
    - absolute = micro-organisms will be represented by their absolute sequence count.
"""
function transform_counts(counts::Matrix; grain::String="absolute")
    @assert grain ∈ ["binary", "relative", "absolute"] 
    if grain == "binary"
        return binary_transform_counts(counts)
    else
        original_counts_are_relative = all(relative_count_mat .<= 1)
        if ((original_counts_are_relative && grain == "relative") || (!original_counts_are_relative && grain == "absolute"))
            return counts
        elseif (original_counts_are_relative && grain == "absolute")
            return integer_transform_counts(counts)
        elseif (!original_counts_are_relative && grain == "relative")
            return normalize_counts(counts)
        end
    end
end

"""
Function to encode microbial counts matrices into hyperdimensional matrices. The grain argument 
refers to the resolution with which abundances are encoded: 
    - binary = microbial presence or absence will be encoded
    - relative = the relative, continuous (between 0 and 1) abundances will be encoded.
    - absolute = the absolute, integer counts will be encoded.
"""
function encode_counts(counts::Matrix, tax_ordered::Vector; grain::String="absolute")
    @assert grain ∈ ["binary", "relative", "absolute"] 
    n, p = size(counts)

    # set appropriate grain
    vector_type = Dict("binary" => BinaryHDV, "relative" => RealHDV, "absolute" => BinaryHDV)[grain]
    grained_counts = transform_counts(counts; grain=grain)
    
    # encode each micro-organism
    @info "Generating random hypervectors..."
    #encoded_taxa = [vector_type() for x in 1:p]
    encoded_taxa = Dict(x => vector_type() for x in tax_ordered)

    # encode samples
    @info "Encoding $grain microbial counts..."
    encoded_samples = Vector{vector_type}()
    for sample_idx in 1:n
        if (grain == "absolute")
            sample_encoding = similar(encoded_taxa[first(tax_ordered)])
            for microbe_idx in 1:p
                count_int = grained_counts[sample_idx, microbe_idx]
                microbe_encoding = encoded_taxa[tax_ordered[microbe_idx]]
                abundance_encoding = similar(microbe_encoding)
                
                for i in 1:count_int
                    abundance_encoding =+ microbe_encoding
                end
                sample_encoding += abundance_encoding #bundling
                #sample_encoding *= abundance_encoding #binding
            end
        end
        push!(encoded_samples, sample_encoding)
    end
    return encoded_samples
end



### read in data
@info "Reading in & formatting data..."
meta = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.mf.csv", DataFrame)
sort!(meta, :"#SampleID")
counts = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.csv", DataFrame)
sort!(counts, :"#SampleID")
taxa = CSV.read("./data/microbiome/Gevers2014_IBD_ileum.tax.csv", DataFrame, missingstring=["k__", ""])

### format data
@assert all(meta[!, "#SampleID"] == counts[!, "#SampleID"])
labels = map(x->Integer(isequal(x, "CD")), meta[!, "label"]) |> Vector
color_labels = map(x->(isequal(x, "CD")) ? :blue : :orange, meta[!, "label"]) |> Vector
@printf("\tThere are %d negative and %d positive samples.\n", values(countmap(labels))...)
relative_count_mat = Matrix(counts[!, 2:end])
absolute_count_mat = integer_transform_counts(relative_count_mat)




### encode samples
encoded_samples = encode_counts(relative_count_mat, taxa[!, :otuID]; grain="absolute")
encoded_samples_mat = reduce(hcat, encoded_samples)



### class encodings
@info "Encoding classes..."
train_idx, test_idx = partition(eachindex(labels), 0.8, rng=123, stratify=labels)
Xtrain, Xtest, ytrain, ytest = encoded_samples[train_idx], encoded_samples[test_idx], labels[train_idx], labels[test_idx]

encoded_classes = Vector{typeof(first(Xtrain))}()
for class in unique(ytrain)
    samples = [idx for idx in 1:length(ytrain) if isequal(ytrain[idx], class)]
    class_encoding = similar(Xtrain[first(samples)])
    for s in samples
        class_encoding += Xtrain[s]
    end
    push!(encoded_classes, class_encoding)
end
encoded_classes_mat = reduce(hcat, encoded_classes)



### ordinate encodings
@info "Ordination..."
pca = fit(PCA, encoded_samples_mat; pratio=0.8);
all_encodings = hcat(encoded_samples_mat, encoded_classes_mat)
colors = vcat(color_labels, :red, :black)
encodings_transformed = projection(pca)' * (all_encodings .- mean(pca));
h = plot(
    encodings_transformed[1,:], 
    encodings_transformed[2,:], 
    seriestype = :scatter, 
    color = colors, 
    markerstrokewidth = 0,
    alpha = 0.8,
    label = "");
explained_var = principalvars(pca) ./ tvar(pca) * 100;
xlab = string("PC1 [", round(explained_var[1]; digits=2), "%]");
ylab = string("PC2 [", round(explained_var[2]; digits=2), "%]");
plot!(xlabel=xlab, ylabel=ylab, framestyle=:box)
savefig("microbiome_crohns_PCA.png")



@info "Class similarity inference..."
train_ypred = [argmax((similarity(x, first(encoded_classes)), similarity(x, last(encoded_classes)))) - 1 for x in Xtrain]
test_ypred = [argmax((similarity(x, first(encoded_classes)), similarity(x, last(encoded_classes)))) - 1 for x in Xtest]
println("On the train data:")
quickeval(train_ypred, ytrain)
println("On the test data:")
quickeval(test_ypred, ytest)



### train classifier
@info "Naive Bayes inference on the encoded counts..."
Xtrain, Xtest, ytrain, ytest = encoded_samples_mat[:, train_idx], encoded_samples_mat[:, test_idx], labels[train_idx], labels[test_idx]

clf = MultinomialNB(unique(labels), size(Xtrain, 1))
fit(clf, Xtrain, ytrain)
train_ypred = StatsBase.predict(clf, Xtrain)
#train_auc_score = auc(scores, ytrain)
test_ypred = StatsBase.predict(clf, Xtest)
#test_auc_score = auc(scores, ytest)
println("On the train data:")
quickeval(train_ypred, ytrain)
#println("\tAUC: $train_auc_score")
println("On the test data:")
quickeval(test_ypred, ytest)
#println("\tAUC: $test_auc_score")



@info "Naive Bayes inference on the raw counts..."
Xtrain, Xtest, ytrain, ytest = absolute_count_mat'[:, train_idx], absolute_count_mat'[:, test_idx], labels[train_idx], labels[test_idx]

clf = MultinomialNB(unique(labels), size(Xtrain, 1))
fit(clf, Xtrain, ytrain)
train_ypred = StatsBase.predict(clf, Xtrain)
test_ypred = StatsBase.predict(clf, Xtest)
println("On the train data:")
quickeval(train_ypred, ytrain)
println("On the test data:")
quickeval(test_ypred, ytest)