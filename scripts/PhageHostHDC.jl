# --------------------------------------------------
# HYPERDIMENSIONAL COMPUTING IN JULIA: PHAGE-HOST PREDICTIONS
#
# @author: dimiboeckaerts
# --------------------------------------------------

# LIBRARIES & DIRECTORIES
# --------------------------------------------------
push!(LOAD_PATH, "/Users/dimi/Documents/GitHub/HyperdimensionalComputing.jl/src/")
using HyperdimensionalComputing
using DataFrames
using ProgressMeter
using CSV
using FASTX
using BioAlignments

data_dir = "/Users/dimi/GoogleDrive/PhD/4_PHAGEHOST_LEARNING/42_DATA/klebsiella_RBP_data"


# FUNCTIONS
# --------------------------------------------------
function file_to_array(file)
    """
    Function that reads a FASTA file and puts its sequences in an array.
    """
    sequences = []
    reader = FASTA.Reader(open(file, "r"))
    for record in reader
        seq = FASTA.sequence(record)
        push!(sequences, seq)
    end
    return sequences
end


# PhageHostLearning: LAYER 1
# --------------------------------------------------
# get loci names for proteins & labels
IM = DataFrame(CSV.File(data_dir*"/interactions_klebsiella.csv"))
loci_names = IM.accession
serotypes = DataFrame(CSV.File(data_dir*"/klebsiella_genomes_031221_serotypes.csv"))
labels = serotypes.sero

# define protein alphabet
alphabet = "GAVLIFPSTYCMKRHWDENQX"
basis = Dict(c=>RealHDV() for c in alphabet)

# load loci proteins in dict
loci_sequences = Dict(accession=>
                file_to_array(data_dir*"/kaptive_results_proteins_"*accession*".fasta")
                for accession in loci_names)

# compute loci embeddings w/ proteins
loci_embeddings = Array{RealHDV}(undef, length(loci_sequences))
for (i, (name, proteins)) in enumerate(loci_sequences)
    protein_hdvs = [sequence_embedding(string(sequence), basis, 3) for sequence in proteins]
    loci_hdv = bind(protein_hdvs)
    loci_embeddings[i] = loci_hdv
end
length(loci_embeddings)

# train layer 1
layer1 = train(labels, loci_embeddings)
retrain!(layer1, labels, loci_embeddings, niters=10)
