# Hyperdimensional Computing in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://dimiboeckaerts.github.io/HyperdimensionalComputing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dimiboeckaerts.github.io/HyperdimensionalComputing.jl/dev)
[![Build Status](https://github.com/dimiboeckaerts/HyperdimensionalComputing.jl/workflows/CI/badge.svg)](https://github.com/dimiboeckaerts/HyperdimensionalComputing.jl/actions)
[![Coverage](https://codecov.io/gh/dimiboeckaerts/HyperdimensionalComputing.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/dimiboeckaerts/HyperdimensionalComputing.jl)


This package implements special types of vectors and associated methods for hyperdimensional computing. Hyperdimensional computing (HDC) is a paragdigm to represent patterns by means of a high-dimensional vectors (typically 10,000 dimensions). Specific operations can be used to create new vectors by combining the information or encoding some kind of position. HDC is an alternative machine learning method that is extremely computationally efficient. It is inspired by the distributed, holographic representation of patterns in the brain. Typically, the high-dimensionality is more important than the nature of the operations. This package provides various types of vectors (binary, graded, bipolar...) with sensible operations for *aggragating*, *binding* and *permutation*. Basic functionality for fitting a k-NN like classifier is also supported.

## Basic use

Several types of vectors are implemented. Random vectors can be initialized of different sizes.

```julia
using HyperdimensionalComputing

x = BipolarHDV()  # default length is 10,000

y = BinaryHDV(20)  # different length

z = RealHDV(Float32)  # specify data type
```

The basic operations are `aggregate` (creating a vector that is similar to the provided vectors), `bind` (creating a vector that is dissimilar to the vectors) and `circshift` (shifting the vector inplace to create a new vector). For `aggregate` and `bind`, we overload `+` and `*` as binary operators, while `Π` is an alias for `circshift`. The latter is lazily implemented. All functions have an inplace version, using the `!` prefix.

```julia
x, y, z = GradedHDV(10), GradedHDV(10), GradedHDV(10)

# aggregation

aggregate([x, y, z])

x + y

# binding

bind([x, y, z])
x * y

# permutation

circshift(x, 2)  # shifts the coordinates
Π(x, 2)  # same

Π!(y, 1)  # inplace
```

See the table for which operations are used for which type.

## Embedding sequences

HDC is particularly powerful for embedding sequences. This is done by creating embeddings for n-grams and aggregating the n-grams found in the sequence.

```julia
# create dictionary for embedding
alphabet = "ATCG"
basis = Dict(c=>RealHDV(100) for (c, hdv) in zip(alphabet, hdvs))

sequence = "TAGTTTGAGGATCCGCTCGCTGCAACGCG"

seq_embedding = sequence_embedding(sequence, basis, 3)  # embedding using 3-grams
```
If the size of the number of n-grams is not too large, it makes sense to precompute these to speed up the encoding process.

```julia
threegrams = compute_3_grams(basis)

sequence_embedding(sequence, threegrams) 
```

## Training

A model is basically trained by making an aggregation of all elements within a class. Training is simple. Prediction is done by nearest-neighbor search based on `similarity`.

```julia
hdvs = [BipolarHDV() for i in 1:1000]  # 1000 vectors
y = rand(Bool, 1000)  # two labels

centers = train(y, hdvs)

predict(BipolarHDV(), centers)  # predict for a random vector
predict(hdvs, centers)  # repredict the labels
```

In practice, this leads to suboptimal performance. One can retrain the model by reaggregating the wrongly classified labels.

```julia
retrain!(centers, y, hdvs, niters=10)
```

## Overview of operations

| Vector | element domain | aggregate | binding | similarity |
| ------ | --------------| ---------| ----------| --------|
| `BinaryHDV` | 0, 1 | majority | xor | Jaccard |
| `BipolarHDV` | -1, 0, 1 | sum and threshold | multiply | cosine |
| `GradedHDV` | [0, 1] |  3π  | fuzzy xor | Jaccard |
| `GradedBipolarHDV` | [-1, 1] | 3π  | fuzzy xor  | cosine |
| `RealHDV` | real | sum weighted to keep vector norm | multiply | cosine |
