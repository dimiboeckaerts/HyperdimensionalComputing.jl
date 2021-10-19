using HyperdimensionalComputing

# get the different types
println("The time to encode a 4-letter alphabet as bipolar or binary hypervectors, respectively:")
@time bip = encode_alphabet(['A', 'C', 'T', 'G'])
@time bin = encode_alphabet(['A', 'C', 'T', 'G'], vectortype="binary")
println()

# benchmark multiplication
println("The time to combine two bipolar or binary hypervectors, respectively, into a new one that is highly dissimilar:")
@time multiply(bip['A'], bip['C'])
@time multiply(bin['A'], bin['C'])
println()

# benchmark aggregation
println("The time to combine two bipolar or binary hypervectors, respectively, into a new one that is highly similar:")
@time add(bip['A'], bip['C'])
@time add(bin['A'], bin['C'])
println()

# benchmark rotation
println("The time to rotate/permute bipolar or binary hypervectors, respectively:")
@time rotate(bip['A'], 1)
@time rotate(bin['A'], 1)
println()

# benchmark sequence encoding
seq = "ATCGATAGCA"
println("The time to encode a 10-letter sequence as bipolar or binary hypervectors, respectively:")
@time encode_sequence(seq, bip, k=3)
@time encode_sequence(seq, bin, k=3)
println()