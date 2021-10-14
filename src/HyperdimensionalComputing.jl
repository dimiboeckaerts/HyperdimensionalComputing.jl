module HyperdimensionalComputing

function encode_alphabet(alphabet::Vector{String}; dim=10000::Int)
    """
    This function encodes an alphabet of characters into hyperdimensional 
    vectors (HVs) as a starting point to encode text or sequences into HVs.
    The vectors are bipolar vectors.

    Input: an alphabet as vector of strings; e.g. ["A", "B", "C"]
    Output: a dictionary with HVs for each of the characters
    """
    encodings = Dict()
    for character in alphabet
        encodings[character] = rand([-1 1], dim)
    end
    return encodings
end

end
