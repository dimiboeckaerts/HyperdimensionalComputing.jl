#=
language_detection.jl; this script shows the application of hypervectors to a language_detection problem.
=#

using CSV, DataFrames, HyperdimensionalComputing


# set 26 letter alphabet and set up preprocessing to fit texts to this format
alphabet_hv = encode_alphabet(vcat('a':'z', ' '))


"""
This function makes all letters in a text lowercase, removes accents and all-round 'normalizes' language-specific variations on letters to a 26-letter alphabet.
"""
function language_preprocessing(text::String)
        #make text lowercase
        text = lowercase(text)

        #remove odd accents and peculiarities
        text = replace(text, "é"=>"e")
        text = replace(text, "ê"=>"e")
        text = replace(text, "è"=>"e")
        text = replace(text, "ë"=>"e")
        text = replace(text, "ę"=>"e")
        text = replace(text, "á"=>"a")
        text = replace(text, "à"=>"a")
        text = replace(text, "â"=>"a")
        text = replace(text, "ä"=>"a")
        text = replace(text, "ą"=>"a")
        text = replace(text, "ç"=>"c")
        text = replace(text, "ć"=>"c")
        text = replace(text, "î"=>"i")
        text = replace(text, "ï"=>"i")
        text = replace(text, "í"=>"i")
        text = replace(text, "ł"=>"l")
        text = replace(text, "ñ"=>"n")
        text = replace(text, "ń"=>"n")
        text = replace(text, "ô"=>"o")
        text = replace(text, "ö"=>"o")
        text = replace(text, "ó"=>"o")
        text = replace(text, "ś"=>"s")
        text = replace(text, "ß"=>"ss")
        text = replace(text, "û"=>"u")
        text = replace(text, "ü"=>"u")
        text = replace(text, "ú"=>"u")
        text = replace(text, "ź"=>"z")
        text = replace(text, "ż"=>"z")
    
        #remove punctuation and double spaces
        text = string([t for t in text if t ∈ vcat('a':'z', ' ')]...)
        text = replace(text, "  "=>" ")

        return text
end


labelled_languages_df = DataFrame(CSV.File("data/labelled_languages_many.csv"))
insertcols!(labelled_languages_df, 2, :"PREPROC"=>Vector{String}(undef, size(labelled_languages_df, 1)))
for i in 1:size(labelled_languages_df, 1)
        labelled_languages_df[i, "PREPROC"] = language_preprocessing(labelled_languages_df[i, "TEXT"])
end

first(labelled_languages_df, 5)

# get some example languages to detect
examples = ["Hyperdimensionaal rekenen: volgens enkele papers zou dit representaties kunnen genereren die even krachtig zijn als word2vec!", 
            "Ceci n'est pas une pipe. Liberté, égalité, fraternité.", 
            "Para bailar la bamba, para bailar la bamba se necesita una poca de gracia. Una poca de gracia pa' mi pa' ti y arriba y arriba. Ah y arriba y arriba por ti seré, por ti seré, por ti seré. Yo no soy marinero. Yo no soy marinero, soy capitán, Soy capitán, soy capitán.",
            "We're no strangers to love, you know the rules and so do I. A full commitment's what I'm thinking of, you wouldn't get this from any other guy.", 
            "My tailor is rich, but my english is poor.", 
            "Der Begriff ”wahrscheinlich” wird im Alltag in verschiedenen Situationen verwendet, hat dabei auch unterschiedliche Bedeutung.", 
            "Véletlen módszerrel bebizonyította minden n és s értékére n-kromatikus s kerületű (legrövidebb kör hossza) gráf létezését."]
example_labels = ["dutch", "french", "spanish", "english", "english", "german", "hungarian"]
examples_preproc = [language_preprocessing(ex) for ex in examples]


# encode both train and test data
encoded_references = Dict((labelled_languages_df[idx, 1]=>encode_sequence(labelled_languages_df[idx, 2], alphabet_hv, k=3) for idx in 1:size(labelled_languages_df, 1)))
encoded_examples = [encode_sequence(ex, alphabet_hv, k=3) for ex in examples_preproc]

# predict
println("Bipolar vector predictions:")
println(cosine_predict(encoded_examples, encoded_references))


# now let's try with binary hypervectors
bit_alphabet_hv = encode_alphabet(vcat('a':'z', ' '), vectortype="binary")

bit_encoded_references = Dict((labelled_languages_df[idx, 1]=>encode_sequence(labelled_languages_df[idx, 2], bit_alphabet_hv, k=3) for idx in 1:size(labelled_languages_df, 1)))
bit_encoded_examples = [encode_sequence(ex, bit_alphabet_hv, k=5) for ex in examples_preproc]

println("Binary vector predictions:")
println(cosine_predict(bit_encoded_examples, bit_encoded_references))

println("Ground truth:")
println(example_labels)