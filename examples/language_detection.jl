#=
language_detection.jl; this script shows the application of hypervectors to a language_detection problem.
=#
using Pkg
Pkg.activate(".")
using CSV, DataFrames, HyperdimensionalComputing, MLJ


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


# load in and preprocess training paragraphs from project Gutenberg
labelled_languages_df = DataFrame(CSV.File("data/labelled_languages_train.csv"))
insertcols!(labelled_languages_df, 2, :"PREPROC"=>Vector{String}(undef, size(labelled_languages_df, 1)))
labelled_languages_df[!, "PREPROC"] = [language_preprocessing(labelled_languages_df[i, "TEXT"]) for i in 1:size(labelled_languages_df, 1)]


# load in and preprocess testing sentences from tatoeba.org
examples_df = DataFrame(CSV.File("data/labelled_languages_test.csv"))
insertcols!(examples_df, 2, :"PREPROC"=>Vector{String}(undef, size(examples_df, 1)))
examples_df[!, "PREPROC"] = [language_preprocessing(examples_df[i, "TEXT"]) for i in 1:size(examples_df, 1)]


for vec in [BipolarHDV, BinaryHDV, GradedHDV, RealHDV]
    println(vec)

    # encode both train and test data
    alphabet = Dict(c=>vec() for c in vcat('a':'z', ' '))
    encoded_references = [sequence_embedding(ref, alphabet, 3) for ref in labelled_languages_df[!, "PREPROC"]]
    encoded_examples = [sequence_embedding(ex, alphabet, 3) for ex in examples_df[!, "PREPROC"]]

    # predict
    centers = train(labelled_languages_df[!, "LANGUAGE"], encoded_references)
    test_predictions = [HyperdimensionalComputing.predict(hv, centers) for hv in encoded_examples]

    # evaluate performance
    println(string("weighted F1-score: ", multiclass_f1score(test_predictions, examples_df[!, "LANGUAGE"])))
end