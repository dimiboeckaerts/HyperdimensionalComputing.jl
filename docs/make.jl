using HyperdimensionalComputing
using Documenter

DocMeta.setdocmeta!(HyperdimensionalComputing, :DocTestSetup, :(using HyperdimensionalComputing); recursive=true)

makedocs(;
    modules=[HyperdimensionalComputing],
    authors="Steff Taelman, Dimi Boeckaerts",
    repo="https://github.com/dimiboeckaerts/HyperdimensionalComputing.jl/blob/{commit}{path}#{line}",
    sitename="HyperdimensionalComputing.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dimiboeckaerts.github.io/HyperdimensionalComputing.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dimiboeckaerts/HyperdimensionalComputing.jl",
)
