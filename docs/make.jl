using FastPIC
using Documenter

DocMeta.setdocmeta!(FastPIC, :DocTestSetup, :(using FastPIC); recursive = true)

makedocs(;
    modules = [FastPIC],
    authors = "FerreolS <ferreol.soulez@univ-lyon1.fr>",
    sitename = "FastPIC.jl",
    format = Documenter.HTML(;
        canonical = "https://FerreolS.github.io/FastPIC.jl",
        edit_link = "master",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo = "github.com/FerreolS/FastPIC.jl",
    devbranch = "master",
)
