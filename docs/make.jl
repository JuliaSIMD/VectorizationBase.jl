using VectorizationBase
using Documenter

makedocs(;
    modules=[VectorizationBase],
    authors="Chris Elrod",
    repo="https://github.com/chriselrod/VectorizationBase.jl/blob/{commit}{path}#L{line}",
    sitename="VectorizationBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://chriselrod.github.io/VectorizationBase.jl",
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=false,
)

deploydocs(;
    repo="github.com/chriselrod/VectorizationBase.jl",
)
