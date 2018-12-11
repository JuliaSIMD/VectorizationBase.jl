using Documenter, VectorizationBase

makedocs(;
    modules=[VectorizationBase],
    format=:html,
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/VectorizationBase.jl/blob/{commit}{path}#L{line}",
    sitename="VectorizationBase.jl",
    authors="Chris Elrod",
    assets=[],
)

deploydocs(;
    repo="github.com/chriselrod/VectorizationBase.jl",
    target="build",
    julia="1.0",
    deps=nothing,
    make=nothing,
)
