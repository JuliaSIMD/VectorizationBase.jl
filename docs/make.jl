using VectorizationBase
using Documenter

makedocs(;
  modules = [VectorizationBase],
  authors = "Chris Elrod",
  repo = "https://github.com/JuliaSIMD/VectorizationBase.jl/blob/{commit}{path}#L{line}",
  sitename = "VectorizationBase.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaSIMD.github.io/VectorizationBase.jl"
  ),
  pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/JuliaSIMD/VectorizationBase.jl")
