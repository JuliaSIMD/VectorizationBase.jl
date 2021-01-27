function __init__()
    nonprecompileable_directory = joinpath(@__DIR__, "nonprecompileable")
    include(joinpath(nonprecompileable_directory, "topology.jl"))
    include(joinpath(nonprecompileable_directory, "cpu_info.jl"))
    include(joinpath(nonprecompileable_directory, "cache_inclusivity.jl"))
    return nothing
end
