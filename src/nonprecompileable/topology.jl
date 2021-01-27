const TOPOLOGY = try
    Hwloc.topology_load();
catch e
    @warn e
    @warn """
        Using Hwloc failed. Please file an issue with the above warning at: https://github.com/JuliaParallel/Hwloc.jl
        Proceeding with generic topology assumptions. This may result in reduced performance.
    """
    nothing
end

if TOPOLOGY isa Nothing
    include(joinpath(@__DIR__, "topology_generic.jl"))
else
    include(joinpath(@__DIR__, "topology_specific.jl"))
end
