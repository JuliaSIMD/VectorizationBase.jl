

@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    include("build_x86.jl")
else
    include("build_generic.jl")
end
    
open(joinpath(@__DIR__, "..", "src", "cpu_info.jl"), "w") do f
    write(f, cpu_info_string)
end


