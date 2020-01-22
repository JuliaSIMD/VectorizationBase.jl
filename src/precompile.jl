function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(VectorizationBase.mask),Val{8},Int64})
    precompile(Tuple{typeof(VectorizationBase.pick_vector_width),Type{Float32}})
    precompile(Tuple{typeof(VectorizationBase.unstable_mask),Int64,Int64})
end
