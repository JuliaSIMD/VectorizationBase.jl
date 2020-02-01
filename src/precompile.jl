function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(VectorizationBase.T_shift),Type{Float32}})
    precompile(Tuple{typeof(VectorizationBase.llvmtype),Type{T} where T})
    precompile(Tuple{typeof(VectorizationBase.mask),Val{16},Int64})
    precompile(Tuple{typeof(VectorizationBase.mask),Val{8},Int64})
    precompile(Tuple{typeof(VectorizationBase.pick_vector_width),Type{Float32}})
end
