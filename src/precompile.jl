function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(VectorizationBase.T_shift),Type{Float64}})
    precompile(Tuple{typeof(VectorizationBase.llvmtype),TypeVar})
    precompile(Tuple{typeof(VectorizationBase.mask_type),Type{Float64}})
end
