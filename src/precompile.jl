function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    for T in (Bool, Int, Float32, Float64)
        for A in (Vector, Matrix)
            precompile(stridedpointer, (A{T},))
            precompile(stridedpointer, (LinearAlgebra.Adjoint{T,A{T}},))
        end
    end

                            
    precompile(offset_ptr, (Symbol, Symbol, Char, Int, Int, Int, Int, Int, Bool, Int))
    precompile(vload_quote, (Symbol, Symbol, Symbol, Int, Int, Int, Int, Bool, Bool, Int, Expr))
    precompile(vstore_quote, (Symbol, Symbol, Symbol, Int, Int, Int, Int, Bool, Bool, Bool, Bool, Int, Expr))

    precompile(reset_features!, ())
    precompile(safe_topology_load!, ())
    precompile(redefine_attr_count, ())
    precompile(redefine_cache, (Int,))

    # precompile(_pick_vector_width, (Type, Vararg{Type,100}))
    # the `"NATIVE_PRECOMPILE_VECTORIZATIONBASE" ∈ keys(ENV)` isn't respected, seems
    # like it gets precompiled anyway given that the first condition is `true`.
    # if VERSION ≥ v"1.7.0-DEV.346" && "NATIVE_PRECOMPILE_VECTORIZATIONBASE" ∈ keys(ENV)
    #     set_features!()
    #     for T ∈ (Float32, Float64)
    #         W = pick_vector_width(T)
    #         precompile(>=, (Int, MM{W, 1, Int}))
    #         for op ∈ (-, Base.FastMath.sub_fast)
    #             precompile(op, (Vec{W, T}, ))
    #         end
    #         for op ∈ (+, -, *, Base.FastMath.add_fast, Base.FastMath.sub_fast, Base.FastMath.mul_fast)
    #             precompile(op, (Vec{W, T}, Vec{W, T}))
    #         end
    #         for op ∈ (VectorizationBase.vfmadd, VectorizationBase.vfmadd_fast)
    #             precompile(op, (Vec{W, T}, Vec{W, T}, Vec{W, T}))
    #         end
    #     end
    # end
end
