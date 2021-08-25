function _precompile_()
  ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
  # for T in (Bool, Int, Float32, Float64)
  #   for A in (Vector, Matrix)
  #     precompile(stridedpointer, (A{T},))
  #     precompile(stridedpointer, (LinearAlgebra.Adjoint{T,A{T}},))
  #   end
  # end

  
  precompile(offset_ptr, (Symbol, Symbol, Char, Int, Int, Int, Int, Int, Bool, Int))
  precompile(vload_quote_llvmcall_core, (Symbol, Symbol, Symbol, Int, Int, Int, Int, Bool, Bool, Int))
  precompile(vstore_quote, (Symbol, Symbol, Symbol, Int, Int, Int, Int, Bool, Bool, Bool, Bool, Int))

  precompile(Tuple{typeof(transpose_vecunroll_quote_W_smaller),Int,Int})   # time: 0.02420761

  precompile(Tuple{typeof(horizontal_reduce_store_expr),Int,Int,NTuple{4, Int},Symbol,Symbol,Bool,Int,Bool})   # time: 0.02125804
  precompile(Tuple{typeof(transpose_vecunroll_quote_W_larger),Int,Int})   # time: 0.01755242
  precompile(Tuple{typeof(shufflevector_instrs),Int,Type,Vector{String},Int})   # time: 0.0159487
  precompile(Tuple{typeof(transpose_vecunroll_quote),Int})   # time: 0.014891806
  precompile(Tuple{typeof(align),Int,Int})   # time: 0.013784537
  precompile(Tuple{typeof(align),Int})   # time: 0.013609074

  precompile(Tuple{typeof(vstore_transpose_quote),Int64,Int64,Int64,Int64,Int64,Int64,Int64,Bool,Bool,Bool,Int64,Int64,Symbol,UInt64,Bool})   # time: 0.006213663
  precompile(Tuple{typeof(vstore_unroll_i_quote),Int64,Int64,Int64,Bool,Bool,Bool,Int64,Bool})   # time: 0.002936335
  
  precompile(Tuple{typeof(_shuffle_load_quote), Symbol, Int, NTuple{9,Int}, Symbol, Symbol, Int, Int, Bool, Int, UInt})
  precompile(Tuple{typeof(_shuffle_store_quote), Symbol, Int, NTuple{9,Int}, Symbol, Symbol, Int, Int, Bool, Bool, Bool, Int, Bool})

  precompile(Tuple{typeof(collapse_expr),Int64,Symbol,Int64})   # time: 0.003906299
  
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
