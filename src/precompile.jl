function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    for T in (Bool, Int, Float32, Float64)
        for A in (Vector, Matrix)
            precompile(stridedpointer, (A{T},))
        end
    end
    function precompile_nt(@nospecialize(T))
        for I ∈ (Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64)
            precompile(vload_quote, (Type{T}, Type{I}, Symbol, Int, Int, Int, Int, Bool, Bool))
            precompile(vload_quote, (Type{T}, Type{I}, Symbol, Int, Int, Int, Int, Bool, Bool, Bool, Bool))
        end
    end
    U = NativeTypes
    while isa(U, Union)
        T, U = U.a, U.b
        precompile_nt(T)
    end
    precompile_nt(U)
    precompile(_pick_vector_width, (Type, Vararg{Type,100}))
    for T ∈ (Float32, Float64)
        W = pick_vector_width(T)
        precompile(>=, (Int, MM{W, 1, Int}))
        for op ∈ (-, Base.FastMath.sub_fast)
            precompile(op, (Vec{W, T}, ))
        end
        for op ∈ (+, -, *, Base.FastMath.add_fast, Base.FastMath.sub_fast, Base.FastMath.mul_fast)
            precompile(op, (Vec{W, T}, Vec{W, T}))
        end
        for op ∈ (VectorizationBase.vfmadd, VectorizationBase.vfmadd_fast)
            precompile(op, (Vec{W, T}, Vec{W, T}, Vec{W, T}))
        end
    end
end
