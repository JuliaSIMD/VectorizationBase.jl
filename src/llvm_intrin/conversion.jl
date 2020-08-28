function convert_func(op, W1, typ1, T1, typ2, T2, W2 = W)
    vtyp1 = vtype(W1, typ1)
    vtyp2 = vtype(W2, typ2)
    instrs = """
    %res = $op $vtyp2 %0 to $vtyp1
    ret $vtyp1 %res
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W1,$T1}, Tuple{_Vec{$W2,$T2}}, data(v)))
    end
end
function identity_func(W, typ1, T1, T2)
    vtyp1 = vtype(W, typ1)
    instrs = """
    ret $vtyp1 %0
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T1}, Tuple{_Vec{$W,$T2}}, data(v)))
    end
end

for bytes1 in (1,2,4,8)
    llvmint = "i$(8bytes1)"
    jint = Symbol(:Int, 8bytes1)
    juint = Symbol(:UInt, 8bytes1)
    for (jfloat,llvmfloat) ∈ ((:Float32,"float"),(:Float64,"double"))
        expr = :(convert_func("sitofp", W, $llvmfloat, $jfloat, $llvmint, $jint))
        @eval @generated Vec{W,$jfloat}(v::Vec{W,$jint}) where {W} = $expr
        expr = :(convert_func("uitofp", W, $llvmfloat, $jfloat, $llvmint, $juint))
        @eval @generated Vec{W,$jfloat}(v::Vec{W,$juint}) where {W} = $expr
        expr = :(convert_func("fptosi", W, $llvmint, $jint, $llvmfloat, $jfloat))
        @eval @generated Vec{W,$jint}(v::Vec{W,$jfloat}) where {W} = $expr
        expr = :(convert_func("fptoui", W, $llvmint, $juint, $llvmfloat, $jfloat))
        @eval @generated Vec{W,$juint}(v::Vec{W,$jfloat}) where {W} = $expr
    end
    for bytes2 in (1,2,4)
        bytes2 == bytes1 && break
        llvmint2 = "i$(8bytes2)"
        jint2 = Symbol(:Int, 8bytes2)
        juint2 = Symbol(:UInt, 8bytes2)
        expr = :(convert_func("trunc", W, $llvmint2, $jint2, $llvmint, $jint))
        @eval @generated Vec{W,$jint2}(v::Vec{W,$jint}) where {W} = $expr
        expr = :(convert_func("trunc", W, $llvmint2, $juint2, $llvmint, $jint))
        @eval @generated Vec{W,$juint2}(v::Vec{W,$jint}) where {W} = $expr
        expr = :(convert_func("trunc", W, $llvmint2, $jint2, $llvmint, $juint))
        @eval @generated Vec{W,$jint2}(v::Vec{W,$juint}) where {W} = $expr
        expr = :(convert_func("trunc", W, $llvmint2, $juint2, $llvmint, $juint))
        @eval @generated Vec{W,$juint2}(v::Vec{W,$juint}) where {W} = $expr

        expr = :(convert_func("sext", W, $llvmint, $jint, $llvmint2, $jint2))
        @eval @generated Vec{W,$jint}(v::Vec{W,$jint2}) where {W} = $expr
        expr = :(convert_func("zext", W, $llvmint, $juint, $llvmint2, $jint2))
        @eval @generated Vec{W,$juint}(v::Vec{W,$jint2}) where {W} = $expr
        expr = :(convert_func("zext", W, $llvmint, $jint, $llvmint2, $juint2))
        @eval @generated Vec{W,$jint}(v::Vec{W,$juint2}) where {W} = $expr
        expr = :(convert_func("zext", W, $llvmint, $juint, $llvmint2, $juint2))
        @eval @generated Vec{W,$juint}(v::Vec{W,$juint2}) where {W} = $expr
    end
    expr = :(identity_func(W, $llvmint, $juint, $jint))
    @eval @generated Vec{W,$juint}(v::Vec{W,$jint}) where {W} = $expr
    expr = :(identity_func(W, $llvmint, $jint, $juint))
    @eval @generated Vec{W,$jint}(v::Vec{W,$juint}) where {W} = $expr
end

for (jfloat,llvmfloat) ∈ ((:Float32,"float"),(:Float64,"double"))
    for (jfloat,llvmfloat) ∈ ((:Float32,"float"),(:Float64,"double"))
    end
end
let
    W = 1
    while W ≤ pick_vector_width(Float64)
        @eval Vec{$W,Float32}(v::Vec{$W,Float64}) = $(convert_func("fptrunc", W, "float", :Float32, "double", :Float64))
        @eval Vec{$W,Float64}(v::Vec{$W,Float32}) = $(convert_func("fpext", W, "double", :Float64, "float", :Float32))
        W += W
    end
end

Base.convert(::Vec{W,T}, s::T) where {W, T <: NativeTypes} = vbroadcast(Val{W}(), s)
Base.convert(::Vec{W,T}, s::T) where {W, T <: Integer} = vbroadcast(Val{W}(), s)
Base.convert(::Vec{W,T}, s::NativeTypes) where {W, T} = vbroadcast(Val{W}(), T(s))
Base.convert(::Vec{W,T1}, s::T2) where {W, T1 <: Integer, T2 <: Integer} = vbroadcast(Val{W}(), s % T1)


@generated function Base.reinterpret(::Type{T1}, v::Vec{W2,T2}) where {W2, T1 <: NativeTypes, T2}
    W1 = W2 * sizeof(T2) ÷ sizeof(T1)
    Expr(:block, Expr(:meta, :inline), :(reinterpret(Vec{$W1,$T1}, v)))
end
@generated function Base.reinterpret(::Type{Vec{W1,T1}}, v::Vec{W2,T2}) where {W1, W2, T1, T2}
    @assert sizeof(T1) * W1 == W2 * sizeof(T2)
    convert_func("bitcast", W1, LLVM_TYPES[T1], T1, LLVM_TYPES[T2], T2, W2)
end

