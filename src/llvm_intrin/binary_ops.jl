# Scalar operations

# @inline Base.@pure vshr(a::Int64, b) = llvmcall("%res = ashr i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b % Int64)
# @inline Base.@pure vshr(a::Int32, b) = llvmcall("%res = ashr i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b % Int32)

# @inline Base.@pure vshr(a::Int16, b) = llvmcall("%res = ashr i16 %0, %1\nret i16 %res", Int16, Tuple{Int16,Int16}, a, b % Int16)
# @inline Base.@pure vshr(a::Int8, b) = llvmcall("%res = ashr i8 %0, %1\nret i8 %res", Int8, Tuple{Int8,Int8}, a, b % Int8)

# @inline Base.@pure vshr(a::UInt64, b) = llvmcall("%res = lshr i64 %0, %1\nret i64 %res", UInt64, Tuple{UInt64,UInt64}, a, b % UInt64)
# @inline Base.@pure vshr(a::UInt32, b) = llvmcall("%res = lshr i32 %0, %1\nret i32 %res", UInt32, Tuple{UInt32,UInt32}, a, b % UInt32)

# @inline Base.@pure vshr(a::UInt16, b) = llvmcall("%res = lshr i16 %0, %1\nret i16 %res", UInt16, Tuple{UInt16,UInt16}, a, b % UInt16)
# @inline Base.@pure vshr(a::UInt8, b) = llvmcall("%res = lshr i8 %0, %1\nret i8 %res", UInt8, Tuple{UInt8,UInt8}, a, b % UInt8)



function binary_op(op, W, @nospecialize(_::Type{T})) where {T}
    ty = LLVM_TYPES[T]
    if isone(W)
        V = T
    else
        ty = "<$W x $ty>"
        V = NTuple{W,VecElement{T}}
    end
    instrs = "%res = $op $ty %0, %1\nret $ty %res"
    call = :(llvmcall($instrs, $V, Tuple{$V,$V}, data(v1), data(v2)))
    W > 1 && (call = Expr(:call, :Vec, call))
    Expr(:block, Expr(:meta, :inline), call)
end
# @generated function binary_operation(::Val{op}, v1::V1, v2::V2) where {op, V1, V2}
#     M1, N1, W1, T1 = description(V1)
#     M2, N2, W2, T2 = description(V2)
    
#     lc = Expr(:call, :llvmcall, join(instrs, "\n"), )
# end
# function binary_op(op, W, @nospecialize(_::Type{T})) where {T}
#     ty = 'i' * string(8*sizeof(T))
#     binary_op(op, W, T, ty)
# end

# Integer
    # vop = Symbol('v', op)
for (op,f) ∈ [("add",:+),("sub",:-),("mul",:*),("shl",:<<)]
    ff = Symbol('v', op); _ff = Symbol(:_, ff)
    @eval @generated $_ff(v1::T, v2::T) where {T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), 1, T)
    @eval @generated $_ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), W, T)
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op, W, T)
    @eval Base.@pure @inline $ff(v1::T, v2::T) where {T} = $_ff(v1, v2)
    @eval @inline $ff(v1, v2) = ((v3, v4) = promote(v1, v2); $ff(v3, v4))
end
for (op,f) ∈ [("div",:÷),("rem",:%)]
    ff = Symbol('v', op); _ff = Symbol(:_, ff)
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
    @eval @generated $_ff(v1::T, v2::T) where {T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
    @eval Base.@pure @inline $ff(v1::T, v2::T) where {T} = $_ff(v1, v2)
    @eval @inline $ff(v1, v2) = ((v3, v4) = promote(v1, v2); $ff(v3, v4))
end
@inline vcld(x, y) = vadd(vdiv(vsub(x,one(x)), y), one(x))
@inline function vdivrem(x, y)
    d = vdiv(x, y)
    r = vsub(x, vmul(d, y))
    d, r
end
for (op,f,s) ∈ [("ashr",:>>,0x01),("lshr",:>>,0x02),("lshr",:>>>,0x03),("and",:&,0x03),("or",:|,0x03),("xor",:⊻,0x03)]
    _ff = Symbol(:_, 'v', op)
    fdef = Expr(:where, :(Base.$f(v1::Vec{W,T}, v2::Vec{W,T})), :W)
    ffdef = Expr(:where, :($_ff(v1::T, v2::T)))
    if iszero(s & 0x01)
        push!(fdef.args, :(T <: Unsigned))
        push!(ffdef.args, :(T <: Unsigned))
    elseif iszero(s & 0x02)
        push!(fdef.args, :(T <: Signed))
        push!(ffdef.args, :(T <: Signed))
    else
        push!(fdef.args, :T)
        push!(ffdef.args, :T)
    end
    @eval @generated $fdef = binary_op($op, W, T)
    @eval @generated $ffdef = binary_op($op, 1, T)
end
for (op,f) ∈ [("lshr",:>>),("ashr",:>>),("and",:&),("or",:|),("xor",:⊻)]
    ff = Symbol('v', op); _ff = Symbol(:_, ff)
    @eval Base.@pure @inline $ff(v1::T, v2::T) where {T} = $_ff(v1, v2)
    @eval @inline $ff(v1, v2) = ((v3, v4) = promote(v1, v2); $ff(v3, v4))
end

for (op,f,ff) ∈ [("fadd",:+,:vadd),("fsub",:-,:vsub),("fmul",:*,:vmul),("fdiv",:/,:vfdiv),("frem",:%,:vrem)]
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T} = binary_op($(op * " nsz arcp contract afn reassoc"), W, T)
end
@inline Base.inv(v::Vec) = vdiv(one(v), v)

@inline Base.:(/)(a::Vec{W,<:Integer}, b::Vec{W,<:Integer}) where {W} = float(a) / float(b)


