# Scalar operations

# @inline Base.@pure vshr(a::Int64, b) = llvmcall("%res = ashr i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b % Int64)
# @inline Base.@pure vshr(a::Int32, b) = llvmcall("%res = ashr i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b % Int32)

# @inline Base.@pure vshr(a::Int16, b) = llvmcall("%res = ashr i16 %0, %1\nret i16 %res", Int16, Tuple{Int16,Int16}, a, b % Int16)
# @inline Base.@pure vshr(a::Int8, b) = llvmcall("%res = ashr i8 %0, %1\nret i8 %res", Int8, Tuple{Int8,Int8}, a, b % Int8)

# @inline Base.@pure vshr(a::UInt64, b) = llvmcall("%res = lshr i64 %0, %1\nret i64 %res", UInt64, Tuple{UInt64,UInt64}, a, b % UInt64)
# @inline Base.@pure vshr(a::UInt32, b) = llvmcall("%res = lshr i32 %0, %1\nret i32 %res", UInt32, Tuple{UInt32,UInt32}, a, b % UInt32)

# @inline Base.@pure vshr(a::UInt16, b) = llvmcall("%res = lshr i16 %0, %1\nret i16 %res", UInt16, Tuple{UInt16,UInt16}, a, b % UInt16)
# @inline Base.@pure vshr(a::UInt8, b) = llvmcall("%res = lshr i8 %0, %1\nret i8 %res", UInt8, Tuple{UInt8,UInt8}, a, b % UInt8)


function binary_op(op, W, @nospecialize(T))
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
    ff = Symbol('v', op); ff_fast = Symbol(ff, :_fast)
    _ff = Symbol('_', ff)
    _ff_fast = Symbol('_', ff_fast)
    @eval begin
        @generated $ff_fast(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), W, T)
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op, W, T)
        
        @generated $_ff_fast(v1::T, v2::T) where {T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), 1, T)
        Base.@pure @inline $ff_fast(v1::T, v2::T) where {T} = $_ff_fast(v1, v2)
        @inline $ff(x::T,y::T) where {T<:IntegerTypesHW} = $_ff_fast(x,y)
    end
end
for (op,f) ∈ [("div",:÷),("rem",:%)]
    ff = Symbol('v', op); _ff = Symbol(:_, ff)
    @eval begin
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
        @generated $_ff(v1::T, v2::T) where {T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
        Base.@pure @inline $ff(v1::T, v2::T) where {T<:IntegerTypesHW} = $_ff(v1, v2)
    end
end
@inline vcld(x, y) = vadd(vdiv(vsub(x, one(x)), y), one(x))
@inline function vdivrem(x, y)
    d = vdiv(x, y)
    r = vsub(x, vmul(d, y))
    d, r
end
for (op,sub) ∈ [
    ("ashr",:SignedHW),
    ("lshr",:UnsignedHW),
    ("lshr",:IntegerTypesHW),
    ("and",:IntegerTypesHW),
    ("or",:IntegerTypesHW),
    ("xor",:IntegerTypesHW)
]
    ff = sub === :UnsignedHW ? :vashr : Symbol('v', op)
    @eval begin
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:$sub}  = binary_op($op, W, T)
        @generated $ff(v1::T, v2::T) where {T<:$sub}  = binary_op($op, 1, T)
    end
end
# for (op,f) ∈ [("lshr",:>>),("ashr",:>>),("and",:&),("or",:|),("xor",:⊻)]
#     ff = Symbol('v', op); _ff = Symbol(:_, ff)
#     @eval Base.@pure @inline $ff(v1::T, v2::T) where {T} = $_ff(v1, v2)
#     @eval @inline $ff(v1, v2) = ((v3, v4) = promote(v1, v2); $ff(v3, v4))
# end

for (op,f) ∈ [("fadd",:vadd),("fsub",:vsub),("fmul",:vmul),("fdiv",:vfdiv),("frem",:vrem)]
    ff = Symbol(f, :_fast)
    @eval begin
        @generated  $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($(op * ' ' * fast_flags(false)), W, T)
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($(op * ' ' * fast_flags( true)), W, T)
    end
end

@inline vdiv(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = vfdiv(v1 - vrem(v1, v2), v2)
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = vfdiv_fast(v1 - vrem_fast(v1, v2), v2)
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = trunc(T, float(a) / float(b))
@inline function vdiv_fast(v1, v2)
    v3, v4 = promote_div(v1, v2)
    vdiv_fast(v3, v4)
end

@inline vfdiv(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = float(a) / float(b)

for f ∈ [:vadd,:vadd_fast,:vsub,:vsub_fast,:vmul,:vmul_fast]
    @eval begin
        @inline function $f(a, b)
            c, d = promote(a, b)
            $f(c, d)
        end
    end
end

