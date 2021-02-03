
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

# Integer
for (op,f) ∈ [("add",:+),("sub",:-),("mul",:*),("shl",:<<)]
    ff = Symbol('v', op); ff_fast = Symbol(ff, :_fast)
    _ff = Symbol('_', ff)
    _ff_fast = Symbol('_', ff_fast)
    @eval begin
        @generated $ff_fast(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), W, T)
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op, W, T)
        
        @generated $_ff_fast(v1::T, v2::T) where {T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), 1, T)
        @inline $ff_fast(v1::T, v2::T) where {T} = $_ff_fast(v1, v2)
        @inline $ff(x::T,y::T) where {T<:IntegerTypesHW} = $_ff_fast(x,y)
    end
end
for (op,f) ∈ [("div",:÷),("rem",:%)]
    ff = Symbol('v', op); _ff = Symbol(:_, ff)
    @eval begin
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
        @generated $_ff(v1::T, v2::T) where {T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
        @inline $ff(v1::T, v2::T) where {T<:IntegerTypesHW} = $_ff(v1, v2)
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

for (op,f) ∈ [("fadd",:vadd),("fsub",:vsub),("fmul",:vmul),("fdiv",:vfdiv),("frem",:vrem)]
    ff = Symbol(f, :_fast)
    @eval begin
        @generated  $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($(op * ' ' * fast_flags(false)), W, T)
        @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($(op * ' ' * fast_flags( true)), W, T)
    end
end

@inline vdiv(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = vfdiv(vsub(v1, vrem(v1, v2)), v2)
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = vfdiv_fast(vsub_fast(v1, vrem_fast(v1, v2)), v2)
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = trunc(T, vfloat_fast(a) / vfloat_fast(b))
@inline function vdiv_fast(v1, v2)
    v3, v4 = promote_div(v1, v2)
    vdiv_fast(v3, v4)
end

@inline vfdiv(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv(vfloat(a), vfloat(b))
@inline vfdiv_fast(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv_fast(vfloat_fast(a), vfloat_fast(b))

for f ∈ [:vadd,:vadd_fast,:vsub,:vsub_fast,:vmul,:vmul_fast]
    @eval begin
        @inline function $f(a, b)
            c, d = promote(a, b)
            $f(c, d)
        end
    end
end

