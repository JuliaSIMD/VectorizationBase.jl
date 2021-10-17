
function binary_op(op, W, @nospecialize(T))
    ty = LLVM_TYPES[T]
    if isone(W)
        V = T
    else
        ty = "<$W x $ty>"
        V = NTuple{W,VecElement{T}}
    end
    instrs = "%res = $op $ty %0, %1\nret $ty %res"
    call = :($LLVMCALL($instrs, $V, Tuple{$V,$V}, data(v1), data(v2)))
    W > 1 && (call = Expr(:call, :Vec, call))
    Expr(:block, Expr(:meta, :inline), call)
end

# Integer
for (op,f) ∈ [("add",:+),("sub",:-),("mul",:*),("shl",:<<)]
  ff = Symbol('v', op)
  fnsw = Symbol(ff,"_nsw")
  fnuw = Symbol(ff,"_nuw")
  fnw = Symbol(ff,"_nw")
  ff_fast = Symbol(ff, :_fast)
  @eval begin
    # @inline $ff(a,b) = $ff_fast(a,b)
    @inline $ff(a::T,b::T) where {T<:Union{FloatingTypes,IntegerTypesHW,AbstractSIMD}} = $ff_fast(a,b)
    # @generated $ff_fast(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), W, T)
    @generated $ff_fast(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W, T)
    @generated $fnsw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nsw"), W, T)
    @generated $fnuw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nuw"), W, T)
    @generated $fnw(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($(op * " nsw nuw"), W, T)
    # @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W, T)
    @inline Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = $ff_fast(v1, v2)
    # @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:IntegerTypesHW} = binary_op($op, W, T)
    @inline $ff_fast(x, y) = $f(x,y)
    
    # @generated $ff_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), 1, T)
    # @generated $ff_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($op, 1, T)
    @generated $fnsw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nsw"), 1, T)
    @generated $fnuw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nuw"), 1, T)
    @generated $fnw(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op($(op * " nsw nuw"), 1, T)
  end
end
@inline vadd_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = Base.add_int(v1,v2)
@inline vsub_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = Base.sub_int(v1,v2)
@inline vmul_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = Base.mul_int(v1,v2)
@inline vshl_fast(v1::T, v2::T) where {T<:IntegerTypesHW} = Base.shl_int(v1,v2)
for (op,f) ∈ [("div",:÷),("rem",:%)]
  ff = Symbol('v', op); #_ff = Symbol(:_, ff)
  sbf = Symbol('s', op, :_int)
  ubf = Symbol('u', op, :_int)
    @eval begin
      @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
      @inline $ff(a::I, b::I) where {I<:SignedHW} = Base.$sbf(a, b)
      @inline $ff(a::I, b::I) where {I<:UnsignedHW} = Base.$ubf(a, b)
      @inline $ff(a::Int128, b::Int128) = Base.$sbf(a, b)
      @inline $ff(a::UInt128, b::UInt128) = Base.$ubf(a, b)
        # @generated $_ff(v1::T, v2::T) where {T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
        # @inline $ff(v1::T, v2::T) where {T<:IntegerTypesHW} = $_ff(v1, v2)
    end
end
# for (op,f) ∈ [("div",:÷),("rem",:%)]
#   @eval begin
#     @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, T)
#     @generated Base.$f(v1::T, v2::T) where {T<:IntegerTypesHW} = binary_op((T <: Signed ? 's' : 'u') * $op, 1, T)
#   end
# end
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

for (op,f) ∈ [("fadd",:vadd),("fsub",:vsub),("fmul",:vmul),("fdiv",:vfdiv)]#,("frem",:vrem)]
  ff = Symbol(f, :_fast)
  fop_fast = f === :vfdiv ? "fdiv fast" : op * ' ' * fast_flags(true)
  fop_contract = op * ' ' * fast_flags(false)
  @eval begin
    @generated  $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($fop_contract, W, T)
    @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = binary_op($fop_fast, W, T)
    @inline $f(v1::Vec{W,Float16}, v2::Vec{W,Float16}) where {W} = $f(convert(Vec{W,Float32}, v1), convert(Vec{W,Float32}, v2))
    @inline $ff(v1::Vec{W,Float16}, v2::Vec{W,Float16}) where {W} = $ff(convert(Vec{W,Float32}, v1), convert(Vec{W,Float32}, v2))
  end
end
@inline vsub(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.sub_float(a,b)
@inline vadd(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.add_float(a,b)
@inline vmul(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.mul_float(a,b)
@inline vsub_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.sub_float_fast(a,b)
@inline vadd_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.add_float_fast(a,b)
@inline vmul_fast(a::T,b::T) where {T<:Union{Float32,Float64}} = Base.mul_float_fast(a,b)

@inline vdiv(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = trunc(vfdiv_fast(v1, v2))
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:FloatingTypes} = trunc(vfdiv_fast(v1, v2))
@inline vdiv_fast(v1::T, v2::T) where {T<:FloatingTypes} = trunc(Base.FastMath.div_float_fast(v1, v2))
@inline vdiv_fast(v1::T, v2::T) where {T <: Number} = v1 ÷ v2
@inline vdiv(v1::T, v2::T) where {T <: Number} = v1 ÷ v2
@inline vdiv(v1::T, v2::T) where {T <: FloatingTypes} = vdiv_fast(v1, v2)
@inline vrem(a,b) = vfnmadd(vdiv_fast(a, b), b, a)
@inline vrem_fast(a,b) = vfnmadd(vdiv_fast(a, b), b, a)
# @inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = trunc(T, vfloat_fast(v1) / vfloat_fast(v2))
@inline vdiv_fast(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = trunc(T, vfloat(v1) / vfloat(v2))
@inline function vdiv_fast(v1, v2)
    v3, v4 = promote_div(v1, v2)
    vdiv_fast(v3, v4)
end
@inline vdiv_fast(v1::VecUnroll{N,1,T,T}, s::T) where {N,T<:SignedHW} = VecUnroll(fmap(Base.sdiv_int, data(v1), s))
@inline vdiv_fast(v1::VecUnroll{N,1,T,T}, s::T) where {N,T<:UnsignedHW} = VecUnroll(fmap(Base.udiv_int, data(v1), s))
@inline vdiv_fast(v1::VecUnroll{N,1,T,T}, v2::VecUnroll{N,1,T,T}) where {N,T<:SignedHW} = VecUnroll(fmap(Base.sdiv_int, data(v1), data(v2)))
@inline vdiv_fast(v1::VecUnroll{N,1,T,T}, v2::VecUnroll{N,1,T,T}) where {N,T<:UnsignedHW} = VecUnroll(fmap(Base.udiv_int, data(v1), data(v2)))

@inline vfdiv(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv(vfloat(a), vfloat(b))
# @inline vfdiv_fast(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv_fast(vfloat_fast(a), vfloat_fast(b))
@inline vfdiv_fast(a::AbstractSIMDVector{W}, b::AbstractSIMDVector{W}) where {W} = vfdiv_fast(vfloat(a), vfloat(b))
@inline vfdiv(a, b) = a / b
@inline vfdiv_fast(a, b) = Base.FastMath.div_fast(a, b)

for (f,op) ∈ ((:vand,:and_int),(:vor,:or_int),(:vxor,:xor_int))
  @eval @inline $f(b1::Bool, b2::Bool) = Base.$op(b1, b2)
end

for f ∈ [:vadd,:vsub,:vmul]
  for s ∈ [Symbol(""),:_fast,:_nsw,:_nuw,:_nw]
    fs = Symbol(f,s)
    @eval begin
      @inline function $fs(a::Union{FloatingTypes,IntegerTypesHW,AbstractSIMD}, b::Union{FloatingTypes,IntegerTypesHW,AbstractSIMD})
        c, d = promote(a, b)
        $fs(c, d)
      end
    end
  end
end
# @inline vsub(a::T, b::T) where {T<:Base.BitInteger} = Base.sub_int(a, b)
for (vf,bf) ∈ [
  (:vadd,:add_int),(:vsub,:sub_int),(:vmul,:mul_int),
  (:vadd_fast,:add_int),(:vsub_fast,:sub_int),(:vmul_fast,:mul_int),
  (:vadd_nsw,:add_int),(:vsub_nsw,:sub_int),(:vmul_nsw,:mul_int),
  (:vadd_nuw,:add_int),(:vsub_nuw,:sub_int),(:vmul_nuw,:mul_int),
  (:vadd_nw,:add_int),(:vsub_nw,:sub_int),(:vmul_nw,:mul_int),
]
  @eval begin
    @inline $vf(a::Int128, b::Int128) = Base.$bf(a, b)
    @inline $vf(a::UInt128, b::UInt128) = Base.$bf(a, b)
  end
end
# @inline vrem(a::Float32, b::Float32) = Base.rem_float_fast(a, b)
# @inline vrem(a::Float64, b::Float64) = Base.rem_float_fast(a, b)

@inline function Base.FastMath.add_fast(a::AbstractSIMD, b::AbstractSIMD, c::AbstractSIMD)
  Base.FastMath.add_fast(Base.FastMath.add_fast(a,b),c)
end
@inline function Base.FastMath.add_fast(a::T, b::T, c::T) where {T<:AbstractSIMD}
  Base.FastMath.add_fast(Base.FastMath.add_fast(a,b),c)
end

@inline function Base.FastMath.add_fast(a::AbstractSIMD, b::AbstractSIMD, c::AbstractSIMD, d::AbstractSIMD)
  x = Base.FastMath.add_fast(a,b)
  y = Base.FastMath.add_fast(c,d)
  Base.FastMath.add_fast(x, y)
end
@inline function Base.FastMath.add_fast(a::T, b::T, c::T, d::T) where {T<:AbstractSIMD}
  x = Base.FastMath.add_fast(a,b)
  y = Base.FastMath.add_fast(c,d)
  Base.FastMath.add_fast(x, y)
end

@inline function Base.FastMath.add_fast(a::AbstractSIMD, b::AbstractSIMD, c::AbstractSIMD, d::AbstractSIMD, e::AbstractSIMD, f::Vararg{Number,K}) where {K}
  x = Base.FastMath.add_fast(a,b)
  y = Base.FastMath.add_fast(c,d)
  Base.FastMath.add_fast(Base.FastMath.add_fast(x, y), e, f...)
end
@inline function Base.FastMath.add_fast(a::T, b::T, c::T, d::T, e::T, f::Vararg{T,K}) where {T<:AbstractSIMD,K}
  x = Base.FastMath.add_fast(a,b)
  y = Base.FastMath.add_fast(c,d)
  Base.FastMath.add_fast(Base.FastMath.add_fast(x, y), e, f...)
end




