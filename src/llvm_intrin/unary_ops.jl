function sub_quote(W::Int, T::Symbol, fast::Bool)::Expr
  vtyp = vtype(W, T)
  instrs = "%res = fneg $(fast_flags(fast)) $vtyp %0\nret $vtyp %res"
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T}}, data(v)))
  end
end

@generated vsub(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} =
  sub_quote(W, JULIA_TYPES[T], false)
@generated vsub_fast(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} =
  sub_quote(W, JULIA_TYPES[T], true)

@inline vsub(v) = -v
@inline vsub_fast(v) = Base.FastMath.sub_fast(v)
@inline vsub(v::Vec{<:Any,<:NativeTypes}) = vsub(zero(v), v)
@inline vsub_fast(v::Vec{<:Any,<:UnsignedHW}) = vsub(zero(v), v)
@inline vsub_fast(v::Vec{<:Any,<:NativeTypes}) = vsub_fast(zero(v), v)
@inline vsub(x::NativeTypes) = Base.FastMath.sub_fast(x)
@inline vsub_fast(x::NativeTypes) = Base.FastMath.sub_fast(x)

@inline vinv(v) = inv(v)
@inline vinv(v::AbstractSIMD{W,<:FloatingTypes}) where {W} = vfdiv(one(v), v)
@inline vinv(v::AbstractSIMD{W,<:IntegerTypesHW}) where {W} = inv(float(v))
@inline Base.FastMath.inv_fast(v::AbstractSIMD) =
  Base.FastMath.div_fast(one(v), v)

@inline vabs(v) = abs(v)
@inline vabs(v::AbstractSIMD{W,<:Unsigned}) where {W} = v
@inline vabs(v::AbstractSIMD{W,<:Signed}) where {W} = ifelse(v > 0, v, -v)

@inline vround(v) = round(v)
@inline vround(v::AbstractSIMD{W,<:Union{Integer,StaticInt}}) where {W} = v
@inline vround(
  v::AbstractSIMD{W,<:Union{Integer,StaticInt}},
  ::RoundingMode
) where {W} = v

function bswap_quote(W::Int, T::Symbol, st::Int)::Expr
  typ = 'i' * string(8st)
  suffix = 'v' * string(W) * typ
  vtyp = "<$W x $typ>"
  decl = "declare $(vtyp) @llvm.bswap.$(suffix)($(vtyp))"
  instrs = """
    %res = call $vtyp @llvm.bswap.$(suffix)($vtyp %0)
    ret $vtyp %res
  """
  ret_type = :(_Vec{$W,$T})
  llvmcall_expr(
    decl,
    instrs,
    ret_type,
    :(Tuple{$ret_type}),
    vtyp,
    [vtyp],
    [:(data(x))]
  )
end
@generated Base.bswap(x::Vec{W,T}) where {T<:IntegerTypesHW,W} =
  bswap_quote(W, JULIA_TYPES[T], sizeof(T))
@inline Base.bswap(x::VecUnroll{<:Any,<:Any,<:IntegerTypesHW}) =
  VecUnroll(fmap(bswap, data(x)))
@inline Base.bswap(x::AbstractSIMDVector{<:Any,<:IntegerTypesHW}) =
  bswap(Vec(x))
@inline Base.bswap(x::AbstractSIMD{<:Any,Float16}) =
  reinterpret(Float16, bswap(reinterpret(UInt16, x)))
@inline Base.bswap(x::AbstractSIMD{<:Any,Float32}) =
  reinterpret(Float32, bswap(reinterpret(UInt32, x)))
@inline Base.bswap(x::AbstractSIMD{<:Any,Float64}) =
  reinterpret(Float64, bswap(reinterpret(UInt64, x)))
