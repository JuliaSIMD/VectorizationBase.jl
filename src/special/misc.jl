@inline function pow_by_square(_v, e::IntegerTypesHW)
  v = float(_v)
  if e < 0
    v = y = inv(v)
    e = -e
  elseif e == 0
    return one(v)
  else
    y = v
  end
  tz = trailing_zeros(e)
  e >>= (tz + one(tz))
  while tz > zero(tz)
    y = Base.FastMath.mul_fast(y, y)
    tz -= one(tz)
  end
  x = y
  while e ≠ zero(e)
    y = Base.FastMath.mul_fast(y, y)
    tz = trailing_zeros(e)
    e >>= (tz + one(tz))
    while tz > zero(tz)
      y = Base.FastMath.mul_fast(y, y)
      tz -= one(tz)
    end
    x = Base.FastMath.mul_fast(x, y)
  end
  return x
end
@generated function pow_by_square(_v, ::StaticInt{E}) where {E}
  e = E
  q = Expr(:block, Expr(:meta, :inline), :(v = float(_v)))
  xdefined = false
  if e < 0
    push!(q.args, :(v = y = inv(v)))
    e = -e
  else
    push!(q.args, :(y = v))
  end
  mf = Base.FastMath.mul_fast
  while e ≠ zero(e)
    xdefined && push!(q.args, :(y = $mf(y, y)))
    tz = trailing_zeros(e)
    e >>= (tz + one(tz))
    while tz > zero(tz)
      push!(q.args, :(y = $mf(y, y)))
      tz -= one(tz)
    end
    if xdefined
      push!(q.args, :(x = $mf(x, y)))
    else
      xdefined = true
      push!(q.args, :(x = y))
    end
  end
  push!(q.args, :x)
  return q
end
# 5 = 101 = 2^2 + 2^0 # x^4 * x^1
# x^5 = x^4 * x

@inline Base.:^(v::AbstractSIMD{W,T}, i::IntegerTypesHW) where {W,T} =
  pow_by_square(v, i)
@inline Base.:^(
  v::AbstractSIMD{W,T},
  i::IntegerTypesHW
) where {W,T<:Union{Float32,Float64}} = pow_by_square(v, i)
@inline Base.:^(v::AbstractSIMD, ::StaticInt{N}) where {N} =
  pow_by_square(v, StaticInt{N}())
@inline Base.FastMath.pow_fast(v::AbstractSIMD, ::StaticInt{N}) where {N} =
  pow_by_square(v, StaticInt{N}())
@inline Base.FastMath.pow_fast(
  v::AbstractSIMD{W,T},
  i::IntegerTypesHW
) where {W,T} = pow_by_square(v, i)
@inline Base.FastMath.pow_fast(
  v::AbstractSIMD{W,T},
  i::IntegerTypesHW
) where {W,T<:Union{Float32,Float64}} = pow_by_square(v, i)
@inline Base.FastMath.pow_fast(v::AbstractSIMD, x::FloatingTypes) =
  exp2(Base.FastMath.log2_fast(v) * x)
@inline Base.FastMath.pow_fast(v::FloatingTypes, x::AbstractSIMD) =
  exp2(Base.FastMath.log2_fast(v) * x)
@inline Base.FastMath.pow_fast(v::AbstractSIMD, x::AbstractSIMD) =
  exp2(Base.FastMath.log2_fast(v) * x)
@inline Base.literal_pow(::typeof(^), x::AbstractSIMD, ::Val{N}) where {N} =
  pow_by_square(x, StaticInt(N))
# @inline relu(x) = (y = zero(x); ifelse(x > y, x, y))
@inline relu(x) = (y = zero(x); ifelse(x < y, y, x))

Base.sign(v::AbstractSIMD) = ifelse(v > 0, one(v), -one(v))

@inline Base.fld(x::AbstractSIMD, y::AbstractSIMD) =
  div(promote_div(x, y)..., RoundDown)
@inline Base.fld(x::AbstractSIMD, y::Real) =
  div(promote_div(x, y)..., RoundDown)
@inline Base.fld(x::Real, y::AbstractSIMD) =
  div(promote_div(x, y)..., RoundDown)

@inline function Base.div(
  x::AbstractSIMD{W,T},
  y::AbstractSIMD{W,T},
  ::RoundingMode{:Down}
) where {W,T<:IntegerTypes}
  d = div(x, y)
  d - (signbit(x ⊻ y) & (d * y != x))
end

@inline Base.mod(
  x::AbstractSIMD{W,T},
  y::AbstractSIMD{W,T}
) where {W,T<:IntegerTypes} = ifelse(y == -1, zero(x), x - fld(x, y) * y)

@inline Base.mod(
  x::AbstractSIMD{W,T},
  y::AbstractSIMD{W,T}
) where {W,T<:Unsigned} = rem(x, y)

@inline function Base.mod(
  x::AbstractSIMD{W,T1},
  y::T2
) where {W,T1<:SignedHW,T2<:UnsignedHW}
  _x, _y = promote_div(x, y)
  unsigned(mod(_x, _y))
end
@inline function Base.mod(
  x::AbstractSIMD{W,T1},
  y::T2
) where {W,T1<:UnsignedHW,T2<:SignedHW}
  _x, _y = promote_div(x, y)
  signed(mod(_x, _y))
end
@inline function Base.mod(
  x::AbstractSIMD{W,T1},
  y::AbstractSIMD{W,T2}
) where {W,T1<:SignedHW,T2<:UnsignedHW}
  _x, _y = promote_div(x, y)
  unsigned(mod(_x, _y))
end
@inline function Base.mod(
  x::AbstractSIMD{W,T1},
  y::AbstractSIMD{W,T2}
) where {W,T1<:UnsignedHW,T2<:SignedHW}
  _x, _y = promote_div(x, y)
  signed(mod(_x, _y))
end

@inline Base.mod(
  i::AbstractSIMD{<:Any,<:IntegerTypes},
  r::AbstractUnitRange{<:IntegerTypes}
) = mod(i - first(r), length(r)) + first(r)

@inline Base.mod(x::AbstractSIMD, y::NativeTypes) = mod(promote_div(x, y)...)
@inline Base.mod(x::NativeTypes, y::AbstractSIMD) = mod(promote_div(x, y)...)

# avoid ambiguity with clamp(::Missing, lo, hi) in Base.Math at math.jl:1258
# but who knows what would happen if you called it
for (X, L, H) in Iterators.product(fill([:Any, :Missing, :AbstractSIMD], 3)...)
  any(==(:AbstractSIMD), (X, L, H)) || continue
  @eval @inline function Base.clamp(x::$X, lo::$L, hi::$H)
    x_, lo_, hi_ = promote(x, lo, hi)
    ifelse(x_ > hi_, hi_, ifelse(x_ < lo_, lo_, x_))
  end
end

@inline Base.FastMath.hypot_fast(x::AbstractSIMD, y::AbstractSIMD) = sqrt(
  Base.FastMath.add_fast(
    Base.FastMath.mul_fast(x, x),
    Base.FastMath.mul_fast(y, y)
  )
)

@inline Base.clamp(
  x::AbstractSIMD{<:Any,<:IntegerTypes},
  r::AbstractUnitRange{<:IntegerTypes}
) = clamp(x, first(r), last(r))

@inline function Base.gcd(
  a::AbstractSIMDVector{W,I},
  b::AbstractSIMDVector{W,I}
) where {W,I<:Base.HWReal}
  aiszero = a == zero(a)
  biszero = b == zero(b)
  absa = abs(a)
  absb = abs(b)
  za = trailing_zeros(a)
  zb = ifelse(biszero, zero(b), trailing_zeros(b))
  k = min(za, zb)
  u = unsigned(ifelse(biszero, zero(a), abs(a >> za)))
  v = unsigned(ifelse(aiszero, zero(b), abs(b >> zb)))
  ne = u ≠ v
  while vany(ne)
    ulev = (u > v) & ne
    t = u
    u = ifelse(ulev, v, u)
    v = ifelse(ulev, t, v)
    d = v - u
    v = ifelse(ne, d >> trailing_zeros(d), v)
    ne = u ≠ v
  end
  ifelse(aiszero, absb, ifelse(biszero, absa, (u << k) % I))
end
@inline Base.gcd(a::VecUnroll, b::Real) = VecUnroll(fmap(gcd, data(a), b))
@inline Base.gcd(a::Real, b::VecUnroll) = VecUnroll(fmap(gcd, a, data(b)))
@inline Base.gcd(a::VecUnroll, b::VecUnroll) =
  VecUnroll(fmap(gcd, data(a), data(b)))
@inline function Base.lcm(a::AbstractSIMD, b::AbstractSIMD)
  z = zero(a)
  isz = (a == z) | (b == z)
  ifelse(isz, z, (b ÷ gcd(b, a)) * a)
end
@inline Base.lcm(a::AbstractSIMD, b::Real) = ((c, d) = promote(a, b); lcm(c, d))
@inline Base.lcm(a::Real, b::AbstractSIMD) = ((c, d) = promote(a, b); lcm(c, d))

@inline function Base.getindex(
  A::Array,
  i::AbstractSIMD,
  j::Vararg{AbstractSIMD,K}
) where {K}
  vload(stridedpointer(A), (i, j...))
end

@inline Base.Sort.midpoint(
  lo::AbstractSIMDVector{W,I},
  hi::AbstractSIMDVector{W,I}
) where {W,I<:Integer} = lo + ((hi - lo) >>> 0x01)
for TType in [:Integer, :(AbstractSIMDVector{W,<:Integer})]
  @eval begin
    @inline function Base.searchsortedlast(
      v::Array,
      x::AbstractSIMDVector{W,I},
      lo::T,
      hi::T,
      o::Base.Ordering
    ) where {W,I,T<:$TType}
      u = convert(T, typeof(x)(1))
      lo = lo - u
      hi = hi + u
      st = lo < hi - u
      @inbounds while vany(st)
        m = Base.Sort.midpoint(lo, hi)
        b = Base.Order.lt(o, x, v[m]) & st
        hi = ifelse(b, m, hi)
        lo = ifelse(b, lo, m)
        st = lo < hi - u
      end
      return convert(typeof(x), lo)
    end
  end
end
@inline function Base.searchsortedlast(
  v::Array,
  x::VecUnroll,
  lo::T,
  hi::T,
  o::Base.Ordering
) where {T<:Integer}
  VecUnroll(fmap(searchsortedlast, v, data(x), lo, hi, o))
end
@inline function Base.searchsortedlast(
  v::Array,
  x::VecUnroll,
  lo::VecUnroll,
  hi::VecUnroll,
  o::Base.Ordering
)
  VecUnroll(fmap(searchsortedlast, v, data(x), data(lo), data(hi), o))
end
