#TODO: Document interface to support static size
# Define maybestaticsize, maybestaticlength, and maybestaticfirstindex 
@inline maybestaticfirst(a) = static_first(a)
@inline maybestaticlast(a) = static_last(a)
@inline maybestaticlength(a) = static_length(a)
@inline maybestaticlength(a::UnitRange{T}) where {T} =
  last(a) - first(a) + oneunit(T)

@inline maybestaticrange(r::Base.OneTo{T}) where {T} =
  ArrayInterface.OptionallyStaticUnitRange(StaticInt{1}(), last(r))
@inline maybestaticrange(r::UnitRange) = r
@inline maybestaticrange(r) = maybestaticfirst(r):maybestaticlast(r)

@inline maybestaticsize(::NTuple{N}, ::Val{1}) where {N} = StaticInt{N}() # should we assert that i == 1?
@inline maybestaticsize(
  ::LinearAlgebra.Adjoint{T,V},
  ::Val{1}
) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticsize(
  ::LinearAlgebra.Transpose{T,V},
  ::Val{1}
) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticsize(A, ::Val{N}) where {N} =
  ArrayInterface.static_size(A)[N]

# These have versions that may allow for more optimizations, so we override base methods with a single `StaticInt` argument.
for (f, ff) ∈ [
  (:(Base.:+), :vadd_fast),
  (:(Base.:-), :vsub_fast),
  (:(Base.:*), :vmul_fast),
  (:(Base.:+), :vadd_nsw),
  (:(Base.:-), :vsub_nsw),
  (:(Base.:*), :vmul_nsw),
  (:(Base.:+), :vadd_nuw),
  (:(Base.:-), :vsub_nuw),
  (:(Base.:*), :vmul_nuw),
  (:(Base.:+), :vadd_nw),
  (:(Base.:-), :vsub_nw),
  (:(Base.:*), :vmul_nw),
  (:(Base.:<<), :vshl),
  (:(Base.:÷), :vdiv),
  (:(Base.:%), :vrem),
  (:(Base.:>>>), :vashr)
]
  @eval begin
    # @inline $f(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{$f(M, N)}()
    # If `M` and `N` are known at compile time, there's no need to add nsw/nuw flags.
    @inline $ff(::StaticInt{M}, ::StaticInt{N}) where {M,N} =
      $f(StaticInt{M}(), StaticInt{N}())
    # @inline $f(::StaticInt{M}, x) where {M} = $ff(M, x)
    # @inline $f(x, ::StaticInt{M}) where {M} = $ff(x, M)
    @inline $ff(::StaticInt{M}, x::T) where {M,T<:IntegerTypesHW} =
      $ff(M % T, x)
    @inline $ff(x::T, ::StaticInt{M}) where {M,T<:IntegerTypesHW} =
      $ff(x, M % T)
    @inline $ff(::StaticInt{M}, x::T) where {M,T} = $ff(T(M), x)
    @inline $ff(x::T, ::StaticInt{M}) where {M,T} = $ff(x, T(M))
  end
end
for f ∈ [:vadd_fast, :vsub_fast, :vmul_fast]
  @eval begin
    @inline $f(::StaticInt{M}, n::T) where {M,T<:Number} = $f(T(M), n)
    @inline $f(m::T, ::StaticInt{N}) where {N,T<:Number} = $f(m, T(N))
  end
end
for f ∈ [:vsub, :vsub_fast, :vsub_nsw, :vsub_nuw, :vsub_nw]
  @eval begin
    @inline $f(::Zero, m::Number) = -m
    @inline $f(::Zero, m::IntegerTypesHW) = -m
    @inline $f(m::Number, ::Zero) = m
    @inline $f(m::IntegerTypesHW, ::Zero) = m
    @inline $f(::Zero, ::Zero) = Zero()
    @inline $f(::Zero, ::StaticInt{N}) where {N} = -StaticInt{N}()
    @inline $f(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
  end
end
for f ∈ [:vadd, :vadd_fast, :vadd_nsw, :vadd_nuw, :vadd_nw]
  @eval begin
    @inline $f(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
    @inline $f(::Zero, ::StaticInt{N}) where {N} = StaticInt{N}()
    @inline $f(::Zero, ::Zero) = Zero()
    @inline $f(a::Number, ::Zero) = a
    @inline $f(a::IntegerTypesHW, ::Zero) = a
    @inline $f(::Zero, a::Number) = a
    @inline $f(::Zero, a::IntegerTypesHW) = a
  end
end

@inline vmul_fast(::StaticInt{N}, ::Zero) where {N} = Zero()
@inline vmul_fast(::Zero, ::StaticInt{N}) where {N} = Zero()
@inline vmul_fast(::Zero, ::Zero) = Zero()
@inline vmul_fast(::StaticInt{N}, ::One) where {N} = StaticInt{N}()
@inline vmul_fast(::One, ::StaticInt{N}) where {N} = StaticInt{N}()
@inline vmul_fast(::One, ::One) = One()
@inline vmul_fast(a::Number, ::One) = a
@inline vmul_fast(a::MM, ::One) = a
@inline vmul_fast(a::IntegerTypesHW, ::One) = a
@inline vmul_fast(::One, a::Number) = a
@inline vmul_fast(::One, a::MM) = a
@inline vmul_fast(::One, a::IntegerTypesHW) = a
@inline vmul_fast(::Zero, ::One) = Zero()
@inline vmul_fast(::One, ::Zero) = Zero()

for T ∈ [:VecUnroll, :AbstractMask, :MM]
  @eval begin
    @inline Base.:(+)(x::$T, ::Zero) = x
    @inline Base.:(+)(::Zero, x::$T) = x
    @inline Base.:(-)(x::$T, ::Zero) = x
    @inline Base.:(*)(x::$T, ::One) = x
    @inline Base.:(*)(::One, x::$T) = x
    @inline Base.:(*)(::$T, ::Zero) = Zero()
    @inline Base.:(*)(::Zero, ::$T) = Zero()
  end
end
@inline Base.:(+)(m::AbstractMask{W}, ::StaticInt{N}) where {N,W} =
  m + vbroadcast(Val{W}(), N)
@inline Base.:(+)(::StaticInt{N}, m::AbstractMask{W}) where {N,W} =
  vbroadcast(Val{W}(), N) + m
# @inline Base.:(*)(::StaticInt{N}, m::Mask{W}) where {N,W} = vbroadcast(Val{W}(), N) * m
@inline vadd_fast(x::VecUnroll, ::Zero) = x
@inline vadd_fast(::Zero, x::VecUnroll) = x
@inline vsub_fast(x::VecUnroll, ::Zero) = x
@inline vmul_fast(x::VecUnroll, ::One) = x
@inline vmul_fast(::One, x::VecUnroll) = x
@inline vmul_fast(::VecUnroll, ::Zero) = Zero()
@inline vmul_fast(::Zero, ::VecUnroll) = Zero()

for V ∈ [:AbstractSIMD, :MM]
  @eval begin
    @inline Base.FastMath.mul_fast(::Zero, x::$V) = Zero()
    @inline Base.FastMath.mul_fast(::One, x::$V) = x
    @inline Base.FastMath.mul_fast(x::$V, ::Zero) = Zero()
    @inline Base.FastMath.mul_fast(x::$V, ::One) = x

    @inline Base.FastMath.add_fast(::Zero, x::$V) = x
    @inline Base.FastMath.add_fast(x::$V, ::Zero) = x

    @inline Base.FastMath.sub_fast(::Zero, x::$V) = -x
    @inline Base.FastMath.sub_fast(x::$V, ::Zero) = x
  end
end
