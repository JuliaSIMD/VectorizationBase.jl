
# nextpow2(W) = vshl(one(W), vsub_fast(8sizeof(W), leading_zeros(vsub_fast(W, one(W)))))


# @inline _pick_vector(::StaticInt{W}, ::Type{T}) where {W,T} = Vec{W,T}
# @inline pick_vector(::Type{T}) where {T} = _pick_vector(pick_vector_width(T), T)
# @inline function pick_vector(::Val{N}, ::Type{T}) where {N, T}
#     _pick_vector(smin(nextpow2(StaticInt{N}()), pick_vector_width(T)), T)
# end
# pick_vector(N::Int, ::Type{T}) where {T} = pick_vector(Val(N), T)

@inline MM(::Union{Val{W},StaticInt{W}}) where {W} = MM{W}(0)
@inline MM(::Union{Val{W},StaticInt{W}}, i) where {W} = MM{W}(i)
@inline MM(::Union{Val{W},StaticInt{W}}, i::AbstractSIMDVector{W}) where {W} = i
@inline MM(::StaticInt{W}, i, ::StaticInt{X}) where {W,X} = MM{W,X}(i)
@inline gep(ptr::Ptr, i::MM) = gep(ptr, data(i))

@inline Base.one(::Type{MM{W,X,I}}) where {W,X,I} = one(I)
@inline staticm1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(vsub_fast(data(i), one(I)))
@inline staticp1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(vadd_nsw(data(i), one(I)))
@inline vadd_fast(i::MM{W,X}, j::IntegerTypesHW) where {W,X} =
  MM{W,X}(vadd_fast(data(i), j))
@inline vadd_fast(i::IntegerTypesHW, j::MM{W,X}) where {W,X} =
  MM{W,X}(vadd_fast(i, data(j)))
@inline vadd_fast(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} =
  MM{W,X}(vadd_fast(data(i), StaticInt{j}()))
@inline vadd_fast(::StaticInt{i}, j::MM{W,X}) where {W,X,i} =
  MM{W,X}(vadd_fast(StaticInt{i}(), data(j)))
@inline vadd_fast(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i
@inline vadd_fast(::StaticInt{0}, j::MM{W,X}) where {W,X} = j
@inline vsub_fast(i::MM{W,X}, j::IntegerTypesHW) where {W,X} =
  MM{W,X}(vsub_fast(data(i), j))
@inline vsub_fast(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} =
  MM{W,X}(vsub_fast(data(i), StaticInt{j}()))
@inline vsub_fast(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i

@inline vadd_nsw(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(vadd_nsw(data(i), j))
@inline vadd_nsw(i::IntegerTypesHW, j::MM{W,X}) where {W,X} = MM{W,X}(vadd_nsw(i, data(j)))
@inline vadd_nsw(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} =
  MM{W,X}(vadd_nsw(data(i), StaticInt{j}()))
@inline vadd_nsw(i::MM, ::Zero) = i
@inline vadd_nsw(::StaticInt{i}, j::MM{W,X}) where {W,X,i} =
  MM{W,X}(vadd_nsw(StaticInt{i}(), data(j)))
@inline vadd_nsw(::Zero, j::MM{W,X}) where {W,X} = j
@inline vsub_nsw(i::MM{W,X}, j::IntegerTypesHW) where {W,X} = MM{W,X}(vsub_nsw(data(i), j))
@inline vsub_nsw(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} =
  MM{W,X}(vsub_nsw(data(i), StaticInt{j}()))
@inline vsub(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i
@inline vsub_fast(i::MM) = i * StaticInt{-1}()
@inline vsub_nsw(i::MM) = i * StaticInt{-1}()
@inline vsub(i::MM) = i * StaticInt{-1}()
@inline vadd(i::MM, j::IntegerTypesHW) = vadd_fast(i, j)
@inline vadd(j::IntegerTypesHW, i::MM) = vadd_fast(j, i)
@inline vsub(i::MM, j::IntegerTypesHW) = vsub_fast(i, j)
@inline vsub(j::IntegerTypesHW, i::MM) = vsub_fast(j, i)
@inline vadd(i::MM, ::StaticInt{j}) where {j} = vadd_fast(i, StaticInt{j}())
@inline vadd(::StaticInt{j}, i::MM) where {j} = vadd_fast(StaticInt{j}(), i)
@inline vsub(i::MM, ::StaticInt{j}) where {j} = vsub_fast(i, StaticInt{j}())
@inline vsub(::StaticInt{j}, i::MM) where {j} = vsub_fast(StaticInt{j}(), i)
@inline vadd(i::MM, ::Zero) = i
@inline vadd(::Zero, i::MM) = i
@inline vsub(::Zero, i::MM) = StaticInt{-1}() * i


@inline vmul_nsw(::StaticInt{M}, i::MM{W,X}) where {M,W,X} =
  MM{W}(vmul_nsw(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul_nsw(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} =
  MM{W}(vmul_nsw(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())

@inline vmul_fast(::StaticInt{M}, i::MM{W,X}) where {M,W,X} =
  MM{W}(vmul_fast(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul_fast(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} =
  MM{W}(vmul_fast(data(i), StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul(a, ::StaticInt{N}) where {N} = vmul_fast(a, StaticInt{N}())
@inline vmul(::StaticInt{N}, a) where {N} = vmul_fast(StaticInt{N}(), a)
@inline vmul(::StaticInt{N}, ::StaticInt{M}) where {N,M} = StaticInt{N}() * StaticInt{M}()

@inline vsub(a, ::StaticInt{N}) where {N} = vsub_fast(a, StaticInt{N}())
@inline vadd(a, ::StaticInt{N}) where {N} = vadd_fast(a, StaticInt{N}())
@inline vsub(::StaticInt{N}, a) where {N} = vsub_fast(StaticInt{N}(), a)
@inline vadd(::StaticInt{N}, a) where {N} = vadd_fast(StaticInt{N}(), a)
@inline vsub(::StaticInt{M}, ::StaticInt{N}) where {M,N} = StaticInt{M}() - StaticInt{N}()
@inline vadd(::StaticInt{M}, ::StaticInt{N}) where {M,N} = StaticInt{M}() + StaticInt{N}()

@inline vrem(i::MM{W,X,I}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = i
@inline vrem(i::MM{W,X}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = MM{W,X}(data(i) % I)
@inline veq(::AbstractIrrational, ::MM{W,<:Integer}) where {W} = zero(Mask{W})
@inline veq(x::AbstractIrrational, i::MM{W}) where {W} = x == Vec(i)
@inline veq(::MM{W,<:Integer}, ::AbstractIrrational) where {W} = zero(Mask{W})
@inline veq(i::MM{W}, x::AbstractIrrational) where {W} = Vec(i) == x

@inline vsub_nsw(i::MM, ::Zero) = i
@inline vsub_nsw(i::NativeTypes, j::MM{W,X}) where {W,X} =
  MM(StaticInt{W}(), vsub_nsw(i, data(j)), -StaticInt{X}())
@inline vsub_fast(i::NativeTypes, j::MM{W,X}) where {W,X} =
  MM(StaticInt{W}(), vsub_fast(i, data(j)), -StaticInt{X}())
@inline vsub_nsw(i::Union{FloatingTypes,IntegerTypesHW}, j::MM{W,X}) where {W,X} =
  MM(StaticInt{W}(), vsub_nsw(i, data(j)), -StaticInt{X}())
@inline vsub_fast(i::Union{FloatingTypes,IntegerTypesHW}, j::MM{W,X}) where {W,X} =
  MM(StaticInt{W}(), vsub_fast(i, data(j)), -StaticInt{X}())

@inline function Base.in(m::MM{W,X,<:Integer}, r::AbstractUnitRange) where {W,X}
  vm = Vec(m)
  (vm ≥ first(r)) & (vm ≤ last(r))
end

for op ∈ (:(+), :(-))
  @eval begin
    @inline Base.$op(vu::VecUnroll{N,1,T,T}, i::MM) where {N,T<:NativeTypes} =
      VecUnroll(fmap($op, data(vu), i))
    @inline Base.$op(i::MM, vu::VecUnroll{N,1,T,T}) where {N,T<:NativeTypes} =
      VecUnroll(fmap($op, i, data(vu)))
    @inline Base.$op(
      vu::VecUnroll{N,1,T,T},
      i::VecUnroll{N,W,I,MM{W,X,I}},
    ) where {N,W,T<:NativeTypes,I,X} = VecUnroll(fmap($op, data(vu), data(i)))
    @inline Base.$op(
      i::VecUnroll{N,W,I,MM{W,X,I}},
      vu::VecUnroll{N,1,T,T},
    ) where {N,W,T<:NativeTypes,I,X} = VecUnroll(fmap($op, data(i), data(vu)))
  end
end
# @inline Base.:(+)(vu::VecUnroll{N,W,T,T}, i::MM) = VecUnroll(fmap(+, data(vu), i))
