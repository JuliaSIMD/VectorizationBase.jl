
# nextpow2(W) = vshl(one(W), vsub_fast(8sizeof(W), leading_zeros(vsub_fast(W, one(W)))))



# @inline _pick_vector(::StaticInt{W}, ::Type{T}) where {W,T} = Vec{W,T}
# @inline pick_vector(::Type{T}) where {T} = _pick_vector(pick_vector_width(T), T)
# @inline function pick_vector(::Val{N}, ::Type{T}) where {N, T}
#     _pick_vector(smin(nextpow2(StaticInt{N}()), pick_vector_width(T)), T)
# end
# pick_vector(N::Int, ::Type{T}) where {T} = pick_vector(Val(N), T)

@inline MM(::Union{Val{W},StaticInt{W}}) where {W} = MM{W}(0)
@inline MM(::Union{Val{W},StaticInt{W}}, i) where {W} = MM{W}(i)
@inline gep(ptr::Ptr, i::MM) = gep(ptr, i.i)

@inline staticm1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(vsub_fast(i.i, one(I)))
@inline staticp1(i::MM{W,X,I}) where {W,X,I} = MM{W,X}(vadd_fast(i.i, one(I)))
@inline vadd_fast(i::MM{W,X}, j::Integer) where {W,X} = MM{W,X}(vadd_fast(i.i, j))
@inline vadd_fast(i::Integer, j::MM{W,X}) where {W,X} = MM{W,X}(vadd_fast(i, j.i))
@inline vadd_fast(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vadd_fast(i.i, j))
@inline vadd_fast(::StaticInt{i}, j::MM{W,X}) where {W,X,i} = MM{W,X}(vadd_fast(i, j.i))
@inline vadd_fast(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i
@inline vadd_fast(::StaticInt{0}, j::MM{W,X}) where {W,X} = j
@inline vsub_fast(i::MM{W,X}, j::Integer) where {W,X} = MM{W,X}(vsub_fast(i.i, j))
@inline vsub_fast(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vsub_fast(i.i, j))
@inline vsub_fast(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i

@inline vadd(i::MM{W,X}, j::Integer) where {W,X} = MM{W,X}(vadd_fast(i.i, j))
@inline vadd(i::Integer, j::MM{W,X}) where {W,X} = MM{W,X}(vadd_fast(i, j.i))
@inline vadd(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vadd_fast(i.i, j))
@inline vadd(::StaticInt{i}, j::MM{W,X}) where {W,X,i} = MM{W,X}(vadd_fast(i, j.i))
@inline vsub(i::MM{W,X}, j::Integer) where {W,X} = MM{W,X}(vsub_fast(i.i, j))
@inline vsub(i::MM{W,X}, ::StaticInt{j}) where {W,X,j} = MM{W,X}(vsub_fast(i.i, j))
@inline vsub(i::MM{W,X}, ::StaticInt{0}) where {W,X} = i
@inline vsub(i::MM) = i * StaticInt{-1}()
@inline vmul(::StaticInt{M}, i::MM{W,X}) where {M,W,X} = MM{W}(vmul_fast(i.i, StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} = MM{W}(vmul_fast(i.i, StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vrem(i::MM{W,X,I}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = i
@inline vrem(i::MM{W,X}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = MM{W,X}(i.i % I)
@inline veq(::AbstractIrrational, ::MM{W,<:Integer}) where {W} = zero(Mask{W})
@inline veq(x::AbstractIrrational, i::MM{W}) where {W} = x == Vec(i)
@inline veq(::MM{W,<:Integer}, ::AbstractIrrational) where {W} = zero(Mask{W})
@inline veq(i::MM{W}, x::AbstractIrrational) where {W} = Vec(i) == x
                   


@inline function Base.in(m::MM{W,X,<:Integer}, r::AbstractUnitRange) where {W,X}
    vm = Vec(m)
    (vm ≥ first(r)) & (vm ≤ last(r))
end

@inline function pick_vector_width_shift(args::Vararg{Any,K}) where {K}
    W = pick_vector_width(args...)
    W, intlog2(W)
end
    

