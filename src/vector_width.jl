
register_size(::Type{T}) where {T} = REGISTER_SIZE
register_size(::Type{T}) where {T<:Union{Signed,Unsigned}} = SIMD_INTEGER_REGISTER_SIZE

intlog2(N::I) where {I <: Integer} = (8sizeof(I) - one(I) - leading_zeros(N)) % I
intlog2(::Type{T}) where {T} = intlog2(sizeof(T))
ispow2(x::Integer) = (x & (x - 1)) == zero(x)
nextpow2(W) = vshl(one(W), vsub_fast(8sizeof(W), leading_zeros(vsub_fast(W, one(W)))))
prevpow2(W) = vshl(one(W), vsub_fast(vsub_fast((8sizeof(W)) % UInt, one(UInt)), leading_zeros(W) % UInt))
prevpow2(W::Signed) = prevpow2(W % Unsigned) % Signed

function pick_vector_width_shift(::Type{T}) where {T<:NativeTypes}
    # W = pick_vector_width(T)
    Wshift = intlog2(register_size(T)) - intlog2(T)
    1 << Wshift, Wshift
end
function pick_vector_width_shift(N::Integer, ::Type{T}) where {T<:NativeTypes}
    Wshift_N = VectorizationBase.intlog2(2N - 1)
    Wshift_st = intlog2(register_size(T)) - VectorizationBase.intlog2(sizeof(T))
    Wshift = min(Wshift_N, Wshift_st)
    W = 1 << Wshift
    W, Wshift
end

function pick_vector_width_shift_from_size(N::Int, size_T::Int)
    Wshift_N = VectorizationBase.intlog2(2N - 1)
    Wshift_st = intlog2(REGISTER_SIZE) - VectorizationBase.intlog2(size_T)
    Wshift = min(Wshift_N, Wshift_st)
    W = 1 << Wshift
    W, Wshift
end

__pick_vector_width(min_W, max_W) = (min_W, max_W)
function __pick_vector_width(min_W::Int, max_W::Int, @nospecialize(_T))::Tuple{Int,Int}
    # function __pick_vector_width(min_W::Int, max_W::Int, _T)::Tuple{Int,Int}
    T = _T.parameters[1]
    if T === Bit
        min_W = 8
    elseif (SIMD_INTEGER_REGISTER_SIZE != REGISTER_SIZE) && T <: Integer # only check subtype if it matters
        max_W = min(max_W, SIMD_INTEGER_REGISTER_SIZE ÷ sizeof(T))
    else
        max_W = min(max_W, REGISTER_SIZE ÷ sizeof(T))
    end
    min_W, max_W
end
function __pick_vector_width(min_W::Int, max_W::Int, @nospecialize(T1), @nospecialize(T2), args...)::Tuple{Int,Int}
    min_W, max_W = __pick_vector_width(min_W, max_W, T1)
    __pick_vector_width(min_W, max_W, T2, args...)
end
function _pick_vector_width(vargs...)::Int
    min_W = 1
    max_W = REGISTER_SIZE
    min_W, max_W = __pick_vector_width(min_W, max_W, vargs...)
    max(min_W, max_W)
end

# function _pick_vector_width(vargs...)
#     min_W = 1
#     max_W = REGISTER_SIZE
#     for v ∈ vargs
#         T = v.parameters[1]
#         if T === Bit
#             min_W = 8
#         elseif (SIMD_INTEGER_REGISTER_SIZE != REGISTER_SIZE) && T <: Integer # only check subtype if it matters
#             max_W = min(max_W, SIMD_INTEGER_REGISTER_SIZE ÷ sizeof(T))
#         else
#             max_W = min(max_W, REGISTER_SIZE ÷ sizeof(T))
#         end
#     end
#     W = max(min_W, max_W)
# end
@generated function pick_vector_width_val(vargs...)
    W = _pick_vector_width(vargs...)
    Expr(:call, Expr(:curly, :StaticInt, W))
end
adjust_W(N, W) = min(nextpow2(N), W)
@generated function pick_vector_width_val(::Union{Val{N},StaticInt{N}}, vargs...) where {N}
    W = adjust_W(N, _pick_vector_width(vargs...))
    Expr(:call, Expr(:curly, :StaticInt, W))
end
pick_vector_width(::Union{StaticInt{N},Val{N}}, args...) where {N} = Int(pick_vector_width_val(StaticInt{N}(), args...))

pick_vector_width(::Type{T}) where {T} = Int(pick_vector_width_val(T))
pick_vector_width(N::Integer, T) = min(nextpow2(N), pick_vector_width(T))

function int_type_symbol(W)
    bits = 8*(SIMD_INTEGER_REGISTER_SIZE ÷ W)
    if bits ≤ 8
        :Int8
    elseif bits ≤ 16
        :Int16
    elseif bits ≤ 32
        :Int32
    else # even if Int === Int32? Or should this be `Int`?
        :Int64
    end
end
@generated int_type(::Union{Val{W},StaticInt{W}}) where {W} = int_type_symbol(W)

pick_vector(::Type{T}) where {T} = _pick_vector(pick_vector_width_val(T), T)
_pick_vector(::StaticInt{W}, ::Type{T}) where {W,T} = Vec{W,T}
@generated pick_vector(::Val{N}, ::Type{T}) where {N, T} =  Expr(:curly, :Vec, pick_vector_width(N, T), T)
pick_vector(N::Int, ::Type{T}) where {T} = pick_vector(Val(N), T)

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
@inline vsub(i::MM) = i * StaticInt{-1}()
@inline vmul(::StaticInt{M}, i::MM{W,X}) where {M,W,X} = MM{W}(vmul_fast(i.i, StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vmul(i::MM{W,X}, ::StaticInt{M}) where {M,W,X} = MM{W}(vmul_fast(i.i, StaticInt{M}()), StaticInt{X}() * StaticInt{M}())
@inline vrem(i::MM{W,X,I}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = i
@inline vrem(i::MM{W,X}, ::Type{I}) where {W,X,I<:IntegerTypesHW} = MM{W,X}(i.i % I)
@inline veq(::AbstractIrrational, ::MM{W,<:Integer}) where {W} = zero(Mask{W})
@inline veq(x::AbstractIrrational, i::MM{W}) where {W} = x == Vec(i)
@inline veq(::MM{W,<:Integer}, ::AbstractIrrational) where {W} = zero(Mask{W})
@inline veq(i::MM{W}, x::AbstractIrrational) where {W} = Vec(i) == x
                   

@generated function Base.promote_rule(::Type{MM{W,X,I}}, ::Type{T2}) where {W,X,I,T2<:NativeTypes}
    if register_size(T2) ≥ sizeof(T2) * W
        return :(Vec{$W,$T2})
    elseif T2 <: Signed
        return :(Vec{$W,$(int_type_symbol(W))})
    elseif T2 <: Unsigned
        return :(Vec{$W,unsigned($(int_type_symbol(W)))})
    else
        return :(Vec{$W,$T2})
    end
end

@inline function Base.in(m::MM{W,X,<:Integer}, r::AbstractUnitRange) where {W,X}
    vm = Vec(m)
    (vm ≥ first(r)) & (vm ≤ last(r))
end

