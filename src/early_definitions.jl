intlog2(N::I) where {I <: Integer} = (8sizeof(I) - one(I) - leading_zeros(N)) % I
intlog2(::Type{T}) where {T} = intlog2(sizeof(T))
nextpow2(W) = (one(W) << (8sizeof(W) - leading_zeros((W - one(W)))))
prevpow2(W) = (one(W) << (((8sizeof(W)) % UInt - one(UInt)) - leading_zeros(W) % UInt))
prevpow2(W::Signed) = prevpow2(W % Unsigned) % Signed

# This file must be loaded early to support `using VectorizationBase` with `--compiled-modules=no`.
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

function pick_vector_width_shift_from_size(N::Int, size_T::Int)::Tuple{Int,Int}
    Wshift_N = VectorizationBase.intlog2(2N - 1)
    Wshift_st = intlog2(register_size()) - VectorizationBase.intlog2(size_T)
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
    elseif (simd_integer_register_size() != register_size()) && T <: Integer # only check subtype if it matters
        max_W = min(max_W, simd_integer_register_size() ÷ sizeof(T))
    else
        max_W = min(max_W, register_size() ÷ sizeof(T))
    end
    min_W, max_W
end
function __pick_vector_width(min_W::Int, max_W::Int, @nospecialize(T1), @nospecialize(T2), args...)::Tuple{Int,Int}
    min_W, max_W = __pick_vector_width(min_W, max_W, T1)
    __pick_vector_width(min_W, max_W, T2, args...)
end
function _pick_vector_width(vargs...)::Int
    min_W = 1
    max_W = register_size()
    min_W, max_W = __pick_vector_width(min_W, max_W, vargs...)
    max(min_W, max_W)
end

# function _pick_vector_width(vargs...)
#     min_W = 1
#     max_W = register_size()
#     for v ∈ vargs
#         T = v.parameters[1]
#         if T === Bit
#             min_W = 8
#         elseif (simd_integer_register_size() != register_size()) && T <: Integer # only check subtype if it matters
#             max_W = min(max_W, simd_integer_register_size() ÷ sizeof(T))
#         else
#             max_W = min(max_W, register_size() ÷ sizeof(T))
#         end
#     end
#     W = max(min_W, max_W)
# end
pick_vector_width_val(::Type{T}) where {T} = sregister_size() ÷ static_sizeof(T)
pick_vector_width_val(::Type{Bit}) = sregister_size()
pick_vector_width_val(::Type{I}) where {I <: Integer} = ssimd_integer_register_size() ÷ static_sizeof(I)
@generated function pick_vector_width_val(vargs...)
    W = _pick_vector_width(vargs...)
    Expr(:call, Expr(:curly, :StaticInt, W))
end
adjust_W(N, W) = min(nextpow2(N), W)
@generated function pick_vector_width_val(::Union{Val{N},StaticInt{N}}, vargs...) where {N}
    W = adjust_W(N, _pick_vector_width(vargs...))
    Expr(:call, Expr(:curly, :StaticInt, W))
end

pick_vector_width(::Union{StaticInt{N},Val{N}}, args...) where {N} = Int(pick_vector_width_val(StaticInt{N}(), args...))::Int

pick_vector_width(::Type{T}) where {T} = Int(pick_vector_width_val(T))::Int
pick_vector_width(N::Integer, T)::Int = min(nextpow2(N), pick_vector_width(T))

function int_type_symbol(W)
    bits = 8*(simd_integer_register_size() ÷ W)
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

function pick_integer_bytes(W::Int, preferred::Int, minbytes::Int = min(preferred,4), sirs::Int = simd_integer_register_size())
    # SIMD quadword integer support requires AVX512DQ
    # preferred = AVX512DQ ? preferred :  min(4, preferred)
    max(minbytes,min(preferred, prevpow2(sirs ÷ W)))
end
function integer_of_bytes(bytes::Int)
    if bytes == 8
        Int64
    elseif bytes == 4
        Int32
    elseif bytes == 2
        Int16
    elseif bytes == 1
        Int8
    else
        throw("$bytes is an invalid number of bytes for integers.")
    end
end
function pick_integer(W::Int, pref::Int, minbytes::Int = min(pref,4))
    integer_of_bytes(pick_integer_bytes(W, pref, minbytes))
end
@generated pick_integer(::Val{W}) where {W} = pick_integer(W, sizeof(Int))
pick_integer(::Val{W}, ::Type{T}) where {W, T} = signorunsign(pick_integer(Val{W}()), issigned(T))

function mask_type_symbol(W)
    if W <= 8
        return :UInt8
    elseif W <= 16
        return :UInt16
    elseif W <= 32
        return :UInt32
    elseif W <= 64
        return :UInt64
    else#if W <= 128
        return :UInt128
    end
end
function mask_type(W)
    if W <= 8
        return UInt8
    elseif W <= 16
        return UInt16
    elseif W <= 32
        return UInt32
    elseif W <= 64
        return UInt64
    else#if W <= 128
        return UInt128
    end
end
mask_type(::Union{Val{1},StaticInt{1}}) = UInt8#Bool
mask_type(::Union{Val{2},StaticInt{2}}) = UInt8
mask_type(::Union{Val{4},StaticInt{4}}) = UInt8
mask_type(::Union{Val{8},StaticInt{8}}) = UInt8
mask_type(::Union{Val{16},StaticInt{16}}) = UInt16
mask_type(::Union{Val{32},StaticInt{32}}) = UInt32
mask_type(::Union{Val{64},StaticInt{64}}) = UInt64

@generated function mask_type(::Type{T}, ::Union{Val{P},StaticInt{P}}) where {T,P}
    mask_type_symbol(pick_vector_width(P, T))
end
@generated function mask_type(::Type{T}) where {T}
    W = max(1, register_size(T) >>> intlog2(T))
    mask_type_symbol(W)
    # mask_type_symbol(pick_vector_width(T))
end

