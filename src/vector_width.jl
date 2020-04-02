

function intlog2(N::I) where {I <: Integer}
    # This version may be slightly faster when vectorized?
    # u = 0x4330000000000000 + N
    # d = reinterpret(Float64,u) - 4503599627370496.0
    # u = (reinterpret(Int64, d) >> 52) - 1023
    # This version is easier to read and understand how it works,
    # and probably faster when evaluated serially
    u = 8sizeof(I) - 1 - leading_zeros(N)
    Base.unsafe_trunc(I, u)
end
intlog2(::Type{T}) where {T} = intlog2(sizeof(T))
ispow2(x::Integer) = (x & (x - 1)) == zero(x)
nextpow2(W) = one(W) << (8sizeof(W) - leading_zeros(W-1))
prevpow2(W) = one(W) << (8sizeof(W) - leading_zeros(W)-1)
    # W -= 1
    # W |= W >> 1
    # W |= W >> 2
    # W |= W >> 4
    # W |= W >> 8
    # W |= W >> 16
    # W + 1

@generated function T_shift(::Type{T}) where {T}
    intlog2(sizeof(T))
end
@generated function pick_vector_width(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    max(1, REGISTER_SIZE >>> shift)
end
@generated function pick_vector_width_shift(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    W = max(1, REGISTER_SIZE >>> shift)
    Wshift = intlog2(W)
    W, Wshift
end
function downadjust_W_and_Wshift(N, W, Wshift)
    N > W && return W, Wshift
    TwoN = N << 1
    while W >= TwoN
        W >>>= 1
        Wshift -= 1
    end
    W, Wshift
end
function pick_vector_width_shift(N::Integer, ::Type{T} = Float64) where {T}
    W, Wshift = pick_vector_width_shift(T)
    downadjust_W_and_Wshift(N, W, Wshift)
end
function pick_vector_width(N::Integer, ::Type{T} = Float64) where {T}
    W = pick_vector_width(T)
    first(downadjust_W_and_Wshift(N, W, 0))
end
function pick_vector_width_shift(N::Integer, size_T::Integer)
    W = max(1, REGISTER_SIZE ÷ size_T)
    Wshift = intlog2(W)
    downadjust_W_and_Wshift(N, W, Wshift)
end
function pick_vector_width(N::Integer, size_T::Integer)
    W = max(1, REGISTER_SIZE ÷ size_T)
    first(downadjust_W_and_Wshift(N, W, 0))
end


pick_vector_width(::Symbol, T) = pick_vector_width(T)
pick_vector_width_shift(::Symbol, T) = pick_vector_width_shift(T)

@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)

@generated pick_vector_width_val(::Type{T} = Float64) where {T} = Val{pick_vector_width(T)}()
@generated pick_vector_width_val(::Val{N}, ::Type{T} = Float64) where {N,T} = Val{pick_vector_width(Val(N), T)}()

@generated function pick_vector_width_val(vargs...)
    sT = 16
    for v ∈ vargs
        T = v.parameters[1]
        if T == Bool
            sTv = REGISTER_SIZE >> 3 # encourage W ≥ 8
        else
            sTv = sizeof(v.parameters[1])
        end
        sT = min(sT, sTv)
    end
    Val{REGISTER_SIZE ÷ sT}()
end
@generated function adjust_W(::Val{N}, ::Val{W}) where {N,W}
    W1 = first(downadjust_W_and_Wshift(N, W, 0))
    Val{W1}()
end
pick_vector_width_val(::Val{N}, vargs...) where {N} = adjust_W(Val{N}(), pick_vector_width_val(vargs...))

@inline valmul(::Val{W}, i) where {W} = W*i
@inline valadd(::Val{W}, i) where {W} = W + i
@inline valsub(::Val{W}, i) where {W} = W - i
@inline valrem(::Val{W}, i) where {W} = i & (W - 1)
@inline valmuladd(::Val{W}, b, c) where {W} = W*b + c

@generated pick_vector(::Type{T}) where {T} = Vec{pick_vector_width(T),T}
pick_vector(N, T) = Vec{pick_vector_width(N, T),T}
@generated pick_vector(::Val{N}, ::Type{T}) where {N, T} =  pick_vector(N, T)

struct _MM{W,I<:Number}
    i::I
    @inline _MM{W}(i::T) where {W,T} = new{W,T}(i)
end
@inline _MM(::Val{W}) where {W} = _MM{W}(0)
@inline _MM(::Val{W}, i) where {W} = _MM{W}(i)
@inline _MM(::Val{W}, ::Static{I}) where {W,I} = _MM{W}(I)

@inline Base.:(+)(i::_MM{W}, j::Integer) where {W} = _MM{W}(i.i + j)
@inline Base.:(+)(i::Integer, j::_MM{W}) where {W} = _MM{W}(i + j.i)
@inline Base.:(+)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i + j)
@inline Base.:(+)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i + j.i)
@inline Base.:(+)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i + j.i)
@inline Base.:(-)(i::_MM{W}, j::Integer) where {W} = _MM{W}(i.i - j)
@inline Base.:(-)(i::Integer, j::_MM{W}) where {W} = _MM{W}(i - j.i)
@inline Base.:(-)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i - j)
@inline Base.:(-)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i - j.i)
@inline Base.:(-)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i - j.i)
# @inline Base.:(*)(i::_MM{W}, j) where {W} = _MM{W}(i.i * j)
# @inline Base.:(*)(i, j::_MM{W}) where {W} = _MM{W}(i * j.i)
# @inline Base.:(*)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i * j)
# @inline Base.:(*)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i * j.i)
# @inline Base.:(*)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i * j.i)
@inline scalar_less(i, j) = i < j
@inline scalar_less(i::_MM, j::Integer) = i.i < j
@inline scalar_less(i::Integer, j::_MM) = i < j.i
@inline scalar_less(i::_MM, ::Static{j}) where {j} = i.i < j
@inline scalar_less(::Static{i}, j::_MM) where {i} = i < j.i
@inline scalar_less(i::_MM, j::_MM) = i.i < j.i
@inline scalar_greater(i, j) = i > j
@inline scalar_greater(i::_MM, j::Integer) = i.i > j
@inline scalar_greater(i::Integer, j::_MM) = i > j.i
@inline scalar_greater(i::_MM, ::Static{j}) where {j} = i.i > j
@inline scalar_greater(::Static{i}, j::_MM) where {i} = i > j.i
@inline scalar_greater(i::_MM, j::_MM) = i.i > j.i
@inline scalar_equal(i, j) = i == j
@inline scalar_equal(i::_MM, j::Integer) = i.i == j
@inline scalar_equal(i::Integer, j::_MM) = i == j.i
@inline scalar_equal(i::_MM, ::Static{j}) where {j} = i.i == j
@inline scalar_equal(::Static{i}, j::_MM) where {i} = i == j.i
@inline scalar_equal(i::_MM, j::_MM) = i.i == j.i
@inline scalar_notequal(i, j) = i != j
@inline scalar_notequal(i::_MM, j::Integer) = i.i != j
@inline scalar_notequal(i::Integer, j::_MM) = i != j.i
@inline scalar_notequal(i::_MM, ::Static{j}) where {j} = i.i != j
@inline scalar_notequal(::Static{i}, j::_MM) where {i} = i != j.i
@inline scalar_notequal(i::_MM, j::_MM) = i.i != j.i

@inline extract_data(i::_MM) = i.i

