

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
    TwoN = 2N
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
@inline valmul(::Val{W}, i) where {W} = W*i
@inline valadd(::Val{W}, i) where {W} = W + i
@inline valsub(::Val{W}, i) where {W} = W - i
@inline valrem(::Val{W}, i) where {W} = i & (W - 1)
@inline valmuladd(::Val{W}, b, c) where {W} = W*b + c

@generated pick_vector(::Type{T}) where {T} = Vec{pick_vector_width(T),T}
pick_vector(N, T) = Vec{pick_vector_width(N, T),T}
@generated pick_vector(::Val{N}, ::Type{T}) where {N, T} =  pick_vector(N, T)

struct _MM{W}
    i::Int
end
@inline _MM(::Val{W}) where {W} = _MM{W}(0)
@inline _MM(::Val{W}, i) where {W} = _MM{W}(i)
@inline Base.:(+)(i::_MM{W}, j) where {W} = _MM{W}(i.i + j)
@inline Base.:(+)(i, j::_MM{W}) where {W} = _MM{W}(i + j.i)
@inline Base.:(+)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i + j)
@inline Base.:(+)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i + j.i)
@inline Base.:(+)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i + j.i)
@inline Base.:(*)(i::_MM{W}, j) where {W} = _MM{W}(i.i * j)
@inline Base.:(*)(i, j::_MM{W}) where {W} = _MM{W}(i * j.i)
@inline Base.:(*)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i * j)
@inline Base.:(*)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i * j.i)
@inline Base.:(*)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i * j.i)
@inline Base.:(<)(i::_MM, j) = i.i < j
@inline Base.:(<)(i, j::_MM) = i < j.i
@inline Base.:(<)(i::_MM, ::Static{j}) where {j} = i.i < j
@inline Base.:(<)(::Static{i}, j::_MM) where {i} = i < j.i
@inline Base.:(<)(i::_MM, j::_MM) = i.i < j.i
@inline Base.:(>)(i::_MM, j) = i.i > j
@inline Base.:(>)(i, j::_MM) = i > j.i
@inline Base.:(>)(i::_MM, ::Static{j}) where {j} = i.i > j
@inline Base.:(>)(::Static{i}, j::_MM) where {i} = i > j.i
@inline Base.:(>)(i::_MM, j::_MM) = i.i > j.i
@inline Base.:(==)(i::_MM, j) = i.i == j
@inline Base.:(==)(i, j::_MM) = i == j.i
@inline Base.:(==)(i::_MM, ::Static{j}) where {j} = i.i == j
@inline Base.:(==)(::Static{i}, j::_MM) where {i} = i == j.i
@inline Base.:(==)(i::_MM, j::_MM) = i.i == j.i
@inline Base.:(!=)(i::_MM, j) = i.i != j
@inline Base.:(!=)(i, j::_MM) = i != j.i
@inline Base.:(!=)(i::_MM, ::Static{j}) where {j} = i.i != j
@inline Base.:(!=)(::Static{i}, j::_MM) where {i} = i != j.i
@inline Base.:(!=)(i::_MM, j::_MM) = i.i != j.i
@inline extract_data(i::_MM) = i.i

