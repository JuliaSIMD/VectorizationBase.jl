
power2check(x) = ( ~(x & (x - 1)) ) == -1

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
function nextpow2(W)
    one(W) << (8sizeof(W) - leading_zeros(W-1))
    # W -= 1
    # W |= W >> 1
    # W |= W >> 2
    # W |= W >> 4
    # W |= W >> 8
    # W |= W >> 16
    # W + 1
end
@generated function T_shift(::Type{T}) where {T}
    intlog2(sizeof(T))
end
@generated function pick_vector_width(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    max(1, REGISTER_SIZE >> shift)
end
@generated function pick_vector_width_shift(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    W = max(1, REGISTER_SIZE >> shift)
    Wshift = intlog2(W)
    W, Wshift
end
function pick_vector_width(N::Integer, ::Type{T} = Float64) where {T}
    W = pick_vector_width(T)
    N > W && return W
    TwoN = 2N
    while W >= TwoN
        W >>= 1
    end
    W
end
function pick_vector_width_shift(N::Integer, ::Type{T} = Float64) where {T}
    W, Wshift = pick_vector_width_shift(T)
    N > W && return W, Wshift
    TwoN = 2N
    while W >= TwoN
        W >>= 1
        Wshift -= 1
    end
    W, Wshift
end
pick_vector_width(::Symbol, T) = pick_vector_width(T)
pick_vector_width_shift(::Symbol, T) = pick_vector_width_shift(T)

@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)
