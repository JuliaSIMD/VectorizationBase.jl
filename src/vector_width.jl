
power2check(x) = ( ~(x & (x - 1)) ) == -1

function intlog2(N)
    # u = UInt64(0x43300000) << 32
    # u += N
    u = 0x4330000000000000 + N
    d = reinterpret(Float64,u) - 4503599627370496.0
    u = (reinterpret(Int64, d) >> 52) - 1023
end

@generated function T_shift(::Type{T}) where {T}
    intlog2(sizeof(T))
end
@generated function pick_vector_width(::Type{T} = Float64) where T
    shift = T_shift(T)
    max(1, REGISTER_SIZE >> shift)
end
@generated function pick_vector_width_shift(::Type{T} = Float64) where T
    shift = T_shift(T)
    W = max(1, REGISTER_SIZE >> shift)
    Wshift = intlog2(W)
    W, Wshift
end
function pick_vector_width(N::Integer, ::Type{T} = Float64) where T
    W = pick_vector_width(T)
    TwoN = 2N
    while W >= TwoN
        W >>= 1
    end
    W
end
function pick_vector_width_shift(N::Integer, ::Type{T} = Float64) where T
    W, Wshift = pick_vector_width_shift(T)
    TwoN = 2N
    while W >= TwoN
        W >>= 1
        Wshift -= 1
    end
    W, Wshift
end
pick_vector_width(::Symbol, T::DataType) = pick_vector_width(T)
pick_vector_width_shift(::Symbol, T::DataType) = pick_vector_width_shift(T)

@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)
