
power2check(x) = ( ~(x & (x - 1)) ) == -1

@generated function T_shift(::Type{T}) where {T}
    round(Int, log2(sizeof(T)))
end
@generated function pick_vector_width(::Type{T} = Float64) where T
    shift = T_shift(T)
    REGISTER_SIZE >> shift
end
@generated function pick_vector_width_shift(::Type{T} = Float64) where T
    shift = T_shift(T)
    W = REGISTER_SIZE >> shift
    Wshift = round(Int, log2(W))
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


@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)
