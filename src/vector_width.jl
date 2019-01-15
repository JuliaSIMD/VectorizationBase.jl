
power2check(x) = ( ~(x & (x - 1)) ) == -1

function pick_vector_width(N::Integer, ::Type{T} = Float64) where T
    T_size = sizeof(T)
    W = REGISTER_SIZE ÷ T_size
    while W >= 2N
        W >>= 1
    end
    W
end
function pick_vector_width(::Type{T} = Float64) where T
    T_size = sizeof(T)
    REGISTER_SIZE ÷ T_size
end

@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)
