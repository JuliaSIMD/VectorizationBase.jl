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

@generated function mask_type(::Type{T}, ::Val{P}) where {T,P}
    mask_type(pick_vector_width(P, T))
end
@generated function mask_type(::Type{T}) where {T}
    mask_type(pick_vector_width(T))
end
@generated function mask_from_remainder(::Type{T}, r) where {T}
    U = mask_type(T)
    quote
        $(Expr(:meta,:inline))
        $(U(2))^r - $(one(U))
    end
end
@generated function max_mask(::Type{T}) where {T}
    W = pick_vector_width(T)
    U = mask_type(W)
    U(2^W-1)
end
