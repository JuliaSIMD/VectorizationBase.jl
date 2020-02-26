


@inline extract_data(m::Mask) = m.u
@inline Base.:(&)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(&)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(&)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(&)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? m.u : zero(m.u))
@inline Base.:(&)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? m.u : zero(m.u))

@inline Base.:(|)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(|)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(|)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(|)(m::Mask{W,U}, b::Bool) where {W,U} = Mask{W}(b ? typemax(U) : m.u)
@inline Base.:(|)(b::Bool, m::Mask{W,U}) where {W,U} = Mask{W}(b ? typemax(U) : m.u)

@inline Base.:(⊻)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(⊻)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(⊻)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(⊻)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? ~m.u : m.u)
@inline Base.:(⊻)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? ~m.u : m.u)

@inline Base.:(<<)(m::Mask{W}, i) where {W} = Mask{W}(m.u << i)
@inline Base.:(>>)(m::Mask{W}, i) where {W} = Mask{W}(m.u >> i)
@inline Base.:(>>>)(m::Mask{W}, i) where {W} = Mask{W}(m.u >>> i)

@inline Base.:(~)(m::Mask{W}) where {W} = Mask{W}( ~m.u )
@inline Base.:(!)(m::Mask{W}) where {W} = Mask{W}( ~m.u )


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

@generated function max_mask(::Type{T}) where {T}
    W = pick_vector_width(T)
    U = mask_type(W)
    Mask{W,U}(one(U)<<W - one(U))
end

@generated function mask(::Type{T}, rem::Integer) where {T}
    M = mask_type(T)
    W = pick_vector_width(T)
    tup = Expr(:tuple, [Base.unsafe_trunc(M, 1 << w - 1) for w in 0:W]...) 
    quote
        $(Expr(:meta,:inline))
        # @inbounds $tup[rem+1]
        Mask{$W,$M}(one($M) << (rem & $(typemax(M))) - $(one(M)))
    end
end

@generated function mask(::Val{W}, rem::Integer) where {W}
    M = mask_type(W)
#    W = pick_vector_width(T)
    tup = Expr(:tuple, [Base.unsafe_trunc(M, 1 << w - 1) for w in 0:W]...) 
    quote
        $(Expr(:meta,:inline))
        # @inbounds $tup[rem+1]
        Mask{$W,$M}(one($M) << (rem & $(typemax(M))) - $(one(M)))
    end
end

unstable_mask(W, rem) = mask(Val(W), rem)

@generated function masktable(::Val{W}, rem::Integer) where {W}
    masks = Expr(:tuple)
    for w ∈ 0:W-1
        push!(masks.args, extract_data(unstable_mask(W, w == 0 ? W : w)))
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(:call, Expr(:curly, :Mask, W), Expr(
            :macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)),
            Expr(:call, :getindex, masks, Expr(:call, :+, 1, :rem))
        ))
    )
end

