
ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}) where {T1,T2,W} = promote_type(T1,T2)
Base.@pure function _ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}) where {T1, T2, W}
    T_canon = promote_type(T1,T2)
    pick_vector_width(T_canon) < W ? T1 : T_canon
end
ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}) where {T1 <: Integer, T2 <: Integer,W} = _ff_promote_rule(T1,T2,Val{W}())
ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}) where {T1 <: FloatingTypes, T2<:FloatingTypes,W} = _ff_promote_rule(T1,T2,Val{W}())

Base.promote_rule(::Type{V}, ::Type{T2}) where {W,T1,T2<:NativeTypes,V<:AbstractSIMDVector{W,T1}} = Vec{W,ff_promote_rule(T1,T2,Val{W}())}

_assemble_vec_unroll(::Val{N}, ::Type{V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = VecUnroll{N,W,T,V}
Base.promote_rule(::Type{VecUnroll{N,W,T1,V}}, ::Type{T2}) where {N,W,T1,V,T2<:NativeTypes} = _assemble_vec_unroll(Val{N}(), promote_type(V,T2))
Base.promote_rule(::Type{<:VecUnroll{N,W,T,V1}}, ::Type{V2}) where {N,W,T,V1,V2<:AbstractSIMDVector{W}} = _assemble_vec_unroll(Val{N}(), promote_type(V1,V2))
Base.promote_rule(::Type{VecUnroll{N,W,T1,V1}}, ::Type{VecUnroll{N,W,T2,V2}}) where {N,W,T1,T2,V1,V2} = _assemble_vec_unroll(Val{N}(), promote_type(V1,V2))

Base.promote_rule(::Type{Bit}, ::Type{T}) where {T <: Number} = T

issigned(x) = issigned(typeof(x))
issigned(::Type{<:Signed}) = True()
issigned(::Type{<:Unsigned}) = False()
issigned(::Type{<:AbstractSIMD{<:Any,T}}) where {T} = issigned(T)
issigned(::Type{T}) where {T} = nothing
"""
 Promote, favoring <:Signed or <:Unsigned of first arg.
"""
@inline promote_div(x::Union{Integer,AbstractSIMD{<:Any,<:Integer}}, y::Union{Integer,AbstractSIMD{<:Any,<:Integer}}) = promote_div(x, y, issigned(x))
@inline promote_div(x, y) = promote(x, y)
@inline promote_div(x, y, ::Nothing) = promote(x, y) # for Integers that are neither Signed or Unsigned, e.g. Bool
@inline function promote_div(x::T1, y::T2, ::True) where {T1,T2}
    T = promote_type(T1, T2)
    signed(x % T), signed(y % T)
end
@inline function promote_div(x::T1, y::T2, ::False) where {T1,T2}
    T = promote_type(T1, T2)
    unsigned(x % T), unsigned(y % T)
end
itosize(i::Union{I,AbstractSIMD{<:Any,I}}, ::Type{J}) where {I,J} = signorunsign(i % J, issigned(I))
signorunsign(i, ::True) = signed(i)
signorunsign(i, ::False) = unsigned(i)

# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{T2}) where {M,N,W,T1,T2<:NativeTypes} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{Vec{W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecUnroll{M,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecTile{M,N,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}

@generated function _ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}, ::StaticInt{RS}) where {T1 <: IntegerTypes, T2 <: FloatingTypes,W, RS}
    T_canon = promote_type(T1,T2)
    (sizeof(T_canon) * W ≤ RS) && return T_canon
    @assert sizeof(T1) * W ≤ RS
    @assert sizeof(T1) == 4
    Float32
end
@inline function ff_promote_rule(::Type{T1}, ::Type{T2}, ::Val{W}) where {T1 <: IntegerTypes, T2 <: FloatingTypes,W}
    _ff_promote_rule(T1, T2, Val{W}(), register_size())
end
@generated function _promote_rule(
    ::Type{V1}, ::Type{V2}, ::StaticInt{RS}
) where {W,T1,T2,V1<:AbstractSIMDVector{W,T1},V2<:AbstractSIMDVector{W,T2},RS}
    T = promote_type(T1,T2) # `T1` and `T2` should be defined in `Base`
    if RS ≥ W * sizeof(T)
        return :(Vec{$W,$T})
    end
    if T === Float64 || T === Float32
        N, r1 = (sizeof(T) * W) ÷ RS
        Wnew, r2 = divrem(W, N)
        @assert iszero(r)
        
        return :(VecUnroll{$(N-1),$Wnew,$T,Vec{$Wnew,$T}})
        # Should we demote `Float64` -> `Float32`?
        # return :(Vec{$W,$T})
        # don't demote to smaller than `Float32`
        # return :(Vec{$W,Float32})
    end
    # They're both of integers
    V1MM = V1 <: MM
    V2MM = V2 <: MM
    if V1MM ⊻ V2MM
        V1MM ? :(Vec{$W,$T2}) : :(Vec{$W,$T1})
    else # either both are `MM` or neither are
        B = W ÷ sizeof(T)
        if !V1MM # if neither are
            B = max(4, B)
        end
        I = integer_of_bytes_symbol(B, T <: Unsigned)
        :(Vec{$W,$I})
    end
end
@inline function Base.promote_rule(::Type{V1}, ::Type{V2}) where {W,T1,T2,V1<:AbstractSIMDVector{W,T1},V2<:AbstractSIMDVector{W,T2}}
    _promote_rule(V1, V2, register_size(promote_type(T1,T2)))
end

maybethrow(::True) = throw(ArgumentError("The arguments were invalid."))
maybethrow(::False) = nothing

# not @generated, because calling `promote_type` on vector types
@inline function Base.promote_rule(
    ::Type{<:VecUnroll{Nm1,Wsplit,T,V1}}, ::Type{V2}
) where {Nm1,Wsplit,T,T2,V1,W,V2<:AbstractSIMDVector{W,T2}}
    maybethrow(ArrayInterface.ne(StaticInt{Nm1}() * StaticInt{Wsplit}() + StaticInt{Wsplit}(), StaticInt{W}()))
    V3 = Vec{Wsplit,T2}
    _assemble_vec_unroll(Val{Nm1}(), promote_type(V1,V3))
end
@inline function Base.promote_rule(
    ::Type{<:VecUnroll{Nm1,Wsplit,T,V1}}, ::Type{V2}
) where {Nm1,Wsplit,T,V1,W,V2<:Mask{W}}
    maybethrow(ArrayInterface.ne(StaticInt{Nm1}() * StaticInt{Wsplit}() + StaticInt{Wsplit}(), StaticInt{W}()))
    V3 = Mask{Wsplit,mask_type(StaticInt{Wsplit}())}
    _assemble_vec_unroll(Val{Nm1}(), promote_type(V1,V3))
end

