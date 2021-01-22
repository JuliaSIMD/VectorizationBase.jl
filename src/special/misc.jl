@inline Base.:^(v::AbstractSIMD{W,T}, i::IntegerTypesHW) where {W,T} = Base.power_by_squaring(v, i)
@inline Base.:^(v::AbstractSIMD{W,T}, i::IntegerTypesHW) where {W,T<:Union{Float32,Float64}} = Base.power_by_squaring(v, i)
# @inline relu(x) = (y = zero(x); ifelse(x > y, x, y)) 
@inline relu(x) = (y = zero(x); ifelse(x < y, y, x))

@inline Base.fld(x::AbstractSIMD, y::AbstractSIMD) = div(promote_div(x,y)..., RoundDown)
@inline Base.fld(x::AbstractSIMD, y::Real) = div(promote_div(x,y)..., RoundDown)
@inline Base.fld(x::Real, y::AbstractSIMD) = div(promote_div(x,y)..., RoundDown)

@inline function Base.div(x::AbstractSIMD{W,T}, y::AbstractSIMD{W,T}, ::RoundingMode{:Down}) where {W,T<:Integer}
    d = div(x, y)
    d - (signbit(x ⊻ y) & (d * y != x))
end

@inline Base.mod(x::AbstractSIMD{W,T}, y::AbstractSIMD{W,T}) where {W,T<:Integer} =
    ifelse(y == -1, zero(x), x - fld(x, y) * y)

@inline Base.mod(x::AbstractSIMD{W,T}, y::AbstractSIMD{W,T}) where {W,T<:Unsigned} =
    rem(x, y)

@inline function Base.mod(x::AbstractSIMD{W,T1}, y::T2) where {W,T1<:SignedHW,T2<:UnsignedHW}
    _x, _y = promote_div(x, y)
    unsigned(mod(_x, _y))
end
@inline function Base.mod(x::AbstractSIMD{W,T1}, y::T2) where {W,T1<:UnsignedHW,T2<:SignedHW}
    _x, _y = promote_div(x, y)
    signed(mod(_x, _y))
end
@inline function Base.mod(x::AbstractSIMD{W,T1}, y::AbstractSIMD{W,T2}) where {W,T1<:SignedHW,T2<:UnsignedHW}
    _x, _y = promote_div(x, y)
    unsigned(mod(_x, _y))
end
@inline function Base.mod(x::AbstractSIMD{W,T1}, y::AbstractSIMD{W,T2}) where {W,T1<:UnsignedHW,T2<:SignedHW}
    _x, _y = promote_div(x, y)
    signed(mod(_x, _y))
end

@inline Base.mod(i::AbstractSIMD{<:Any,<:Integer}, r::AbstractUnitRange{<:Integer}) =
    mod(i-first(r), length(r)) + first(r)

@inline Base.mod(x::AbstractSIMD, y::NativeTypes) = mod(promote_div(x,y)...)
@inline Base.mod(x::NativeTypes, y::AbstractSIMD) = mod(promote_div(x,y)...)

# avoid ambiguity with clamp(::Missing, lo, hi) in Base.Math at math.jl:1258
# but who knows what would happen if you called it
for (X,L,H) in Iterators.product(fill([:Any, :Missing, :AbstractSIMD], 3)...)
    any(==(:AbstractSIMD), (X,L,H)) || continue
    @eval @inline function Base.clamp(x::$X, lo::$L, hi::$H)
        x_, lo_, hi_ = promote(x, lo, hi)
        ifelse(x_ > hi_, hi_, ifelse(x_ < lo_, lo_, x_))
    end
end

@inline Base.clamp(x::AbstractSIMD{<:Any,<:Integer}, r::AbstractUnitRange{<:Integer}) =
    clamp(x, first(r), last(r))
