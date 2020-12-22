@inline Base.:^(v::AbstractSIMD{W,T}, i::Integer) where {W,T} = Base.power_by_squaring(v, i)
@inline Base.:^(v::AbstractSIMD{W,T}, i::Integer) where {W,T<:Union{Float32,Float64}} = Base.power_by_squaring(v, i)
@inline relu(x) = (y = zero(x); IfElse.ifelse(x > y, x, y))

@inline Base.fld(x::AbstractSIMDVector, y::AbstractSIMDVector) = div(promote(x,y)..., RoundDown)

@inline function Base.div(x::AbstractSIMDVector{W1,T}, y::AbstractSIMDVector{W2,T}, ::RoundingMode{:Down}) where {W1,W2,T<:Integer}
    d = div(x, y)
    d - (signbit(x ⊻ y) & (d * y != x))
end

@inline Base.mod(x::AbstractSIMDVector{W1,T}, y::AbstractSIMDVector{W2,T}) where {W1,W2,T<:Integer} =
    ifelse(y == -1, zero(x), x - fld(x, y) * y)

@inline Base.mod(i::AbstractSIMDVector{<:Any,<:Integer}, r::AbstractUnitRange{<:Integer}) =
    mod(i-first(r), length(r)) + first(r)

# avoid ambiguity with clamp(::Missing, lo, hi) in Base.Math at math.jl:1258
# but who knows what would happen if you called it
for (X,L,H) in Iterators.product(fill([:Any, :Missing, :AbstractSIMD], 3)...)
    any(==(:AbstractSIMD), (X,L,H)) || continue
    @eval @inline function Base.clamp(x::$X, lo::$L, hi::$H)
        x_, lo_, hi_ = promote(x, lo, hi)
        ifelse(x_ > hi_, hi_, ifelse(x_ < lo_, lo_, x_))
    end
end

@inline Base.clamp(x::AbstractSIMD{<:Any, <:Integer}, r::AbstractUnitRange{<:Integer}) =
    clamp(x, first(r), last(r))
