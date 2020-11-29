@inline Base.:^(v::AbstractSIMD{W,T}, i::Integer) where {W,T} = Base.power_by_squaring(v, i)
@inline Base.:^(v::AbstractSIMD{W,T}, i::Integer) where {W,T<:Union{Float32,Float64}} = Base.power_by_squaring(v, i)
@inline relu(x) = (y = zero(x); IfElse.ifelse(x > y, x, y))

