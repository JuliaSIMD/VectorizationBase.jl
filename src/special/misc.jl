@inline Base.:^(v::AbstractSIMD{W,T}, i::Integer) where {W,T} = Base.power_by_squaring(v, i)


