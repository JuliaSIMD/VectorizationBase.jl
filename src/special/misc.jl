@inline Base.:^(v::AbstractSIMD, i::Integer) = Base.power_by_squaring(v, i)


