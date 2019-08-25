



align(x) = (x + REGISTER_SIZE-1) & -REGISTER_SIZE
align(x::Ptr{T}) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x)))


