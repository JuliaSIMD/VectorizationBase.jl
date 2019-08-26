
align(x) = (x + REGISTER_SIZE-1) & -REGISTER_SIZE
align(x::Ptr{T}) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x)))

align(x, n) = (nm1 = n - 1; (x + nm1) & ~nm1)
