"""
    align(x::Union{Int,Ptr}, [n])

Return aligned memory address with minimum increment. `align` assumes `n` is a
power of 2.
"""
function align end
@inline align(x::Integer) = (x + REGISTER_SIZE-1) & -REGISTER_SIZE
@inline align(x::Ptr{T}, args...) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x), args...))
@inline align(x::Integer, n) = (nm1 = n - 1; (x + nm1) & -n)
@inline align(x::Integer, ::Type{T}) where {T} = align(x, REGISTER_SIZE ÷ sizeof(T))
@inline align(x::Integer, ::Type{T}) where {T<:Union{Float64,Int64,UInt64}} = align(x, REGISTER_SIZE >>> 3)
@inline align(x::Integer, ::Type{T}) where {T<:Union{Float32,Int32,UInt32}} = align(x, REGISTER_SIZE >>> 2)
@inline align(x::Integer, ::Type{T}) where {T<:Union{Float16,Int16,UInt16}} = align(x, REGISTER_SIZE >>> 1)
@inline align(x::Integer, ::Type{T}) where {T<:Union{Int8,UInt8}} = align(x, REGISTER_SIZE)

# @generated align(::Val{L}, ::Type{T}) where {L,T} = align(L, T)
aligntrunc(x::Integer, n) = x & -n
aligntrunc(x::Integer) = aligntrunc(x, REGISTER_SIZE)
aligntrunc(x::Integer, ::Type{T}) where {T} = aligntrunc(x, REGISTER_SIZE ÷ sizeof(T))
alignment(x::Integer, N = 64) = reinterpret(Int, x) % N

function valloc(N::Int, ::Type{T} = Float64) where {T}
    # We want alignment to both vector and cacheline-sized boundaries
    a = max(REGISTER_SIZE, CACHELINE_SIZE) 
    reinterpret(Ptr{T}, align(reinterpret(UInt,Libc.malloc(sizeof(T)*N + a - 1)), a))
end
