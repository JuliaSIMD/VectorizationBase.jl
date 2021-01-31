"""
    align(x::Union{Int,Ptr}, [n])

Return aligned memory address with minimum increment. `align` assumes `n` is a
power of 2.
"""
function align end
@inline align(x::Integer) = vadd_fast(x, Int(register_size()-One())) & Int(-register_size())
@inline align(x::Ptr{T}, arg) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x), arg))
@inline align(x::Ptr{T}) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x)))
@inline align(x::Integer, n) = (nm1 = n - One(); (x + nm1) & -n)
@inline align(x::Integer, ::StaticInt{N}) where {N} = (nm1 = N - 1; (x + nm1) & -N)
@inline align(x::Integer, ::Type{T}) where {T} = align(x, register_size() ÷ static_sizeof(T))

# @generated align(::Val{L}, ::Type{T}) where {L,T} = align(L, T)
aligntrunc(x::Integer, n) = x & -n
aligntrunc(x::Integer) = aligntrunc(x, register_size())
aligntrunc(x::Integer, ::Type{T}) where {T} = aligntrunc(x, register_size() ÷ sizeof(T))
alignment(x::Integer, N = 64) = reinterpret(Int, x) % N

function valloc(N::Integer, ::Type{T} = Float64, a = max(register_size(), cache_linesize())) where {T}
    # We want alignment to both vector and cacheline-sized boundaries
    size_T = max(1, sizeof(T))
    reinterpret(Ptr{T}, align(reinterpret(UInt,Libc.malloc(size_T*N + a - 1)), a))
end

