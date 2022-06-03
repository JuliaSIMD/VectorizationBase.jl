"""
    align(x::Union{Int,Ptr}, [n])

Return aligned memory address with minimum increment. `align` assumes `n` is a
power of 2.
"""
function align end
@inline align(x::Union{Integer,StaticInt}) = (x + Int(register_size() - One())) & Int(-register_size())
@inline align(x::Ptr{T}, arg) where {T} =
  reinterpret(Ptr{T}, align(reinterpret(UInt, x), arg))
@inline align(x::Ptr{T}) where {T} = reinterpret(Ptr{T}, align(reinterpret(UInt, x)))
@inline align(x::Union{Integer,StaticInt}, n) = (nm1 = n - One(); (x + nm1) & -n)
@inline align(x::Union{Integer,StaticInt}, ::StaticInt{N}) where {N} = (nm1 = N - 1; (x + nm1) & -N)
@inline align(x::Union{Integer,StaticInt}, ::Type{T}) where {T} =
  align(x, register_size() ÷ static_sizeof(T))

# @generated align(::Val{L}, ::Type{T}) where {L,T} = align(L, T)
aligntrunc(x::Union{Integer,StaticInt}, n) = x & -n
aligntrunc(x::Union{Integer,StaticInt}) = aligntrunc(x, register_size())
aligntrunc(x::Union{Integer,StaticInt}, ::Type{T}) where {T} = aligntrunc(x, register_size() ÷ sizeof(T))
alignment(x::Union{Integer,StaticInt}, N = 64) = reinterpret(Int, x) % N

function valloc(
  N::Union{Integer,StaticInt},
  ::Type{T} = Float64,
  a = max(register_size(), cache_linesize()),
) where {T}
  # We want alignment to both vector and cacheline-sized boundaries
  size_T = max(1, sizeof(T))
  reinterpret(Ptr{T}, align(reinterpret(UInt, Libc.malloc(size_T * N + a - 1)), a))
end
