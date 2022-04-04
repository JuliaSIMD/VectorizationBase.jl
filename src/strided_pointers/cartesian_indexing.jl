
# Overloadable method, e.g to insert OffsetPrecalc's precalculated stride multiples
@inline tdot(ptr::AbstractStridedPointer, ::Tuple{}, ::Tuple{}) = (pointer(ptr), Zero())
@inline tdot(ptr::AbstractStridedPointer{T}, a, b) where {T} = tdot(pointer(ptr), a, b)

@inline function tdot(p::Ptr{T}, a::Tuple{A}, b::Tuple{B,Vararg}) where {T,A,B}
  p, lazymul(first(a), first(b))
end
@inline function tdot(
  p::Ptr{T},
  a::Tuple{A},
  b::Tuple{B,Vararg},
  c::Tuple{C,Vararg},
) where {T,A,B,C}
  p, lazymul(first(a), first(b), first(c))
end

@inline function tdot(
  p::Ptr{T},
  a::Tuple{A1,A2,Vararg},
  b::Tuple{B1,B2,Vararg},
) where {T,A1,A2,B1,B2}
  i = lazymul(first(a), first(b))
  p, j = tdot(p, tail(a), tail(b))
  add_indices(p, i, j)
end
@inline function tdot(
  p::Ptr{T},
  a::Tuple{A1,A2,Vararg},
  b::Tuple{B1,B2,Vararg},
  c::Tuple{C1,C2,Vararg},
) where {T,A1,A2,B1,B2,C1,C2}
  i = lazymul(first(a), first(b), first(c))
  p, j = tdot(p, tail(a), tail(b), tail(c))
  add_indices(p, i, j)
end

@inline function tdot(
  p::Ptr{T},
  a::Tuple{A1,A2,Vararg},
  b::Tuple{B1,B2,Vararg},
  c::Tuple{C1},
) where {T,A1,A2,B1,B2,C1}
  i = lazymul(first(a), first(b), first(c))
  p, j = tdot(p, tail(a), tail(b))
  add_indices(p, i, j)
end
