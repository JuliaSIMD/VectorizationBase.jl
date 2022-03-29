
@inline _maybefirst(x) = x
@inline _maybefirst(x::VecUnroll) = first(data(x))
@inline _maybetail(x) = x
@inline _maybetail(::VecUnroll{0}) = ()
@inline _maybetail(x::VecUnroll) = VecUnroll(Base.tail(data(x)))


@inline _vload_map(_, ::Tuple, ::Tuple{}, __, ___) = ()
@inline function _vload_map(p, i, m, ::J, ::A) where {J,A}
  x = _vload(p, map(_maybefirst, i), first(m), J(), A())
  r = _vload_map(p, map(_maybetail,i), Base.tail(m), J(), A())
  (x, r...)
end

@inline function _vload(
  p::AbstractStridedPointer,
  i::Tuple{Vararg{Union{IntegerIndex,MM,VecUnroll{N,<:Any,<:Any,<:IntegerIndex}}}},
  m::VecUnroll{N,<:Any,Bit}, ::J, ::A
) where {N,J,A}
  VecUnroll(_vload_map(p, i, data(m), J(), A()))
end

