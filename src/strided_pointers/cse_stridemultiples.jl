struct OffsetPrecalc{
  T,
  N,
  C,
  B,
  R,
  X,
  M,
  P<:AbstractStridedPointer{T,N,C,B,R,X,M},
  I
} <: AbstractStridedPointer{T,N,C,B,R,X,M}
  ptr::P
  precalc::I
end
@inline Base.pointer(ptr::OffsetPrecalc) = pointer(getfield(ptr, :ptr))
@inline Base.similar(ptr::OffsetPrecalc, p::Ptr) =
  OffsetPrecalc(similar(getfield(ptr, :ptr), p), getfield(ptr, :precalc))
# @inline pointerforcomparison(p::OffsetPrecalc) = pointerforcomparison(getfield(p, :ptr))
# @inline pointerforcomparison(p::OffsetPrecalc, i::Tuple) = pointerforcomparison(p.ptr, i)
@inline offsetprecalc(x, ::Any) = x
@inline offsetprecalc(x::OffsetPrecalc, ::Val) = x
@inline offsetprecalc(x::StridedBitPointer, ::Val) = x
# @inline pointerforcomparison(p::AbstractStridedPointer) = pointer(p)
# @inline pointerforcomparison(p::AbstractStridedPointer, i) = gep(p, i)
@inline ArrayInterface.offsets(p::OffsetPrecalc) = offsets(getfield(p, :ptr))

@inline Base.strides(p::OffsetPrecalc) = static_strides(getfield(p, :ptr))
@inline ArrayInterface.static_strides(p::OffsetPrecalc) =
  static_strides(getfield(p, :ptr))

@inline function LayoutPointers.similar_no_offset(sptr::OffsetPrecalc, ptr::Ptr)
  OffsetPrecalc(
    similar_no_offset(getfield(sptr, :ptr), ptr),
    getfield(sptr, :precalc)
  )
end
@inline function LayoutPointers.similar_with_offset(
  sptr::OffsetPrecalc,
  ptr::Ptr,
  off::Tuple
)
  OffsetPrecalc(
    similar_with_offset(getfield(sptr, :ptr), ptr, off),
    getfield(sptr, :precalc)
  )
end
@inline LayoutPointers.bytestrides(p::OffsetPrecalc) =
  bytestrides(getfield(p, :ptr))
@inline LayoutPointers.bytestrideindex(p::OffsetPrecalc) =
  LayoutPointers.bytestrideindex(getfield(p, :ptr))

"""
Basically:

if I ∈ [3,5,7,9]
c[(I - 1) >> 1]
else
b * I
end

because

c = b .* [3, 5, 7, 9]
"""
@generated function lazymul(
  ::StaticInt{I},
  b,
  c::Tuple{Vararg{Any,N}}
) where {I,N}
  Is = (I - 1) >> 1
  ex = if (isodd(I) && 1 ≤ Is ≤ N) && (c.parameters[Is] !== nothing)
    Expr(:call, GlobalRef(Core, :getfield), :c, Is, false)
  elseif ((I ∈ (6, 10)) && ((I >> 2) ≤ N)) && (c.parameters[I>>2] !== nothing)
    Expr(
      :call,
      :lazymul,
      Expr(:call, Expr(:curly, :StaticInt, 2)),
      Expr(:call, GlobalRef(Core, :getfield), :c, I >> 2, false)
    )
  else
    Expr(:call, :lazymul, Expr(:call, Expr(:curly, :StaticInt, I)), :b)
  end
  Expr(:block, Expr(:meta, :inline), ex)
end
@inline lazymul(a, b, c) = lazymul(a, b)
@inline lazymul(a::StaticInt, b, ::Nothing) = lazymul(a, b)

_unwrap(@nospecialize(_::Type{StaticInt{N}})) where {N} = N
_unwrap(@nospecialize(_)) = nothing
# descript is a tuple of (unrollfactor) for each ind; if it shouldn't preallocate, unrollfactor may be set to 1
function precalc_quote_from_descript(
  @nospecialize(descript),
  contig::Int,
  @nospecialize(X)
)
  precalc = Expr(:tuple)
  anyprecalcs = anydynamicprecals = false
  pstrideextracts = Expr(:block)
  for (i, uf) ∈ enumerate(descript)
    if i > length(X)
      break
    elseif i == contig || uf < 3
      push!(precalc.args, nothing)
    else
      t = Expr(:tuple)
      Xᵢ = X[i]
      anyprecalcs = true
      if Xᵢ === nothing
        anydynamicprecals = true
        pstride_i = Symbol(:pstride_, i)
        push!(
          pstrideextracts.args,
          Expr(
            :(=),
            pstride_i,
            Expr(:call, GlobalRef(Core, :getfield), :pstride, i, false)
          )
        )
        for u = 3:2:uf
          push!(t.args, Expr(:call, :vmul_nw, u, pstride_i))
        end
      else
        for u = 3:2:uf
          push!(t.args, static(u * Xᵢ))
        end
      end
      push!(precalc.args, t)
    end
  end
  q = Expr(:block, Expr(:meta, :inline))
  if anydynamicprecals
    push!(q.args, :(pstride = static_strides(p)))
    push!(q.args, pstrideextracts)
  end
  if anyprecalcs
    push!(q.args, Expr(:call, :OffsetPrecalc, :p, precalc))
  else
    push!(q.args, :p)
  end
  q
end
@generated function offsetprecalc(
  p::AbstractStridedPointer{T,N,C,B,R,X,O},
  ::Val{descript}
) where {T,N,C,B,R,X,O,descript}
  x = known(X)
  any(isnothing, x) || return Expr(:block, Expr(:meta, :inline), :p)
  precalc_quote_from_descript(descript, C, x)
end

@inline tdot(ptr::OffsetPrecalc{T}, a, b) where {T} =
  tdot(pointer(ptr), a, b, getfield(ptr, :precalc))
@inline tdot(ptr::OffsetPrecalc, ::Tuple{}, ::Tuple{}) = (pointer(ptr), Zero())
