

@inline vstore!(ptr::AbstractStridedPointer{T}, v::Number) where {T<:Number} =
  __vstore!(pointer(ptr), convert(T, v), False(), False(), False(), register_size())

using LayoutPointers: nopromote_axis_indicator

@inline _vload(
  ptr::AbstractStridedPointer{T,0},
  i::Tuple{},
  ::A,
  ::StaticInt{RS},
) where {T,A<:StaticBool,RS} = __vload(pointer(ptr), A(), StaticInt{RS}())
@inline gep(ptr::AbstractStridedPointer{T,0}, i::Tuple{}) where {T} = pointer(ptr)

# terminating
@inline _offset_index(i::Tuple{}, offset::Tuple{}) = ()
@inline _offset_index(i::Tuple{I1}, offset::Tuple{I2,I3,Vararg}) where {I1,I2,I3} =
  (vsub_nsw(only(i), first(offset)),)
@inline _offset_index(i::Tuple{I1,I2,Vararg}, offset::Tuple{I3}) where {I1,I2,I3} =
  (vsub_nsw(first(i), first(offset)),)
@inline _offset_index(i::Tuple{I1}, offset::Tuple{I2}) where {I1,I2} =
  (vsub_nsw(only(i), only(offset)),)
# iterating
@inline _offset_index(
  i::Tuple{I1,I2,Vararg},
  offset::Tuple{I3,I4,Vararg},
) where {I1,I2,I3,I4} =
  (vsub_nsw(first(i), first(offset)), _offset_index(Base.tail(i), Base.tail(offset))...)

@inline offset_index(ptr, i) = _offset_index(i, offsets(ptr))
@inline linear_index(ptr, i) = tdot(ptr, offset_index(ptr, i), strides(ptr))

# Fast compile path?
@inline function _vload(
  ptr::AbstractStridedPointer,
  i::Tuple,
  ::A,
  ::StaticInt{RS},
) where {A<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vload(p, li, A(), StaticInt{RS}())
end
@inline function _vload(
  ptr::AbstractStridedPointer,
  i::Tuple,
  m::Union{AbstractMask,Bool},
  ::A,
  ::StaticInt{RS},
) where {A<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vload(p, li, m, A(), StaticInt{RS}())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T},
  i::Tuple{I},
  ::A,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vload(p, li, A(), StaticInt{RS}())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T},
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vload(p, li, m, A(), StaticInt{RS}())
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function _vload(
  ptr::AbstractStridedPointer{T,1},
  i::Tuple{I},
  ::A,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vload(p, li, A(), StaticInt{RS}())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T,1},
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vload(p, li, m, A(), StaticInt{RS}())
end

# align, noalias, nontemporal
@inline function _vstore!(
  ptr::AbstractStridedPointer,
  v,
  i::Tuple,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer,
  v,
  i::Tuple,
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)

  __vstore!(p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T},
  v,
  i::Tuple{I},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T},
  v,
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T,1},
  v,
  i::Tuple{I},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T,1},
  v,
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end


@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer,
  v,
  i::Tuple,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer,
  v,
  i::Tuple,
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer{T},
  v,
  i::Tuple{I},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(f, p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer{T},
  v,
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(f, p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer{T,1},
  v,
  i::Tuple{I},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
  f::F,
  ptr::AbstractStridedPointer{T,1},
  v,
  i::Tuple{I},
  m::Union{AbstractMask,Bool},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS},
) where {F,T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, m, A(), S(), NT(), StaticInt{RS}())
end


@inline function gep(
  ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}},
  i::Tuple{Vararg{Any,N}},
) where {T,N,C,B,R,X}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end
@inline function gep(
  ptr::AbstractStridedPointer{T,N,C,B,R,X,O},
  i::Tuple,
) where {T,N,C,B,R,X,O}
  p, li = linear_index(ptr, i)
  gep(p, li)
end
@inline function gep(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T,I}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end
@inline function gep(
  ptr::AbstractStridedPointer{T,1,C,B,R,X,O},
  i::Tuple{I},
) where {T,I,C,B,R,X,O}
  p, li = linear_index(ptr, i)
  gep(p, li)
end
@inline function gep(
  ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}},
  i::Tuple{I},
) where {T,I,C,B,R,X}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end


# There is probably a smarter way to do indexing adjustment here.
# The reasoning for the current approach of geping for Zero() on extracted inds
# and for offsets otherwise is best demonstrated witht his motivational example:
#
# A = OffsetArray(rand(10,11), 6:15, 5:15);
# for i in 6:15
# s += A[i,i]
# end
# first access is at zero-based index
# (first(6:16) - ArrayInterface.offsets(a)[1]) * ArrayInterface.strides(A)[1] + (first(6:16) - ArrayInterface.offsets(a)[2]) * ArrayInterface.strides(A)[2]
# equal to
#  (6 - 6)*1 + (6 - 5)*10 = 10
# i.e., the 1-based index 11.
# So now we want to adjust the offsets and pointer's value
# so that indexing it with a single `i` (after summing strides) is correct.
# Moving to 0-based indexing is easiest for combining strides. So we gep by 0 on these inds.
# E.g., gep(stridedpointer(A), (0,0))
# ptr += (0 - 6)*1 + (0 - 5)*10 = -56
# new stride = 1 + 10 = 11
# new_offset = 0
# now if we access the 6th element
# (6 - new_offse) * new_stride
# ptr += (6 - 0) * 11 = 66
# cumulative:
# ptr = pointer(A) + 66 - 56  = pointer(A) + 10
# so initial load is of pointer(A) + 10 -> the 11th element w/ 1-based indexing


function double_index_quote(C, B, R::NTuple{N,Int}, I1::Int, I2::Int) where {N}
  # place into position of second arg
  J1 = I1 + 1
  J2 = I2 + 1
  @assert (J1 != B) & (J2 != B)
  Cnew = ((C == J1) | (C == J2)) ? -1 : (C - (J1 < C))
  strd = Expr(:tuple)
  offs = Expr(:tuple)
  inds = Expr(:tuple)
  Rtup = Expr(:tuple)
  si = Expr(:curly, GlobalRef(ArrayInterface, :StrideIndex), N - 1, Rtup, Cnew)
  for n = 1:N
    if n == J1
      push!(inds.args, :(Zero()))
    elseif n == J2
      arg1 = Expr(:call, getfield, :strd, J1, false)
      arg2 = Expr(:call, getfield, :strd, J2, false)
      push!(strd.args, Expr(:call, :+, arg1, arg2))
      push!(offs.args, :(Zero()))
      push!(inds.args, :(Zero()))
      push!(Rtup.args, max(R[J1], R[J2]))
    else
      push!(strd.args, Expr(:call, getfield, :strd, n, false))
      push!(offs.args, Expr(:call, getfield, :offs, n, false))
      push!(inds.args, Expr(:call, getfield, :offs, n, false))
      push!(Rtup.args, R[n])
    end
  end
  gepedptr = Expr(:call, :gep, :ptr, inds)
  newptr =
    Expr(:call, :stridedpointer, gepedptr, Expr(:call, si, strd, offs), :(StaticInt{$B}()))
  Expr(:block, Expr(:meta, :inline), :(strd = strides(ptr)), :(offs = offsets(ptr)), newptr)
end
@generated function double_index(
  ptr::AbstractStridedPointer{T,N,C,B,R},
  ::Val{I1},
  ::Val{I2},
) where {T,N,C,B,R,I1,I2}
  double_index_quote(C, B, R, I1, I2)
end

using LayoutPointers: FastRange
# `FastRange{<:Union{Integer,StaticInt}}` can ignore the offset
@inline vload(r::FastRange{T,Zero}, i::Tuple{I}) where {T<:Union{Integer,StaticInt},I} =
  convert(T, getfield(r, :o)) + convert(T, getfield(r, :s)) * first(i)

@inline function vload(r::FastRange{T}, i::Tuple{I}) where {T<:FloatingTypes,I}
  convert(T, getfield(r, :f)) +
  convert(T, getfield(r, :s)) * (only(i) + convert(T, getfield(r, :o)))
end
@inline function gesp(r::FastRange{T,Zero}, i::Tuple{I}) where {I,T<:Union{Integer,StaticInt}}
  s = getfield(r, :s)
  FastRange{T}(Zero(), s, only(i) * s + getfield(r, :o))
end
@inline function gesp(r::FastRange{T}, i::Tuple{I}) where {I,T<:FloatingTypes}
  FastRange{T}(getfield(r, :f), getfield(r, :s), only(i) + getfield(r, :o))
end
@inline gesp(r::FastRange{T,Zero}, i::Tuple{NullStep}) where {T<:Union{Integer,StaticInt}} = r
@inline gesp(r::FastRange{T}, i::Tuple{NullStep}) where {T<:FloatingTypes} = r
@inline increment_ptr(r::FastRange{T,Zero}, i::Tuple{I}) where {I,T<:Union{Integer,StaticInt}} =
  only(i) * s + getfield(r, :o)
@inline increment_ptr(r::FastRange{T}, i::Tuple{I}) where {I,T<:Union{Integer,StaticInt}} =
  only(i) + getfield(r, :o)
@inline increment_ptr(r::FastRange) = getfield(r, :o)
@inline increment_ptr(r::FastRange{T}, o, i::Tuple{I}) where {I,T} = vadd_nsw(only(i), o)
@inline increment_ptr(r::FastRange{T,Zero}, o, i::Tuple{I}) where {I,T} =
  vadd_nsw(vmul_nsw(only(i), getfield(r, :s)), o)

@inline reconstruct_ptr(r::FastRange{T}, o) where {T} =
  FastRange{T}(getfield(r, :f), getfield(r, :s), o)

@inline vload(r::FastRange, i, m::AbstractMask) = (v = vload(r, i); ifelse(m, v, zero(v)))
@inline vload(r::FastRange, i, m::Bool) = (v = vload(r, i); ifelse(m, v, zero(v)))
@inline _vload(r::FastRange, i, _, __) = vload(r, i)
@inline _vload(r::FastRange, i, m::AbstractMask, __, ___) = vload(r, i, m)
@inline _vload(r::FastRange, i, m::VecUnroll{<:Any,<:Any,<:Union{Bool,Bit}}, __, ___) =
  vload(r, i, m)
function _vload_fastrange_unroll(
  AU::Int,
  F::Int,
  N::Int,
  AV::Int,
  W::Int,
  M::UInt,
  X::Int,
  mask::Bool,
  vecunrollmask::Bool,
)
  t = Expr(:tuple)
  inds = unrolled_indicies(1, AU, F, N, AV, W, X)
  q = quote
    $(Expr(:meta, :inline))
    gptr = gesp(r, data(u))
  end
  vecunrollmask && push!(q.args, :(masktup = data(vm)))
  gf = GlobalRef(Core, :getfield)
  for n = 1:N
    l = Expr(:call, :vload, :gptr, inds[n])
    if vecunrollmask
      push!(l.args, :($gf(masktup, $n, false)))
    elseif mask & (M % Bool)
      push!(l.args, :m)
    end
    M >>= 1
    push!(t.args, l)
  end
  push!(q.args, :(VecUnroll($t)))
  q
end

"""
For structs wrapping arrays, using `GC.@preserve` can trigger heap allocations.
`preserve_buffer` attempts to extract the heap-allocated part. Isolating it by itself
will often allow the heap allocations to be elided. For example:

```julia
julia> using StaticArrays, BenchmarkTools

julia> # Needed until a release is made featuring https://github.com/JuliaArrays/StaticArrays.jl/commit/a0179213b741c0feebd2fc6a1101a7358a90caed
       Base.elsize(::Type{<:MArray{S,T}}) where {S,T} = sizeof(T)

julia> @noinline foo(A) = unsafe_load(A,1)
foo (generic function with 1 method)

julia> function alloc_test_1()
           A = view(MMatrix{8,8,Float64}(undef), 2:5, 3:7)
           A[begin] = 4
           GC.@preserve A foo(pointer(A))
       end
alloc_test_1 (generic function with 1 method)

julia> function alloc_test_2()
           A = view(MMatrix{8,8,Float64}(undef), 2:5, 3:7)
           A[begin] = 4
           pb = parent(A) # or `LoopVectorization.preserve_buffer(A)`; `perserve_buffer(::SubArray)` calls `parent`
           GC.@preserve pb foo(pointer(A))
       end
alloc_test_2 (generic function with 1 method)

julia> @benchmark alloc_test_1()
BenchmarkTools.Trial:
  memory estimate:  544 bytes
  allocs estimate:  1
  --------------
  minimum time:     17.227 ns (0.00% GC)
  median time:      21.352 ns (0.00% GC)
  mean time:        26.151 ns (13.33% GC)
  maximum time:     571.130 ns (78.53% GC)
  --------------
  samples:          10000
  evals/sample:     998

julia> @benchmark alloc_test_2()
BenchmarkTools.Trial:
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     3.275 ns (0.00% GC)
  median time:      3.493 ns (0.00% GC)
  mean time:        3.491 ns (0.00% GC)
  maximum time:     4.998 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     1000
```
"""
@inline preserve_buffer(A::AbstractArray) = A
@inline preserve_buffer(
  A::Union{
    LinearAlgebra.Transpose,
    LinearAlgebra.Adjoint,
    Base.ReinterpretArray,
    Base.ReshapedArray,
    PermutedDimsArray,
    SubArray,
  },
) = preserve_buffer(parent(A))
@inline preserve_buffer(x) = x

function llvmptr_comp_quote(cmp, Tsym)
  pt = Expr(:curly, GlobalRef(Core, :LLVMPtr), Tsym, 0)
  instrs = "%cmpi1 = icmp $cmp i8* %0, %1\n%cmpi8 = zext i1 %cmpi1 to i8\nret i8 %cmpi8"
  Expr(
    :block,
    Expr(:meta, :inline),
    :($(Base.llvmcall)($instrs, Bool, Tuple{$pt,$pt}, p1, p2)),
  )
end
@inline llvmptrd(p::Ptr) = reinterpret(Core.LLVMPtr{Float64,0}, p)
@inline llvmptrd(p::AbstractStridedPointer) = llvmptrd(pointer(p))
for (op, f, cmp) ∈ [
  (:(<), :vlt, "ult"),
  (:(>), :vgt, "ugt"),
  (:(≤), :vle, "ule"),
  (:(≥), :vge, "uge"),
  (:(==), :veq, "eq"),
  (:(≠), :vne, "ne"),
]
  @eval begin
    @generated function $f(p1::Core.LLVMPtr{T,0}, p2::Core.LLVMPtr{T,0}) where {T}
      llvmptr_comp_quote($cmp, JULIA_TYPES[T])
    end
    @inline Base.$op(p1::P, p2::P) where {P<:AbstractStridedPointer} =
      $f(llvmptrd(p1), llvmptrd(p2))
    @inline Base.$op(p1::P, p2::P) where {P<:StridedBitPointer} =
      $op(linearize(p1), linearize(p2))
    @inline Base.$op(p1::P, p2::P) where {P<:FastRange} =
      $op(getfield(p1, :o), getfield(p2, :o))
    @inline $f(p1::Ptr, p2::Ptr, sp::AbstractStridedPointer) =
      $f(llvmptrd(p1), llvmptrd(p2))
    @inline $f(p1::NTuple{N,Int}, p2::NTuple{N,Int}, sp) where {N} =
      $op(reconstruct_ptr(sp, p1), reconstruct_ptr(sp, p2))
    @inline $f(a, b, c) = $f(a, b)
  end
end
@inline linearize(p::StridedBitPointer) = -sum(map(*, strides(p), offsets(p)))
