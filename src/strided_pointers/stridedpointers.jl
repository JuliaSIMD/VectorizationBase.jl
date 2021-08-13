# @generated function static_tuple(t::SDTuple{N,X,P}) where {N,X,P}
#     i = 0
#     sdt = Expr(:tuple)
#     Xv = tuple_type_to_value_tuple(X)
#     for n ∈ 1:N
#         push!(sdt.args, Xv[n] == -1 ? Expr(:ref, :x, (i += 1)) : Expr(:call, Expr(:curly, :StaticInt, Xv[n])))
#     end
#     q = Expr(:block, Expr(:meta, :inline))
#     i > 0 && push!(q.args, :(x = t.x))
#     push!(q.args, sdt)
#     q
# end

@generated function mulsizeof(::Type{T}, x::Number) where {T}
    st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :vmul_nw, st, :x))
end
@generated function mulsizeof(::Type{T}, ::StaticInt{N}) where {T,N}
    st = Base.allocatedinline(T) ? sizeof(T) : sizeof(Int)
    Expr(:block, Expr(:meta,:inline), Expr(:call, Expr(:curly, :StaticInt, N*st)))
end
@inline mulsizeof(::Type{T}, ::Tuple{}) where {T} = ()
@inline mulsizeof(::Type{T}, x::Tuple{X}) where {T,X} = (mulsizeof(T, first(x)), )
@inline mulsizeof(::Type{T}, x::Tuple) where {T} = (mulsizeof(T, first(x)), mulsizeof(T, Base.tail(x))...)

@inline bytestrides(A::AbstractArray{T}) where {T} = mulsizeof(T, ArrayInterface.strides(A))

# @generated function mulsizeof(::Type{T}, ::Type{S}) where {T, N, S <: Tuple{Vararg{Any,N}}}
#     Smul = Expr(:curly, :Tuple)
#     for n in 1:N
#         Sₙ = S.parameters[n]
#         push!(Smul.args, Sₙ == -1 ? -1 : sizeof(T) * Sₙ)
#     end
#     Smul
# end
# @inline function mulsizeof(::Type{T}, t::SDTuple{N,X,P}) where {X,N,P,T}
#     SDTuple{N,mulsizeof(T, X),P}(mulsizeof(T, t.x))
# end

@inline memory_reference(A::AbstractArray) = memory_reference(device(A), A)
@inline memory_reference(A::BitArray) = Base.unsafe_convert(Ptr{Bit}, A.chunks), A.chunks
@inline memory_reference(::CPUPointer, A) = pointer(A), preserve_buffer(A)
@inline memory_reference(::CPUPointer, A::Union{LinearAlgebra.Adjoint, Base.ReshapedArray, Base.PermutedDimsArray, LinearAlgebra.Transpose}) = memory_reference(CPUPointer(), parent(A))
@inline function memory_reference(::CPUPointer, A::Base.ReinterpretArray{T}) where {T}
  p, m = memory_reference(CPUPointer(), parent(A))
  reinterpret(Ptr{T}, p), m
end
@inline ind_diff(::Base.Slice, ::Any) = Zero()
@inline ind_diff(x::AbstractRange, o) = vsub_nsw(static_first(x), o)
@inline ind_diff(x::Integer, o) = vsub_nsw(x, o)
@inline function memory_reference(::CPUPointer, A::SubArray)
  p, m = memory_reference(CPUPointer(), parent(A))
  pA = parent(A)
  offset = ArrayInterface.reduce_tup(+, map(vmul_nsw, map(ind_diff, A.indices, offsets(pA)), strides(pA)))
  p + sizeof(eltype(A))*offset, m
end
@inline function memory_reference(::ArrayInterface.CPUTuple, A)
    r = Ref(A)
    Base.unsafe_convert(Ptr{eltype(A)}, r), r
end
@inline function memory_reference(::ArrayInterface.CheckParent, A)
    P = parent(A)
    if P === A
        memory_reference(ArrayInterface.CPUIndex(), A)
    else
        memory_reference(device(P), P)
    end
end
@inline memory_reference(::ArrayInterface.CPUIndex, A) = throw("Memory access for $(typeof(A)) not implemented yet.")

@inline ArrayInterface.contiguous_axis(::Type{A}) where {T,N,C,A<:AbstractStridedPointer{T,N,C}} = StaticInt{C}()
@inline ArrayInterface.contiguous_batch_size(::Type{A}) where {T,N,C,B,A<:AbstractStridedPointer{T,N,C,B}} = StaticInt{B}()
@inline ArrayInterface.stride_rank(::Type{A}) where {T,N,C,B,R,A<:AbstractStridedPointer{T,N,C,B,R}} = map(StaticInt, R)
@inline memory_reference(A::AbstractStridedPointer) = pointer(A), nothing

@inline Base.eltype(::AbstractStridedPointer{T}) where {T} = T

struct StridedPointer{T,N,C,B,R,X,O} <: AbstractStridedPointer{T,N,C,B,R,X,O}
    p::Ptr{T}
    strd::X
    offsets::O
end
@inline StridedPointer{T,N,C,B,R}(ptr::Ptr{T}, strd::X, o::O) where {T,N,C,B,R,X,O} = StridedPointer{T,N,C,B,R,X,O}(ptr, strd, o)
@inline StridedPointer{T,N,C,B,R,X}(ptr::Ptr{T}, strd::X, o::O) where {T,N,C,B,R,X,O} = StridedPointer{T,N,C,B,R,X,O}(ptr, strd, o)

@inline function stridedpointer(A::AbstractArray)#{T}) where {T <: NativeTypes}
    p, r = memory_reference(A)
    stridedpointer(p, contiguous_axis(A), contiguous_batch_size(A), val_stride_rank(A), bytestrides(A), offsets(A))
end
@inline function stridedpointer_preserve(A::AbstractArray)#{T}) where {T <: NativeTypes}
    p, r = memory_reference(A)
    stridedpointer(p, contiguous_axis(A), contiguous_batch_size(A), val_stride_rank(A), bytestrides(A), offsets(A)), r
end
@inline function stridedpointer(
    ptr::Ptr{T}, ::StaticInt{C}, ::StaticInt{B}, ::Val{R}, strd::X, offsets::O
) where {T,C,B,R,N,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
    StridedPointer{T,N,C,B,R,X,O}(ptr, strd, offsets)
end
@inline bytestrides(A::StridedPointer) = getfield(A, :strd)
@inline Base.strides(ptr::StridedPointer) = getfield(ptr, :strd)
@inline ArrayInterface.strides(ptr::StridedPointer) = getfield(ptr, :strd)
@inline ArrayInterface.offsets(ptr::StridedPointer) = getfield(ptr, :offsets)
@inline ArrayInterface.contiguous_axis_indicator(ptr::AbstractStridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(StaticInt{C}(), Val{N}())
@inline val_stride_rank(::AbstractStridedPointer{T,N,C,B,R}) where {T,N,C,B,R} = Val{R}()
@generated val_dense_dims(::AbstractStridedPointer{T,N}) where {T,N} = Val{ntuple(==(0), Val(N))}()

@generated function zerotuple(::Val{N}) where {N}
    t = Expr(:tuple);
    for n in 1:N
        push!(t.args, :(Zero()))
    end
    Expr(:block, Expr(:meta,:inline), t)
end
@inline center(p::AbstractStridedPointer{T,N}) where {T,N} = gesp(p, zerotuple(Val(N)))
@inline function zero_offsets(sptr::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
    StridedPointer{T,N,C,B,R}(getfield(sptr, :p), getfield(sptr, :strd), zerotuple(Val{N}()))
end
@inline zstridedpointer(A) = zero_offsets(stridedpointer(A))
@inline function zstridedpointer_preserve(A::AbstractArray)#{T}) where {T <: NativeTypes}
  p, r = stridedpointer_preserve(A)
  zero_offsets(p), r
end

@inline function Base.similar(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
    StridedPointer{T,N,C,B,R,X,O}(ptr, getfield(sptr, :strd), getfield(sptr, :offsets))
end
# @inline noalias!(p::StridedPointer) = similar(p, noalias!(pointer(p)))
@inline function similar_with_offset(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}, off) where {T,N,C,B,R,X,O}
    StridedPointer{T,N,C,B,R,X}(ptr, getfield(sptr, :strd), off)
end
@inline function similar_no_offset(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
    StridedPointer{T,N,C,B,R,X}(ptr, getfield(sptr, :strd), zerotuple(Val{N}()))
end

@inline Base.pointer(ptr::StridedPointer) = getfield(ptr, :p)
Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractStridedPointer{T}) where {T} = pointer(ptr)
# Shouldn't need to special case Array
# function stridedpointer(A::Array{T,N}) where {T,N}
#     StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
# end

@inline vstore!(ptr::AbstractStridedPointer{T}, v::Number) where {T<:Number} = __vstore!(pointer(ptr), convert(T,v), False(), False(), False(), register_size())

@generated function nopromote_axis_indicator(::AbstractStridedPointer{<:Any,N}) where {N}
    t = Expr(:tuple); foreach(n -> push!(t.args, True()), 1:N)
    Expr(:block, Expr(:meta, :inline), t)
end

@inline _vload(ptr::AbstractStridedPointer{T,0}, i::Tuple{}, ::A) where {T,A<:StaticBool,RS} = __vload(pointer(ptr), A())
@inline gep(ptr::AbstractStridedPointer{T,0}, i::Tuple{}) where {T} = pointer(ptr)

# terminating
@inline _offset_index(i::Tuple{}, offset::Tuple{}) = ()
@inline _offset_index(i::Tuple{I1}, offset::Tuple{I2,I3,Vararg}) where {I1,I2,I3} = (vsub_nsw(only(i), first(offset)),)
@inline _offset_index(i::Tuple{I1,I2,Vararg}, offset::Tuple{I3}) where {I1,I2,I3} = (vsub_nsw(first(i), first(offset)),)
@inline _offset_index(i::Tuple{I1}, offset::Tuple{I2}) where {I1,I2}           = (vsub_nsw(only(i), only(offset)),)
# iterating
@inline _offset_index(i::Tuple{I1,I2,Vararg}, offset::Tuple{I3,I4,Vararg}) where {I1,I2,I3,I4} = (vsub_nsw(first(i), first(offset)), _offset_index(Base.tail(i),Base.tail(offset))...)

@inline offset_index(ptr, i) = _offset_index(i, offsets(ptr))
@inline linear_index(ptr, i) = tdot(ptr, offset_index(ptr, i), strides(ptr))

# Fast compile path?
@inline function _vload(ptr::AbstractStridedPointer, i::Tuple, ::A) where {A<:StaticBool}
    p, li = linear_index(ptr, i)
    __vload(p, li, A())
end
@inline function _vload(
  ptr::AbstractStridedPointer, i::Tuple, m::Union{AbstractMask,Bool}, ::A
) where {A<:StaticBool}
  p, li = linear_index(ptr, i)
  __vload(p, li, m, A())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T}, i::Tuple{I}, ::A
) where {T,I,A<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vload(p, li, A())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T}, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A
) where {T,I,A<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vload(p, li, m, A())
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function _vload(
  ptr::AbstractStridedPointer{T,1}, i::Tuple{I}, ::A
) where {T,I,A<:StaticBool}
  p, li = linear_index(ptr, i)
  __vload(p, li, A())
end
@inline function _vload(
  ptr::AbstractStridedPointer{T,1}, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A
) where {T,I,A<:StaticBool}
  p, li = linear_index(ptr, i)
  __vload(p, li, m, A())
end

# align, noalias, nontemporal
@inline function _vstore!(
  ptr::AbstractStridedPointer, v, i::Tuple, ::A, ::S, ::NT
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer, v, i::Tuple, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)

  __vstore!(p, v, li, m, A(), S(), NT())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, ::A, ::S, ::NT
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(p, v, li, m, A(), S(), NT())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T,1}, v, i::Tuple{I}, ::A, ::S, ::NT
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  ptr::AbstractStridedPointer{T,1}, v, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(p, v, li, m, A(), S(), NT())
end


@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer, v, i::Tuple, ::A, ::S, ::NT
) where {F,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer, v, i::Tuple, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {F,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, m, A(), S(), NT())
end
@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, ::A, ::S, ::NT
) where {F, T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(f, p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {F, T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = tdot(ptr, i, strides(ptr))
  __vstore!(f, p, v, li, m, A(), S(), NT())
end
@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer{T,1}, v, i::Tuple{I}, ::A, ::S, ::NT
) where {F, T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, A(), S(), NT())
end
@inline function _vstore!(
  f::F, ptr::AbstractStridedPointer{T,1}, v, i::Tuple{I}, m::Union{AbstractMask,Bool}, ::A, ::S, ::NT
) where {F, T,I,A<:StaticBool,S<:StaticBool,NT<:StaticBool}
  p, li = linear_index(ptr, i)
  __vstore!(f, p, v, li, m, A(), S(), NT())
end


@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple) where {T,N,C,B,R,X,O}
  p, li = linear_index(ptr, i)
  gep(p, li)
end
@inline function gep(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T, I}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}) where {T, I,C,B,R,X,O}
  p, li = linear_index(ptr, i)
  gep(p, li)
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}) where {T, I,C,B,R,X}
  p, li = tdot(ptr, i, strides(ptr))
  gep(p, li)
end

struct StridedBitPointer{N,C,B,R,X} <: AbstractStridedPointer{Bit,N,C,B,R,X,NTuple{N,Int}}
  p::Ptr{Bit}
  strd::X
  offsets::NTuple{N,Int}
end
function StridedBitPointer{N,C,B,R}(p::Ptr{Bit}, strd::X, offsets) where {N,C,B,R,X}
  StridedBitPointer{N,C,B,R,X}(p, strd, offsets)
end
@inline Base.pointer(p::StridedBitPointer) = p.p
# @inline stridedpointer(A::BitVector) = StridedBitPointer{1,1,0,(1,)}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), (StaticInt{1}(),), (StaticInt{1}(),))
@inline bytestrides(A::StridedBitPointer) = getfield(A, :strd)
@inline Base.strides(ptr::StridedBitPointer) = getfield(ptr, :strd)
@inline ArrayInterface.strides(ptr::StridedBitPointer) = getfield(ptr, :strd)
@inline ArrayInterface.offsets(ptr::StridedBitPointer) = getfield(ptr, :offsets)

@inline function stridedpointer(
  ptr::Ptr{Bit}, ::StaticInt{C}, ::StaticInt{B}, ::Val{R}, strd::X, offsets::O
) where {C,B,R,N,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}}
  StridedBitPointer{N,C,B,R,X}(ptr, strd, offsets)
end

# @inline stridedpointer(A::BitVector) = StridedBitPointer{1,1,0,(1,)}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), (StaticInt{1}(),), (1,))
# @generated function stridedpointer(A::BitArray{N}) where {N}
#     q = quote;
#         s = size(A)
#         @assert iszero(s[1] & 7) "For performance reasons, `BitArray{N}` where `N > 1` are required to have a multiple of 8 rows.";
#     end
#     sone = :(StaticInt{1}());
#     strd = Expr(:tuple, sone, :s_2); offsets = Expr(:tuple, sone, sone);
#     last_stride = next_stride = :s_2
#     push!(q.args, :(s_2 = size(A,1)));
#     R = Expr(:tuple, 1, 2)
#     for n ∈ 3:N
#         next_stride = Symbol(:s_, n)
#         push!(q.args, Expr(:(=), next_stride, Expr(:call, :(*), Expr(:ref, :s, n-1), last_stride)))
#         push!(strd.args, next_stride)
#         # push!(offsets.args, :(StaticInt{1}()))
#         push!(offsets.args, 1)
#         last_stride = next_stride
#         push!(R.args, n)
#     end
#     push!(q.args, :(StridedBitPointer{$N,1,0,$R}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), $strd, $offsets)))
#     q
# end
@inline function similar_with_offset(sptr::StridedBitPointer{N,C,B,R}, ptr::Ptr{Bit}, offset::Tuple{Vararg{Integer,N}}) where {N,C,B,R}
    StridedBitPointer{N,C,B,R}(ptr, getfield(sptr, :strd), offset)
end
@inline function similar_no_offset(sptr::StridedBitPointer{N,C,B,R}, ptr::Ptr{Bit}) where {N,C,B,R}
  StridedBitPointer{N,C,B,R}(ptr, getfield(sptr, :strd), ntuple(zero, Val{N}()))
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


function double_index_quote(C,B,R::NTuple{N,Int},I1,I2,typ) where {N}
    # place into position of second arg
    J1 = I1 + 1; J2 = I2 + 1;
    @assert (J1 != B) & (J2 != B)
    Cnew = ((C == J1) | (C == J2)) ? -1 : (C - (J1 < C))
    push!(typ.args, Cnew); push!(typ.args, B);
    strd = Expr(:tuple); offs = Expr(:tuple);
    inds = Expr(:tuple); Rtup = Expr(:tuple)
    for n in 1:N
        if n == J1
            push!(inds.args, :(Zero()))
        elseif n == J2
            push!(strd.args, Expr(:call, :vadd_nw, Expr(:ref, :strd, J1), Expr(:ref, :strd, J2)))
            push!(offs.args, :(Zero()))
            push!(inds.args, :(Zero()))
            push!(Rtup.args, max(R[J1], R[J2]))
        else
            push!(strd.args, Expr(:ref, :strd, n))
            push!(offs.args, Expr(:ref, :offs, n))
            push!(inds.args, Expr(:ref, :offs, n))
            push!(Rtup.args, R[n])
        end
    end
    push!(typ.args, Rtup)
    gepedptr = Expr(:call, :gep, :ptr, inds)
    newptr = Expr(:call, typ, gepedptr, strd, offs)
    Expr(:block, Expr(:meta,:inline), :(strd = getfield(ptr, :strd)), :(offs = getfield(ptr, :offsets)), newptr)
end
@generated function double_index(ptr::StridedPointer{T,N,C,B,R}, ::Val{I1}, ::Val{I2}) where {T,N,C,B,R,I1,I2}
    double_index_quote(C,B,R,I1,I2, Expr(:curly, :StridedPointer, :T, N - 1))
end
@generated function double_index(ptr::StridedBitPointer{N,C,B,R}, ::Val{I1}, ::Val{I2}) where {N,C,B,R,I1,I2}
    double_index_quote(C,B,R,I1,I2, Expr(:curly, :StridedBitPointer, N - 1))
end

@inline stridedpointer(ptr::AbstractStridedPointer) = ptr
@inline stridedpointer_preserve(ptr::AbstractStridedPointer) = (ptr,nothing)

struct FastRange{T,F,S,O}# <: AbstractStridedPointer{T,1,1,0,(1,),Tuple{S},Tuple{O}}# <: AbstractRange{T}
  f::F
  s::S
  o::O
end
FastRange{T}(f::F,s::S) where {T<:Integer,F,S} = FastRange{T,Zero,S,F}(Zero(),s,f)
FastRange{T}(f::F,s::S,o::O) where {T,F,S,O} = FastRange{T,F,S,O}(f,s,o)

FastRange{T}(f,s) where {T<:FloatingTypes} = FastRange{T}(f,s,fast_int64_to_double())
FastRange{T}(f::F,s::S,::True) where {T<:FloatingTypes,F,S} = FastRange{T,F,S,Int}(f,s,0)
FastRange{T}(f::F,s::S,::False) where {T<:FloatingTypes,F,S} = FastRange{T,F,S,Int32}(f,s,zero(Int32))

@inline function memory_reference(r::AbstractRange{T}) where {T}
  s = ArrayInterface.static_step(r)
  FastRange{T}(ArrayInterface.static_first(r) - s, s), nothing
end
@inline memory_reference(r::FastRange) = (r,nothing)
@inline bytestrides(::FastRange{T}) where {T} = (static_sizeof(T),)
@inline ArrayInterface.offsets(::FastRange) = (One(),)
@inline val_stride_rank(::FastRange) = Val{(1,)}()
@inline val_dense_dims(::FastRange) = Val{(true,)}()
@inline ArrayInterface.contiguous_axis(::FastRange) = One()
@inline ArrayInterface.contiguous_batch_size(::FastRange) = Zero()

@inline stridedpointer(fr::FastRange, ::StaticInt{1}, ::StaticInt{0}, ::Val{(1,)}, ::Tuple{X}, ::Tuple{One}) where {X<:Integer} = fr

# `FastRange{<:Integer}` can ignore the offset
@inline vload(r::FastRange{T,Zero}, i::Tuple{I}) where {T<:Integer,I} = convert(T, getfield(r, :o)) + convert(T, getfield(r, :s)) * first(i)

@inline function vload(r::FastRange{T}, i::Tuple{I}) where {T<:FloatingTypes,I}
  convert(T, getfield(r, :f)) + convert(T, getfield(r, :s)) * (only(i) + convert(T, getfield(r, :o)))
end
@inline function gesp(r::FastRange{T,Zero}, i::Tuple{I}) where {I,T<:Integer}
  s = getfield(r, :s)
  FastRange{T}(Zero(), s, only(i)*s + getfield(r, :o))
end
@inline function gesp(r::FastRange{T}, i::Tuple{I}) where {I,T<:FloatingTypes}
    FastRange{T}(getfield(r, :f), getfield(r, :s), only(i) + getfield(r, :o))
end
@inline gesp(r::FastRange{T,Zero}, i::Tuple{NullStep}) where {T<:Integer} = r
@inline gesp(r::FastRange{T}, i::Tuple{NullStep}) where {T<:FloatingTypes} = r
@inline increment_ptr(r::FastRange{T,Zero}, i::Tuple{I}) where {I,T<:Integer} = only(i)*s + getfield(r, :o)
@inline increment_ptr(r::FastRange{T}, i::Tuple{I}) where {I,T<:Integer} = only(i) + getfield(r, :o)
@inline increment_ptr(r::FastRange) = getfield(r,:o)
@inline increment_ptr(r::FastRange{T}, o, i::Tuple{I}) where {I,T} = vadd_nsw(only(i), o)
@inline increment_ptr(r::FastRange{T,Zero}, o, i::Tuple{I}) where {I,T} = vadd_nsw(vmul_nsw(only(i), getfield(r, :s)), o)

@inline reconstruct_ptr(r::FastRange{T}, o) where {T} = FastRange{T}(getfield(r,:f), getfield(r, :s), o)


@inline function zero_offsets(fr::FastRange{T,Zero}) where {T<:Integer}
  s = getfield(fr,:s)
  FastRange{T}(Zero(), s, getfield(fr,:o) + s)
end
@inline function zero_offsets(fr::FastRange{T}) where {T<:FloatingTypes}
  FastRange{T}(getfield(fr,:f), getfield(fr,:s), getfield(fr,:o) + 1)
end

# `FastRange{<:FloatingTypes}` must use an integer offset because `ptrforcomparison` needs to be exact/integral.

# @inline pointerforcomparison(r::FastRange) = getfield(r, :o)
# @inline pointerforcomparison(r::FastRange, i::Tuple{I}) where {I} = getfield(r, :o) + first(i)


@inline vload(r::FastRange, i, m::AbstractMask) = (v = vload(r, i); ifelse(m, v, zero(v)))
@inline vload(r::FastRange, i, m::Bool) = (v = vload(r, i); ifelse(m, v, zero(v)))
@inline _vload(r::FastRange, i, _, __) = vload(r, i)
@inline _vload(r::FastRange, i, m::AbstractMask, __, ___) = vload(r, i, m)
@inline _vload(r::FastRange, i, m::VecUnroll{<:Any,<:Any,<:Union{Bool,Bit}}, __, ___) = vload(r, i, m)
function _vload_fastrange_unroll(AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, X::Int, mask::Bool, vecunrollmask::Bool)
  t = Expr(:tuple)
  inds = unrolled_indicies(1, AU, F, N, AV, W, X)
  q = quote
    $(Expr(:meta, :inline))
    gptr = gesp(r, data(u))
  end
  vecunrollmask && push!(q.args, :(masktup = data(vm)))
  gf = GlobalRef(Core, :getfield)
  for n in 1:N
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
# discard unnueeded align/reg size info
# @inline vload(r::FastRange, i, ::A) where {A<:StaticBool} = vload(r,i)
# @inline vload(r::FastRange, i, m, ::A) where {A<:StaticBool} = vload(r,i,m)
# @inline Base.getindex(r::FastRange, i::Integer) = vload(r, (i,))
@inline Base.eltype(::FastRange{T}) where {T} = T

# @generated function vload(r::FastRange{T}, i::Unroll{AU,F,N,AV,W,M,X,I}) where {T,I}

# end

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
@inline preserve_buffer(A::Union{LinearAlgebra.Transpose,LinearAlgebra.Adjoint,Base.ReinterpretArray,Base.ReshapedArray,PermutedDimsArray,SubArray}) = preserve_buffer(parent(A))
@inline preserve_buffer(x) = x

function llvmptr_comp_quote(cmp, Tsym)
    pt = Expr(:curly, GlobalRef(Core, :LLVMPtr), Tsym, 0)
    instrs = "%cmpi1 = icmp $cmp i8* %0, %1\n%cmpi8 = zext i1 %cmpi1 to i8\nret i8 %cmpi8"
    Expr(:block, Expr(:meta,:inline), :(llvmcall($instrs, Bool, Tuple{$pt,$pt}, p1, p2)))
end
# @inline llvmptr(p::Ptr{T}) where {T} = reinterpret(Core.LLVMPtr{T,0}, p)
# @inline llvmptr(p::AbstractStridedPointer) = llvmptr(pointer(p))
@inline llvmptrd(p::Ptr) = reinterpret(Core.LLVMPtr{Float64,0}, p)
@inline llvmptrd(p::AbstractStridedPointer) = llvmptrd(pointer(p))
for (op,f,cmp) ∈ [(:(<),:vlt,"ult"), (:(>),:vgt,"ugt"), (:(≤),:vle,"ule"), (:(≥),:vge,"uge"), (:(==),:veq,"eq"), (:(≠),:vne,"ne")]
  @eval begin
    @generated function $f(p1::Core.LLVMPtr{T,0}, p2::Core.LLVMPtr{T,0}) where {T}
      llvmptr_comp_quote($cmp, JULIA_TYPES[T])
    end
    @inline Base.$op(p1::P, p2::P) where {P <: AbstractStridedPointer} = $f(llvmptrd(p1), llvmptrd(p2))
    # @inline Base.$op(p1::P, p2::P) where {P <: StridedBitPointer} = $f(llvmptr(center(p1)), llvmptr(center(p2)))
    @inline Base.$op(p1::P, p2::P) where {P <: StridedBitPointer} = $op(linearize(p1), linearize(p2))
    @inline Base.$op(p1::P, p2::P) where {P <: FastRange} = $op(getfield(p1, :o), getfield(p2, :o))
    @inline $f(p1::Ptr, p2::Ptr, sp::AbstractStridedPointer) = $f(llvmptrd(p1), llvmptrd(p2))
    @inline $f(p1::NTuple{N,Int}, p2::NTuple{N,Int}, sp) where {N} = $op(reconstruct_ptr(sp, p1), reconstruct_ptr(sp, p2))
    @inline $f(a,b,c) = $f(a,b)
  end
end
@inline linearize(p::StridedBitPointer) = -sum(map(*, getfield(p, :strd), getfield(p, :offsets)))



# for (op) ∈ [(:(<)), (:(>)), (:(≤)), (:(≥)), (:(==)), (:(≠))]
#     @eval begin
#         @inline Base.$op(p1::P, p2::P) where {P <: AbstractStridedPointer} = $op(pointer(p1), pointer(p2))
#         @inline Base.$op(p1::P, p2::P) where {P <: StridedBitPointer} = $op(linearize(p1), linearize(p2))
#         @inline Base.$op(p1::P, p2::P) where {P <: FastRange} = $op(getfield(p1, :o), getfield(p2, :o))
#     end
# end


