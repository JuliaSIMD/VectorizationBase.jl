
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

@inline mulsizeof(::Type{T}, x::Number) where {T} = vmul_fast(sizeof(T), x)
@generated mulsizeof(::Type{T}, ::StaticInt{N}) where {T,N} = Expr(:call, Expr(:curly, :StaticInt, N*sizeof(T)))
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
@inline memory_reference(::CPUPointer, A) = pointer(A)
@inline function memory_reference(::ArrayInterface.CheckParent, A)
    P = parent(A)
    if P === A
        memory_reference(ArrayInterface.CPUIndex(), A)
    else
        memory_reference(device(P), P)
    end
end
@inline memory_reference(::ArrayInterface.CPUIndex, A) = throw("Memory access for $(typeof(A)) not implemented yet.")


"""
  abstract type AbstractStridedPointer{T,N,C,B,R,X,O} end

T: element type
N: dimensionality
C: contiguous dim
B: batch size
R: rank of strides
X: strides
O: offsets
"""
abstract type AbstractStridedPointer{T<:NativeTypes,N,C,B,R,X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}}} end

@inline Base.eltype(::AbstractStridedPointer{T}) where {T} = T

struct StridedPointer{T,N,C,B,R,X,O} <: AbstractStridedPointer{T,N,C,B,R,X,O}
    p::Ptr{T}
    strd::X
    offsets::O
end
@inline StridedPointer{T,N,C,B,R}(ptr::Ptr{T}, strd::X, o::O) where {T,N,C,B,R,X,O} = StridedPointer{T,N,C,B,R,X,O}(ptr, strd, o)
@inline StridedPointer{T,N,C,B,R,X}(ptr::Ptr{T}, strd::X, o::O) where {T,N,C,B,R,X,O} = StridedPointer{T,N,C,B,R,X,O}(ptr, strd, o)

@inline function stridedpointer(A::AbstractArray{T}) where {T <: NativeTypes}
    stridedpointer(memory_reference(A), contiguous_axis(A), contiguous_batch_size(A), stride_rank(A), bytestrides(A), offsets(A))
end
@inline function stridedpointer(
    ptr::Ptr{T}, ::Contiguous{C}, ::ContiguousBatch{B}, ::StrideRank{R}, strd::X, offsets::O
) where {T<:NativeTypes,C,B,R,N,X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}}}
    StridedPointer{T,N,C,B,R,X,O}(ptr, strd, offsets)
end
@inline Base.strides(ptr::AbstractStridedPointer) = ptr.strd
@inline ArrayInterface.offsets(ptr::AbstractStridedPointer) = ptr.offsets
@inline ArrayInterface.contiguous_axis_indicator(ptr::AbstractStridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(Contiguous{C}(), Val{N}())


@generated function zerotuple(::Val{N}) where {N}
    t = Expr(:tuple);
    for n in 1:N
        push!(t.args, :(Zero()))
    end
    Expr(:block, Expr(:meta,:inline), t)
end

@generated function zero_offsets(sptr::StridedPointer{T,N,C,B,R,X}) where {T,N,C,B,R,X}
    o = Expr(:tuple); foreach(n -> push!(o.args, :(StaticInt{0}())), 1:N)
    O = Expr(:curly, :Tuple); foreach(n -> push!(O.args, :(StaticInt{0})), 1:N)
    Expr(:block, Expr(:meta, :inline), :(StridedPointer{$T,$N,$C,$B,$R,$X,$O}(sptr.p, sptr.strd, $o)))
end

@inline function Base.similar(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
    StridedPointer{T,N,C,B,R,X,O}(ptr, sptr.strd, sptr.offsets)
end
# @inline noalias!(p::StridedPointer) = similar(p, noalias!(pointer(p)))
@inline function similar_no_offset(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
    StridedPointer{T,N,C,B,R,X}(ptr, sptr.strd, zerotuple(Val{N}()))
end

@inline Base.pointer(ptr::StridedPointer) = ptr.p
Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractStridedPointer{T}) where {T} = pointer(ptr)
# Shouldn't need to special case Array
# function stridedpointer(A::Array{T,N}) where {T,N}
#     StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
# end

@inline vload(ptr::AbstractStridedPointer) = vload(pointer(ptr))
@inline vstore!(ptr::AbstractStridedPointer{T}, v::T) where {T} = vstore!(pointer(ptr), v)

@generated function nopromote_axis_indicator(::AbstractStridedPointer{<:Any,N}) where {N}
    t = Expr(:tuple); foreach(n -> push!(t.args, Expr(:call, Expr(:curly, :Val, true))), 1:N)
    Expr(:block, Expr(:meta, :inline), t)
end

@inline vload(ptr::AbstractStridedPointer{T,0}, i::Tuple{}, ::Val{A}) where {A,T} = vload(pointer(ptr), Val{A}())
@inline gep(ptr::AbstractStridedPointer{T,0}, i::Tuple{}) where {T} = pointer(ptr)

# Fast compile path?
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}, ::Val{A}) where {A,T,N,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}, m, ::Val{A}) where {A,T,N,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}, ::Val{A}) where {A,T,N,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}, m, ::Val{A}) where {A,T,N,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}, ::Val{A}) where {A,T,I}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}, m, ::Val{A}) where {A,T,I}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}, ::Val{A}) where {A,T,I,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}, m, ::Val{A}) where {A,T,I,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}, ::Val{A}) where {A,T,I,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}, m, ::Val{A}) where {A,T,I,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end

# align, noalias, nontemporal
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X,O}
    gep(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T, I}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}) where {T, I,C,B,R,X,O}
    gep(pointer(ptr), tdot(ptr, map(vsub_fast, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}) where {T, I,C,B,R,X}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end

struct StridedBitPointer{N,C,B,R,X,O} <: AbstractStridedPointer{Bit,N,C,B,R,X,O}
    p::Ptr{Bit}
    strd::X
    offsets::NTuple{N,Int}
end
function StridedBitPointer{N,C,B,R}(p::Ptr{Bit}, strd::X, offsets::O) where {N,C,B,R,X,O}
    StridedBitPointer{N,C,B,R,X,O}(p, strd, offsets)
end
@inline Base.pointer(p::StridedBitPointer) = p.p
# @inline stridedpointer(A::BitVector) = StridedBitPointer{1,1,0,(1,)}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), (StaticInt{1}(),), (StaticInt{1}(),))
@inline stridedpointer(A::BitVector) = StridedBitPointer{1,1,0,(1,)}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), (StaticInt{1}(),), (1,))
@generated function stridedpointer(A::BitArray{N}) where {N}
    q = quote;
        s = size(A)
        @assert iszero(s[1] & 7) "For performance reasons, `BitArray{N}` where `N > 1` are required to have a multiple of 8 rows.";
    end
    sone = :(StaticInt{1}());
    strd = Expr(:tuple, sone, :s_2); offsets = Expr(:tuple, sone, sone);
    last_stride = next_stride = :s_2
    push!(q.args, :(s_2 = size(A,1))); # >>> 3
    R = Expr(:tuple, 1, 2)
    for n ∈ 3:N
        next_stride = Symbol(:s_, n)
        push!(q.args, Expr(:(=), next_stride, Expr(:call, :(*), Expr(:ref, :s, n-1), last_stride)))
        push!(strd.args, next_stride)
        # push!(offsets.args, :(StaticInt{1}()))
        push!(offsets.args, 1)
        last_stride = next_stride
        push!(R.args, n)
    end
    push!(q.args, :(StridedBitPointer{$N,1,0,$R}(Base.unsafe_convert(Ptr{Bit}, pointer(A.chunks)), $strd, $offsets)))
    q
end
@inline function similar_no_offset(sptr::StridedBitPointer{N,C,B,R,X,O}, ptr::Ptr{Bit}) where {N,C,B,R,X,O}
    StridedBitPointer{N,C,B,R}(ptr, sptr.strd, zerotuple(Val{N}()))
end

@generated function gesp(ptr::StridedBitPointer{N,C,B,R}, i::Tuple{Vararg{Any,N}}) where {N,C,B,R}
    quote
        $(Expr(:meta,:inline))
        offs = ptr.offsets
        StridedBitPointer{$N,$C,$B,$R}(ptr.p, ptr.strd, Base.Cartesian.@ntuple $N n -> vsub_fast(offs[n], i[n]))
    end
end
@generated function pointerforcomparison(p::StridedBitPointer{N}) where {N}
    inds = Expr(:tuple); foreach(_ -> push!(inds.args, :(Zero())), 1:N)
    Expr(:block, Expr(:meta,:inline), Expr(:call, :gep, :p, inds))
end
# @inline tdot(ptr::StridedBitPointer, a, b, c) = tdot(Bool, a, b, c) >>> StaticInt(3)

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
            push!(strd.args, Expr(:call, :vadd_fast, Expr(:ref, :strd, J1), Expr(:ref, :strd, J2)))
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
    Expr(:block, Expr(:meta,:inline), :(strd = ptr.strd), :(offs = ptr.offsets), newptr)
end
@generated function double_index(ptr::StridedPointer{T,N,C,B,R}, ::Val{I1}, ::Val{I2}) where {T,N,C,B,R,I1,I2}
    double_index_quote(C,B,R,I1,I2, Expr(:curly, :StridedPointer, :T, N - 1))
end
@generated function double_index(ptr::StridedBitPointer{N,C,B,R}, ::Val{I1}, ::Val{I2}) where {N,C,B,R,I1,I2}
    double_index_quote(C,B,R,I1,I2, Expr(:curly, :StridedBitPointer, N - 1))
end

@inline stridedpointer(ptr::AbstractStridedPointer) = ptr

struct FastRange{T,F,S,O}# <: AbstractRange{T}
    f::F
    s::S
    offset::O
end
FastRange{T}(f::F,s::S,o::O) where {T,F,S,O} = FastRange{T,F,S,O}(f,s,o)
@inline function stridedpointer(r::AbstractRange{T}) where {T}
    FastRange{T}(ArrayInterface.static_first(r), ArrayInterface.static_step(r), One())
end
@inline function gesp(r::FastRange{T}, i::Tuple{I}) where {I,T}
    ii = first(i) - r.offset
    f = r.f
    s = r.s
    FastRange{T}(f + ii * s, r.s, Zero())
end
@inline vload(r::FastRange{T}, i::Tuple{I}) where {T,I} = convert(T, r.f) + convert(T, r.s) * (first(i) - convert(T, r.offset))
@inline vload(r::FastRange, i::Tuple, m::Mask) = vload(r, i)
@inline vload(r::FastRange, i::Tuple, m::Bool) = vload(r, i)
# @inline Base.getindex(r::FastRange, i::Integer) = vload(r, (i,))
@inline Base.eltype(::FastRange{T}) where {T} = T

