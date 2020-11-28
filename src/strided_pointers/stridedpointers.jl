
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

@inline mulsizeof(::Type{T}, x::Number) where {T} = vmul(sizeof(T), x)
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
@inline memory_reference(::ArrayInterface.CPUIndex, A) = throw("Not implemented yet.")


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
@inline Base.strides(ptr::StridedPointer) = ptr.strd
@inline ArrayInterface.offsets(ptr::StridedPointer) = ptr.offsets
@inline ArrayInterface.contiguous_axis_indicator(ptr::AbstractStridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(Contiguous{C}(), Val{N}())


@generated function zerotuple(::Val{N}) where {N}
    t = Expr(:tuple); foreach(n -> push!(t.args, Expr(:call, :Zero)), 1:N)
    Expr(:block, Expr(:meta,:inline), t)
end

@generated function zero_offsets(sptr::StridedPointer{T,N,C,B,R,X}) where {T,N,C,B,R,X}
    o = Expr(:tuple); foreach(n -> push!(o.args, :(StaticInt{0}())), 1:N)
    O = Expr(:curly, :Tuple); foreach(n -> push!(O.args, :(StaticInt{0})), 1:N)
    Expr(:block, Expr(:meta, :inline), :(StridedPointer{$T,$N,$C,$B,$R,$X,$O}(sptr.p, sptr.strd, $o)))
end

# @inline function Base.similar(sptr::StridedPointer{T,N,C,B,R,X,O}, ptr::Ptr{T}) where {T,N,C,B,R,X,O}
#     StridedPointer{T,N,C,B,R,X,O}(ptr, sptr.strd, sptr.offsets)
# end
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

@inline vload(ptr::AbstractStridedPointer{T,0}, i::Tuple{}, ::Val{A}) where {A,T} = vload(pointer(ptr))

# Fast compile path?
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}, ::Val{A}) where {A,T,N,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}, m, ::Val{A}) where {A,T,N,C,B,R,X}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}, ::Val{A}) where {A,T,N,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}, m, ::Val{A}) where {A,T,N,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}, ::Val{A}) where {A,T,I}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}, m, ::Val{A}) where {A,T,I}
    vload(pointer(ptr), tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}, ::Val{A}) where {A,T,I,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), Val{A}())
end
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}, m, ::Val{A}) where {A,T,I,C,B,R,X,O}
    vload(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m, Val{A}())
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
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,N,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,O,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(
    ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,I,C,B,R,X,A,S,NT}
    vstore!(pointer(ptr), v, tdot(ptr, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X,O}
    gep(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T, I}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}) where {T, I,C,B,R,X,O}
    gep(pointer(ptr), tdot(ptr, map(vsub, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}) where {T, I,C,B,R,X}
    gep(pointer(ptr), tdot(ptr, i, strides(ptr), nopromote_axis_indicator(ptr)))
end


struct StridedBitPointer{N,C,B,R,X,O} <: AbstractStridedPointer{Bool,N,C,B,R,X,O}
    p::Ptr{Bool}
    strd::X
    offsets::O
end
function StridedBitPointer{N,C,B,R}(p::Ptr{Bool}, strd::X, offsets::O) where {N,C,B,R,X,O}
    StridedBitPointer{N,C,B,R,X,O}(p, strd, offsets)
end
@inline stridedpointer(A::BitVector) = StridedBitPointer(Base.unsafe_convert(Ptr{Bool}, pointer(A.chunks)), (Static{1}(),), (Static{1}(),))
@generated function stridedpointer(A::BitArray{N}) where {N}
    q = quote;
        s = size(A)
        @assert iszero(s[1] & 7) "For performance reasons, `BitArray{N}` where `N > 1` are required to have a multiple of 8 rows.";
    end
    sone = :(StaticInt{1}());
    strd = Expr(:tuple, sone, :s_2); offsets = Expr(:tuple, sone, sone);
    last_stride = next_stride = :s_2
    push!(q.args, :(s_2 = size(A,1))); # >>> 3
    for n ∈ 3:N
        next_stride = Symbol(:s_, n)
        push!(q.args, Expr(:(=), next_stride, Expr(:call, :(*), Expr(:ref, :s, n-1), last_stride)))
        push!(strd.args, next_stride)
        push!(offsets.args, :(StaticInt{1}()))
        last_stride = next_stride
    end
    push!(q.args, :(StridedBitPointer{$N,1,0,$R}(Base.unsafe_convert(Ptr{UInt8}, pointer(A.chunks)), $strd, $offsets)))
    q
end

@inline tdot(ptr::StridedBitPointer, a, b, c) = tdot(Bool, a, b, c) >>> StaticInt(3)

