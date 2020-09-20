
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

@inline bytestrides(A::AbstractArray{T}) where {T} = mulsizeof(T, strides(A))

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

struct StridedPointer{T,N,C,B,R,X,O} <: AbstractStridedPointer{T,N,C,B,R,X,O}
    p::Ptr{T}
    strd::X
    offsets::O
end
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
@inline ArrayInterface.contiguous_axis_indicator(ptr::StridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(Contiguous{C}(), Val{N}())

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

# Fast compile path?
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}) where {T,N,C,B,R,X}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,StaticInt{0}}}, v, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X}
    gep(pointer(ptr), tdot(T, i, strides(ptr), nopromote_axis_indicator(ptr)))
end

@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X,O}
    vload(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X,O}
    vload(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X,O}
    vstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X,O}
    vstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X,O}
    vnoaliasstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, v, i::Tuple{Vararg{Any,N}}, m) where {T, N,C,B,R,X,O}
    vnoaliasstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T,N,C,B,R,X,O}, i::Tuple{Vararg{Any,N}}) where {T, N,C,B,R,X,O}
    gep(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end

# LinearIndexing -- no offset
# Danger, this means 1-dim indexing is 0-based?
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T, I}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{T}, i::Tuple{I}, m) where {T, I}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{T}, v, i::Tuple{I}) where {T, I}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m) where {T, I}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T}, v, i::Tuple{I}) where {T, I}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T}, v, i::Tuple{I}, m) where {T, I}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T}, i::Tuple{I}) where {T, I}
    gep(pointer(ptr), tdot(T, i, strides(ptr), nopromote_axis_indicator(ptr)))
end

# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}) where {T, I,C,B,R,X,O}
    vload(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}, m) where {T, I,C,B,R,X,O}
    vload(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}) where {T, I,C,B,R,X,O}
    vstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, m) where {T, I,C,B,R,X,O}
    vstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}) where {T, I,C,B,R,X,O}
    vnoaliasstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, v, i::Tuple{I}, m) where {T, I,C,B,R,X,O}
    vnoaliasstore!(pointer(ptr), v, tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,O}, i::Tuple{I}) where {T, I,C,B,R,X,O}
    gep(pointer(ptr), tdot(T, map(vsub, i, offsets(ptr)), strides(ptr), nopromote_axis_indicator(ptr)))
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}) where {T, I,C,B,R,X}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}, m) where {T, I,C,B,R,X}
    vload(pointer(ptr), tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}) where {T, I,C,B,R,X}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, m) where {T, I,C,B,R,X}
    vstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}) where {T, I,C,B,R,X}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vnoaliasstore!(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, v, i::Tuple{I}, m) where {T, I,C,B,R,X}
    vnoaliasstore!(pointer(ptr), v, tdot(T, i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function gep(ptr::AbstractStridedPointer{T,1,C,B,R,X,Tuple{StaticInt{0}}}, i::Tuple{I}) where {T, I,C,B,R,X}
    gep(pointer(ptr), tdot(T, i, strides(ptr), nopromote_axis_indicator(ptr)))
end



