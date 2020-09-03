
# @generated function static_tuple(t::SDTuple{N,X,P}) where {N,X,P}
#     i = 0
#     sdt = Expr(:tuple)
#     Xv = tuple_type_to_value_tuple(X)
#     for n ∈ 1:N
#         push!(sdt.args, Xv[n] == -1 ? Expr(:ref, :x, (i += 1)) : Expr(:call, Expr(:curly, :Static, Xv[n])))
#     end
#     q = Expr(:block, Expr(:meta, :inline))
#     i > 0 && push!(q.args, :(x = t.x))
#     push!(q.args, sdt)
#     q
# end

@inline mulsizeof(::Type{T}, x::Number) where {T} = vmul(sizeof(T), x)
@generated mulsizeof(::Type{T}, ::Static{N}) where {T,N} = Expr(:call, Expr(:curly, :Static, N*sizeof(T)))
@inline mulsizeof(::Type{T}, ::Tuple{}) where {T} = ()
@inline mulsizeof(::Type{T}, x::Tuple{X}) where {T,X} = (mulsizeof(T, first(x)), )
@inline mulsizeof(::Type{T}, x::Tuple) where {T} = (mulsizeof(T, first(x)), mulsizeof(T, Base.tail(x))...)
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
@inline stridedpointer(A::AbstractArray) = stridedpointer(device(A), A)
@inline function stridedpointer(::CPUPointer, A::AbstractArray{T}) where {T <: NativeTypes}
    stridedpointer(pointer(A), contiguous_axis(A), contiguous_batch_size(A), stride_rank(A), mulsizeof(T, sdstrides(A)), sdoffsets(A))
end
@inline function stridedpointer(
    ::ArrayInterface.CheckParent, A::AbstractArray{T},
    C = contiguous_axis(A), B = contiguous_batch_size(A), R = stride_rank(A), X = mulsizeof(T, sdstrides(A)), O = sdoffsets(A)
) where {T <: NativeTypes}
    P = parent(A)
    if P === A
        stridedpointer(ArrayInterface.CPUIndex(), P, C, B, R, X, O)
    else
        stridedpointer(device(A), P, C, B, R, X, O)
    end
end
@inline function stridedpointer(::ArrayInterface.CPUPointer, A::AbstractArray{T}, C, B, R, X, O) where {T<:NativeTypes}
    stridedpointer(pointer(A), C, B, R, X, O)
end
@inline function stridedpointer(
    ptr::Ptr{T}, ::Contiguous{C}, ::ContiguousBatch{B}, ::StrideRank{R}, strd::X, offsets::O
) where {T<:NativeTypes,C,B,R,N,X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}}}
    StridedPointer{T,N,C,B,R,X,O}(ptr, strd, offsets)
end
@inline Base.strides(ptr::StridedPointer) = ptr.strd
@inline ArrayInterface.sdoffsets(ptr::StridedPointer) = ptr.offsets
@inline ArrayInterface.contiguous_axis_indicator(ptr::StridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(Contiguous{C}(), Val{N}())

@inline Base.pointer(ptr::StridedPointer) = ptr.p
Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractStridedPointer{T}) where {T} = pointer(ptr)
# Shouldn't need to special case Array
# function stridedpointer(A::Array{T,N}) where {T,N}
#     StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
# end

@inline vload(ptr::AbstractStridedPointer) = vload(pointer(ptr))
@inline vstore!(ptr::AbstractStridedPointer{T}, v::T) where {T} = vstore!(pointer(ptr), v)

# Fast compile path?
@inline function vload(ptr::AbstractStridedPointer{<:Any,N,<:Any,<:Any,<:Any,NTuple{N,Static{0}}}, i::Tuple{Vararg{Any,N}}) where {N}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{<:Any,N,<:Any,<:Any,<:Any,NTuple{N,Static{0}}}, i::Tuple{Vararg{Any,N}}, m) where {N}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,N,<:Any,<:Any,<:Any,NTuple{N,Static{0}}}, v, i::Tuple{Vararg{Any,N}}) where {N}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,N,<:Any,<:Any,<:Any,NTuple{N,Static{0}}}, v, i::Tuple{Vararg{Any,N}}, m) where {N}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end

@inline function vload(ptr::AbstractStridedPointer{<:Any,N}, i::Tuple{Vararg{Any,N}}) where {N}
    vload(pointer(ptr), tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{<:Any,N}, i::Tuple{Vararg{Any,N}}, m) where {N}
    vload(pointer(ptr), tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,N}, v, i::Tuple{Vararg{Any,N}}) where {N}
    vstore!(pointer(ptr), v, tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,N}, v, i::Tuple{Vararg{Any,N}}, m) where {N}
    vstore!(pointer(ptr), v, tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end

# LinearIndexing -- no offset
# Danger, this means 1-dim indexing is 0-based?
@inline function vload(ptr::AbstractStridedPointer, i::Tuple{I}) where {I}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer, i::Tuple{I}, m) where {I}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer, v, i::Tuple{I}) where {I}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer, v, i::Tuple{I}, m) where {I}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end

# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{<:Any,1}, i::Tuple{I}) where {I}
    vload(pointer(ptr), tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{<:Any,1}, i::Tuple{I}, m) where {I}
    vload(pointer(ptr), tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,1}, v, i::Tuple{I}) where {I}
    vstore!(pointer(ptr), v, tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,1}, v, i::Tuple{I}, m) where {I}
    vstore!(pointer(ptr), v, tdot(map(-, i, sdoffsets(ptr)), strides(ptr), contiguous_axis_indicator(ptr)), m)
end
# Ambiguity: 1-dimensional + 1-dim index -> Cartesian (offset) indexing
@inline function vload(ptr::AbstractStridedPointer{<:Any,1,<:Any,<:Any,<:Any,Tuple{Static{0}}}, i::Tuple{I}) where {I}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vload(ptr::AbstractStridedPointer{<:Any,1,<:Any,<:Any,<:Any,Tuple{Static{0}}}, i::Tuple{I}, m) where {I}
    vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,1,<:Any,<:Any,<:Any,Tuple{Static{0}}}, v, i::Tuple{I}) where {I}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
end
@inline function vstore!(ptr::AbstractStridedPointer{<:Any,1,<:Any,<:Any,<:Any,Tuple{Static{0}}}, v, i::Tuple{I}, m) where {I}
    vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
end



