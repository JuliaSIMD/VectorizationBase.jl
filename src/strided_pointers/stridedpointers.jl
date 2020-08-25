


@inline mulsizeof(::Type{T}, ::Tuple{}) where {T,X} = ()
@inline mulsizeof(::Type{T}, x::Tuple{X}) where {T,X} = (mulsizeof(T, first(x)), )
@inline mulsizeof(::Type{T}, x::Tuple) where {T} = (mulsizeof(T, first(x)), mulsizeof(T, Base.tail(x))...)
@inline function mulsizeof(::Type{T}, t::SDTuple{X,N,P}) where {X,N,P,T}
    Xnew = ntuple(n -> X[n] == -1 ? -1 : 8X[n], Val{N}())
    SDTuple{Xnew,N,P}(mulsizeof(T, t.x))
end


abstract type AbstractStridedPointer{T,C,B,R,X,N,P} end

struct StridedPointer{T,C,B,R,X,N,P} <: AbstractStridedPointer{T,C,B,R,X,N,P}
    p::Ptr{T}
    st::SDTuple{X,N,P}
end
@inline stridedpointer(A::AbstractArray) = stridedpointer(device(A), A)
@inline function stridedpointer(::CPUPointer, A::AbstractArray{T}) where {T}
    stridedpointer(pointer(A), contiguous_axis(A), contiguous_batch_size(A), striderank(A), mulsizeof(T, sdstrides(A)))
end
@inline function stridedpointer(ptr::Ptr{T}, ::Contiguous{C}, ::ContiguousBatch{B}, ::StrideRank{R}, st::SDTuple{X,N,P}) where {T,C,B,R,X,N,P}
    StridedPointer{T,C,B,R,X,N,P}(ptr, st)
end


# Shouldn't need to special case Array
# function stridedpointer(A::Array{T,N}) where {T,N}
#     StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
# end

vload(ptr::AbstractStridedPointer, i) = vload(ptr, 

