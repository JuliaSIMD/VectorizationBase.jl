module VectorizationBase

import LinearAlgebra: conj, adjoint

export  Vec,
        VE,
        SVec,
        AbstractStructVec,
        AbstractSIMDVector,
        extract_data,
        pick_vector_width,
        num_vector_load_expr,
        vectorizable

const VE{T} = Core.VecElement{T}
const Vec{N,T} = NTuple{N,VE{T}}

abstract type AbstractStructVec{N,T} end
struct SVec{N,T} <: AbstractStructVec{N,T}
    data::Vec{N,T}
    SVec{N,T}(v) where {N,T} = new(v)
end
# SVec{N,T}(x) where {N,T} = SVec(ntuple(i -> VE(T(x)), Val(N)))
@inline function SVec{N,T}(x::Number) where {N,T}
    SVec(ntuple(i -> VE(T(x)), Val(N)))
end
@inline function SVec{N,T}(x::Vararg{<:Number,N}) where {N,T}
    SVec(ntuple(i -> VE(T(x[i])), Val(N)))
end
@inline function SVec(v::Vec{N,T}) where {N,T}
    SVec{N,T}(v)
end
@inline Base.length(::AbstractStructVec{N}) where N = N
@inline Base.size(::AbstractStructVec{N}) where N = (N,)
@inline Base.eltype(::AbstractStructVec{N,T}) where {N,T} = T
@inline conj(v::AbstractStructVec) = v # so that things like dot products work.
@inline adjoint(v::AbstractStructVec) = v # so that things like dot products work.
@inline Base.getindex(v::SVec, i::Integer) = v.data[i].value

function Base.one(::Type{<:AbstractStructVec{N,T}}) where {N,T}
    SVec{N,T}(one(T))
end
function Base.one(::AbstractStructVec{N,T}) where {N,T}
    SVec{N,T}(one(T))
end
function Base.zero(::Type{<:AbstractStructVec{N,T}}) where {N,T}
    SVec{N,T}(zero(T))
end
function Base.zero(::AbstractStructVec{N,T}) where {N,T}
    SVec{N,T}(zero(T))
end

const AbstractSIMDVector{N,T} = Union{Vec{N,T},AbstractStructVec{N,T}}

@inline extract_data(v) = v
@inline extract_data(v::SVec) = v.data


"""
A wrapper to the base pointer type, that supports pointer arithmetic.
"""
struct vpointer{T}
    ptr::Ptr{T}
    @inline vpointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
@inline Base.:+(ptr::vpointer{T}, i) where {T} = vpointer(ptr.ptr + sizeof(T)*i)
@inline Base.:+(i, ptr::vpointer{T}) where {T} = vpointer(ptr.ptr + sizeof(T)*i)
@inline Base.:-(ptr::vpointer{T}, i) where {T} = vpointer(ptr.ptr - sizeof(T)*i)
@inline vpointer(A) = vpointer(pointer(A))
@inline Base.eltype(::vpointer{T}) where {T} = T
@inline Base.unsafe_load(ptr::vpointer) = unsafe_load(ptr.ptr)
@inline Base.unsafe_load(ptr::vpointer, i::Integer) = unsafe_load(ptr.ptr, i)
@inline Base.unsafe_store!(ptr::vpointer{T}, v::T) where {T} = Base.unsafe_store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::vpointer{T}, v::T, i::Integer) where {T} = Base.unsafe_store!(ptr.ptr, v, i)
@inline Base.getindex(ptr::vpointer{T}) where {T} = Base.unsafe_load(ptr.ptr)
@inline Base.getindex(ptr::vpointer{T}, i) where {T} = Base.unsafe_load(ptr.ptr, i)

"""
vectorizable(x) returns a representation of x convenient for vectorization.
The generic fallback simply returns pointer(x):

@inline vectorizable(x) = pointer(x)

however pointers are sometimes not the ideal representation, and othertimes
they are not possible in Julia (eg for stack-allocated objects). This interface
allows one to customize behavior via making use of the type system.
"""
@inline vectorizable(x) = vpointer(x)
@inline vectorizable(x::vpointer) = x


function mask_type(W)
    if W <= 8
        return UInt8
    elseif W <= 16
        return UInt16
    elseif W <= 32
        return UInt32
    elseif W <= 64
        return UInt64
    else#if W <= 128
        return UInt128
    end
end

@generated function mask_type(::Type{T}, ::Val{P}) where {T,P}
    W = pick_vector_width(P, T)
    if W <= 8
        return UInt8
    elseif W <= 16
        return UInt16
    elseif W <= 32
        return UInt32
    elseif W <= 64
        return UInt64
    else#if W <= 128
        return UInt128
    end
end

include("cpu_info.jl")
include("vector_width.jl")
include("number_vectors.jl")

end # module
