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
end
SVec{N,T}(x::T) where {N,T} = SVec(ntuple(i -> VE(x), Val(N)))
@inline Base.length(::AbstractStructVec{N}) where N = N
@inline Base.size(::AbstractStructVec{N}) where N = (N,)
@inline Base.eltype(::AbstractStructVec{N,T}) where {N,T} = T
@inline conj(v::AbstractStructVec) = v # so that things like dot products work.
@inline adjoint(v::AbstractStructVec) = v # so that things like dot products work.

const AbstractSIMDVector{N,T} = Union{Vec{N,T},AbstractStructVec{N,T}}

@inline extract_data(v::Vec) = v
@inline extract_data(v::SVec) = v.data

"""
vectorizable(x) returns a representation of x convenient for vectorization.
The generic fallback simply returns pointer(x):

@inline vectorizable(x) = pointer(x)

however pointers are sometimes not the ideal representation, and othertimes
they are not possible in Julia (eg for stack-allocated objects). This interface
allows one to customize behavior via making use of the type system.
"""
@inline vectorizable(x) = pointer(x)




include("cpu_info.jl")
include("vector_width.jl")
include("number_vectors.jl")

end # module
