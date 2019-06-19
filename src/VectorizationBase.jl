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
@inline SVec(v::SVec) = v
@inline Base.length(::AbstractStructVec{N}) where N = N
@inline Base.size(::AbstractStructVec{N}) where N = (N,)
@inline Base.eltype(::AbstractStructVec{N,T}) where {N,T} = T
@inline conj(v::AbstractStructVec) = v # so that things like dot products work.
@inline adjoint(v::AbstractStructVec) = v # so that things like dot products work.
@inline Base.getindex(v::SVec, i::Integer) = v.data[i].value

@inline function SVec{N,T}(v::SVec{N,T2}) where {N,T,T2}
    @inbounds SVec(ntuple(n -> Core.VecElement{T}(T(v[n])), Val(N)))
end

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




include("vectorizable.jl")
include("cpu_info.jl")
include("vector_width.jl")
include("number_vectors.jl")
include("masks.jl")

end # module
