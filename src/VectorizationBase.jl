module VectorizationBase

import ArrayInterface, LinearAlgebra, Libdl, Hwloc
using ArrayInterface: contiguous_axis, contiguous_axis_indicator, Contiguous, SDTuple, CPUPointer, ContiguousBatch, StrideRank, known_length, known_first, known_last
# using LinearAlgebra: Adjoint, 

# const LLVM_SHOULD_WORK = Sys.ARCH !== :i686 && isone(length(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist())))

## Until SIMDPirates stops importing it
# isfile(joinpath(@__DIR__, "cpu_info.jl")) || throw("File $(joinpath(@__DIR__, "cpu_info.jl")) does not exist. Please run `using Pkg; Pkg.build()`.")

export Vec

# using Base: llvmcall
using Base: llvmcall, VecElement
# @inline llvmcall(s::String, args...) = Base.llvmcall(s, args...)
# @inline llvmcall(s::Tuple{String,String}, args...) = Base.llvmcall(s, args...)

# export Vec, VE, Vec, Mask, MM,
#     gep, gesp,
#     data,
#     pick_vector_width,
#     pick_vector_width_shift,
#     stridedpointer,
#     PackedStridedPointer, RowMajorStridedPointer,
#     StaticStridedPointer, StaticStridedStruct,
#     vload, vstore!, vbroadcast, Static, mask, masktable

# @static if VERSION < v"1.4"
#     # I think this is worth using, and simple enough that I may as well.
#     # I'll uncomment when I find a place to use it.
#     function only(x)
#         @boundscheck length(x) == 0 && throw(ArgumentError("Collection is empty, must contain exactly 1 element"))
#         @boundscheck length(x) > 1 && throw(ArgumentError("Collection has multiple elements, must contain exactly 1 element"))
#         @inbounds x[1]
#     end
#     export only
# end

# const IntTypes = Union{Int8, Int16, Int32, Int64} # Int128
# const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64} # UInt128
# const IntegerTypes = Union{IntTypes, UIntTypes, Ptr, Bool}
const FloatingTypes = Union{Float32, Float64} # Float16
# const ScalarTypes = Union{IntegerTypes, FloatingTypes}
# const SUPPORTED_FLOATS = [Float32, Float64]
# const SUPPORTED_TYPES = [Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8]

const NativeTypes = Union{Bool,Base.HWReal}


const _Vec{W,T<:Number} = NTuple{W,Core.VecElement{T}}
# const _Vec{W,T<:Number} = Tuple{VecElement{T},Vararg{VecElement{T},W}}
@eval struct Static{N} <: Number
    (f::Type{<:Static})() = $(Expr(:new,:f))
end
Base.@pure Static(N) = Static{N}()

abstract type AbstractSIMDVector{W,T <: Union{Static,NativeTypes}} <: Real end
struct Vec{W,T} <: AbstractSIMDVector{W,T}
    data::NTuple{W,Core.VecElement{T}}
    @inline function Vec(x::NTuple{W,Core.VecElement{T}}) where {W,T}
        @assert W === pick_vector_width(W, T)# || W === 8
        # @assert ispow2(W) && (W ≤ max(pick_vector_width(W, T), 8))
        new{W,T}(x)
    end
end
struct VecUnroll{N,W,T,V<:AbstractSIMDVector{W,T}}# <: AbstractSIMDVector{W,T}
    data::NTuple{N,V}
end

@inline Base.copy(v::AbstractSIMDVector) = v
@inline asvec(x::_Vec) = Vec(x)
@inline asvec(x) = x

# struct VecUnroll{N,W,T} <: AbstractSIMDVector{W,T}
#     data::NTuple{N,Vec{W,T}}
# end

@inline unrolleddata(x) = x
@inline unrolleddata(x::VecUnroll) = x.data
# struct VecTile{M,N,W,T} <: AbstractSIMDVector{W,T}
    # data::NTuple{N,VecUnroll{M,Vec{W,T}}}
# end
# description(::Type{T}) where {T <: NativeTypes} = (-1,-1,-1,T)
# description(::Type{Vec{W,T}}) where {W, T <: NativeTypes} = (-1,-1,W,T)
# description(::Type{VecUnroll{M,W,T}}) where {M, W, T <: NativeTypes} = (M,-1,W,T)
# description(::Type{VecTile{M,N,W,T}}) where {M, W, T <: NativeTypes} = (M,N,W,T)
# function description(::Type{T1}, ::Type{T2}) where {T1, T2}
#     M1,N1,W1,T1 = description(T1)
#     M2,N2,W2,T2 = description(T2)
# end

function vec_quote(W, Wpow2, offset = 0)
    tup = Expr(:tuple); Wpow2 += offset
    foreach(w -> push!(tup.args, Expr(:call, :VecElement, Expr(:ref, :x, w+offset))), 1+offset:min(W,Wpow2))
    foreach(w -> push!(tup.args, Expr(:call, :VecElement, Expr(:call, :zero, :T))), W+1:Wpow2)
    Expr(:call, :Vec, tup)
end
@generated function Vec(x::Vararg{T,W}) where {W, T <: NativeTypes}
    Wpow2 = pick_vector_width(W, T)
    if W ≤ Wpow2
        vec_quote(W, Wpow2)
    else
        tup = Expr(:tuple)
        offset = 0
        while offset < W
            push!(tup.args, vec_quote(W, Wpow2, offset)); offset += Wpow2
        end
        Expr(:call, :VecUnroll, tup)
    end
end


struct Mask{W,U<:Unsigned} <: AbstractSIMDVector{W,Bool}
    u::U
end
const AbstractMask{W} = Union{Mask{W}, Vec{W,Bool}}
@inline Mask{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u)
# Const prop is good enough; added an @inferred test to make sure.
# Removed because confusion can cause more harm than good.
# @inline Mask(u::U) where {U<:Unsigned} = Mask{sizeof(u)<<3,U}(u)

@inline Base.broadcastable(v::AbstractSIMDVector) = Ref(v)

# Vec{N,T}(x) where {N,T} = Vec(ntuple(i -> VE(T(x)), Val(N)))
# @inline function Vec{N,T}(x::Number) where {N,T}
    # Vec(ntuple(i -> VE(T(x)), Val(N)))
# end
# @inline function Vec{N,T}(x::Vararg{<:Number,N}) where {N,T}
    # Vec(ntuple(i -> VE(T(x[i])), Val(N)))
# end
# @inline function Vec(v::Vec{N,T}) where {N,T}
    # Vec{N,T}(v)
# end
# @inline Vec(u::Unsigned) = u # Unsigned integers are treated as vectors of bools
# @inline Vec{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u) # Unsigned integers are treated as vectors of bools
# @inline Vec(v::Vec{W,T}) where {W,T} = v
# @inline Vec{W}(v::Vec{W,T}) where {W,T} = v
# @inline Vec{W,T}(v::Vec{W,T}) where {W,T} = v
# @inline Vec{W}(v::Vec{W,T}) where {W,T} = Vec{W,T}(v)
# @inline vbroadcast(::Val, b::Bool) = b


@inline Base.length(::AbstractSIMDVector{W}) where W = W
@inline Base.size(::AbstractSIMDVector{W}) where W = (W,)
@inline Base.eltype(::AbstractSIMDVector{W,T}) where {W,T} = T
@inline Base.conj(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.adjoint(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.transpose(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.getindex(v::Vec, i::Integer) = v.data[i].value

# @inline function Vec{N,T}(v::Vec{N,T2}) where {N,T,T2}
    # @inbounds Vec(ntuple(n -> Core.VecElement{T}(T(v[n])), Val(N)))
# end

# @inline Base.one(::Type{<:AbstractSIMDVector{W,T}}) where {W,T} = Vec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.one(::AbstractSIMDVector{W,T}) where {W,T} = Vec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.zero(::Type{<:AbstractSIMDVector{W,T}}) where {W,T} = Vec(vbroadcast(Vec{W,T}, zero(T)))
# @inline Base.zero(::AbstractSIMDVector{W,T}) where {W,T} = Vec(vbroadcast(Vec{W,T}, zero(T)))


# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args

@inline data(v) = v
@inline data(v::Vec) = v.data
#@inline data(v::AbstractSIMDVector) = v.data
# @inline extract_value(v::Vec, i) = v[i].value
# @inline extract_value(v::Vec, i) = v.data[i].value


function Base.show(io::IO, v::Vec{W,T}) where {W,T}
    print(io, "Vec{$W,$T}<")
    for w ∈ 1:W
        print(io, repr(v[w]))
        w < W && print(io, ", ")
    end
    print(io, ">")
end
Base.bitstring(m::Mask{W}) where {W} = bitstring(data(m))[end-W+1:end]
function Base.show(io::IO, m::Mask{W}) where {W}
    bits = m.u
    print(io, "Mask{$W,Bool}<")
    for w ∈ 0:W-1
        print(io, bits & 1)
        bits >>= 1
        w < W-1 && print(io, ", ")
    end
    print(io, ">")
end
function Base.show(io::IO, vu::VecUnroll{N,W,T}) where {N,W,T}
    println(io, "$N x Vec{$W, $T}")
    for n in 1:N
        show(io, vu.data[n]);
        n == N || println(io)
    end
end

"""
The name `MM` type refers to _MM registers such as `XMM`, `YMM`, and `ZMM`.
`MMX` from the original MMX SIMD instruction set is a [meaningless initialism](https://en.wikipedia.org/wiki/MMX_(instruction_set)#Naming).

The `MM` type is used to represent SIMD indexes. 
"""
struct MM{W,I<:Number} <: AbstractSIMDVector{W,I}
    i::I
    @inline MM{W}(i::T) where {W,T} = new{W,T}(i)
end


include("static.jl")
include("cartesianvindex.jl")
# include("vectorizable.jl")
# include("strideprodcsestridedpointers.jl")
include("topology.jl")
@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    include("cpu_info_x86_llvm.jl")
else
    include("cpu_info_generic.jl")
end
include("vector_width.jl")
include("fmap.jl")
include("llvm_types.jl")
include("lazymul.jl")
include("strided_pointers/stridedpointers.jl")
include("strided_pointers/bitpointers.jl")
include("strided_pointers/cartesian_indexing.jl")
include("strided_pointers/cse_stridemultiples.jl")
include("llvm_intrin/binary_ops.jl")
include("llvm_intrin/conversion.jl")
include("llvm_intrin/masks.jl")
include("llvm_intrin/intrin_funcs.jl")
include("llvm_intrin/memory_addr.jl")
include("llvm_intrin/unary_ops.jl")
include("llvm_intrin/vbroadcast.jl")
include("llvm_intrin/vector_ops.jl")
include("promotion.jl")
include("number_vectors.jl")
include("ranges.jl")
include("alignment.jl")
include("precompile.jl")
_precompile_()

end # module
