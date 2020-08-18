module VectorizationBase

using LinearAlgebra, Libdl
const LLVM_SHOULD_WORK = Sys.ARCH !== :i686 && isone(length(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist())))

## Until SIMDPirates stops importing it
# isfile(joinpath(@__DIR__, "cpu_info.jl")) || throw("File $(joinpath(@__DIR__, "cpu_info.jl")) does not exist. Please run `using Pkg; Pkg.build()`.")

# using Base: llvmcall
using Base: llvmcall, VecElement
# @inline llvmcall(s::String, args...) = Base.llvmcall(s, args...)
# @inline llvmcall(s::Tuple{String,String}, args...) = Base.llvmcall(s, args...)

export Vec, VE, SVec, Mask, _MM,
    gep, gesp,
    data,
    pick_vector_width,
    pick_vector_width_shift,
    stridedpointer,
    PackedStridedPointer, RowMajorStridedPointer,
    StaticStridedPointer, StaticStridedStruct,
    vload, vstore!, vbroadcast, Static, mask, masktable

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

abstract type AbstractSIMDVector{W,T <: NativeTypes} <: Real end
struct Vec{W,T} <: AbstractStructVec{W,T}
    data::NTuple{W,Core.VecElement{T}}
    @inline function Vec(x::NTuple{W,Core.VecElement{T}}) where {W,T}
        @assert W === pick_vector_width(W, T) || W === 8
        # @assert ispow2(W) && (W ≤ max(pick_vector_width(W, T), 8))
        new{W,T}(x)
    end
end
struct VecUnroll{M,W,T} <: AbstractStructVec{W,T}
    data::NTuple{M,Vec{W,T}}
end
struct VecTile{M,N,W,T} <: AbstractStructVec{W,T}
    data::NTuple{N,VecUnroll{M,Vec{W,T}}}
end
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


struct Mask{W,U<:Unsigned} <: AbstractStructVec{W,Bool}
    u::U
end
const AbstractMask{W} = Union{Mask{W}, SVec{W,Bool}}
@inline Mask{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u)
# Const prop is good enough; added an @inferred test to make sure.
@inline Mask(u::U) where {U<:Unsigned} = Mask{sizeof(u)<<3,U}(u)

@inline Base.broadcastable(v::AbstractStructVec) = Ref(v)

# SVec{N,T}(x) where {N,T} = SVec(ntuple(i -> VE(T(x)), Val(N)))
# @inline function SVec{N,T}(x::Number) where {N,T}
    # SVec(ntuple(i -> VE(T(x)), Val(N)))
# end
# @inline function SVec{N,T}(x::Vararg{<:Number,N}) where {N,T}
    # SVec(ntuple(i -> VE(T(x[i])), Val(N)))
# end
# @inline function SVec(v::Vec{N,T}) where {N,T}
    # SVec{N,T}(v)
# end
# @inline Vec(u::Unsigned) = u # Unsigned integers are treated as vectors of bools
# @inline Vec{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u) # Unsigned integers are treated as vectors of bools
# @inline SVec(v::SVec{W,T}) where {W,T} = v
# @inline SVec{W}(v::SVec{W,T}) where {W,T} = v
# @inline SVec{W,T}(v::SVec{W,T}) where {W,T} = v
# @inline SVec{W}(v::Vec{W,T}) where {W,T} = SVec{W,T}(v)
# @inline vbroadcast(::Val, b::Bool) = b


@inline Base.length(::AbstractStructVec{N}) where N = N
@inline Base.size(::AbstractStructVec{N}) where N = (N,)
@inline Base.eltype(::AbstractStructVec{N,T}) where {N,T} = T
@inline Base.conj(v::AbstractStructVec) = v # so that things like dot products work.
@inline Base.adjoint(v::AbstractStructVec) = v # so that things like dot products work.
@inline Base.transpose(v::AbstractStructVec) = v # so that things like dot products work.
@inline Base.getindex(v::SVec, i::Integer) = v.data[i].value

# @inline function SVec{N,T}(v::SVec{N,T2}) where {N,T,T2}
    # @inbounds SVec(ntuple(n -> Core.VecElement{T}(T(v[n])), Val(N)))
# end

# @inline Base.one(::Type{<:AbstractStructVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.one(::AbstractStructVec{W,T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.zero(::Type{<:AbstractStructVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, zero(T)))
# @inline Base.zero(::AbstractStructVec{W,T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, zero(T)))


# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args

@inline data(v) = v
@inline data(v::Vec) = v.data
@inline data(v::Vec{1}) = v.data[1].value
#@inline data(v::AbstractStructVec) = v.data
# @inline extract_value(v::Vec, i) = v[i].value
# @inline extract_value(v::SVec, i) = v.data[i].value


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
    bits = bitstring(m)
    bitv = split(bits, "")
    print(io, "Mask{$W,Bool}<")
    for w ∈ 0:W-1
        print(io, bitv[W-w])
        w < W-1 && print(io, ", ")
    end
    print(io, ">")
end

struct _MM{W,I<:Number}
    i::I
    @inline _MM{W}(i::T) where {W,T} = new{W,T}(i)
end


include("cartesianvindex.jl")
include("static.jl")
include("vectorizable.jl")
include("strideprodcsestridedpointers.jl")
@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    @static if Base.libllvm_version >= v"8" && VERSION >= v"1.4" && LLVM_SHOULD_WORK
        include("cpu_info_x86_llvm.jl")
    else
        include("cpu_info_x86_cpuid.jl")
    end
else
    include("cpu_info_generic.jl")
end

include("vector_width.jl")
include("number_vectors.jl")
include("masks.jl")
include("alignment.jl")
include("precompile.jl")
_precompile_()

end # module
