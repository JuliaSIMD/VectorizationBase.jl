module VectorizationBase

import ArrayInterface, LinearAlgebra, Libdl, Hwloc, IfElse
using ArrayInterface:
    StaticInt, Zero, One, StaticBool, True, False,
    contiguous_axis, contiguous_axis_indicator, contiguous_batch_size, stride_rank,
    device, CPUPointer, CPUIndex, eq, ne, lt, le, gt, ge,
    known_length, known_first, known_last, strides, offsets,
    static_first, static_last, static_length
import IfElse: ifelse

asbool(::Type{True}) = true
asbool(::Type{False}) = false
# TODO: see if `@inline` is good enough.
# @inline asvalbool(r) = Val(map(Bool, r))
# @inline asvalint(r) = Val(map(Int, r))
@generated function asvalint(r::T) where {T<:Tuple{Vararg{StaticInt}}}
    t = Expr(:tuple)
    for s ∈ T.parameters
        push!(t.args, s.parameters[1])
    end
    Expr(:call, Expr(:curly, :Val, t))
end
@generated function asvalbool(r::T) where {T<:Tuple{Vararg{StaticBool}}}
    t = Expr(:tuple)
    for b ∈ T.parameters
        push!(t.args, b === True)
    end
    Expr(:call, Expr(:curly, :Val, t))
end
@inline val_stride_rank(A) = asvalint(stride_rank(A))
@inline val_dense_dims(A) = asvalbool(ArrayInterface.dense_dims(A))

# doesn't export `Zero` and `One` by default, as these names could conflict with an AD library
export Vec, Mask, MM, stridedpointer, vload, vstore!, StaticInt, True, False, Bit,
    vbroadcast, mask, vfmadd, vfmsub, vfnmadd, vfnmsub,
    VecUnroll, Unroll, pick_vector_width

using Base: llvmcall, VecElement, HWReal, tail

const FloatingTypes = Union{Float32, Float64} # Float16

const SignedHW = Union{Int8,Int16,Int32,Int64}
const UnsignedHW = Union{UInt8,UInt16,UInt32,UInt64}
const IntegerTypesHW = Union{SignedHW,UnsignedHW}
const IntegerTypes = Union{StaticInt,IntegerTypesHW}

struct Bit; data::Bool; end # Dummy for Ptr
const Boolean = Union{Bit,Bool}
const NativeTypesExceptBit = Union{Bool,HWReal}
const NativeTypes = Union{NativeTypesExceptBit, Bit}

const _Vec{W,T<:Number} = NTuple{W,Core.VecElement{T}}

abstract type AbstractSIMD{W,T <: Union{<:StaticInt,NativeTypes}} <: Real end
abstract type AbstractSIMDVector{W,T} <: AbstractSIMD{W,T} end
struct VecUnroll{N,W,T,V<:Union{NativeTypes,AbstractSIMD{W,T}}} <: AbstractSIMD{W,T}
    data::Tuple{V,Vararg{V,N}}
    # @inline VecUnroll{N,W,T,V}(data::Tuple{V,Vararg{V,N}}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
    # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,Vec{W,T}}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
    @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMD{W,T}} = new{N,W,T,V}(data)
    # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
    @inline (VecUnroll(data::Tuple{T,Vararg{T,N}})::VecUnroll{N,T,T}) where {N,T<:NativeTypes} = new{N,1,T,T}(data)
end
# const AbstractSIMD{W,T} = Union{AbstractSIMDVector{W,T},VecUnroll{<:Any,W,T}}
const VecOrScalar = Union{AbstractSIMDVector,NativeTypes}
const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
# const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
const IntegerTypesV = Union{AbstractSIMD{<:Any,<:IntegerTypes},IntegerTypesHW}
struct Vec{W,T} <: AbstractSIMDVector{W,T}
    data::NTuple{W,Core.VecElement{T}}
    @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:NativeTypes} = new{W,T}(x)
    @generated function Vec(x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}) where {_W,T<:NativeTypes}
        W = _W + 1
        # @assert W === pick_vector_width(W, T)# || W === 8
        vtyp = Expr(:curly, :Vec, W, T)
        Expr(:block, Expr(:meta,:inline), Expr(:(::), Expr(:call, vtyp, :x), vtyp))
    end
    # @inline function Vec(x::NTuple{W,<:Core.VecElement}) where {W}
    #     T = eltype(x)
    #     @assert W === pick_vector_width(W, T)
    #     # @assert ispow2(W) && (W ≤ max(pick_vector_width(W, T), 8))
    #     new{W,T}(x)
    # end
end

# @inline (VecUnroll(data::Tuple{Vec{W,T},Vararg{Vec{W,T},N}})::VecUnroll{N,W,T,Vec{W,T}}) where {N,W,T} = VecUnroll{N,W,T,Vec{W,T}}(data)
# @inline (VecUnroll(data::Tuple{Mask{W,U},Vararg{Mask{W,T},N}})::VecUnroll{N,W,Bit,Mask{W,U}}) where {N,W,U} = VecUnroll{N,W,Bit,Mask{W,U}}(data)
# @inline (VecUnroll(data::Tuple{MM{W,X,I},Vararg{MM{W,X,I},N}})::VecUnroll{N,W,I,MM{W,X,I}}) where {N,W,X,I} = VecUnroll{N,W,I,MM{W,X,I}}(data)


@inline Base.copy(v::AbstractSIMDVector) = v
@inline asvec(x::_Vec) = Vec(x)
@inline asvec(x) = x
@inline data(vu::VecUnroll) = getfield(vu, :data)

# struct VecUnroll{N,W,T} <: AbstractSIMDVector{W,T}
#     data::NTuple{N,Vec{W,T}}
# end

@inline unrolleddata(x) = x
@inline unrolleddata(x::VecUnroll) = getfield(x, :data)
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


@inline _demoteint(::Type{T}) where {T} = T
@inline _demoteint(::Type{Int64}) = Int32
@inline _demoteint(::Type{UInt64}) = UInt32



struct Mask{W,U<:UnsignedHW} <: AbstractSIMDVector{W,Bit}
    u::U
    @inline function Mask{W,U}(u::Unsigned) where {W,U} # ignores U...
        U2 = mask_type(Val{W}())
        new{W,U2}(u % U2)
    end
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

Vec{W,T}(x::Vararg{NativeTypes,W}) where {W,T<:NativeTypes} = Vec(ntuple(w -> Core.VecElement{T}(x[w]), Val{W}()))
Vec{1,T}(x::Union{Float32,Float64}) where {T<:NativeTypes} = T(x)
Vec{1,T}(x::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Bool}) where {T<:NativeTypes} = T(x)
# Vec{1,T}(x::Integer) where {T<:HWReal} = T(x)

@inline Base.length(::AbstractSIMDVector{W}) where W = W
@inline Base.size(::AbstractSIMDVector{W}) where W = (W,)
@inline Base.eltype(::AbstractSIMD{W,T}) where {W,T} = T
@inline Base.conj(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.adjoint(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.transpose(v::AbstractSIMDVector) = v # so that things like dot products work.
# @inline Base.getindex(v::Vec, i::Integer) = v.data[i].value

# Not using getindex/setindex as names to emphasize that these are generally treated as single objects, not collections.
@generated function extractelement(v::Vec{W,T}, i::I) where {W,I <: IntegerTypesHW,T}
    typ = LLVM_TYPES[T]
    instrs = """
        %res = extractelement <$W x $typ> %0, i$(8sizeof(I)) %1
        ret $typ %res
    """
    call = :(llvmcall($instrs, $T, Tuple{_Vec{$W,$T},$I}, data(v), i))
    Expr(:block, Expr(:meta, :inline), call)
end
@generated function insertelement(v::Vec{W,T}, x::T, i::I) where {W,I <: IntegerTypesHW,T}
    typ = LLVM_TYPES[T]
    instrs = """
        %res = insertelement <$W x $typ> %0, $typ %1, i$(8sizeof(I)) %2
        ret <$W x $typ> %res
    """
    call = :(Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T},$T,$I}, data(v), x, i)))
    Expr(:block, Expr(:meta, :inline), call)
end
@inline (v::AbstractSIMDVector)(i::IntegerTypesHW) = extractelement(v, i - one(i))
@inline (v::AbstractSIMDVector)(i::Integer) = extractelement(v, Int(i) - 1)
Base.@propagate_inbounds (vu::VecUnroll)(i::Integer, j::Integer) = vu.data[j](i)

@inline Base.Tuple(v::Vec{W}) where {W} = ntuple(v, Val{W}())

# @inline function Vec{N,T}(v::Vec{N,T2}) where {N,T,T2}
    # @inbounds Vec(ntuple(n -> Core.VecElement{T}(T(v[n])), Val(N)))
# end

# @inline Base.one(::Type{<:AbstractSIMDVector{W,T}}) where {W,T} = Vec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.one(::AbstractSIMDVector{W,T}) where {W,T} = Vec(vbroadcast(Vec{W,T}, one(T)))
# @inline Base.zero(::Type{<:AbstractSIMDVector{W,T}}) where {W,T} = Vec(vbroadcast(Vec{W,T}, zero(T)))
# @inline Base.zero(::AbstractSIMDVector{W,T}) where {W,T} = Vec(vbroadcast(Vec{W,T}, zero(T)))


# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args

@inline data(v) = v
@inline data(v::Vec) = getfield(v, :data)
#@inline data(v::AbstractSIMDVector) = v.data
# @inline extract_value(v::Vec, i) = v[i].value
# @inline extract_value(v::Vec, i) = v.data[i].value


function Base.show(io::IO, v::AbstractSIMDVector{W,T}) where {W,T}
    name = typeof(v)
    print(io, "$(name)<")
    for w ∈ 1:W
        print(io, repr(extractelement(v, w-1)))
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
    println(io, "$(N+1) x Vec{$W, $T}")
    for n in 1:N+1
        show(io, vu.data[n]);
        n > N || println(io)
    end
end

"""
The name `MM` type refers to _MM registers such as `XMM`, `YMM`, and `ZMM`.
`MMX` from the original MMX SIMD instruction set is a [meaningless initialism](https://en.wikipedia.org/wiki/MMX_(instruction_set)#Naming).

The `MM{W,X}` type is used to represent SIMD indexes of width `W` with stride `X`.
"""
struct MM{W,X,I<:Union{HWReal,StaticInt}} <: AbstractSIMDVector{W,I}
    i::I
    @inline MM{W,X}(i::T) where {W,X,T<:Union{HWReal,StaticInt}} = new{W,X::Int,T}(i)
end
@inline MM(i::MM{W,X}) where {W,X} = MM{W,X}(i.i)
@inline MM{W}(i::Union{HWReal,StaticInt}) where {W} = MM{W,1}(i)
@inline MM{W}(i::Union{HWReal,StaticInt}, ::StaticInt{X}) where {W,X} = MM{W,X}(i)
@inline data(i::MM) = getfield(i, :i)

@inline extractelement(i::MM{W,X,I}, j) where {W,X,I} = i.i + (X % I) * (j % I)

# function Base.getproperty(::AbstractSIMD, s)
#     throw("""
# `Base.getproperty` not defined on AbstractSIMD.
# If you wish to access the underlying data, e.g. for use with `Base.llvmcall`, use `data(v) instead.`
# If you wish to convert to work with the data as a tuple, it is recommended to use `Tuple(v)`.
# Accessing individual elements can be done via `v(1)` for the first element.
# Alternatively, `VectorizationBase.extractelement(v, i)` will access the `i+1`st element (0-indexed) and
# `VectorizationBase.insertelement(v, x, i)` will insert `x` into position `i+1` (0-indexed).
# """)
# end

"""
  pause()

For use in spin-and-wait loops, like spinlocks.
"""
@inline pause() = ccall(:jl_cpu_pause, Cvoid, ())

# notinthreadedregion() = iszero(ccall(:jl_in_threaded_region, Cint, ()))
# function assert_init_has_finished()
#     _init_has_finished[] || throw(ErrorException("bad stuff happened"))
#     return nothing
# end

include("static.jl")
include("cartesianvindex.jl")
include("topology.jl")
include("cpu_info.jl")
# include("cache_inclusivity.jl")
include("early_definitions.jl")
include("promotion.jl")
include("llvm_types.jl")
include("lazymul.jl")
include("strided_pointers/stridedpointers.jl")
# include("strided_pointers/bitpointers.jl")
include("strided_pointers/cartesian_indexing.jl")
include("strided_pointers/grouped_strided_pointers.jl")
include("strided_pointers/cse_stridemultiples.jl")
include("llvm_intrin/binary_ops.jl")
include("vector_width.jl")
include("ranges.jl")
include("llvm_intrin/conversion.jl")
include("llvm_intrin/masks.jl")
include("llvm_intrin/intrin_funcs.jl")
include("llvm_intrin/memory_addr.jl")
include("llvm_intrin/unary_ops.jl")
include("llvm_intrin/vbroadcast.jl")
(VERSION ≥ v"1.6.0-DEV.674") && include("llvm_intrin/verf.jl")
include("llvm_intrin/vector_ops.jl")
include("llvm_intrin/nonbroadcastingops.jl")
include("llvm_intrin/integer_fma.jl")
include("base_defs.jl")
include("fmap.jl")
include("alignment.jl")
include("special/misc.jl")
# include("special/log.jl")

demoteint(::Type{T}, W) where {T} = False()
demoteint(::Type{UInt64}, W::StaticInt) = gt(W, pick_vector_width(UInt64))
demoteint(::Type{Int64}, W::StaticInt) = gt(W, pick_vector_width(Int64))


@generated function simd_vec(::DemoteInt, y::_T, x::Vararg{_T,_W}) where {DemoteInt<:StaticBool,_T,_W}
    W = 1 + _W
    T = DemoteInt === True ? _demoteint(_T) : _T
    trunc = T !== _T
    Wfull = nextpow2(W)
    ty = LLVM_TYPES[T]
    init = W == Wfull ? "undef" : "zeroinitializer"
    instrs = ["%v0 = insertelement <$Wfull x $ty> $init, $ty %0, i32 0"]
    Tup = Expr(:curly, :Tuple, T)
    for w ∈ 1:_W
        push!(instrs, "%v$w = insertelement <$Wfull x $ty> %v$(w-1), $ty %$w, i32 $w")
        push!(Tup.args, T)
    end
    push!(instrs, "ret <$Wfull x $ty> %v$_W")
    llvmc = :(llvmcall($(join(instrs,"\n")), _Vec{$Wfull,$T}, $Tup))
    trunc ? push!(llvmc.args, :(y % $T)) : push!(llvmc.args, :y)
    for w ∈ 1:_W
        ref = Expr(:ref, :x, w)
        trunc && (ref = Expr(:call, :%, ref, T))
        push!(llvmc.args, ref)
    end
    quote
        $(Expr(:meta,:inline))
        Vec($llvmc)
    end
end
function vec_quote(demote, W, Wpow2, offset::Int = 0)
    call = Expr(:call, :simd_vec, Expr(:call, demote ? :True : :False)); Wpow2 += offset
    iszero(offset) && push!(call.args, :y)
    foreach(w -> push!(call.args, Expr(:ref, :x, w)), max(1,offset):min(W,Wpow2)-1)
    # foreach(w -> push!(call.args, Expr(:call, :VecElement, Expr(:call, :zero, :T))), W+1:Wpow2)
    call
end
@generated function _vec(::StaticInt{_Wpow2}, ::DemoteInt, y::T, x::Vararg{T,_W}) where {DemoteInt<:StaticBool, _Wpow2, _W, T <: NativeTypes}
    W = _W + 1
    demote = DemoteInt === True
    Wpow2 = demote ? 2_Wpow2 : _Wpow2
    if W ≤ Wpow2
        vec_quote(demote, W, Wpow2)
    else
        tup = Expr(:tuple)
        offset = 0
        while offset < W
            push!(tup.args, vec_quote(demote, W, Wpow2, offset)); offset += Wpow2
        end
        Expr(:call, :VecUnroll, tup)
    end
end
@inline function Vec(y::T, x::Vararg{T,_W}) where {_W, T <: NativeTypes}
    W = StaticInt{_W}() + One()
    _vec(pick_vector_width(W, T), demoteint(T, W), y, x...)
end

@inline reduce_to_onevec(f::F, vu::VecUnroll) where {F} = ArrayInterface.reduce_tup(f, data(vu))

include("precompile.jl")
_precompile_()

function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return
    reset_features!()
    if unwrap(cpu_name()) !== Symbol(Sys.CPU_NAME::String)
        @info "Defining CPU name."
        define_cpu_name()
    end
    safe_topology_load!()
    redefine_attr_count()
    foreach(redefine_cache, 1:4)
    return nothing
end


end # module
