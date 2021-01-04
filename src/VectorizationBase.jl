module VectorizationBase

import ArrayInterface, LinearAlgebra, Libdl, Hwloc
using ArrayInterface: StaticInt, Zero, One, contiguous_axis, contiguous_axis_indicator, contiguous_batch_size, stride_rank,
    Contiguous, CPUPointer, ContiguousBatch, StrideRank, device,
    known_length, known_first, known_last, strides, offsets,
    static_first, static_last, static_length
import IfElse: ifelse

using Preferences

include("preferences.jl")

# using LinearAlgebra: Adjoint,

# const LLVM_SHOULD_WORK = Sys.ARCH !== :i686 && isone(length(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist())))

## Until SIMDPirates stops importing it
# isfile(joinpath(@__DIR__, "cpu_info.jl")) || throw("File $(joinpath(@__DIR__, "cpu_info.jl")) does not exist. Please run `using Pkg; Pkg.build()`.")

export Vec, Mask, MM, stridedpointer, vload, vstore!, StaticInt, vbroadcast, mask

# using Base: llvmcall
using Base: llvmcall, VecElement, HWReal
# @inline llvmcall(s::String, args...) = Base.llvmcall(s, args...)
# @inline llvmcall(s::Tuple{String,String}, args...) = Base.llvmcall(s, args...)

# export Vec, VE, Vec, Mask, MM,
#     gep, gesp,
#     data,
#     pick_vector_width,
#     pick_vector_width_shift,
#     stridedpointer,
#     PackedStridedPointer, RowMajorStridedPointer,
#     StaticIntStridedPointer, StaticIntStridedStruct,
#     vload, vstore!, vbroadcast, StaticInt, mask, masktable

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

const SignedHW = Union{Int8,Int16,Int32,Int64}
const UnsignedHW = Union{UInt8,UInt16,UInt32,UInt64}
const IntegerTypesHW = Union{SignedHW,UnsignedHW}
const IntegerTypes = Union{StaticInt,IntegerTypesHW}

struct Bit; data::Bool; end # Dummy for Ptr
const Boolean = Union{Bit,Bool}
const NativeTypesExceptBit = Union{Bool,HWReal}
const NativeTypes = Union{NativeTypesExceptBit, Bit}

const _Vec{W,T<:Number} = NTuple{W,Core.VecElement{T}}
# const _Vec{W,T<:Number} = Tuple{VecElement{T},Vararg{VecElement{T},W}}
# @eval struct StaticInt{N} <: Number
#     (f::Type{<:StaticInt})() = $(Expr(:new,:f))
# end
# Base.@pure StaticInt(N) = StaticInt{N}()

abstract type AbstractSIMD{W,T <: Union{<:StaticInt,NativeTypes}} <: Real end
abstract type AbstractSIMDVector{W,T} <: AbstractSIMD{W,T} end
const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
# const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
const IntegerTypesV = Union{AbstractSIMD{<:Any,<:IntegerTypes},IntegerTypesHW}
struct Vec{W,T} <: AbstractSIMDVector{W,T}
    data::NTuple{W,Core.VecElement{T}}
    @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:Union{NativeTypes,StaticInt}} = new{W,T}(x)
    @generated function Vec(x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}) where {_W,T<:Union{NativeTypes,StaticInt}}
        W = _W + 1
        # @assert W === pick_vector_width(W, T)# || W === 8
        Expr(:block, Expr(:meta,:inline), Expr(:call, Expr(:curly, :Vec, W, T), :x))
    end
    # @inline function Vec(x::NTuple{W,<:Core.VecElement}) where {W}
    #     T = eltype(x)
    #     @assert W === pick_vector_width(W, T)
    #     # @assert ispow2(W) && (W ≤ max(pick_vector_width(W, T), 8))
    #     new{W,T}(x)
    # end
end
struct VecUnroll{N,W,T,V<:AbstractSIMDVector{W,T}} <: AbstractSIMD{W,T}
    data::Tuple{V,Vararg{V,N}}
    @inline VecUnroll(data::Tuple{V,Vararg{V,N}}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
end

@inline Base.copy(v::AbstractSIMDVector) = v
@inline asvec(x::_Vec) = Vec(x)
@inline asvec(x) = x
@inline data(vu::VecUnroll) = vu.data

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

@generated function simd_vec(y::T, x::Vararg{T,_W}) where {T,_W}
    W = 1 + _W
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
    llvmc = :(llvmcall($(join(instrs,"\n")), _Vec{$Wfull,$T}, $Tup, y))
    for w ∈ 1:_W
        push!(llvmc.args, Expr(:ref, :x, w))
    end
    quote
        $(Expr(:meta,:inline))
        Vec($llvmc)
    end
end

function vec_quote(W, Wpow2, offset = 0)
    call = Expr(:call, :simd_vec); Wpow2 += offset
    iszero(offset) && push!(call.args, :y)
    foreach(w -> push!(call.args, Expr(:ref, :x, w)), max(1,offset):min(W,Wpow2)-1)
    # foreach(w -> push!(call.args, Expr(:call, :VecElement, Expr(:call, :zero, :T))), W+1:Wpow2)
    call
end
@generated function Vec(y::T, x::Vararg{T,_W}) where {_W, T <: NativeTypes}
    W = _W + 1
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


struct Mask{W,U<:Unsigned} <: AbstractSIMDVector{W,Bit}
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


function Base.show(io::IO, v::AbstractSIMDVector{W,T}) where {W,T}
    print(io, "Vec{$W,$T}<")
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
@inline data(i::MM) = i.i

@inline extractelement(i::MM{W,X,I}, j) where {W,X,I} = i.i + (X % I) * (j % I)

"""
  pause()

For use in spin-and-wait loops, like spinlocks.
"""
@inline pause() = ccall(:jl_cpu_pause, Cvoid, ())
const CACHE_INCLUSIVITY = let
    if !((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686))
         (false,false,false,false)
    else
        # source: https://github.com/m-j-w/CpuId.jl/blob/401b638cb5a020557bce7daaf130963fb9c915f0/src/CpuInstructions.jl#L38
        # credit Markus J. Weber, copyright: https://github.com/m-j-w/CpuId.jl/blob/master/LICENSE.md
        function get_cache_edx(subleaf)
            Base.llvmcall(
                """
                    ; leaf = %0, subleaf = %1, %2 is some label
                    ; call 'cpuid' with arguments loaded into registers EAX = leaf, ECX = subleaf
                    %2 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid",
                        "={ax},={bx},={cx},={dx},{ax},{cx},~{dirflag},~{fpsr},~{flags}"
                        (i32 4, i32 %0) #2
                    ; retrieve the result values and return eax and edx contents
                    %3 = extractvalue { i32, i32, i32, i32 } %2, 0
                    %4 = extractvalue { i32, i32, i32, i32 } %2, 3
                    %5  = insertvalue [2 x i32] undef, i32 %3, 0
                    %6  = insertvalue [2 x i32]   %5 , i32 %4, 1
                    ; return the value
                    ret [2 x i32] %6
                """
                # llvmcall requires actual types, rather than the usual (...) tuple
                , Tuple{UInt32,UInt32}, Tuple{UInt32}, subleaf % UInt32
            )
        end
        # eax0, edx1 = get_cache_edx(0x00000000)
        l1_inc = false
        eax1, edx1 = get_cache_edx(0x00000001)
        l2_inc = ((edx1 & 0x00000002) != 0x00000000) & (eax1 & 0x1f != 0x00000000)
        eax2, edx2 = get_cache_edx(0x00000002)
        l3_inc = ((edx2 & 0x00000002) != 0x00000000) & (eax2 & 0x1f != 0x00000000)
        eax3, edx3 = get_cache_edx(0x00000003)
        l4_inc = ((edx3 & 0x00000002) != 0x00000000) & (eax3 & 0x1f != 0x00000000)
        (l1_inc, l2_inc, l3_inc, l4_inc)
    end
end

include("static.jl")
include("cartesianvindex.jl")
# include("vectorizable.jl")
# include("strideprodcsestridedpointers.jl")
const TOPOLOGY = try
    Hwloc.topology_load();
catch e
    @warn e
    @warn """
        Using Hwloc failed. Please file an issue with the above warning at: https://github.com/JuliaParallel/Hwloc.jl
        Proceeding with generic topology assumptions. This may result in reduced performance.
    """
    nothing
end
if TOPOLOGY === nothing
    include("topology_generic.jl")
else
    include("topology.jl")
end
@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    include("cpu_info_x86_llvm.jl")
else
    include("cpu_info_generic.jl")
end
include("vector_width.jl")
include("llvm_types.jl")
include("lazymul.jl")
include("strided_pointers/stridedpointers.jl")
# include("strided_pointers/bitpointers.jl")
include("strided_pointers/cartesian_indexing.jl")
include("strided_pointers/grouped_strided_pointers.jl")
include("strided_pointers/cse_stridemultiples.jl")
include("llvm_intrin/binary_ops.jl")
include("llvm_intrin/conversion.jl")
include("llvm_intrin/masks.jl")
include("llvm_intrin/intrin_funcs.jl")
include("llvm_intrin/memory_addr.jl")
include("llvm_intrin/unary_ops.jl")
include("llvm_intrin/vbroadcast.jl")
include("llvm_intrin/vector_ops.jl")
include("llvm_intrin/nonbroadcastingops.jl")
include("fmap.jl")
include("promotion.jl")
include("ranges.jl")
include("alignment.jl")
include("special/misc.jl")


# function reduce_to_onevec_quote(Nm1)
#     N = Nm1 + 1
#     q = Expr(:block, Expr(:meta,:inline))
#     assign = Expr(:tuple); syms = Vector{Symbol}(undef, N)
#     for n ∈ 1:N
#         x_n = Symbol(:x_, n)
#         push!(assign.args, x_n); syms[n] = x_n;
#     end
#     push!(q.args, Expr(:(=), assign, :(data(vu))))
#     while N > 1
#         tz = trailing_zeros(N)
#         for h ∈ 1:tz
#             N >>= 1
#             for n ∈ 1:N
#                 push!(q.args, Expr(:(=), syms[n], Expr(:call, :f, syms[n], syms[n+N])))
#             end
#         end
#         if N > 1 # N must be odd
#             push!(q.args, Expr(:(=), syms[N-1], Expr(:call, :f, syms[N-1], syms[N])))
#             N -= 1
#         end
#     end
#     push!(q.args, first(syms))
#     q
# end
# @generated function reduce_to_onevec(f::F, vu::VecUnroll{Nm1}) where {F, Nm1}
#     reduce_to_onevec_quote(Nm1)
# end
@inline reduce_to_onevec(f::F, vu::VecUnroll) where {F} = ArrayInterface.reduce_tup(f, data(vu))

include("precompile.jl")
_precompile_()



end # module
