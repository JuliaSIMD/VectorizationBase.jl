module VectorizationBase

using LinearAlgebra, Libdl
const LLVM_SHOULD_WORK = isone(length(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist())))

# isfile(joinpath(@__DIR__, "cpu_info.jl")) || throw("File $(joinpath(@__DIR__, "cpu_info.jl")) does not exist. Please run `using Pkg; Pkg.build()`.")

# using Base: llvmcall
# using Base: llvmcall
@inline llvmcall(s::String, args...) = Base.llvmcall(s, args...)
@inline llvmcall(s::Tuple{String,String}, args...) = Base.llvmcall(s, args...)

export Vec, VE, SVec, Mask, _MM,
    firstval, gep, gesp,
    extract_data,
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
const IntTypes = Union{Int8, Int16, Int32, Int64, Int128}
const UIntTypes = Union{UInt8, UInt16, UInt32, UInt64, UInt128}
const IntegerTypes = Union{IntTypes, UIntTypes, Ptr, Bool}
const FloatingTypes = Union{Float16, Float32, Float64}
const ScalarTypes = Union{IntegerTypes, FloatingTypes}

const VE{T} = Core.VecElement{T}
const Vec{W,T<:Number} = NTuple{W,VE{T}}
const _Vec{W,T<:Number} = Tuple{VE{T},Vararg{VE{T},W}}

abstract type AbstractStructVec{W,T<:Number} end
struct SVec{W,T} <: AbstractStructVec{W,T}
    data::Vec{W,T}
    # SVec{N,T}(v) where {N,T} = new(v)
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
@inline SVec(u::Unsigned) = u # Unsigned integers are treated as vectors of bools
@inline SVec{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u) # Unsigned integers are treated as vectors of bools
@inline SVec(v::SVec{W,T}) where {W,T} = v
@inline SVec{W}(v::SVec{W,T}) where {W,T} = v
@inline SVec{W,T}(v::SVec{W,T}) where {W,T} = v
@inline SVec{W}(v::Vec{W,T}) where {W,T} = SVec{W,T}(v)
@inline vbroadcast(::Val, b::Bool) = b
@generated function vbroadcast(::Type{_Vec{_W,Ptr{T}}}, s::Ptr{T}) where {_W, T}
    W = _W + 1
    typ = "i$(8sizeof(Int))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        llvmcall( $(join(instrs,"\n")), Vec{$W,Ptr{$T}}, Tuple{Ptr{$T}}, s )
    end
end
@generated function vbroadcast(::Type{_Vec{_W,T}}, s::T) where {_W, T <: Integer}
    W = _W + 1
    typ = "i$(8sizeof(T))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, s )
    end
end
@generated function vbroadcast(::Type{_Vec{_W,T}}, s::T) where {_W, T <: Union{Float16,Float32,Float64}}
    W = _W + 1
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, s )
    end
end
@generated function vbroadcast(::Type{_Vec{_W,T}}, ptr::Ptr{T}) where {_W, T}
    W = _W + 1
    typ = llvmtype(T)
    ptyp = JuliaPointerType
    vtyp = "<$W x $typ>"
    instrs = String[]
    alignment = Base.datatype_alignment(T)
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%res = load $typ, $typ* %ptr, align $alignment")
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %res, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{Ptr{$T}}, ptr )
    end
end
@generated function vzero(::Type{_Vec{_W,T}}) where {_W,T}
    W = _W + 1
    typ = llvmtype(T)
    instrs = "ret <$W x $typ> zeroinitializer"
    quote
        $(Expr(:meta,:inline))
        llvmcall($instrs, Vec{$W,$T}, Tuple{}, )
    end
end
@generated function vzero(::Val{W}, ::Type{T}) where {W,T}
    typ = llvmtype(T)
    instrs = "ret <$W x $typ> zeroinitializer"
    quote
        $(Expr(:meta,:inline))
        SVec(llvmcall($instrs, Vec{$W,$T}, Tuple{}, ))
    end
end


# @inline vzero(::Type{Vec{W,T}}) where {W,T} = vzero(Val{W}(), T)
@inline vbroadcast(::Val{W}, s::T) where {W,T} = SVec(vbroadcast(Vec{W,T}, s))
@inline vbroadcast(::Val{W}, ptr::Ptr{T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, ptr))
@inline vbroadcast(::Type{_Vec{_W,T1}}, s) where {_W,T1} = vbroadcast(_Vec{_W,T1}, convert(T1,s))
@inline vbroadcast(::Type{_Vec{_W,T1}}, s::T2) where {_W,T1<:Integer,T2<:Integer} = vbroadcast(_Vec{_W,T1}, s % T1)
@inline vbroadcast(::Type{_Vec{_W,T}}, ptr::Ptr) where {_W,T} = vbroadcast(_Vec{_W,T}, Base.unsafe_convert(Ptr{T},ptr))
@inline vbroadcast(::Type{SVec{W,T}}, s) where {W,T} = SVec(vbroadcast(Vec{W,T}, s))
@inline vbroadcast(::Type{_Vec{_W,T}}, v::_Vec{_W,T}) where {_W,T} = v
@inline vbroadcast(::Type{SVec{W,T}}, v::SVec{W,T}) where {W,T} = v
@inline vbroadcast(::Type{SVec{W,T}}, v::Vec{W,T}) where {W,T} = SVec(v)

@inline vone(::Type{_Vec{_W,T}}) where {_W,T} = vbroadcast(_Vec{_W,T}, one(T))
# @inline vzero(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Vec{W,T}, zero(T))
@inline vone(::Type{SVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
@inline vzero(::Type{SVec{W,T}}) where {W,T} = SVec(vzero(Vec{W,T}))
@inline vone(::Type{T}) where {T} = one(T)
@inline vzero(::Type{T}) where {T<:Number} = zero(T)
@inline vzero() = SVec(vzero(pick_vector_width_val(Float64)))
@inline sveczero(::Type{T}) where {T} = Svec(vzero(pick_vector_width_val(T)))
@inline sveczero() = Svec(vzero(pick_vector_width_val(Float64)))
@inline VectorizationBase.SVec{W}(s::T) where {W,T<:Number} = SVec(vbroadcast(Vec{W,T}, s))
@inline VectorizationBase.SVec{W,T}(s::T) where {W,T<:Number} = SVec(vbroadcast(Vec{W,T}, s))
@inline VectorizationBase.SVec{W,T}(s::Number) where {W,T} = SVec(vbroadcast(Vec{W,T}, convert(T, s)))

@inline VectorizationBase.SVec{W}(v::Vec{W,T}) where {W,T<:Number} = SVec(v)
@inline VectorizationBase.SVec{W}(v::SVec{W,T}) where {W,T<:Number} = v

@inline SVec{W,T}(v::Vararg{T,W}) where {W,T} = @inbounds SVec{W,T}(ntuple(Val(W)) do w Core.VecElement(v[w]) end)
@inline SVec{W,T1}(v::Vararg{T2,W}) where {W,T1<:Number,T2<:Number} = @inbounds SVec{W,T1}(ntuple(Val(W)) do w Core.VecElement(convert(T1,v[w])) end)
@inline SVec{1,T1}(v::Vararg{T2,1}) where {T1<:Number,T2<:Number} = @inbounds SVec{1,T1}((Core.VecElement(convert(T1,v[1])),))


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

@inline Base.one(::Type{<:AbstractStructVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
@inline Base.one(::AbstractStructVec{W,T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
@inline Base.zero(::Type{<:AbstractStructVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, zero(T)))
@inline Base.zero(::AbstractStructVec{W,T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, zero(T)))


# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args
const AbstractSIMDVector{W,T} = Union{Vec{W,T},AbstractStructVec{W,T}}

@inline extract_data(v) = v
@inline extract_data(v::SVec) = v.data
@inline extract_data(v::AbstractStructVec) = v.data
@inline extract_value(v::Vec, i) = v[i].value
@inline extract_value(v::SVec, i) = v.data[i].value

@inline firstval(x::Vec) = first(x).value
@inline firstval(x::SVec) = first(extract_data(x)).value
@inline firstval(x) = first(x)

function Base.show(io::IO, v::SVec{W,T}) where {W,T}
    print(io, "SVec{$W,$T}<")
    for w ∈ 1:W
        print(io, repr(v[w]))
        w < W && print(io, ", ")
    end
    print(io, ">")
end
Base.bitstring(m::Mask{W}) where {W} = bitstring(extract_data(m))[end-W+1:end]
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
    @static if Base.libllvm_version >= v"8" && LLVM_SHOULD_WORK
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
