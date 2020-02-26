module VectorizationBase

using LinearAlgebra

export Vec, VE, SVec, Mask, _MM,
    firstval, gep,
    extract_data,
    pick_vector_width,
    pick_vector_width_shift,
    stridedpointer,
    ZeroInitializedPointer,
    PackedStridedPointer, RowMajorStridedPointer,
    StaticStridedPointer, StaticStridedStruct,
    vload, vstore!, vbroadcast, Static

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

const VE{T} = Core.VecElement{T}
const Vec{W,T<:Number} = NTuple{W,VE{T}}

abstract type AbstractStructVec{W,T<:Number} end
struct SVec{W,T} <: AbstractStructVec{W,T}
    data::Vec{W,T}
    # SVec{N,T}(v) where {N,T} = new(v)
end
struct Mask{W,U<:Unsigned} <: AbstractStructVec{W,Bool}
    u::U
end
@inline Mask{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u)
# Const prop is good enough; added an @inferred test to make sure.
@inline Mask(u::U) where {U<:Unsigned} = Mask{sizeof(u)<<3,U}(u)

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
@generated function vbroadcast(::Type{Vec{W,Ptr{T}}}, s::Ptr{T}) where {W, T}
    typ = "i$(8sizeof(Int))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Base.llvmcall( $(join(instrs,"\n")), Vec{$W,Ptr{$T}}, Tuple{Ptr{$T}}, s )
    end
end
@generated function vbroadcast(::Type{Vec{W,T}}, s::T) where {W, T <: Integer}
    typ = "i$(8sizeof(T))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Base.llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, s )
    end
end
@generated function vbroadcast(::Type{Vec{W,T}}, s::T) where {W, T <: Union{Float16,Float32,Float64}}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Base.llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, s )
    end
end
@generated function vbroadcast(::Type{Vec{W,T}}, ptr::Ptr{T}) where {W, T}
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
        Base.llvmcall( $(join(instrs,"\n")), Vec{$W,$T}, Tuple{Ptr{$T}}, ptr )
    end
end
@generated function vzero(::Type{Vec{W,T}}) where {W,T}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = """
    ret $vtyp zeroinitializer
    """
    quote
        $(Expr(:meta,:inline))
        Base.llvmcall($instrs, Vec{$W,$T}, Tuple{}, )
    end
end
# @inline vzero(::Type{Vec{W,T}}) where {W,T} = vzero(Val{W}(), T)
@inline vbroadcast(::Val{W}, s::T) where {W,T} = SVec(vbroadcast(Vec{W,T}, s))
@inline vbroadcast(::Val{W}, ptr::Ptr{T}) where {W,T} = SVec(vbroadcast(Vec{W,T}, ptr))
@inline vbroadcast(::Type{Vec{W,T1}}, s::T2) where {W,T1,T2} = vbroadcast(Vec{W,T1}, convert(T1,s))
@inline vbroadcast(::Type{Vec{W,T1}}, s::T2) where {W,T1<:Integer,T2<:Integer} = vbroadcast(Vec{W,T1}, s % T1)
@inline vbroadcast(::Type{Vec{W,T}}, ptr::Ptr) where {W,T} = vbroadcast(Vec{W,T}, Base.unsafe_convert(Ptr{T},ptr))
@inline vbroadcast(::Type{SVec{W,T}}, s) where {W,T} = SVec(vbroadcast(Vec{W,T}, s))
@inline vbroadcast(::Type{Vec{W,T}}, v::Vec{W,T}) where {W,T} = v
@inline vbroadcast(::Type{SVec{W,T}}, v::SVec{W,T}) where {W,T} = v
@inline vbroadcast(::Type{SVec{W,T}}, v::Vec{W,T}) where {W,T} = SVec(v)

@inline vone(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Vec{W,T}, one(T))
# @inline vzero(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Vec{W,T}, zero(T))
@inline vone(::Type{SVec{W,T}}) where {W,T} = SVec(vbroadcast(Vec{W,T}, one(T)))
@inline vzero(::Val{W}, ::Type{T}) where {W,T} = SVec(vzero(Vec{W,T}))
@inline vzero(::Type{SVec{W,T}}) where {W,T} = SVec(vzero(Vec{W,T}))
@inline vone(::Type{T}) where {T} = one(T)
@inline vzero(::Type{T}) where {T} = zero(T)
@inline vzero() = vzero(pick_vector_width_val(Float64), Float64)
@inline VectorizationBase.SVec{W}(s::T) where {W,T<:Number} = SVec(vbroadcast(Vec{W,T}, s))
@inline VectorizationBase.SVec{W,T}(s::T) where {W,T<:Number} = SVec(vbroadcast(Vec{W,T}, s))
@inline VectorizationBase.SVec{W,T}(s::Number) where {W,T} = SVec(vbroadcast(Vec{W,T}, convert(T, s)))

@inline VectorizationBase.SVec{W}(v::Vec{W,T}) where {W,T<:Number} = SVec(v)
@inline VectorizationBase.SVec{W}(v::SVec{W,T}) where {W,T<:Number} = v

@inline SVec{W,T}(v::Vararg{T,W}) where {W,T} = @inbounds SVec{W,T}(ntuple(Val(W)) do w Core.VecElement(v[w]) end)
@inline SVec{W,T1}(v::Vararg{T2,W}) where {W,T1,T2} = @inbounds SVec{W,T1}(ntuple(Val(W)) do w Core.VecElement(convert(T1,v[w])) end)


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



const AbstractSIMDVector{N,T} = Union{Vec{N,T},AbstractStructVec{N,T}}

@inline extract_data(v) = v
@inline extract_data(v::SVec) = v.data
@inline extract_data(v::AbstractStructVec) = v.data
@inline extract_value(v::Vec{W,T}, i) where {W,T} = v[i].value
@inline extract_value(v::SVec{W,T}, i) where {W,T} = v.data[i].value

@inline firstval(x::Vec) = first(x).value
@inline firstval(x::SVec) = first(extract_data(x)).value
@inline firstval(x) = first(x)

function Base.show(io::IO, v::SVec{W,T}) where {W,T}
    print(io, "SVec{$W,$T}<")
    for w ∈ 1:W
        print(io, v[w])
        w < W && print(io, ", ")
    end
    print(">")
end
Base.bitstring(m::Mask{W}) where {W} = bitstring(extract_data(m))[end-W+1:end]
function Base.show(io::IO, m::Mask{W}) where {W}
    bits = bitstring(m)
    bitv = split(bits, "")
    print(io, "Mask{$W,Bool}<")
    for w ∈ 1:W
        print(io, bitv[w])
        w < W && print(io, ", ")
    end
    print(">")
end

include("static.jl")
include("vectorizable.jl")
include("cpu_info.jl")
include("vector_width.jl")
include("number_vectors.jl")
include("masks.jl")
include("alignment.jl")
include("precompile.jl")
_precompile_()

end # module
