## 


# Convert Julia types to LLVM types
const LLVMTYPE = Dict{DataType,String}(
    Bool => "i8",   # Julia represents Tuple{Bool} as [1 x i8]    
    Int8 => "i8",
    Int16 => "i16",
    Int32 => "i32",
    Int64 => "i64",
    Int128 => "i128",
    UInt8 => "i8",
    UInt16 => "i16",
    UInt32 => "i32",
    UInt64 => "i64",
    UInt128 => "i128",
    Float16 => "half",
    Float32 => "float",
    Float64 => "double",
    Nothing => "void"
)
llvmtype(x)::String = LLVMTYPE[x]
const JuliaPointerType = LLVMTYPE[Int]

const LLVMCompatible = Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,Float16,Float32,Float64}


@generated function load(ptr::Ptr{T}) where {T <: LLVMCompatible}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%res = load $typ, $typ* %ptr, align $alignment")
    push!(instrs, "ret $typ %res")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($(join(instrs, "\n")), T, Tuple{Ptr{T}}, ptr)
    end
end
@generated function store!(ptr::Ptr{T}, v::T) where {T <: LLVMCompatible}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "store $typ %1, $typ* %ptr, align $alignment")
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($(join(instrs, "\n")), Cvoid, Tuple{Ptr{T}, T}, ptr, v)
    end
end

# Fall back definitions
@inline load(ptr::Ptr) = Base.unsafe_load(ptr)
@inline load(ptr::Ptr, i::Int) = load(gep(ptr, i))
@inline store!(ptr::Ptr{T},v::T) where {T} = Base.unsafe_store!(ptr, v)
@inline load(::Type{T1}, ptr::Ptr{T2}) where {T1, T2} = load(Base.unsafe_convert(Ptr{T1}, ptr))
@inline store!(ptr::Ptr{T1}, v::T2) where {T1,T2} = store!(ptr, convert(T1, v))

@inline tdot(a::Tuple{Int}, b::Tuple{Int}) = @inbounds a[1] * b[1]
@inline tdot(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2]
@inline tdot(a::Tuple{Int,Int,Int}, b::Tuple{Int,Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
# @inline tdot(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = @inbounds a[1]*b[1] + tdot(Base.tail(a), Base.tail(b))

@inline tdot(a::Tuple{Int}, b::Tuple{N,Int}) where {N} = @inbounds a[1]*b[1]
@inline tdot(a::Tuple{N,Int}, b::Tuple{Int}) where {N} = @inbounds a[1]*b[1]
# @inline tdot(a::Tuple{Int,Int}, b::Tuple{Int}) where {N} = @inbounds a[1]*b[1]
@inline tdot(a::NTuple{M,Int}, b::NTuple{N,Int}) where {M,N} = @inbounds a[1]*b[1] + tdot(Base.tail(a), Base.tail(b))


"""
A wrapper to the base pointer type, that supports pointer arithmetic.
Note that `VectorizationBase.load` and `VectorizationBase.store!` are 0-indexed,
while `Base.unsafe_load` and `Base.unsafe_store!` are 1-indexed.
x = [1, 2, 3, 4, 5, 6, 7, 8];
ptrx = Pointer(x);
load(ptrx)
# 1
load(ptrx + 1)
# 2
ptrx[]
# 1
(ptrx+1)[]
# 2
ptrx[1]
# 1
ptrx[2]
# 2
"""
abstract type AbstractPointer{T} end

@generated function gepbyte(ptr::Ptr, i::I) where {T, I <: Integer}
    ptyp = JuliaPointerType
    ityp = llvmtype(I)
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to i8*")
    push!(instrs, "%offsetptr = getelementptr inbounds i8, i8* %ptr, $ityp %1")
    push!(instrs, "%iptr = ptrtoint i8* %offsetptr to $ptyp")
    push!(instrs, "ret $ptyp %iptr")
    quote
        Base.llvmcall(
            $(join(instrs, "\n")),
            Ptr{$T}, Tuple{Ptr{$T}, $I},
            ptr, i
        )
    end
end
@generated function gep(ptr::Ptr{T}, i::I) where {T, I <: Integer}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    ityp = llvmtype(I)
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%offsetptr = getelementptr inbounds $typ, $typ* %ptr, $ityp %1")
    push!(instrs, "%iptr = ptrtoint $typ* %offsetptr to $ptyp")
    push!(instrs, "ret $ptyp %iptr")
    quote
        Base.llvmcall(
            $(join(instrs, "\n")),
            Ptr{$T}, Tuple{Ptr{$T}, $I},
            ptr, i
        )
    end
end
@inline gep(ptr::AbstractPointer, i::Integer) = gep(ptr.ptr, i)
@generated function gep(ptr::Ptr{T}, i::NTuple{W,Core.VecElement{I}}) where {W, T, I <: Integer}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    ityp = llvmtype(I)
    vityp = "<$W x $ityp>"
    vptyp = "<$W x $ptyp>"
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%offsetptr = getelementptr inbounds $typ, $typ* %ptr, $vityp %1")
    push!(instrs, "%iptr = ptrtoint <$W x $typ*> %offsetptr to $vptyp")
    push!(instrs, "ret $vptyp %iptr")
    quote
        Base.llvmcall(
            $(join(instrs, "\n")),
            NTuple{$W,Core.VecElement{Ptr{$T}}}, Tuple{Ptr{$T}, NTuple{W,Core.VecElement{$I}}},
            ptr, i
        )
    end    
end
@inline gep(ptr::Ptr, v::SVec) = gep(ptr, extract_data(v))
@inline gep(ptr::AbstractPointer, v::SVec) = gep(ptr, extract_data(v))
@inline gep(ptr::AbstractPointer, i::NTuple{W,Core.VecElement{I}}) where {W,I<:Integer} = gep(ptr.ptr, i)
@inline gep(ptr::AbstractPointer{Cvoid}, i::Integer) where {T} = ptr.ptr + i

struct Reference{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline Reference(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
@inline Base.eltype(::AbstractPointer{T}) where {T} = T
# @inline gep(ptr::Pointer, i::Tuple{<:Integer}) = gep(ptr, first(i))
@inline store!(ptr::AbstractPointer{T1}, v::T2, args...) where {T1,T2} = store!(ptr, convert(T1, v), args...)

abstract type AbstractStridedPointer{T} <: AbstractPointer{T} end
# abstract type AbstractPackedStridedObject{T,N} <: AbstractStridedPointer{T} end
struct PackedStridedPointer{T,N} <: AbstractStridedPointer{T}#AbstractPackedStridedPointer{T,N}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
struct RowMajorStridedPointer{T,N} <: AbstractStridedPointer{T}#AbstractPackedStridedPointer{T,N}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
struct SparseStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
abstract type AbstractStaticStridedPointer{T,X} <: AbstractStridedPointer{T} end
struct StaticStridedPointer{T,X} <: AbstractStaticStridedPointer{T,X}
    ptr::Ptr{T}
end

struct ZeroInitializedPointer{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline ZeroInitializedPointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
const AbstractUnitPointer{T} = Union{Pointer{T},ZeroInitializedPointer{T}}
struct ZeroInitializedPackedStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
const AbstractPackedStridedPointer{T,N} = Union{PackedStridedPointer{T,N},ZeroInitializedPackedStridedPointer{T,N}}
@inline function gep(ptr::AbstractPackedStridedPointer{Cvoid}, i::NTuple)
    @inbounds ptr.ptr + first(i) + tdot(Base.tail(i), ptr.strides)
end
@inline gep(ptr::AbstractPackedStridedPointer{T}, i::NTuple{N,I}) where {T,N,I<:Integer} = @inbounds gep(ptr, first(i) + tdot(Base.tail(i), ptr.strides))
@inline function gep(ptr::AbstractPackedStridedPointer{Cvoid}, i::Tuple{Int})
    ptr.ptr + first(i)
end
@inline gep(ptr::AbstractPackedStridedPointer{T,0}, i::Tuple{I}) where {T,I<:Integer} = @inbounds gep(ptr.ptr, first(i))



struct ZeroInitializedRowMajorStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
const AbstractRowMajorStridedPointer{T,N} = Union{RowMajorStridedPointer{T,N},ZeroInitializedRowMajorStridedPointer{T,N}}
@inline LinearAlgebra.Transpose(ptr::RowMajorStridedPointer) = PackedStridedPointer(ptr.ptr, ptr.strides)
@inline function gep(ptr::AbstractRowMajorStridedPointer{Cvoid,N}, i::NTuple) where {N}
    j = last(i)
    s = ptr.strides
    @inbounds for n ∈ 1:N
        j += s[1 + N - n]*i[n]
    end
    j
end
@inline gep(ptr::AbstractRowMajorStridedPointer{T,0}, i::Tuple{I}) where {T,I<:Integer} = @inbounds gep(ptr.ptr, i[1])
@inline gep(ptr::AbstractRowMajorStridedPointer{T,1}, i::Tuple{I,I}) where {T,I<:Integer} = @inbounds gep(ptr.ptr, i[1]*ptr.strides[1] + i[2])
@inline gep(ptr::AbstractRowMajorStridedPointer{T,2}, i::Tuple{I,I,I}) where {T,I<:Integer} = @inbounds gep(ptr.ptr, i[1]*ptr.strides[2] + i[2]*ptr.strides[1] + i[3])
@inline gep(ptr::AbstractRowMajorStridedPointer{T}, i::NTuple{N,I}) where {T,N,I<:Integer} = (ri = reverse(i); @inbounds gep(ptr.ptr, first(ri) + tdot(ptr.strides, Base.tail(ri))))
@inline function gep(ptr::AbstractRowMajorStridedPointer{Cvoid,0}, i::Tuple{Int})
    ptr.ptr + first(i)
end
@inline gep(ptr::AbstractRowMajorStridedPointer{T}, i::NTuple) where {T} = gep(PackedStridedPointer(ptr.ptr, reverse(ptr.strides)), i)

struct ZeroInitializedSparseStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
const AbstractSparseStridedPointer{T,N} = Union{SparseStridedPointer{T,N},ZeroInitializedSparseStridedPointer{T,N}}
@inline gep(ptr::AbstractSparseStridedPointer{T}, i::Integer) where {T} = @inbounds gep(ptr.ptr, first(ptr.strides)*i)
@inline gep(ptr::AbstractSparseStridedPointer{T}, i::NTuple) where {T} = @inbounds gep(ptr.ptr, tdot(i, ptr.strides))
struct ZeroInitializedStaticStridedPointer{T,X} <: AbstractStaticStridedPointer{T,X}
    ptr::Ptr{T}
end
@generated function LinearAlgebra.Transpose(ptr::StaticStridedPointer{T,X}) where {T,X}
    tup = Expr(:curly, :Tuple)
    N = length(X.parameters)
    for n ∈ N:-1:1
        push!(tup.args, (X.parameters[n])::Int)
    end
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticStridedPointer, T, tup), Expr(:(.), :ptr, QuoteNode(:ptr))))
end
@generated function LinearAlgebra.Transpose(ptr::ZeroInitializedStaticStridedPointer{T,X}) where {T,X}
    tup = Expr(:curly, :Tuple)
    N = length(X.parameters)
    for n ∈ N:-1:1
        push!(tup.args, (X.parameters[n])::Int)
    end
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :ZeroInitializedStaticStridedPointer, T, tup), Expr(:(.), :ptr, QuoteNode(:ptr))))
end

@generated function gep(ptr::AbstractStaticStridedPointer{T,X}, i::Integer) where {T,X}
    s = first(X.parameters)::Int
    g = if s == 1
        Expr(:call, :gep, Expr(:(.), :ptr. QuoteNode(:ptr)), :i)
    else
        Expr(:call, :gep, Expr(:(.), :ptr. QuoteNote(:ptr)), Expr(:call, :*, :i, s))
    end
    Expr(:block, Expr(:meta,:inline), g)
end
function indprod(X::Core.SimpleVector, i)
    Xᵢ = (X[i])::Int
    iᵢ = Expr(:ref, :i, i)
    Xᵢ == 1 ? iᵢ : Expr(:call, :*, Xᵢ, iᵢ)
end
@generated function gep(ptr::AbstractStaticStridedPointer{T,X}, i::NTuple{N}) where {T,X,N}
    Xv = X.parameters
    M = min(N, length(X.parameters))
    if M == 1
        ind = indprod(Xv, 1)
    else
        ind = Expr(:call, :+)
        for m ∈ 1:M
            push!(ind.args, indprod(Xv, m))
        end
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(
            :macrocall,
            Symbol("@inbounds"),
            LineNumberNode(@__LINE__, Symbol(@__FILE__)),
            Expr(:call, :gep, Expr(:(.), :ptr, QuoteNode(:ptr)), ind)
        )
    )
end

struct StaticStridedStruct{T,X,S} <: AbstractStaticStridedPointer{T,X}
    ptr::S
    offset::Int # keeps track of offset, incase of nested gep calls
end

const AbstractInitializedStridedPointer{T} = Union{
    PackedStridedPointer{T},
    RowMajorStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T},
    StaticStridedStruct{T}
}
const AbstractZeroInitializedStridedPointer{T} = Union{
    ZeroInitializedPackedStridedPointer{T},
    ZeroInitializedSparseStridedPointer{T},
    ZeroInitializedStaticStridedPointer{T}
}
const AbstractInitializedPointer{T} = Union{
    Pointer{T},
    PackedStridedPointer{T},
    RowMajorStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T},
    StaticStridedStruct{T}
}
const AbstractZeroInitializedPointer{T} = Union{
    ZeroInitializedPointer{T},
    ZeroInitializedPackedStridedPointer{T},
    ZeroInitializedSparseStridedPointer{T},
    ZeroInitializedStaticStridedPointer{T}
}

@inline Base.stride(ptr::AbstractPackedStridedPointer, i) = isone(i) ? 1 : @inbounds ptr.strides[i-1]
@inline Base.stride(ptr::AbstractSparseStridedPointer, i) = @inbounds ptr.strides[i]
@generated function Base.stride(::AbstractStaticStridedPointer{T,X}, i) where {T,X}
    Expr(:block, Expr(:meta, :inline), Expr(:getindex, Expr(:tuple, X.parameters...), :i))
end
@inline LinearAlgebra.stride1(ptr::AbstractPackedStridedPointer) = 1
@inline LinearAlgebra.stride1(ptr::AbstractSparseStridedPointer) = @inbounds first(ptr.strides)
@inline LinearAlgebra.stride1(::AbstractStaticStridedPointer{T,<:Tuple{X,Vararg}}) where {T,X} = X

@inline gep(ptr::AbstractPointer, i::CartesianIndex) = gep(ptr, i.I)

@inline Base.similar(::Pointer{T}, ptr::Ptr{T}) where {T} = Pointer(ptr)
@inline Base.similar(::ZeroInitializedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedPointer(ptr)
@inline Base.similar(p::PackedStridedPointer{T}, ptr::Ptr{T}) where {T} = PackedStridedPointer(ptr, p.strides)
@inline Base.similar(p::ZeroInitializedPackedStridedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedPackedStridedPointer(ptr, p.strides)
@inline Base.similar(p::SparseStridedPointer{T}, ptr::Ptr{T}) where {T} = SparseStridedPointer(ptr, p.strides)
@inline Base.similar(p::ZeroInitializedSparseStridedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedSparseStridedPointer(ptr, p.strides)
@inline Base.similar(p::StaticStridedPointer{T,X}, ptr::Ptr{T}) where {T,X} = StaticStridedPointer{T,X}(ptr)
@inline Base.similar(p::ZeroInitializedStaticStridedPointer{T,X}, ptr::Ptr{T}) where {T,X} = ZeroInitializedStaticStridedPointer{T,X}(ptr)

@inline Base.:+(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, gep(ptr.ptr, i))
@inline Base.:+(i, ptr::AbstractPointer{T}) where {T} = similar(ptr, gep(ptr.ptr, i))
@inline Base.:-(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, gep(ptr.ptr, - i))

# Now, to define indexing
@inline load(ptr::AbstractZeroInitializedPointer{T}) where {T} = zero(T)
@inline load(ptr::AbstractZeroInitializedPointer{T}, i) where {T} = zero(T)
@inline Base.unsafe_load(ptr::AbstractZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.unsafe_load(ptr::AbstractZeroInitializedPointer{T}, i) where {T} = zero(T)
@inline Base.getindex(ptr::AbstractZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.getindex(ptr::AbstractZeroInitializedPointer{T}, i) where {T} = zero(T)

@inline load(ptr::AbstractInitializedPointer) = load(ptr.ptr)
@inline Base.unsafe_load(ptr::AbstractInitializedPointer) = load(ptr.ptr)
@inline Base.getindex(ptr::AbstractInitializedPointer) = load(ptr.ptr)

@inline load(ptr::AbstractInitializedPointer{T}, ::Tuple{}) where {T} = load(ptr.ptr)
@inline load(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i))
@inline Base.unsafe_load(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i - 1))
@inline Base.getindex(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i))

@inline store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)

@inline store!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i), v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i - 1), v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i), v)


@inline Pointer(A) = Pointer(pointer(A))
@inline ZeroInitializedPointer(A) = ZeroInitializedPointer(pointer(A))
@inline Base.pointer(ptr::AbstractPointer) = ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractPointer{T}) where {T} = ptr.ptr


@inline zeroinitialized(A::Pointer) = ZeroInitializedPointer(A.ptr)



@inline stridedpointer(x) = x#Pointer(x)
@inline stridedpointer(x::Ptr) = PackedStridedPointer(x, tuple())
@inline stridedpointer(x::Union{LowerTriangular,UpperTriangular}) = stridedpointer(parent(x))
# @inline stridedpointer(x::AbstractArray) = stridedpointer(parent(x))
@inline stridedpointer(A::AbstractArray) = @inbounds PackedStridedPointer(pointer(A), Base.tail(strides(A)))
@inline stridedpointer(A::AbstractArray{T,0}) where {T} = pointer(A)
@inline stridedpointer(A::SubArray{T,0,P,S}) where {T,P,S <: Tuple{Int,Vararg}} = pointer(A)
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P,S <: Tuple{Int,Vararg}} = SparseStridedPointer(pointer(A), strides(A))
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P,S} = PackedStridedPointer(pointer(A), Base.tail(strides(A)))
@inline stridedpointer(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T,A <: AbstractVector{T}} = stridedpointer(parent(B))

@inline function stridedpointer(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T,N,A <: AbstractArray{T,N}}
    pB = parent(B)
    RowMajorStridedPointer(pointer(pB), Base.tail(strides(pB)))
end
@inline function stridedpointer(C::Union{Adjoint{T,A},Transpose{T,A}}) where {T, P, B, A <: SubArray{T,2,P,Tuple{Int,Vararg},B}}
    pC = parent(C)
    SparseStridedPointer(pointer(pC), reverse(strides(pC)))
end

@inline stridedpointer(x::Number) = x
@inline stridedpointer(x::AbstractRange) = x
@inline stridedpointer(ptr::Pointer) = PackedStridedPointer(pointer(ptr), tuple())
@inline stridedpointer(ptr::AbstractPointer) = ptr


# @inline StaticStridedStruct{T,X}(s::S) where {T,X,S} = StaticStridedStruct{T,X,S}(s, 0)
# @inline StaticStridedStruct{T,X}(s::S, i::Int) where {T,X,S} = StaticStridedStruct{T,X,S}(s, i)
# @inline gep(ptr::StaticStridedStruct{T,X,S}, i::Integer) where {T,X,S} = StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + i)
# # Trying to avoid generated functions
# @inline gep(ptr::StaticStridedStruct{T,X,S}, i::Tuple{<:Integer}) where {T,X,S} = @inbounds StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + i[1])
# @inline function gep(ptr::StaticStridedStruct{T,Tuple{A},S}, i::Tuple{<:Integer,<:Integer}) where {T,A,S}
#     @inbounds StaticStridedStruct{T,Tuple{A},S}(ptr.ptr, ptr.offset + i[1] + A * i[2])
# end
# @inline function gep(ptr::StaticStridedStruct{T,Tuple{A,B},S}, i::Tuple{<:Integer,<:Integer,<:Integer}) where {T,A,B,S}
#     @inbounds StaticStridedStruct{T,Tuple{A,B},S}(ptr.ptr, ptr.offset + i[1] + A*i[2] + B*i[3])
# end
# @inline function gep(ptr::StaticStridedStruct{T,Tuple{A,B,C},S}, i::Tuple{<:Integer,<:Integer,<:Integer,<:Integer}) where {T,A,B,C,S}
#     @inbounds StaticStridedStruct{T,Tuple{A,B,C},S}(ptr.ptr, ptr.offset + i[1] + A*i[2] + B*i[3] + C*i[4] )
# end

# @generated tupletype_to_tuple(::Type{T}) where {T<:Tuple} = Expr(:block, Expr(:meta,:inline), Expr(:tuple, T.parameters...))
# @inline function gep(ptr::StaticStridedStruct{T,X,S}, i::NTuple{N}) where {T,X,S,N}
#     strides = tupletype_to_tuple(X)
#     StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + first(i) + tdot(strides, Base.tail(i)))
# end

@inline load(r::AbstractRange, i::Tuple{<:Integer}) = @inbounds r[i[1] + 1]


@inline subsetview(ptr::PackedStridedPointer, ::Val{1}, i::Integer) = SparseStridedPointer(gep(ptr.ptr, i), ptr.strides)
@generated function subsetview(ptr::PackedStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    I > N + 1 && return :ptr
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != I-1]...)
    offset = Expr(:call, :*, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, I - 1)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gep, :p, offset)),
        Expr(:call, :PackedStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::RowMajorStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    if N + 1 == I
        return Expr(
            :block, Expr(:meta, :inline),
            Expr(:(=), :p, Expr(:call, :gep, Expr(:(.), :ptr, QuoteNode(:ptr)), :i)),
            Expr(:call, :SparseStridedPointer, :p, Expr(:call, :reverse, Expr(:(.), :ptr, QuoteNode(:strides))))
        )
    elseif N + 1 < I
        return :ptr
    end
    strideind = N + 1 - I
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != strideind]...)
    offset = Expr(:call, :*, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, strideind)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gep, :p, offset)),
        Expr(:call, :RowMajorStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::SparseStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    I > N && return :ptr
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != I]...)
    offset = Expr(:call, :*, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, I)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gep, :p, offset)),
        Expr(:call, :SparseStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::StaticStridedPointer{T, X}, ::Val{I}, i::Integer) where {I, T, X}
    I > length(X.parameters) && return :ptr
    Xa = Expr(:curly, :Tuple)
    Xparam = X.parameters
    for n ∈ 1:length(Xparam)
        n == I && continue
        push!(Xa.args, Xparam[n])
    end
    offset = Expr(:call, :*, :i, Xparam[I])
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :gp, Expr(:call, :gep, :p, offset)),
        Expr(:call, Expr(:curly, :StaticStridedPointer, T, Xa), :gp)
    )
end

@generated function subsetview(ptr::StaticStridedStruct{T, X}, ::Val{I}, i::Integer) where {I, T, X}
    I > length(X.parameters) && return :ptr
    Xa = Expr(:curly, :Tuple)
    Xparam = X.parameters
    for n ∈ 1:length(Xparam)
        n == I && continue
        push!(Xa.args, Xparam[n])
    end
    offset = Expr(:call, :*, :i, Xparam[I])
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :offset, Expr(:call, :+, Expr(:(.), :ptr, QuoteNote(:offset)), offset)),
        Expr(:call, Expr(:curly, :StaticStridedStruct, T, Xa), :p, :offset)
    )
end

@inline stridedpointer(A::AbstractArray, indices::Tuple) = stridedpointer(view(A, indices...))
@inline stridedpointer(A::AbstractArray, ::Type{Transpose}) = stridedpointer(transpose(A))
@inline stridedpointer(A::AbstractArray, ::Type{Adjoint}) = stridedpointer(adjoint(A))
@inline stridedpointer(A::AbstractArray, ::Nothing) = stridedpointer(A)

@inline filter_strides_by_dimequal1(sz::NTuple{N,Int}, st::NTuple{N,Int}) where {N} = @inbounds ntuple(n -> sz[n] == 1 ? 0 : st[n], Val{N}())

@inline function stridedpointer_for_broadcast(A::AbstractArray{T,N}) where {T,N}
    PackedStridedPointer(pointer(A), filter_strides_by_dimequal1(Base.tail(size(A)), Base.tail(strides(A))))
end
@inline stridedpointer_for_broadcast(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T,A <: AbstractVector{T}} = stridedpointer_for_broadcast(parent(B))
@inline stridedpointer_for_broadcast(A::SubArray{T,0,P,S}) where {T,P,S <: Tuple{Int,Vararg}} = pointer(A)
@inline function stridedpointer_for_broadcast(A::SubArray{T,N,P,S}) where {T,N,P,S <: Tuple{Int,Vararg}}
    SparseStridedPointer(pointer(A), filter_strides_by_dimequal1(size(A), strides(A)))
end
@inline function stridedpointer_for_broadcast(A::SubArray{T,N,P,S}) where {T,N,P,S}
    PackedStridedPointer(pointer(A), filter_strides_by_dimequal1(Base.tail(size(A)), Base.tail(strides(A))))
end


struct MappedStridedPointer{F, T, P <: AbstractPointer{T}}
    f::F
    ptr::P
end
@inline load(ptr::MappedStridedPointer) = ptr.f(load(ptr.ptr))

