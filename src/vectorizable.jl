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

# llvmtype(::Type{Bool8}) = "i8"
# llvmtype(::Type{Bool16}) = "i16"
# llvmtype(::Type{Bool32}) = "i32"
# llvmtype(::Type{Bool64}) = "i64"
# llvmtype(::Type{Bool128}) = "i128"


const LLVMCompatible = Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,Float16,Float32,Float64}


@generated function load(ptr::Ptr{T}) where {T <: LLVMCompatible}
    # @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    # vtyp = "<$N x $typ>"
    # decls = String[]
    # push!(decls, "!0 = !{!0}")
    instrs = String[]
    # if Aligned
    #     align = N * sizeof(T)
    # else
        align = 1#sizeof(T)   # This is overly optimistic
    # end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    # push!(flags, "!noalias !0")
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%res = load $typ, $typ* %ptr" * join(flags, ", "))
    push!(instrs, "ret $typ %res")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($(join(instrs, "\n")),
        T, Tuple{Ptr{T}}, ptr)
    end
end
@generated function store!(ptr::Ptr{T}, v::T) where {T <: LLVMCompatible}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    # vtyp = "<$N x $typ>"
    # decls = String[]
    # push!(decls, "!0 = !{!0}")
    instrs = String[]
    # if Aligned
    # align = N * sizeof(T)
    # else
    align = 1#sizeof(T)   # This is overly optimistic
    # end
    flags = [""]
    if align > 0
        push!(flags, "align $align")
    end
    # push!(flags, "!noalias !0")
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "store $typ %1, $typ* %ptr" * join(flags, ", "))
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($(join(instrs, "\n")),
        Cvoid, Tuple{Ptr{T}, T}, ptr, v)
    end
end
# Fall back definitions
@inline load(ptr::Ptr) = Base.unsafe_load(ptr)
@inline store!(ptr::Ptr{T},v::T) where {T} = Base.unsafe_store!(ptr, v)
@inline load(::Type{T1}, ptr::Ptr{T2}) where {T1, T2} = load(Base.unsafe_convert(Ptr{T1}, ptr))
@inline store!(ptr::Ptr{T1}, v::T2) where {T1,T2} = store!(ptr, convert(T1, v))


@inline tdot(a::Tuple{Int}, b::Tuple{Int}) = @inbounds a[1] * b[1]
@inline tdot(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2]
@inline tdot(a::Tuple{Int,Int,Int}, b::Tuple{Int,Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline tdot(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = @inbounds first(a)*first(b) + tdot(Base.tail(a), Base.tail(b))




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
struct Pointer{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline Pointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
@inline Base.eltype(::AbstractPointer{T}) where {T} = T
@inline gep(ptr::Pointer, i::Tuple{<:Integer}) = gep(ptr, first(i))
@inline store!(ptr::AbstractPointer{T1}, v::T2, args...) where {T1,T2} = store!(ptr, convert(T1, v), args...)
@inline vectorizable(A) = Pointer(pointer(A))

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
struct StaticStridedPointer{T,X} <: AbstractStridedPointer{T}
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


struct ZeroInitializedSparseStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
const AbstractSparseStridedPointer{T,N} = Union{SparseStridedPointer{T,N},ZeroInitializedSparseStridedPointer{T,N}}
@inline gep(ptr::AbstractSparseStridedPointer{T}, i::Integer) where {T} = @inbounds gep(ptr.ptr, first(ptr.strides)*i)
@inline gep(ptr::AbstractSparseStridedPointer{T}, i::NTuple) where {T} = @inbounds gep(ptr.ptr, tdot(i, ptr.strides))
struct ZeroInitializedStaticStridedPointer{T,X} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
end
const AbstractStaticStridedPointer{T,X} = Union{StaticStridedPointer{T,X},ZeroInitializedStaticStridedPointer{T,X}}
# @generated function unitstride(::AbstractStaticStridedPointer{T,X}) where {T,X}
    # Expr(:block, Expr(:meta,:inline), first(X.parameters)::Int)
# end
@generated function gep(ptr::AbstractStaticStridedPointer{T,X}, i::Integer) where {T,X}
    s = first(X.parameters)::Int
    g = if s == 1
        Expr(:call, :gep, :ptr, :i)
    else
        Expr(:call, :gep, :ptr, Expr(:call, :*, :i, s))
    end
    Expr(:block, Expr(:meta,:inline), g)
end
@generated function gep(ptr::AbstractStaticStridedPointer{T,X}, i::NTuple{N}) where {T,X,N}
    s = first(X.parameters)::Int
    ex = Expr(:call, :+)
    i1 = Expr(:ref, :i, 1)
    push!(ex.args, s == 1 ? i1 : Expr(:call, :*, s, i1))
    for n ∈ 2:N
        push!(ex.args, Expr(:call, :*, (X.parameters[n])::Int, Expr(:ref, :i, n)))
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(
            :macrocall,
            Symbol("@inbounds"),
            LineNumberNode(@__LINE__, @__FILE__),
            Expr(:call, :gep, :ptr, ex)
        )
    )
end
const AbstractInitializedStridedPointer{T} = Union{
    PackedStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T}
}
const AbstractZeroInitializedStridedPointer{T} = Union{
    ZeroInitializedPackedStridedPointer{T},
    ZeroInitializedSparseStridedPointer{T},
    ZeroInitializedStaticStridedPointer{T}
}
const AbstractInitializedPointer{T} = Union{
    Pointer{T},
    PackedStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T}
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
@inline stride1(x) = stride(x, 1)
@inline stride1(ptr::AbstractPackedStridedPointer) = 1
@inline stride1(ptr::AbstractSparseStridedPointer) = @inbounds first(ptr.strides)
@generated function stride1(::AbstractStaticStridedPointer{T,X}) where {T,X}
    Expr(:block, Expr(:meta, :inline), first(X.parameters)::Int)
end


@inline gep(ptr::AbstractPointer, i::CartesianIndex) = gep(ptr, i.I)

@inline Base.similar(::Pointer{T}, ptr::Ptr{T}) where {T} = Pointer(ptr)
@inline Base.similar(::ZeroInitializedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedPointer(ptr)
@inline Base.similar(p::PackedStridedPointer{T}, ptr::Ptr{T}) where {T} = PackedStridedPointer(ptr, p.strides)
@inline Base.similar(p::ZeroInitializedPackedStridedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedPackedStridedPointer(ptr, p.strides)
@inline Base.similar(p::SparseStridedPointer{T}, ptr::Ptr{T}) where {T} = SparseStridedPointer(ptr, p.strides)
@inline Base.similar(p::ZeroInitializedSparseStridedPointer{T}, ptr::Ptr{T}) where {T} = ZeroInitializedSparseStridedPointer(ptr, p.strides)
@inline Base.similar(p::StaticStridedPointer{T,X}, ptr::Ptr{T}) where {T,X} = StaticStridedPointer{T,X}(ptr)
@inline Base.similar(p::ZeroInitializedStaticStridedPointer{T,X}, ptr::Ptr{T}) where {T,X} = ZeroInitializedStaticStridedPointer{T,X}(ptr)

@inline elstride(::AbstractPointer{T}) where {T} = sizeof(T)
@inline elstride(::AbstractPointer{Cvoid}) = 1
# @inline unitstride(ptr::AbstractSparseStridedPointer{T}) where {T} = sizeof(T) * first(ptr.strides)
# @generated function unitstride(
#     ::AbstractStaticStridedPointer{T,X}
# ) where {T,X}
#     s = sizeof(T)*first(X.parameters)::Int
#     Expr(:block, Expr(:meta,:inline), s)
# end

# Pointer arithmetic
# for ptype ∈ (:Pointer, :PackedStridedPointer, :SparseStridedPointer, :StaticStridedPointer,
             # :ZeroInitializedPointer, :ZeroInitializedPackedStridedPointer, :ZeroInitializedSparseStridedPointer, :ZeroInitializedStaticStridedPointer)

@inline Base.:+(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, ptr.ptr + elstride(ptr)*i)
@inline Base.:+(i, ptr::AbstractPointer{T}) where {T} = similar(ptr, ptr.ptr + elstride(ptr)*i)
@inline Base.:-(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, ptr.ptr - elstride(ptr)*i)

# end

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

@inline load(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i))
@inline Base.unsafe_load(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i - 1))
@inline Base.getindex(ptr::AbstractInitializedPointer, i) = load(gep(ptr, i))

@inline store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)

@inline store!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i), v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i - 1), v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T, i) where {T} = store!(gep(ptr, i), v)



# @inline Base.stride(ptr::AbstractPointer{Cvoid}) = 1
# @inline Base.stride(ptr::AbstractPointer{T}) where {T} = sizeof(T)
# @inline Base.stride(ptr::DynamicStridedPointer{T}) where {T} = sizeof(T)*ptr.stride
# @inline Base.stride(ptr::StaticStridedPointer{T,S}) where {T,S} = sizeof(T)*S
# @inline Base.stride(ptr::DynamicStridedPointer{Cvoid}) = ptr.stride
# @inline Base.stride(ptr::StaticStridedPointer{Cvoid,S}) where {S} = S

@inline Pointer(A) = Pointer(pointer(A))
@inline ZeroInitializedPointer(A) = ZeroInitializedPointer(pointer(A))
# @inline DynamicStridedPointer(A::AbstractArray) = DynamicStridedPointer(pointer(A), stride(A,1))

@inline Base.pointer(ptr::AbstractPointer) = ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractPointer{T}) where {T} = ptr.ptr


@inline zeroinitialized(A::Pointer) = ZeroInitializedPointer(A.ptr)



@inline stridedpointer(x) = x#Pointer(x)
# @inline stridedpointer(x::AbstractArray) = stridedpointer(parent(x))
@inline stridedpointer(A::AbstractArray) = @inbounds PackedStridedPointer(pointer(A), Base.tail(strides(A)))
@inline stridedpointer(A::AbstractArray{T,0}) where {T} = pointer(A)
# @inline stridedpointer(A::DenseArray) = @inbounds PackedStridedPointer(pointer(A), Base.tail(strides(A)))

# @inline function broadcaststridedpointer(A::DenseArray{T,N}) where {T,N}
#     stridesA = strides(A)
#     sizeA = size(A)
#     PackedStridedPointer(
#         pointer(A),
#         ntuple(n -> sizeA[n+1] == 1 ? 0 : stridesA[n+1], Val(N-1))
#     )
# end
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
@inline stridedpointer(ptr::Ptr) = ptr
@inline stridedpointer(ptr::AbstractPointer) = ptr


struct StaticStridedStruct{T,X,S} <: AbstractStridedPointer{T}
    ptr::S
    offset::Int # keeps track of offset, incase of nested gep calls
end
@inline StaticStridedStruct{T,X}(s::S) where {T,X,S} = StaticStridedStruct{T,X,S}(s, 0)
@inline StaticStridedStruct{T,X}(s::S, i::Int) where {T,X,S} = StaticStridedStruct{T,X,S}(s, i)
@inline gep(ptr::StaticStridedStruct{T,X,S}, i::Integer) where {T,X,S} = StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + i)
# Trying to avoid generated functions
@inline gep(ptr::StaticStridedStruct{T,X,S}, i::Tuple{<:Integer}) where {T,X,S} = @inbounds StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + i[1])
@inline function gep(ptr::StaticStridedStruct{T,Tuple{A},S}, i::Tuple{<:Integer,<:Integer}) where {T,A,S}
    @inbounds StaticStridedStruct{T,Tuple{A},S}(ptr.ptr, ptr.offset + i[1] + A * i[2])
end
@inline function gep(ptr::StaticStridedStruct{T,Tuple{A,B},S}, i::Tuple{<:Integer,<:Integer,<:Integer}) where {T,A,B,S}
    @inbounds StaticStridedStruct{T,Tuple{A,B},S}(ptr.ptr, ptr.offset + i[1] + A*i[2] + B*i[3])
end
@inline function gep(ptr::StaticStridedStruct{T,Tuple{A,B,C},S}, i::Tuple{<:Integer,<:Integer,<:Integer,<:Integer}) where {T,A,B,C,S}
    @inbounds StaticStridedStruct{T,Tuple{A,B,C},S}(ptr.ptr, ptr.offset + i[1] + A*i[2] + B*i[3] + C*i[4] )
end

@generated tupletype_to_tuple(::Type{T}) where {T<:Tuple} = Expr(:block, Expr(:meta,:inline), Expr(:tuple, T.parameters...))
@inline function gep(ptr::StaticStridedStruct{T,X,S}, i::NTuple{N}) where {T,X,S,N}
    strides = tupletype_to_tuple(X)
    StaticStridedStruct{T,X,S}(ptr.ptr, ptr.offset + first(i) + tdot(strides, Base.tail(i)))
end

@inline load(r::AbstractRange, i::Tuple{<:Integer}) = @inbounds r[i[1] + 1]

