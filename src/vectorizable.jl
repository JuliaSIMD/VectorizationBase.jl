
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
    Float64 => "double"
)
llvmtype(x)::String = LLVMTYPE[x]


# llvmtype(::Type{Bool8}) = "i8"
# llvmtype(::Type{Bool16}) = "i16"
# llvmtype(::Type{Bool32}) = "i32"
# llvmtype(::Type{Bool64}) = "i64"
# llvmtype(::Type{Bool128}) = "i128"


const LLVMCompatible = Union{Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,Float16,Float32,Float64}


@generated function load(ptr::Ptr{T}) where {T <: LLVMCompatible}
    # @assert isa(Aligned, Bool)
    ptyp = llvmtype(Int)
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
    ptyp = llvmtype(Int)
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



@inline tdot(a::Tuple{Int}, b::Tuple{Int}) = @inbounds a[1] + b[1]
@inline tdot(a::Tuple{Int,Int}, b::Tuple{Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2]
@inline tdot(a::Tuple{Int,Int,Int}, b::Tuple{Int,Int,Int}) = @inbounds a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline tdot(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = first(a)*first(b) + tdot(Base.tail(a), Base.tail(b))




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
@inline unitstep(::AbstractPointer, i::Integer) = i
struct Pointer{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline Pointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end

abstract type AbstractStridedPointer{T} <: AbstractPointer{T} end
struct PackedStridedPointer{T,N} <: AbstractStridedPointer{T}
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
@inline linearizeindices(ptr::AbstractPackedStridedPointer, i::NTuple) = first(i) + tdot(Base.tail(i), ptr.strides)

struct ZeroInitializedSparseStridedPointer{T,N} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
const AbstractSparseStridedPointer{T,N} = Union{SparseStridedPointer{T,N},ZeroInitializedSparseStridedPointer{T,N}}
@inline unitstep(ptr::AbstractSparseStridedPointer, i::Integer) = first(ptr.strides)*i
@inline linearizeindices(ptr::AbstractSparseStridedPointer, i::NTuple) = tdot(i, ptr.strides)
struct ZeroInitializedStaticStridedPointer{T,X} <: AbstractStridedPointer{T}
    ptr::Ptr{T}
end
const AbstractStaticStridedPointer{T,X} = Union{StaticStridedPointer{T,X},ZeroInitializedStaticStridedPointer{T,X}}
@generated function unitstride(::AbstractStaticStridedPointer{T,X}) where {T,X}
    Expr(:block, Expr(:meta,:inline), first(X.parameters)::Int)
end
@generated function unitstep(::AbstractStaticStridedPointer{T,X}, i::Integer) where {T,X}
    s = first(X.parameters)::Int
    Expr(:block, Expr(:meta,:inline), s == 1 ? :i : Expr(:*, s, :i))
end
@generated function linearindices(::AbstractStaticStridedPointer{T,X}, i::NTuple{N}) where {T,X,N}
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
            ex
        )
    )
end
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
@inline load(ptr::AbstractZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)
@inline Base.unsafe_load(ptr::AbstractZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.unsafe_load(ptr::AbstractZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)
@inline Base.getindex(ptr::AbstractZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.getindex(ptr::AbstractZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)

@inline load(ptr::AbstractInitializedPointer) = load(ptr.ptr)
@inline unsafe_load(ptr::AbstractInitializedPointer) = load(ptr.ptr)
@inline Base.getindex(ptr::AbstractInitializedPointer) = load(ptr.ptr)

@inline load(ptr::AbstractInitializedPointer, i::Integer) = load((ptr + unitstep(ptr, i)).ptr)
@inline unsafe_load(ptr::AbstractInitializedPointer, i::Integer) = load((ptr + unitstep(ptr, i - 1)).ptr)
@inline Base.getindex(ptr::AbstractInitializedPointer, i::Integer) = load((ptr + unitstep(ptr, i)).ptr)




# @inline Base.stride(ptr::AbstractPointer{Cvoid}) = 1
# @inline Base.stride(ptr::AbstractPointer{T}) where {T} = sizeof(T)
# @inline Base.stride(ptr::DynamicStridedPointer{T}) where {T} = sizeof(T)*ptr.stride
# @inline Base.stride(ptr::StaticStridedPointer{T,S}) where {T,S} = sizeof(T)*S
# @inline Base.stride(ptr::DynamicStridedPointer{Cvoid}) = ptr.stride
# @inline Base.stride(ptr::StaticStridedPointer{Cvoid,S}) where {S} = S

@inline Pointer(A) = Pointer(pointer(A))
@inline ZeroInitializedPointer(A) = ZeroInitializedPointer(pointer(A))
# @inline DynamicStridedPointer(A::AbstractArray) = DynamicStridedPointer(pointer(A), stride(A,1))


@inline Base.eltype(::AbstractPointer{T}) where {T} = T
@inline load(ptr::AbstractPointer) = load(ptr.ptr)
@inline load(ptr::AbstractPointer{T}, i::Integer) where {T} = load(ptr.ptr + i * sizeof(T))
@inline Base.unsafe_load(ptr::AbstractPointer) = load(ptr.ptr)
@inline Base.unsafe_load(ptr::AbstractPointer{T}, i::Integer) where {T} = load(ptr.ptr + (i-1) * sizeof(T))
@inline store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline store!(ptr::AbstractPointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + i * sizeof(T), v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + (i-1)*sizeof(T), v)
@inline Base.getindex(ptr::AbstractPointer{T}) where {T} = load(ptr.ptr)
@inline Base.getindex(ptr::AbstractPointer{T}, i::Integer) where {T} = load(ptr.ptr + i*sizeof(T) )
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr, v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + i * sizeof(T), v)
@inline Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractPointer{T}) where {T} = ptr.ptr
@inline Base.pointer(ptr::AbstractPointer) = ptr.ptr


"""
vectorizable(x) returns a representation of x convenient for vectorization.
The generic fallback simply returns pointer(x):

@inline vectorizable(x) = pointer(x)

however pointers are sometimes not the ideal representation, and othertimes
they are not possible in Julia (eg for stack-allocated objects). This interface
allows one to customize behavior via making use of the type system.
"""
@inline vectorizable(x) = Pointer(x)
@inline vectorizable(x::AbstractPointer) = x
@inline vectorizable(x::Symmetric) = vectorizable(x.data)
@inline vectorizable(x::LinearAlgebra.AbstractTriangular) = vectorizable(x.data)
@inline vectorizable(x::Diagonal) = vectorizable(x.diag)

@inline zeroinitialized(A::Pointer) = ZeroInitializedPointer(A.ptr)

@generated function vectorizable(A::SubArray{T,N,P,S,B}) where {T,N,P,S,B}
    if first(S.parameters) <: Integer # nonunit stride 1
        quote
            $(Expr(:meta,:inline))
            DynamicStridedPointer{$T}(pointer(A), stride(A,1))
        end
    else
        quote
            $(Expr(:meta,:inline))
            Pointer{$T}(pointer(A))
        end
    end
end



# ### vectorizables
# # Extensible interface code is at risk of world age issues if it uses generated functions.
# # Extensibility is necessary here, therefore generated functions are avoided.
# abstract type AbstractStrideDescription{I} end
# struct Dense{I,N} <: AbstractStrideDescription{I}
#     strides::NTuple{N,Int}
# end
# @inline Dense{I}(strides::NTuple{N,Int}) = Dense{I,N}(strides)
# struct Packed{I,N} <: AbstractStrideDescription{I}
#     strides::NTuple{N,Int}
# end
# @inline Packed{I}(strides::NTuple{N,Int}) = Packed{I,N}(strides)
# struct Spaced{I,N} <: AbstractStrideDescription{I}
#     strides::NTuple{N,Int}
# end
# struct Static{I,X} <: AbstractStrideDescription{I} end
# @inline indices(A::DenseArray, ::Val{I}) where {I} = Dense{I}(Base.tail(strides(A)))
# @inline subarray_indices(strides::NTuple{N,Int}, i::Integer, ::Val{I}) where {I} = Spaced{I}(strides)
# @inline subarray_indices(strides::NTuple{N,Int}, i::UnitRange, ::Val{I}) where {I} = Packed{I,N}(Base.tail(strides))
# @inline indices(A::SubArray, ::Val{I}) where {I} = subarray_indices(strides(A), A.indices, Val{I}())

# @inline function vectorizables(args::Tuple{Dense{I1},a::Dense{I2}}) where {I1, I2}
    
# end


