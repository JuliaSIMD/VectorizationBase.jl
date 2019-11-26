
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
struct Pointer{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline Pointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
struct ZeroInitializedPointer{T} <: AbstractPointer{T}
    ptr::Ptr{T}
    @inline ZeroInitializedPointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end

@inline Base.:+(ptr::Pointer{T}, i) where {T} = Pointer(ptr.ptr + sizeof(T)*i)
@inline Base.:+(i, ptr::Pointer{T}) where {T} = Pointer(ptr.ptr + sizeof(T)*i)
@inline Base.:-(ptr::Pointer{T}, i) where {T} = Pointer(ptr.ptr - sizeof(T)*i)
@inline Base.:+(ptr::Pointer{Cvoid}, i) = Pointer(ptr.ptr + i)
@inline Base.:+(i, ptr::Pointer{Cvoid}) = Pointer(ptr.ptr + i)
@inline Base.:-(ptr::Pointer{Cvoid}, i) = Pointer(ptr.ptr - i)
@inline Base.:+(ptr::ZeroInitializedPointer{T}, i) where {T} = Pointer(ptr.ptr + sizeof(T)*i)
@inline Base.:+(i, ptr::ZeroInitializedPointer{T}) where {T} = Pointer(ptr.ptr + sizeof(T)*i)
@inline Base.:-(ptr::ZeroInitializedPointer{T}, i) where {T} = Pointer(ptr.ptr - sizeof(T)*i)
@inline Base.:+(ptr::ZeroInitializedPointer{Cvoid}, i) = Pointer(ptr.ptr + i)
@inline Base.:+(i, ptr::ZeroInitializedPointer{Cvoid}) = Pointer(ptr.ptr + i)
@inline Base.:-(ptr::ZeroInitializedPointer{Cvoid}, i) = Pointer(ptr.ptr - i)
@inline Pointer(A) = Pointer(pointer(A))
@inline ZeroInitializedPointer(A) = ZeroInitializedPointer(pointer(A))
@inline Base.eltype(::AbstractPointer{T}) where {T} = T
@inline load(ptr::Pointer) = load(ptr.ptr)
@inline load(ptr::Pointer{T}, i::Integer) where {T} = load(ptr.ptr + i * sizeof(T))
@inline Base.unsafe_load(ptr::Pointer) = load(ptr.ptr)
@inline Base.unsafe_load(ptr::Pointer{T}, i::Integer) where {T} = load(ptr.ptr + (i-1) * sizeof(T))
@inline load(ptr::ZeroInitializedPointer{T}) where {T} = zero(T)
@inline load(ptr::ZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)
@inline Base.unsafe_load(ptr::ZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.unsafe_load(ptr::ZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)
@inline store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline store!(ptr::AbstractPointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + i * sizeof(T), v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + (i-1)*sizeof(T), v)
@inline Base.getindex(ptr::Pointer{T}) where {T} = load(ptr.ptr)
@inline Base.getindex(ptr::Pointer{T}, i::Integer) where {T} = load(ptr.ptr + i*sizeof(T) )
@inline Base.getindex(ptr::ZeroInitializedPointer{T}) where {T} = zero(T)
@inline Base.getindex(ptr::ZeroInitializedPointer{T}, i::Integer) where {T} = zero(T)
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
