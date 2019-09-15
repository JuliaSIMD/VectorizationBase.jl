
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
    decls = []
    # push!(decls, "!0 = !{!0}")
    instrs = []
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
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
        T, Tuple{Ptr{T}}, ptr)
    end
end
@generated function store!(ptr::Ptr{T}, v::T) where {T <: LLVMCompatible}
    ptyp = llvmtype(Int)
    typ = llvmtype(T)
    # vtyp = "<$N x $typ>"
    decls = []
    # push!(decls, "!0 = !{!0}")
    instrs = []
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
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
        Cvoid, Tuple{Ptr{T}, T}, ptr, v)
    end
end
# Fall back definitions
@inline load(ptr::Ptr) = Base.unsafe_load(ptr)
@inline store!(ptr::Ptr{T},v::T) where {T} = Base.unsafe_store!(ptr, v)






"""
A wrapper to the base pointer type, that supports pointer arithmetic.
"""
struct vpointer{T}
    ptr::Ptr{T}
    @inline vpointer(ptr::Ptr{T}) where {T} = new{T}(ptr)
end
@inline Base.:+(ptr::vpointer{T}, i) where {T} = vpointer(ptr.ptr + sizeof(T)*i)
@inline Base.:+(i, ptr::vpointer{T}) where {T} = vpointer(ptr.ptr + sizeof(T)*i)
@inline Base.:-(ptr::vpointer{T}, i) where {T} = vpointer(ptr.ptr - sizeof(T)*i)
@inline vpointer(A) = vpointer(pointer(A))
@inline Base.eltype(::vpointer{T}) where {T} = T
@inline load(ptr::vpointer) = load(ptr.ptr)
@inline Base.unsafe_load(ptr::vpointer) = load(ptr.ptr)
@inline Base.unsafe_load(ptr::vpointer{T}, i::Integer) where {T} = load(ptr.ptr + (i-1) * sizeof(T))
@inline store!(ptr::vpointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::vpointer{T}, v::T) where {T} = store!(ptr.ptr, v)
@inline Base.unsafe_store!(ptr::vpointer{T}, v::T, i::Integer) where {T} = store!(ptr.ptr + (i-1)*sizeof(T), v)
@inline Base.getindex(ptr::vpointer{T}) where {T} = load(ptr.ptr)
@inline Base.getindex(ptr::vpointer{T}, i) where {T} = load(ptr.ptr + (i-1)*sizeof(T) )
@inline Base.unsafe_convert(::Type{Ptr{T}}, ptr::vpointer{T}) where {T} = ptr.ptr
@inline Base.pointer(ptr::vpointer) = ptr.ptr

"""
vectorizable(x) returns a representation of x convenient for vectorization.
The generic fallback simply returns pointer(x):

@inline vectorizable(x) = pointer(x)

however pointers are sometimes not the ideal representation, and othertimes
they are not possible in Julia (eg for stack-allocated objects). This interface
allows one to customize behavior via making use of the type system.
"""
@inline vectorizable(x) = vpointer(x)
@inline vectorizable(x::vpointer) = x
