
const LLVM_TYPES = IdDict{Type{<:NativeTypes},String}(
    Float32 => "float",
    Float64 => "double",
    Int8 => "i8",
    Int16 => "i16",
    Int32 => "i32",
    Int64 => "i64",
    UInt8 => "i8",
    UInt16 => "i16",
    UInt32 => "i32",
    UInt64 => "i64",
    Bool => "i8"
)
const JULIAPOINTERTYPE = 'i' * string(8sizeof(Int))

vtype(W, typ) = isone(W) ? typ : "<$W x $typ>"

function suffix(W::Int, ::Type{T}) where {T <: NativeTypes}
    s = suffix(T)
    W == -1 ? s : 'v' * string(W) * s
end
suffix(::Type{Ptr{T}}) where {T} = "p0" * suffix(T)
function suffix(@nospecialize(T))
    if T === Float32 || T === Float64
        t = 'f'
    else
        t = 'i'
    end
    t * string(8sizeof(T))
end

# Type-dependent LLVM constants
function llvmconst(T, val)
    iszero(val) && return "zeroinitializer"
    typ = llvmtype(T)
    "$typ $val"
end
function llvmconst(::Type{Bool}, val)
    Bool(val) || return "zeroinitializer"
    typ = "i1"
    "$typ $(Int(val))"
end
function llvmconst(N::Integer, T, val)
    isa(val, Number) && iszero(val) && return "zeroinitializer"
    typ = llvmtype(T)
    "<" * join(["$typ $(val)" for i in 1:N], ", ") * ">"
end
function llvmconst(N::Integer, ::Type{Bool}, val)
    Bool(val) || return "zeroinitializer"
    typ = "i1"
    "<" * join(["$typ $(Int(val))" for i in 1:N], ", ") * ">"
end
function llvmtypedconst(T, val)
    typ = llvmtype(T)
    iszero(val) && return "$typ zeroinitializer"
    "$typ $val"
end
function llvmtypedconst(::Type{Bool}, val)
    typ = "i1"
    Bool(val) || return "$typ zeroinitializer"
    "$typ $(Int(val))"
end


