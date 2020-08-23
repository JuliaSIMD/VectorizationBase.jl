
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
const LLVM_TYPES_SYM = IdDict{Symbol,String}(
    :Float32 => "float",
    :Float64 => "double",
    :Int8 => "i8",
    :Int16 => "i16",
    :Int32 => "i32",
    :Int64 => "i64",
    :UInt8 => "i8",
    :UInt16 => "i16",
    :UInt32 => "i32",
    :UInt64 => "i64",
    :Bool => "i8"
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
    "$(LLVM_TYPES[T]) $val"
end
function llvmconst(::Type{Bool}, val)
    Bool(val) ? "i1 1" : "zeroinitializer"
end
function llvmconst(W::Integer, T, val)
    isa(val, Number) && iszero(val) && return "zeroinitializer"
    typ = LLVM_TYPES[T]
    '<' * join(("$typ $(val)" for _ in Base.OneTo(W)), ", ") * '>'
end
function llvmconst(W::Integer, ::Type{Bool}, val)
    Bool(val) || return "zeroinitializer"
    typ = "i1"
    '<' * join(("$typ $(Int(val))" for _ in Base.OneTo(W)), ", ") * '>'
end
function llvmconst(W::Int, v::String)
    '<' * join((v for _ in Base.OneTo(W)), ", ") * '>'
end
# function llvmtypedconst(T, val)
#     typ = LLVM_TYPES[T]
#     iszero(val) && return "$typ zeroinitializer"
#     "$typ $val"
# end
# function llvmtypedconst(::Type{Bool}, val)
#     Bool(val) ? "i1 1" : "i1 zeroinitializer"
# end

@static if VERSION ≥ v"1.6.0-DEV.674"
    function llvmcall_expr(decl, instr, ret, args, lret, largs, arg_syms)
        mod = """
            $decl

            define $lret @entry($(join(largs, ", "))) alwaysinline {
            top:
                $instr
            }
        """
        # attributes #0 = { alwaysinline }
        call = Expr(:call, :llvmcall, (mod, "entry"), ret, args)
        foreach(arg -> push!(call.args, arg), arg_syms)
        if first(lret) === '<'
            call = Expr(:call, :Vec, call)
        end
        Expr(:block, Expr(:meta, :inline), call)
    end
else
    function llvmcall_expr(decl, instr, ret, args, lret, largs, arg_syms)
        call = Expr(:call, :llvmcall, (decl, instr), ret, args)
        foreach(arg -> push!(call.args, arg), arg_syms)
        if first(lret) === '<'
            call = Expr(:call, :Vec, call)
        end
        Expr(:block, Expr(:meta, :inline), call)
    end
end

