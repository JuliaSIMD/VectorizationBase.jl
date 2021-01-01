
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
    Bool => "i8",
    Bit => "i1"
)
const JULIA_TYPES = IdDict{Type{<:NativeTypes},Symbol}(
    Float32 => :Float32,
    Float64 => :Float64,
    Int8 => :Int8,
    Int16 => :Int16,
    Int32 => :Int32,
    Int64 => :Int64,
    UInt8 => :UInt8,
    UInt16 => :UInt16,
    UInt32 => :UInt32,
    UInt64 => :UInt64,
    Bool => :Bool,
    Bit => :Bit
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
    :Bool => "i8",
    :Bit => "i1",
    :Nothing => "void"
)
const TYPE_LOOKUP = IdDict{Symbol,Type{<:NativeTypes}}(
    :Float32 => Float32,
    :Float64 => Float64,
    :Int8 => Int8,
    :Int16 => Int16,
    :Int32 => Int32,
    :Int64 => Int64,
    :UInt8 => UInt8,
    :UInt16 => UInt16,
    :UInt32 => UInt32,
    :UInt64 => UInt64,
    :Bool => Bool,
    :Bit => Bit
)
const JULIA_TYPE_SIZE = IdDict{Symbol,Int}(
    :Float32 => 4,
    :Float64 => 8,
    :Int8 => 1,
    :Int16 => 2,
    :Int32 => 4,
    :Int64 => 8,
    :UInt8 => 1,
    :UInt16 => 2,
    :UInt32 => 4,
    :UInt64 => 8,
    :Bool => 1,
    :Bit => 1
)


function _get_alignment(W::Int, sym::Symbol)::Int
    sym === :Bit && return 1
    T = TYPE_LOOKUP[sym]
    if W > 1
        Base.datatype_alignment(_Vec{W,T})
    else
        Base.datatype_alignment(T)
    end
end

    
const JULIAPOINTERTYPE = 'i' * string(8sizeof(Int))

vtype(W, typ) = isone(abs(W)) ? typ : "<$W x $typ>"
vtype(W, T::DataType) = vtype(W, LLVM_TYPES[T])
julia_type(W, T) = isone(abs(W)) ? T : _Vec{W,T}

ptr_suffix(T) = "p0" * suffix(T)
ptr_suffix(W, T) = suffix(W, ptr_suffix(T))
suffix(W::Int, s::String) = W == -1 ? s : 'v' * string(W) * s
suffix(W::Int, T) = suffix(W, suffix(T))
suffix(::Type{Ptr{T}}) where {T} = "p0" * suffix(T)
suffix_jlsym(W::Int, s::Symbol) = suffix(W, suffix(T))
function suffix(T::Symbol)::String
    if T === :Float64
        "f64"
    elseif T === :Float32
        "f32"
    else
        string('i', 8JULIA_TYPE_SIZE[T])
    end
end
suffix(@nospecialize(T))::String = suffix(JULIA_TYPES[T])

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

function llvmname(op, WR, WA, T, TA)
    lret = LLVM_TYPES[T]
    ln = "llvm.$op.$(suffix(WR,T))"
    (isone(abs(WR)) || T !== first(TA)) ? ln * '.' * suffix(maximum(WA),first(TA)) : ln
end

function llvmcall_expr(op, WR, R, WA, TA, ::Nothing = nothing)
    ff = llvmname(op, WR, WA, R, TA)
    argt = Expr(:tuple)
    foreach(WT -> push!(argt.args, julia_type(WT[1], WT[2])), zip(WA,TA))
    call = Expr(:call, :ccall, ff, :llvmcall, julia_type(WR, R), argt)
    foreach(n -> push!(call.args, Expr(:call, :data, Symbol(:v, n))), 1:length(TA))
    Expr(:block, Expr(:meta, :inline), isone(abs(WR)) ? call : Expr(:call, :Vec, call))
end
function llvmcall_expr(op, WR, R, WA, TA, flags::String)
    lret = LLVM_TYPES[R]
    lvret = vtype(WR, lret)
    lop = llvmname(op, WR, WA, R, TA)
    instr = "$lvret $flags @$lop"
    larg_types = vtype.(WA, TA)
    decl = "declare $lvret @$(lop)(" * join(larg_types, ", ") * ')'
    args_for_call = ("$T %$(n-1)" for (n,T) ∈ enumerate(larg_types))
    instrs = """%res = call $flags $lvret @$(lop)($(join(args_for_call, ", ")))
        ret $lvret %res"""
    args = Expr(:curly, :Tuple); foreach(WT -> push!(args.args, julia_type(WT[1], WT[2])), zip(WA,TA))
    arg_syms = [Expr(:call, :data, Symbol(:v,n)) for n ∈ 1:length(TA)]
    # println(instrs)
    llvmcall_expr(decl, instrs, julia_type(WR, R), args, lvret, larg_types, arg_syms)
end

@static if VERSION ≥ v"1.6.0-DEV.674"
    function llvmcall_expr(decl::String, instr::String, @nospecialize(ret), @nospecialize(args), lret::String, largs, arg_syms, callonly::Bool = false)
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
        callonly && return call
        Expr(:block, Expr(:meta, :inline), call)
    end
else
    function llvmcall_expr(decl::String, instr::String, @nospecialize(ret), @nospecialize(args), lret::String, largs, arg_syms, callonly::Bool = false)
        call = Expr(:call, :llvmcall, (decl, instr), ret, args)
        foreach(arg -> push!(call.args, arg), arg_syms)
        if first(lret) === '<'
            call = Expr(:call, :Vec, call)
        end
        callonly && return call
        Expr(:block, Expr(:meta, :inline), call)
    end
end

