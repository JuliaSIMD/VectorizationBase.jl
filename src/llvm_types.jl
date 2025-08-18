
fast_flags(fast::Bool) = fast ? "nsz arcp contract afn reassoc" : "nsz contract"
# fast_flags(fast::Bool) = fast ? "fast" : "nsz contract"

const LLVM_TYPES = IdDict{Type{<:NativeTypes},String}(
  Float16 => "half",
  Float32 => "float",
  Float64 => "double",
  Bit => "i1",
  Bool => "i8",
  Int8 => "i8",
  UInt8 => "i8",
  Int16 => "i16",
  UInt16 => "i16",
  Int32 => "i32",
  UInt32 => "i32",
  Int64 => "i64",
  UInt64 => "i64"
  # Int128 => "i128",
  # UInt128 => "i128",
  # UInt256 => "i256",
  # UInt512 => "i512",
  # UInt1024 => "i1024",
)
const JULIA_TYPES = IdDict{Type{<:NativeTypes},Symbol}(
  Float16 => :Float16,
  Float32 => :Float32,
  Float64 => :Float64,
  Int8 => :Int8,
  Int16 => :Int16,
  Int32 => :Int32,
  Int64 => :Int64,
  # Int128 => :Int128,
  UInt8 => :UInt8,
  UInt16 => :UInt16,
  UInt32 => :UInt32,
  UInt64 => :UInt64,
  # UInt128 => :UInt128,
  Bool => :Bool,
  Bit => :Bit
  # UInt256 => :UInt256,
  # UInt512 => :UInt512,
  # UInt1024 => :UInt1024,
)
const LLVM_TYPES_SYM = IdDict{Symbol,String}(
  :Float16 => "half",
  :Float32 => "float",
  :Float64 => "double",
  :Int8 => "i8",
  :Int16 => "i16",
  :Int32 => "i32",
  :Int64 => "i64",
  # :Int128 => "i128",
  :UInt8 => "i8",
  :UInt16 => "i16",
  :UInt32 => "i32",
  :UInt64 => "i64",
  # :UInt128 => "i128",
  :Bool => "i8",
  :Bit => "i1",
  :Nothing => "void"
  # :UInt256 => "i256",
  # :UInt512 => "i512",
  # :UInt1024 => "i1024",
)
const TYPE_LOOKUP = IdDict{Symbol,Type{<:NativeTypes}}(
  :Float16 => Float16,
  :Float32 => Float32,
  :Float64 => Float64,
  :Int8 => Int8,
  :Int16 => Int16,
  :Int32 => Int32,
  :Int64 => Int64,
  # :Int128 => Int128,
  :UInt8 => UInt8,
  :UInt16 => UInt16,
  :UInt32 => UInt32,
  :UInt64 => UInt64,
  # :UInt128 => UInt128,
  :Bool => Bool,
  :Bit => Bit
  # :UInt256 => UInt256,
  # :UInt512 => UInt512,
  # :UInt1024 => UInt1024
)
const JULIA_TYPE_SIZE = IdDict{Symbol,Int}(
  :Float16 => 2,
  :Float32 => 4,
  :Float64 => 8,
  :Int8 => 1,
  :Int16 => 2,
  :Int32 => 4,
  :Int64 => 8,
  # :Int128 => 16,
  :UInt8 => 1,
  :UInt16 => 2,
  :UInt32 => 4,
  :UInt64 => 8,
  # :UInt128 => 16,
  :Bool => 1,
  :Bit => 1
  # :UInt256 => 32,
  # :UInt512 => 64,
  # :UInt1024 => 128,
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

"""
use opaque pointer
Ref:
- Switch LLVM codegen of Ptr{T} to an actual pointer type.
  https://github.com/JuliaLang/julia/pull/53687
"""
const USE_OPAQUE_PTR = VERSION >= v"1.12-DEV"

@static if !USE_OPAQUE_PTR
  const JULIAPOINTERTYPE = 'i' * string(8sizeof(Int))
else 
  const JULIAPOINTERTYPE = "ptr"
end

vtype(W, typ::String) = (isone(abs(W)) ? typ : "<$W x $typ>")::String
vtype(W, T::DataType) = vtype(W, LLVM_TYPES[T])::String
vtype(W, T::Symbol) = vtype(W, get(LLVM_TYPES_SYM, T, T))::String
push_julia_type!(x, W, T) =
  if W ≤ 1
    push!(x, T)
    nothing
  else
    push!(x, Expr(:curly, :_Vec, W, T))
    nothing
  end
append_julia_type!(x, Ws, Ts) =
  for i ∈ eachindex(Ws)
    push_julia_type!(x, Ws[i], Ts[i])
  end

ptr_suffix(W, T) = suffix(W, ptr_suffix(T))
suffix(W::Int, s::String) = W == -1 ? s : 'v' * string(W) * s
suffix_jlsym(W::Int, s::Symbol) = suffix(W, suffix(s))
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
@static if !USE_OPAQUE_PTR
  ptr_suffix(T) = "p0" * suffix(T)
  suffix(::Type{Ptr{T}}) where {T} = "p0" * suffix(T)
else 
  ptr_suffix(T) = "p0"
  suffix(::Type{Ptr{T}}) where {T} = "p0"
end 
suffix(W::Int, T) = suffix(W, suffix(T))

# Type-dependent LLVM constants
function llvmconst(T, val)::String
  iszero(val) ? "zeroinitializer" : "$(LLVM_TYPES[T]) $val"
end
function llvmconst(::Type{Bool}, val)::String
  Bool(val) ? "i1 1" : "zeroinitializer"
end
function llvmconst(W::Int, @nospecialize(T), val)::String
  isa(val, Number) && iszero(val) && return "zeroinitializer"
  typ = (LLVM_TYPES[T])::String
  '<' * join(("$typ $(val)" for _ in Base.OneTo(W)), ", ") * '>'
end
function llvmconst(W::Int, ::Type{Bool}, val)::String
  Bool(val) ?
  '<' * join(("i1 $(Int(val))" for _ in Base.OneTo(W)), ", ") * '>' :
  "zeroinitializer"
end
function llvmconst(W::Int, v::String)::String
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

function _llvmcall_expr(ff, WR, R, argt)
  if WR ≤ 1
    Expr(:call, :ccall, ff, :llvmcall, R, argt)
  else
    Expr(:call, :ccall, ff, :llvmcall, Expr(:curly, :_Vec, WR, R), argt)
  end
end

function llvmname(op::String, WR::Int, WA, T::Symbol, TA::Symbol)
  lret = LLVM_TYPES_SYM[T]
  ln = WR ≤ 1 ? "llvm.$op" : "llvm.$op.$(suffix(WR,T))"
  (isone(abs(WR)) || T !== TA) ? ln * '.' * suffix(maximum(WA), TA) : ln
end

function build_llvmcall_expr(op, WR, R::Symbol, WA, TA, ::Nothing = nothing)
  ff = llvmname(op, WR, WA, R, first(TA))
  argt = Expr(:tuple)
  append_julia_type!(argt.args, WA, TA)
  call = _llvmcall_expr(ff, WR, R, argt)
  for n ∈ eachindex(TA)
    push!(call.args, Expr(:call, :data, Symbol(:v, n)))
  end
  Expr(
    :block,
    Expr(:meta, :inline),
    isone(abs(WR)) ? call : Expr(:call, :Vec, call)
  )
end
function build_llvmcall_expr(op, WR, R::Symbol, WA, TA, flags::String)
  lret = LLVM_TYPES_SYM[R]
  lvret = vtype(WR, lret)
  lop = llvmname(op, WR, WA, R, first(TA))
  # instr = "$lvret $flags @$lop"
  larg_types = map(vtype, WA, TA)::Vector{String}
  decl = "declare $lvret @$(lop)(" * join(larg_types, ", ")::String * ')'
  args_for_call = ("$T %$(n-1)" for (n, T) ∈ enumerate(larg_types))
  instrs = """%res = call $flags $lvret @$(lop)($(join(args_for_call, ", ")))
      ret $lvret %res"""
  args = Expr(:curly, :Tuple)
  append_julia_type!(args.args, WA, TA)
  arg_syms = Vector{Expr}(undef, length(TA))
  for n ∈ eachindex(TA)
    arg_syms[n] = Expr(:call, :data, Symbol(:v, n))
  end
  if WR ≤ 1
    llvmcall_expr(decl, instrs, R, args, lvret, larg_types, arg_syms)
  else
    llvmcall_expr(
      decl,
      instrs,
      Expr(:curly, :_Vec, WR, R),
      args,
      lvret,
      larg_types,
      arg_syms
    )
  end
end

@static if VERSION ≥ v"1.6.0-DEV.674"
  function llvmcall_expr(
    decl::String,
    instr::String,
    ret::Union{Symbol,Expr},
    args::Expr,
    lret::String,
    largs::Vector{String},
    arg_syms::Vector,
    callonly::Bool = false,
    touchesmemory::Bool = false
  )
    mod = """
        $decl

        define $lret @entry($(join(largs, ", "))) alwaysinline {
        top:
            $instr
        }
    """
    # attributes #0 = { alwaysinline }
    call = Expr(
      :call,
      LLVMCALL,
      (mod::String, "entry")::Tuple{String,String},
      ret,
      args
    )
    for arg ∈ arg_syms
      push!(call.args, arg)
    end
    call = Expr(:(::), call, ret)
    if first(lret) === '<'
      call = Expr(:call, :Vec, call)
    end
    callonly && return call
    meta = if VERSION ≥ v"1.8.0-beta"
      purity = if touchesmemory
        Expr(:purity, false, false, true, true, false)
      else
        Expr(:purity, true, true, true, true, false)
      end
      VERSION >= v"1.9.0-DEV.1019" && push!(purity.args, true)
      VERSION >= v"1.11" && push!(purity.args,
        #= inaccessiblememonly =# true,
        #= noub =# true,
        #= noub_if_noinbounds =# false,
        #= consistent_overlay =# false,
        #= nortcall =# true,
      )
      Expr(:meta, purity, :inline)
    else
      Expr(:meta, :inline)
    end
    Expr(:block, meta, call)
    # Expr(:block, Expr(:meta, :inline), )
  end
else
  function llvmcall_expr(
    decl::String,
    instr::String,
    ret::Union{Symbol,Expr},
    args::Expr,
    lret::String,
    largs::Vector{String},
    arg_syms::Vector,
    callonly::Bool = false,
    touchesmemory::Bool = false
  )
    call = Expr(:call, LLVMCALL, (decl, instr), ret, args)
    foreach(arg -> push!(call.args, arg), arg_syms)
    if first(lret) === '<'
      call = Expr(:call, :Vec, call)
    end
    callonly && return call
    Expr(:block, Expr(:meta, :inline), call)
  end
end
