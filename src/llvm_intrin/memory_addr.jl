####################################################################################################
###################################### Memory Addressing ###########################################
####################################################################################################

# Operation names and dispatch pipeline:
# `vload` and `vstore!` are the user API functions. They load from an `AbstractStridedPointer` and
# are indexed via a `::Tuple` or `Unroll{AU,F,N,AV,W,M,X,<:Tuple}`.
# These calls are forwarded to `_vload` and `_vstore!`, appending information like  `register_size()`,
# whether the operations can be assumed to be aligned, and for `vstore!` whether to add alias scope
# metadata and a nontemporal hint (non-temporal requires alignment).
#
# The tuple (Cartesian) indices are then linearized and put in terms of the number of bytes, and
# forwarded to `__vload` and `__vstore!`.
#
# The name mangling was introduced to help with discoverability, and to mayke the dispatch chain clearer.
# `methods(vload)` and `methods(vstore!)` now return much fewer methods, so users have an easier time
# assessing the API.

"""
Unroll{AU,F,N,AV,W,M,X}(i::I)

  - AU: Unrolled axis
  - F: Factor, step size per unroll. If AU == AV, `F == W` means successive loads. `1` would mean offset by `1`, e.g. `x{1:8]`, `x[2:9]`, and `x[3:10]`.
  - N: How many times is it unrolled
  - AV: Vectorized axis # 0 means not vectorized, some sort of reduction
  - W: vector width
  - M: bitmask indicating whether each factor is masked
  - X: stride between loads of vectors along axis `AV`.
  - i::I - index
"""
struct Unroll{AU,F,N,AV,W,M,X,I}
  i::I
end
@inline Unroll{AU,F,N,AV,W}(i::I) where {AU,F,N,AV,W,I} =
  Unroll{AU,F,N,AV,W,zero(UInt),1,I}(i)
@inline Unroll{AU,F,N,AV,W,M}(i::I) where {AU,F,N,AV,W,M,I} =
  Unroll{AU,F,N,AV,W,M,1,I}(i)
@inline Unroll{AU,F,N,AV,W,M,X}(i::I) where {AU,F,N,AV,W,M,X,I} =
  Unroll{AU,F,N,AV,W,M,X,I}(i)
@inline data(u::Unroll) = getfield(u, :i)
const TupleIndex =
  Union{Tuple,Unroll{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Tuple}}
@inline function linear_index(
  ptr::AbstractStridedPointer,
  u::Unroll{AU,F,N,AV,W,M,X,I}
) where {AU,F,N,AV,W,M,X,I<:TupleIndex}
  p, i = linear_index(ptr, data(u))
  # Unroll{AU,F,N,AV,W,M,typeof(i)}(i)
  p, Unroll{AU,F,N,AV,W,M,X}(i)
end
unroll_params(::Type{Unroll{AU,F,N,AV,W,M,X,I}}) where {AU,F,N,AV,W,M,X,I} =
  (AU, F, N, AV, W, M, X, I)
const NestedUnroll{W,AV,X,I,AUO,FO,NO,MO,AUI,FI,NI,MI} =
  Unroll{AUO,FO,NO,AV,W,MO,X,Unroll{AUI,FI,NI,AV,W,MI,X,I}}

const VectorIndexCore{W} = Union{Vec{W},MM{W},Unroll{<:Any,<:Any,<:Any,<:Any,W}}
const VectorIndex{W} =
  Union{VectorIndexCore{W},LazyMulAdd{<:Any,<:Any,<:VectorIndexCore{W}}}
const IntegerIndex = Union{IntegerTypes,LazyMulAdd{<:Any,<:Any,<:IntegerTypes}}
const Index = Union{IntegerIndex,VectorIndex}

const VectorIndexNoUnrollCore{W} = Union{Vec{W},MM{W}}
const VectorIndexNoUnroll{W} = Union{
  VectorIndexNoUnrollCore{W},
  LazyMulAdd{<:Any,<:Any,<:VectorIndexNoUnrollCore{W}}
}
const IndexNoUnroll = Union{IntegerIndex,VectorIndexNoUnroll}

# const BoolVec = Union{AbstractMask,VecUnroll{<:Any, <:Any, Bool, <: AbstractMask}}

const SCOPE_METADATA = """
!1 = !{!\"noaliasdomain\"}
!2 = !{!\"noaliasscope\", !1}
!3 = !{!2}
"""
const LOAD_SCOPE_FLAGS = ", !alias.scope !3";
const STORE_SCOPE_FLAGS = ", !noalias !3";

# use TBAA?
# const TBAA_STR = """
# !4 = !{!5, !5, i64 0}
# !5 = !{!"jtbaa_mutab", !6, i64 0}
# !6 = !{!"jtbaa_value", !7, i64 0}
# !7 = !{!"jtbaa_data", !8, i64 0}
# !8 = !{!"jtbaa", !9, i64 0}
# !9 = !{!"jtbaa"}
# """;
# const TBAA_FLAGS = ", !tbaa !4";
const TBAA_STR = TBAA_FLAGS = ""
# const TBAA_STR = """
# !4 = !{!"jtbaa", !5, i64 0}
# !5 = !{!"jtbaa"}
# !6 = !{!"jtbaa_data", !4, i64 0}
# !7 = !{!8, !8, i64 0}
# !8 = !{!"jtbaa_arraybuf", !6, i64 0}
# """;
# const TBAA_FLAGS = ", !tbaa !7";
const LOAD_SCOPE_TBAA = SCOPE_METADATA * TBAA_STR
const LOAD_SCOPE_TBAA_FLAGS = LOAD_SCOPE_FLAGS * TBAA_FLAGS

"""
An omnibus offset constructor.

The general motivation for generating the memory addresses as LLVM IR rather than combining multiple lllvmcall Julia functions is
that we want to minimize the `inttoptr` and `ptrtoint` calculations as we go back and fourth. These can get in the way of some
optimizations, such as memory address calculations.
It is particulary import for `gather` and `scatter`s, as these functions take a `Vec{W,Ptr{T}}` argument to load/store a
`Vec{W,T}` to/from. If `sizeof(T) < sizeof(Int)`, converting the `<W x \$(typ)*` vectors of pointers in LLVM to integer
vectors as they're represented in Julia will likely make them too large to fit in a single register, splitting the operation
into multiple operations, forcing a corresponding split of the `Vec{W,T}` vector as well.
This would all be avoided by not promoting/widenting the `<W x \$(typ)>` into a vector of `Int`s.

For this last issue, an alternate workaround would be to wrap a `Vec` of 32-bit integers with a type that defines it as a pointer for use with
internal llvmcall functions, but I haven't really explored this optimization.
"""
function offset_ptr(
  T_sym::Symbol,
  ind_type::Symbol,
  indargname::Char,
  ibits::Int,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  forgep::Bool,
  rs::Int
)
  sizeof_T = JULIA_TYPE_SIZE[T_sym]
  i = 0
  Morig = M
  isbit = T_sym === :Bit
  if isbit
    typ = "i1"
    vtyp = isone(W) ? typ : "<$W x i1>"
    M = max(1, M >> 3)
    O >>= 3
    if !((isone(X) | iszero(X)) && (ind_type !== :Vec))
      throw(
        ArgumentError(
          "indexing BitArrays with a vector not currently supported."
        )
      )
    end
  else
    typ = LLVM_TYPES_SYM[T_sym]
    vtyp = vtype(W, typ) # vtyp is dest type
  end
  instrs = String[]
  if M == 0
    ind_type = :StaticInt
  elseif ind_type === :StaticInt
    M = 0
  end
  if isone(W)
    X = 1
  end
  if iszero(M)
    tz = intlog2(sizeof_T)
    tzf = sizeof_T
    index_gep_typ = typ
  else
    tz = min(trailing_zeros(M), 3)
    tzf = 1 << tz
    index_gep_typ = ((tzf == sizeof_T) | iszero(M)) ? typ : "i$(tzf << 3)"
    M >>= tz
  end
  # after this block, we will have a index_gep_typ pointer
  if iszero(O)
    push!(
      instrs,
      "%ptr.$(i) = inttoptr $(JULIAPOINTERTYPE) %0 to $(index_gep_typ)*"
    )
    i += 1
  else # !iszero(O)
    if !iszero(O & (tzf - 1)) # then index_gep_typ works for the constant offset
      offset_gep_typ = "i8"
      offset = O
    else # then we need another intermediary
      offset_gep_typ = index_gep_typ
      offset = O >> tz
    end
    push!(
      instrs,
      "%ptr.$(i) = inttoptr $(JULIAPOINTERTYPE) %0 to $(offset_gep_typ)*"
    )
    i += 1
    push!(
      instrs,
      "%ptr.$(i) = getelementptr inbounds $(offset_gep_typ), $(offset_gep_typ)* %ptr.$(i-1), i32 $(offset)"
    )
    i += 1
    if forgep && iszero(M) && (iszero(X) || isone(X))
      push!(
        instrs,
        "%ptr.$(i) = ptrtoint $(offset_gep_typ)* %ptr.$(i-1) to $(JULIAPOINTERTYPE)"
      )
      i += 1
      return instrs, i
    elseif offset_gep_typ != index_gep_typ
      push!(
        instrs,
        "%ptr.$(i) = bitcast $(offset_gep_typ)* %ptr.$(i-1) to $(index_gep_typ)*"
      )
      i += 1
    end
  end
  # will do final type conversion
  if ind_type === :Vec
    if isone(M)
      indname = indargname
    else
      indname = "indname"
      constmul = llvmconst(W, "i$(ibits) $M")
      push!(
        instrs,
        "%$(indname) = mul nsw <$W x i$(ibits)> %$(indargname), $(constmul)"
      )
    end
    push!(
      instrs,
      "%ptr.$(i) = getelementptr inbounds $(index_gep_typ), $(index_gep_typ)* %ptr.$(i-1), <$W x i$(ibits)> %$(indname)"
    )
    i += 1
    if forgep
      push!(
        instrs,
        "%ptr.$(i) = ptrtoint <$W x $index_gep_typ*> %ptr.$(i-1) to <$W x $JULIAPOINTERTYPE>"
      )
      i += 1
    elseif index_gep_typ != vtyp
      push!(
        instrs,
        "%ptr.$(i) = bitcast <$W x $index_gep_typ*> %ptr.$(i-1) to <$W x $typ*>"
      )
      i += 1
    end
    return instrs, i
  end
  if ind_type === :Integer
    if isbit
      if (Morig & 7) == 0 || ispow2(Morig)
        if abs(Morig) ≥ 8
          M = Morig >> 3
          if M != 1
            indname = "indname"
            push!(instrs, "%$(indname) = mul nsw i$(ibits) %$(indargname), $M")
          else
            indname = indargname
          end
        else
          shifter = 3 - intlog2(Morig)
          push!(instrs, "%shiftedind = ashr i$(ibits) %$(indargname), $shifter")
          if Morig > 0
            indname = "shiftedind"
          else
            indname = "indname"
            push!(instrs, "%$(indname) = mul i$(ibits) %shiftedind, -1")
          end
        end
      else
        throw(
          ArgumentError(
            "Scale factors on bit accesses must be an integer multiple of 8 or an integer power of 2."
          )
        )
        indname = "0"
      end
    else
      if isone(M)
        indname = indargname
      elseif iszero(M)
        indname = "0"
      else
        indname = "indname"
        push!(instrs, "%$(indname) = mul nsw i$(ibits) %$(indargname), $M")
      end
      # TODO: if X != 1 and X != 0, check if it is better to gep -> gep, or broadcast -> add -> gep
    end
    push!(
      instrs,
      "%ptr.$(i) = getelementptr inbounds $(index_gep_typ), $(index_gep_typ)* %ptr.$(i-1), i$(ibits) %$(indname)"
    )
    i += 1
  end
  # ind_type === :Integer || ind_type === :StaticInt
  if !(isone(X) | iszero(X)) # vec
    # LLVM assumes integers are signed for indexing
    # therefore, we need to set the integer size to be at least `nextpow2(intlog2(X*W-1)+2)` bits
    # to avoid overflow
    vibytes = max(min(4, rs ÷ W), nextpow2(intlog2(X * W - 1) + 2) >> 3)
    vityp = "i$(8vibytes)"
    vi = join((X * w for w ∈ 0:W-1), ", $vityp ")
    if typ !== index_gep_typ
      push!(
        instrs,
        "%ptr.$(i) = bitcast $(index_gep_typ)* %ptr.$(i-1) to $(typ)*"
      )
      i += 1
    end
    push!(
      instrs,
      "%ptr.$(i) = getelementptr inbounds $(typ), $(typ)* %ptr.$(i-1), <$W x $(vityp)> <$vityp $vi>"
    )
    i += 1
    if forgep
      push!(
        instrs,
        "%ptr.$(i) = ptrtoint <$W x $typ*> %ptr.$(i-1) to <$W x $JULIAPOINTERTYPE>"
      )
      i += 1
    end
    return instrs, i
  end
  if forgep # if forgep, just return now
    push!(
      instrs,
      "%ptr.$(i) = ptrtoint $(index_gep_typ)* %ptr.$(i-1) to $JULIAPOINTERTYPE"
    )
    i += 1
  elseif index_gep_typ != vtyp
    push!(
      instrs,
      "%ptr.$(i) = bitcast $(index_gep_typ)* %ptr.$(i-1) to $(vtyp)*"
    )
    i += 1
  end
  instrs, i
end

gep_returns_vector(W::Int, X::Int, M::Int, ind_type::Symbol) =
  (!isone(W) && ((ind_type === :Vec) || !(isone(X) | iszero(X))))
#::Type{T}, ::Type{I}, W::Int = 1, ivec::Bool = false, constmul::Int = 1) where {T <: NativeTypes, I <: Integer}
function gep_quote(
  ::Type{T},
  ind_type::Symbol,
  ::Type{I},
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  rs::Int
) where {T,I}
  T_sym = JULIA_TYPES[T]
  I_sym = JULIA_TYPES[I]
  gep_quote(T_sym, ind_type, I_sym, W, X, M, O, rs)
end
function gep_quote(
  T_sym::Symbol,
  ind_type::Symbol,
  I_sym::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  rs::Int
)
  sizeof_T = JULIA_TYPE_SIZE[T_sym]
  sizeof_I = JULIA_TYPE_SIZE[I_sym]
  if W > 1 && ind_type !== :Vec
    X, Xr = divrem(X, sizeof_T)
    iszero(Xr) || throw(
      ArgumentError(
        "sizeof($T_sym) == $sizeof_T, but stride between vector loads is given as $X, which is not a positive integer multiple."
      )
    )
  end
  if iszero(O) &&
     (iszero(X) | isone(X)) &&
     (iszero(M) || ind_type === :StaticInt)
    return Expr(:block, Expr(:meta, :inline), :ptr)
  end
  ibits = 8sizeof_I
  # ::Type{T}, ind_type::Symbol, indargname = '1', ibytes::Int, W::Int = 1, X::Int = 1, M::Int = 1, O::Int = 0, forgep::Bool = false
  instrs, i = offset_ptr(T_sym, ind_type, '1', ibits, W, X, M, O, true, rs)
  ret = Expr(:curly, :Ptr, T_sym)
  lret = JULIAPOINTERTYPE
  if gep_returns_vector(W, X, M, ind_type)
    ret = Expr(:curly, :_Vec, W, ret)
    lret = "<$W x $lret>"
  end

  args = Expr(:curly, :Tuple, Expr(:curly, :Ptr, T_sym))
  largs = String[JULIAPOINTERTYPE]
  arg_syms = Union{Symbol,Expr}[:ptr]

  if !(iszero(M) || ind_type === :StaticInt)
    push!(arg_syms, Expr(:call, :data, :i))
    if ind_type === :Integer
      push!(args.args, I_sym)
      push!(largs, "i$(ibits)")
    else
      push!(args.args, Expr(:curly, :_Vec, W, I_sym))
      push!(largs, "<$W x i$(ibits)>")
    end
  end
  push!(instrs, "ret $lret %ptr.$(i-1)")
  llvmcall_expr("", join(instrs, "\n"), ret, args, lret, largs, arg_syms)
end

@generated function _gep(
  ptr::Ptr{T},
  i::I,
  ::StaticInt{RS}
) where {I<:IntegerTypes,T<:NativeTypes,RS}
  gep_quote(T, :Integer, I, 1, 1, 1, 0, RS)
end
@generated function _gep(
  ptr::Ptr{T},
  ::StaticInt{N},
  ::StaticInt{RS}
) where {N,T<:NativeTypes,RS}
  gep_quote(T, :StaticInt, Int, 1, 1, 0, N, RS)
end
@generated function _gep(
  ptr::Ptr{T},
  i::LazyMulAdd{M,O,I},
  ::StaticInt{RS}
) where {T<:NativeTypes,I<:IntegerTypes,O,M,RS}
  gep_quote(T, :Integer, I, 1, 1, M, O, RS)
end
@inline _gep(ptr, i::IntegerIndex, ::StaticInt) = ptr + _materialize(i)
@generated function _gep(
  ptr::Ptr{T},
  i::Vec{W,I},
  ::StaticInt{RS}
) where {W,T<:NativeTypes,I<:IntegerTypes,RS}
  gep_quote(T, :Vec, I, W, 1, 1, 0, RS)
end
@generated function _gep(
  ptr::Ptr{T},
  i::LazyMulAdd{M,O,Vec{W,I}},
  ::StaticInt{RS}
) where {W,T<:NativeTypes,I<:IntegerTypes,M,O,RS}
  gep_quote(T, :Vec, I, W, 1, M, O, RS)
end
@inline gep(ptr::Ptr, i) = _gep(ptr, i, register_size())

@inline increment_ptr(ptr::AbstractStridedPointer) = pointer(ptr)
@inline function increment_ptr(ptr::AbstractStridedPointer, i::Tuple)
  ioffset = _offset_index(i, offsets(ptr))
  p, li = tdot(ptr, ioffset, strides(ptr))
  _gep(p, li, Zero())
end
# @inline function increment_ptr(ptr::AbstractStridedPointer{T,N,C,B,R,X,NTuple{N,Zero}}, i::Tuple) where {T,N,C,B,R,X}
#   p, li = tdot(ptr, i, strides(ptr))
#   _gep(p, li, Zero())
# end
@inline increment_ptr(p::StridedBitPointer) = offsets(p)
@inline bmap(::F, ::Tuple{}, y::Tuple{}) where {F} = ()
@inline bmap(::F, ::Tuple{}, y::Tuple) where {F} = ()
@inline bmap(::F, x::Tuple{X,Vararg{Any,K}}, ::Tuple{}) where {F,X,K} = x
#  (first(x), bmap(f, Base.tail(x), ())...)
@inline bmap(
  f::F,
  x::Tuple{X,Vararg{Any,KX}},
  y::Tuple{Y,Vararg{Any,KY}}
) where {F,X,Y,KX,KY} =
  (f(first(x), first(y)), bmap(f, Base.tail(x), Base.tail(y))...)
@inline increment_ptr(p::StridedBitPointer, i::Tuple) =
  bmap(vsub_nsw, offsets(p), i)
@inline increment_ptr(p::AbstractStridedPointer, o, i::Tuple) =
  increment_ptr(reconstruct_ptr(p, o), i)
@inline function reconstruct_ptr(sp::AbstractStridedPointer, p::Ptr)
  similar_with_offset(sp, p, offsets(sp))
end
# @inline function reconstruct_ptr(sp::AbstractStridedPointer, p::Ptr)
#   similar_no_offset(sp, p)
# end
@inline function reconstruct_ptr(
  ptr::StridedBitPointer{N,C,B,R},
  offs::NTuple{N,Int}
) where {N,C,B,R}
  stridedpointer(
    pointer(ptr),
    ArrayInterface.StrideIndex{N,R,C}(strides(ptr), offs),
    StaticInt{B}()
  )
end

@inline function gesp(
  ptr::AbstractStridedPointer,
  i::Tuple{Vararg{IntegerIndex}}
)
  similar_no_offset(ptr, increment_ptr(ptr, i))
end
@inline function gesp(
  ptr::StridedBitPointer{N,C,B,R},
  i::Tuple{Vararg{IntegerIndex,K}}
) where {N,C,B,R,K}
  stridedpointer(
    pointer(ptr),
    ArrayInterface.StrideIndex{N,R,C}(strides(ptr), increment_ptr(ptr, i)),
    StaticInt{B}()
  )
end
@inline vsub_nsw(::NullStep, _) = Zero()
@inline vsub_nsw(::NullStep, ::LazyMulAdd) = Zero() # avoid ambiguity
@inline vsub_nsw(::NullStep, ::NullStep) = Zero()
@inline vsub_nsw(::NullStep, ::StaticInt) = Zero()

@inline select_null_offset(i::Tuple{}, off::Tuple{}) = ()
@inline select_null_offset(i::Tuple{I1,Vararg}, off::Tuple{O1}) where {I1,O1} =
  (Zero(),)
@inline select_null_offset(
  i::Tuple{NullStep,Vararg},
  off::Tuple{O1}
) where {O1} = (first(off),)
@inline select_null_offset(
  i::Tuple{I1,I2,Vararg},
  off::Tuple{O1,O2,Vararg}
) where {I1,I2,O1,O2} =
  (Zero(), select_null_offset(Base.tail(i), Base.tail(off))...)
@inline select_null_offset(
  i::Tuple{NullStep,I2,Vararg},
  off::Tuple{O1,O2,Vararg}
) where {I2,O1,O2} =
  (first(off), select_null_offset(Base.tail(i), Base.tail(off))...)

@inline function gesp(
  ptr::AbstractStridedPointer,
  i::Tuple{Vararg{Union{NullStep,IntegerIndex}}}
)
  ioffset = _offset_index(i, offsets(ptr))
  offs = select_null_offset(i, offsets(ptr))
  similar_with_offset(ptr, gep(zero_offsets(ptr), ioffset), offs)
end
@inline gesp(
  ptr::AbstractStridedPointer,
  i::Tuple{NullStep,Vararg{NullStep,N}}
) where {N} = ptr
@inline gesp(ptr::AbstractStridedPointer, i::Tuple{Vararg{Any,N}}) where {N} =
  gesp(ptr, Tuple(CartesianVIndex(i)))#flatten

function vload_quote(
  ::Type{T},
  ::Type{I},
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int
) where {T<:NativeTypes,I<:Integer}
  T_sym = JULIA_TYPES[T]
  I_sym = JULIA_TYPES[I]
  vload_quote(T_sym, I_sym, ind_type, W, X, M, O, mask, align, rs)
end
function vload_quote(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int
)
  isbit = T_sym === :Bit
  if !isbit && W > 1
    sizeof_T = JULIA_TYPE_SIZE[T_sym]
    if W * sizeof_T > rs
      if ((T_sym === :Int64) | (T_sym === :UInt64)) && ((W * sizeof_T) == 2rs)
        return vload_trunc_quote(
          T_sym,
          I_sym,
          ind_type,
          W,
          X,
          M,
          O,
          mask,
          align,
          rs,
          :(_Vec{$W,$T_sym})
        )
        # else
        #   return vload_split_quote(W, sizeof_T, mask, align, rs, T_sym)
      end
    else
      return vload_quote(
        T_sym,
        I_sym,
        ind_type,
        W,
        X,
        M,
        O,
        mask,
        align,
        rs,
        :(_Vec{$W,$T_sym})
      )
    end
  end
  jtyp = isbit ? (isone(W) ? :Bool : mask_type_symbol(W)) : T_sym
  vload_quote(T_sym, I_sym, ind_type, W, X, M, O, mask, align, rs, jtyp)
  # jtyp_expr = Expr(:(.), :Base, QuoteNode(jtyp)) # reduce latency, hopefully
  # vload_quote(T_sym, I_sym, ind_type, W, X, M, O, mask, align, rs, jtyp_expr)
end
function vload_trunc_quote(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int,
  ret::Union{Symbol,Expr}
)
  call = vload_quote_llvmcall(
    T_sym,
    I_sym,
    ind_type,
    W,
    X,
    M,
    O,
    mask,
    align,
    rs,
    ret
  )
  call = Expr(:call, :%, call, Core.ifelse(T_sym === :Int64, :Int32, :UInt32))
  Expr(:block, Expr(:meta, :inline), call)
end
# function vload_split_quote(W::Int, sizeof_T::Int, mask::Bool, align::Bool, rs::Int, T_sym::Symbol)
#     D, r1 = divrem(W * sizeof_T, rs)
#     Wnew, r2 = divrem(W, D)
#     (iszero(r1) & iszero(r2)) || throw(ArgumentError("If loading more than a vector, Must load a multiple of the vector width."))
#     q = Expr(:block,Expr(:meta,:inline))
#     # ind_type = :StaticInt, :Integer, :Vec
#     push!(q.args, :(isplit = splitvectortotuple(StaticInt{$D}(), StaticInt{$Wnew}(), i)))
#     mask && push!(q.args, :(msplit = splitvectortotuple(StaticInt{$D}(), StaticInt{$Wnew}(), m)))
#     t = Expr(:tuple)
#     alignval = Expr(:call, align ? :True : :False)
#     for d ∈ 1:D
#         call = Expr(:call, :__vload, :ptr)
#         push!(call.args, Expr(:ref, :isplit, d))
#         mask && push!(call.args, Expr(:ref, :msplit, d))
#         push!(call.args, alignval, Expr(:call, Expr(:curly, :StaticInt, rs)))
#         v_d = Symbol(:v_, d)
#         push!(q.args, Expr(:(=), v_d, call))
#         push!(t.args, v_d)
#     end
#     push!(q.args, :(VecUnroll($t)::VecUnroll{$(D-1),$Wnew,$T_sym,Vec{$Wnew,$T_sym}}))
#     q
# end

@inline function _mask_scalar_load(
  ptr::Ptr{T},
  i::IntegerIndex,
  m::AbstractMask{1},
  ::A,
  ::StaticInt{RS}
) where {T,A,RS}
  Bool(m) ? __vload(ptr, i, A(), StaticInt{RS}()) : zero(T)
end
@inline function _mask_scalar_load(
  ptr::Ptr{T},
  m::AbstractMask{1},
  ::A,
  ::StaticInt{RS}
) where {T,A,RS}
  Bool(m) ? __vload(ptr, i, A(), StaticInt{RS}()) : zero(T)
end
@inline function _mask_scalar_load(
  ptr::Ptr{T},
  i::IntegerIndex,
  m::AbstractMask{W},
  ::A,
  ::StaticInt{RS}
) where {T,A,RS,W}
  s = __vload(ptr, i, A(), StaticInt{RS}())
  ifelse(
    m,
    _vbroadcast(StaticInt{W}(), s, StaticInt{RS}()),
    _vzero(StaticInt{W}(), T, StaticInt{RS}())
  )
end
@inline function _mask_scalar_load(
  ptr::Ptr{T},
  m::AbstractMask{W},
  ::A,
  ::StaticInt{RS}
) where {T,A,RS,W}
  s = __vload(ptr, A(), StaticInt{RS}())
  ifelse(
    m,
    _vbroadcast(StaticInt{W}(), s, StaticInt{RS}()),
    _vzero(StaticInt{W}(), T, StaticInt{RS}())
  )
end

function vload_quote_llvmcall(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int,
  ret::Union{Symbol,Expr}
)
  if mask && W == 1
    if M == O == 0
      return quote
        $(Expr(:meta, :inline))
        _mask_scalar_load(ptr, m, $(align ? :True : :False)(), StaticInt{$rs}())
      end
    else
      return quote
        $(Expr(:meta, :inline))
        _mask_scalar_load(
          ptr,
          i,
          m,
          $(align ? :True : :False)(),
          StaticInt{$rs}()
        )
      end
    end
  end

  decl, instrs, args, lret, largs, arg_syms = vload_quote_llvmcall_core(
    T_sym,
    I_sym,
    ind_type,
    W,
    X,
    M,
    O,
    mask,
    align,
    rs
  )

  return llvmcall_expr(
    decl,
    instrs,
    ret,
    args,
    lret,
    largs,
    arg_syms,
    true,
    true
  )
end
function vload_quote_llvmcall_core(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int
)
  sizeof_T = JULIA_TYPE_SIZE[T_sym]

  reverse_load = ((W > 1) & (X == -sizeof_T)) & (ind_type !== :Vec)
  if reverse_load
    X = sizeof_T
    O -= (W - 1) * sizeof_T
  end
  # if (X == -sizeof_T) & (!mask)
  #     return quote
  #         $(Expr(:meta,:inline))
  #         vload(ptr, i
  #     end
  # end
  sizeof_I = JULIA_TYPE_SIZE[I_sym]
  ibits = 8sizeof_I
  if W > 1 && ind_type !== :Vec
    X, Xr = divrem(X, sizeof_T)
    iszero(Xr) || throw(
      ArgumentError(
        "sizeof($T_sym) == $sizeof_T, but stride between vector loads is given as $X, which is not a positive integer multiple."
      )
    )
  end
  instrs, i = offset_ptr(T_sym, ind_type, '1', ibits, W, X, M, O, false, rs)
  grv = gep_returns_vector(W, X, M, ind_type)
  # considers booleans to only occupy 1 bit in memory, so they must be handled specially
  isbit = T_sym === :Bit
  if isbit
    # @assert !grv "gather's not are supported with `BitArray`s."
    mask = false # TODO: not this?
    # typ = "i$W"
    alignment = (align & (!grv)) ? cld(W, 8) : 1
    typ = "i1"
  else
    alignment =
      (align & (!grv)) ? _get_alignment(W, T_sym) : _get_alignment(0, T_sym)
    typ = LLVM_TYPES_SYM[T_sym]
  end

  decl = LOAD_SCOPE_TBAA
  dynamic_index = !(iszero(M) || ind_type === :StaticInt)

  vtyp = vtype(W, typ)
  if mask
    if reverse_load
      decl *= truncate_mask!(instrs, '1' + dynamic_index, W, 0, true) * "\n"
    else
      truncate_mask!(instrs, '1' + dynamic_index, W, 0, false)
    end
  end
  if grv
    loadinstr =
      "$vtyp @llvm.masked.gather." *
      suffix(W, T_sym) *
      '.' *
      ptr_suffix(W, T_sym)
    decl *= "declare $loadinstr(<$W x $typ*>, i32, <$W x i1>, $vtyp)"
    m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
    passthrough = mask ? "zeroinitializer" : "undef"
    push!(
      instrs,
      "%res = call $loadinstr(<$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m, $vtyp $passthrough)" *
      LOAD_SCOPE_TBAA_FLAGS
    )
  elseif mask
    suff = suffix(W, T_sym)
    loadinstr = "$vtyp @llvm.masked.load." * suff * ".p0" * suff
    decl *= "declare $loadinstr($vtyp*, i32, <$W x i1>, $vtyp)"
    push!(
      instrs,
      "%res = call $loadinstr($vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0, $vtyp zeroinitializer)" *
      LOAD_SCOPE_TBAA_FLAGS
    )
  else
    push!(
      instrs,
      "%res = load $vtyp, $vtyp* %ptr.$(i-1), align $alignment" *
      LOAD_SCOPE_TBAA_FLAGS
    )
  end
  if isbit
    lret = string('i', max(8, nextpow2(W)))
    if W > 1
      if reverse_load
        # isbit means mask is set to false, so we definitely need to declare `bitreverse`
        bitreverse = "i$(W) @llvm.bitreverse.i$(W)(i$(W))"
        decl *= "declare $bitreverse"
        resbit = "resbitreverse"
        push!(instrs, "%$(resbit) = call $bitreverse(i$(W) %res")
      else
        resbit = "res"
      end
      if W < 8
        push!(instrs, "%resint = bitcast <$W x i1> %$(resbit) to i$(W)")
        push!(instrs, "%resfinal = zext i$(W) %resint to i8")
      elseif ispow2(W)
        push!(instrs, "%resfinal = bitcast <$W x i1> %$(resbit) to i$(W)")
      else
        Wpow2 = nextpow2(W)
        push!(instrs, "%resint = bitcast <$W x i1> %$(resbit) to i$(W)")
        push!(instrs, "%resfinal = zext i$(W) %resint to i$(Wpow2)")
      end
    else
      push!(instrs, "%resfinal = zext i1 %res to i8")
    end
    push!(instrs, "ret $lret %resfinal")
  else
    lret = vtyp
    if reverse_load
      reversemask = '<' * join(map(x -> string("i32 ", W - x), 1:W), ", ") * '>'
      push!(
        instrs,
        "%resreversed = shufflevector $vtyp %res, $vtyp undef, <$W x i32> $reversemask"
      )
      push!(instrs, "ret $vtyp %resreversed")
    else
      push!(instrs, "ret $vtyp %res")
    end
  end
  args = Expr(:curly, :Tuple, Expr(:curly, :Ptr, T_sym))
  largs = String[JULIAPOINTERTYPE]
  arg_syms = Union{Symbol,Expr}[:ptr]
  if dynamic_index
    push!(arg_syms, :(data(i)))
    if ind_type === :Integer
      push!(args.args, I_sym)
      push!(largs, "i$(ibits)")
    else
      push!(args.args, :(_Vec{$W,$I_sym}))
      push!(largs, "<$W x i$(ibits)>")
    end
  end
  if mask
    push!(arg_syms, :(data(m)))
    push!(args.args, mask_type(nextpow2(W)))
    push!(largs, "i$(max(8,nextpow2(W)))")
  end
  return decl, join(instrs, "\n"), args, lret, largs, arg_syms
end
function vload_quote(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  rs::Int,
  ret::Union{Symbol,Expr}
)
  call = vload_quote_llvmcall(
    T_sym,
    I_sym,
    ind_type,
    W,
    X,
    M,
    O,
    mask,
    align,
    rs,
    ret
  )
  if (W > 1) & (T_sym === :Bit)
    call = Expr(:call, Expr(:curly, :Mask, W), call)
  end
  Expr(:block, Expr(:meta, :inline), call)
end

# vload_quote(T, ::Type{I}, ind_type::Symbol, W::Int, X, M, O, mask, align = false)

# ::Type{T}, ::Type{I}, ind_type::Symbol, W::Int, X::Int, M::Int, O::Int, mask::Bool, align::Bool, rs::Int

function index_summary(::Type{StaticInt{N}}) where {N}
  #I,    ind_type, W, X, M, O
  Int, :StaticInt, 1, 1, 0, N
end
function index_summary(::Type{I}) where {I<:IntegerTypesHW}
  #I, ind_type, W, X, M, O
  I, :Integer, 1, 1, 1, 0
end
function index_summary(::Type{Vec{W,I}}) where {W,I<:IntegerTypes}
  #I, ind_type, W, X, M, O
  I, :Vec, W, 1, 1, 0
end
function index_summary(::Type{MM{W,X,I}}) where {W,X,I<:IntegerTypes}
  #I, ind_type, W,  X, M, O
  IT, ind_type, _, __, M, O = index_summary(I)
  # inherit from parent, replace `W` and `X`
  IT, ind_type, W, X, M, O
end
function index_summary(
  ::Type{LazyMulAdd{LMAM,LMAO,LMAI}}
) where {LMAM,LMAO,LMAI}
  I, ind_type, W, X, M, O = index_summary(LMAI)
  I, ind_type, W, X * LMAM, M * LMAM, LMAO + O * LMAM
end

# no index, no mask
@generated function __vload(
  ptr::Ptr{T},
  ::A,
  ::StaticInt{RS}
) where {T<:NativeTypes,A<:StaticBool,RS}
  vload_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, A === True, RS)
end
# no index, mask
@generated function __vload(
  ptr::Ptr{T},
  ::A,
  m::AbstractMask,
  ::StaticInt{RS}
) where {T<:NativeTypesExceptFloat16,A<:StaticBool,RS}
  vload_quote(T, Int, :StaticInt, 1, 1, 0, 0, true, A === True, RS)
end
# index, no mask
@generated function __vload(
  ptr::Ptr{T},
  i::I,
  ::A,
  ::StaticInt{RS}
) where {A<:StaticBool,T<:NativeTypes,I<:Index,RS}
  IT, ind_type, W, X, M, O = index_summary(I)
  vload_quote(T, IT, ind_type, W, X, M, O, false, A === True, RS)
end
# index, mask
@generated function __vload(
  ptr::Ptr{T},
  i::I,
  m::AbstractMask,
  ::A,
  ::StaticInt{RS}
) where {A<:StaticBool,T<:NativeTypesExceptFloat16,I<:Index,RS}
  IT, ind_type, W, X, M, O = index_summary(I)
  vload_quote(T, IT, ind_type, W, X, M, O, true, A === True, RS)
end

# Float16 with mask
@inline function __vload(
  ptr::Ptr{Float16},
  ::A,
  m::AbstractMask,
  ::StaticInt{RS}
) where {A<:StaticBool,RS}
  reinterpret(
    Float16,
    __vload(reinterpret(Ptr{Int16}, ptr), A(), m, StaticInt{RS}())
  )
end
@inline function __vload(
  ptr::Ptr{Float16},
  i::I,
  m::AbstractMask,
  ::A,
  ::StaticInt{RS}
) where {A<:StaticBool,I<:Index,RS}
  reinterpret(
    Float16,
    __vload(reinterpret(Ptr{Int16}, ptr), i, m, A(), StaticInt{RS}())
  )
end

@inline function _vload_scalar(
  ptr::Ptr{Bit},
  i::Union{Integer,StaticInt},
  ::A,
  ::StaticInt{RS}
) where {RS,A<:StaticBool}
  d = i >> 3
  r = i & 7
  u = __vload(Base.unsafe_convert(Ptr{UInt8}, ptr), d, A(), StaticInt{RS}())
  (u >> r) % Bool
end
@inline function __vload(
  ptr::Ptr{Bit},
  i::IntegerTypesHW,
  ::A,
  ::StaticInt{RS}
) where {A<:StaticBool,RS}
  _vload_scalar(ptr, i, A(), StaticInt{RS}())
end
# avoid ambiguities
@inline __vload(
  ptr::Ptr{Bit},
  ::StaticInt{N},
  ::A,
  ::StaticInt{RS}
) where {N,A<:StaticBool,RS} =
  _vload_scalar(ptr, StaticInt{N}(), A(), StaticInt{RS}())

# Entry points, `vload` and `vloada`
# No index, so forward straight to `__vload`
@inline vload(ptr::AbstractStridedPointer) =
  __vload(ptr, False(), register_size())
@inline vloada(ptr::AbstractStridedPointer) =
  __vload(ptr, True(), register_size())
# Index, so forward to `_vload` to linearize.
@inline vload(ptr::AbstractStridedPointer, i::Union{Tuple,Unroll}) =
  _vload(ptr, i, False(), register_size())
@inline vloada(ptr::AbstractStridedPointer, i::Union{Tuple,Unroll}) =
  _vload(ptr, i, True(), register_size())
@inline vload(
  ptr::AbstractStridedPointer,
  i::Union{Tuple,Unroll},
  m::Union{AbstractMask,Bool,VecUnroll}
) = _vload(ptr, i, m, False(), register_size())
@inline vloada(
  ptr::AbstractStridedPointer,
  i::Union{Tuple,Unroll},
  m::Union{AbstractMask,Bool,VecUnroll}
) = _vload(ptr, i, m, True(), register_size())

@inline function __vload(
  ptr::Ptr{T},
  i::Number,
  b::Bool,
  ::A,
  ::StaticInt{RS}
) where {T,A<:StaticBool,RS}
  b ? __vload(ptr, i, A(), StaticInt{RS}()) : zero(T)
end
@inline vwidth_from_ind(i::Tuple) =
  vwidth_from_ind(i, StaticInt(1), StaticInt(0))
@inline vwidth_from_ind(
  i::Tuple{},
  ::StaticInt{W},
  ::StaticInt{U}
) where {W,U} = (StaticInt{W}(), StaticInt{U}())
@inline function vwidth_from_ind(
  i::Tuple{<:AbstractSIMDVector{W},Vararg},
  ::Union{StaticInt{1},StaticInt{W}},
  ::StaticInt{U}
) where {W,U}
  vwidth_from_ind(Base.tail(i), StaticInt{W}(), StaticInt{W}(U))
end
@inline function vwidth_from_ind(
  i::Tuple{<:VecUnroll{U,W},Vararg},
  ::Union{StaticInt{1},StaticInt{W}},
  ::Union{StaticInt{0},StaticInt{U}}
) where {U,W}
  vwidth_from_ind(Base.tail(i), StaticInt{W}(), StaticInt{W}(U))
end

@inline function _vload(
  ptr::Ptr{T},
  i::Tuple,
  b::Bool,
  ::A,
  ::StaticInt{RS}
) where {T,A<:StaticBool,RS}
  if b
    _vload(ptr, i, A(), StaticInt{RS}())
  else
    zero_init(T, vwidth_from_ind(i), StaticInt{RS}())
  end
end
@inline function _vload(
  ptr::Ptr{T},
  i::Unroll{AU,F,N,AV,W,M,X,I},
  b::Bool,
  ::A,
  ::StaticInt{RS}
) where {T,AU,F,N,AV,W,M,X,I,A<:StaticBool,RS}
  m = max_mask(Val{W}()) & b
  _vload(ptr, i, m, A(), StaticInt{RS}())
end
@inline function __vload(
  ptr::Ptr{T},
  i::I,
  m::Bool,
  ::A,
  ::StaticInt{RS}
) where {T,A<:StaticBool,RS,I<:IntegerIndex}
  m ? __vload(ptr, i, A(), StaticInt{RS}()) : zero(T)
end
@inline function __vload(
  ptr::Ptr{T},
  i::I,
  m::Bool,
  ::A,
  ::StaticInt{RS}
) where {T,A<:StaticBool,RS,W,I<:VectorIndex{W}}
  _m = max_mask(Val{W}()) & m
  __vload(ptr, i, _m, A(), StaticInt{RS}())
end

function vstore_quote(
  ::Type{T},
  ::Type{I},
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  noalias::Bool,
  nontemporal::Bool,
  rs::Int
) where {T<:NativeTypes,I<:Integer}
  T_sym = JULIA_TYPES[T]
  I_sym = JULIA_TYPES[I]
  vstore_quote(
    T_sym,
    I_sym,
    ind_type,
    W,
    X,
    M,
    O,
    mask,
    align,
    noalias,
    nontemporal,
    rs
  )
end
function vstore_quote(
  T_sym::Symbol,
  I_sym::Symbol,
  ind_type::Symbol,
  W::Int,
  X::Int,
  M::Int,
  O::Int,
  mask::Bool,
  align::Bool,
  noalias::Bool,
  nontemporal::Bool,
  rs::Int
)
  sizeof_T = JULIA_TYPE_SIZE[T_sym]
  sizeof_I = JULIA_TYPE_SIZE[I_sym]
  ibits = 8sizeof_I

  reverse_store =
    (((W > 1) & (X == -sizeof_T)) & (ind_type !== :Vec)) & (T_sym ≢ :Bit) # TODO: check if `T_sym` can actually be `Bit`; don't we cast?
  if reverse_store
    X = sizeof_T
    O -= (W - 1) * sizeof_T
  end

  if W > 1 && ind_type !== :Vec
    X, Xr = divrem(X, sizeof_T)
    iszero(Xr) || throw(
      ArgumentError(
        "sizeof($T_sym) == $sizeof_T, but stride between vector loads is given as $X, which is not a positive integer multiple."
      )
    )
  end
  instrs, i = offset_ptr(T_sym, ind_type, '2', ibits, W, X, M, O, false, rs)

  grv = gep_returns_vector(W, X, M, ind_type)
  align != nontemporal # should I do this?
  alignment =
    (align & (!grv)) ? _get_alignment(W, T_sym) : _get_alignment(0, T_sym)

  decl = noalias ? SCOPE_METADATA * TBAA_STR : TBAA_STR
  metadata = noalias ? STORE_SCOPE_FLAGS * TBAA_FLAGS : TBAA_FLAGS
  dynamic_index = !(iszero(M) || ind_type === :StaticInt)

  typ = LLVM_TYPES_SYM[T_sym]
  lret = vtyp = vtype(W, typ)
  if mask
    if reverse_store
      decl *= truncate_mask!(instrs, '2' + dynamic_index, W, 0, true) * "\n"
    else
      truncate_mask!(instrs, '2' + dynamic_index, W, 0, false)
    end
  end
  if reverse_store
    reversemask = '<' * join(map(x -> string("i32 ", W - x), 1:W), ", ") * '>'
    push!(
      instrs,
      "%argreversed = shufflevector $vtyp %1, $vtyp undef, <$W x i32> $reversemask"
    )
    argtostore = "%argreversed"
  else
    argtostore = "%1"
  end
  if grv
    storeinstr =
      "void @llvm.masked.scatter." *
      suffix(W, T_sym) *
      '.' *
      ptr_suffix(W, T_sym)
    decl *= "declare $storeinstr($vtyp, <$W x $typ*>, i32, <$W x i1>)"
    m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
    push!(
      instrs,
      "call $storeinstr($vtyp $(argtostore), <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m)" *
      metadata
    )
    # push!(instrs, "call $storeinstr($vtyp $(argtostore), <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m)")
  elseif mask
    suff = suffix(W, T_sym)
    storeinstr = "void @llvm.masked.store." * suff * ".p0" * suff
    decl *= "declare $storeinstr($vtyp, $vtyp*, i32, <$W x i1>)"
    push!(
      instrs,
      "call $storeinstr($vtyp $(argtostore), $vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0)" *
      metadata
    )
  elseif nontemporal
    push!(
      instrs,
      "store $vtyp $(argtostore), $vtyp* %ptr.$(i-1), align $alignment, !nontemporal !{i32 1}" *
      metadata
    )
  else
    push!(
      instrs,
      "store $vtyp $(argtostore), $vtyp* %ptr.$(i-1), align $alignment" *
      metadata
    )
  end
  push!(instrs, "ret void")
  ret = :Cvoid
  lret = "void"
  ptrtyp = Expr(:curly, :Ptr, T_sym)
  args = if W > 1
    Expr(
      :curly,
      :Tuple,
      ptrtyp,
      Expr(:curly, :NTuple, W, Expr(:curly, :VecElement, T_sym))
    )
  else
    Expr(:curly, :Tuple, ptrtyp, T_sym)
  end
  largs = String[JULIAPOINTERTYPE, vtyp]
  arg_syms = Union{Symbol,Expr}[:ptr, Expr(:call, :data, :v)]
  if dynamic_index
    push!(arg_syms, :(data(i)))
    if ind_type === :Integer
      push!(args.args, I_sym)
      push!(largs, "i$(ibits)")
    else
      push!(args.args, :(_Vec{$W,$I_sym}))
      push!(largs, "<$W x i$(ibits)>")
    end
  end
  if mask
    push!(arg_syms, :(data(m)))
    push!(args.args, mask_type(W))
    push!(largs, "i$(max(8,nextpow2(W)))")
  end
  llvmcall_expr(
    decl,
    join(instrs, "\n"),
    ret,
    args,
    lret,
    largs,
    arg_syms,
    false,
    true
  )
end

# no index, no mask, scalar store
@generated function __vstore!(
  ptr::Ptr{T},
  v::VT,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  VT<:NativeTypes,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  if VT !== T
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(ptr, convert($T, v), $A(), $S(), $NT(), StaticInt{$RS}()))
    )
  end
  vstore_quote(
    T,
    Int,
    :StaticInt,
    1,
    1,
    0,
    0,
    false,
    A === True,
    S === True,
    NT === True,
    RS
  )
end
# no index, no mask, vector store
@generated function __vstore!(
  ptr::Ptr{T},
  v::V,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  W,
  VT<:NativeTypes,
  V<:AbstractSIMDVector{W,VT},
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  if V !== Vec{W,T}
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(
        ptr,
        convert(Vec{$W,$T}, v),
        $A(),
        $S(),
        $NT(),
        StaticInt{$RS}()
      ))
    )
  end
  vstore_quote(
    T,
    Int,
    :StaticInt,
    W,
    sizeof(T),
    0,
    0,
    false,
    A === True,
    S === True,
    NT === True,
    RS
  )
end
# index, no mask, scalar store
@generated function __vstore!(
  ptr::Ptr{T},
  v::VT,
  i::I,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  VT<:NativeTypes,
  I<:Index,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  IT, ind_type, W, X, M, O = index_summary(I)
  if VT !== T || W > 1
    if W > 1
      return Expr(
        :block,
        Expr(:meta, :inline),
        :(__vstore!(
          ptr,
          convert(Vec{$W,$T}, v),
          i,
          $A(),
          $S(),
          $NT(),
          StaticInt{$RS}()
        ))
      )
    else
      return Expr(
        :block,
        Expr(:meta, :inline),
        :(__vstore!(
          ptr,
          convert($T, v),
          i,
          $A(),
          $S(),
          $NT(),
          StaticInt{$RS}()
        ))
      )
    end
  end
  vstore_quote(
    T,
    IT,
    ind_type,
    W,
    X,
    M,
    O,
    false,
    A === True,
    S === True,
    NT === True,
    RS
  )
end
# index, no mask, vector store
@generated function __vstore!(
  ptr::Ptr{T},
  v::V,
  i::I,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  W,
  VT<:NativeTypes,
  V<:AbstractSIMDVector{W,VT},
  I<:Index,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  if V !== Vec{W,T}
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(
        ptr,
        convert(Vec{$W,$T}, v),
        i,
        $A(),
        $S(),
        $NT(),
        StaticInt{$RS}()
      ))
    )
  end
  IT, ind_type, _W, X, M, O = index_summary(I)
  # don't want to require vector indices...
  (W == _W || _W == 1) || throw(
    ArgumentError(
      "Vector width: $W, index width: $(_W). They must either be equal, or index width == 1."
    )
  )
  if (W != _W) & (_W == 1)
    X *= sizeof(T)
  end
  vstore_quote(
    T,
    IT,
    ind_type,
    W,
    X,
    M,
    O,
    false,
    A === True,
    S === True,
    NT === True,
    RS
  )
end
# index, mask, scalar store
@generated function __vstore!(
  ptr::Ptr{T},
  v::VT,
  i::I,
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  T<:NativeTypesExceptBit,
  VT<:NativeTypes,
  I<:Index,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  IT, ind_type, _W, X, M, O = index_summary(I)
  (W == _W || _W == 1) || throw(
    ArgumentError(
      "Vector width: $W, index width: $(_W). They must either be equal, or index width == 1."
    )
  )
  if W == 1
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(
        Bool(m) && __vstore!(
          ptr,
          convert($T, v),
          data(i),
          $A(),
          $S(),
          $NT(),
          StaticInt{$RS}()
        )
      )
    )
  else
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(
        ptr,
        convert(Vec{$W,$T}, v),
        i,
        m,
        $A(),
        $S(),
        $NT(),
        StaticInt{$RS}()
      ))
    )
  end
  # vstore_quote(T, IT, ind_type, W, X, M, O, true, A===True, S===True, NT===True, RS)
end
# no index, mask, vector store
@generated function __vstore!(
  ptr::Ptr{T},
  v::V,
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  W,
  VT<:NativeTypes,
  V<:AbstractSIMDVector{W,VT},
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  if W == 1
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(
        Bool(m) && __vstore!(
          ptr,
          convert($T, v),
          data(i),
          $A(),
          $S(),
          $NT(),
          StaticInt{$RS}()
        )
      )
    )
  elseif V !== Vec{W,T}
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(
        ptr,
        convert(Vec{$W,$T}, v),
        m,
        $A(),
        $S(),
        $NT(),
        StaticInt{$RS}()
      ))
    )
  end
  vstore_quote(
    T,
    Int,
    :StaticInt,
    W,
    sizeof(T),
    0,
    0,
    true,
    A === True,
    S === True,
    NT === True,
    RS
  )
end
# index, mask, vector store
@generated function __vstore!(
  ptr::Ptr{T},
  v::V,
  i::I,
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  W,
  VT<:NativeTypes,
  V<:AbstractSIMDVector{W,VT},
  I<:Index,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  if W == 1
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(
        Bool(m) && __vstore!(
          ptr,
          convert($T, v),
          data(i),
          $A(),
          $S(),
          $NT(),
          StaticInt{$RS}()
        )
      )
    )
  elseif V !== Vec{W,T}
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(__vstore!(
        ptr,
        convert(Vec{$W,$T}, v),
        i,
        m,
        $A(),
        $S(),
        $NT(),
        StaticInt{$RS}()
      ))
    )
  end
  IT, ind_type, _W, X, M, O = index_summary(I)
  (W == _W || _W == 1) || throw(
    ArgumentError(
      "Vector width: $W, index width: $(_W). They must either be equal, or index width == 1."
    )
  )
  if (W != _W) & (_W == 1)
    X *= sizeof(T)
  end
  vstore_quote(
    T,
    IT,
    ind_type,
    W,
    X,
    M,
    O,
    true,
    A === True,
    S === True,
    NT === True,
    RS
  )
end

# no index, mask, vector store
@generated function __vstore!(
  ptr::Ptr{Float16},
  v::V,
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  V<:AbstractSIMDVector{W,Float16},
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(
    reinterpret(Ptr{Int16}, ptr),
    reinterpret(Int16, v),
    m,
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end
# index, mask, vector store
@inline function __vstore!(
  ptr::Ptr{Float16},
  v::V,
  i::I,
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  V<:AbstractSIMDVector{W,Float16},
  I<:Index,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(
    reinterpret(Ptr{Int16}, ptr),
    reinterpret(Int16, v),
    i,
    m,
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end

# BitArray stores
@inline function __vstore!(
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {B<:Union{Bit,Bool},W,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  __vstore!(
    Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr),
    tounsigned(v),
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end
@inline bytetobit(x::Union{Vec,VecUnroll}) = x >> 3
@inline bytetobit(x::Union{MM,LazyMulAdd}) = data(x) >> 3
@inline bytetobit(x::Union{MM,LazyMulAdd,Unroll}) = data(x) >> 3

@inline function __vstore!(
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  i::VectorIndex{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {B<:Union{Bit,Bool},W,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  __vstore!(
    Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr),
    tounsigned(v),
    bytetobit(i),
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end
@inline function __vstore!(
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  i::VectorIndex{W},
  m::AbstractMask,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {B<:Union{Bit,Bool},W,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  ishift = data(i) >> 3
  p = Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr)
  u =
    bitselect(data(m), __vload(p, ishift, A(), StaticInt{RS}()), tounsigned(v))
  __vstore!(p, u, ishift, A(), S(), NT(), StaticInt{RS}())
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  B<:Union{Bit,Bool},
  F<:Function,
  W,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(
    f,
    Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr),
    tounsigned(v),
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  i::VectorIndex{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  B<:Union{Bit,Bool},
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(
    f,
    Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr),
    tounsigned(v),
    data(i) >> 3,
    A(),
    S(),
    NT(),
    StaticInt{RS}()
  )
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{Bit},
  v::AbstractSIMDVector{W,B},
  i::VectorIndex{W},
  m::AbstractMask,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  B<:Union{Bit,Bool},
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  ishift = data(i) >> 3
  p = Base.unsafe_convert(Ptr{mask_type(StaticInt{W}())}, ptr)
  u =
    bitselect(data(m), __vload(p, ishift, A(), StaticInt{RS}()), tounsigned(v))
  __vstore!(f, p, u, ishift, A(), S(), NT(), StaticInt{RS}())
end

# Can discard `f` if we have a vector index
@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::AbstractSIMDVector{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS,
  W
}
  __vstore!(ptr, f(v), A(), S(), NT(), StaticInt{RS}())
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::AbstractSIMDVector{W},
  i::IntegerIndex,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS,
  W
}
  __vstore!(ptr, f(v), i, A(), S(), NT(), StaticInt{RS}())
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::AbstractSIMDVector{W},
  i::VectorIndex{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  # __vstore!(ptr, convert(Vec{W,T}, v), i, A(), S(), NT(), StaticInt{RS}()) # discard `f`
  __vstore!(ptr, v, i, A(), S(), NT(), StaticInt{RS}()) # discard `f`
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::AbstractSIMDVector{W},
  i::VectorIndex{W},
  m::AbstractMask{W},
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  W,
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  # __vstore!(ptr, convert(Vec{W,T}, v), i, m, A(), S(), NT(), StaticInt{RS}())
  __vstore!(ptr, f(v), i, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function __vstore!(
  f::F,
  ptr::Ptr,
  v::AbstractSIMDVector,
  i::Index,
  m::Bool,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {F<:Function,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  # __vstore!(ptr, convert(Vec{W,T}, v), i, m, A(), S(), NT(), StaticInt{RS}())
  m && __vstore!(f, ptr, v, i, A(), S(), NT(), StaticInt{RS}())
end

@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::NativeTypes,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(ptr, v, A(), S(), NT(), StaticInt{RS}())
end
@inline function __vstore!(
  f::F,
  ptr::Ptr{T},
  v::NativeTypes,
  i::IntegerIndex,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {
  T<:NativeTypesExceptBit,
  F<:Function,
  A<:StaticBool,
  S<:StaticBool,
  NT<:StaticBool,
  RS
}
  __vstore!(ptr, v, i, A(), S(), NT(), StaticInt{RS}())
end

for (store, align, alias, nontemporal) ∈ [
  (:vstore!, False(), False(), False()),
  (:vstorea!, True(), False(), False()),
  (:vstorent!, True(), False(), True()),
  (:vnoaliasstore!, False(), True(), False()),
  (:vnoaliasstorea!, True(), True(), False()),
  (:vnoaliasstorent!, True(), True(), True())
]
  @eval begin
    @inline function $store(ptr::AbstractStridedPointer, v)
      __vstore!(ptr, v, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll}
    )
      _vstore!(ptr, v, i, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll},
      m::Union{AbstractMask,VecUnroll}
    )
      _vstore!(ptr, v, i, m, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll},
      b::Bool
    )
      b && _vstore!(ptr, v, i, $align, $alias, $nontemporal, register_size())
    end

    @inline function $store(
      f::F,
      ptr::AbstractStridedPointer,
      v
    ) where {F<:Function}
      __vstore!(f, ptr, v, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      f::F,
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll}
    ) where {F<:Function}
      _vstore!(f, ptr, v, i, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      f::F,
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll},
      m::Union{AbstractMask,VecUnroll}
    ) where {F<:Function}
      _vstore!(f, ptr, v, i, m, $align, $alias, $nontemporal, register_size())
    end
    @inline function $store(
      f::F,
      ptr::AbstractStridedPointer,
      v,
      i::Union{Tuple,Unroll},
      b::Bool
    ) where {F<:Function}
      b && _vstore!(f, ptr, v, i, $align, $alias, $nontemporal, register_size())
    end
  end
end
@inline function __vstore!(
  ptr::Ptr,
  v,
  i,
  b::Bool,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS}
  b && __vstore!(ptr, v, i, A(), S(), NT(), StaticInt{RS}())
  nothing
end

@generated function prefetch(ptr::Ptr{Cvoid}, ::Val{L}, ::Val{R}) where {L,R}
  L ∈ (0, 1, 2, 3) || throw(
    ArgumentError(
      "Prefetch intrinsic requires a locality argument of 0, 1, 2, or 3, but received $L."
    )
  )
  R ∈ (0, 1) || throw(
    ArgumentError(
      "Prefetch intrinsic requires a read/write argument of 0, 1, but received $R."
    )
  )
  decl = "declare void @llvm.prefetch(i8*, i32, i32, i32)"
  instrs = """
      %addr = inttoptr $JULIAPOINTERTYPE %0 to i8*
      call void @llvm.prefetch(i8* %addr, i32 $R, i32 $L, i32 1)
      ret void
  """
  llvmcall_expr(
    decl,
    instrs,
    :Cvoid,
    :(Tuple{Ptr{Cvoid}}),
    "void",
    [JULIAPOINTERTYPE],
    [:ptr],
    false,
    true
  )
end
@inline prefetch(ptr::Ptr{T}, ::Val{L}, ::Val{R}) where {T,L,R} =
  prefetch(Base.unsafe_convert(Ptr{Cvoid}, ptr), Val{L}(), Val{R}())

@inline function prefetch(
  ptr::Union{AbstractStridedPointer,Ptr},
  i,
  ::Val{Locality},
  ::Val{ReadOrWrite}
) where {Locality,ReadOrWrite}
  prefetch(gep(ptr, i), Val{Locality}(), Val{ReadOrWrite}())
end
@inline prefetch(ptr) = nothing
@inline prefetch(ptr::Ptr) =
  prefetch(reinterpret(Ptr{Cvoid}, ptr), Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, ::Val{L}) where {L} =
  prefetch(ptr, Val{L}(), Val{0}())
@inline prefetch(ptr::Ptr, i) = prefetch(ptr, i, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, i, ::Val{L}) where {L} =
  prefetch(ptr, i, Val{L}(), Val{0}())

@inline prefetch0(x, i) =
  prefetch(gep(stridedpointer(x), (data(i),)), Val{3}(), Val{0}())
@inline prefetch0(x, I::Tuple) =
  prefetch(gep(stridedpointer(x), data.(I)), Val{3}(), Val{0}())
@inline prefetch0(x, i, j) =
  prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{3}(), Val{0}())
@inline prefetch1(x, i) =
  prefetch(gep(stridedpointer(x), (data(i),)), Val{2}(), Val{0}())
@inline prefetch1(x, i, j) =
  prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{2}(), Val{0}())
@inline prefetch2(x, i) =
  prefetch(gep(stridedpointer(x), (data(i),)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) =
  prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{1}(), Val{0}())

@generated function lifetime_start!(ptr::Ptr{T}, ::Val{L}) where {L,T}
  ptyp = LLVM_TYPES[T]
  decl = "declare void @llvm.lifetime.start(i64, $ptyp* nocapture)"
  instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to $ptyp*\ncall void @llvm.lifetime.start(i64 $L, $ptyp* %ptr)\nret void"
  llvmcall_expr(
    decl,
    instrs,
    :Cvoid,
    :(Tuple{Ptr{$T}}),
    "void",
    [JULIAPOINTERTYPE],
    [:ptr],
    false,
    true
  )
end
@generated function lifetime_end!(ptr::Ptr{T}, ::Val{L}) where {L,T}
  ptyp = LLVM_TYPES[T]
  decl = "declare void @llvm.lifetime.end(i64, $ptyp* nocapture)"
  instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to $ptyp*\ncall void @llvm.lifetime.end(i64 $L, $ptyp* %ptr)\nret void"
  llvmcall_expr(
    decl,
    instrs,
    :Cvoid,
    :(Tuple{Ptr{$T}}),
    "void",
    [JULIAPOINTERTYPE],
    [:ptr],
    false,
    true
  )
end
# @generated function lifetime_start!(ptr::Ptr{T}, ::Val{L}) where {L,T}
#   decl = "declare void @llvm.lifetime.start(i64, i8* nocapture)"
#   instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to i8*\ncall void @llvm.lifetime.start(i64 $(L*sizeof(T)), i8* %ptr)\nret void"
#   llvmcall_expr(decl, instrs, :Cvoid, :(Tuple{Ptr{$T}}), "void", [JULIAPOINTERTYPE], [:ptr], false, true)
# end
# @generated function lifetime_end!(ptr::Ptr{T}, ::Val{L}) where {L,T}
#   decl = "declare void @llvm.lifetime.end(i64, i8* nocapture)"
#   instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to i8*\ncall void @llvm.lifetime.end(i64 $(L*sizeof(T)), i8* %ptr)\nret void"
#   llvmcall_expr(decl, instrs, :Cvoid, :(Tuple{Ptr{$T}}), "void", [JULIAPOINTERTYPE], [:ptr], false, true)
# end

@inline lifetime_start!(ptr::Ptr) = lifetime_start!(ptr, Val{-1}())
@inline lifetime_end!(ptr::Ptr) = lifetime_end!(ptr, Val{-1}())
# Fallback is to do nothing. Intention is (e.g.) for PaddedMatrices/StackPointers.
@inline lifetime_start!(::Any) = nothing
@inline lifetime_end!(::Any) = nothing

@generated function compressstore!(
  ptr::Ptr{T},
  v::Vec{W,T},
  mask::AbstractMask{W,U}
) where {W,T<:NativeTypes,U<:Unsigned}
  typ = LLVM_TYPES[T]
  vtyp = "<$W x $typ>"
  mtyp_input = LLVM_TYPES[U]
  mtyp_trunc = "i$W"
  instrs = String["%ptr = inttoptr $JULIAPOINTERTYPE %1 to $typ*"]
  truncate_mask!(instrs, '2', W, 0)
  decl = "declare void @llvm.masked.compressstore.$(suffix(W,T))($vtyp, $typ*, <$W x i1>)"
  push!(
    instrs,
    "call void @llvm.masked.compressstore.$(suffix(W,T))($vtyp %0, $typ* %ptr, <$W x i1> %mask.0)\nret void"
  )
  llvmcall_expr(
    decl,
    join(instrs, "\n"),
    :Cvoid,
    :(Tuple{_Vec{$W,$T},Ptr{$T},$U}),
    "void",
    [vtyp, JULIAPOINTERTYPE, "i$(8sizeof(U))"],
    [:(data(v)), :ptr, :(data(mask))],
    false,
    true
  )
end

@generated function expandload(
  ptr::Ptr{T},
  mask::AbstractMask{W,U}
) where {W,T<:NativeTypes,U<:Unsigned}
  typ = LLVM_TYPES[T]
  vtyp = "<$W x $typ>"
  vptrtyp = "<$W x $typ*>"
  mtyp_input = LLVM_TYPES[U]
  mtyp_trunc = "i$W"
  instrs = String[]
  push!(instrs, "%ptr = inttoptr $JULIAPOINTERTYPE %0 to $typ*")
  if mtyp_input == mtyp_trunc
    push!(instrs, "%mask = bitcast $mtyp_input %1 to <$W x i1>")
  else
    push!(instrs, "%masktrunc = trunc $mtyp_input %1 to $mtyp_trunc")
    push!(instrs, "%mask = bitcast $mtyp_trunc %masktrunc to <$W x i1>")
  end
  decl = "declare $vtyp @llvm.masked.expandload.$(suffix(W,T))($typ*, <$W x i1>, $vtyp)"
  push!(
    instrs,
    "%res = call $vtyp @llvm.masked.expandload.$(suffix(W,T))($typ* %ptr, <$W x i1> %mask, $vtyp zeroinitializer)\nret $vtyp %res"
  )
  llvmcall_expr(
    decl,
    join(instrs, "\n"),
    :(_Vec{$W,$T}),
    :(Tuple{Ptr{$T},$U}),
    vtyp,
    [JULIAPOINTERTYPE, "i$(8sizeof(U))"],
    [:ptr, :(data(mask))],
    false,
    true
  )
end

# fallback definitions
@generated function __vload(
  p::Ptr{T},
  i::Index,
  ::A,
  ::StaticInt{RS}
) where {T,A,RS}
  if Base.allocatedinline(T)
    Expr(:block, Expr(:meta, :inline), :(unsafe_load(p + convert(Int, i))))
  else
    Expr(
      :block,
      Expr(:meta, :inline),
      :(ccall(
        :jl_value_ptr,
        Ref{$T},
        (Ptr{Cvoid},),
        unsafe_load(Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p) + convert(Int, i))
      ))
    )
  end
end
@generated function __vstore!(
  p::Ptr{T},
  v::T,
  i::Index,
  ::A,
  ::S,
  ::NT,
  ::StaticInt{RS}
) where {T,A,S,NT,RS}
  if Base.allocatedinline(T)
    Expr(
      :block,
      Expr(:meta, :inline),
      :(unsafe_store!(p + convert(Int, i), v); return nothing)
    )
  else
    Expr(
      :block,
      Expr(:meta, :inline),
      :(
        unsafe_store!(
          Base.unsafe_convert(Ptr{Ptr{Cvoid}}, p) + convert(Int, i),
          Base.pointer_from_objref(v)
        );
        return nothing
      )
    )
  end
end

# @inline function Base.getindex(A::AbstractArray{<:Number}, i::Unroll)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(sp, i)
#   end
#   return x
# end
# @inline function Base.getindex(A::ArrayInterface.AbstractArray2{<:Number}, i::Unroll)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(sp, i)
#   end
#   return x
# end
# @inline function Base.setindex!(A::AbstractArray{<:Number}, v, i::Unroll)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     vstore!(sp, v, i)
#   end
#   return v
# end
# @inline function Base.setindex!(A::ArrayInterface.AbstractArray2{<:Number}, v, i::Unroll)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     vstore!(sp, v, i)
#   end
#   return v
# end

# @inline function Base.getindex(A::AbstractArray{<:Number}, i::AbstractSIMD)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(zero_offsets(sp), (vsub_nw(i,ArrayInterface.offset1(A)),))
#   end
#   return x
# end
# @inline function Base.getindex(A::ArrayInterface.AbstractArray2{<:Number}, i::AbstractSIMD)
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(zero_offsets(sp), (vsub_nw(i,ArrayInterface.offset1(A)),))
#   end
#   return x
# end

# @inline function Base.getindex(A::AbstractArray{<:Number}, i::Vararg{Union{Integer,AbstractSIMD},K}) where {K}
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(sp, i)
#   end
#   return x
# end
# @inline function Base.getindex(A::ArrayInterface.AbstractArray2{<:Number}, i::Vararg{Union{Integer,AbstractSIMD},K}) where {K}
#   sp, pres = stridedpointer_preserve(A)
#   GC.@preserve pres begin
#     x = vload(sp, i)
#   end
#   return x
# end
