module VectorizationBase
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
  @eval Base.Experimental.@max_methods 1
end
import StaticArrayInterface, LinearAlgebra, Libdl, IfElse, LayoutPointers
const ArrayInterface = StaticArrayInterface
using StaticArrayInterface:
  contiguous_axis,
  contiguous_axis_indicator,
  contiguous_batch_size,
  stride_rank,
  device,
  CPUPointer,
  CPUIndex,
  known_length,
  known_first,
  known_last,
  static_size,
  static_strides,
  offsets,
  static_first,
  static_last,
  static_length
import IfElse: ifelse

using CPUSummary:
  cache_type,
  num_cache,
  num_cache_levels,
  num_cores,
  num_l1cache,
  num_l2cache,
  cache_associativity,
  num_l3cache,
  sys_threads,
  cache_inclusive,
  num_l4cache,
  cache_linesize,
  num_machines,
  cache_size,
  num_sockets
using HostCPUFeatures:
  register_size,
  static_sizeof,
  fast_int64_to_double,
  pick_vector_width,
  pick_vector_width_shift,
  prevpow2,
  simd_integer_register_size,
  fma_fast,
  smax,
  smin,
  has_feature,
  has_opmask_registers,
  register_count,
  static_sizeof,
  cpu_name,
  register_size,
  unwrap,
  intlog2,
  nextpow2,
  fast_half

using SIMDTypes:
  Bit,
  FloatingTypes,
  SignedHW,
  UnsignedHW,
  IntegerTypesHW,
  NativeTypesExceptBitandFloat16,
  NativeTypesExceptBit,
  NativeTypesExceptFloat16,
  NativeTypes,
  _Vec
using LayoutPointers:
  AbstractStridedPointer,
  StridedPointer,
  StridedBitPointer,
  memory_reference,
  stridedpointer,
  zstridedpointer,
  similar_no_offset,
  similar_with_offset,
  grouped_strided_pointer,
  stridedpointers,
  bytestrides,
  DensePointerWrapper,
  zero_offsets

using Static
using Static: One, Zero, eq, ne, lt, le, gt, ge

@inline function promote(x::X, y::Y) where {X,Y}
  T = promote_type(X, Y)
  convert(T, x), convert(T, y)
end
@inline function promote(x::X, y::Y, z::Z) where {X,Y,Z}
  T = promote_type(promote_type(X, Y), Z)
  convert(T, x), convert(T, y), convert(T, z)
end

asbool(::Type{True}) = true
asbool(::Type{False}) = false
# TODO: see if `@inline` is good enough.
# @inline asvalbool(r) = Val(map(Bool, r))
# @inline asvalint(r) = Val(map(Int, r))
@generated function asvalint(r::T) where {T<:Tuple{Vararg{StaticInt}}}
  t = Expr(:tuple)
  for s ∈ T.parameters
    push!(t.args, s.parameters[1])
  end
  Expr(:call, Expr(:curly, :Val, t))
end
@generated function asvalbool(r::T) where {T<:Tuple{Vararg{StaticBool}}}
  t = Expr(:tuple)
  for b ∈ T.parameters
    push!(t.args, b === True)
  end
  Expr(:call, Expr(:curly, :Val, t))
end
@inline val_stride_rank(A) = asvalint(stride_rank(A))
@inline val_dense_dims(A) = asvalbool(ArrayInterface.dense_dims(A))

# doesn't export `Zero` and `One` by default, as these names could conflict with an AD library
export Vec,
  Mask,
  EVLMask,
  MM,
  stridedpointer,
  vload,
  vstore!,
  StaticInt,
  True,
  False,
  Bit,
  vbroadcast,
  mask,
  vfmadd,
  vfmsub,
  vfnmadd,
  vfnmsub,
  VecUnroll,
  Unroll,
  pick_vector_width

using Base: llvmcall, VecElement, HWReal, tail
const LLVMCALL = GlobalRef(Base, :llvmcall)

const Boolean = Union{Bit,Bool}

abstract type AbstractSIMD{W,T<:Union{<:StaticInt,NativeTypes}} <: Real end
abstract type AbstractSIMDVector{W,T} <: AbstractSIMD{W,T} end
"""
    VecUnroll{N,W,T,V<:Union{NativeTypes,AbstractSIMD{W,T}}} <: AbstractSIMD{W,T}

`VecUnroll` supports optimizations when interleaving instructions across different memory storage schemes.
`VecUnroll{N,W,T} is typically a tuple of `N+1` `AbstractSIMDVector{W,T}`s. For example, a `VecUnroll{3,8,Float32}`is a collection of 4×`Vec{8,Float32}`.

# Examples

```jldoctest; setup=:(using VectorizationBase)
julia> rgbs = [
         (
           R = Float32(i) / 255,
           G = Float32(i + 100) / 255,
           B = Float32(i + 200) / 255
         ) for i = 0:7:49
       ]
8-element Vector{NamedTuple{(:R, :G, :B), Tuple{Float32, Float32, Float32}}}:
 (R = 0.0, G = 0.39215687, B = 0.78431374)
 (R = 0.02745098, G = 0.41960785, B = 0.8117647)
 (R = 0.05490196, G = 0.44705883, B = 0.8392157)
 (R = 0.08235294, G = 0.4745098, B = 0.8666667)
 (R = 0.10980392, G = 0.5019608, B = 0.89411765)
 (R = 0.13725491, G = 0.5294118, B = 0.92156863)
 (R = 0.16470589, G = 0.5568628, B = 0.9490196)
 (R = 0.19215687, G = 0.58431375, B = 0.9764706)

julia> ret = vload(
         stridedpointer(reinterpret(reshape, Float32, rgbs)),
         Unroll{1,1,3,2,8,zero(UInt),1}((1, 1))
       )
3 x Vec{8, Float32}
Vec{8, Float32}<0.0f0, 0.02745098f0, 0.05490196f0, 0.08235294f0, 0.10980392f0, 0.13725491f0, 0.16470589f0, 0.19215687f0>
Vec{8, Float32}<0.39215687f0, 0.41960785f0, 0.44705883f0, 0.4745098f0, 0.5019608f0, 0.5294118f0, 0.5568628f0, 0.58431375f0>
Vec{8, Float32}<0.78431374f0, 0.8117647f0, 0.8392157f0, 0.8666667f0, 0.89411765f0, 0.92156863f0, 0.9490196f0, 0.9764706f0>

julia> typeof(ret)
VecUnroll{2, 8, Float32, Vec{8, Float32}}
```

While the `R`, `G`, and `B` are interleaved in `rgb`s, they have effectively been split out in `ret`
(the first contains all 8 `R` values, with `G` and `B` in the second and third, respectively).

To optimize for the user's CPU, in real code it would typically be better to use `Int(pick_vector_width(Float32))`  # # following two definitions are for checking that you aren't accidentally creating `VecUnroll{0}`s.
in place of `8` (`W`) in the `Unroll` construction.  # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMD{W,T}} = (@assert(N > 0); new{N,W,T,V}(data))
"""
struct VecUnroll{N,W,T,V<:Union{NativeTypes,AbstractSIMD{W,T}}} <:
       AbstractSIMD{W,T}
  data::Tuple{V,Vararg{V,N}}
  @inline (VecUnroll(
    data::Tuple{V,Vararg{V,N}}
  )) where {N,W,T,V<:AbstractSIMD{W,T}} = new{N,W,T,V}(data)
  @inline (VecUnroll(data::Tuple{T,Vararg{T,N}})) where {N,T<:NativeTypes} =
    new{N,1,T,T}(data)
  # # following two definitions are for checking that you aren't accidentally creating `VecUnroll{0}`s.
  # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMD{W,T}} = (@assert(N > 0); new{N,W,T,V}(data))
  # @inline (VecUnroll(data::Tuple{T,Vararg{T,N}})::VecUnroll{N,T,T}) where {N,T<:NativeTypes} = (@assert(N > 0); new{N,1,T,T}(data))

  # @inline VecUnroll{N,W,T,V}(data::Tuple{V,Vararg{V,N}}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
  # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,Vec{W,T}}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
  # @inline (VecUnroll(data::Tuple{V,Vararg{V,N}})::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = new{N,W,T,V}(data)
end
# @inline VecUnroll(data::Tuple) = VecUnroll(promote(data))
# const AbstractSIMD{W,T} = Union{AbstractSIMDVector{W,T},VecUnroll{<:Any,W,T}}
const IntegerTypes = Union{StaticInt,IntegerTypesHW}
const VecOrScalar = Union{AbstractSIMDVector,NativeTypes}
const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
# const NativeTypesV = Union{AbstractSIMD,NativeTypes,StaticInt}
const IntegerTypesV = Union{AbstractSIMD{<:Any,<:IntegerTypes},IntegerTypesHW}

struct Vec{W,T} <: AbstractSIMDVector{W,T}
  data::NTuple{W,Core.VecElement{T}}
  @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:NativeTypes} =
    new{W,T}(x)
  @generated function Vec(
    x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}
  ) where {_W,T<:NativeTypes}
    W = _W + 1
    # @assert W === pick_vector_width(W, T)# || W === 8
    vtyp = Expr(:curly, :Vec, W, T)
    Expr(:block, Expr(:meta, :inline), Expr(:(::), Expr(:call, vtyp, :x), vtyp))
  end
  # @inline function Vec(x::NTuple{W,<:Core.VecElement}) where {W}
  #     T = eltype(x)
  #     @assert W === pick_vector_width(W, T)
  #     # @assert ispow2(W) && (W ≤ max(pick_vector_width(W, T), 8))
  #     new{W,T}(x)
  # end
end

Base.:*(::Vec, y::Zero) = y
Base.:*(x::Zero, ::Vec) = x

@inline Base.copy(v::AbstractSIMDVector) = v
@inline asvec(x::_Vec) = Vec(x)
@inline asvec(x) = x
@inline data(vu::VecUnroll) = getfield(vu, :data)

@inline unrolleddata(x) = x
@inline unrolleddata(x::VecUnroll) = getfield(x, :data)

@inline _demoteint(::Type{T}) where {T} = T
@inline _demoteint(::Type{Int64}) = Int32
@inline _demoteint(::Type{UInt64}) = UInt32

# abstract type AbstractMask{W,U<:Union{UnsignedHW,UInt128,UInt256,UInt512,UInt1024}} <: AbstractSIMDVector{W,Bit} end
abstract type AbstractMask{W,U<:Union{UnsignedHW,UInt128}} <:
              AbstractSIMDVector{W,Bit} end
struct Mask{W,U} <: AbstractMask{W,U}
  u::U
  @inline function Mask{W,U}(u::Unsigned) where {W,U} # ignores U...
    U2 = mask_type(StaticInt{W}())
    new{W,U2}(u % U2)
  end
end
struct EVLMask{W,U} <: AbstractMask{W,U}
  u::U
  evl::UInt32
  @inline function EVLMask{W,U}(u::Unsigned, evl) where {W,U} # ignores U...
    U2 = mask_type(StaticInt{W}())
    new{W,U2}(u % U2, evl % UInt32)
  end
end
const AnyMask{W} =
  Union{AbstractMask{W},VecUnroll{<:Any,W,Bit,<:AbstractMask{W}}}
@inline Mask{W}(u::U) where {W,U<:Unsigned} = Mask{W,U}(u)
@inline EVLMask{W}(u::U, i) where {W,U<:Unsigned} = EVLMask{W,U}(u, i)
@inline Mask{1}(b::Bool) = b
@inline EVLMask{1}(b::Bool, i) = b
@inline Mask(m::EVLMask{W,U}) where {W,U} = Mask{W,U}(getfield(m, :u))
# Const prop is good enough; added an @inferred test to make sure.
# Removed because confusion can cause more harm than good.

@inline Base.broadcastable(v::AbstractSIMDVector) = Ref(v)

Vec{W,T}(x::Vararg{NativeTypes,W}) where {W,T<:NativeTypes} =
  Vec(ntuple(w -> Core.VecElement{T}(x[w]), Val{W}()))
Vec{1,T}(x::Union{Float32,Float64}) where {T<:NativeTypes} = T(x)
Vec{1,T}(
  x::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Bool}
) where {T<:NativeTypes} = T(x)

@inline Base.length(::AbstractSIMDVector{W}) where {W} = W
@inline Base.size(::AbstractSIMDVector{W}) where {W} = (W,)
@inline Base.eltype(::AbstractSIMD{W,T}) where {W,T} = T
@inline Base.conj(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.adjoint(v::AbstractSIMDVector) = v # so that things like dot products work.
@inline Base.transpose(v::AbstractSIMDVector) = v # so that things like dot products work.

# Not using getindex/setindex as names to emphasize that these are generally treated as single objects, not collections.
@generated function extractelement(
  v::Vec{W,T},
  i::I
) where {W,I<:IntegerTypesHW,T}
  typ = LLVM_TYPES[T]
  instrs = """
      %res = extractelement <$W x $typ> %0, i$(8sizeof(I)) %1
      ret $typ %res
  """
  call = :($LLVMCALL($instrs, $T, Tuple{_Vec{$W,$T},$I}, data(v), i))
  Expr(:block, Expr(:meta, :inline), call)
end
@generated function insertelement(
  v::Vec{W,T},
  x::T,
  i::I
) where {W,I<:IntegerTypesHW,T}
  typ = LLVM_TYPES[T]
  instrs = """
      %res = insertelement <$W x $typ> %0, $typ %1, i$(8sizeof(I)) %2
      ret <$W x $typ> %res
  """
  call = :(Vec(
    $LLVMCALL($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T},$T,$I}, data(v), x, i)
  ))
  Expr(:block, Expr(:meta, :inline), call)
end
@inline (v::AbstractSIMDVector)(i::IntegerTypesHW) =
  extractelement(v, i - one(i))
@inline (v::AbstractSIMDVector)(i::Integer) = extractelement(v, Int(i) - 1)
Base.@propagate_inbounds (vu::VecUnroll)(i::Integer, j::Integer) =
  getfield(vu, :data)[j](i)

@inline Base.Tuple(v::Vec{W}) where {W} = ntuple(v, Val{W}())

# Use with care in function signatures; try to avoid the `T` to stay clean on Test.detect_unbound_args

@inline data(v) = v
@inline data(v::Vec) = getfield(v, :data)

function Base.show(io::IO, v::AbstractSIMDVector{W,T}) where {W,T}
  name = typeof(v)
  print(io, "$(name)<")
  for w ∈ 1:W
    print(io, repr(extractelement(v, w - 1)))
    w < W && print(io, ", ")
  end
  print(io, ">")
end
Base.bitstring(m::AbstractMask{W}) where {W} = bitstring(data(m))[end-W+1:end]
function Base.show(io::IO, m::AbstractMask{W}) where {W}
  bits = data(m)
  if m isa EVLMask
    print(io, "EVLMask{$W,Bit}<")
  else
    print(io, "Mask{$W,Bit}<")
  end
  for w ∈ 0:W-1
    print(io, (bits & 0x01) % Int)
    bits >>= 0x01
    w < W - 1 && print(io, ", ")
  end
  print(io, ">")
end
function Base.show(io::IO, vu::VecUnroll{N,W,T,V}) where {N,W,T,V}
  println(io, "$(N+1) x $V")
  d = data(vu)
  for n = 1:N+1
    show(io, d[n])
    n > N || println(io)
  end
end

"""
The name `MM` type refers to _MM registers such as `XMM`, `YMM`, and `ZMM`.
`MMX` from the original MMX SIMD instruction set is a [meaningless initialism](https://en.wikipedia.org/wiki/MMX_(instruction_set)#Naming).

The `MM{W,X}` type is used to represent SIMD indexes of width `W` with stride `X`.
"""
struct MM{W,X,I<:Union{HWReal,StaticInt}} <: AbstractSIMDVector{W,I}
  i::I
  @inline MM{W,X}(i::T) where {W,X,T<:Union{HWReal,StaticInt}} =
    new{W,X::Int,T}(i)
end
@inline MM(i::MM{W,X}) where {W,X} = MM{W,X}(getfield(i, :i))
@inline MM{W}(i::Union{HWReal,StaticInt}) where {W} = MM{W,1}(i)
@inline MM{W}(i::Union{HWReal,StaticInt}, ::StaticInt{X}) where {W,X} =
  MM{W,X}(i)
@inline data(i::MM) = getfield(i, :i)

@inline extractelement(i::MM{W,X,I}, j) where {W,X,I<:HWReal} =
  getfield(i, :i) + (X % I) * (j % I)
@inline extractelement(i::MM{W,X,I}, j) where {W,X,I<:StaticInt} =
  getfield(i, :i) + X * j

Base.propertynames(::AbstractSIMD) = ()
function Base.getproperty(::AbstractSIMD, ::Symbol)
  throw(
    ErrorException(
      """
`Base.getproperty` not defined on AbstractSIMD.
If you wish to work with the data as a tuple, it is recommended to use `Tuple(v)`. Once you have an ordinary tuple, you can access
individual elements normally. Alternatively, you can index using parenthesis, e.g. `v(1)` indexes the first element.
Parenthesis are used instead of `getindex`/square brackets because `AbstractSIMD` objects represent a single number, and
for `x::Number`, `x[1] === x`.

If you wish to perform a reduction on the collection, the naming convention is prepending the base function with a `v`. These functions
are not overloaded, because for `x::Number`, `sum(x) === x`. Functions include `vsum`, `vprod`, `vmaximum`, `vminimum`, `vany`, and `vall`.

If you wish to define a new operation applied to the entire vector, do not define it in terms of operations on the individual eleemnts.
This will often lead to bad code generation -- bad in terms of both performance, and often silently producing incorrect results!
Instead, implement them in terms of existing functions defined on `::AbstractSIMD`. Please feel free to file an issue if you would like
clarification, and especially if you think the function may be useful for others and should be included in `VectorizationBase.jl`.
"""
    )
  )
end

"""
pause()

For use in spin-and-wait loops, like spinlocks.
"""
@inline pause() = ccall(:jl_cpu_pause, Cvoid, ())

include("static.jl")
include("cartesianvindex.jl")
include("early_definitions.jl")
include("promotion.jl")
include("llvm_types.jl")
include("lazymul.jl")
include("strided_pointers/stridedpointers.jl")
# include("strided_pointers/bitpointers.jl")
include("strided_pointers/cartesian_indexing.jl")
include("strided_pointers/cse_stridemultiples.jl")
include("llvm_intrin/binary_ops.jl")
include("vector_width.jl")
include("ranges.jl")
include("llvm_intrin/conversion.jl")
include("llvm_intrin/masks.jl")
include("llvm_intrin/intrin_funcs.jl")
include("llvm_intrin/memory_addr.jl")
include("llvm_intrin/unary_ops.jl")
include("llvm_intrin/vbroadcast.jl")
include("llvm_intrin/vector_ops.jl")
include("llvm_intrin/nonbroadcastingops.jl")
include("llvm_intrin/integer_fma.jl")
include("llvm_intrin/conflict.jl")
include("llvm_intrin/vfmaddsub.jl")
include("vecunroll/memory.jl")
include("vecunroll/mappedloadstore.jl")
include("vecunroll/fmap.jl")
include("base_defs.jl")
include("alignment.jl")
include("special/misc.jl")
include("special/double.jl")
include("special/exp.jl")
include("special/verf.jl")
# include("special/log.jl")

demoteint(::Type{T}, W) where {T} = False()
demoteint(::Type{UInt64}, W::StaticInt) = gt(W, pick_vector_width(UInt64))
demoteint(::Type{Int64}, W::StaticInt) = gt(W, pick_vector_width(Int64))

@generated function simd_vec(
  ::DemoteInt,
  y::_T,
  x::Vararg{_T,_W}
) where {DemoteInt<:StaticBool,_T,_W}
  W = 1 + _W
  T = DemoteInt === True ? _demoteint(_T) : _T
  trunc = T !== _T
  Wfull = nextpow2(W)
  ty = LLVM_TYPES[T]
  init = W == Wfull ? "undef" : "zeroinitializer"
  instrs = ["%v0 = insertelement <$Wfull x $ty> $init, $ty %0, i32 0"]
  Tup = Expr(:curly, :Tuple, T)
  for w ∈ 1:_W
    push!(
      instrs,
      "%v$w = insertelement <$Wfull x $ty> %v$(w-1), $ty %$w, i32 $w"
    )
    push!(Tup.args, T)
  end
  push!(instrs, "ret <$Wfull x $ty> %v$_W")
  llvmc = :($LLVMCALL($(join(instrs, "\n")), _Vec{$Wfull,$T}, $Tup))
  trunc ? push!(llvmc.args, :(y % $T)) : push!(llvmc.args, :y)
  for w ∈ 1:_W
    ref = Expr(:ref, :x, w)
    trunc && (ref = Expr(:call, :%, ref, T))
    push!(llvmc.args, ref)
  end
  meta = Expr(:meta, :inline)
  if VERSION >= v"1.8.0-beta"
    push!(meta.args, Expr(:purity, true, true, true, true, false))
  end
  quote
    $meta
    Vec($llvmc)
  end
end
function vec_quote(demote, W, Wpow2, offset::Int = 0)
  call = Expr(:call, :simd_vec, Expr(:call, demote ? :True : :False))
  Wpow2 += offset
  iszero(offset) && push!(call.args, :y)
  foreach(
    w -> push!(call.args, Expr(:call, getfield, :x, w, false)),
    max(1, offset):min(W, Wpow2)-1
  )
  foreach(w -> push!(call.args, Expr(:call, :zero, :T)), W+1:Wpow2)
  call
end
@generated function _vec(
  ::StaticInt{_Wpow2},
  ::DemoteInt,
  y::T,
  x::Vararg{T,_W}
) where {DemoteInt<:StaticBool,_Wpow2,_W,T<:NativeTypes}
  W = _W + 1
  demote = DemoteInt === True
  Wpow2 = demote ? 2_Wpow2 : _Wpow2
  if W ≤ Wpow2
    vec_quote(demote, W, Wpow2)
  else
    tup = Expr(:tuple)
    offset = 0
    while offset < W
      push!(tup.args, vec_quote(demote, W, Wpow2, offset))
      offset += Wpow2
    end
    Expr(:call, :VecUnroll, tup)
  end
end
@static if VERSION >= v"1.8.0-beta"
  Base.@assume_effects total @inline function Vec(
    y::T,
    x::Vararg{T,_W}
  ) where {_W,T<:NativeTypes}
    W = StaticInt{_W}() + One()
    _vec(pick_vector_width(W, T), demoteint(T, W), y, x...)
  end
else
  @inline function Vec(y::T, x::Vararg{T,_W}) where {_W,T<:NativeTypes}
    W = StaticInt{_W}() + One()
    _vec(pick_vector_width(W, T), demoteint(T, W), y, x...)
  end
end
@inline reduce_to_onevec(f::F, vu::VecUnroll) where {F} =
  ArrayInterface.reduce_tup(f, data(vu))

if VERSION >= v"1.7.0" && hasfield(Method, :recursion_relation)
  dont_limit = Returns(true)
  for f in (vconvert, _vconvert)
    for m in methods(f)
      m.recursion_relation = dont_limit
    end
  end
end

include("precompile.jl")
_precompile_()

end # module
