#
# We use these definitions because when we have other SIMD operations with masks
# LLVM optimizes the masks better.
function truncate_mask!(instrs, input, W, suffix, reverse_load::Bool = false)
  mtyp_input = "i$(max(8,nextpow2(W)))"
  mtyp_trunc = "i$(W)"
  if reverse_load
    bitreverse = "i$(W) @llvm.bitreverse.i$(W)"
    decl = "declare $bitreverse(i$(W))"
    bitrevmask = "bitrevmask.$(suffix)"
    if mtyp_input == mtyp_trunc
      str = """
          %$(bitrevmask) = call $(bitreverse)($(mtyp_trunc) %$(input))
          %mask.$(suffix) = bitcast $mtyp_input %$(bitrevmask) to <$W x i1>
      """
    else
      str = """
          %masktrunc.$(suffix) = trunc $mtyp_input %$input to $mtyp_trunc
          %$(bitrevmask) = call $(bitreverse)($(mtyp_trunc) %masktrunc.$(suffix))
          %mask.$(suffix) = bitcast $mtyp_trunc %$(bitrevmask) to <$W x i1>
      """
    end
  else
    decl = ""
    if mtyp_input == mtyp_trunc
      str = "%mask.$(suffix) = bitcast $mtyp_input %$input to <$W x i1>"
    else
      str = "%masktrunc.$(suffix) = trunc $mtyp_input %$input to $mtyp_trunc\n%mask.$(suffix) = bitcast $mtyp_trunc %masktrunc.$(suffix) to <$W x i1>"
    end
  end
  push!(instrs, str)
  decl
end
function zext_mask!(instrs, input, W, suffix)
  mtyp_input = "i$(max(8,nextpow2(W)))"
  mtyp_trunc = "i$(W)"
  str = if mtyp_input == mtyp_trunc
    "%res.$(suffix) = bitcast <$W x i1> %$input to $mtyp_input"
  else
    "%restrunc.$(suffix) = bitcast <$W x i1> %$input to $mtyp_trunc\n%res.$(suffix) = zext $mtyp_trunc %restrunc.$(suffix) to $mtyp_input"
  end
  push!(instrs, str)
end
function binary_mask_op_instrs(W, op)
  mtyp_input = "i$(max(8,nextpow2(W)))"
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  truncate_mask!(instrs, '1', W, 1)
  push!(instrs, "%combinedmask = $op <$W x i1> %mask.0, %mask.1")
  zext_mask!(instrs, "combinedmask", W, 1)
  push!(instrs, "ret $mtyp_input %res.1")
  join(instrs, "\n")
end
function binary_mask_op(W, U, op, evl::Symbol = Symbol(""))
  instrs = binary_mask_op_instrs(W, op)
  mask = Expr(:curly, evl === Symbol("") ? :Mask : :EVLMask, W)
  gf = GlobalRef(Core, :getfield)
  gf1 = Expr(:call, gf, :m1, 1, false)
  gf2 = Expr(:call, gf, :m2, 1, false)
  llvmc = Expr(:call, GlobalRef(Base, :llvmcall), instrs, U, :(Tuple{$U,$U}), gf1, gf2)
  call = Expr(:call, mask, llvmc)
  evl === Symbol("") ||
    push!(call.args, Expr(:call, evl, :($gf(m1, :evl)), :($gf(m2, :evl))))
  Expr(:block, Expr(:meta, :inline), call)
end

@inline data(m::Mask) = getfield(m, :u)
@inline data(m::EVLMask) = getfield(m, :u)
@inline Base.convert(::Type{Mask{W,U}}, m::EVLMask{W,U}) where {W,U} =
  Mask{W,U}(getfield(m, :u))
for (f, op, evl) ∈ [
  (:vand, "and", :min),
  (:vor, "or", :max),
  (:vxor, "xor", Symbol("")),
  (:veq, "icmp eq", Symbol("")),
  (:vne, "icmp ne", Symbol("")),
]
  @eval begin
    @generated function $f(m1::AbstractMask{W,U}, m2::AbstractMask{W,U}) where {W,U}
      binary_mask_op(
        W,
        U,
        $op,
        ((m1 <: EVLMask) && (m2 <: EVLMask)) ? $(QuoteNode(evl)) : Symbol(""),
      )
    end
  end
end
for f ∈ [:vand, :vor, :vxor] # ignore irrelevant bits, so just bitcast to `Bool`
  @eval @inline $f(a::Vec{W,Bool}, b::Vec{W,Bool}) where {W} =
    vreinterpret(Bool, $f(vreinterpret(UInt8, a), vreinterpret(UInt8, b)))
end
for f ∈ [:vne, :veq] # Here we truncate.
  @eval @inline $f(a::Vec{W,Bool}, b::Vec{W,Bool}) where {W} =
    convert(Bool, $f(convert(Bit, a), convert(Bit, b)))
end

@generated function vconvert(
  ::Type{Vec{W,I}},
  m::AbstractMask{W,U},
) where {W,I<:IntegerTypesHW,U<:Union{UInt8,UInt16,UInt32,UInt64}}
  bits = 8sizeof(I)
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  push!(
    instrs,
    "%res = zext <$W x i1> %mask.0 to <$W x i$(bits)>\nret <$W x i$(bits)> %res",
  )
  gf = Expr(:call, GlobalRef(Core, :getfield), :m, 1, false)
  llvmc = Expr(
    :call,
    GlobalRef(Base, :llvmcall),
    join(instrs, "\n"),
    :(_Vec{$W,$I}),
    :(Tuple{$U}),
    gf,
  )
  Expr(:block, Expr(:meta, :inline), Expr(:call, :Vec, llvmc))
end


@generated function splitint(
  i::S,
  ::Type{T},
) where {S<:Base.BitInteger,T<:Union{Bool,Base.BitInteger}}
  sizeof_S = sizeof(S)
  sizeof_T = sizeof(T)
  if sizeof_T > sizeof_S
    return :(i % T)
  elseif sizeof_T == sizeof_S
    return :i
  end
  W, r = divrem(sizeof_S, sizeof_T)
  @assert iszero(r)
  vtyp = "<$W x i$(8sizeof_T)>"
  instrs = """
      %split = bitcast i$(8sizeof_S) %0 to $vtyp
      ret $vtyp %split
  """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{$S}, i))
  end
end
@generated function fuseint(v::Vec{W,I}) where {W,I<:Union{Bool,Base.BitInteger}}
  @assert ispow2(W)
  bytes = W * sizeof(I)
  bits = 8bytes
  @assert bytes ≤ 16
  T = (I <: Signed) ? Symbol(:Int, bits) : Symbol(:UInt, bits)
  vtyp = "<$W x i$(8sizeof(I))>"
  styp = "i$(bits)"
  instrs = """
      %fused = bitcast $vtyp %0 to $styp
      ret $styp %fused
  """
  quote
    $(Expr(:meta, :inline))
    $LLVMCALL($instrs, $T, Tuple{_Vec{$W,$I}}, data(v))
  end
end


function vadd_expr(W, U, instr)
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  truncate_mask!(instrs, '1', W, 1)
  push!(
    instrs,
    """%uv.0 = zext <$W x i1> %mask.0 to <$W x i8>
%uv.1 = zext <$W x i1> %mask.1 to <$W x i8>
%res = $instr <$W x i8> %uv.0, %uv.1
ret <$W x i8> %res""",
  )
  Expr(
    :block,
    Expr(:meta, :inline),
    :(Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        _Vec{$W,UInt8},
        Tuple{$U,$U},
        getfield(m1, :u),
        getfield(m2, :u),
      ),
    )),
  )
end
@generated vadd_fast(m1::AbstractMask{W,U}, m2::AbstractMask{W,U}) where {W,U} =
  vadd_expr(W, U, "add")
@generated vsub_fast(m1::AbstractMask{W,U}, m2::AbstractMask{W,U}) where {W,U} =
  vadd_expr(W, U, "sub")
@inline vadd(m1::AbstractMask{W,U}, m2::AbstractMask{W,U}) where {W,U} = vadd_fast(m1, m2)
@inline vsub(m1::AbstractMask{W,U}, m2::AbstractMask{W,U}) where {W,U} = vsub_fast(m1, m2)

@inline Base.:(&)(m::AbstractMask{W,U}, b::Bool) where {W,U} =
  Mask{W,U}(Core.ifelse(b, getfield(m, :u), zero(getfield(m, :u))))
@inline Base.:(&)(b::Bool, m::AbstractMask{W,U}) where {W,U} =
  Mask{W,U}(Core.ifelse(b, getfield(m, :u), zero(getfield(m, :u))))

@inline Base.:(|)(m::AbstractMask{W,U}, b::Bool) where {W,U} =
  Mask{W,U}(Core.ifelse(b, getfield(max_mask(Mask{W,U}), :u), getfield(m, :u)))
@inline Base.:(|)(b::Bool, m::AbstractMask{W,U}) where {W,U} =
  Mask{W,U}(Core.ifelse(b, getfield(max_mask(Mask{W,U}), :u), getfield(m, :u)))

@inline function Base.:(&)(m::EVLMask{W,U}, b::Bool) where {W,U}
  EVLMask{W,U}(
    Core.ifelse(b, getfield(m, :u), zero(getfield(m, :u))),
    Core.ifelse(b, getfield(m, :evl), 0x00000000),
  )
end
@inline function Base.:(&)(b::Bool, m::EVLMask{W,U}) where {W,U}
  EVLMask{W,U}(
    Core.ifelse(b, getfield(m, :u), zero(getfield(m, :u))),
    Core.ifelse(b, getfield(m, :evl), 0x00000000),
  )
end

@inline function Base.:(|)(m::EVLMask{W,U}, b::Bool) where {W,U}
  EVLMask{W,U}(
    Core.ifelse(b, getfield(max_mask(Mask{W,U}), :u), getfield(m, :u)),
    Core.ifelse(b, W % UInt32, getfield(m, :evl)),
  )
end
@inline function Base.:(|)(b::Bool, m::EVLMask{W,U}) where {W,U}
  EVLMask{W,U}(
    Core.ifelse(b, getfield(max_mask(Mask{W,U}), :u), getfield(m, :u)),
    Core.ifelse(b, W % UInt32, getfield(m, :evl)),
  )
end

@inline Base.:(⊻)(m::AbstractMask{W,U}, b::Bool) where {W,U} =
  Mask{W,U}(Core.ifelse(b, ~getfield(m, :u), getfield(m, :u)))
@inline Base.:(⊻)(b::Bool, m::AbstractMask{W,U}) where {W,U} =
  Mask{W,U}(Core.ifelse(b, ~getfield(m, :u), getfield(m, :u)))

@inline vshl(m::AbstractMask{W,U}, i::IntegerTypesHW) where {W,U} =
  Mask{W,U}(shl(getfield(m, :u), i))
@inline vashr(m::AbstractMask{W,U}, i::IntegerTypesHW) where {W,U} =
  Mask{W,U}(shr(getfield(m, :u), i))
@inline vlshr(m::AbstractMask{W,U}, i::IntegerTypesHW) where {W,U} =
  Mask{W,U}(shr(getfield(m, :u), i))

@inline zero_mask(::AbstractSIMDVector{W}) where {W} = Mask(zero_mask(Val(W)))
@inline zero_mask(::VecUnroll{N,W}) where {N,W} = VecUnroll{N}(Mask(zero_mask(Val(W))))
@inline max_mask(::AbstractSIMDVector{W}) where {W} = Mask(max_mask(Val(W)))
@inline max_mask(::VecUnroll{N,W}) where {N,W} = VecUnroll{N}(Mask(max_mask(Val(W))))
@inline zero_mask(::NativeTypes) = false
@inline max_mask(::NativeTypes) = true

for (U, W) in [(UInt8, 8), (UInt16, 16), (UInt32, 32), (UInt64, 64)]
  @eval @inline vany(m::AbstractMask{$W,$U}) = getfield(m, :u) != $(zero(U))
  @eval @inline vall(m::AbstractMask{$W,$U}) = getfield(m, :u) == $(typemax(U))
end
# TODO: use vector reduction intrsincs
@inline function vany(m::AbstractMask{W}) where {W}
  mm = getfield(max_mask(Val{W}()), :u)
  mu = getfield(m, :u)
  (mu & mm) !== zero(mu)
end
@inline function vall(m::AbstractMask{W}) where {W}
  mm = getfield(max_mask(Val{W}()), :u)
  mu = getfield(m, :u)
  mm & mu === mm
end
@inline vany(b::Bool) = b
@inline vall(b::Bool) = b
@inline vsum(m::AbstractMask) = count_ones(getfield(m, :u))
@inline vprod(m::AbstractMask) = vall(m)

@generated function vnot(m::AbstractMask{W,U}) where {W,U}
  mtyp_input = "i$(8sizeof(U))"
  mtyp_trunc = "i$(W)"
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  mask = llvmconst(W, "i1 true")
  push!(instrs, "%resvec.0 = xor <$W x i1> %mask.0, $mask")
  zext_mask!(instrs, "resvec.0", W, 1)
  push!(instrs, "ret $mtyp_input %res.1")
  quote
    $(Expr(:meta, :inline))
    Mask{$W}($LLVMCALL($(join(instrs, "\n")), $U, Tuple{$U}, getfield(m, :u)))
  end
end
@inline vnot(x::Bool) = Base.not_int(x)
# @inline Base.:(~)(m::Mask) = !m

@inline Base.count_ones(m::AbstractMask) = count_ones(getfield(m, :u))
@inline vadd(m::AbstractMask, i::IntegerTypesHW) = i + count_ones(m)
@inline vadd(i::IntegerTypesHW, m::AbstractMask) = i + count_ones(m)

@generated function vzero(::Type{M}) where {W,M<:Mask{W}}
  Expr(
    :block,
    Expr(:meta, :inline),
    Expr(:call, Expr(:curly, :Mask, W), Expr(:call, :zero, mask_type_symbol(W))),
  )
end
@generated function vzero(::Type{M}) where {W,M<:EVLMask{W}}
  Expr(
    :block,
    Expr(:meta, :inline),
    Expr(
      :call,
      Expr(:curly, :EVLMask, W),
      Expr(:call, :zero, mask_type_symbol(W)),
      0x00000000,
    ),
  )
end
@inline vzero(::Mask{W,U}) where {W,U} = Mask{W}(zero(U))
@inline vzero(::EVLMask{W,U}) where {W,U} = EVLMask{W}(zero(U), 0x00000000)
@inline Base.zero(::Type{M}) where {W,M<:AbstractMask{W}} = vzero(M)
@inline zero_mask(::Union{Val{W},StaticInt{W}}) where {W} =
  EVLMask{W}(zero(VectorizationBase.mask_type(Val{W}())), 0x00000000)

@generated function max_mask(::Union{Val{W},StaticInt{W}}) where {W}
  U = mask_type(W)
  :(EVLMask{$W,$U}($(one(U) << W - one(U)), $(UInt32(W))))
end
@inline max_mask(::Type{T}) where {T} = max_mask(pick_vector_width(T))
@generated max_mask(::Type{Mask{W,U}}) where {W,U} =
  EVLMask{W,U}(one(U) << W - one(U), W % UInt32)

@generated function valrem(::Union{Val{W},StaticInt{W}}, l::T) where {W,T<:Union{Integer,StaticInt}}
  ex = ispow2(W) ? :(l & $(T(W - 1))) : Expr(:call, Base.urem_int, :l, T(W))
  Expr(:block, Expr(:meta, :inline), ex)
end

function bzhi_quote(b)
  T = b == 32 ? :UInt32 : :UInt64
  typ = 'i' * string(b)
  instr = "i$b @llvm.x86.bmi.bzhi.$b"
  decl = "declare $instr(i$b, i$b) nounwind readnone"
  instrs = "%res = call $instr(i$b %0, i$b %1)\n ret i$b %res"
  llvmcall_expr(decl, instrs, T, :(Tuple{$T,$T}), typ, [typ, typ], [:a, :b])
end
@generated bzhi(a::UInt32, b::UInt32) = bzhi_quote(32)
@generated bzhi(a::UInt64, b::UInt64) = bzhi_quote(64)

# @generated function _mask(::Union{Val{W},StaticInt{W}}, l::I, ::True) where {W,I<:Union{Integer,StaticInt}}
#   # if `has_opmask_registers()` then we can use bitmasks directly, so we create them via bittwiddling
#   M = mask_type(W)
#   quote # If the arch has opmask registers, we can generate a bitmask and then move it into the opmask register
#     $(Expr(:meta,:inline))
#     evl = valrem(Val{$W}(), (l % $M) - one($M))
#     EVLMask{$W,$M}($(typemax(M)) >>> ($(M(8sizeof(M))-1) - evl), evl + one(evl))
#   end
# end
@generated function _mask_bzhi(::Union{Val{W},StaticInt{W}}, l::I) where {W,I<:Union{Integer,StaticInt}}
  U = mask_type_symbol(W)
  T = W > 32 ? :UInt64 : :UInt32
  quote
    $(Expr(:meta, :inline))
    m = valrem(StaticInt{$W}(), l % $T)
    m = Core.ifelse((m % UInt8) == 0x00, $W % $T, m)
    EVLMask{$W,$U}(bzhi(-1 % $T, m) % $U, m)
  end
end
# @inline function _mask_bzhi(::Union{Val{W},StaticInt{W}}, l::I) where {W,I<:Union{Integer,StaticInt}}
#   U = mask_type(StaticInt(W))
#   # m = ((l) % UInt32) & ((W-1) % UInt32)
#   m = valrem(StaticInt{W}(), l % UInt32)
#   m = Core.ifelse((m % UInt8) == 0x00, W % UInt32, m)
#   # m = Core.ifelse(zero(m) == m, -1 % UInt32, m)
#   EVLMask{W,U}(bzhi(-1 % UInt32, m) % U, m)
# end
# @inline function _mask(::Union{Val{W},StaticInt{W}}, l::I, ::True) where {W,I<:Union{Integer,StaticInt}}
#   U = mask_type(StaticInt(W))
#   m = ((l-one(l)) % UInt32) & ((W-1) % UInt32)
#   m += one(m)
#   EVLMask{W,U}(bzhi(-1 % UInt32, m) % U, m)
# end
# @generated function _mask(::Union{Val{W},StaticInt{W}}, l::I, ::True) where {W,I<:Union{Integer,StaticInt}}
#   M = mask_type_symbol(W)
#   quote
#     $(Expr(:meta,:inline))
#     evl = valrem(Val{$W}(), vsub_nw((l % $M), one($M)))
#     EVLMask{$W}(data(evl ≥ MM{$W}(0)), vadd_nw(evl, one(evl)))
#   end
# end

function mask_shift_quote(W::Int, bmi::Bool)
  if (((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686))) && bmi
    W ≤ 64 && return Expr(:block, Expr(:meta, :inline), :(_mask_bzhi(StaticInt{$W}(), l)))
  end
  MT = mask_type(W)
  quote # If the arch has opmask registers, we can generate a bitmask and then move it into the opmask register
    $(Expr(:meta, :inline))
    evl = valrem(Val{$W}(), (l % $MT) - one($MT))
    EVLMask{$W,$MT}($(typemax(MT)) >>> ($(MT(8sizeof(MT)) - 1) - evl), evl + one(evl))
  end
end
@generated _mask_shift(::StaticInt{W}, l, ::True) where {W} = mask_shift_quote(W, true)
@generated _mask_shift(::StaticInt{W}, l, ::False) where {W} = mask_shift_quote(W, false)
@static if Base.libllvm_version ≥ v"12"
  function active_lane_mask_quote(W::Int)
    quote
      $(Expr(:meta, :inline))
      upper = (l % UInt32) & $((UInt32(W - 1)))
      upper = Core.ifelse(upper == 0x00000000, $(W % UInt32), upper)
      mask(Val{$W}(), 0x00000000, upper)
    end
  end
else
  function active_lane_mask_quote(W::Int)
    quote
      $(Expr(:meta, :inline))
      mask(Val{$W}(), 0x00000000, vsub_nw(l % UInt32, 0x00000001) & $(UInt32(W - 1)))
    end
  end
end
function mask_cmp_quote(W::Int, RS::Int, bmi::Bool)
  M = mask_type_symbol(W)
  bytes = min(RS ÷ W, 8)
  bytes < 4 && return mask_shift_quote(W, bmi)
  T = integer_of_bytes_symbol(bytes, true)
  quote
    $(Expr(:meta, :inline))
    evl = valrem(Val{$W}(), vsub_nw((l % $T), one($T)))
    EVLMask{$W}(data(evl ≥ MM{$W}(zero($T))), vadd_nw(evl, one(evl)))
  end
end
@generated _mask_cmp(
  ::Union{Val{W},StaticInt{W}},
  l::I,
  ::StaticInt{RS},
  ::True,
) where {W,RS,I<:Union{Integer,StaticInt}} = mask_cmp_quote(W, RS, true)
@generated _mask_cmp(
  ::Union{Val{W},StaticInt{W}},
  l::I,
  ::StaticInt{RS},
  ::False,
) where {W,RS,I<:Union{Integer,StaticInt}} = mask_cmp_quote(W, RS, false)
@generated _mask(::Union{Val{W},StaticInt{W}}, l::I, ::True) where {W,I<:Union{Integer,StaticInt}} =
  mask_shift_quote(W, true)
@generated function _mask(::Union{Val{W},StaticInt{W}}, l::I, ::False) where {W,I<:Union{Integer,StaticInt}}
  # Otherwise, it's probably more efficient to use a comparison, as this will probably create some type that can be used directly for masked moves/blends/etc
  if W > 16
    Expr(
      :block,
      Expr(:meta, :inline),
      :(_mask_shift(StaticInt{$W}(), l, has_feature(Val(:x86_64_bmi)))),
    )
    # mask_shift_quote(W)
    # elseif (Base.libllvm_version ≥ v"11") && ispow2(W)
    # elseif ((Sys.ARCH ≢ :x86_64) && (Sys.ARCH ≢ :i686)) && (Base.libllvm_version ≥ v"11") && ispow2(W)
  elseif false
    # cmpval = Base.libllvm_version ≥ v"12" ? -one(I) : zero(I)
    active_lane_mask_quote(W)
  else
    Expr(
      :block,
      Expr(:meta, :inline),
      :(_mask_cmp(
        Val{$W}(),
        l,
        simd_integer_register_size(),
        has_feature(Val(:x86_64_bmi)),
      )),
    )
  end
end
# This `mask` method returns a constant, independent of `has_opmask_registers()`; that only effects method of calculating
# the constant. So it'd be safe to bake in a value.
@inline mask(::Union{Val{W},StaticInt{W}}, L) where {W} =
  _mask(StaticInt(W), L, has_feature(Val(:x86_64_avx512f)) & ge_one_fma(cpu_name()))
@inline mask(::Union{Val{W},StaticInt{W}}, ::StaticInt{L}) where {W,L} =
  _mask(StaticInt(W), L, has_feature(Val(:x86_64_avx512f)) & ge_one_fma(cpu_name()))
@inline mask(::Type{T}, l::Union{Integer,StaticInt}) where {T} =
  _mask(pick_vector_width(T), l, has_feature(Val(:x86_64_avx512f)) & ge_one_fma(cpu_name()))

# @generated function masktable(::Union{Val{W},StaticInt{W}}, rem::Union{Integer,StaticInt}) where {W}
#     masks = Expr(:tuple)
#     for w ∈ 0:W-1
#         push!(masks.args, data(mask(Val(W), w == 0 ? W : w)))
#     end
#     Expr(
#         :block,
#         Expr(:meta,:inline),
#         Expr(:call, Expr(:curly, :Mask, W), Expr(
#             :macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)),
#             Expr(:call, :getindex, masks, Expr(:call, :+, 1, Expr(:call, :valrem, Expr(:call, Expr(:curly, W)), :rem)))
#         ))
#     )
# end

@inline tomask(m::Unsigned) = Mask{sizeof(m)}(m)
@inline tomask(m::Mask) = m
@generated function tomask(v::Vec{W,Bool}) where {W}
  usize = W > 8 ? nextpow2(W) : 8
  utyp = "i$(usize)"
  U = mask_type_symbol(W)
  instrs = String[]
  push!(instrs, "%bitvec = trunc <$W x i8> %0 to <$W x i1>")
  zext_mask!(instrs, "bitvec", W, 0)
  push!(instrs, "ret i$(usize) %res.0")
  quote
    $(Expr(:meta, :inline))
    Mask{$W}($LLVMCALL($(join(instrs, "\n")), $U, Tuple{_Vec{$W,Bool}}, data(v)))
  end
end
@inline tomask(v::AbstractSIMDVector{W,Bool}) where {W} =
  tomask(vconvert(Vec{W,Bool}, data(v)))
# @inline tounsigned(m::Mask) = getfield(m, :u)
# @inline tounsigned(m::Vec{W,Bool}) where {W} = getfield(tomask(m), :u)
@inline tounsigned(v) = getfield(tomask(v), :u)

@generated function vrem(m::Mask{W,U}, ::Type{I}) where {W,U,I<:IntegerTypesHW}
  bits = 8sizeof(I)
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  push!(
    instrs,
    "%res = zext <$W x i1> %mask.0 to <$W x i$(bits)>\nret <$W x i$(bits)> %res",
  )
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($(join(instrs, "\n")), _Vec{$W,$I}, Tuple{$U}, data(m)))
  end
end
Vec(m::Mask{W}) where {W} = m % int_type(Val{W}())

# @inline getindexzerobased(m::Mask, i) = (getfield(m, :u) >>> i) % Bool
# @inline function extractelement(m::Mask{W}, i::Union{Integer,StaticInt}) where {W}
#     @boundscheck i > W && throw(BoundsError(m, i))
#     getindexzerobased(m, i)
# end
@generated function extractelement(v::Mask{W,U}, i::I) where {W,U,I}
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  push!(instrs, "%res1 = extractelement <$W x i1> %mask.0, i$(8sizeof(I)) %1")
  push!(instrs, "%res8 = zext i1 %res1 to i8\nret i8 %res8")
  instrs_string = join(instrs, "\n")
  call = :($LLVMCALL($instrs_string, Bool, Tuple{$U,$I}, data(v), i))
  Expr(:block, Expr(:meta, :inline), call)
end
@generated function insertelement(
  v::Mask{W,U},
  x::T,
  i::I,
) where {W,T,U,I<:Union{Bool,IntegerTypesHW}}
  mtyp_input = "i$(max(8,nextpow2(W)))"
  instrs = String["%bit = trunc i$(8sizeof(T)) %1 to i1"]
  truncate_mask!(instrs, '0', W, 0)
  push!(instrs, "%bitvec = insertelement <$W x i1> %mask.0, i1 %bit, i$(8sizeof(I)) %2")
  zext_mask!(instrs, "bitvec", W, 1)
  push!(instrs, "ret $(mtyp_input) %res.1")
  instrs_string = join(instrs, "\n")
  call = :(Mask{$W}($LLVMCALL($instrs_string, $U, Tuple{$U,$T,$I}, data(v), x, i)))
  Expr(:block, Expr(:meta, :inline), call)
end


# @generated function Base.isodd(i::MM{W,1}) where {W}
#     U = mask_type(W)
#     evenfirst = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa % U
#     # Expr(:block, Expr(:meta, :inline), :(isodd(getfield(i, :i)) ? Mask{$W}($oddfirst) : Mask{$W}($evenfirst)))
#     Expr(:block, Expr(:meta, :inline), :(Mask{$W}($evenfirst >> (getfield(i, :i) & 0x03))))
# end
# @generated function Base.iseven(i::MM{W,1}) where {W}
#     U = mask_type(W)
#     oddfirst = 0x55555555555555555555555555555555 % U
#     # evenfirst = oddfirst << 1
#     # Expr(:block, Expr(:meta, :inline), :(isodd(getfield(i, :i)) ? Mask{$W}($evenfirst) : Mask{$W}($oddfirst)))
#     Expr(:block, Expr(:meta, :inline), :(Mask{$W}($oddfirst >> (getfield(i, :i) & 0x03))))
# end
@inline Base.isodd(i::MM{W,1}) where {W} = Mask{W}(
  (0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa % mask_type(Val{W}())) >>> (getfield(i, :i) & 0x03),
)
@inline Base.iseven(i::MM{W,1}) where {W} = Mask{W}(
  (0x55555555555555555555555555555555 % mask_type(Val{W}())) >>> (getfield(i, :i) & 0x03),
)

function cmp_quote(W, cond, vtyp, T1, T2 = T1)
  instrs = String["%m = $cond $vtyp %0, %1"]
  zext_mask!(instrs, 'm', W, '0')
  push!(instrs, "ret i$(max(8,nextpow2(W))) %res.0")
  U = mask_type_symbol(W)
  quote
    $(Expr(:meta, :inline))
    Mask{$W}(
      $LLVMCALL(
        $(join(instrs, "\n")),
        $U,
        Tuple{_Vec{$W,$T1},_Vec{$W,$T2}},
        data(v1),
        data(v2),
      ),
    )
  end
end
function icmp_quote(W, cond, bytes, T1, T2 = T1)
  vtyp = vtype(W, "i$(8bytes)")
  cmp_quote(W, "icmp " * cond, vtyp, T1, T2)
end
function fcmp_quote(W, cond, T)
  vtyp = vtype(W, T === Float32 ? "float" : "double")
  cmp_quote(W, "fcmp nsz arcp contract reassoc " * cond, vtyp, T)
end
# @generated function compare(::Val{cond}, v1::Vec{W,I}, v2::Vec{W,I}) where {cond, W, I}
# cmp_quote(W, cond, sizeof(I), I)
# end
# for (f,cond) ∈ [(:(==), :eq), (:(!=), :ne), (:(>), :ugt), (:(≥), :uge), (:(<), :ult), (:(≤), :ule)]
for (f, cond) ∈ [(:veq, "eq"), (:vne, "ne")]
  @eval @generated function $f(
    v1::Vec{W,T1},
    v2::Vec{W,T2},
  ) where {W,T1<:IntegerTypesHW,T2<:IntegerTypesHW}
    if sizeof(T1) != sizeof(T2)
      return Expr(
        :block,
        Expr(:meta, :inline),
        :((v3, v4) = promote(v1, v2)),
        Expr(:call, $f, :v3, :v4),
      )
    end
    icmp_quote(W, $cond, sizeof(T1), T1, T2)
  end
end
for (f, cond) ∈ [(:vgt, "ugt"), (:vge, "uge"), (:vlt, "ult"), (:vle, "ule")]
  @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Unsigned}
    icmp_quote(W, $cond, sizeof(T), T)
  end
end
for (f, cond) ∈ [(:vgt, "sgt"), (:vge, "sge"), (:vlt, "slt"), (:vle, "sle")]
  @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Signed}
    icmp_quote(W, $cond, sizeof(T), T)
  end
end

# for (f,cond) ∈ [(:veq, "oeq"), (:vgt, "ogt"), (:vge, "oge"), (:vlt, "olt"), (:vle, "ole"), (:vne, "one")]
for (f, cond) ∈ [
  (:veq, "oeq"),
  (:vgt, "ogt"),
  (:vge, "oge"),
  (:vlt, "olt"),
  (:vle, "ole"),
  (:vne, "une"),
]
  # for (f,cond) ∈ [(:veq, "ueq"), (:vgt, "ugt"), (:vge, "uge"), (:vlt, "ult"), (:vle, "ule"), (:vne, "une")]
  @eval @generated function $f(
    v1::Vec{W,T},
    v2::Vec{W,T},
  ) where {W,T<:Union{Float32,Float64}}
    fcmp_quote(W, $cond, T)
  end
end

@inline function vgt(
  v1::AbstractSIMDVector{W,S},
  v2::AbstractSIMDVector{W,U},
) where {W,S<:SignedHW,U<:UnsignedHW}
  (v1 > zero(S)) & (vconvert(U, v1) > v2)
end
@inline function vgt(
  v1::AbstractSIMDVector{W,U},
  v2::AbstractSIMDVector{W,S},
) where {W,S<:SignedHW,U<:UnsignedHW}
  (v2 < zero(S)) | (vconvert(S, v1) > v2)
end

@inline function vge(
  v1::AbstractSIMDVector{W,S},
  v2::AbstractSIMDVector{W,U},
) where {W,S<:SignedHW,U<:UnsignedHW}
  (v1 ≥ zero(S)) & (vconvert(U, v1) ≥ v2)
end
@inline function vge(
  v1::AbstractSIMDVector{W,U},
  v2::AbstractSIMDVector{W,S},
) where {W,S<:SignedHW,U<:UnsignedHW}
  (v2 < zero(S)) | (vconvert(S, v1) ≥ v2)
end

@inline vlt(
  v1::AbstractSIMDVector{W,S},
  v2::AbstractSIMDVector{W,U},
) where {W,S<:SignedHW,U<:UnsignedHW} = vgt(v2, v1)
@inline vlt(
  v1::AbstractSIMDVector{W,U},
  v2::AbstractSIMDVector{W,S},
) where {W,S<:SignedHW,U<:UnsignedHW} = vgt(v2, v1)
@inline vle(
  v1::AbstractSIMDVector{W,S},
  v2::AbstractSIMDVector{W,U},
) where {W,S<:SignedHW,U<:UnsignedHW} = vge(v2, v1)
@inline vle(
  v1::AbstractSIMDVector{W,U},
  v2::AbstractSIMDVector{W,S},
) where {W,S<:SignedHW,U<:UnsignedHW} = vge(v2, v1)
for (op, f) ∈ [(:vgt, :(>)), (:vge, :(≥)), (:vlt, :(<)), (:vle, :(≤))]
  @eval begin
    @inline function $op(
      v1::V1,
      v2::V2,
    ) where {
      V1<:Union{IntegerTypesHW,AbstractSIMDVector{<:Any,<:IntegerTypesHW}},
      V2<:Union{IntegerTypesHW,AbstractSIMDVector{<:Any,<:IntegerTypesHW}},
    }
      V3 = promote_type(V1, V2)
      $op(itosize(v1, V3), itosize(v2, V3))
    end
    @inline function $op(v1, v2)
      v3, v4 = promote(v1, v2)
      $op(v3, v4)
    end
    @inline $op(s1::IntegerTypesHW, s2::IntegerTypesHW) = $f(s1, s2)
    @inline $op(s1::Union{Float32,Float64}, s2::Union{Float32,Float64}) = $f(s1, s2)
  end
end
for (op, f) ∈ [(:veq, :(==)), (:vne, :(≠))]
  @eval begin
    @inline $op(a, b) = ((c, d) = promote(a, b); $op(c, d))
    @inline $op(s1::NativeTypes, s2::NativeTypes) = $f(s1, s2)
  end
end

@generated function vifelse(m::AbstractMask{W,U}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,U,T}
  typ = LLVM_TYPES[T]
  vtyp = vtype(W, typ)
  selty = vtype(W, "i1")
  f = "select"
  if Base.libllvm_version ≥ v"9" && ((T === Float32) || (T === Float64))
    f *= " nsz arcp contract reassoc"
  end
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  push!(instrs, "%res = $f $selty %mask.0, $vtyp %1, $vtyp %2\nret $vtyp %res")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        _Vec{$W,$T},
        Tuple{$U,_Vec{$W,$T},_Vec{$W,$T}},
        data(m),
        data(v1),
        data(v2),
      ),
    )
  end
end

@inline vifelse(m::Vec{W,Bool}, s1::T, s2::T) where {W,T<:NativeTypes} =
  vifelse(m, Vec{W,T}(s1), Vec{W,T}(s2))
@inline vifelse(m::AbstractMask{W}, s1::T, s2::T) where {W,T<:NativeTypes} =
  vifelse(m, Vec{W,T}(s1), Vec{W,T}(s2))
@inline vifelse(m::AbstractMask{W,U}, s1, s2) where {W,U} =
  ((x1, x2) = promote(s1, s2); vifelse(m, x1, x2))
@inline vifelse(m::AbstractMask{W}, v1::VecUnroll{N,W}, v2::VecUnroll{N,W}) where {N,W} =
  VecUnroll(fmap(vifelse, m, getfield(v1, :data), getfield(v2, :data)))

@inline Base.Bool(m::AbstractMask{1,UInt8}) = (getfield(m, :u) & 0x01) === 0x01
@inline vconvert(::Type{Bool}, m::AbstractMask{1,UInt8}) = (getfield(m, :u) & 0x01) === 0x01
@inline vifelse(m::AbstractMask{1}, s1::T, s2::T) where {T<:NativeTypes} =
  Base.ifelse(Bool(m), s1, s2)
@inline vifelse(
  f::F,
  m::AbstractSIMD{W,B},
  a::Vararg{NativeTypesV,K},
) where {F<:Function,K,W,B<:Union{Bool,Bit}} = vifelse(m, f(a...), a[K])
@inline vifelse(f::F, m::Bool, a::Vararg{NativeTypesV,K}) where {F<:Function,K} =
  ifelse(m, f(a...), a[K])

@inline vconvert(::Type{EVLMask{W,U}}, b::Bool) where {W,U} = b & max_mask(StaticInt{W}())

@inline vifelse(m::AbstractMask{W}, a::AbstractMask{W}, b::AbstractMask{W}) where {W} =
  bitselect(m, b, a)

@inline Base.isnan(v::AbstractSIMD) = v != v
@inline Base.isfinite(x::AbstractSIMD) = iszero(x - x)

@inline Base.flipsign(x::AbstractSIMD, y::AbstractSIMD) = vifelse(y > zero(y), x, -x)
for T ∈ [:Float32, :Float64]
  @eval begin
    @inline Base.flipsign(x::AbstractSIMD, y::$T) = vifelse(y > zero(y), x, -x)
    @inline Base.flipsign(x::$T, y::AbstractSIMD) = vifelse(y > zero(y), x, -x)
  end
end
@inline Base.flipsign(x::AbstractSIMD, y::Real) = ifelse(y > zero(y), x, -x)
@inline Base.flipsign(x::Real, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
@inline Base.flipsign(x::Signed, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
@inline Base.isodd(x::AbstractSIMD{W,T}) where {W,T<:Union{Integer,StaticInt}} = (x & one(T)) != zero(T)
@inline Base.iseven(x::AbstractSIMD{W,T}) where {W,T<:Union{Integer,StaticInt}} = (x & one(T)) == zero(T)

@generated function vifelse(m::Vec{W,Bool}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,T}
  typ = LLVM_TYPES[T]
  vtyp = vtype(W, typ)
  selty = vtype(W, "i1")
  f = "select"
  if Base.libllvm_version ≥ v"9" && ((T === Float32) || (T === Float64))
    f *= " nsz arcp contract reassoc"
  end
  instrs = String["%mask.0 = trunc <$W x i8> %0 to <$W x i1>"]
  push!(instrs, "%res = $f $selty %mask.0, $vtyp %1, $vtyp %2\nret $vtyp %res")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        _Vec{$W,$T},
        Tuple{_Vec{$W,Bool},_Vec{$W,$T},_Vec{$W,$T}},
        data(m),
        data(v1),
        data(v2),
      ),
    )
  end
end
@inline vifelse(b::Bool, w, x) = ((y, z) = promote(w, x); vifelse(b, y, z))
@inline vifelse(b::Bool, w::T, x::T) where {T<:Union{NativeTypes,AbstractSIMDVector}} =
  Core.ifelse(b, w, x)
@inline vifelse(b::Bool, w::T, x::T) where {T<:VecUnroll} =
  VecUnroll(fmap(Core.ifelse, b, getfield(w, :data), getfield(x, :data)))

@generated function vifelse(
  m::AbstractMask{W},
  vu1::VecUnroll{Nm1,Wsplit},
  vu2::VecUnroll{Nm1,Wsplit},
) where {W,Wsplit,Nm1}
  N = Nm1 + 1
  @assert N * Wsplit == W
  U = mask_type_symbol(Wsplit)
  quote
    $(Expr(:meta, :inline))
    vifelse(vconvert(VecUnroll{$Nm1,$Wsplit,Bit,Mask{$Wsplit,$U}}, m), vu1, vu2)
  end
end

@inline vmul(v::AbstractSIMDVector, m::AbstractMask) = vifelse(m, v, zero(v))
@inline vmul(m::AbstractMask, v::AbstractSIMDVector) = vifelse(m, v, zero(v))
@inline vmul(m1::AbstractMask, m2::AbstractMask) = m1 & m2
@inline vmul(v::AbstractSIMDVector, b::Bool) = b ? v : zero(v)
@inline vmul(b::Bool, v::AbstractSIMDVector) = b ? v : zero(v)
@inline vmul(v::VecUnroll{N,W,T}, b::Bool) where {N,W,T} = b ? v : zero(v)
@inline vmul(b::Bool, v::VecUnroll{N,W,T}) where {N,W,T} = b ? v : zero(v)



@static if Base.libllvm_version ≥ v"11"
  """
    mask(::Union{StaticInt{W},Val{W}}, base, N)
    mask(base::MM{W}, N)

  The two arg (`base`, `N`) method takes a base (current index) and last index of a loop.
  Idiomatic use for three-arg version may look like

  ```julia
  using VectorizationBase
  sp = stridedpointer(x);
  for i ∈ 1:8:N
      m = mask(Val(8), (MM{8}(i),), N) # if using an integer base, also needs a `Val` or `StaticInt` to indicate size.
      v = vload(sp, (MM{8}(i),), m)
      # do something with `v`
  end
  ```
  or, a full runnable example:
  ```julia
  using VectorizationBase, SLEEFPirates
  x = randn(117); y = similar(x);
  function vexp!(y, x)
      W = VectorizationBase.pick_vector_width(eltype(x));
      L = length(y);
      spx = stridedpointer(x); spy = stridedpointer(y);
      i = MM(W, 1); # use an `MM` index.
      while (m = mask(i,L); m !== VectorizationBase.zero_mask(W))
          yᵢ = exp(vload(spx, (i,), m))
          vstore!(spy, yᵢ, (i,), m)
          i += W
      end
  end

  vexp!(y, x)
  @assert y ≈ exp.(x)

  # A sum optimized for short vectors (e.g., 10-20 elements)
  function simd_sum(x)
      W = VectorizationBase.pick_vector_width(eltype(x));
      L = length(x);
      spx = stridedpointer(x);
      i = MM(W, 1); # use an `MM` index.
      s = VectorizationBase.vzero(W, eltype(x))
      while (m = mask(i,L); m !== VectorizationBase.zero_mask(W))
          s += vload(spx, (i,), m)
          i += W
      end
      VectorizationBase.vsum(s)
  end
  # or
  function simd_sum(x)
      W = VectorizationBase.pick_vector_width(eltype(x));
      L = length(x);
      spx = stridedpointer(x);
      i = MM(W, 1); # use an `MM` index.
      s = VectorizationBase.vzero(W, eltype(x))
      cond = true
      m = mask(i,L)
      while cond
          s += vload(spx, (i,), m)
          i += W
          m = mask(i,L)
          cond = m !== VectorizationBase.zero_mask(W)
      end
      VectorizationBase.vsum(s)
  end
  ```

  ```julia
  julia> VectorizationBase.mask(Val(8), 1, 6) # starting with `i = 1`, if vector is of length 6, 6 lanes are on
  Mask{8,Bool}<1, 1, 1, 1, 1, 1, 0, 0>

  julia> VectorizationBase.mask(Val(8), 81, 93) # if `i = 81` and the vector is of length 93, we want all lanes on.
  Mask{8,Bool}<1, 1, 1, 1, 1, 1, 1, 1>

  julia> VectorizationBase.mask(Val(8), 89, 93) # But after `i += 8`, we're at `i = 89`, and now want just 5 lanes on.
  Mask{8,Bool}<1, 1, 1, 1, 1, 0, 0, 0>
  ```
  """
  @generated function mask(
    ::Union{Val{W},StaticInt{W}},
    base::T,
    N::T,
  ) where {W,T<:IntegerTypesHW}
    # declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i64(i64 %base, i64 %n)
    bits = 8sizeof(T)
    typ = "i$(bits)"
    decl = "declare <$W x i1> @llvm.get.active.lane.mask.v$(W)i1.$(typ)($(typ), $(typ))"
    instrs = [
      "%m = call <$W x i1> @llvm.get.active.lane.mask.v$(W)i1.$(typ)($(typ) %0, $(typ) %1)",
    ]
    zext_mask!(instrs, 'm', W, 0)
    push!(instrs, "ret i$(max(nextpow2(W),8)) %res.0")
    args = [:base, :N]
    call = llvmcall_expr(
      decl,
      join(instrs, "\n"),
      mask_type_symbol(W),
      :(Tuple{$T,$T}),
      "i$(max(nextpow2(W),8))",
      [typ, typ],
      args,
      true,
    )
    Expr(
      :block,
      Expr(:meta, :inline),
      :(EVLMask{$W}($call, ((N % UInt32) - (base % UInt32)) + 0x00000001)),
    )
  end
  @inline mask(i::MM{W}, N::T) where {W,T<:IntegerTypesHW} =
    mask(Val{W}(), getfield(i, :i), N)
end


@generated function Base.vcat(m1::AbstractMask{_W1}, m2::AbstractMask{_W2}) where {_W1,_W2}
  if _W1 == _W2
    W = _W1
  else
    W = _W1 + _W2
    U = integer_of_bytes_symbol(W, true)
    return Expr(
      :block,
      Expr(:meta, :inline),
      :(Mask{$W}(((data(m1) % $U) << $_W2) | (data(m2) % $U))),
    )
  end
  mtyp_input = "i$(max(8,nextpow2(W)))"
  instrs = String[]
  truncate_mask!(instrs, '0', W, 0)
  truncate_mask!(instrs, '1', W, 1)

  W2 = W + W
  shuffmask = Vector{String}(undef, W2)
  for w ∈ eachindex(shuffmask)
    shuffmask[w] = string(w - 1)
  end
  mask = '<' * join(map(x -> string("i32 ", x), shuffmask), ", ") * '>'

  push!(
    instrs,
    "%combinedmask = shufflevector <$W x i1> %mask.0, <$W x i1> %mask.1, <$(W2) x i32> $mask",
  )

  mtyp_output = "i$(max(8,nextpow2(W2)))"
  zext_mask!(instrs, "combinedmask", W2, 1)
  push!(instrs, "ret $mtyp_output %res.1")
  instrj = join(instrs, "\n")
  U = mask_type_symbol(W)
  U2 = mask_type_symbol(W2)
  Expr(
    :block,
    Expr(:meta, :inline),
    :(Mask{$W2}($LLVMCALL($instrj, $U2, Tuple{$U,$U}, getfield(m1, :u), getfield(m2, :u)))),
  )
end
# @inline function Base.vcat(m1::AbstractMask{W}, m2::AbstractMask{W}) where {W}
#     U = mask_type(Val(W))
#     u1 = data(m1) % U
#     u2 = data(m2) % U
#     (u1 << W) | u2
# end

@inline ifelse(b::Bool, m1::Mask{W}, m2::Mask{W}) where {W} =
  Mask{W}(Core.ifelse(b, getfield(m1, :u), getfield(m2, :u)))
@inline ifelse(b::Bool, m1::Mask{W}, m2::EVLMask{W}) where {W} =
  Mask{W}(Core.ifelse(b, getfield(m1, :u), getfield(m2, :u)))
@inline ifelse(b::Bool, m1::EVLMask{W}, m2::Mask{W}) where {W} =
  Mask{W}(Core.ifelse(b, getfield(m1, :u), getfield(m2, :u)))
@inline ifelse(b::Bool, m1::EVLMask{W}, m2::EVLMask{W}) where {W} = EVLMask{W}(
  Core.ifelse(b, getfield(m1, :u), getfield(m2, :u)),
  Core.ifelse(b, getfield(m1, :evl), getfield(m2, :evl)),
)

@inline vconvert(::Type{<:AbstractMask{W}}, b::Bool) where {W} =
  b ? max_mask(Val(W)) : zero_mask(Val(W))
@inline vconvert(::Type{Mask{W}}, b::Bool) where {W} =
  b ? max_mask(Val(W)) : zero_mask(Val(W))
@inline vconvert(::Type{EVLMask{W}}, b::Bool) where {W} =
  b ? max_mask(Val(W)) : zero_mask(Val(W))

@inline Base.max(x::AbstractMask, y::AbstractMask) = x | y
@inline Base.min(x::AbstractMask, y::AbstractMask) = x & y
@inline Base.FastMath.max_fast(x::AbstractMask, y::AbstractMask) = x | y
@inline Base.FastMath.min_fast(x::AbstractMask, y::AbstractMask) = x & y
