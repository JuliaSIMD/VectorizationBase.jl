# unroll
@inline Base.Broadcast.broadcastable(u::Unroll) = (u,)


"""
Returns a vector of expressions for a set of unrolled indices.


"""
function unrolled_indicies(D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, X::Int)
  baseind = Expr(:tuple)
  for d in 1:D
    i = Expr(:call, :Zero)
    if d == AV && W > 1
      i = Expr(:call, Expr(:curly, :MM, W, X), i)
    end
    push!(baseind.args, i)
  end
  inds = Vector{Expr}(undef, N)
  inds[1] = baseind
  for n in 1:N-1
    ind = copy(baseind)
    i = Expr(:call, Expr(:curly, :StaticInt, n*F))
    if AU == AV && W > 1
      i = Expr(:call, Expr(:curly, :MM, W, X), i)
    end
    ind.args[AU] = i
    inds[n+1] = ind
  end
  inds
end

# This creates a generic expression that simply calls `vload` for each of the specified `Unroll`s without any fanciness.
function vload_unroll_quote(
  D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, X::Int, mask::Bool, align::Bool, rs::Int, vecunrollmask::Bool
  )
  t = Expr(:tuple)
  inds = unrolled_indicies(D, AU, F, N, AV, W, X)
  # TODO: Consider doing some alignment checks before accepting user's `align`?
  alignval = Expr(:call, align ? :True : :False)
  rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
  q = quote
    $(Expr(:meta, :inline))
    gptr = similar_no_offset(sptr, gep(pointer(sptr), data(u)))
  end
  vecunrollmask && push!(q.args, :(masktup = data(vm)))
  gf = GlobalRef(Core, :getfield)
  for n in 1:N
    l = Expr(:call, :_vload, :gptr, inds[n])
    if vecunrollmask
      push!(l.args, :($gf(masktup, $n, false)))
    elseif mask && (M % Bool)
      push!(l.args, :sm)
    end
    push!(l.args, alignval, rsexpr)
    M >>= 1
    push!(t.args, l)
  end
  push!(q.args, :(VecUnroll($t)))
  q
end
# so I could call `linear_index`, then
# `IT, ind_type, W, X, M, O = index_summary(I)`
# `gesp` to set offset multiplier (`M`) and offset (`O`) to `0`.
# call, to build extended load quote (`W` below is `W*N`):
# vload_quote_llvmcall(
#     T_sym::Symbol, I_sym::Symbol, ind_type::Symbol, W::Int, X::Int, M::Int, O::Int, mask::Bool, align::Bool, rs::Int, ret::Expr
# )

function interleave_memory_access(AU, C, F, X, UN, size_T, B)
  ((((AU == C) && (C > 0)) && (F == 1)) && (abs(X) == (UN*size_T)) && (B < 1))
end

# if either
# x = rand(3,L);
# foo(x[1,i],x[2,i],x[3,i])
# `(AU == 1) & (AV == 2) & (F == 1) & (stride(p,2) == N)`
# or
# x = rand(3L);
# foo(x[3i - 2], x[3i - 1], x[3i   ])
# Index would be `(MM{W,3}(1),)`
# so we have `AU == AV == 1`, but also `X == N == F`.
function shuffle_load_quote(
  ::Type{T}, integer_params::NTuple{9,Int}, ::Type{I}, align::Bool, rs::Int, MASKFLAG::UInt
  ) where {T,I}
  Sys.CPU_NAME === "znver1" && return nothing
  IT, ind_type, _W, _X, M, O = index_summary(I)
  size_T = sizeof(T)
  T_sym = JULIA_TYPES[T]
  I_sym = JULIA_TYPES[IT]

  _shuffle_load_quote(
    T_sym, size_T, integer_params, I_sym, ind_type, M, O, align, rs, MASKFLAG
  )
end
function _shuffle_load_quote(
  T_sym::Symbol, size_T::Int, integer_params::NTuple{9,Int}, I_sym::Symbol, ind_type::Symbol, M::Int, O::Int, align::Bool, rs::Int, MASKFLAG::UInt
)
  N, C, B, AU, F, UN, AV, W, X = integer_params
  # we don't require vector indices for `Unroll`s...
  # @assert _W == W "W from index $(_W) didn't equal W from Unroll $W."
  mask = MASKFLAG ≠ zero(UInt)
  if mask && ((MASKFLAG & ((one(UInt) << UN) - one(UInt))) ≠ ((one(UInt) << UN) - one(UInt)))
    return nothing
    # throw(ArgumentError("`shuffle_load_quote` currently requires masking either all or none of the unrolled loads."))
  end
  if mask && Base.libllvm_version < v"11"
    return nothing
  end
  # We need to unroll in a contiguous dimension for this to be a shuffle store, and we need the step between the start of the vectors to be `1`
  # @show X, UN, size_T
  ((AV > 0) && interleave_memory_access(AU, C, F, X, UN, size_T, B)) || return nothing
  Wfull = W * UN
  (mask && (Wfull > 128)) && return nothing
  # `X` is stride between indices, e.g. `X = 3` means our final vectors should be `<x[0], x[3], x[6], x[9]>`
  # We need `X` to equal the steps (the unrolling factor)
  vloadexpr = vload_quote_llvmcall(
    T_sym, I_sym, ind_type, Wfull, size_T, M, O, mask, align, rs, :(_Vec{$Wfull,$T_sym})
  )
  q = quote
    $(Expr(:meta,:inline))
    ptr = pointer(sptr)
    i = data(u)
  end
  X < 0 && push!(q.args, :(ptr -= $(size_T*(UN*(W-1)))))
  if mask
    return nothing
    if X > 0
      mask_expr = :(mask(StaticInt{$W}(), 0, vmul_nw($UN, getfield(sm, :evl))))
      for n ∈ 1:UN-1
        mask_expr = :(vcat($mask_expr, mask(StaticInt{$W}(), $(n*W), vmul_nw($UN, getfield(sm, :evl)))))
      end
      # push!(q.args, :(m = mask(StaticInt{$Wfull}(), vmul_nw($UN, getfield(sm, :evl)))))
    else
      # FIXME
      return nothing
      vrange = :(VectorizationBase.vrange(Val{$W}(),$(integer_of_bytes(min(size_T,rs÷W))),Val{0}(),Val{-1}()))
      mask_expr = :(($vrange + $(UN*W)) ≤ vmul_nw($UN, getfield(sm, :evl)))
      for n ∈ UN-1:-1:1
        mask_expr = :(vcat($mask_expr, ($vrange + $(n*W)) ≤ vmul_nw($UN, getfield(sm, :evl))))
      end
    end
    push!(q.args, :(m = $mask_expr))
  end
  push!(q.args, :(v = $vloadexpr))
  vut = Expr(:tuple)
  Wrange = X > 0 ? (0:1:W-1) : (W-1:-1:0)
  for n ∈ 0:UN-1
    shufftup = Expr(:tuple)
    for w ∈ Wrange
      push!(shufftup.args, n + UN*w)
    end
    push!(vut.args, :(shufflevector(v, Val{$shufftup}())))
  end
  push!(q.args, Expr(:call, :VecUnroll, vut))
  q
end
function init_transpose_memop_masking!(q::Expr, M::UInt, N::Int, evl::Bool)
  domask = M ≠ zero(UInt)
  if domask
    if (M & ((one(UInt) << N) - one(UInt))) ≠ ((one(UInt) << N) - one(UInt))
      throw(ArgumentError("`vload_transpose_quote` currently requires masking either all or none of the unrolled loads."))
    end
    if evl
      push!(q.args, :(_evl = getfield(sm, :evl)))
    else
      push!(q.args, :(u_1 = getfield(sm, :u)))
    end
  end
  domask
end
function push_transpose_mask!(q::Expr, mq::Expr, domask::Bool, n::Int, npartial::Int, w::Int, W::Int, evl::Bool, RS::Int, mask::UInt)
  Utyp = mask_type_symbol(n)
  if domask
    mw_w = Symbol(:mw_,w)
    if evl
      mm_evl_cmp = Symbol(:mm_evl_cmp_,n)
      if w == 1
        isym = integer_of_bytes_symbol(min(4, RS ÷ n))
        vmmtyp = :(VectorizationBase._vrange(Val{$n}(), $isym, Val{0}(), Val{1}()))
        push!(q.args, :($mm_evl_cmp = $vmmtyp))
        push!(q.args, :($mw_w = vmul_nw(_evl,$(UInt32(n))) > $mm_evl_cmp))
      else
        push!(q.args, :($mw_w = (vsub_nsw(vmul_nw(_evl,$(UInt32(n))), $(UInt32(n*(w-1))))%Int32) > ($mm_evl_cmp)))
      end
      if n == npartial
        push!(mq.args, mw_w )
      else
        push!(mq.args, :(Mask{$n}($mask % $Utyp) & $mw_w ))
      end
    else
      push!(q.args, :($mw_w = Core.ifelse($(Symbol(:u_,w)) % Bool, $mask % $Utyp, zero($Utyp))))
      if w < W
        push!(q.args, Expr(:(=), Symbol(:u_,w+1), Expr(:call, :(>>>), Symbol(:u_,w), 1)))
      end
      push!(mq.args, :(Mask{$n}( $mw_w )))
    end
  elseif n ≠ npartial
    push!(mq.args, :(Mask{$n}( $mask % $Utyp )))
  end
  nothing
end

function vload_transpose_quote(D::Int,AU::Int,F::Int,N::Int,AV::Int,W::Int,X::Int,align::Bool,RS::Int,st::Int,M::UInt,evl::Bool)
  ispow2(W) || throw(ArgumentError("Vector width must be a power of 2, but recieved $W."))
  isone(F) || throw(ArgumentError("No point to doing a transposed store if unroll step factor $F != 1"))
  C = AU # the point of tranposing
  q = Expr(:block, Expr(:meta,:inline), :(gptr = similar_no_offset(sptr, gep(pointer(sptr), data(u)))))
  domask = init_transpose_memop_masking!(q, M, N, evl)
  alignval = Expr(:call, align ? :True : :False)
  rsexpr = Expr(:call, Expr(:curly, :StaticInt, RS))
  vut = Expr(:tuple)
  # vds = Vector{Symbol}(undef, N)
  # for n ∈ 1:N
  #     vds[n] = vdn = Symbol(:vd_,n)
  #     push!(q.args, :($vdn = getfield(vd, $n, false)))
  # end
  # AU = 1, AV = 2, N = 3, W = 8, M = 0x7 (<1,1,1,0>), mask unknown, hypothetically 0x1f <1 1 1 1 1 0 0 0>
  # load 5 vetors of length 3, replace last 3 with undef
  i = 0
  Wmax = RS ÷ st
  while N > 0
    # for store, we do smaller unrolls first.
    # for load, we start with larger unrolls.
    # The idea is that this is probably a better order to make use of execution resources.
    # For load, we want to begin immediately issuing loads, but then we'd also like to do other work, e.g. shuffles at the same time.
    # Before issueing the loads, register pressure is also probably low, and it will be higher towards the end. Another reason to do transposes early.
    # For stores, we also want to begin immediately issueing stores, so we start with smaller unrolls so that there is less work
    # to do beforehand. This also immediately frees up registers for use while transposing, in case of register pressure.
    if N ≥ Wmax
      npartial = n = Wmax
      mask = -1 % UInt
    else
      npartial = N
      n = nextpow2(npartial)
      # if npartial == n
      #     mask = -1 % UInt
      # else
      mask = (one(UInt) << (npartial)) - one(UInt)
      # end
    end
    N -= npartial
    if n == 1
      # this can only happen on the first iter, so `StaticInt{0}()` is fine
      ind = Expr(:tuple)
      for d ∈ 1:D
        if AV == d
          push!(ind.args, :(MM{$W,$X}(StaticInt{0}())))
        elseif AU == d
          push!(ind.args, :(StaticInt{$i}()))
        else
          push!(ind.args, :(StaticInt{0}()))
        end
      end
      loadq = :(_vload(gptr, $ind))
      # we're not shuffling when `n == 1`, so we just push the mask
      domask && push!(loadq.args, :sm)
      push!(loadq.args, alignval, rsexpr)
      push!(q.args, :(vl_1 = $loadq))
      push!(vut.args, :vl_1)
    elseif W == 1
      loadq = loadq_expr!(q,D,AU,AV,n,i,X,W,W,domask,npartial,evl,RS,mask,alignval,rsexpr)
      loadsym = Symbol(:vloadw1_, i, :_, n)
      push!(q.args, Expr(:(=), loadsym, loadq))
      for nn ∈ 1:npartial
        push!(vut.args, :(extractelement($loadsym, $(nn-1))))
      end
    else
      # dname is a `VecUnroll{(W-1),N}`
      t = Expr(:tuple)
      dname = Symbol(:vud_,i,:_,n)
      for w ∈ 1:W
        # if domask, these get masked
        loadq = loadq_expr!(q,D,AU,AV,n,i,X,w,W,domask,npartial,evl,RS,mask,alignval,rsexpr)
        push!(t.args, loadq)
      end
      push!(q.args, :($dname = data(transpose_vecunroll(VecUnroll($t)))))
      for nn ∈ 1:npartial
        extract = :(getfield($dname, $nn, false))
        push!(vut.args, extract)
      end
    end
    # M >>>= 1
    i += npartial
  end
  push!(q.args, :(VecUnroll($vut)))
  q
end
function loadq_expr!(q,D,AU,AV,n,i,X,w,W,domask,npartial,evl,RS,mask,alignval,rsexpr)
  ind = Expr(:tuple)
  for d ∈ 1:D
    if AU == d
      push!(ind.args, :(MM{$n}(StaticInt{$i}())))
    elseif AV == d
      push!(ind.args, :(StaticInt{$(X*(w-1))}()))
    else
      push!(ind.args, :(StaticInt{0}()))
    end
  end
  # transposing mask does what?
  loadq = :(_vload(gptr, $ind))
  push_transpose_mask!(q, loadq, domask, n, npartial, w, W, evl, RS, mask)
  push!(loadq.args, alignval, rsexpr)
  loadq
end

# @inline staticunrolledvectorstride(_, __) = nothing
# @inline staticunrolledvectorstride(::StaticInt{M}, ::StaticInt{X}) where {M,X} = StaticInt{M}() * StaticInt{X}()
# @inline staticunrolledvectorstride(sp::AbstractStridedPointer, ::Unroll{AU,F,UN,AV,W,M,X}) where {AU,F,UN,AV,W,M,X} = staticunrolledvectorstride(strides(ptr)[AV], StaticInt{X}())

@generated function staticunrolledvectorstride(sptr::T, ::Unroll{AU,F,UN,AV,W,M,X}) where {T,AU,F,UN,AV,W,M,X}
  AV > 0 || return nothing
  SM = T.parameters[AV]
  if SM <: StaticInt
    return Expr(:block, Expr(:meta,:inline), Expr(:call, *, Expr(:call, SM), Expr(:call, Expr(:curly, :StaticInt, X))))
  else
    return nothing
  end
end

function should_transpose_memop(F,C,AU,AV,UN,M)
  (F == 1) & (C == AU) & (C ≠ AV) || return false
  max_mask = (one(UInt) << UN) - one(UInt)
  (M == zero(UInt)) | ((max_mask & M) == max_mask)
end

function bitload(AU::Int,W::Int,AV::Int,F::Int,UN::Int,RS::Int,mask::Bool)
  if AU ≠ AV
    1 < W < 8 && throw(ArgumentError("Must unroll in vectorized direction for `Bit` loads with W < 8."))
    return
  end
  if (1 < W < 8) && F ≠ W
    throw(ArgumentError("Must take steps of size $W along unrolled and vectorized axis when loading from bits."))
  end
  loadq = :(__vload(pointer(sptr), MM{$(W*UN)}(ind)))
  mask && push!(loadq.args, :sm)
  push!(loadq.args, :(False()), :(StaticInt{$RS}()))
  quote
    $(Expr(:meta,:inline))
    ind = getfield(u,:i)
    VecUnroll(splitvectortotuple(StaticInt{$UN}(),StaticInt{$W}(), $loadq))
  end
end

@generated function _vload_unroll(
  sptr::AbstractStridedPointer{T,N,C,B}, u::Unroll{AU,F,UN,AV,W,M,UX,I}, ::A, ::StaticInt{RS}, ::StaticInt{X}
) where {T<:NativeTypes,N,C,B,AU,F,UN,AV,W,M,UX,I<:IndexNoUnroll,A<:StaticBool,RS,X}
  1+2
  if T === Bit
    bitlq = bitload(AU,W,AV,F,UN,RS,false)
    bitlq === nothing || return bitlq
  end
  align = A === True
  should_transpose = should_transpose_memop(F,C,AU,AV,UN,zero(UInt64))
  if (W == N) & ((sizeof(T)*W) == RS) & should_transpose
    return vload_transpose_quote(N,AU,F,UN,AV,W,UX,align,RS,sizeof(T),zero(UInt),false)
  end
  maybeshufflequote = shuffle_load_quote(T, (N, C, B, AU, F, UN, AV, W, X), I, align, RS, zero(UInt))
  maybeshufflequote === nothing || return maybeshufflequote
  if should_transpose
    vload_transpose_quote(N,AU,F,UN,AV,W,UX,align,RS,sizeof(T),zero(UInt),false)
  else
    vload_unroll_quote(N, AU, F, UN, AV, W, M, UX, false, align, RS, false)
  end
end
@generated function _vload_unroll(
  sptr::AbstractStridedPointer{T,N,C,B}, u::Unroll{AU,F,UN,AV,W,M,UX,I}, ::A, ::StaticInt{RS}, ::Nothing
  ) where {T<:NativeTypes,N,C,B,AU,F,UN,AV,W,M,UX,I<:IndexNoUnroll,A<:StaticBool,RS}
  # 1+2
  # @show AU,F,UN,AV,W,M,UX,I
  if T === Bit
    bitlq = bitload(AU,W,AV,F,UN,RS,false)
    bitlq === nothing || return bitlq
  end
  should_transpose = should_transpose_memop(F,C,AU,AV,UN,zero(UInt64))
  if should_transpose
    vload_transpose_quote(N,AU,F,UN,AV,W,UX,A === True,RS,sizeof(T),zero(UInt),false)
  else
    vload_unroll_quote(N, AU, F, UN, AV, W, M, UX, false, A === True, RS, false)
  end
end
@generated function _vload_unroll(
  sptr::AbstractStridedPointer{T,D,C,B}, u::Unroll{AU,F,N,AV,W,M,UX,I}, sm::EVLMask{W}, ::A, ::StaticInt{RS}, ::StaticInt{X}
  ) where {A<:StaticBool,AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,RS,UX,X,C,B}
  if T === Bit
    bitlq = bitload(AU,W,AV,F,N,RS,true)
    bitlq === nothing || return bitlq
  end
  1+2
  should_transpose = should_transpose_memop(F,C,AU,AV,N,M)
  align = A === True
  if (W == N) & ((sizeof(T)*W) == RS) & should_transpose
    return vload_transpose_quote(D,AU,F,N,AV,W,UX,align,RS,sizeof(T),M,true)
  end
  # maybeshufflequote = shuffle_load_quote(T, (D, C, B, AU, F, N, AV, W, X), I, align, RS, M)
  # maybeshufflequote === nothing || return maybeshufflequote
  if should_transpose
    return vload_transpose_quote(D,AU,F,N,AV,W,UX,align,RS,sizeof(T),M,true)
  end
  vload_unroll_quote(D, AU, F, N, AV, W, M, UX, true, align, RS, false)
end
@generated function _vload_unroll(
  sptr::AbstractStridedPointer{T,D,C}, u::Unroll{AU,F,N,AV,W,M,UX,I}, sm::AbstractMask{W}, ::A, ::StaticInt{RS}, ::Any
  ) where {A<:StaticBool,AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,RS,UX,C}
  1+2
  if T === Bit
    bitlq = bitload(AU,W,AV,F,N,RS,true)
    bitlq === nothing || return bitlq
  end
  align = A === True
  if should_transpose_memop(F,C,AU,AV,N,M)
    isevl = sm <: EVLMask
    return vload_transpose_quote(D,AU,F,N,AV,W,UX,align,RS,sizeof(T),M,isevl)
  end
  vload_unroll_quote(D, AU, F, N, AV, W, M, UX, true, align, RS, false)
end
# @generated function _vload_unroll(
#     sptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,UX,I}, vm::VecUnroll{Nm1,W,B}, ::A, ::StaticInt{RS}, ::StaticInt{X}
# ) where {A<:StaticBool,AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,RS,UX,Nm1,B<:Union{Bool,Bit},X}
#     Nm1+1 == N || throw(ArgumentError("Nm1 + 1 = $(Nm1 + 1) ≠ $N = N"))
#     maybeshufflequote = shuffle_load_quote(T, (N, C, B, AU, F, UN, AV, W, X), I, align, RS, 2)
#     maybeshufflequote === nothing || return maybeshufflequote
#     vload_unroll_quote(D, AU, F, N, AV, W, M, UX, true, A === True, RS, true)
# end
@generated function _vload_unroll(
  sptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,UX,I}, vm::VecUnroll{Nm1,<:Any,<:Union{Bool,Bit}}, ::A, ::StaticInt{RS}, ::Any
  ) where {A<:StaticBool,AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,RS,UX,Nm1}
  Nm1+1 == N || throw(ArgumentError("Nm1 + 1 = $(Nm1 + 1) ≠ $N = N"))
  vload_unroll_quote(D, AU, F, N, AV, W, M, UX, true, A === True, RS, true)
end

@inline function _vload(ptr::AbstractStridedPointer, u::Unroll, ::A, ::StaticInt{RS}) where {A<:StaticBool,RS}
  p, li = linear_index(ptr, u)
  sptr = similar_no_offset(ptr, p)
  _vload_unroll(sptr, li, A(), StaticInt{RS}(), staticunrolledvectorstride(strides(ptr), u))
end
@inline function _vload(ptr::AbstractStridedPointer, u::Unroll, m::AbstractMask, ::A, ::StaticInt{RS}) where {A<:StaticBool,RS}
  p, li = linear_index(ptr, u)
  sptr = similar_no_offset(ptr, p)
  _vload_unroll(sptr, li, m, A(), StaticInt{RS}(), staticunrolledvectorstride(strides(ptr), u))
end
@inline function _vload(
    ptr::AbstractStridedPointer, u::Unroll, m::VecUnroll{Nm1,W,B}, ::A, ::StaticInt{RS}
) where {A<:StaticBool,RS,Nm1,W,B<:Union{Bool,Bit}}
  p, li = linear_index(ptr, u)
  sptr = similar_no_offset(ptr, p)
  _vload_unroll(sptr, li, m, A(), StaticInt{RS}(), staticunrolledvectorstride(strides(ptr), u))
end
@inline function _vload(ptr::AbstractStridedPointer{T}, u::Unroll{AU,F,N,AV,W}, m::Bool, ::A, ::StaticInt{RS}) where {A<:StaticBool,RS,AU,F,N,AV,W,T}
  if m
    _vload(ptr, u, A(), StaticInt{RS}())
  else
    zero_vecunroll(StaticInt{N}(), StaticInt{W}(), T, StaticInt{RS}())
  end
end
@generated function vload(r::FastRange{T}, u::Unroll{AU,F,N,AV,W,M,X,I}) where {AU,F,N,AV,W,M,X,I,T}
  _vload_fastrange_unroll(AU,F,N,AV,W,M,X,false,false)
end
@generated function vload(r::FastRange{T}, u::Unroll{AU,F,N,AV,W,M,X,I}, m::AbstractMask) where {AU,F,N,AV,W,M,X,I,T}
  _vload_fastrange_unroll(AU,F,N,AV,W,M,X,true,false)
end
@generated function vload(r::FastRange{T}, u::Unroll{AU,F,N,AV,W,M,X,I}, vm::VecUnroll{Nm1,<:Any,<:Union{Bool,Bit}}) where {AU,F,N,AV,W,M,X,I,T,Nm1}
  Nm1+1 == N || throw(ArgumentError("Nm1 + 1 = $(Nm1 + 1) ≠ $N = N"))
  _vload_fastrange_unroll(AU,F,N,AV,W,M,X,false,true)
end


function vstore_unroll_quote(
  D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, X::Int, mask::Bool, align::Bool, noalias::Bool, nontemporal::Bool, rs::Int, vecunrollmask::Bool
  )
  t = Expr(:tuple)
  inds = unrolled_indicies(D, AU, F, N, AV, W, X)
  q = quote
    $(Expr(:meta, :inline))
    gptr = similar_no_offset(sptr, gep(pointer(sptr), data(u)))
    # gptr = gesp(sptr, getfield(u, :i))
    t = data(vu)
  end
  alignval = Expr(:call, align ? :True : :False)
  noaliasval = Expr(:call, noalias ? :True : :False)
  nontemporalval = Expr(:call, nontemporal ? :True : :False)
  rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
  if vecunrollmask
    push!(q.args, :(masktup = data(vm)))
  end
  gf = GlobalRef(Core, :getfield)
  for n in 1:N
    l = Expr(:call, :_vstore!, :gptr, Expr(:call, gf, :t, n, false), inds[n])
    if vecunrollmask
      push!(l.args, :($gf(masktup, $n, false)))
    elseif mask && (M % Bool)
      push!(l.args, :sm)
    end
    push!(l.args, alignval, noaliasval, nontemporalval, rsexpr)
    M >>= 1
    push!(q.args, l)
  end
  q
end

function shuffle_store_quote(
  ::Type{T}, integer_params::NTuple{9,Int}, ::Type{I}, align::Bool, alias::Bool, notmp::Bool, rs::Int, mask::Bool
  ) where {T,I}
  Sys.CPU_NAME === "znver1" && return nothing
  IT, ind_type, _W, _X, M, O = index_summary(I)
  T_sym = JULIA_TYPES[T]
  I_sym = JULIA_TYPES[IT]
  size_T = sizeof(T)
  _shuffle_store_quote(
    T_sym, size_T, integer_params, I_sym, ind_type, M, O, align, alias, notmp, rs, mask
  )
end
function _shuffle_store_quote(
  T_sym::Symbol, size_T::Int, integer_params::NTuple{9,Int}, I_sym::Symbol, ind_type::Symbol, M::Int, O::Int, align::Bool, alias::Bool, notmp::Bool, rs::Int, mask::Bool
  )
  N, C, B, AU, F, UN, AV, W, X = integer_params
  
  # we don't require vector indices for `Unroll`s...
  # @assert _W == W "W from index $(_W) didn't equal W from Unroll $W."
  # We need to unroll in a contiguous dimension for this to be a shuffle store, and we need the step between the start of the vectors to be `1`
  interleave_memory_access(AU, C, F, X, UN, size_T, B) || return nothing
  (mask && (Base.libllvm_version < v"11")) && return nothing
  # `X` is stride between indices, e.g. `X = 3` means our final vectors should be `<x[0], x[3], x[6], x[9]>`
  # We need `X` to equal the steps (the unrolling factor)
  Wfull = W * UN
  (mask && (Wfull > 128)) && return nothing
  # the approach for combining is to keep concatenating vectors to double their length
  # until we hit ≥ half Wfull, then we `vresize` the remainder, and shuffle in the final combination before storing.

  # mask = false
  vstoreexpr = vstore_quote(
    T_sym, I_sym, ind_type, Wfull, size_T, M, O,
    mask, align, alias, notmp, rs
  )
  q = quote
    $(Expr(:meta,:inline))
    ptr = pointer(sptr)
    t = data(vu)
    i = data(u)
  end
  X < 0 && push!(q.args, :(ptr -= $(size_T*(UN*(W-1)))))
  syms = Vector{Symbol}(undef, UN)
  gf = GlobalRef(Core, :getfield)
  for n ∈ 1:UN
    syms[n] = vs = Symbol(:v_,n)
    push!(q.args, Expr(:(=), vs, Expr(:call, gf, :t, n, false)))
  end
  Wtemp = W
  Nvec = UN
  # first, we start concatenating vectors
  while 2Wtemp < Wfull
    Wnext = 2Wtemp
    Nvecnext = (Nvec >>> 1)
    for n ∈ 1:Nvecnext
      v1 = syms[2n-1]
      v2 = syms[2n  ]
      vc = Symbol(v1, :_, v2)
      push!(q.args, Expr(:(=), vc, Expr(:call, :vcat, v1, v2)))
      syms[n] = vc
    end
    if isodd(Nvec)
      syms[Nvecnext+1] = syms[Nvec]
      Nvec = Nvecnext + 1
    else
      Nvec = Nvecnext
    end
    Wtemp = Wnext
  end
  shufftup = Expr(:tuple)
  for w ∈ ((X > 0) ? (0:1:W-1) : (W-1:-1:0))
    for n ∈ 0:UN-1
      push!(shufftup.args, W*n + w)
    end
  end
  mask && push!(q.args, :(m = mask(StaticInt{$Wfull}(), vmul_nw($UN, $gf(sm, :evl)))))
  push!(q.args, Expr(:(=), :v, Expr(:call, :shufflevector, syms[1], syms[2], Expr(:call, Expr(:curly, :Val, shufftup)))))
  push!(q.args, vstoreexpr)
  q
end
function sparse_index_tuple(N, d, o)
  t = Expr(:tuple)
  for n ∈ 1:N
    if n == d
      push!(t.args, :(StaticInt{$o}()))
    else
      push!(t.args, :(StaticInt{0}()))
    end
  end
  t
end
function vstore_transpose_quote(D,AU,F,N,AV,W,X,align,alias,notmp,RS,st,Tsym,M,evl)
  ispow2(W) || throw(ArgumentError("Vector width must be a power of 2, but recieved $W."))
  isone(F) || throw(ArgumentError("No point to doing a transposed store if unroll step factor $F != 1"))
  C = AU # the point of tranposing
  q = Expr(
    :block, Expr(:meta,:inline), :(vd = data(vu)),
    :(gptr = similar_no_offset(sptr, gep(pointer(sptr), data(u))))
  )
  alignval = Expr(:call, align ? :True : :False)
  aliasval = Expr(:call, alias ? :True : :False)
  notmpval = Expr(:call, notmp ? :True : :False)
  rsexpr = Expr(:call, Expr(:curly, :StaticInt, RS))
  domask = init_transpose_memop_masking!(q, M, N, evl)

  vds = Vector{Symbol}(undef, N)
  for n ∈ 1:N
    vds[n] = vdn = Symbol(:vd_,n)
    push!(q.args, :($vdn = getfield(vd, $n, false)))
  end
  i = 0
  # Use `trailing_zeros` to decompose unroll amount, `N`, into a sum of powers-of-2
  Wmax = RS ÷ st
  while N > 0
    r = N % Wmax
    if r == 0
      npartial = n = Wmax
      mask = ~zero(UInt)
    else
      npartial = r
      n = nextpow2(npartial)
      if npartial == n
        mask = ~zero(UInt)
      else
        mask = (one(UInt) << (npartial)) - one(UInt)
      end
    end
    N -= npartial
    if n == 1
      # this can only happen on the first iter, so `StaticInt{0}()` is fine
      ind = Expr(:tuple)
      for d ∈ 1:D
        if AV == d
          push!(ind.args, :(MM{$W,$X}(StaticInt{0}())))
        else
          push!(ind.args, :(StaticInt{0}()))
        end
      end
      storeq = :(_vstore!(gptr, $(vds[1]), $ind))
      domask && push!(storeq.args, :sm)
      push!(storeq.args, alignval, aliasval, notmpval, rsexpr)
      push!(q.args, storeq)
      # elseif n < W
      # elseif n == W
    else
      t = Expr(:tuple)
      for nn ∈ 1:npartial
        push!(t.args, vds[i+nn])
      end
      for nn ∈ npartial+1:n
        # if W == 1
        #     push!(t.args, :(zero($Tsym)))
        # else
        push!(t.args, :(_vundef(StaticInt{$W}(), $Tsym)))
        # end
      end
      dname = Symbol(:vud_,i,:_,n)
      if W == 1
        push!(q.args, :($dname = transpose_vecunroll(VecUnroll($t))))
      else
        push!(q.args, :($dname = data(transpose_vecunroll(VecUnroll($t)))))
      end
      # dname is a `VecUnroll{(W-1),N}`
      for w ∈ 1:W
        ind = Expr(:tuple)
        for d ∈ 1:D
          if AU == d
            push!(ind.args, :(MM{$n}(StaticInt{$i}())))
          elseif AV == d
            push!(ind.args, :(StaticInt{$(X*(w-1))}()))
          else
            push!(ind.args, :(StaticInt{0}()))
          end
        end
        # transposing mask does what?
        storeq = if W == 1
          :(_vstore!(gptr, $dname, $ind))
        else
          :(_vstore!(gptr, getfield($dname, $w, false), $ind))
        end
        push_transpose_mask!(q, storeq, domask, n, npartial, w, W, evl, RS, mask)
        push!(storeq.args, alignval, aliasval, notmpval, rsexpr)
        push!(q.args, storeq)

      end
    end
    # M >>>= 1
    i += npartial
  end
  # @show
  q
end

@generated function _vstore_unroll!(
  sptr::AbstractStridedPointer{T,D,C,B}, vu::VecUnroll{Nm1,W,VUT,<:VecOrScalar}, u::Unroll{AU,F,N,AV,W,M,UX,I}, ::A, ::S, ::NT, ::StaticInt{RS}, ::StaticInt{X}
) where {AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,Nm1,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,C,B,UX,X,VUT}
  N == Nm1 + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(Nm1+1)."))
  VUT === T || return Expr(:block,Expr(:meta,:inline), :(_vstore_unroll!(sptr, vconvert($T,vu), u, $(A()), $(S()), $(NT()), $(StaticInt(RS)), $(StaticInt(X)))))
  if (T === Bit) && (F == W < 8) && (UX == 1) && (AV == AU == C > 0)
    return quote
      $(Expr(:meta,:inline))
      __vstore!(pointer(sptr), vu, MM{$(N*W)}(_materialize(data(u))), $A(), $S(), $NT(), StaticInt{$RS}())
    end
  end
  # 1+1
  align =  A === True
  alias =  S === True
  notmp = NT === True
  should_transpose = should_transpose_memop(F,C,AU,AV,N,zero(UInt64))
  if (W == N) & ((sizeof(T)*W) == RS) & should_transpose
        # should transpose means we'll transpose, but we'll only prefer it over the
        # `shuffle_store_quote` implementation if W == N, and we're using the entire register.
        # Otherwise, llvm's shuffling is probably more clever/efficient when the conditions for
        # `shuffle_store_quote` actually hold.
    return vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],zero(UInt),false)
  end
  maybeshufflequote = shuffle_store_quote(T,(D,C,B,AU,F,N,AV,W,X), I, align, alias, notmp, RS, false)
  maybeshufflequote === nothing || return maybeshufflequote
  if should_transpose
    vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],zero(UInt),false)
  else
    vstore_unroll_quote(D, AU, F, N, AV, W, M, UX, false, align, alias, notmp, RS, false)
  end
end
@generated function _vstore_unroll!(
  sptr::AbstractStridedPointer{T,D,C,B}, vu::VecUnroll{Nm1,W,VUT,<:VecOrScalar}, u::Unroll{AU,F,N,AV,W,M,UX,I}, ::A, ::S, ::NT, ::StaticInt{RS}, ::Nothing
) where {AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,Nm1,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,C,B,UX,VUT}
  VUT === T || return Expr(:block,Expr(:meta,:inline), :(_vstore_unroll!(sptr, vconvert($T,vu), u, $(A()), $(S()), $(NT()), $(StaticInt(RS)), nothing)))
  if (T === Bit) && (F == W < 8) && (UX == 1) && (AV == AU == C > 0)
    return quote
      $(Expr(:meta,:inline))
      __vstore!(pointer(sptr), vu, MM{$(N*W)}(_materialize(data(u))), $A(), $S(), $NT(), StaticInt{$RS}())
    end
  end
  align =  A === True
  alias =  S === True
  notmp = NT === True
  N == Nm1 + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(Nm1+1)."))
  if should_transpose_memop(F,C,AU,AV,N,zero(UInt64))
    vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],zero(UInt),false)
  else
    vstore_unroll_quote(D, AU, F, N, AV, W, M, UX, false, align, alias, notmp, RS, false)
  end
end
@generated function flattenmask(m::AbstractMask{W}, ::Val{M}, ::StaticInt{N}) where {W,N,M}
  WN = W*N
  MT = mask_type(WN)
  MTS = mask_type_symbol(WN)
  q = Expr(:block, Expr(:meta,:inline), :(u = zero($MTS)), :(mu = data(m)), :(mf = (one($MTS)<<$W)-one($MTS)))
  M = (bitreverse(M) >>> (8sizeof(M)-N))
  n = N
  while true
    push!(q.args, :(u |= $(M % Bool ? :mu : :mf)))
    (n -= 1) == 0 && break
    push!(q.args, :(u <<= $(MT(W))))
    M >>= 1
  end
  push!(q.args, :(Mask{$WN}(u)))
  q
end
@generated function flattenmask(vm::VecUnroll{Nm1,W,Bit}, ::Val{M}) where {W,Nm1,M}
  N = Nm1+1
  WN = W*N
  MT = mask_type(WN)
  MTS = mask_type_symbol(WN)
  q = Expr(:block, Expr(:meta,:inline), :(u = zero($MTS)), :(mu = data(vm)), :(mf = (one($MTS)<<$W)-one($MTS)))
  M = (bitreverse(M) >>> (8sizeof(M)-N))
  n = 0
  while true
    n += 1
    if M % Bool
      push!(q.args, :(u |= data(getfield(mu,$n))))
    else
      push!(q.args, :(u |= mf))
    end
    n == N && break
    push!(q.args, :(u <<= $(MT(W))))
    M >>= 1
  end
  push!(q.args, :(Mask{$WN}(u)))
  q
end
@generated function _vstore_unroll!(
  sptr::AbstractStridedPointer{T,D,C,B}, vu::VecUnroll{Nm1,W,VUT,VUV}, u::Unroll{AU,F,N,AV,W,M,UX,I}, sm::EVLMask{W},
  ::A, ::S, ::NT, ::StaticInt{RS},::StaticInt{X}
) where {AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,Nm1,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,UX,VUT,VUV<:VecOrScalar,X,B,C}
  N == Nm1 + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(Nm1+1)."))
  VUT === T || return Expr(:block,Expr(:meta,:inline), :(_vstore_unroll!(sptr, vconvert($T,vu), u, sm, $(A()), $(S()), $(NT()), $(StaticInt(RS)), $(StaticInt(X)))))
  if (T === Bit) && (F == W < 8) && (UX == 1) && (AV == AU == C > 0)
    return quote
      $(Expr(:meta,:inline))
      msk = flattenmask(sm, Val{$M}(), StaticInt{$N}())
      __vstore!(pointer(sptr), vu, MM{$(N*W)}(_materialize(data(u))), msk, $A(), $S(), $NT(), StaticInt{$RS}())
    end
  end
  align =  A === True
  alias =  S === True
  notmp = NT === True
  should_transpose = should_transpose_memop(F,C,AU,AV,N,M)
  if (W == N) & ((sizeof(T)*W) == RS) & should_transpose
    vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],M,true)
  end
  # maybeshufflequote = shuffle_store_quote(T,(D,C,B,AU,F,N,AV,W,X), I, align, alias, notmp, RS, true)
  # maybeshufflequote === nothing || return maybeshufflequote
  if should_transpose
    vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],M,true)
  else
    vstore_unroll_quote(D, AU, F, N, AV, W, M, UX, true, align, alias, notmp, RS, false)
  end
end
@generated function _vstore_unroll!(
  sptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{Nm1,W,VUT,VUV}, u::Unroll{AU,F,N,AV,W,M,UX,I}, sm::AbstractMask{W},
  ::A, ::S, ::NT, ::StaticInt{RS}, ::Any
) where {AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,Nm1,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,UX,VUT,VUV<:VecOrScalar,C}
  N == Nm1 + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(Nm1+1)."))
  VUT === T || return Expr(:block,Expr(:meta,:inline), :(_vstore_unroll!(sptr, vconvert($T,vu), u, sm, $(A()), $(S()), $(NT()), $(StaticInt(RS)), nothing)))
  if (T === Bit) && (F == W < 8) && (UX == 1) && (AV == AU == C > 0)
    return quote
      $(Expr(:meta,:inline))
      msk = flattenmask(sm, Val{$M}(), StaticInt{$N}())
      __vstore!(pointer(sptr), vu, MM{$(N*W)}(_materialize(data(u))), msk, $A(), $S(), $NT(), StaticInt{$RS}())
    end    
  end
  align =  A === True
  alias =  S === True
  notmp = NT === True
  if should_transpose_memop(F,C,AU,AV,N,M)
    vstore_transpose_quote(D,AU,F,N,AV,W,UX,align,alias,notmp,RS,sizeof(T),JULIA_TYPES[T],M,sm<:EVLMask)
  else
    vstore_unroll_quote(D, AU, F, N, AV, W, M, UX, true, A===True, S===True, NT===True, RS, false)
  end
end
# TODO: add `m::VecUnroll{Nm1,W,Bool}`
@generated function _vstore_unroll!(
    sptr::AbstractStridedPointer{T,D}, vu::VecUnroll{Nm1,W,VUT,VUV}, u::Unroll{AU,F,N,AV,W,M,UX,I}, vm::VecUnroll{Nm1,<:Any,B}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {AU,F,N,AV,W,M,I<:IndexNoUnroll,T,D,Nm1,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS,UX,B<:Union{Bit,Bool},VUT,VUV<:VecOrScalar}
  N == Nm1 + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(Nm1+1)."))
  VUT === T || return Expr(:block,Expr(:meta,:inline), :(_vstore_unroll!(sptr, vconvert($T,vu), u, vm, $(A()), $(S()), $(NT()), $(StaticInt(RS)))))
  if (T === Bit) && (F == W < 8) && (X == 1) && (AV == AU == C > 0)
    return quote
      $(Expr(:meta,:inline))
      msk = flattenmask(vm, Val{$M}())
      __vstore!(pointer(sptr), vu, MM{$(N*W)}(_materialize(data(u))), msk, $A(), $S(), $NT(), StaticInt{$RS}())
    end
  end
  vstore_unroll_quote(D, AU, F, N, AV, W, M, UX, true, A===True, S===True, NT===True, RS, true)
end

@inline function _vstore!(
    ptr::AbstractStridedPointer, vu::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
  p, li = linear_index(ptr, u)
  sptr = similar_no_offset(ptr, p)
  _vstore_unroll!(sptr, vu, li, A(), S(), NT(), StaticInt{RS}(), staticunrolledvectorstride(strides(sptr), u))
end
@inline function _vstore!(
  ptr::AbstractStridedPointer, vu::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W}, m::AbstractMask{W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
  p, li = linear_index(ptr, u)
  sptr = similar_no_offset(ptr, p)
  _vstore_unroll!(sptr, vu, li, m, A(), S(), NT(), StaticInt{RS}(), staticunrolledvectorstride(strides(sptr), u))
end
@inline function _vstore!(
    ptr::AbstractStridedPointer, vu::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W}, m::VecUnroll{Nm1,<:Any,B}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,Nm1,W,B<:Union{Bool,Bit},AU,F,N,AV}
    p, li = linear_index(ptr, u)
    sptr = similar_no_offset(ptr, p)
    # @show vu u m
    _vstore_unroll!(sptr, vu, li, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    ptr::AbstractStridedPointer, vu::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W}, m::Bool, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,AU,F,N,AV,W,Nm1}
    m && _vstore!(ptr, vu, u, A(), S(), NT(), StaticInt{RS}())
    nothing
end

@inline function _vstore!(
    sptr::AbstractStridedPointer, v::V, u::Unroll{AU,F,N,AV,W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,W,T,V<:AbstractSIMDVector{W,T},AU,F,N,AV}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),W,T,Vec{W,T}}, v), u, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    sptr::AbstractStridedPointer, v::V, u::Unroll{AU,F,N,AV,W}, m::Union{Bool,AbstractMask,VecUnroll}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,W,T,V<:AbstractSIMDVector{W,T},AU,F,N,AV}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),W,T,Vec{W,T}}, v), u, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, x::NativeTypes, u::Unroll{AU,F,N,AV,W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,W,T<:NativeTypes,AU,F,N,AV}
    # @show typeof(x), VecUnroll{Int(StaticInt{N}()-One()),W,T,Vec{W,T}}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),W,T,Vec{W,T}}, x), u, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, x::NativeTypes, u::Unroll{AU,F,N,AV,W}, m::Union{Bool,AbstractMask,VecUnroll}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,W,T<:NativeTypes,AU,F,N,AV}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),W,T,Vec{W,T}}, x), u, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, v::NativeTypes, u::Unroll{AU,F,N,-1,1}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),1,T,T}, v), u, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, v::NativeTypes, u::Unroll{AU,F,N,-1,1}, m::Union{Bool,AbstractMask,VecUnroll}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N}
    _vstore!(sptr, vconvert(VecUnroll{Int(StaticInt{N}()-One()),1,T,T}, v), u, m, A(), S(), NT(), StaticInt{RS}())
end

@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, vs::VecUnroll{Nm1,1}, u::Unroll{AU,F,N,AV,W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,W,AV}
    vb = _vbroadcast(StaticInt{W}(), vs, StaticInt{RS}())
    _vstore!(sptr, vb, u, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    ptr::AbstractStridedPointer{T}, vu::VecUnroll{Nm1,1}, u::Unroll{AU,F,N,AV,1}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,AV}
    p, li = linear_index(ptr, u)
    sptr = similar_no_offset(ptr, p)
    _vstore_unroll!(sptr, vu, li, A(), S(), NT(), StaticInt{RS}(), staticunrolledvectorstride(strides(sptr), u))
end
for M ∈ [:Bool, :AbstractMask]
    @eval begin
        @inline function _vstore!(
            sptr::AbstractStridedPointer{T}, vs::VecUnroll{Nm1,1,T,T}, u::Unroll{AU,F,N,AV,W}, m::$M, ::A, ::S, ::NT, ::StaticInt{RS}
        ) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,W,AV}
            vb = _vbroadcast(StaticInt{W}(), vs, StaticInt{RS}())
            _vstore!(sptr, vb, u, m, A(), S(), NT(), StaticInt{RS}())
        end
        @inline function _vstore!(
            ptr::AbstractStridedPointer{T}, vu::VecUnroll{Nm1,1,T,T}, u::Unroll{AU,F,N,AV,1,M}, m::$M, ::A, ::S, ::NT, ::StaticInt{RS}
        ) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,AV,M}
            p, li = linear_index(ptr, u)
            sptr = similar_no_offset(ptr, p)
            _vstore_unroll!(sptr, vu, li, m, A(), S(), NT(), StaticInt{RS}(), staticunrolledvectorstride(strides(sptr), u))
        end
    end
end
@inline function _vstore!(
    sptr::AbstractStridedPointer{T}, vs::VecUnroll{Nm1,1,T,T}, u::Unroll{AU,F,N,AV,W}, m::VecUnroll{Nm1,<:Any,<:Union{Bool,Bit}}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,W,AV}
    vb = _vbroadcast(StaticInt{W}(), vs, StaticInt{RS}())
    _vstore!(sptr, vb, u, m, A(), S(), NT(), StaticInt{RS}())
end
@inline function _vstore!(
    ptr::AbstractStridedPointer{T}, vu::VecUnroll{Nm1,1,T,T}, u::Unroll{AU,F,N,AV,1,M}, m::VecUnroll{Nm1,<:Any,<:Union{Bool,Bit}}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,T<:NativeTypes,AU,F,N,Nm1,AV,M}
    p, li = linear_index(ptr, u)
    sptr = similar_no_offset(ptr, p)
    _vstore_unroll!(sptr, vu, li, m, A(), S(), NT(), StaticInt{RS}())
end

function vload_double_unroll_quote(
    D::Int, NO::Int, NI::Int, AUO::Int, FO::Int, AV::Int, W::Int, MO::UInt,
    X::Int, C::Int, AUI::Int, FI::Int, MI::UInt, mask::Bool, A::Bool, RS::Int, svus::Int
)
    # UO + 1 ≠ NO && throw(ArgumentError("Outer unroll being stores is unrolled $(UO+1) times, but index indicates it was unrolled $NO times."))
    # UI + 1 ≠ NI && throw(ArgumentError("Inner unroll being stores is unrolled $(UI+1) times, but index indicates it was unrolled $NI times."))
    q = Expr(
        :block, Expr(:meta,:inline),
        :(id = getfield(getfield(u, :i), :i)),
        :(gptr = similar_no_offset(sptr, gep(pointer(sptr), id)))
    )
    aexpr = Expr(:call, A ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, RS))
    if (AUO == C) & ((AV ≠ C) | ((AV == C) & (X == NO))) # outer unroll is along contiguous axis, so we swap outer and inner
        # we loop over `UI+1`, constructing VecUnrolls "slices", and store those
        unroll = :(Unroll{$AUO,$FO,$NO,$AV,$W,$MO,$X}(Zero()))
        # tupvec = Vector{Expr}(undef, NI)
        vds = Vector{Symbol}(undef, NI)
        for ui ∈ 0:NI-1
            if ui == 0
                loadq = :(_vload_unroll(gptr, $unroll)) # VecUnroll($tup)
            else
                inds = sparse_index_tuple(D, AUI, ui*FI)
                loadq = :(_vload_unroll(gesp(gptr, $inds), $unroll)) # VecUnroll($tup)
            end
            if mask & (MI % Bool)
                push!(loadq.args, :m)
            end
            MI >>>= 1
            push!(loadq.args, aexpr, rsexpr)
            if svus == typemax(Int)
                push!(loadq.args, nothing)
            else
                push!(loadq.args, :(StaticInt{$svus}()))
            end
            vds[ui+1] = vul = Symbol(:vul_, ui)
            push!(q.args, Expr(:(=), vul, :(getfield($loadq, 1))))
        end
        otup = Expr(:tuple)
        for t ∈ 1:NO # transpose them
            tup = Expr(:tuple)
            # tup = ui == 0 ? Expr(:tuple) : tupvec[ui+1]
            for ui ∈ 1:NI
                # push!(tup.args, :(getfield($(vds[t]), $(ui+1), false)))
                push!(tup.args, :(getfield($(vds[ui]), $t, false)))
            end
            push!(otup.args, :(VecUnroll($tup)))
        end
        push!(q.args, :(VecUnroll($otup)))
    else # we loop over `UO+1` and do the loads
        unroll = :(Unroll{$AUI,$FI,$NI,$AV,$W,$MI,$X}(Zero()))
        tup = Expr(:tuple)
        for uo ∈ 0:NO-1
            if uo == 0
                loadq = :(_vload_unroll(gptr, $unroll))
            else
                inds = sparse_index_tuple(D, AUO, uo*FO)
                loadq = :(_vload_unroll(gesp(gptr, $inds), $unroll))
            end
            if mask & (MO % Bool)
                push!(loadq.args, :m)
            end
            MO >>>= 1
            push!(loadq.args, aexpr, rsexpr)
            if svus == typemax(Int)
                push!(loadq.args, nothing)
            else
                push!(loadq.args, :(StaticInt{$svus}()))
            end
            push!(tup.args, loadq)
        end
        push!(q.args, :(VecUnroll($tup)))
    end
    return q
end
@generated function _vload_unroll(
    sptr::AbstractStridedPointer{T,D,C}, u::UU, ::A, ::StaticInt{RS}, ::StaticInt{SVUS}
) where {T, A<:StaticBool,RS, D,C, SVUS, UU <: NestedUnroll}
    AUO,FO,NO,AV,W,MO,X,U = unroll_params(UU)
    AUI,FI,NI,AV,W,MI,X,I = unroll_params( U)
    vload_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, false, A === True, RS, SVUS)
end

@generated function _vload_unroll(
    sptr::AbstractStridedPointer{T,D,C}, u::UU, ::A, ::StaticInt{RS}, ::Nothing
) where {T, A<:StaticBool,RS, D,C, UU <: NestedUnroll}
    AUO,FO,NO,AV,W,MO,X,U = unroll_params(UU)
    AUI,FI,NI,AV,W,MI,X,I = unroll_params( U)
    vload_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, false, A === True, RS, typemax(Int))
end
@generated function _vload_unroll(
    sptr::AbstractStridedPointer{T,D,C}, u::UU, m::AbstractMask{W}, ::A, ::StaticInt{RS}, ::StaticInt{SVUS}
) where {W, T, A<:StaticBool,RS, D,C, SVUS, UU <: NestedUnroll{W}}
    AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
    AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
    vload_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, true, A === True, RS, SVUS)
end
@generated function _vload_unroll(
    sptr::AbstractStridedPointer{T,D,C}, u::UU, m::AbstractMask{W}, ::A, ::StaticInt{RS}, ::Nothing
) where {W, T, A<:StaticBool,RS, D,C, UU <: NestedUnroll{W}}
    AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
    AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
    vload_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, true, A === True, RS, typemax(Int))
end

# Unroll{AU,F,N,AV,W,M,X,I}
function vstore_double_unroll_quote(
    D::Int, NO::Int, NI::Int, AUO::Int, FO::Int, AV::Int, W::Int, MO::UInt,
    X::Int, C::Int, AUI::Int, FI::Int, MI::UInt, mask::Bool, A::Bool, S::Bool, NT::Bool, RS::Int, svus::Int
)
    # UO + 1 ≠ NO && throw(ArgumentError("Outer unroll being stores is unrolled $(UO+1) times, but index indicates it was unrolled $NO times."))
    # UI + 1 ≠ NI && throw(ArgumentError("Inner unroll being stores is unrolled $(UI+1) times, but index indicates it was unrolled $NI times."))
    q = Expr(
        :block, Expr(:meta,:inline),
        :(vd = getfield(v, :data)), :(id = getfield(getfield(u, :i), :i)),
        :(gptr = similar_no_offset(sptr, gep(pointer(sptr), id)))
    )
    aexpr = Expr(:call, A ? :True : :False)
    sexpr = Expr(:call, S ? :True : :False)
    ntexpr = Expr(:call, NT ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, RS))
    if (AUO == C) & ((AV ≠ C) | ((AV == C) & (X == NO))) # outer unroll is along contiguous axis, so we swap outer and inner
        # so we loop over `UI+1`, constructing VecUnrolls "slices", and store those
        unroll = :(Unroll{$AUO,$FO,$NO,$AV,$W,$MO,$X}(Zero()))
        vds = Vector{Symbol}(undef, NO)
        for t ∈ 1:NO
            vds[t] = vdt = Symbol(:vd_,t)
            push!(q.args, :($vdt = getfield(getfield(vd, $t, false), 1)))
        end
        # tupvec = Vector{Expr}(undef, NI)
        for ui ∈ 0:NI-1
            tup = Expr(:tuple)
            # tup = ui == 0 ? Expr(:tuple) : tupvec[ui+1]
            for t ∈ 1:NO
                # push!(tup.args, :(getfield($(vds[t]), $(ui+1), false)))
                push!(tup.args, :(getfield($(vds[t]), $(ui+1), false)))
            end
            # tupvec[ui+1] = tup
            if ui == 0
                storeq = :(_vstore_unroll!(gptr, VecUnroll($tup), $unroll))
            else
                inds = sparse_index_tuple(D, AUI, ui*FI)
                storeq = :(_vstore_unroll!(gesp(gptr, $inds), VecUnroll($tup), $unroll))
            end
            if mask & (MI % Bool)
                push!(storeq.args, :m)
            end
            MI >>>= 1
            push!(storeq.args, aexpr, sexpr, ntexpr, rsexpr)
            if svus == typemax(Int)
                push!(storeq.args, nothing)
            else
                push!(storeq.args, :(StaticInt{$svus}()))
            end
            push!(q.args, storeq)
        end
    else # we loop over `UO+1` and do the stores
        unroll = :(Unroll{$AUI,$FI,$NI,$AV,$W,$MI,$X}(Zero()))
        for uo ∈ 0:NO-1
            if uo == 0
                storeq = :(_vstore_unroll!(gptr, getfield(vd, 1, false), $unroll))
            else
                inds = sparse_index_tuple(D, AUO, uo*FO)
                storeq = :(_vstore_unroll!(gesp(gptr, $inds), getfield(vd, $(uo+1), false), $unroll))
            end
            if mask & (MO % Bool)
                push!(storeq.args, :m)
            end
            MO >>>= 1
            push!(storeq.args, aexpr, sexpr, ntexpr, rsexpr)
            if svus == typemax(Int)
                push!(storeq.args, nothing)
            else
                push!(storeq.args, :(StaticInt{$svus}()))
            end
            push!(q.args, storeq)
        end
    end
    return q
end

@inline function _vstore_unroll!(
  sptr::AbstractStridedPointer{T1,D,C}, v::VecUnroll{<:Any,W,T2,<:VecUnroll{<:Any,W,T2,Vec{W,T2}}},
  u::UU, ::A, ::S, ::NT, ::StaticInt{RS}, ::SVUS
) where {T1,D,C,W,T2,UU,A,S,NT,RS,SVUS}
  _vstore_unroll!(sptr, vconvert(T1, v), u, A(), S(), NT(), StaticInt{RS}(), SVUS())
end
@inline function _vstore_unroll!(
  sptr::AbstractStridedPointer{T1,D,C}, v::VecUnroll{<:Any,W,T2,<:VecUnroll{<:Any,W,T2,Vec{W,T2}}},
  u::UU, m::M, ::A, ::S, ::NT, ::StaticInt{RS}, ::SVUS
) where {T1,D,C,W,T2,UU,A,S,NT,RS,SVUS,M}
  _vstore_unroll!(sptr, vconvert(T1, v), u, m, A(), S(), NT(), StaticInt{RS}(), SVUS())
end
@generated function _vstore_unroll!(
    sptr::AbstractStridedPointer{T,D,C}, v::VecUnroll{<:Any,W,T,<:VecUnroll{<:Any,W,T,Vec{W,T}}},
    u::UU, ::A, ::S, ::NT, ::StaticInt{RS}, ::StaticInt{SVUS}
) where {W, T, A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS, D,C, SVUS, UU <: NestedUnroll{W}}
  AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
  AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
  vstore_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, false, A === True, S === True, NT === True, RS, SVUS)
end
@generated function _vstore_unroll!(
    sptr::AbstractStridedPointer{T,D,C}, v::VecUnroll{<:Any,W,T,<:VecUnroll{<:Any,W,T,Vec{W,T}}},
    u::UU, ::A, ::S, ::NT, ::StaticInt{RS}, ::Nothing
) where {W, T, A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS, D,C, UU <: NestedUnroll{W}}
  AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
  AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
  vstore_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, false, A === True, S === True, NT === True, RS, typemax(Int))
end
@generated function _vstore_unroll!(
    sptr::AbstractStridedPointer{T,D,C}, v::VecUnroll{<:Any,W,T,<:VecUnroll{<:Any,W,T,Vec{W,T}}},
    u::UU, m::AbstractMask{W}, ::A, ::S, ::NT, ::StaticInt{RS}, ::StaticInt{SVUS}
) where {W, T, A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS, D,C, SVUS, UU <: NestedUnroll{W}}
  AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
  AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
  vstore_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, true, A === True, S === True, NT === True, RS, SVUS)
end
@generated function _vstore_unroll!(
    sptr::AbstractStridedPointer{T,D,C}, v::VecUnroll{<:Any,W,T,<:VecUnroll{<:Any,W,T,Vec{W,T}}},
    u::UU, m::AbstractMask{W}, ::A, ::S, ::NT, ::StaticInt{RS}, ::Nothing
) where {W, T, A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS, D,C, UU <: NestedUnroll{W}}
  AUO,FO,NO,AV,_W,MO,X,U = unroll_params(UU)
  AUI,FI,NI,AV,_W,MI,X,I = unroll_params( U)
  vstore_double_unroll_quote(D, NO, NI, AUO, FO, AV, W, MO, X, C, AUI, FI, MI, true, A === True, S === True, NT === True, RS, typemax(Int))
end

function vstore_unroll_i_quote(Nm1, Wsplit, W, A, S, NT, rs::Int, mask::Bool)
    N = Nm1 + 1
    N*Wsplit == W || throw(ArgumentError("Vector of length $W can't be split into $N pieces of size $Wsplit."))
    q = Expr(:block, Expr(:meta, :inline), :(vt = data(v)), :(im = _materialize(i)))
    if mask
        let U = mask_type_symbol(Wsplit)
            push!(q.args, :(mt = data(vconvert(VecUnroll{$Nm1,$Wsplit,Bit,Mask{$Wsplit,$U}}, m))))
        end
    end
    j = 0
    alignval = Expr(:call, A ? :True : :False)
    aliasval = Expr(:call, S ? :True : :False)
    notmpval = Expr(:call, NT ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
    for n ∈ 1:N
        shufflemask = Expr(:tuple)
        for w ∈ 1:Wsplit
            push!(shufflemask.args, j)
            j += 1
        end
        ex = :(__vstore!(ptr, vt[$n], shufflevector(im, Val{$shufflemask}())))
        mask && push!(ex.args, Expr(:call, GlobalRef(Core, :getfield), :mt, n, false))
        push!(ex.args, alignval, aliasval, notmpval, rsexpr)
        push!(q.args, ex)
    end
    q
end
@generated function __vstore!(
    ptr::Ptr{T}, v::VecUnroll{Nm1,Wsplit}, i::VectorIndex{W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,Nm1,Wsplit,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS}
    vstore_unroll_i_quote(Nm1, Wsplit, W, A===True, S===True, NT===True, RS, false)
end
@generated function __vstore!(
    ptr::Ptr{T}, v::VecUnroll{Nm1,Wsplit}, i::VectorIndex{W}, m::AbstractMask{W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,Nm1,Wsplit,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool,RS}
    vstore_unroll_i_quote(Nm1, Wsplit, W, A===True, S===True, NT===True, RS, true)
end
function vstorebit_unroll_i_quote(Nm1::Int, Wsplit::Int, W::Int, A::Bool, S::Bool, NT::Bool, rs::Int, mask::Bool)
    N = Nm1 + 1
    N*Wsplit == W || throw(ArgumentError("Vector of length $W can't be split into $N pieces of size $Wsplit."))
    # W == 8 || throw(ArgumentError("There is only a need for splitting a mask of size 8, but the mask is of size $W."))
    # q = Expr(:block, Expr(:meta, :inline), :(vt = data(v)), :(im = _materialize(i)), :(u = 0x00))
    q = Expr(:block, Expr(:meta, :inline), :(vt = data(v)), :(u = 0x00))
    j = 0
    gf = GlobalRef(Core, :getfield)
    while true
        push!(q.args, :(u |= data($(Expr(:call, gf, :vt, (N-j), false)))))
        j += 1
        j == N && break
        push!(q.args, :(u <<= $Wsplit))
    end
    alignval = Expr(:call, A ? :True : :False)
    aliasval = Expr(:call, A ? :True : :False)
    notmpval = Expr(:call, A ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
    mask && push!(q.args, :(u = bitselect(data(m), __vload(Base.unsafe_convert(Ptr{$(mask_type_symbol(W))}, ptr), (data(i) >> 3), $alignval, $rsexpr), u)))
    call = Expr(:call, :__vstore!, :(reinterpret(Ptr{UInt8}, ptr)), :u, :(data(i) >> 3))
    push!(call.args, alignval, aliasval, notmpval, rsexpr)
    push!(q.args, call)
    q
end
@generated function __vstore!(
    ptr::Ptr{Bit}, v::VecUnroll{Nm1,Wsplit,Bit,M}, i::MM{W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {Nm1,Wsplit,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool, RS, M <: AbstractMask{Wsplit}}
  vstorebit_unroll_i_quote(Nm1, Wsplit, W, A===True, S===True, NT===True, RS, false)
end
@generated function __vstore!(
    ptr::Ptr{Bit}, v::VecUnroll{Nm1,Wsplit,Bit,M}, i::MM{W}, m::Mask{W}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {Nm1,Wsplit,W,S<:StaticBool,A<:StaticBool,NT<:StaticBool, RS, M <: AbstractMask{Wsplit}}
  vstorebit_unroll_i_quote(Nm1, Wsplit, W, A===True, S===True, NT===True, RS, true)
end

# If `::Function` vectorization is masked, then it must not be reduced by `::Function`.
@generated function _vstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,X,I}, m, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,AV,A<:StaticBool,S<:StaticBool,NT<:StaticBool,RS,X,G<:Function}
    N == U + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(U+1)."))
    # mask means it isn't vectorized
    AV > 0 || throw(ArgumentError("AV ≤ 0, but masking what, exactly?"))
    Expr(:block, Expr(:meta, :inline), :(_vstore!(ptr, vu, u, m, $(A()), $(S()), $(NT()), StaticInt{$RS}())))
end

function transposeshuffle(split, W, offset::Bool)
    tup = Expr(:tuple)
    w = 0
    S = 1 << split
    i = offset ? S : 0
    while w < W
        for s ∈ 0:S-1
            push!(tup.args, w + s + i)
        end
        for s ∈ 0:S-1
            # push!(tup.args, w + W + s)
            push!(tup.args, w + W + s + i)
        end
        w += 2S
    end
    Expr(:call, Expr(:curly, :Val, tup))
end

function horizontal_reduce_store_expr(W::Int, Ntotal::Int, (C,D,AU,F)::NTuple{4,Int}, op::Symbol, reduct::Symbol, noalias::Bool, RS::Int, mask::Bool)
    N = ((C == AU) && isone(F)) ? prevpow2(Ntotal) : 0
    q = Expr(:block, Expr(:meta, :inline), :(v = data(vu)))
    mask && push!(q.args, :(masktuple = data(m)))
    # mask && push!(q.args, :(unsignedmask = data(tomask(m))))
    # store = noalias ? :vnoaliasstore! : :vstore!
    falseexpr = Expr(:call, :False)
    aliasexpr = noalias ? Expr(:call, :True) : falseexpr
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, RS))
    ispow2(W) || throw(ArgumentError("Horizontal store requires power-of-2 vector widths."))
    gf = GlobalRef(Core, :getfield)
    if N > 1
        push!(q.args, :(gptr = gesp(ptr, $gf(u, :i))))
        push!(q.args, :(bptr = pointer(gptr)))
        extractblock = Expr(:block)
        vectors = [Symbol(:v_, n) for n ∈ 0:N-1]
        for n ∈ 1:N
            push!(extractblock.args, Expr(:(=), vectors[n], Expr(:call, gf, :v, n, false)))
        end
        push!(q.args, extractblock)
        ncomp = 0
        minWN = min(W,N)
        while ncomp < N
            Nt = minWN;
            Wt = W
            splits = 0
            while Nt > 1
                Nt >>>= 1
                shuffle0 = transposeshuffle(splits, Wt, false)
                shuffle1 = transposeshuffle(splits, Wt, true)
                splits += 1
                for nh ∈ 1:Nt
                    n1 = 2nh
                    n0 = n1 - 1
                    v0 = vectors[n0 + ncomp]; v1 = vectors[n1 + ncomp]; vh = vectors[nh + ncomp];
                    # combine n0 and n1
                    push!(q.args, Expr(
                        :(=), vh, Expr(
                            :call, op,
                            Expr(:call, :shufflevector, v0, v1, shuffle0),
                            Expr(:call, :shufflevector, v0, v1, shuffle1))
                    ))
                end
            end
            # v0 is now the only vector
            v0 = vectors[ncomp + 1]
            while Wt > minWN
                Wh = Wt >>> 1
                v0new = Symbol(v0, Wt)
                push!(q.args, Expr(
                    :(=), v0new, Expr(
                        :call, op,
                        Expr(:call, :shufflevector, v0, Expr(:call, Expr(:curly, :Val, Expr(:tuple, [w for w ∈ 0:Wh-1]...)))),
                        Expr(:call, :shufflevector, v0, Expr(:call, Expr(:curly, :Val, Expr(:tuple, [w for w ∈ Wh:Wt-1]...)))))
                )
                      )
                v0 = v0new
                Wt = Wh
            end
            if ncomp == 0
                storeexpr = Expr(:call, :__vstore!, :bptr, v0)
            else
                storeexpr = Expr(:call, :_vstore!, :gptr, v0)
                zeroexpr = Expr(:call, Expr(:curly, :StaticInt, 0))
                ind = Expr(:tuple); foreach(_ -> push!(ind.args, zeroexpr), 1:D)
                ind.args[AU] = Expr(:call, Expr(:curly, :StaticInt, F*ncomp))
                push!(storeexpr.args, ind)
            end
            if mask
                boolmask = Expr(:call, :Vec)
                for n ∈ ncomp+1:ncomp+minWN
                    push!(boolmask.args, Expr(:call, gf, :masktuple, n, false))
                end
                push!(storeexpr.args, Expr(:call, :tomask, boolmask))
            end
            # mask && push!(storeexpr.args, :(Mask{$minWN}(unsignedmask)))
            push!(storeexpr.args, falseexpr, aliasexpr, falseexpr, rsexpr)
            push!(q.args, storeexpr)
            # mask && push!(q.args, :(unsignedmask >>>= $minWN))
            ncomp += minWN
        end
    else
        push!(q.args, :(gptr = gesp(ptr, $gf(u, :i))))
    end
    if N < Ntotal
        zeroexpr = Expr(:call, Expr(:curly, :StaticInt, 0))
        ind = Expr(:tuple); foreach(_ -> push!(ind.args, zeroexpr), 1:D)
        for n ∈ N+1:Ntotal
            (n > N+1) && (ind = copy(ind)) # copy to avoid overwriting old
            ind.args[AU] = Expr(:call, Expr(:curly, :StaticInt, F*(n-1)))
            scalar = Expr(:call, reduct, Expr(:call, gf, :v, n, false))
            storeexpr = Expr(:call, :_vstore!, :gptr, scalar, ind, falseexpr, aliasexpr, falseexpr, rsexpr)
            if mask
                push!(q.args, Expr(:&&, Expr(:call, gf, :masktuple, n, false), storeexpr))
            else
                push!(q.args, storeexpr)
            end
        end
    end
    q
end
@inline function _vstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,X,I}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X}
    _vstore!(ptr, vu, u, A(), S(), NT(), StaticInt{RS}())
end
# function _vstore!(
@generated function _vstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,1,M,X,I}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X}
    N == U + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(U+1)."))
    if (G === typeof(identity)) || (AV > 0) || (W == 1)
        return Expr(:block, Expr(:meta, :inline), :(_vstore!(ptr, vu, u, $A(), $S(), $NT(), StaticInt{$RS}())))
    elseif G === typeof(vsum)
        op = :+; reduct = :vsum
    elseif G === typeof(vprod)
        op = :*; reduct = :vprod
    elseif G === typeof(vmaximum)
        op = :max; reduct = :vmaximum
    elseif G === typeof(vminimum)
        op = :min; reduct = :vminimum
    elseif G === typeof(vall)
        op = :&; reduct = :vall
    elseif G === typeof(vany)
        op = :|; reduct = :vany
    else
        throw("Function $G not recognized.")
    end
    horizontal_reduce_store_expr(W, N, (C,D,AU,F), op, reduct, S === True, RS, false)
end
@generated function _vstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,1,M,X,I}, m::VecUnroll{U,1,Bool,Bool}, ::A, ::S, ::NT, ::StaticInt{RS}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A<:StaticBool, S<:StaticBool, NT<:StaticBool, RS,X}
    N == U + 1 || throw(ArgumentError("The unrolled index specifies unrolling by $N, but sored `VecUnroll` is unrolled by $(U+1)."))
    1+2
    if (G === typeof(identity)) || (AV > 0) || (W == 1)
        return Expr(:block, Expr(:meta, :inline), :(_vstore!(ptr, vu, u, $A(), $S(), $NT(), StaticInt{$RS}())))
    elseif G === typeof(vsum)
        op = :+; reduct = :vsum
    elseif G === typeof(vprod)
        op = :*; reduct = :vprod
    elseif G === typeof(vmaximum)
        op = :max; reduct = :vmaximum
    elseif G === typeof(vminimum)
        op = :min; reduct = :vminimum
    elseif G === typeof(vall)
        op = :&; reduct = :vall
    elseif G === typeof(vany)
        op = :|; reduct = :vany
    else
        throw("Function $G not recognized.")
    end
    horizontal_reduce_store_expr(W, N, (C,D,AU,F), op, reduct, S === True, RS, true)
end




function lazymulunroll_load_quote(M,O,N,maskall,masklast,align,rs)
    t = Expr(:tuple)
    alignval = Expr(:call, align ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
    gf = GlobalRef(Core, :getfield)
    for n in 1:N+1
        ind = if (M != 1) | (O != 0)
            :(LazyMulAdd{$M,$O}(u[$n]))
        else
            Expr(:call, gf, :u, n, false)
        end
        call = if maskall
            Expr(:call, :__vload, :ptr, ind, Expr(:call, gf, :mt, n, false), alignval, rsexpr)
        elseif masklast && n == N+1
            Expr(:call, :__vload, :ptr, ind, :m, alignval, rsexpr)
        else
            Expr(:call, :__vload, :ptr, ind, alignval, rsexpr)
        end
        push!(t.args, call)
    end
    q = Expr(:block, Expr(:meta, :inline), :(u = data(um)))
    maskall && push!(q.args, :(mt = data(m)))
    push!(q.args, Expr(:call, :VecUnroll, t))
    q
end
@generated function __vload(ptr::Ptr{T}, um::VecUnroll{N,W,I,V}, ::A, ::StaticInt{RS}) where {T,N,W,I,V,A<:StaticBool,RS}
    lazymulunroll_load_quote(1,0,N,false,false,A === True,RS)
end
@generated function __vload(
    ptr::Ptr{T}, um::VecUnroll{N,W,I,V}, m::VecUnroll{N,W,Bit,M}, ::A, ::StaticInt{RS}
) where {T,N,W,I,V,A<:StaticBool,U,RS,M<:AbstractMask{W,U}}
    lazymulunroll_load_quote(1,0,N,true,false,A===True,RS)
end
@generated function __vload(
    ptr::Ptr{T}, um::VecUnroll{N,W1,I,V}, m::AbstractMask{W2,U}, ::A, ::StaticInt{RS}
) where {T,N,W1,W2,I,V,A<:StaticBool,U,RS}
    if W1 == W2
        lazymulunroll_load_quote(1,0,N,false,true,A===True,RS)
    elseif W2 == (N+1)*W1
        quote
            $(Expr(:meta,:inline))
            __vload(ptr, um, VecUnroll(splitvectortotuple(StaticInt{$(N+1)}(), StaticInt{$W1}(), m)), $A(), StaticInt{$RS}())
        end
    else
        throw(ArgumentError("Trying to load using $(N+1) indices of length $W1, while applying a mask of length $W2."))
    end
end
@generated function __vload(ptr::Ptr{T}, um::LazyMulAdd{M,O,VecUnroll{N,W,I,V}}, ::A, ::StaticInt{RS}) where {T,M,O,N,W,I,V,A<:StaticBool,RS}
    lazymulunroll_load_quote(M,O,N,false,false,A===True,RS)
end
@generated function __vload(
    ptr::Ptr{T}, um::LazyMulAdd{M,O,VecUnroll{N,W,I,V}}, m::VecUnroll{N,W,Bit,MSK}, ::A, ::StaticInt{RS}
) where {T,M,O,N,W,I,V,A<:StaticBool,U,RS,MSK<:AbstractMask{W,U}}
    lazymulunroll_load_quote(M,O,N,true,false,A===True,RS)
end
@generated function __vload(
    ptr::Ptr{T}, um::LazyMulAdd{M,O,VecUnroll{N,W1,I,V}}, m::AbstractMask{W2}, ::A, ::StaticInt{RS}
) where {T,M,O,N,W1,W2,I,V,A<:StaticBool,RS}
    if W1 == W2
        lazymulunroll_load_quote(M,O,N,false,true,A===True,RS)
    elseif W1 * (N+1) == W2
        quote
            $(Expr(:meta,:inline))
            __vload(ptr, um, VecUnroll(splitvectortotuple(StaticInt{$(N+1)}(), StaticInt{$W1}(), m)), $A(), StaticInt{$RS}())
        end
    else
        throw(ArgumentError("Trying to load using $(N+1) indices of length $W1, while applying a mask of length $W2."))
    end
end
function lazymulunroll_store_quote(M,O,N,mask,align,noalias,nontemporal,rs)
    gf = GlobalRef(Core, :getfield)
    q = Expr(:block, Expr(:meta, :inline), :(u = $gf($gf(um, :data), :data)), :(v = $gf($gf(vm, :data), :data)))
    alignval = Expr(:call, align ? :True : :False)
    noaliasval = Expr(:call, noalias ? :True : :False)
    nontemporalval = Expr(:call, nontemporal ? :True : :False)
    rsexpr = Expr(:call, Expr(:curly, :StaticInt, rs))
    for n in 1:N+1
        push!(q.args, Expr(:call, :vstore!, :ptr, Expr(:call, gf, :v, n, false), :(LazyMulAdd{$M,$O}(u[$n])), alignval, noaliasval, nontemporalval, rsexpr))
    end
    q
end


@generated function vload(r::FastRange{T}, i::Unroll{1,W,N,1,W,M,X,Tuple{I}}) where {T,I,W,N,M,X}
    q = quote
        $(Expr(:meta,:inline))
        s = vload(r, data(i))
        step = getfield(r, :s)
        mm = Vec(MM{$W,$X}(Zero())) * step
        v = Base.FastMath.add_fast(s + mm)
    end
    t = Expr(:tuple, :v)
    for n ∈ 1:N-1
        # push!(t.args, :(MM{$W,$W}(Base.FastMath.add_fast(s, $(T(n*W))))))
        push!(t.args, :(Base.FastMath.add_fast(v, Base.FastMath.mul_fast($(T(n*W)), step))))
    end
    push!(q.args, :(VecUnroll($t)))
    q
end
@generated function vload(r::FastRange{T}, i::Unroll{1,W,N,1,W,M,X,Tuple{I}}, m::AbstractMask{W}) where {T,I,W,N,M,X}
    q = quote
        $(Expr(:meta,:inline))
        s = vload(r, data(i))
        step = getfield(r, :s)
        mm = Vec(MM{$W,$X}(Zero())) * step
        v = Base.FastMath.add_fast(s + mm)
        z = zero(v)
    end
    t = if M % Bool
        Expr(:tuple, :(ifelse(m, v, z)))
    else
        Expr(:tuple, :v)
    end
    for n ∈ 1:N-1
        M >>>= 1
        if M % Bool
            push!(t.args, :(ifelse(m, Base.FastMath.add_fast(v, Base.FastMath.mul_fast($(T(n*W)), step)), z)))
        else
            push!(t.args, :(Base.FastMath.add_fast(v, Base.FastMath.mul_fast($(T(n*W)), step))))
        end
    end
    push!(q.args, :(VecUnroll($t)))
    q
end
@generated function vload(r::FastRange{T}, i::Unroll{1,W,N,1,W,M,X,Tuple{I}}, m::VecUnroll{Nm1,W,B}) where {T,I,W,N,M,X,Nm1,B<:Union{Bit,Bool}}
    q = quote
        $(Expr(:meta,:inline))
        s = vload(r, data(i))
        step = getfield(r, :s)
        mm = Vec(MM{$W,$X}(Zero())) * step
        v = Base.FastMath.add_fast(s + mm)
        z = zero(v)
    end
    t = Expr(:tuple, :(ifelse(getfield(m,$1,false), v, z)))
    for n ∈ 1:N-1
        push!(t.args, :(ifelse(getfield(m,$(n+1),false), Base.FastMath.add_fast(v, Base.FastMath.mul_fast($(T(n*W)), step)), z)))
    end
    push!(q.args, :(VecUnroll($t)))
    q
end
