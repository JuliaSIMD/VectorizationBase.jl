
function shufflevector_instrs(W::Int, @nospecialize(T), I::Vector{String}, W2::Int)
  W2 > W && throw(
    ArgumentError(
      "W for vector 1 must be at least W for vector two, but W₁ = $W < W₂ = $W2.",
    ),
  )
  typ::String = (LLVM_TYPES[T])::String
  vtyp1::String = "<$W x $typ>"
  M::Int = length(I)
  vtyp3::String = "<$M x i32>"
  vtypr::String = "<$M x $typ>"
  mask::String = '<' * join(I, ", ")::String * '>'
  if ((W2 == 0) | (W2 == W))
    v2 = W2 == 0 ? "undef" : "%1"
    M,
    """
     %res = shufflevector $vtyp1 %0, $vtyp1 $v2, $vtyp3 $mask
     ret $vtypr %res
 """
  else
    vtyp0 = "<$W2 x $typ>"
    maskpad =
      '<' *
      join(map(w -> string("i32 ", w > W2 ? "undef" : string(w - 1)), 1:W), ", ") *
      '>'
    M,
    """
     %pad = shufflevector $vtyp0 %1, $vtyp0 undef, <$W x i32> $maskpad
     %res = shufflevector $vtyp1 %0, $vtyp1 %pad, $vtyp3 $mask
     ret $vtypr %res    
 """
  end
end
function tupletostringvector(@nospecialize(x::NTuple{N,Int})) where {N}
  y = Vector{String}(undef, N)
  @inbounds for n ∈ 1:N
    y[n] = string("i32 ", x[n])
  end
  y
end
@generated function shufflevector(v1::Vec{W,T}, v2::Vec{W2,T}, ::Val{I}) where {W,W2,T,I}
  W ≥ W2 || throw(
    ArgumentError(
      "`v1` should be at least as long as `v2`, but `v1` is a `Vec{$W,$T}` and `v2` is a `Vec{$W2,$T}`.",
    ),
  )
  M, instrs = shufflevector_instrs(W, T, tupletostringvector(I), W2)
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T},_Vec{$W2,$T}}, data(v1), data(v2)),
    )
  end
end
@inline shufflevector(x::T, y::T, ::Val{(0,1)}) where {T<:NativeTypes} = Vec(x, y)
@inline shufflevector(x::T, y::T, ::Val{(1,0)}) where {T<:NativeTypes} = Vec(y, x)
@generated function shufflevector(v1::Vec{W,T}, ::Val{I}) where {W,T,I}
  if length(I) == 1
    return Expr(:block, Expr(:meta,:inline), :(extractelement(v1, $(only(I)))))
  end
  M, instrs = shufflevector_instrs(W, T, tupletostringvector(I), 0)
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}}, data(v1)))
  end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::Vec{L,T}) where {W,L,T}
  typ = LLVM_TYPES[T]
  mask =
    '<' * join(map(x -> string("i32 ", x ≥ L ? "undef" : string(x)), 0:W-1), ", ") * '>'
  instrs = """
      %res = shufflevector <$L x $typ> %0, <$L x $typ> undef, <$W x i32> $mask
      ret <$W x $typ> %res
  """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{_Vec{$L,$T}}, data(v)))
  end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::T) where {W,T<:NativeTypes}
  typ = LLVM_TYPES[T]
  vtyp = vtype(W, typ)
  instrs = """
      %ie = insertelement $vtyp undef, $typ %0, i32 0
      ret $vtyp %ie
  """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{$T}, v))
  end
end
@generated function shufflevector(i::MM{W,X}, ::Val{I}) where {W,X,I}
  allincr = true
  L = length(I)
  for l ∈ 2:L
    allincr &= (I[l] == I[l-1] + 1)
  end
  allincr || return Expr(:block, Expr(:meta, :inline), :(shufflevector(Vec(i), Val{$I}())))
  Expr(:block, Expr(:meta, :inline), :(MM{$L,$X}(extractelement(i, $(first(I))))))
end
@generated function Base.vcat(a::Vec{W1,T}, b::Vec{W2,T}) where {W1,W2,T}
  W1 ≥ W2 || throw(
    ArgumentError(
      "`v1` should be at least as long as `v2`, but `v1` is a `Vec{$W1,$T}` and `v2` is a `Vec{$W2,$T}`.",
    ),
  )
  mask = Vector{String}(undef, 2W1)
  for w ∈ 0:W1+W2-1
    mask[w+1] = string("i32 ", w)
  end
  for w ∈ W1+W2:2W1-1
    mask[w+1] = "i32 undef"
  end
  M, instrs = shufflevector_instrs(W1, T, mask, W2)
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$M,$T}, Tuple{_Vec{$W1,$T},_Vec{$W2,$T}}, data(a), data(b)))
  end
end

@inline Base.vcat(
  a::VecUnroll{N,W1,T,Vec{W1,T}},
  b::VecUnroll{N,W2,T,Vec{W2,T}},
) where {N,W1,W2,T} = VecUnroll(fmap(vcat, data(a), data(b)))
@generated function Base.hcat(
  a::VecUnroll{N1,W,T,V},
  b::VecUnroll{N2,W,T,V},
) where {N1,N2,W,T,V}
  q = Expr(:block, Expr(:meta, :inline), :(da = data(a)), :(db = data(b)))
  t = Expr(:tuple)
  for (d, N) ∈ ((:da, N1), (:db, N2))
    for n ∈ 1:N
      push!(t.args, Expr(:call, :getfield, d, n, false))
    end
  end
  push!(q.args, :(VecUnroll($t)))
  q
end


function transpose_vecunroll_quote(W)
  ispow2(W) || throw(
    ArgumentError(
      "Only supports powers of 2 for vector width and unrolling factor, but recieved $W = $W.",
    ),
  )
  log2W = intlog2(W)
  q = Expr(:block, Expr(:meta, :inline), :(vud = data(vu)))
  N = W # N vectors of length W
  vectors1 = [Symbol(:v_, n) for n ∈ 0:N-1]
  vectors2 = [Symbol(:v_, n + N) for n ∈ 0:N-1]
  # z = Expr(:call, Expr(:curly, Expr(:(.), :VectorizationBase, QuoteNode(:MM)), W), 0)
  # for n ∈ 1:N
  #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vload)), :ptrA, Expr(:tuple, z, n-1))))
  # end
  for n ∈ 1:N
    push!(q.args, Expr(:(=), vectors1[n], Expr(:call, :getfield, :vud, n, false)))
  end
  Nhalf = N >>> 1
  vecstride = 1
  partition_stride = 2
  for nsplits = 0:log2W-1
    shuffle0 = transposeshuffle(nsplits, W, false)
    shuffle1 = transposeshuffle(nsplits, W, true)
    for partition ∈ 0:(W>>>(nsplits+1))-1
      for _n1 ∈ 1:vecstride
        n1 = partition * partition_stride + _n1
        n2 = n1 + vecstride
        v11 = vectors1[n1]
        v12 = vectors1[n2]
        v21 = vectors2[n1]
        v22 = vectors2[n2]
        shuff1 = Expr(:call, :shufflevector, v11, v12, shuffle0)
        shuff2 = Expr(:call, :shufflevector, v11, v12, shuffle1)
        push!(q.args, Expr(:(=), v21, shuff1))
        push!(q.args, Expr(:(=), v22, shuff2))
      end
    end
    vectors1, vectors2 = vectors2, vectors1
    vecstride <<= 1
    partition_stride <<= 1
    # @show vecstride <<= 1
  end
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, vectors1[n])
  end
  # for n ∈ 1:N
  #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vstore!)), :ptrB, vectors1[n], Expr(:tuple, z, n-1))))
  # end
  push!(q.args, Expr(:call, :VecUnroll, t))
  q
end
function subset_tup(W, o)
  t = Expr(:tuple)
  for w ∈ o:W-1+o
    push!(t.args, w)
  end
  Expr(:call, Expr(:curly, :Val, t))
end
function transpose_vecunroll_quote_W_larger(N, W)
  (ispow2(W) & ispow2(N)) || throw(
    ArgumentError(
      "Only supports powers of 2 for vector width and unrolling factor, but recieved $N and $W.",
    ),
  )
  log2W = intlog2(W)
  log2N = intlog2(N)
  q = Expr(:block, Expr(:meta, :inline), :(vud = data(vu)))
  # N = W # N vectors of length W
  vectors1 = [Symbol(:v_, n) for n ∈ 0:N-1]
  vectors2 = [Symbol(:v_, n + N) for n ∈ 0:N-1]
  # z = Expr(:call, Expr(:curly, Expr(:(.), :VectorizationBase, QuoteNode(:MM)), W), 0)
  # for n ∈ 1:N
  #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vload)), :ptrA, Expr(:tuple, z, n-1))))
  # end
  for n ∈ 1:N
    push!(q.args, Expr(:(=), vectors1[n], Expr(:call, :getfield, :vud, n, false)))
  end
  Nhalf = N >>> 1
  vecstride = 1
  partition_stride = 2
  for nsplits = 0:log2N-1
    shuffle0 = transposeshuffle(nsplits, W, false)
    shuffle1 = transposeshuffle(nsplits, W, true)
    for partition ∈ 0:(N>>>(nsplits+1))-1
      for _n1 ∈ 1:vecstride
        n1 = partition * partition_stride + _n1
        n2 = n1 + vecstride
        v11 = vectors1[n1]
        v12 = vectors1[n2]
        v21 = vectors2[n1]
        v22 = vectors2[n2]
        shuff1 = Expr(:call, :shufflevector, v11, v12, shuffle0)
        shuff2 = Expr(:call, :shufflevector, v11, v12, shuffle1)
        push!(q.args, Expr(:(=), v21, shuff1))
        push!(q.args, Expr(:(=), v22, shuff2))
      end
    end
    vectors1, vectors2 = vectors2, vectors1
    vecstride <<= 1
    partition_stride <<= 1
    # @show vecstride <<= 1
  end
  # @show vecstride, partition_stride
  t = Expr(:tuple)
  o = 0
  for i ∈ 1:(1<<(log2W-log2N))
    extract = subset_tup(N, o)
    for n ∈ 1:N
      push!(t.args, Expr(:call, :shufflevector, vectors1[n], extract))
    end
    o += N
  end
  # for n ∈ 1:N
  #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vstore!)), :ptrB, vectors1[n], Expr(:tuple, z, n-1))))
  # end
  push!(q.args, Expr(:call, :VecUnroll, t))
  q
end
function transpose_vecunroll_quote_W_smaller(N, W)
  (ispow2(W) & ispow2(N)) || throw(
    ArgumentError(
      "Only supports powers of 2 for vector width and unrolling factor, but recieved $N and $W.",
    ),
  )
  N, W = W, N
  log2W = intlog2(W)
  log2N = intlog2(N)
  q = Expr(:block, Expr(:meta, :inline), :(vud = data(vu)))
  # N = W # N vectors of length W
  vectors1 = [Symbol(:v_, n) for n ∈ 0:N-1]
  vectors2 = [Symbol(:v_, n + N) for n ∈ 0:N-1]
  # z = Expr(:call, Expr(:curly, Expr(:(.), :VectorizationBase, QuoteNode(:MM)), W), 0)
  # for n ∈ 1:N
  #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vload)), :ptrA, Expr(:tuple, z, n-1))))
  # end
  vectors3 = [Symbol(:vpiece_, w) for w ∈ 0:W-1]
  for w ∈ 1:W
    push!(q.args, Expr(:(=), vectors3[w], Expr(:call, :getfield, :vud, w, false)))
  end
  Wtemp = W
  exprs = Vector{Expr}(undef, W >>> 1)
  initstride = W >>> (log2W - log2N)

  Ntemp = N
  # Wtemp = W >>> 1
  Wratio_init = W ÷ N
  Wratio = Wratio_init
  1, 3
  2, 4
  vcat(vcat(1, 3), vcat(5, 7))
  vcat(vcat(2, 4), vcat(6, 8))
  while Wratio > 1
    Wratioh = Wratio >>> 1
    for w ∈ 0:(Wratioh)-1
      i = (2N) * w
      j = i + N
      for n ∈ 1:N
        exprs[n+N*w] = if Wratio == Wratio_init
          Expr(:call, :vcat, vectors3[i+n], vectors3[j+n])
        else
          Expr(:call, :vcat, exprs[i+n], exprs[j+n])
        end
      end
    end
    Wratio = Wratioh
  end
  for n ∈ 1:N
    push!(q.args, Expr(:(=), vectors1[n], exprs[n]))
  end
  Nhalf = N >>> 1
  vecstride = 1
  partition_stride = 2
  for nsplits = 0:log2N-1
    shuffle0 = transposeshuffle(nsplits, W, false)
    shuffle1 = transposeshuffle(nsplits, W, true)
    for partition ∈ 0:(N>>>(nsplits+1))-1
      for _n1 ∈ 1:vecstride
        n1 = partition * partition_stride + _n1
        n2 = n1 + vecstride
        v11 = vectors1[n1]
        v12 = vectors1[n2]
        v21 = vectors2[n1]
        v22 = vectors2[n2]
        shuff1 = Expr(:call, :shufflevector, v11, v12, shuffle0)
        shuff2 = Expr(:call, :shufflevector, v11, v12, shuffle1)
        push!(q.args, Expr(:(=), v21, shuff1))
        push!(q.args, Expr(:(=), v22, shuff2))
      end
    end
    vectors1, vectors2 = vectors2, vectors1
    vecstride <<= 1
    partition_stride <<= 1
    # @show vecstride <<= 1
  end
  # @show vecstride, partition_stride
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, vectors1[n])
  end
  push!(q.args, Expr(:call, :VecUnroll, t))
  q
end
@generated function transpose_vecunroll(vu::VecUnroll{N,W}) where {N,W}
  # N+1 == W || throw(ArgumentError("Transposing is currently only supported for sets of vectors of size equal to their length, but received $(N+1) vectors of length $W."))
  # 1+2
  if N + 1 == W
    W == 1 && return :vu
    transpose_vecunroll_quote(W)
  elseif W == 1
    v = Expr(:call, :Vec)
    for n ∈ 0:N
      push!(v.args, Expr(:call, GlobalRef(Core, :getfield), :vud, n + 1, false))
    end
    Expr(:block, Expr(:meta, :inline), :(vud = data(vu)), v)
  elseif N + 1 < W
    transpose_vecunroll_quote_W_larger(N + 1, W)
  else# N+1 > W
    transpose_vecunroll_quote_W_smaller(N + 1, W)
  end
  # code below lets LLVM do it.
  # q = Expr(:block, Expr(:meta,:inline), :(vud = data(vu)))
  # S = W
  # syms = Vector{Symbol}(undef, W)
  # gf = GlobalRef(Core, :getfield)
  # for w ∈ 1:W
  #     syms[w] = v = Symbol(:v_, w)
  #     push!(q.args, Expr(:(=), v, Expr(:call, gf, :vud, w, false)))
  # end
  # while S > 1
  #     S >>>= 1
  #     for s ∈ 1:S
  #         v1 = syms[2s-1]
  #         v2 = syms[2s  ]
  #         vc = Symbol(v1,:_,v2)
  #         push!(q.args, Expr(:(=), vc, Expr(:call, :vcat, v1, v2)))
  #         syms[s] = vc
  #     end        
  # end
  # t = Expr(:tuple)
  # v1 = syms[1];# v2 = syms[2]
  # for w1 ∈ 0:N
  #     shufftup = Expr(:tuple)
  #     for w2 ∈ 0:N
  #         push!(shufftup.args, w2*W + w1)
  #     end
  #     push!(t.args, Expr(:call, :shufflevector, v1, Expr(:call, Expr(:curly, :Val, shufftup))))
  #     # push!(t.args, Expr(:call, :shufflevector, v1, v2, Expr(:call, Expr(:curly, :Val, shufftup))))
  # end
  # push!(q.args, Expr(:call, :VecUnroll, t))
  # q
end
@generated function vec_to_vecunroll(v::AbstractSIMDVector{W}) where {W}
  t = Expr(:tuple)
  for w ∈ 0:W-1
    push!(t.args, :(extractelement(v, $w)))
  end
  Expr(:block, Expr(:meta, :inline), :(VecUnroll($t)))
end

@inline shufflevector(vxu::VecUnroll, ::Val{I}) where {I} =
  VecUnroll(fmap(shufflevector, data(vxu), Val{I}()))

shuffleexpr(s::Expr) = Expr(:block, Expr(:meta, :inline), :(shufflevector(vx, Val{$s}())))
"""
  vpermilps177(vx::AbstractSIMD)

  Vec(0, 1, 2, 3, 4, 5, 6, 7) ->
    Vec(1, 0, 3, 2, 5, 4, 7, 6)
"""
@generated function vpermilps177(vx::AbstractSIMD{W}) where {W}
  s = Expr(:tuple)
  for w ∈ 1:2:W
    push!(s.args, w, w - 1)
  end
  shuffleexpr(s)
end
"""
  vmovsldup(vx::AbstractSIMD)

  Vec(0, 1, 2, 3, 4, 5, 6, 7) ->
    Vec(0, 0, 2, 2, 4, 4, 6, 6),
"""
@generated function vmovsldup(vx::AbstractSIMD{W}) where {W}
  sl = Expr(:tuple)
  for w ∈ 1:2:W
    push!(sl.args, w - 1, w - 1)
  end
  shuffleexpr(sl)
end
"""
  vmovshdup(vx::AbstractSIMD)

  Vec(0, 1, 2, 3, 4, 5, 6, 7) ->
    Vec(1, 1, 3, 3, 5, 5, 7, 7)
"""
@generated function vmovshdup(vx::AbstractSIMD{W}) where {W}
  sh = Expr(:tuple)
  for w ∈ 1:2:W
    push!(sh.args, w, w)
  end
  shuffleexpr(sh)
end

@generated function uppervector(vx::AbstractSIMD{W}) where {W}
  s = Expr(:tuple)
  for i ∈ W>>>1:W-1
    push!(s.args, i)
  end
  shuffleexpr(s)
end
@generated function lowervector(vx::AbstractSIMD{W}) where {W}
  s = Expr(:tuple)
  for i ∈ 0:(W>>>1)-1
    push!(s.args, i)
  end
  shuffleexpr(s)
end
@inline splitvector(vx::AbstractSIMD) = lowervector(vx), uppervector(vx)

@generated function extractupper(vx::AbstractSIMD{W}) where {W}
  s = Expr(:tuple)
  for i ∈ 0:(W>>>1)-1
    push!(s.args, 2i)
  end
  shuffleexpr(s)
end
@generated function extractlower(vx::AbstractSIMD{W}) where {W}
  s = Expr(:tuple)
  for i ∈ 0:(W>>>1)-1
    push!(s.args, 2i + 1)
  end
  shuffleexpr(s)
end
