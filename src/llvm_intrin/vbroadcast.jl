
@inline vzero(::Val{1}, ::Type{T}) where {T<:NativeTypes} = zero(T)
@inline vzero(::StaticInt{1}, ::Type{T}) where {T<:NativeTypes} = zero(T)
@inline _vzero(::StaticInt{W}, ::Type{Float16}, ::StaticInt{RS}) where {W,RS} =
  _vzero_float16(StaticInt{W}(), StaticInt{RS}(), fast_half())
@inline _vzero_float16(::StaticInt{W}, ::StaticInt{RS}, ::False) where {W,RS} =
  _vzero(StaticInt{W}(), Float32, StaticInt{RS}())
function _vzero_expr(W::Int, typ::String, T::Symbol, st::Int, RS::Int)
  isone(W) && return Expr(:block, Expr(:meta, :inline), Expr(:call, :zero, T))
  # if W * st > RS
  #   d, r1 = divrem(st * W, RS)
  #   Wnew, r2 = divrem(W, d)
  #   (iszero(r1) & iszero(r2)) || throw(ArgumentError("If broadcasting to greater than 1 vector length, should make it an integer multiple of the number of vectors."))
  #   t = Expr(:tuple)
  #   for i ∈ 1:d
  #       push!(t.args, :v)
  #   end
  #   # return Expr(:block, Expr(:meta,:inline), :(v = vzero(StaticInt{$Wnew}(), $T)), :(VecUnroll{$(d-1),$Wnew,$T,Vec{$Wnew,$T}}($t)))
  #   return Expr(:block, Expr(:meta,:inline), :(v = _vzero(StaticInt{$Wnew}(), $T, StaticInt{$RS}())), :(VecUnroll($t)))
  #   # return Expr(:block, Expr(:meta,:inline), :(v = _vzero(StaticInt{$Wnew}(), $T, StaticInt{$RS}())), :(VecUnroll($t)::VecUnroll{$(d-1),$Wnew,$T,Vec{$Wnew,$T}}))
  # end
  instrs = "ret <$W x $typ> zeroinitializer"
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{}))
  end
end
@generated _vzero_float16(::StaticInt{W}, ::StaticInt{RS}) where {W,RS} =
  _vzero_expr(W, "half", :Float16, 2, RS)
@generated _vzero(
  ::StaticInt{W},
  ::Type{T},
  ::StaticInt{RS}
) where {W,T<:NativeTypesExceptFloat16,RS} =
  _vzero_expr(W, LLVM_TYPES[T], JULIA_TYPES[T], sizeof(T), RS)
function vundef_expr(W::Int, typ::String, T::Symbol)
  if T === :Bit
    W == 1 ? false : Mask(zero_mask(Val(W)))
  elseif W == 1
    instrs = "ret $typ undef"
    quote
      $(Expr(:meta, :inline))
      $LLVMCALL($instrs, $T, Tuple{})
    end
  else
    instrs = "ret <$W x $typ> undef"
    quote
      $(Expr(:meta, :inline))
      Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{}))
    end
  end
end
@generated function _vundef(
  ::StaticInt{W},
  ::Type{T}
) where {W,T<:NativeTypesExceptFloat16}
  vundef_expr(W, LLVM_TYPES[T], JULIA_TYPES[T])
end
@generated function _vundef(::StaticInt{W}, ::Type{Float16}) where {W}
  _vundef_float16(StaticInt{W}(), fast_half())
end
@generated _vundef_float16(::StaticInt{W}, ::True) where {W} =
  vundef_expr(W, "half", :Float16)
@generated _vundef_float16(::StaticInt{W}, ::False) where {W} =
  vundef_expr(W, "float", :Float32)
@inline _vundef(::T) where {T<:NativeTypes} = _vundef(StaticInt{1}(), T)
@inline _vundef(::Vec{W,T}) where {W,T} = _vundef(StaticInt{W}(), T)
@generated _vundef(::VecUnroll{N,W,T}) where {N,W,T} = Expr(
  :block,
  Expr(:meta, :inline),
  :(VecUnroll(
    Base.Cartesian.@ntuple $(N + 1) n -> _vundef(StaticInt{$W}(), $T)
  ))
)
function vbroadcast_expr(W::Int, typ::String, T::Symbol, st::Int, RS::Int)
  isone(W) && return :s
  # if st * W > RS
  #   d, r1 = divrem(st * W, RS)
  #   Wnew, r2 = divrem(W, d)
  #   (iszero(r1) & iszero(r2)) || throw(ArgumentError("If broadcasting to greater than 1 vector length, should make it an integer multiple of the number of vectors."))
  #   t = Expr(:tuple)
  #   for i ∈ 1:d
  #     push!(t.args, :v)
  #   end
  #   return Expr(:block, Expr(:meta,:inline), :(v = _vbroadcast(StaticInt{$Wnew}(), s, StaticInt{$RS}())), :(VecUnroll($t)))
  # end
  vtyp = vtype(W, typ)
  instrs = """
    %ie = insertelement $vtyp undef, $typ %0, i32 0
    %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
    ret $vtyp %v
  """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{$T}, s))
  end
end
@inline _vbroadcast(::StaticInt{W}, s::Float16, ::StaticInt{RS}) where {W,RS} =
  _vbroadcast_float16(StaticInt{W}(), s, StaticInt{RS}(), fast_half())
@inline _vbroadcast_float16(
  ::StaticInt{W},
  s::Float16,
  ::StaticInt{RS},
  ::False
) where {W,RS} =
  _vbroadcast(StaticInt{W}(), convert(Float32, s), StaticInt{RS}())
@generated _vbroadcast_float16(
  ::StaticInt{W},
  s::Float16,
  ::StaticInt{RS},
  ::True
) where {W,RS} = vbroadcast_expr(W, "half", :Float16, 2, RS)
@inline function _vbroadcast(
  ::StaticInt{W},
  s::Bool,
  ::StaticInt{RS}
) where {W,RS}
  t = Mask(max_mask(StaticInt{W}()))
  f = Mask(zero_mask(StaticInt{W}()))
  Core.ifelse(s, t, f)
end
@generated function _vbroadcast(
  ::StaticInt{W},
  s::_T,
  ::StaticInt{RS}
) where {W,_T<:NativeTypesExceptFloat16,RS}
  if (_T <: Integer) && (sizeof(_T) * W > RS) && sizeof(_T) ≥ 8
    intbytes = max(4, RS ÷ W)
    T = integer_of_bytes(intbytes)
    if _T <: Unsigned
      T = unsigned(T)
    end
    # ssym = :(s % $T)
    if T ≢ _T
      return Expr(
        :block,
        Expr(:meta, :inline),
        :(_vbroadcast(StaticInt{$W}(), convert($T, s), StaticInt{$RS}()))
      )
    end
  end
  vbroadcast_expr(W, LLVM_TYPES[_T], JULIA_TYPES[_T], sizeof(_T), RS)
end
@inline _vbroadcast(
  ::StaticInt{W},
  m::EVLMask{W},
  ::StaticInt{RS}
) where {W,RS} = Mask(m)
@inline vzero(::Union{Val{W},StaticInt{W}}, ::Type{T}) where {W,T} =
  _vzero(StaticInt{W}(), T, register_size(T))
@inline vbroadcast(::Union{Val{W},StaticInt{W}}, s::T) where {W,T} =
  _vbroadcast(StaticInt{W}(), s, register_size(T))
@inline function _vbroadcast(
  ::StaticInt{W},
  vu::VecUnroll{N,1,T,T},
  ::StaticInt{RS}
) where {W,N,T,RS}
  VecUnroll(fmap(_vbroadcast, StaticInt{W}(), data(vu), StaticInt{RS}()))
end

@generated function vbroadcast(
  ::Union{Val{W},StaticInt{W}},
  ptr::Ptr{T}
) where {W,T}
  isone(W) && return Expr(:block, Expr(:meta, :inline), :(vload(ptr)))
  typ = LLVM_TYPES[T]
  ptyp = JULIAPOINTERTYPE
  vtyp = "<$W x $typ>"
  alignment = Base.datatype_alignment(T)
  instrs = @static if USE_OPAQUE_PTR
    "%res = load $typ, ptr %0, align $alignment"
  else
    """
    %ptr = inttoptr $ptyp %0 to $typ*
    %res = load $typ, $typ* %ptr, align $alignment
    """
  end
  instrs *= """
      %ie = insertelement $vtyp undef, $typ %res, i32 0
      %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
      ret $vtyp %v
  """
  quote
    $(Expr(:meta, :inline))
    Vec($LLVMCALL($instrs, _Vec{$W,$T}, Tuple{Ptr{$T}}, ptr))
  end
end

@inline vbroadcast(
  ::Union{Val{W},StaticInt{W}},
  v::AbstractSIMDVector{W}
) where {W} = v

@generated function vbroadcast(
  ::Union{Val{W},StaticInt{W}},
  v::V
) where {W,L,T,V<:AbstractSIMDVector{L,T}}
  N, r = divrem(L, W)
  @assert iszero(r)
  V = if T === Bit
    :(Mask{$W,$(mask_type_symbol(W))})
  else
    :(Vec{$W,$T})
  end
  Expr(
    :block,
    Expr(:meta, :inline),
    :(vconvert(VecUnroll{$(N - 1),$W,$T,$V}, v))
  )
end

@inline Vec{W,T}(v::Vec{W,T}) where {W,T} = v

@inline Base.zero(::Type{Vec{W,T}}) where {W,T} =
  _vzero(StaticInt{W}(), T, StaticInt{W}() * static_sizeof(T))
@inline Base.zero(::Vec{W,T}) where {W,T} = zero(Vec{W,T})
@inline Base.one(::Vec{W,T}) where {W,T} = vbroadcast(Val{W}(), one(T))

@inline Base.one(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Val{W}(), one(T))
@inline Base.oneunit(::Type{Vec{W,T}}) where {W,T} =
  vbroadcast(Val{W}(), one(T))
@inline vzero(::Type{T}) where {T<:Number} = zero(T)
@inline vzero() = vzero(pick_vector_width(Float64), Float64)

@inline Vec{W,T}(s::Real) where {W,T} = vbroadcast(Val{W}(), T(s))
@inline Vec{W}(s::T) where {W,T<:NativeTypes} = vbroadcast(Val{W}(), s)
@inline Vec(s::T) where {T<:NativeTypes} = vbroadcast(pick_vector_width(T), s)

@generated function _vzero(
  ::Type{VecUnroll{N,W,T,V}},
  ::StaticInt{RS}
) where {N,W,T,V,RS}
  t = Expr(:tuple)
  z = W == 1 ? :(zero($T)) : :(_vzero(StaticInt{$W}(), $T, StaticInt{$RS}()))
  for _ ∈ 0:N
    push!(t.args, z)
  end
  Expr(:block, Expr(:meta, :inline), :(VecUnroll($t)))
end
@inline Base.zero(::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V} =
  _vzero(VecUnroll{N,W,T,V}, register_size())
@inline Base.zero(::VecUnroll{N,W,T,V}) where {N,W,T,V} =
  zero(VecUnroll{N,W,T,V})

@inline Base.one(::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V} =
  VecUnroll{N}(one(V))

@generated function VecUnroll{N,W,T,V}(
  x::S
) where {N,W,T,V<:AbstractSIMDVector{W,T},S<:Real}
  t = Expr(:tuple)
  for n ∈ 0:N
    push!(t.args, :(convert($V, x)))
  end
  Expr(:block, Expr(:meta, :inline), :(VecUnroll($t)))
end
@generated function VecUnroll{N,1,T,T}(x::S) where {N,T<:NativeTypes,S<:Real}
  t = Expr(:tuple)
  for n ∈ 0:N
    push!(t.args, :(convert($T, x)))
  end
  Expr(:block, Expr(:meta, :inline), :(VecUnroll($t)))
end
@inline VecUnroll{N,W,T}(x::NativeTypesV) where {N,W,T} =
  VecUnroll{N,W,T,Vec{W,T}}(x)
@inline VecUnroll{N}(x::V) where {N,W,T,V<:AbstractSIMDVector{W,T}} =
  VecUnroll{N,W,T,V}(x)
@inline VecUnroll{N}(x::T) where {N,T<:NativeTypes} = VecUnroll{N,1,T,T}(x)

@generated function zero_vecunroll(
  ::StaticInt{N},
  ::StaticInt{W},
  ::Type{T},
  ::StaticInt{RS}
) where {N,W,T,RS}
  Expr(
    :block,
    Expr(:meta, :inline),
    :(_vzero(VecUnroll{$(N - 1),$W,$T,Vec{$W,$T}}, StaticInt{$RS}()))
  )
end
@inline zero_init(
  ::Type{T},
  ::StaticInt{1},
  ::StaticInt{0},
  ::StaticInt{RS}
) where {T,RS} = zero(T)
@inline zero_init(
  ::Type{T},
  ::StaticInt{W},
  ::StaticInt{0},
  ::StaticInt{RS}
) where {W,T,RS} = _vzero(StaticInt{W}(), T, StaticInt{RS}())
@inline zero_init(
  ::Type{T},
  ::StaticInt{W},
  ::StaticInt{U},
  ::StaticInt{RS}
) where {W,U,T,RS} = _vzero(VecUnroll{U,W,T,Vec{W,T}}, StaticInt{RS}())
@inline zero_init(
  ::Type{T},
  ::Tuple{StaticInt{W},StaticInt{U}},
  ::StaticInt{RS}
) where {W,U,T,RS} = zero_init(T, StaticInt{W}(), StaticInt{U}(), StaticInt{RS})
@generated function vbroadcast_vecunroll(
  ::StaticInt{N},
  ::StaticInt{W},
  s::T,
  ::StaticInt{RS}
) where {N,W,T,RS}
  q = Expr(
    :block,
    Expr(:meta, :inline),
    :(v = _vbroadcast(StaticInt{$W}(), s, StaticInt{$RS}()))
  )
  t = Expr(:tuple)
  for n ∈ 1:N
    push!(t.args, :v)
  end
  push!(q.args, :(VecUnroll($t)))
  q
end
