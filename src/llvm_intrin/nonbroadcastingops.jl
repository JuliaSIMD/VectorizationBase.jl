
@generated function addscalar(v::Vec{W,T}, s::T) where {W,T<:IntegerTypesHW}
  typ = "i$(8sizeof(T))"
  vtyp = "<$W x $typ>"
  instrs = String[]
  push!(instrs, "%ie = insertelement $vtyp zeroinitializer, $typ %1, i32 0")
  push!(instrs, "%v = add $vtyp %0, %ie")
  push!(instrs, "ret $vtyp %v")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end
@generated function addscalar(v::Vec{W,T}, s::T) where {W,T<:FloatingTypes}
  typ = LLVM_TYPES[T]
  vtyp = "<$W x $typ>"
  instrs = String[]
  push!(instrs, "%ie = insertelement $vtyp zeroinitializer, $typ %1, i32 0")
  push!(instrs, "%v = fadd nsz arcp contract afn reassoc $vtyp %0, %ie")
  push!(instrs, "ret $vtyp %v")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end

@generated function mulscalar(v::Vec{W,T}, s::T) where {W,T<:IntegerTypesHW}
  typ = "i$(8sizeof(T))"
  vtyp = "<$W x $typ>"
  instrs = String[]
  push!(
    instrs,
    "%ie = insertelement $vtyp $(llvmconst(W, T, 1)), $typ %1, i32 0"
  )
  push!(instrs, "%v = mul $vtyp %0, %ie")
  push!(instrs, "ret $vtyp %v")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end
@generated function mulscalar(v::Vec{W,T}, s::T) where {W,T<:FloatingTypes}
  typ = LLVM_TYPES[T]
  vtyp = "<$W x $typ>"
  instrs = String[]
  push!(
    instrs,
    "%ie = insertelement $vtyp $(llvmconst(W, T, 1.0)), $typ %1, i32 0"
  )
  push!(instrs, "%v = fmul nsz arcp contract afn reassoc $vtyp %0, %ie")
  push!(instrs, "ret $vtyp %v")
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end

function scalar_maxmin(W::Int, @nospecialize(_::Type{T}), ismax::Bool) where {T}
  if T <: Integer
    typ = "i$(8sizeof(T))"
    comp =
      (T <: Signed) ? (ismax ? "icmp sgt" : "icmp slt") :
      (ismax ? "icmp ugt" : "icmp ult")
    basevalue = llvmconst(W, T, ismax ? typemin(T) : typemax(T))
  else
    opzero = ismax ? -Inf : Inf
    comp = ismax ? "fcmp ogt" : "fcmp olt"
    basevalue = llvmconst(W, T, repr(reinterpret(UInt64, opzero)))
    if T === Float64
      typ = "double"
      # basevalue = llvmconst(W, T, repr(reinterpret(UInt64, opzero)))
    elseif T === Float32
      typ = "float"
      # basevalue = llvmconst(W, T, repr(reinterpret(UInt32, Float32(opzero))))
      # elseif T === Float16
      # typ = "half"
      # basevalue = llvmconst(W, T, repr(reinterpret(UInt16, Float16(opzero))))
    else
      throw("T === $T not currently supported.")
    end
  end
  _scalar_maxmin(W, typ, comp, basevalue)
end
function _scalar_maxmin(W::Int, typ::String, comp::String, basevalue::String)
  vtyp = "<$W x $typ>"
  String[
    "%ie = insertelement $vtyp $(basevalue), $typ %1, i32 0",
    "%selection = $comp $vtyp %0, %ie",
    "%v = select <$W x i1> %selection, $vtyp %0, $vtyp %ie",
    "ret $vtyp %v"
  ]
end
@generated function maxscalar(v::Vec{W,T}, s::T) where {W,T<:NativeTypes}
  instrs = scalar_maxmin(W, T, true)
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end
@generated function minscalar(v::Vec{W,T}, s::T) where {W,T<:NativeTypes}
  instrs = scalar_maxmin(W, T, false)
  quote
    $(Expr(:meta, :inline))
    Vec(
      $LLVMCALL(
        $(join(instrs, "\n")),
        NTuple{$W,Core.VecElement{$T}},
        Tuple{NTuple{$W,Core.VecElement{$T}},$T},
        data(v),
        s
      )
    )
  end
end
for (f, op) ∈ [
  (:addscalar, :(+)),
  (:mulscalar, :(*)),
  (:maxscalar, :max),
  (:minscalar, :min)
]
  @eval begin
    @inline $f(v::VecUnroll, s) = VecUnroll((
      $f(first(getfield(v, :data)), s),
      Base.tail(getfield(v, :data))...
    ))
    @inline $f(v::Vec{W,T}, s::NativeTypes) where {W,T<:NativeTypes} =
      $f(v, vconvert(T, s))
    @inline $f(s::NativeTypes, v::AbstractSIMD{W,T}) where {W,T<:NativeTypes} =
      $f(v, s)
    @inline $f(a, b) = $op(a, b)
  end
end
