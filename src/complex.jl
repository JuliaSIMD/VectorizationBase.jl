
abstract type AbstractComplexVec{W,T} <: AbstractSIMDVector{W,T}
const AbstractComplex{W,T} = Union{AbstractComplexVec{W,T},Complex{<:AbstractSIMD{W,T}}}
struct ComplexVec{W,T} <: AbstractComplexVec{W,T}
  data::NTuple{W,Core.VecElement{T}}
  @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:NativeTypes} = new{W,T}(x)
  @generated function Vec(x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}) where {_W,T<:NativeTypes}
    W = _W + 1
    vtyp = Expr(:curly, :Vec, W, T)
    Expr(:block, Expr(:meta,:inline), Expr(:(::), Expr(:call, vtyp, :x), vtyp))
  end
end
struct AdjointComplexVec{W,T} <: AbstractComplexVec{W,T}
  data::NTuple{W,Core.VecElement{T}}
  @inline Vec{W,T}(x::NTuple{W,Core.VecElement{T}}) where {W,T<:NativeTypes} = new{W,T}(x)
  @generated function Vec(x::Tuple{Core.VecElement{T},Vararg{Core.VecElement{T},_W}}) where {_W,T<:NativeTypes}
    W = _W + 1
    vtyp = Expr(:curly, :Vec, W, T)
    Expr(:block, Expr(:meta,:inline), Expr(:(::), Expr(:call, vtyp, :x), vtyp))
  end
end
@inline Base.adjoint(v::ComplexVec) = AdjointComplexVec(getfield(v,:data))
@inline Base.adjoint(v::AdjointComplexVec) = ComplexVec(getfield(v,:data))
@inline data(v::ComplexVec) = getfield(v,:data)
@inline function data(v::AdjointComplexVec{W}) where {W}
  vv = Vec(getfield(v, :data))
  data(ifelse(isodd(MM{W}(Zero())), vv, -vv))
end

@inline asvec(x::AbstractComplexVec) = Vec(data(x))
@inline ascomplex(x::Vec) = ComplexVec(data(x))
@inline vadd_fast(x, y)
@inline function vmul_fast(x::ComplexVec, y::ComplexVec)
  vx = asvec(x); vy = asvec(y);
  xu = uppervector(vx)
  xl = lowervector(vx)
  yp = vpermilps177(vy)
  ascomplex(vfmaddsub(xu, vy, Base.FastMath.mul_fast(xl, yp)))
  # xre * yre - xim * yim
  # xre * yim + xim * yre

end
@inline vmul(x::ComplexVec, y::ComplexVec) = vmul_fast(x, y)

@inline vfma(x::AbstractComplex, y::AbstractComplex, z::AbstractComplex) = vmuladd(x, y, z)
@inline vfmadd(x::AbstractComplex, y::AbstractComplex, z::AbstractComplex) = vmuladd(x, y, z)
@inline vfmadd_fast(x::AbstractComplex, y::AbstractComplex, z::AbstractComplex) = vmuladd(x, y, z)
@inline function vmuladd(x::ComplexVec, y::ComplexVec, z::ComplexVec)
  vx = asvec(x); vy = asvec(y); vz = asvec(z);
  # xre * yre - xim * yim + zre
  # xre * yim + xim * yre + zim
  xu = uppervector(vx)
  xl = lowervector(vx)
  yp = vpermilps177(vy)
  ascomplex(vfmaddsub(xu, vy, vfmaddsub(xl, yp, vz)))
end
@inline function vmuladd(x::Complex, y::ComplexVec, z::ComplexVec)
  xre = real(x)
  xim = imag(x)
  vy = asvec(y); vz = asvec(z);
  # xre * yre - xim * yim + zre
  # xre * yim + xim * yre + zim
  yp = vpermilps177(vy)
  ascomplex(vfmaddsub(xre, vy, vfmaddsub(xim, yp, vz)))
end
@inline function vmuladd(x::ComplexVec, y::Complex, z::ComplexVec)
  vx = asvec(x); vz = asvec(z);
  yre = real(y)
  yim = imag(y)
  # xre * yre - xim * yim + zre
  # xim * yre + xre * yim + zim
  xp = vpermilps177(vx)
  ascomplex(vfmaddsub(vx, vre, vfmaddsub(xp, yim, vz)))
end

@inline vfnmadd_fast(x::ComplexVec, y::ComplexVec, z::ComplexVec) = vfnmadd(x, y, z)
@inline function vfnmadd(x::ComplexVec, y::ComplexVec, z::ComplexVec)
  vx = asvec(x); vy = asvec(y); vz = asvec(z);
  # - xre * yre + xim * yim + zre
  # - xre * yim - xim * yre + zim
  xu = uppervector(vx)
  xl = lowervector(vx)
  yp = vpermilps177(vy)
  ascomplex(vfmaddsub(xu, vy, vfmaddsub(xl, yp, z)))
end

