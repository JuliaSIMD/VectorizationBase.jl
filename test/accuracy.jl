# The SLEEF.jl package is licensed under the MIT "Expat" License:

# > Copyright (c) 2016: Mustafa Mohamad and other contributors:
# > 
# > https://github.com/musm/SLEEF.jl/graphs/contributors
# > 
# > Permission is hereby granted, free of charge, to any person obtaining a copy
# > of this software and associated documentation files (the "Software"), to deal
# > in the Software without restriction, including without limitation the rights
# > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# > copies of the Software, and to permit persons to whom the Software is
# > furnished to do so, subject to the following conditions:
# > 
# > The above copyright notice and this permission notice shall be included in all
# > copies or substantial portions of the Software.
# > 
# > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# > SOFTWARE.
# > 

# SLEEF.jl includes ported code from the following project

#     - [SLEEF](https://github.com/shibatch/SLEEF) [public domain] Author Naoki Shibata

using Base.Math: significand_bits

isnzero(x::T) where {T<:AbstractFloat} = signbit(x)
ispzero(x::T) where {T<:AbstractFloat} = !signbit(x)

# function cmpdenorm(x::Tx, y::Ty) where {Tx <: AbstractFloat, Ty <: AbstractFloat}
#     sizeof(Tx) < sizeof(Ty) ? y = Tx(y) : x = Ty(x) # cast larger type to smaller type
#     (isnan(x) && isnan(y)) && return true
#     (isnan(x) || isnan(y)) && return false
#     (isinf(x) != isinf(y)) && return false
#     (x == Tx(Inf)  && y == Ty(Inf))  && return true
#     (x == Tx(-Inf) && y == Ty(-Inf)) && return true
#     if y == 0
#         (ispzero(x) && ispzero(y)) && return true
#         (isnzero(x) && isnzero(y)) && return true
#         return false
#     end
#     (!isnan(x) && !isnan(y) && !isinf(x) && !isinf(y)) && return sign(x) == sign(y)
#     return false
# end

# the following compares the ulp between x and y.
# First it promotes them to the larger of the two types x,y
infh(::Type{Float64}) = 1e300
infh(::Type{Float32}) = 1e37
function countulp(::Type{T}, __x, __y) where {T}
  _x, _y = promote(__x, __y)
  x, y = convert(T, _x), convert(T, _y) # Cast to smaller type
  iszero(y) && return iszero(x) ? zero(x) : T(1004)
  ulpc = convert(T, abs(_x - _y) / ulp(y))
  nanulp = VectorizationBase.ifelse(isnan(x) ⊻ isnan(y), T(10000), T(0))
  infulp = VectorizationBase.ifelse(
    (sign(x) == sign(y)) & (abs(y) > infh(T)),
    T(0),
    T(10001)
  )

  ulpc = VectorizationBase.ifelse(
    isinf(x),
    infulp,
    VectorizationBase.ifelse(isfinite(y), ulpc, T(10003))
  )
  ulpc = VectorizationBase.ifelse(isnan(x) | isnan(y), nanulp, ulpc)
  ulpc = VectorizationBase.ifelse(
    iszero(y),
    VectorizationBase.ifelse(iszero(x), T(0), T(10002)),
    ulpc
  )
  return ulpc
end

DENORMAL_MIN(::Type{Float64}) = 2.0^-1074
DENORMAL_MIN(::Type{Float32}) = 2.0f0^-149

function ulp(
  x::Union{<:VectorizationBase.AbstractSIMD{<:Any,T},T}
) where {T<:AbstractFloat}
  e = exponent(x)
  # ulpc = max(VectorizationBase.vscalef(T(1.0), e - significand_bits(T)), DENORMAL_MIN(T))
  ulpc = max(ldexp(T(1.0), e - significand_bits(T)), DENORMAL_MIN(T))
  ulpc = VectorizationBase.ifelse(x == T(0.0), DENORMAL_MIN(T), ulpc)
  return ulpc
end

countulp(x::T, y::T) where {T<:AbstractFloat} = countulp(T, x, y)
countulp(
  x::VectorizationBase.AbstractSIMD{W,T},
  y::VectorizationBase.AbstractSIMD{W,T}
) where {W,T<:AbstractFloat} = countulp(T, x, y)

# test the accuracy of a function where fun_table is a Dict mapping the function you want
# to test to a reference function
# xx is an array of values (which may be tuples for multiple arugment functions)
# tol is the acceptable tolerance to test against
function test_acc(
  f1,
  f2,
  T,
  xx,
  tol,
  ::StaticInt{W} = pick_vector_width(T);
  debug = false,
  tol_debug = 5
) where {W}
  @testset "accuracy $(f1)" begin
    reference = map(f2 ∘ big, xx)
    comp = similar(xx)
    i = 0
    spc = VectorizationBase.zstridedpointer(comp)
    spx = VectorizationBase.zstridedpointer(xx)
    GC.@preserve xx comp begin
      while i < length(xx)
        vstore!(spc, f1(vload(spx, (MM{W}(i),))), (MM{W}(i),))
        i += W
      end
    end
    rmax = 0.0
    rmean = 0.0
    xmax = map(zero, first(xx))
    for i ∈ eachindex(xx)
      q = comp[i]
      c = reference[i]
      u = countulp(T, q, c)
      rmax = max(rmax, u)
      xmax = rmax == u ? xx[i] : xmax
      rmean += u
      if xx[i] == 36.390244f0
        @show f1, q, f2, T(c), xx[i], T(c)
      end
      if debug && u > tol_debug
        @show f1, q, f2, T(c), xx[i], T(c)
      end
    end
    rmean = rmean / length(xx)

    fmtxloc = isa(xmax, Tuple) ? join(xmax, ", ") : string(xmax)
    println(
      rpad(f1, 18, " "),
      ": max ",
      rmax,
      rpad(" at x = " * fmtxloc, 40, " "),
      ": mean ",
      rmean
    )

    t = @test trunc(rmax; digits = 1) <= tol
  end
end
