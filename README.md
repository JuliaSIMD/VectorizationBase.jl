# VectorizationBase

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/VectorizationBase.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/VectorizationBase.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/VectorizationBase.jl.svg?branch=master)](https://travis-ci.com/chriselrod/VectorizationBase.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/VectorizationBase.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/VectorizationBase-jl)
[![Codecov](https://codecov.io/gh/chriselrod/VectorizationBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/VectorizationBase.jl)

---

This library provides some basic functionality meant to aid in in vectorizing code on x86 architectures (most laptops, desktops, and servers). Building it depends on [CpuID.jl](https://github.com/m-j-w/CpuId.jl), which creates a file defining a fewconstants important to vectorization, such as the SIMD widths. For example:

```julia
julia> using VectorizationBase

julia> VectorizationBase.pick_vector_width(Float64)
8

julia> VectorizationBase.pick_vector_width(Float32)
16

julia> VectorizationBase.pick_vector(Float64)
NTuple{8,VecElement{Float64}}
```
This means that on the computer that ran this code, a single vector register can hold 8 elements of type `Float64` or 16 `Float32`.

For more information on `VecElement`s, please see [Julia's documentation](https://docs.julialang.org/en/v1/base/simd-types/) or Kristoffer Carlson's blog post [SIMD and SIMD-intrinsics in Julia](http://kristofferc.github.io/post/intrinsics/). In short, they are lowered as [LLVM vectors](https://llvm.org/docs/LangRef.html#vector-type), which (while housing multiple elements) are treated as single values, to be operated on in parallel with SIMD instructions. This is contrast with an array, which is an aggregate collection of values.

The libraries [SIMDPirates](https://github.com/chriselrod/SIMDPirates.jl) and [SLEEFPirates](https://github.com/chriselrod/SLEEFPirates.jl) (which are themselves forks of [SIMD.jl](https://github.com/eschnett/SIMD.jl) and [SLEEF.jl](https://github.com/musm/SLEEF.jl)) define many functions operating on tuples of `VecElement`s, for writing explicit SIMD code.

Perhaps used more often than `VectorizationBase.pick_vector_width` is `VectorizationBase.pick_vector_width_shift`:
```julia
julia> W, Wshift = VectorizationBase.pick_vector_width_shift(Float64)
(8, 3)
```
This function returns the ideal vector width for the type, as well as `log2(width)`. This is useful for calculating the number of loop iterations. For example, if we wish to iterate over a vector of length 117:
```julia
julia> 117 >> Wshift
14

julia> 117 & (W - 1)
5

julia> 14W + 5
117
```
We would want 14 iterations of 8, leaving us a remainder of 5. We could either loop over this remainder one at a time, or use masked instructions. `VectorizationBase.mask` provides a convenience function for producing a mask:
```julia
julia> VectorizationBase.mask(Float64, 5)
0x1f

julia> bitstring(ans)
"00011111"
```
Masks are to be read from right to left. For example, we could define a masked load as follows:
```julia
@inline function vload8d(ptr::Ptr{Float64}, mask::UInt8)
	Base.llvmcall(
		("declare <8 x double> @llvm.masked.load.v8f64(<8 x double>*, i32, <8 x i1>, <8 x double>)",
		"""%ptr = inttoptr i64 %0 to <8 x double>*
		%mask = bitcast i8 %1 to <8 x i1>
		%res = call <8 x double> @llvm.masked.load.v8f64(<8 x double>* %ptr, i32 8, <8 x i1> %mask, <8 x double> zeroinitializer)
		ret <8 x double> %res"""),
		Vec{8, Float64}, Tuple{Ptr{Float64}, UInt8}, ptr, mask
	)
end
```
This allows us to load just 5 elements (and therefore avoid segmentation faults):
```julia
julia> x = rand(5); x'
1×5 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.928247  0.889502  0.533114  0.285248  0.275795

julia> y = rand(8); y'
1×8 LinearAlgebra.Adjoint{Float64,Array{Float64,1}}:
 0.402572  0.771600  0.242454  0.699283  0.618579  0.804612  0.904894  0.234704
 
julia> vload8d(pointer(x), 0x1f)
(VecElement{Float64}(0.9282465954844357), VecElement{Float64}(0.8895020822839887), VecElement{Float64}(0.5331136178366147), VecElement{Float64}(0.28524793374254176), VecElement{Float64}(0.2757945162086832), VecElement{Float64}(0.0), VecElement{Float64}(0.0), VecElement{Float64}(0.0))

julia> vload8d(pointer(y), 0x1f)
(VecElement{Float64}(0.4025719724148169), VecElement{Float64}(0.7715998492280507), VecElement{Float64}(0.242453946944301), VecElement{Float64}(0.6992828239389028), VecElement{Float64}(0.6185788376359711), VecElement{Float64}(0.0), VecElement{Float64}(0.0), VecElement{Float64}(0.0))

julia> vload8d(pointer(y) + 3sizeof(Float64), 0x1f)
(VecElement{Float64}(0.6992828239389028), VecElement{Float64}(0.6185788376359711), VecElement{Float64}(0.8046118255195078), VecElement{Float64}(0.904893953223624), VecElement{Float64}(0.23470368695369492), VecElement{Float64}(0.0), VecElement{Float64}(0.0), VecElement{Float64}(0.0))
```

This compiles to efficient native code on the host machine:
```asm
julia> @code_native debuginfo=:none vload8d(pointer(x), 0x1f)
	.text
	kmovd	%esi, %k1
	vmovupd	(%rdi), %zmm0 {%k1} {z}
	retq
	nopl	(%rax,%rax)
```
You don't need to know LLVM to use this library, `SIMDPirates`, or `SLEEFPirates`; they should abstract everything away. If you have any feature requests, or would like any helper functions such as those mentioned here, please don't hesitate to file an issue or open a pull request.
