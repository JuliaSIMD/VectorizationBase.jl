# VectorizationBase

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/VectorizationBase.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/VectorizationBase.jl/latest)
![CI](https://github.com/chriselrod/VectorizationBase.jl/workflows/CI/badge.svg)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/VectorizationBase.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/VectorizationBase-jl)
[![Codecov](https://codecov.io/gh/chriselrod/VectorizationBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/VectorizationBase.jl)

---

This is a library providing basic SIMD support in Julia. VectorizationBase exists in large part to serve the needs of [LoopVectorization.jl](https://github.com/chriselrod/LoopVectorization.jl)'s code gen, prioritizing this over a stable user-facing API. Thus, you may wish to consider [SIMD.jl](https://github.com/eschnett/SIMD.jl) as an alternative when writing explicit SIMD code in Julia. That said, the `Vec` and `VecUnroll` types are meant to "just work" as much as possible when passed to user-defined functions, so it should be reasonably stable in practice. Other parts of the code -- e.g, loading and storing vectors as well as the `stridedpointer` function -- will hopefully converge reasonably soon, and have support for various `AbstractArray` types propogated through the ecosystem by taking advantage of [ArrayInterface.jl](https://github.com/SciML/ArrayInterface.jl), so that VectorizationBase can begin to offer a stable, ergonomic, and well supported API fairly soon.

It additionally provides some information on the host computer it is running on, which can be used to automate target-specific optimizations. Currently, x86_64 support is best on that front, but I'm looking to improve the quality of information provided for other architectures.
 



