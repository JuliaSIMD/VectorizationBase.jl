# VectorizationBase

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSIMD.github.io/VectorizationBase.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaSIMD.github.io/VectorizationBase.jl/dev)
[![CI](https://github.com/JuliaSIMD/VectorizationBase.jl/workflows/CI/badge.svg)](https://github.com/JuliaSIMD/VectorizationBase.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/JuliaSIMD/VectorizationBase.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/JuliaSIMD/VectorizationBase.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)
[![Codecov](https://codecov.io/gh/JuliaSIMD/VectorizationBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaSIMD/VectorizationBase.jl)

---

This is a library providing basic SIMD support in Julia. VectorizationBase exists in large part to serve the needs of [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)'s code gen, prioritizing this over a stable user-facing API. Thus, you may wish to consider [SIMD.jl](https://github.com/eschnett/SIMD.jl) as an alternative when writing explicit SIMD code in Julia. That said, the `Vec` and `VecUnroll` types are meant to "just work" as much as possible when passed to user-defined functions, so it should be reasonably stable in practice. Other parts of the code -- e.g, loading and storing vectors as well as the `stridedpointer` function -- will hopefully converge reasonably soon, and have support for various `AbstractArray` types propogated through the ecosystem by taking advantage of [ArrayInterface.jl](https://github.com/SciML/ArrayInterface.jl), so that VectorizationBase can begin to offer a stable, ergonomic, and well supported API fairly soon.

It additionally provides some information on the host computer it is running on, which can be used to automate target-specific optimizations. Currently, x86_64 support is best on that front, but I'm looking to improve the quality of information provided for other architectures.

`Vec`s are `Number`s and behave as a single objects; they just happen to contain multiple `Float64`. Therefore, it will behave like a single number rather than a collection with respect to indexing and reductions:
```julia
julia> using VectorizationBase

julia> vx = Vec(ntuple(_ -> 10randn(), VectorizationBase.pick_vector_width(Float64))...)
Vec{8,Float64}<14.424983437388981, -7.7378330531368045, -3.499708331670689, -3.358981392002452, 22.519898671389406, -13.08647686033593, 13.96943264299162, -9.518537139443254>

julia> vx[1]
Vec{8,Float64}<14.424983437388981, -7.7378330531368045, -3.499708331670689, -3.358981392002452, 22.519898671389406, -13.08647686033593, 13.96943264299162, -9.518537139443254>

julia> sum(vx)
Vec{8,Float64}<14.424983437388981, -7.7378330531368045, -3.499708331670689, -3.358981392002452, 22.519898671389406, -13.08647686033593, 13.96943264299162, -9.518537139443254>

julia> a = 1.2;

julia> a[1]
1.2

julia> sum(a)
1.2
```

To extract elements from a `Vec`, you call it, using parenthesis to index as you would in Fortran or MATLAB:
```julia
julia> vx(1), vx(2)
(14.424983437388981, -7.7378330531368045)

julia> ntuple(vx, Val(8))
(14.424983437388981, -7.7378330531368045, -3.499708331670689, -3.358981392002452, 22.519898671389406, -13.08647686033593, 13.96943264299162, -9.518537139443254)

julia> Tuple(vx) # defined for convenience
(14.424983437388981, -7.7378330531368045, -3.499708331670689, -3.358981392002452, 22.519898671389406, -13.08647686033593, 13.96943264299162, -9.518537139443254)
```
Unfortunately, this means no support for indexing with `begin`/`end`.


Reductions are like the ordinary version, but prefixed with `v`:
```julia
julia> using VectorizationBase: vsum, vprod, vmaximum, vminimum

julia> vsum(vx), sum(Tuple(vx))
(13.712777975180877, 13.712777975180877)

julia> vprod(vx), prod(Tuple(vx))
(-5.141765647043406e7, -5.141765647043406e7)

julia> vmaximum(vx), maximum(Tuple(vx))
(22.519898671389406, 22.519898671389406)

julia> vminimum(vx), minimum(Tuple(vx))
(-13.08647686033593, -13.08647686033593)
```




