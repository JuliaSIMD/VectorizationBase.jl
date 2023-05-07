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


Here is an example of using `vload`:
```julia
julia> A = rand(8,8);

julia> vload(stridedpointer(A), (MM(W, 1), 1))
Vec{8, Float64}<0.23659378106523243, 0.1572296679962767, 0.4139998988982545, 0.4068544124895789, 0.6365683129363592, 0.10041731176364777, 0.6198701180649783, 0.18351031426464992>

julia> A[1:W,1]'
1×8 adjoint(::Vector{Float64}) with eltype Float64:
 0.236594  0.15723  0.414  0.406854  0.636568  0.100417  0.61987  0.18351

julia> vload(stridedpointer(A), (1, MM(W, 1)))
Vec{8, Float64}<0.23659378106523243, 0.43800087768259754, 0.5833216557209256, 0.8076063696863035, 0.12069215155721758, 0.6015627184700922, 0.1390837892914757, 0.9139206013822945>

julia> A[1,1:W]'
1×8 adjoint(::Vector{Float64}) with eltype Float64:
 0.236594  0.438001  0.583322  0.807606  0.120692  0.601563  0.139084  0.913921

julia> vload(stridedpointer(A), (MM(W,1), MM(W, 1)))
Vec{8, Float64}<0.23659378106523243, 0.7580627352162604, 0.044776171518136954, 0.218587536875811, 0.4596625543892163, 0.2933303822991349, 0.30481677678671315, 0.3595115888246907>

julia> getindex.(Ref(A), 1:W, 1:W)'
1×8 adjoint(::Vector{Float64}) with eltype Float64:
 0.236594  0.758063  0.0447762  0.218588  0.459663  0.29333  0.304817  0.359512
 ```
 The basic idea is that you have a tuple of indices. The `MM` type indicates that it is vectorized. In the above example, we vectorize the load along colums, then rows, and then both. This is equivalent to loading the column, row, and diagonal.
 Note that you can pass a `Mask` argument to mask off extra loads/stores.
