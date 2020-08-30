
# abstract type AbstractLazyMul end
struct LazyMul{N,T} <: Number
    data::T
end
@inline LazyMul{N}(data::T) where {N,T} = LazyMul{N,T}(data)
@inline data(lm::LazyMul) = lm.data

# struct LazyMulAdd{N,M,T} <: AbstractLazyMul
    # data::T
# end
# @inline LazyMulAdd{N,M}(data::T) where {N,M,T} = LazyMulAdd{N,M,T}(data)


@inline lazymul(a, b) = vmul(a, b)
@inline lazymul(::Static{N}, b) where {N} = LazyMul{N}(b)
@inline lazymul(a, ::Static{N}) where {N} = LazyMul{N}(a)
@inline lazymul(::Static{N}, b::MM) where {N} = LazyMul{N}(Vec(b))
@inline lazymul(a::MM, ::Static{N}) where {N} = LazyMul{N}(Vec(a))
@inline lazymul(::Static{M}, ::Static{N}) where {M, N} = Static{M*N}()

@inline lazymul(a::LazyMul{M}, ::Static{N}) where {M,N} = LazyMul{M*N}(data(a))
@inline lazymul(::Static{M}, b::LazyMul{N}) where {M,N} = LazyMul{M*N}(data(b))
@inline lazymul(a::LazyMul{M,<:MM}, ::Static{N}) where {M,N} = LazyMul{M*N}(Vec(data(a)))
@inline lazymul(::Static{M}, b::LazyMul{N,<:MM}) where {M,N} = LazyMul{M*N}(Vec(data(b)))
@inline lazymul(a::LazyMul{M}, b::LazyMul{N}) where {M,N} = LazyMul{M*N}(vmul(data(a), data(b)))

@inline lazymul_no_prmote(a, b) = vmul_no_promote(a, b)
@inline lazymul_no_prmote(::Static{N}, b) where {N} = LazyMul{N}(b)
@inline lazymul_no_prmote(a, ::Static{N}) where {N} = LazyMul{N}(a)
@inline lazymul_no_prmote(::Static{M}, ::Static{N}) where {M, N} = Static{M*N}()

@inline lazymul_no_prmote(a::LazyMul{M}, ::Static{N}) where {M,N} = LazyMul{M*N}(data(a))
@inline lazymul_no_prmote(::Static{M}, b::LazyMul{N}) where {M,N} = LazyMul{M*N}(data(b))
@inline lazymul_no_prmote(a::LazyMul{M}, b::LazyMul{N}) where {M,N} = LazyMul{M*N}(vmul(data(a), data(b)))


