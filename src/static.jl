
struct Static{N} end
Base.@pure Static(N) = Static{N}()

struct StaticUnitRange{L,U} <: AbstractRange{Int} end
Base.@pure StaticUnitRange(L,U) = StaticUnitRange{L,U}()
struct StaticLowerUnitRange{L} <: AbstractRange{Int}
    U::Int
end
struct StaticUpperUnitRange{U} <: AbstractRange{Int}
    L::Int
end


@inline Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline Base.:(:)(::Static{L}, U::Int) where {L} = StaticLowerUnitRange{L}(U)
@inline Base.:(:)(L::Int, ::Static{U}) where {U} = StaticUpperUnitRange{U}(L)

@inline StaticLowerUnitRange{L}(::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline StaticUpperUnitRange{U}(::Static{L}) where {L,U} = StaticUnitRange{L,U}()

@inline maybestaticsize(A, ::Val{I}) where {I} = size(A, I)
@inline maybestaticlength(A) = length(A)

@inline maybestaticsize(::NTuple{N}, ::Val{1}) where {N} = Static{N}() # should we assert that i == 1?
@inline maybestaticlength(::NTuple{N}) where {N} = Static{N}()


@inline Base.:+(::Static{N}, i) where {N} = N + i
@inline Base.:+(i, ::Static{N}) where {N} = N + i
@inline Base.:+(::Static{M}, ::Static{N}) where {M,N} = M + N
@inline Base.:*(::Static{N}, i) where {N} = N * i
@inline Base.:*(i, ::Static{N}) where {N} = N * i
@inline Base.:*(::Static{M}, ::Static{N}) where {M,N} = M * N
@inline Base.:-(::Static{N}, i) where {N} = N - i
@inline Base.:-(i, ::Static{N}) where {N} = i - N
@inline Base.:-(::Static{M}, ::Static{N}) where {M,N} = M - N
@inline Base.:>>(::Static{N}, i) where {N} = N >> i
@inline Base.:>>(i, ::Static{N}) where {N} = i >> N
@inline Base.:>>(::Static{M}, ::Static{N}) where {M,N} = M >> N
@inline Base.:<<(::Static{N}, i) where {N} = N << i
@inline Base.:<<(i, ::Static{N}) where {N} = i << N
@inline Base.:<<(::Static{M}, ::Static{N}) where {M,N} = M << N
@inline Base.:>>>(::Static{N}, i) where {N} = N >>> i
@inline Base.:>>>(i, ::Static{N}) where {N} = i >>> N
@inline Base.:>>>(::Static{M}, ::Static{N}) where {M,N} = M >>> N
@inline Base.:&(::Static{N}, i) where {N} = N & i
@inline Base.:&(i, ::Static{N}) where {N} = N & i
@inline Base.:&(::Static{M}, ::Static{N}) where {M,N} = M & N
@inline Base.:>(::Static{N}, i) where {N} = N > i
@inline Base.:>(i, ::Static{N}) where {N} = i > N
@inline Base.:>(::Static{M}, ::Static{N}) where {M,N} = M > N
@inline Base.:<(::Static{N}, i) where {N} = N < i
@inline Base.:<(i, ::Static{N}) where {N} = i < N
@inline Base.:<(::Static{M}, ::Static{N}) where {M,N} = M < N


@generated staticm1(::Static{N}) where {N} = Static{N-1}()
@inline staticm1(N::Integer) = N - 1
