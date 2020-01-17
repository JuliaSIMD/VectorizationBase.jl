
struct Static{N} end
Base.@pure Static(N) = Static{N}()

struct StaticUnitRange{L,U} end
Base.@pure StaticUnitRange(L,U) = StaticUnitRange{L,U}()
struct StaticLowerUnitRange{L}
    U::Int
end
struct StaticUpperUnitRange{U}
    L::Int
end


@inline Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline Base.:(:)(::Static{L}, U::Int) where {L} = StaticLowerUnitRange{L}(U)
@inline Base.:(:)(L::Int, ::Static{U}) where {U} = StaticUpperUnitRange{U}(L)

@inline maybestaticsize(A, args...) = size(A, args...)
@inline maybestaticlength(A) = length(A)

@inline maybestaticsize(::NTuple{N}, i) where {N} = Static{N}() # should we assert that i == 1?
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

