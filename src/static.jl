
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

@inline Base.first(::StaticUnitRange{L}) where {L} = L
@inline Base.first(::StaticLowerUnitRange{L}) where {L} = L
@inline Base.first(r::StaticUpperUnitRange) = r.L
@inline Base.last(::StaticUnitRange{L,U}) where {L,U} = U
@inline Base.last(r::StaticLowerUnitRange) = r.U
@inline Base.last(::StaticUpperUnitRange{U}) where {U} = U
@inline Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline Base.:(:)(::Static{L}, U::Int) where {L} = StaticLowerUnitRange{L}(U)
@inline Base.:(:)(L::Int, ::Static{U}) where {U} = StaticUpperUnitRange{U}(L)
@inline unwrap(::Static{N}) where {N} = N
@inline unwrap(::Type{Static{N}}) where {N} = N

@inline StaticLowerUnitRange{L}(::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline StaticUpperUnitRange{U}(::Static{L}) where {L,U} = StaticUnitRange{L,U}()

@inline maybestaticsize(A, ::Val{I}) where {I} = size(A, I)
@inline maybestaticlength(A) = length(A)

@inline maybestaticsize(::NTuple{N}, ::Val{1}) where {N} = Static{N}() # should we assert that i == 1?
@inline maybestaticlength(::NTuple{N}) where {N} = Static{N}()
@inline maybestaticsize(::Adjoint{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = Static{1}()
@inline maybestaticsize(::Transpose{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = Static{1}()
@inline maybestaticlength(B::Adjoint) = maybestaticlength(parent(B))
@inline maybestaticlength(B::Transpose) = maybestaticlength(parent(B))
@inline maybestaticsize(B::Adjoint{T,A}, ::Val{1}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{2}())
@inline maybestaticsize(B::Adjoint{T,A}, ::Val{2}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{1}())
@inline maybestaticsize(B::Transpose{T,A}, ::Val{1}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{2}())
@inline maybestaticsize(B::Transpose{T,A}, ::Val{2}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{1}())

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
@inline Base.:(==)(::Static{M}, i) where {M} = M == i
@inline Base.:(==)(i, ::Static{M}) where {M} = M == i
@inline Base.:(==)(::Static{M}, ::Static{N}) where {M,N} = false
@inline Base.:(==)(::Static{M}, ::Static{M}) where {M} = true

function static_promote(i, j)
    i == j || throw("$i ≠ $j")
    i
end
function static_promote(::Static{M}, i) where {M}
    M == i || throw("$M ≠ $i")
    Static{M}()
end
function static_promote(i, ::Static{M}) where {M}
    M == i || throw("$M ≠ $i")
    Static{M}()
end
static_promote(::Static{M}, ::Static{N}) where {M, N} = throw("$M ≠ $N")
static_promote(::Static{M}, ::Static{M}) where {M} = Static{M}()


@generated staticm1(::Static{N}) where {N} = Static{N-1}()
@inline staticm1(N::Integer) = N - 1
