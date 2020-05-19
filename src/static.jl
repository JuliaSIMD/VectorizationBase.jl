#TODO: Document interface to support static size
# Define maybestaticsize, maybestaticlength, and maybestaticfirstindex



struct Static{N} end
Base.@pure Static(N) = Static{N}()

Base.CartesianIndex(I::Tuple{<:Static,Vararg}) = CartesianVIndex(I)
Base.CartesianIndex(I::Tuple{<:Integer,<:Static,Vararg}) = CartesianVIndex(I)
Base.CartesianIndex(I::Tuple{<:Integer,<:Integer,<:Static,Vararg}) = CartesianVIndex(I)
Base.CartesianIndex(I::Tuple{<:Integer,<:Integer,<:Integer,<:Static,Vararg}) = CartesianVIndex(I)
Base.CartesianIndex(I::Tuple{<:Integer,<:Integer,<:Integer,<:Integer,<:Static,Vararg}) = CartesianVIndex(I)

abstract type AbstractStaticUnitRange <: AbstractUnitRange{Int} end
# rank 2, but 3 unknowns; 5 types can express all different posibilities
# 1: all unknown, UnitRange; 2: all three known (only two must be made explicit):
struct StaticUnitRange{L,U} <: AbstractStaticUnitRange end
Base.@pure StaticUnitRange(L,U) = StaticUnitRange{L,U}() # Do I use this definition anywhere?
# Then each specifying one of the three parameters
struct StaticLowerUnitRange{L} <: AbstractStaticUnitRange
    U::Int
end
struct StaticUpperUnitRange{U} <: AbstractStaticUnitRange
    L::Int
end
struct StaticLengthUnitRange{N} <: AbstractStaticUnitRange
    L::Int
end



@inline Base.first(::StaticUnitRange{L}) where {L} = L
@inline Base.first(::StaticLowerUnitRange{L}) where {L} = L
@inline Base.first(r::StaticUpperUnitRange) = r.L
@inline Base.first(r::StaticLengthUnitRange) = r.L
@inline Base.last(::StaticUnitRange{L,U}) where {L,U} = U
@inline Base.last(r::StaticLowerUnitRange) = r.U
@inline Base.last(::StaticUpperUnitRange{U}) where {U} = U
@inline Base.last(r::StaticLengthUnitRange{N}) where {N} = r.L + N - 1

@inline Base.iterate(x::AbstractStaticUnitRange) = (i = unwrap(first(x)); (i,i))

@inline Base.:(:)(::Static{L}, ::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline Base.:(:)(::Static{L}, U::Int) where {L} = StaticLowerUnitRange{L}(U)
@inline Base.:(:)(L::Int, ::Static{U}) where {U} = StaticUpperUnitRange{U}(L)
@inline unwrap(x) = x
@inline unwrap(::Val{N}) where {N} = N
@inline unwrap(::Type{Val{N}}) where {N} = N
@inline unwrap(::Static{N}) where {N} = N
@inline unwrap(::Type{Static{N}}) where {N} = N

@inline StaticLowerUnitRange{L}(::Static{U}) where {L,U} = StaticUnitRange{L,U}()
@inline StaticUpperUnitRange{U}(::Static{L}) where {L,U} = StaticUnitRange{L,U}()
@generated StaticLengthUnitRange{N}(::Static{L}) where {L,N} = StaticUnitRange{L,L+N-1}()

@inline maybestaticsize(A, ::Val{I}) where {I} = size(A, I)
@inline maybestaticsize(A::AbstractArray{<:Any,0}, ::Val{1:2}) = (1, 1)
@inline maybestaticsize(A::AbstractVector, ::Val{1:2}) = (length(A), 1)
@inline maybestaticsize(A::AbstractMatrix, ::Val{1:2}) = size(A)
@inline maybestaticsize(A::AbstractArray, ::Val{1:2}) = (size(A,1),size(A,2))
@inline maybestaticlength(A) = length(A)

@inline maybestaticfirstindex(A::AbstractArray, ::Val{I}) where {I} = firstindex(A, I)
@inline maybestaticfirstindex(A::Array, ::Val{I}) where {I} = Static{1}()
                  

# @inline maybestaticeachindex(A::AbstractArray) = maybestaticrange(eachindex(A))

@inline maybestaticrange(r::Base.OneTo) = StaticLowerUnitRange{1}(last(r))
@inline maybestaticrange(r::UnitRange) = r
@inline maybestaticrange(r::AbstractStaticUnitRange) = r
@inline maybestaticrange(r) = maybestaticfirst(r):maybestaticlast(r)
# @inline maybestaticaxis(A::AbstractArray, ::Val{I}) where {I} = maybestaticfirstindex(A, Val{I}()):maybestaticsize(A, Val{I}())

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

@inline Base.iszero(::Static{0}) = true
@inline Base.iszero(::Static) = false
@generated function Base.divrem(N::Integer, ::Static{L}) where {L}
    if ispow2(L)
        quote
            $(Expr(:meta,:inline))
            d = N >>> $(intlog2(L))
            r = N & $(L-1)
            d, r
        end
    else
        quote
            $(Expr(:meta,:inline))
            divrem(N, $L)
        end
    end
end
@generated Base.divrem(::Static{N}, ::Static{D}) where {N,D} = divrem(N, D)

@inline Base.:+(::Static{N}, i) where {N} = N + i
@inline Base.:+(i, ::Static{N}) where {N} = N + i
@inline Base.:+(::Static{M}, ::Static{N}) where {M,N} = M + N
@inline Base.:*(::Static{N}, i) where {N} = N * i
@inline Base.:*(i, ::Static{N}) where {N} = N * i
@inline Base.:*(::Static{M}, ::Static{N}) where {M,N} = M * N
@inline Base.:-(::Static{N}, i) where {N} = N - i
@inline Base.:-(i, ::Static{N}) where {N} = i - N
@inline Base.:-(::Static{M}, ::Static{N}) where {M,N} = M - N
@inline Base.checked_add(::Static{N}, i) where {N} = Base.checked_add(N, i)
@inline Base.checked_add(i, ::Static{N}) where {N} = Base.checked_add(i, N)
@generated Base.checked_add(::Static{M}, ::Static{N}) where {M,N} = Static{Base.checked_add(M, N)}()
@inline Base.checked_sub(::Static{N}, i) where {N} = Base.checked_sub(N, i)
@inline Base.checked_sub(i, ::Static{N}) where {N} = Base.checked_sub(i, N)
@generated Base.checked_sub(::Static{M}, ::Static{N}) where {M,N} = Static{Base.checked_sub(M, N)}()
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

@generated staticp1(::Static{N}) where {N} = Static{N+1}()
@inline staticp1(N) = vadd(N, one(N))
@inline staticp1(i::Tuple{}) = tuple()
@inline staticp1(i::Tuple{I}) where {I} = @inbounds (staticp1(i[1]),)
@inline staticp1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticp1(i[1]), staticp1(i[2]))
@inline staticp1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticp1(i[1]), staticp1(Base.tail(i))...)
@generated staticm1(::Static{N}) where {N} = Static{N-1}()
@inline staticm1(N) = vsub(N, one(N))
@inline staticm1(i::Tuple{}) = tuple()
@inline staticm1(i::Tuple{I}) where {I} = @inbounds (staticm1(i[1]),)
@inline staticm1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticm1(i[1]), staticm1(i[2]))
@inline staticm1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticm1(i[1]), staticm1(Base.tail(i))...)
@generated staticmul(::Type{T}, ::Static{N}) where {T,N} = Static{sizeof(T) * N}()
@inline staticmul(::Type{T}, N) where {T} = vmul(N, sizeof(T))
@inline staticmul(::Type{T}, i::Tuple{}) where {T} = tuple()
@inline staticmul(::Type{T}, i::Tuple{I}) where {T,I} = @inbounds (vmul(i[1], sizeof(T)),)
@inline staticmul(::Type{T}, i::Tuple{I1,I2}) where {T,I1,I2} = @inbounds (vmul(sizeof(T), i[1]), vmul(sizeof(T), i[2]))
@inline staticmul(::Type{T}, i::Tuple{I1,I2,I3,Vararg}) where {T,I1,I2,I3} = @inbounds (vmul(sizeof(T), i[1]), staticmul(T, Base.tail(i))...)
@inline Base.ntuple(f::F, ::Static{N}) where {F,N} = ntuple(f, Val{N}())

struct LazyP1{T}
    data::T
end
@inline staticm1(d::LazyP1) = d.data


@inline maybestaticfirst(A) = first(A)
@inline maybestaticfirst(::StaticUnitRange{L}) where {L} = Static{L}()
@inline maybestaticfirst(::StaticLowerUnitRange{L}) where {L} = Static{L}()
@inline maybestaticfirst(::Base.OneTo) = Static{1}()

@inline _maybestaticfirst(A::Tuple{}) = tuple()
@inline _maybestaticfirst(A::Tuple{I}) where {I} = (maybestaticfirst(@inbounds(A[1])),)
@inline _maybestaticfirst(A::Tuple{I1,I2}) where {I1,I2} = @inbounds (maybestaticfirst(A[1]), maybestaticfirst(A[2]))
@inline _maybestaticfirst(A::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = (maybestaticfirst(@inbounds A[1]), _maybestaticfirst(Base.tail(A))...)
@inline maybestaticfirst(A::CartesianIndices) = CartesianVIndex(_maybestaticfirst(A.indices))


@inline maybestaticlast(A) = last(A)
@inline maybestaticlast(::StaticUnitRange{L,U}) where {L,U} = Static{U}()
@inline maybestaticlast(::StaticUpperUnitRange{U}) where {U} = Static{U}()
@inline _maybestaticlast(A::Tuple{}) = tuple()
@inline _maybestaticlast(A::Tuple{I}) where {I} = (maybestaticlast(@inbounds(A[1])),)
@inline _maybestaticlast(A::Tuple{I1,I2}) where {I1,I2} = @inbounds (maybestaticlast(A[1]), maybestaticlast(A[2]))
@inline _maybestaticlast(A::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = (maybestaticlast(@inbounds A[1]), _maybestaticlast(Base.tail(A))...)
@inline maybestaticlast(A::CartesianIndices) = CartesianVIndex(_maybestaticlast(A.indices))

