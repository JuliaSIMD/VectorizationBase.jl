#TODO: Document interface to support static size
# Define maybestaticsize, maybestaticlength, and maybestaticfirstindex


@inline static(::Val{N}) where {N} = StaticInt{N}()
@inline static(::Nothing) = nothing
@generated function static_sizeof(::Type{T}) where {T}
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticInt, sizeof(T))))
end

# @inline static_last(::Type{T}) where {T} = static(known_last(T))
# @inline static_first(::Type{T}) where {T} = static(known_first(T))
# @inline static_length(::Type{T}) where {T} = static(known_length(T))

@inline maybestaticfirst(a) = static_first(a)
@inline maybestaticlast(a) = static_last(a)
@inline maybestaticlength(a) = static_length(a)

# @inline _maybestaticfirst(a, ::Nothing) = first(a)
# @inline _maybestaticfirst(::Any, L) = StaticInt{L}()
# @inline maybestaticfirst(a::T) where {T} = _maybestaticfirst(a, known_first(T))

# @inline _maybestaticlast(a, ::Nothing) = last(a)
# @inline _maybestaticlast(::Any, L) = StaticInt{L}()
# @inline maybestaticlast(a::T) where {T} = _maybestaticlast(a, known_last(T))



# @inline _maybestaticlength(a, L::Int, U::Int) = StaticInt{U-L}()
# @inline _maybestaticlength(a, ::Any, ::Any) = length(a)
# @inline _maybestaticlength(a, ::Nothing) = _maybestaticlength(a, known_first(a), known_last(a))
# @inline _maybestaticlength(::Any, L::Int) = StaticInt{L}()
# @inline maybestaticlength(a::T) where {T} = _maybestaticlength(a, known_length(T))

@inline maybestaticrange(r::Base.OneTo{T}) where {T} = ArrayInterface.OptionallyStaticUnitRange(StaticInt{1}(), last(r))
@inline maybestaticrange(r::UnitRange) = r
# @inline maybestaticrange(r::AbstractStaticIntUnitRange) = r
@inline maybestaticrange(r) = maybestaticfirst(r):maybestaticlast(r)
# @inline maybestaticaxis(A::AbstractArray, ::Val{I}) where {I} = maybestaticfirstindex(A, Val{I}()):maybestaticsize(A, Val{I}())

@inline maybestaticsize(::NTuple{N}, ::Val{1}) where {N} = StaticInt{N}() # should we assert that i == 1?
@inline maybestaticlength(::NTuple{N}) where {N} = StaticInt{N}()
@inline maybestaticsize(::LinearAlgebra.Adjoint{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticsize(::LinearAlgebra.Transpose{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticlength(B::LinearAlgebra.Adjoint) = maybestaticlength(parent(B))
@inline maybestaticlength(B::LinearAlgebra.Transpose) = maybestaticlength(parent(B))
@inline maybestaticsize(A, ::Val{N}) where {N} = ArrayInterface.size(A)[N]
# @inline maybestaticsize(B::LinearAlgebra.Adjoint{T,A}, ::Val{1}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{2}())
# @inline maybestaticsize(B::LinearAlgebra.Adjoint{T,A}, ::Val{2}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{1}())
# @inline maybestaticsize(B::LinearAlgebra.Transpose{T,A}, ::Val{1}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{2}())
# @inline maybestaticsize(B::LinearAlgebra.Transpose{T,A}, ::Val{2}) where {T,A<:AbstractMatrix{T}} = maybestaticsize(parent(B), Val{1}())


# These have versions that may allow for more optimizations, so we override base methods with a single `StaticInt` argument.
for (f,ff) ∈ [(:(Base.:+),:vadd), (:(Base.:-),:vsub), (:(Base.:*),:vmul), (:(Base.:<<),:vshl), (:(Base.:÷),:vdiv), (:(Base.:%), :vrem), (:(Base.:>>>),:vashr)]
    @eval begin
        # @inline $f(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{$f(M, N)}()
    # If `M` and `N` are known at compile time, there's no need to add nsw/nuw flags.
        @inline $ff(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{$f(M, N)}()
        # @inline $f(::StaticInt{M}, x) where {M} = $ff(M, x)
        @inline $ff(::StaticInt{M}, x) where {M} = $ff(M, x)
        # @inline $f(x, ::StaticInt{M}) where {M} = $ff(x, M)
        @inline $ff(x, ::StaticInt{M}) where {M} = $ff(x, M)
    end
end
for f ∈ [:vadd, :vsub, :vmul]
    @eval @inline $f(::StaticInt{M}, n::Number) where {M} = $f(M, n)
    @eval @inline $f(m::Number, ::StaticInt{N}) where {N} = $f(m, N)
end

@inline vsub(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
# @inline vsub(::Zero, ::StaticInt{N}) where {N} = StaticInt{-N}()
@inline vsub(::Zero, ::Zero) = Zero()
@inline vsub(a::Number, ::Zero) = a
@inline vsub(a, ::Zero) = a

@inline vadd(::StaticInt{N}, ::Zero) where {N} = StaticInt{N}()
@inline vadd(::Zero, ::StaticInt{N}) where {N} = StaticInt{N}()
@inline vadd(::Zero, ::Zero) = Zero()
@inline vadd(a::Number, ::Zero) = a
@inline vadd(::Zero, a::Number) = a

@inline vmul(::StaticInt{N}, ::Zero) where {N} = Zero()
@inline vmul(::Zero, ::StaticInt{N}) where {N} = Zero()
@inline vmul(::Zero, ::Zero) = Zero()
@inline vmul(::StaticInt{N}, ::One) where {N} = StaticInt{N}()
@inline vmul(::One, ::StaticInt{N}) where {N} = StaticInt{N}()
@inline vmul(::One, ::One) = One()
@inline vmul(a::Number, ::One) = a
@inline vmul(::One, a::Number) = a
@inline vmul(::Zero, ::One) = Zero()
@inline vmul(::One, ::Zero) = Zero()
@inline vmul(i::MM{W,X}, ::StaticInt{N}) where {W,X,N} = MM{W}(vmul(data(i), StaticInt{N}()), StaticInt{X}() * StaticInt{N}())
@inline vmul(i::MM{W,X}, ::StaticInt{1}) where {W,X} = i
@inline vmul(::StaticInt{N}, i::MM{W,X}) where {W,X,N} = MM{W}(vmul(data(i), StaticInt{N}()), StaticInt{X}() * StaticInt{N}())
@inline vmul(::StaticInt{1}, i::MM{W,X}) where {W,X} = i

@generated staticp1(::StaticInt{N}) where {N} = StaticInt{N+1}()
@inline staticp1(N) = vadd(N, one(N))
@inline staticp1(i::Tuple{}) = tuple()
@inline staticp1(i::Tuple{I}) where {I} = @inbounds (staticp1(i[1]),)
@inline staticp1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticp1(i[1]), staticp1(i[2]))
@inline staticp1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticp1(i[1]), staticp1(Base.tail(i))...)
@generated staticm1(::StaticInt{N}) where {N} = StaticInt{N-1}()
@inline staticm1(N) = vsub(N, one(N))
@inline staticm1(i::Tuple{}) = tuple()
@inline staticm1(i::Tuple{I}) where {I} = @inbounds (staticm1(i[1]),)
@inline staticm1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticm1(i[1]), staticm1(i[2]))
@inline staticm1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticm1(i[1]), staticm1(Base.tail(i))...)
@generated staticmul(::Type{T}, ::StaticInt{N}) where {T,N} = sizeof(T) * N
@generated staticmul(::Type{T}, ::Val{N}) where {T,N} = sizeof(T) * N
@inline staticmul(::Type{T}, N) where {T} = vmul(N, sizeof(T))
@inline staticmul(::Type{T}, i::Tuple{}) where {T} = tuple()
@inline staticmul(::Type{T}, i::Tuple{I}) where {T,I} = @inbounds (vmul(i[1], sizeof(T)),)
@inline staticmul(::Type{T}, i::Tuple{I1,I2}) where {T,I1,I2} = @inbounds (vmul(sizeof(T), i[1]), vmul(sizeof(T), i[2]))
@inline staticmul(::Type{T}, i::Tuple{I1,I2,I3,Vararg}) where {T,I1,I2,I3} = @inbounds (vmul(sizeof(T), i[1]), staticmul(T, Base.tail(i))...)
for T ∈ [:VecUnroll, :Mask]
    @eval begin
        @inline Base.:(+)(x::$T, ::Zero) = x
        @inline Base.:(+)(::Zero, x::$T) = x
        @inline Base.:(-)(x::$T, ::Zero) = x
        @inline Base.:(*)(x::$T, ::One) = x
        @inline Base.:(*)(::One, x::$T) = x
        @inline Base.:(*)(::$T, ::Zero) = Zero()
        @inline Base.:(*)(::Zero, ::$T) = Zero()
    end
end
@inline Base.:(+)(m::Mask{W}, ::StaticInt{N}) where {N,W} = m + vbroadcast(Val{W}(), N)
@inline Base.:(+)(::StaticInt{N}, m::Mask{W}) where {N,W} = vbroadcast(Val{W}(), N) + m
# @inline Base.:(*)(::StaticInt{N}, m::Mask{W}) where {N,W} = vbroadcast(Val{W}(), N) * m
@inline vadd(x::VecUnroll, ::Zero) = x
@inline vadd(::Zero, x::VecUnroll) = x
@inline vsub(x::VecUnroll, ::Zero) = x
@inline vmul(x::VecUnroll, ::One) = x
@inline vmul(::One, x::VecUnroll) = x
@inline vmul(::VecUnroll, ::Zero) = Zero()
@inline vmul(::Zero, ::VecUnroll) = Zero()


