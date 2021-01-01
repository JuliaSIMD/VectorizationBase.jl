#TODO: Document interface to support static size
# Define maybestaticsize, maybestaticlength, and maybestaticfirstindex


@inline static(::Val{N}) where {N} = StaticInt{N}()
@inline static(::Nothing) = nothing
@generated function static_sizeof(::Type{T}) where {T}
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticInt, sizeof(T))))
end

@inline maybestaticfirst(a) = static_first(a)
@inline maybestaticlast(a) = static_last(a)
@inline maybestaticlength(a) = static_length(a)
@inline maybestaticlength(a::UnitRange{T}) where {T} = last(a) - first(a) + oneunit(T)

@inline maybestaticrange(r::Base.OneTo{T}) where {T} = ArrayInterface.OptionallyStaticUnitRange(StaticInt{1}(), last(r))
@inline maybestaticrange(r::UnitRange) = r
@inline maybestaticrange(r) = maybestaticfirst(r):maybestaticlast(r)

@inline maybestaticsize(::NTuple{N}, ::Val{1}) where {N} = StaticInt{N}() # should we assert that i == 1?
@inline maybestaticsize(::LinearAlgebra.Adjoint{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticsize(::LinearAlgebra.Transpose{T,V}, ::Val{1}) where {T,V<:AbstractVector{T}} = One()
@inline maybestaticsize(A, ::Val{N}) where {N} = ArrayInterface.size(A)[N]

# These have versions that may allow for more optimizations, so we override base methods with a single `StaticInt` argument.
for (f,ff) ∈ [(:(Base.:+),:vadd), (:(Base.:-),:vsub), (:(Base.:*),:vmul), (:(Base.:<<),:vshl), (:(Base.:÷),:vdiv), (:(Base.:%), :vrem), (:(Base.:>>>),:vashr)]
    @eval begin
        # @inline $f(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{$f(M, N)}()
    # If `M` and `N` are known at compile time, there's no need to add nsw/nuw flags.
        @inline $ff(::StaticInt{M}, ::StaticInt{N}) where {M, N} = $f(StaticInt{M}(),StaticInt{N}())
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

@inline staticp1(::StaticInt{N}) where {N} = StaticInt{N}() + One()
@inline staticp1(N) = vadd(N, One())
@inline staticp1(i::Tuple{}) = tuple()
@inline staticp1(i::Tuple{I}) where {I} = @inbounds (staticp1(i[1]),)
@inline staticp1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticp1(i[1]), staticp1(i[2]))
@inline staticp1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticp1(i[1]), staticp1(Base.tail(i))...)
@inline staticm1(::StaticInt{N}) where {N} = StaticInt{N}() - One()
@inline staticm1(N) = vsub(N, one(N))
@inline staticm1(i::Tuple{}) = tuple()
@inline staticm1(i::Tuple{I}) where {I} = @inbounds (staticm1(i[1]),)
@inline staticm1(i::Tuple{I1,I2}) where {I1,I2} = @inbounds (staticm1(i[1]), staticm1(i[2]))
@inline staticm1(i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3} = @inbounds (staticm1(i[1]), staticm1(Base.tail(i))...)
@inline staticmul(::Type{T}, ::StaticInt{N}) where {T,N} = static_sizeof(T) * StaticInt{N}()
@inline staticmul(::Type{T}, ::Val{N}) where {T,N} = static_sizeof(T) * StaticInt{N}()
@inline staticmul(::Type{T}, N) where {T} = vmul(N, sizeof(T))
@inline staticmul(::Type{T}, i::Tuple{}) where {T} = tuple()
@inline staticmul(::Type{T}, i::Tuple{I}) where {T,I} = @inbounds (vmul(i[1], sizeof(T)),)
@inline staticmul(::Type{T}, i::Tuple{I1,I2}) where {T,I1,I2} = @inbounds (vmul(sizeof(T), i[1]), vmul(sizeof(T), i[2]))
@inline staticmul(::Type{T}, i::Tuple{I1,I2,I3,Vararg}) where {T,I1,I2,I3} = @inbounds (vmul(sizeof(T), i[1]), staticmul(T, Base.tail(i))...)
for T ∈ [:VecUnroll, :Mask, :MM]
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


