
# abstract type AbstractLazyMul end
struct LazyMulAdd{M,O,T} <: Number
    data::T
end
# O for offset is kind of hard to read next to default of 0?
@inline LazyMulAdd{M,O}(data::T) where {M,O,T} = LazyMulAdd{M,O,T}(data)
@inline LazyMulAdd{M}(data::T) where {M,T} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd(data::T, ::Static{M}) where {M,T} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd(data::T, ::Static{M}, ::Static{O}) where {M,O,T} = LazyMulAdd{M,O,T}(data)

@inline data(lm::LazyMulAdd) = data(lm.data) # for use with indexing

@inline Base.convert(::Type{T}, a::LazyMulAdd{M,O,I}) where {M,O,I,T<:Number} = convert(T, vadd(vmul(M, a.data), Static{O}()))
Base.promote_rule(::Type{LazyMulAdd{M,O,I}}, ::Type{T}) where {M,O,I<:Number,T} = symmetric_promote_rule(I, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,MM{W,X,I}}}, ::Type{T}) where {M,O,W,X,I,T} = symmetric_promote_rule(Vec{W,I}, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,Vec{W,I}}}, ::Type{T}) where {M,O,W,I,T} = symmetric_promote_rule(Vec{W,I}, T)

@inline lazymul(a, b) = vmul(a, b)
@inline lazymul(::Static{M}, b) where {M} = LazyMulAdd{M}(b)
@inline lazymul(::Static{1}, b) = b
@inline lazymul(::Static{0}, b) = Static{0}()
@inline lazymul(a, ::Static{M}) where {M} = LazyMulAdd{M}(a)
@inline lazymul(a, ::Static{1}) = a
@inline lazymul(a, ::Static{0}) = Static{0}()
# @inline lazymul(::Static{M}, b::MM{W,X}) where {W,M,X} = LazyMulAdd{M}(MM{W}(b.data, Static{M}()*Static{X}()))
# @inline lazymul(a::MM{W,X}, ::Static{M}) where {W,M,X} = LazyMulAdd{M}(MM{W}(a.data, Static{M}()*Static{X}()))
@inline lazymul(::Static{M}, b::MM{W,X}) where {W,M,X} = MM{W}(vmul(Static{M}(), data(b)), Static{X}() * Static{M}())
@inline lazymul(a::MM{W,X}, ::Static{M}) where {W,M,X} = MM{W}(vmul(Static{M}(), data(b)), Static{X}() * Static{M}())
# @inline lazymul(::Static{M}, b::MM{W,X,Static{N}}) where {W,M,X,N} = MM{W}(Static{N}() * Static{M}(), Static{X}() * Static{M}())
# @inline lazymul(a::MM{W,X,Static{N}}, ::Static{M}) where {W,M,X,N} = MM{W}(Static{N}() * Static{M}(), Static{X}() * Static{M}())

# @inline lazymul(::Static{N}, b::MM) where {N} = LazyMulAdd{N}(Vec(b))
# @inline lazymul(a::MM, ::Static{N}) where {N} = LazyMulAdd{N}(Vec(a))
@inline lazymul(::Static{M}, ::Static{N}) where {M, N} = Static{M*N}()
@inline lazymul(::Static{0}, ::Static) = Static{0}()
@inline lazymul(::Static{1}, ::Static{M}) where {M} = Static{M}()
@inline lazymul(::Static, ::Static{0}) = Static{0}()
@inline lazymul(::Static{M}, ::Static{1}) where {M} = Static{M}()
@inline lazymul(::Static{0}, ::Static{0}) = Static{0}()
@inline lazymul(::Static{0}, ::Static{1}) = Static{0}()
@inline lazymul(::Static{1}, ::Static{0}) = Static{0}()
@inline lazymul(::Static{1}, ::Static{1}) = Static{1}()

@inline lazymul(a::LazyMulAdd{M}, ::Static{N}) where {M,N} = LazyMulAdd(a.data, Static{M}() * Static{N}())
@inline lazymul(::Static{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(b.data, Static{M}() * Static{N}())

@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::Static{N}) where {M,N,W,X} = LazyMulAdd(MM{W}(a.data, Static{M}()*Static{N}()*Static{X}()), Static{M}()*Static{N}())
@inline lazymul(::Static{M}, b::LazyMulAdd{N,<:MM{W,X}}) where {M,N,W,X} = LazyMulAdd(MM{W}(b.data, Static{M}()*Static{N}()*Static{X}()), Static{M}()*Static{N}())
@inline lazymul(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(vmul(a.data, b.data), Static{M}()*Static{N}())

@inline lazymul_no_promote(a, b) = vmul_no_promote(a, b)
@inline lazymul_no_promote(::Static{N}, b) where {N} = LazyMulAdd{N}(b)
@inline lazymul_no_promote(a, ::Static{N}) where {N} = LazyMulAdd{N}(a)
@inline lazymul_no_promote(::Static{N}, b::MM{W,X,Static{M}}) where {N,W,X,M} = MM{W,X}(Static{M}() * Static{N}())
@inline lazymul_no_promote(a::MM{W,X,Static{M}}, ::Static{N}) where {N,W,X,M} = MM{W,X}(Static{M}() * Static{N}())
@inline lazymul_no_promote(::Static{M}, ::Static{N}) where {M, N} = Static{M*N}()

@inline lazymul_no_promote(a::LazyMulAdd{M}, ::Static{N}) where {M,N} = LazyMulAdd(a.data, Static{M}()*Static{N}())
@inline lazymul_no_promote(::Static{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(b.data, Static{M}()*Static{N}())
@inline lazymul_no_promote(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(vmul(a.data, b.data), Static{M}()*Static{N}())

@inline lazyadd(a, b) = vadd(a, b)
@inline lazyadd(a::LazyMulAdd{M,O,T}, ::Static{A}) where {M,O,T,A} = LazyMulAdd(a.data, Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(::Static{A}, a::LazyMulAdd{M,O,T}) where {M,O,T,A} = LazyMulAdd(a.data, Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(a::LazyMulAdd{M,O,T}, ::MM{W,X,Static{A}}) where {M,O,T<:Integer,A,W,X} = LazyMulAdd(MM{W,X}(a.data), Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(::MM{W,X,Static{A}}, a::LazyMulAdd{M,O,T}) where {M,O,T<:Integer,A,W,X} = LazyMulAdd(MM{W,X}(a.data), Static{M}(), Static{O}()+Static{A}())

@inline lazyadd(a::LazyMulAdd{M,O}, b::LazyMulAdd{M,A}) where {M,O,A} = LazyMulAdd(vadd(a.data, b.data), Static{M}(), Static{O}()+Static{A}())

@inline vadd(a::LazyMulAdd{M,O,MM{W,X,I}}, b::Integer) where {M,O,W,X,I} = MM{W,X}(vadd(vmul(Static{M}(), data(a)), vadd(Static{0}(), b)))
@inline vadd(b::Integer, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I} = MM{W,X}(vadd(vmul(Static{M}(), data(a)), vadd(Static{0}(), b)))


