
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


@inline lazymul(a, b) = vmul(a, b)
@inline lazymul(::Static{M}, b) where {M} = LazyMulAdd{M}(b)
@inline lazymul(::Static{1}, b) = b
@inline lazymul(::Static{0}, b) = Static{0}()
@inline lazymul(a, ::Static{M}) where {M} = LazyMulAdd{M}(a)
@inline lazymul(a, ::Static{1}) = a
@inline lazymul(a, ::Static{0}) = Static{0}()
@inline lazymul(::Static{M}, b::MM{W,X}) where {W,M,X} = LazyMulAdd{M}(MM{W}(b.data, Static{M}()*Static{X}()))
@inline lazymul(a::MM{W,X}, ::Static{M}) where {W,M,X} = LazyMulAdd{M}(MM{W}(a.data, Static{M}()*Static{X}()))
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

@inline lazymul_no_prmote(a, b) = vmul_no_promote(a, b)
@inline lazymul_no_prmote(::Static{N}, b) where {N} = LazyMulAdd{N}(b)
@inline lazymul_no_prmote(a, ::Static{N}) where {N} = LazyMulAdd{N}(a)
@inline lazymul_no_prmote(::Static{M}, ::Static{N}) where {M, N} = Static{M*N}()

@inline lazymul_no_prmote(a::LazyMulAdd{M}, ::Static{N}) where {M,N} = LazyMulAdd(a.data, Static{M}()*Static{N}())
@inline lazymul_no_prmote(::Static{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(b.data, Static{M}()*Static{N}())
@inline lazymul_no_prmote(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(vmul(a.data, b.data), Static{M}()*Static{N}())

@inline lazyadd(a, b) = vadd(a, b)
@inline lazyadd(a::LazyMulAdd{M,O,T}, ::Static{A}) where {M,O,T,A} = LazyMulAdd(a.data, Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(::Static{A}, a::LazyMulAdd{M,O,T}) where {M,O,T,A} = LazyMulAdd(a.data, Static{M}(), Static{O}()+Static{A}())

@inline lazyadd(a::LazyMulAdd{M,O}, b::LazyMulAdd{M,A}) where {M,O,A} = LazyMulAdd(vadd(a.data, b.data), Static{M}(), Static{O}()+Static{A}())

@inline lazyadd(a::LazyMulAdd{M,O,MM{W,X,T1}}, b::LazyMulAdd{N,A,T2}) where {M,N,O,A,W,X,T1,T2} = LazyMulAdd(vadd(a.data, b.data), Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(a::LazyMulAdd{M,O,T1}, b::LazyMulAdd{N,A,MM{W,X,T2}}) where {M,N,O,A,W,X,T1,T2} = LazyMulAdd(vadd(a.data, b.data), Static{M}(), Static{O}()+Static{A}())
@inline lazyadd(a::LazyMulAdd{M,O,MM{W,X1,T1}}, b::LazyMulAdd{N,A,MM{W,X2,T2}}) where {M,N,O,A,W,X,T1,T2} = LazyMulAdd(vadd(a.data, b.data), Static{M}(), Static{O}()+Static{A}())
# @inline lazyadd(a::LazyMulAdd{M,O,T1}, b::LazyMulAdd{N,A,T}) where {M,O,A,T1,T2} = LazyMulAdd(a.data, Static{M}(), Static{O}()+Static{A}())


