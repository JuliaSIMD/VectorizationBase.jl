
# abstract type AbstractLazyMul end
struct LazyMulAdd{M,O,T<:NativeTypesV} <: Number
    data::T
    @inline LazyMulAdd{M,O,T}(data::T) where {M,O,T<:NativeTypesV} = new{M,O,T}(data)
end
# O for offset is kind of hard to read next to default of 0?
@inline LazyMulAdd{M,O}(data::T) where {M,O,T<:Union{Base.HWReal,AbstractSIMD}} = LazyMulAdd{M,O,T}(data)
@inline LazyMulAdd{M}(data::T) where {M,T<:NativeTypesV} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd{0,O}(data::T) where {M,O,T<:Union{Base.HWReal,AbstractSIMD}} = @assert false#StaticInt{O}()
@inline LazyMulAdd{0}(data::T) where {M,T<:NativeTypesV} = @assert false#StaticInt{0}()
@inline LazyMulAdd(data::T, ::StaticInt{M}) where {M,T} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd(data::T, ::StaticInt{M}, ::StaticInt{O}) where {M,O,T} = LazyMulAdd{M,O,T}(data)

@inline data(lm::LazyMulAdd) = data(lm.data) # for use with indexing

@inline Base.convert(::Type{T}, a::LazyMulAdd{M,O,I}) where {M,O,I,T<:Number} = convert(T, vadd(vmul(M, a.data), StaticInt{O}()))
Base.promote_rule(::Type{LazyMulAdd{M,O,I}}, ::Type{T}) where {M,O,I<:Number,T} = symmetric_promote_rule(I, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,MM{W,X,I}}}, ::Type{T}) where {M,O,W,X,I,T} = symmetric_promote_rule(Vec{W,I}, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,Vec{W,I}}}, ::Type{T}) where {M,O,W,I,T} = symmetric_promote_rule(Vec{W,I}, T)

@inline lazymul(a, b) = vmul(a, b)
@inline lazymul(::StaticInt{M}, b) where {M} = LazyMulAdd{M}(b)
@inline lazymul(::StaticInt{1}, b) = b
@inline lazymul(::StaticInt{0}, b) = StaticInt{0}()
@inline lazymul(a, ::StaticInt{M}) where {M} = LazyMulAdd{M}(a)
@inline lazymul(a, ::StaticInt{1}) = a
@inline lazymul(a, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(a::LazyMulAdd, ::StaticInt{1}) = a
@inline lazymul(::StaticInt{1}, a::LazyMulAdd) = a
@inline lazymul(a::LazyMulAdd, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{0}, a::LazyMulAdd) = StaticInt{0}()
# @inline lazymul(a::LazyMulAdd, ::StaticInt{0}) = StaticInt{0}()
# @inline lazymul(::StaticInt{M}, b::MM{W,X}) where {W,M,X} = LazyMulAdd{M}(MM{W}(b.data, StaticInt{M}()*StaticInt{X}()))
# @inline lazymul(a::MM{W,X}, ::StaticInt{M}) where {W,M,X} = LazyMulAdd{M}(MM{W}(a.data, StaticInt{M}()*StaticInt{X}()))
@inline lazymul(::StaticInt{M}, b::MM{W,X}) where {W,M,X} = MM{W}(vmul(StaticInt{M}(), data(b)), StaticInt{X}() * StaticInt{M}())
@inline lazymul(a::MM{W,X}, ::StaticInt{M}) where {W,M,X} = MM{W}(vmul(StaticInt{M}(), data(a)), StaticInt{X}() * StaticInt{M}())
@inline lazymul(a::MM{W,X}, ::StaticInt{0}) where {W,X} = StaticInt{0}()
@inline lazymul(::StaticInt{0}, a::MM{W,X}) where {W,X} = StaticInt{0}()
@inline lazymul(a::MM{W,X}, ::StaticInt{1}) where {W,X} = a
@inline lazymul(::StaticInt{1}, a::MM{W,X}) where {W,X} = a

@inline lazymul_no_promote(::Type{T}, a, b) where {T} = lazymul_no_promote(a, b)
@inline lazymul_no_promote(::Type{T}, a::MM{W,X}, b::StaticInt) where {W,X,T} = MM{W}(vmul(a.i, b), mulsizeof(T, StaticInt{X}()))
@inline lazymul_no_promote(::Type{T}, a::MM{W,X}, b::Integer) where {W,X,T} = MM{W}(vmul(a.i, b), mulsizeof(T, StaticInt{X}()))
@inline lazymul_no_promote(::Type{T}, b::StaticInt, a::MM{W,X}) where {W,X,T} = MM{W}(vmul(b, a.i), mulsizeof(T, StaticInt{X}()))
@inline lazymul_no_promote(::Type{T}, b::Integer, a::MM{W,X}) where {W,X,T} = MM{W}(vmul(b, a.i), mulsizeof(T, StaticInt{X}()))

# @inline lazymul(::StaticInt{M}, b::MM{W,X,StaticInt{N}}) where {W,M,X,N} = MM{W}(StaticInt{N}() * StaticInt{M}(), StaticInt{X}() * StaticInt{M}())
# @inline lazymul(a::MM{W,X,StaticInt{N}}, ::StaticInt{M}) where {W,M,X,N} = MM{W}(StaticInt{N}() * StaticInt{M}(), StaticInt{X}() * StaticInt{M}())

# @inline lazymul(::StaticInt{N}, b::MM) where {N} = LazyMulAdd{N}(Vec(b))
# @inline lazymul(a::MM, ::StaticInt{N}) where {N} = LazyMulAdd{N}(Vec(a))
@inline lazymul(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{M*N}()
@inline lazymul(::StaticInt{0}, ::StaticInt) = StaticInt{0}()
@inline lazymul(::StaticInt{1}, ::StaticInt{M}) where {M} = StaticInt{M}()
@inline lazymul(::StaticInt, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{M}, ::StaticInt{1}) where {M} = StaticInt{M}()
@inline lazymul(::StaticInt{0}, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{0}, ::StaticInt{1}) = StaticInt{0}()
@inline lazymul(::StaticInt{1}, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{1}, ::StaticInt{1}) = StaticInt{1}()

@inline lazymul(a::LazyMulAdd{M}, ::StaticInt{N}) where {M,N} = LazyMulAdd(a.data, StaticInt{M}() * StaticInt{N}())
@inline lazymul(::StaticInt{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(b.data, StaticInt{M}() * StaticInt{N}())

@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{N}) where {M,N,W,X} = LazyMulAdd(MM{W}(a.data, StaticInt{M}()*StaticInt{N}()*StaticInt{X}()), StaticInt{M}()*StaticInt{N}())
@inline lazymul(::StaticInt{M}, b::LazyMulAdd{N,<:MM{W,X}}) where {M,N,W,X} = LazyMulAdd(MM{W}(b.data, StaticInt{M}()*StaticInt{N}()*StaticInt{X}()), StaticInt{M}()*StaticInt{N}())
@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{0}) where {M,W,X} = StaticInt{0}()
@inline lazymul(::StaticInt{0}, b::LazyMulAdd{N,<:MM{W,X}}) where {N,W,X} = StaticInt{0}()
@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{1}) where {M,W,X} = a
@inline lazymul(::StaticInt{1}, b::LazyMulAdd{N,<:MM{W,X}}) where {N,W,X} = b
@inline lazymul(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(vmul(a.data, b.data), StaticInt{M}()*StaticInt{N}())

@inline lazymul_no_promote(a, b) = vmul_no_promote(a, b)
@inline lazymul_no_promote(::StaticInt{N}, b) where {N} = LazyMulAdd{N}(b)
@inline lazymul_no_promote(a, ::StaticInt{N}) where {N} = LazyMulAdd{N}(a)
@inline lazymul_no_promote(::StaticInt{N}, b::MM{W,X,StaticInt{M}}) where {N,W,X,M} = MM{W,X}(StaticInt{M}() * StaticInt{N}())
@inline lazymul_no_promote(a::MM{W,X,StaticInt{M}}, ::StaticInt{N}) where {N,W,X,M} = MM{W,X}(StaticInt{M}() * StaticInt{N}())
@inline lazymul_no_promote(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{M*N}()


@inline lazymul_no_promote(a::LazyMulAdd{M}, ::StaticInt{N}) where {M,N} = LazyMulAdd(a.data, StaticInt{M}()*StaticInt{N}())
@inline lazymul_no_promote(::StaticInt{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(b.data, StaticInt{M}()*StaticInt{N}())
@inline lazymul_no_promote(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N} = LazyMulAdd(vmul(a.data, b.data), StaticInt{M}()*StaticInt{N}())

@inline lazymul_no_promote(::StaticInt{0}, b) = StaticInt{0}()
@inline lazymul_no_promote(a, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul_no_promote(::StaticInt{0}, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul_no_promote(::StaticInt{0}, ::StaticInt{M}) where {M} = StaticInt{0}()
@inline lazymul_no_promote(::StaticInt{M}, ::StaticInt{0}) where {M} = StaticInt{0}()
@inline lazymul_no_promote(::StaticInt{0}, b::MM{W,X,StaticInt{M}}) where {W,X,M} = StaticInt{0}()
@inline lazymul_no_promote(a::MM{W,X,StaticInt{M}}, ::StaticInt{0}) where {W,X,M} = StaticInt{0}()
@inline lazymul_no_promote(a::LazyMulAdd{M}, ::StaticInt{0}) where {M} = StaticInt{0}()
@inline lazymul_no_promote(::StaticInt{0}, b::LazyMulAdd{N}) where {N} = StaticInt{0}()



# @inline lazyadd(a, b) = vadd(a, b)
@inline vadd(a::LazyMulAdd{M,O,T}, ::StaticInt{A}) where {M,O,T,A} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
@inline vadd(::StaticInt{A}, a::LazyMulAdd{M,O,T}) where {M,O,T,A} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
@inline vadd(a::LazyMulAdd{M,O,T}, ::StaticInt{0}) where {M,O,T} = a
@inline vadd(::StaticInt{0}, a::LazyMulAdd{M,O,T}) where {M,O,T} = a
# @inline vadd(a::LazyMulAdd{M,O,T}, ::StaticInt{A}) where {M,O,T<:MM,A} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
# @inline vadd(::StaticInt{A}, a::LazyMulAdd{M,O,T}) where {M,O,T,A} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
@inline vadd(a::LazyMulAdd{M,O,T}, ::MM{W,X,StaticInt{A}}) where {M,O,T<:Integer,A,W,X} = LazyMulAdd(MM{W,X}(a.data), StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
@inline vadd(::MM{W,X,StaticInt{A}}, a::LazyMulAdd{M,O,T}) where {M,O,T<:Integer,A,W,X} = LazyMulAdd(MM{W,X}(a.data), StaticInt{M}(), StaticInt{O}()+StaticInt{A}())

@inline vadd(a::LazyMulAdd{M,O}, b::LazyMulAdd{M,A}) where {M,O,A} = LazyMulAdd(vadd(a.data, b.data), StaticInt{M}(), StaticInt{O}()+StaticInt{A}())

@inline vadd(a::LazyMulAdd{M,O,MM{W,X,I}}, b::Integer) where {M,O,W,X,I} = MM{W,X}(vadd(vmul(StaticInt{M}(), data(a)), vadd(StaticInt{0}(), b)))
@inline vadd(::StaticInt{N}, a::LazyMulAdd{M,O,MM{W,X,I}}) where {N,M,O,W,X,I} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
@inline vadd(a::LazyMulAdd{M,O,MM{W,X,I}}, ::StaticInt{N}) where {N,M,O,W,X,I} = LazyMulAdd(a.data, StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
@inline vadd(b::Integer, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I} = MM{W,X}(vadd(vmul(StaticInt{M}(), data(a)), vadd(StaticInt{0}(), b)))
@inline vadd(a::LazyMulAdd{M,O,MM{W,X,I}}, ::StaticInt{0}) where {M,O,W,X,I} = a
@inline vadd(::StaticInt{0}, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I} = a
# @inline vadd(::StaticInt{M}, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I} = a


