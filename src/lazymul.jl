struct LazyMulAdd{M,O,T<:NativeTypesV} <: Number
    data::T
    @inline LazyMulAdd{M,O,T}(data::T) where {M,O,T<:NativeTypesV} = new{M,O,T}(data)
end
# O for offset is kind of hard to read next to default of 0?
@inline LazyMulAdd{M,O}(data::T) where {M,O,T<:Union{Base.HWReal,AbstractSIMD}} = LazyMulAdd{M,O,T}(data)
@inline LazyMulAdd{M}(data::T) where {M,T<:NativeTypesV} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd{0,O}(data::T) where {O,T<:Union{Base.HWReal,AbstractSIMD}} = @assert false#StaticInt{O}()
@inline LazyMulAdd{0}(data::T) where {T<:NativeTypesV} = @assert false#StaticInt{0}()
@inline LazyMulAdd(data::T, ::StaticInt{M}) where {M,T} = LazyMulAdd{M,0,T}(data)
@inline LazyMulAdd(data::T, ::StaticInt{M}, ::StaticInt{O}) where {M,O,T} = LazyMulAdd{M,O,T}(data)

@inline data(lm::LazyMulAdd) = data(getfield(lm, :data)) # calls data on inner for use with indexing (normally `data` only goes through one layer)

@inline _materialize(a::LazyMulAdd{M,O,I}) where {M,O,I} = vadd_fast(vmul_fast(StaticInt{M}(), getfield(a, :data)), StaticInt{O}())
@inline _materialize(x) = x
@inline Base.convert(::Type{T}, a::LazyMulAdd{M,O,I}) where {M,O,I,T<:Number} = convert(T, _materialize(a))
@inline Base.convert(::Type{LazyMulAdd{M,O,I}}, a::LazyMulAdd{M,O,I}) where {M,O,I} = a
# @inline Base.convert(::Type{LazyMulAdd{M,O,I}}, a::LazyMulAdd{M}) where {M,O,I} = a
# @inline Base.convert(::Type{LazyMulAdd{M,T,I}}, a::LazyMulAdd{M,StaticInt{O},I}) where {M,O,I,T} = a

Base.promote_rule(::Type{LazyMulAdd{M,O,I}}, ::Type{T}) where {M,O,I<:Number,T} = promote_type(I, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,MM{W,X,I}}}, ::Type{T}) where {M,O,W,X,I,T} = promote_type(MM{W,X,I}, T)
Base.promote_rule(::Type{LazyMulAdd{M,O,Vec{W,I}}}, ::Type{T}) where {M,O,W,I,T} = promote_type(Vec{W,I}, T)

@inline lazymul(a, b) = vmul_fast(a, b)
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
# @inline lazymul(::StaticInt{M}, b::MM{W,X}) where {W,M,X} = LazyMulAdd{M}(MM{W}(getfield(b, :data), StaticInt{M}()*StaticInt{X}()))
# @inline lazymul(a::MM{W,X}, ::StaticInt{M}) where {W,M,X} = LazyMulAdd{M}(MM{W}(getfield(a, :data), StaticInt{M}()*StaticInt{X}()))
@inline lazymul(a::MM{W,X}, ::StaticInt{M}) where {W,M,X} = MM{W}(vmul_fast(StaticInt{M}(), data(a)), StaticInt{X}() * StaticInt{M}())
@inline lazymul(::StaticInt{M}, b::MM{W,X}) where {W,M,X} = MM{W}(vmul_fast(StaticInt{M}(), data(b)), StaticInt{X}() * StaticInt{M}())
@inline lazymul(a::MM{W,X}, ::StaticInt{1}) where {W,X} = a
@inline lazymul(::StaticInt{1}, a::MM{W,X}) where {W,X} = a
@inline lazymul(a::MM{W,X}, ::StaticInt{0}) where {W,X} = StaticInt{0}()
@inline lazymul(::StaticInt{0}, a::MM{W,X}) where {W,X} = StaticInt{0}()

@inline lazymul(::StaticInt{M}, ::StaticInt{N}) where {M, N} = StaticInt{M}()*StaticInt{N}()

@inline lazymul(::StaticInt{M}, ::StaticInt{1}) where {M} = StaticInt{M}()
@inline lazymul(::StaticInt,    ::StaticInt{0}) = StaticInt{0}()

@inline lazymul(::StaticInt{1}, ::StaticInt{M}) where {M} = StaticInt{M}()
@inline lazymul(::StaticInt{0}, ::StaticInt   ) = StaticInt{0}()

@inline lazymul(::StaticInt{0}, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{0}, ::StaticInt{1}) = StaticInt{0}()
@inline lazymul(::StaticInt{1}, ::StaticInt{0}) = StaticInt{0}()
@inline lazymul(::StaticInt{1}, ::StaticInt{1}) = StaticInt{1}()

@inline function lazymul(a::LazyMulAdd{M,O}, ::StaticInt{N}) where {M,O,N}
    LazyMulAdd(getfield(a, :data), StaticInt{M}() * StaticInt{N}(), StaticInt{O}() * StaticInt{N}())
end
@inline function lazymul(::StaticInt{M}, b::LazyMulAdd{N,O}) where {M,O,N}
    LazyMulAdd(getfield(b, :data), StaticInt{M}() * StaticInt{N}(), StaticInt{O}() * StaticInt{M}())
end

@inline function lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{N}) where {M,N,W,X}
    LazyMulAdd(MM{W}(getfield(a, :data), StaticInt{M}()*StaticInt{N}()*StaticInt{X}()), StaticInt{M}()*StaticInt{N}())
end
@inline function lazymul(::StaticInt{M}, b::LazyMulAdd{N,<:MM{W,X}}) where {M,N,W,X}
    LazyMulAdd(MM{W}(getfield(b, :data), StaticInt{M}()*StaticInt{N}()*StaticInt{X}()), StaticInt{M}()*StaticInt{N}())
end
@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{0}) where {M,W,X} = StaticInt{0}()
@inline lazymul(::StaticInt{0}, b::LazyMulAdd{N,<:MM{W,X}}) where {N,W,X} = StaticInt{0}()
@inline lazymul(a::LazyMulAdd{M,<:MM{W,X}}, ::StaticInt{1}) where {M,W,X} = a
@inline lazymul(::StaticInt{1}, b::LazyMulAdd{N,<:MM{W,X}}) where {N,W,X} = b
@inline function lazymul(a::LazyMulAdd{M}, b::LazyMulAdd{N}) where {M,N}
    LazyMulAdd(vmul_fast(getfield(a, :data), getfield(b, :data)), StaticInt{M}()*StaticInt{N}())
end


@inline function Base.:(>>>)(a::LazyMulAdd{M,O}, ::StaticInt{N}) where {M,O,N}
    LazyMulAdd(getfield(a, :data), StaticInt{M}() >>> StaticInt{N}(), StaticInt{O}() >>> StaticInt{N}())
end

# The approach with `add_indices` is that we try and make `vadd_fast` behave well
# but for `i` and `j` type combinations where it's difficult,
# we can add specific `add_indices` methods that increment the pointer.
@inline function add_indices(p::Ptr, i, j) # generic fallback
    p, vadd_fast(i, j)
end
@inline vadd_fast(i::LazyMulAdd, ::Zero) = i
@inline vadd_fast(::Zero, i::LazyMulAdd) = i

# These following two definitions normally shouldn't be hit
@inline function vadd_fast(a::LazyMulAdd{M,O,MM{W,X,I}}, b::Integer) where {M,O,W,X,I}
    MM{W}(vadd_fast(vmul_fast(StaticInt{M}(), data(a)), b), StaticInt{X}() * StaticInt{M}())
end
@inline function vadd_fast(b::Integer, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I}
    MM{W}(vadd_fast(vmul_fast(StaticInt{M}(), data(a)), b), StaticInt{X}() * StaticInt{M}())
end
@inline vadd_fast(a::LazyMulAdd{M,O,MM{W,X,I}}, ::Zero) where {M,O,W,X,I} = a
@inline vadd_fast(::Zero, a::LazyMulAdd{M,O,MM{W,X,I}}) where {M,O,W,X,I} = a

# because we should hit this instead:
@inline add_indices(p::Ptr, b::Integer, a::LazyMulAdd{M,O}) where {M,O} = (p + b, a)
@inline add_indices(p::Ptr, a::LazyMulAdd{M,O}, b::Integer) where {M,O} = (p + b, a)
# but in the case of `VecUnroll`s, which skip the `add_indices`, it's useful to still have the former two definitions.
# However, this also forces us to write:
@inline add_indices(p::Ptr, ::StaticInt{N}, a::LazyMulAdd{M,O}) where {M,O,N} = (p, vadd_fast(a, StaticInt{N}()))
@inline add_indices(p::Ptr, a::LazyMulAdd{M,O}, ::StaticInt{N}) where {M,O,N} = (p, vadd_fast(a, StaticInt{N}()))

@inline function vadd_fast(::StaticInt{N}, a::LazyMulAdd{M,O}) where {N,M,O}
    LazyMulAdd(getfield(a, :data), StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
end
@inline function vadd_fast(a::LazyMulAdd{M,O}, ::StaticInt{N}) where {N,M,O}
    LazyMulAdd(getfield(a, :data), StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
end
@inline function vadd_fast(::StaticInt{N}, a::LazyMulAdd{M,O,MM{W,X,I}}) where {N,M,O,W,X,I}
    LazyMulAdd(getfield(a, :data), StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
end
@inline function vadd_fast(a::LazyMulAdd{M,O,MM{W,X,I}}, ::StaticInt{N}) where {N,M,O,W,X,I}
    LazyMulAdd(getfield(a, :data), StaticInt{M}(), StaticInt{O}()+StaticInt{N}())
end

@inline function vadd_fast(a::LazyMulAdd{M,O}, b::LazyMulAdd{M,A}) where {M,O,A}
    LazyMulAdd(vadd_fast(getfield(a, :data), getfield(b, :data)), StaticInt{M}(), StaticInt{O}()+StaticInt{A}())
end

@inline add_indices(p::Ptr, a::LazyMulAdd{M,O,V}, b::LazyMulAdd{N,P,J}) where {M,O,V<:AbstractSIMDVector,N,P,J<:IntegerTypes} = (gep(p, b), a)
@inline add_indices(p::Ptr, b::LazyMulAdd{N,P,J}, a::LazyMulAdd{M,O,V}) where {M,O,V<:AbstractSIMDVector,N,P,J<:IntegerTypes} = (gep(p, b), a)
@inline add_indices(p::Ptr, a::LazyMulAdd{M,O,V}, b::LazyMulAdd{M,P,J}) where {M,O,V<:AbstractSIMDVector,P,J<:IntegerTypes} = (p, vadd_fast(a, b))
@inline add_indices(p::Ptr, b::LazyMulAdd{M,P,J}, a::LazyMulAdd{M,O,V}) where {M,O,V<:AbstractSIMDVector,P,J<:IntegerTypes} = (p, vadd_fast(a, b))


@inline add_indices(p::Ptr, a::AbstractSIMDVector, b::LazyMulAdd{M,O,I}) where {M,O,I<:IntegerTypes} = (gep(p, b), a)
@inline add_indices(p::Ptr, b::LazyMulAdd{M,O,I}, a::AbstractSIMDVector) where {M,O,I<:IntegerTypes} = (gep(p, b), a)
@inline function add_indices(p::Ptr, ::MM{W,X,StaticInt{A}}, a::LazyMulAdd{M,O,T}) where {M,O,T<:IntegerTypes,A,W,X}
    gep(p, a), MM{W,X}(StaticInt{A}())
end
@inline function add_indices(p::Ptr, a::LazyMulAdd{M,O,T}, ::MM{W,X,StaticInt{A}}) where {M,O,T<:IntegerTypes,A,W,X}
    gep(p, a), MM{W,X}(StaticInt{A}())
end

@generated function add_indices(p::Ptr, a::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, b::LazyMulAdd{N,P,J}) where {M,O,W,X,I,N,P,J<:IntegerTypes}
    d, r = divrem(M, N)
    if iszero(r)
        quote
            $(Expr(:meta,:inline))
            p, VectorizationBase.LazyMulAdd{$N,$(I*M)}(MM{$W,$d}(getfield(b, :data)))
        end
    else
        quote
            $(Expr(:meta,:inline))
            gep(p, b), a
        end
    end
end
@inline add_indices(p::Ptr, b::LazyMulAdd{N,P,J}, a::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}) where {M,O,W,X,I,N,P,J<:IntegerTypes} = add_indices(p, a, b)
@generated function vadd_fast(a::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, b::LazyMulAdd{N,P,J}) where {M,O,W,X,I,N,P,J<:IntegerTypes}
    d, r = divrem(M, N)
    if iszero(r)
        quote
            $(Expr(:meta,:inline))
            VectorizationBase.LazyMulAdd{$N,$(I*M)}(MM{$W,$d}(getfield(b, :data)))
        end
    else
        quote
            $(Expr(:meta,:inline))
            vadd_fast(a, _materialize(b))
        end
    end
end
@inline vadd_fast(b::LazyMulAdd{N,P,J}, a::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}) where {M,O,W,X,I,N,P,J<:IntegerTypes} = vadd_fast(a, b)

@inline vadd_fast(a::VecUnroll, b::LazyMulAdd) = VecUnroll(fmap(vadd_fast, getfield(a, :data), b))
@inline vadd_fast(b::LazyMulAdd, a::VecUnroll) = VecUnroll(fmap(vadd_fast, b, getfield(a, :data)))

