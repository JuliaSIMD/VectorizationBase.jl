
register_size(::Type{T}) where {T} = REGISTER_SIZE
register_size(::Type{T}) where {T<:Union{Signed,Unsigned}} = SIMD_NATIVE_INTEGERS ? REGISTER_SIZE : sizeof(T)

intlog2(N::I) where {I <: Integer} = (8sizeof(I) - one(I) - leading_zeros(N)) % I
intlog2(::Type{T}) where {T} = intlog2(sizeof(T))
ispow2(x::Integer) = (x & (x - 1)) == zero(x)
nextpow2(W) = vshl(one(W), vsub(8sizeof(W), leading_zeros(vsub(W, one(W)))))
prevpow2(W) = vshl(one(W), vsub(vsub((8sizeof(W)) % UInt, one(UInt)), leading_zeros(W) % UInt))
prevpow2(W::Signed) = prevpow2(W % Unsigned) % Signed

function pick_vector_width(::Type{T} = Float64) where {T<:NativeTypes}
    max(1, register_size(T) >>> intlog2(T))
end
function pick_vector_width_shift(::Type{T} = Float64) where {T<:NativeTypes}
    W = pick_vector_width(T)
    Wshift = intlog2(register_size(T)) - intlog2(T)
    W, Wshift
end

# For the sake of convenient mask support, we allow 8 so that the mask can be a full byte
# max_vector_width(::Type{T}) where {T} = max(8, pick_vector_width(T))


pick_vector_width(::Type{T1}, ::Type{T2}) where {T1,T2} = min(pick_vector_width(T1), pick_vector_width(T2))
@inline pick_vector_width(::Type{T1}, ::Type{T2}, ::Type{T3}, args::Vararg{Any,K}) where {T1,T2,T3,K} = min(pick_vector_width(T1), pick_vector_width(T2, T3, args...))

@inline function pick_vector_width(N::Integer, args::Vararg{Any,K}) where {K}
    min(nextpow2(N), pick_vector_width(args...))
end
@inline function pick_vector_width_shift(N::Integer, args::Vararg{Any,K}) where {K}
    W = pick_vector_width(N, args...)
    W, intlog2(W)
end

pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)
pick_vector_width_val(::Type{T} = Float64) where {T} = Val{pick_vector_width(T)}()
pick_vector_width_val(::Val{N}, ::Type{T} = Float64) where {N,T} = Val{pick_vector_width(Val(N), T)}()


# @inline vadd(::Static{i}, j) where {i} = vadd(i, j)
# @inline vadd(i, ::Static{j}) where {j} = vadd(i, j)
# @inline vsub(::Static{i}, j) where {i} = vsub(i, j)
# @inline vsub(i, ::Static{j}) where {j} = vsub(i, j)

@inline staticmul(::Val{W}, i) where {W} = vmul(Static(W), i)
@inline staticmuladd(::Val{W}, b, c) where {W} = vadd(vmul(Static(W), b), c)
@inline valmul(::Val{W}, i) where {W} = vmul(W, i)
@inline valadd(::Val{W}, i) where {W} = vadd(W, i)
@inline valsub(::Val{W}, i) where {W} = vsub(W, i)
@inline valsub(i, ::Val{W}) where {W} = vsub(i, W)
@inline valrem(::Val{W}, i) where {W} = i & vsub(W, 1)
@inline valmuladd(::Val{W}, b, c) where {W} = vadd(vmul(W, b), c)
@inline valmulsub(::Val{W}, b, c) where {W} = vsub(vmul(W, b), c)
@inline valmul(::Val{W}, i::T) where {W,T<:Integer} = vmul((W % T), i)
@inline valadd(::Val{W}, i::T) where {W,T<:Integer} = vadd((W % T), i)
@inline valsub(::Val{W}, i::T) where {W,T<:Integer} = vsub((W % T), i)
@inline valrem(::Val{W}, i::T) where {W,T<:Integer} = i & vsub((W % T), one(T))
@inline valmuladd(::Val{W}, b::T, c::T) where {W,T<:Integer} = vadd(vmul(W % T, b), c)
@inline valmulsub(::Val{W}, b::T, c::T) where {W,T<:Integer} = vsub(vmul(W % T, b), c)

@generated pick_vector(::Type{T}) where {T} = Vec{pick_vector_width(T),T}
pick_vector(N, T) = Vec{pick_vector_width(N, T),T}
@generated pick_vector(::Val{N}, ::Type{T}) where {N, T} =  pick_vector(N, T)

@inline MM(::Val{W}) where {W} = MM{W}(0)
@inline MM(::Val{W}, i) where {W} = MM{W}(i)
# @inline MM{W}(a::LazyMul) where {W} = MM{W}(extract_data(a))
@inline gep(ptr::Ptr, i::MM) = gep(ptr, i.i)

@inline staticm1(i::MM{W,I}) where {W,I} = MM{W}(vsub(i.i, one(I)))
@inline staticp1(i::MM{W,I}) where {W,I} = MM{W}(vadd(i.i, one(I)))
@inline vadd(i::MM{W}, j::Integer) where {W} = MM{W}(vadd(i.i, j))
@inline vadd(i::Integer, j::MM{W}) where {W} = MM{W}(vadd(i, j.i))
@inline vadd(i::MM{W}, ::Static{j}) where {W,j} = MM{W}(vadd(i.i, j))
@inline vadd(::Static{i}, j::MM{W}) where {W,i} = MM{W}(vadd(i, j.i))
@inline vadd(i::MM{W,X}, j::MM{W,S}) where {W,X,S} = MM{W}(vadd(i.i, j.i), Static{X}() + Static{S}())
@inline vsub(i::MM{W}, j::Integer) where {W} = MM{W}(vsub(i.i, j))
@inline vsub(i::MM{W}, ::Static{j}) where {W,j} = MM{W}(vsub(i.i, j))
# @inline vmul_no_promote(i, j) = vmul(i,j)
# @inline vmul_no_promote(i, j::MM{W}) where {W} = MM{W}(vmul(i, j.i))
# @inline vmuladdnp(a, b, c) = vadd(vmul(a,b), c)
# @inline vmuladdnp(a, b::MM{W}, c) where {W} = vadd(MM{W}(vmul(a,b.i)), c)
# @inline vmuladdnp(a, b::MM{W}, c::MM{W}) where {W} = vadd(vmul(a,b), c)
# @inline vmuladdnp(a, b::MM{W}, c::Vec) where {W} = vadd(vmul(a,b), c)
# @inline vmuladdnp(a, b::MM{W}, c::_Vec) where {W} = vadd(vmul(a,b), c)

@inline Base.:(+)(i::MM{W}, j::Integer) where {W} = MM{W}(vadd(i.i, j))
@inline Base.:(+)(i::Integer, j::MM{W}) where {W} = MM{W}(vadd(i, j.i))
@inline Base.:(+)(i::MM{W}, ::Static{j}) where {W,j} = MM{W}(vadd(i.i, j))
@inline Base.:(+)(::Static{i}, j::MM{W}) where {W,i} = MM{W}(vadd(i, j.i))
# @inline Base.:(+)(i::MM{W}, j::MM{W}) where {W} = MM{W}(i.i + j.i)
@inline Base.:(-)(i::MM{W}, j::Integer) where {W} = MM{W}(vsub(i.i, j))
# @inline Base.:(-)(i::Integer, j::MM{W}) where {W} = MM{W}(i - j.i)
@inline Base.:(-)(i::MM{W}, ::Static{j}) where {W,j} = MM{W}(vsub(i.i, j))
# @inline Base.:(-)(::Static{i}, j::MM{W}) where {W,i} = MM{W}(i - j.i)
# @inline Base.:(-)(i::MM{W}, j::MM{W}) where {W} = MM{W}(i.i - j.i)
# @inline Base.:(*)(i::MM{W}, j) where {W} = MM{W}(i.i * j)
# @inline Base.:(*)(i, j::MM{W}) where {W} = MM{W}(i * j.i)
# @inline Base.:(*)(i::MM{W}, ::Static{j}) where {W,j} = MM{W}(i.i * j)
# @inline Base.:(*)(::Static{i}, j::MM{W}) where {W,i} = MM{W}(i * j.i)
# @inline Base.:(*)(i::MM{W}, j::MM{W}) where {W} = MM{W}(i.i * j.i)
@inline scalar_less(i, j) = i < j
@inline scalar_less(i::MM, j::Integer) = i.i < j
@inline scalar_less(i::Integer, j::MM) = i < j.i
@inline scalar_less(i::MM, ::Static{j}) where {j} = i.i < j
@inline scalar_less(::Static{i}, j::MM) where {i} = i < j.i
@inline scalar_less(i::MM, j::MM) = i.i < j.i
@inline scalar_greater(i, j) = i > j
@inline scalar_greater(i::MM, j::Integer) = i.i > j
@inline scalar_greater(i::Integer, j::MM) = i > j.i
@inline scalar_greater(i::MM, ::Static{j}) where {j} = i.i > j
@inline scalar_greater(::Static{i}, j::MM) where {i} = i > j.i
@inline scalar_greater(i::MM, j::MM) = i.i > j.i
@inline scalar_equal(i, j) = i == j
@inline scalar_equal(i::MM, j::Integer) = i.i == j
@inline scalar_equal(i::Integer, j::MM) = i == j.i
@inline scalar_equal(i::MM, ::Static{j}) where {j} = i.i == j
@inline scalar_equal(::Static{i}, j::MM) where {i} = i == j.i
@inline scalar_equal(i::MM, j::MM) = i.i == j.i
@inline scalar_notequal(i, j) = i != j
@inline scalar_notequal(i::MM, j::Integer) = i.i != j
@inline scalar_notequal(i::Integer, j::MM) = i != j.i
@inline scalar_notequal(i::MM, ::Static{j}) where {j} = i.i != j
@inline scalar_notequal(::Static{i}, j::MM) where {i} = i != j.i
@inline scalar_notequal(i::MM, j::MM) = i.i != j.i

@inline extract_data(i::MM) = i.i




# @inline _vload(ptr::Ptr{T}, i::Integer) where {T} = vload(ptr + vmul(sizeof(T), i))
# @inline _vload(ptr::Ptr, v::Vec{<:Any,<:Integer}) = vload(ptr, v.data)
# @inline _vload(ptr::Ptr, v::Vec{<:Any,<:Integer}) = vload(ptr, v)
# @inline _vload(ptr::Ptr{T}, i::MM{W}) where {W,T} = vload(Val{W}(), ptr + vmul(sizeof(T), i.i))
# @inline _vload(ptr::AbstractPointer, i) = _vload(ptr.ptr, offset(ptr, i))
# @inline vload(ptr::AbstractPointer{T}, i::Tuple) where {T} = _vload(ptr.ptr, offset(ptr, i))


