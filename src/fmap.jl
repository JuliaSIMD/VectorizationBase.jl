
# @inline tuplefirst(x) = x
# @inline tuplefirst(x::Tuple) = first(x)
# @inline tuplesecond(x) = x
# @inline tuplesecond(x::Tuple) = x[2]
# @inline tuplethird(x) = x
# @inline tuplethird(x::Tuple) = x[3]
# @inline tuplefourth(x) = x
# @inline tuplefourth(x::Tuple) = x[4]
# @inline tupletail(x) = x
# @inline tupletail(x::Tuple) = Base.tail(x)

# struct TupleLength{N} end
# @inline tuple_len(::Any) = nothing
# @inline tuple_len(::Tuple{Vararg{Any,N}}) where {N} = TupleLength{N}()
# @inline tuple_len(::TupleLength{N}, args...) where {N} = TupleLength{N}()
# @inline tuple_len(::Nothing, a, args...) = tuple_len(tuple_len(a), args...)
# @inline tuple_len(a, args...) = tuple_len(tuple_len(a), args...)

# unary # 2^2 - 2 = 2 definitions
@inline fmap(f::F, x::Tuple{X}) where {F,X} = (f(first(x)),)
@inline fmap(f::F, x::NTuple) where {F} = (f(first(x)), fmap(f, Base.tail(x))...)

# binary # 2^3 - 2 = 6 definitions
@inline fmap(f::F, x::Tuple{X}, y::Tuple{Y}) where {F,X,Y} = (f(first(x), first(y)),)
@inline fmap(f::F, x::Tuple{X}, y) where {F,X} = (f(first(x), y),)
@inline fmap(f::F, x, y::Tuple{Y}) where {F,Y} = (f(x, first(y)),)
@inline fmap(f::F, x::Tuple{Vararg{Any,N}}, y::Tuple{Vararg{Any,N}}) where {F,N} = (f(first(x), first(y)), fmap(f, Base.tail(x), Base.tail(y))...)
@inline fmap(f::F, x::Tuple, y) where {F} = (f(first(x), y), fmap(f, Base.tail(x), y)...)
@inline fmap(f::F, x, y::Tuple) where {F} = (f(x, first(y)), fmap(f, x, Base.tail(y))...)


fmap(f::F, x::Tuple{X}, y::Tuple) where {F,X} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple{Y}) where {F,Y} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple) where {F} = throw("Dimension mismatch.")

# ternary # 2^4 - 2 = 14 definitions, or 3
# @inline fmap(f::F, x, y, z) where {F} = fmap(f, tuple_len(x, y, z), x, y, z)
# @inline fmap(f::F, ::TupleLength{1}, x, y, z) where {F} = (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)),)
# @inline fmap(f::F, ::TupleLength{2}, x, y, z) where {F} = (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)), f(tuplesecond(x), tuplesecond(y), tuplesecond(z)))
# @inline function fmap(f::F, ::TupleLength{3}, x, y, z) where {F}
#     (
#         f(tuplefirst(x), tuplefirst(y), tuplefirst(z)),
#         f(tuplesecond(x), tuplesecond(y), tuplesecond(z)),
#         f(tuplethird(x), tuplethird(y), tuplethird(z))
#     )
# end
# @inline function fmap(f::F, ::TupleLength{4}, x, y, z) where {F}
#     (
#         f(tuplefirst(x), tuplefirst(y), tuplefirst(z)),
#         f(tuplesecond(x), tuplesecond(y), tuplesecond(z)),
#         f(tuplethird(x), tuplethird(y), tuplethird(z)),
#         f(tuplefourth(x), tuplefourth(y), tuplefourth(z))
#     )
# end
@generated function fmap(f::F, x::Vararg{Any,N}) where {F,N}
    q = Expr(:block, Expr(:meta, :inline))
    t = Expr(:tuple)
    U = 1
    call = Expr(:call, :f)
    syms = Vector{Symbol}(undef, N)
    istup = Vector{Bool}(undef, N)
    for n ∈ 1:N
        syms[n] = xₙ = Symbol(:x_, n)
        push!(q.args, Expr(:(=), xₙ, Expr(:ref, :x, n)))
        istup[n] = ist = (x[n] <: Tuple)
        if ist
            U = length(x[n].parameters)
            push!(call.args, Expr(:ref, xₙ, 1))
        else
            push!(call.args, xₙ)
        end
    end
    push!(t.args, call)
    for u ∈ 2:U
        call = Expr(:call, :f)
        for n ∈ 1:N
            xₙ = syms[n]
            if istup[n]
                push!(call.args, Expr(:ref, xₙ, u))
            else
                push!(call.args, xₙ)
            end
        end
        push!(t.args, call)
    end
    push!(q.args, t); q
end
# @inline function fmap(f::F, ::TupleLength, x, y, z) where {F}
#     (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)), fmap(f, tupletail(x), tupletail(y), tupletail(z))...)
# end
# @inline fmap(f::F, ::TupleLength, x, y, z) where {F} = (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)), fmap(f, tupletail(x), tupletail(y), tupletail(z))...)

# quaternary # 2^5 - 2 = 30 definitions, or 3
# @inline fmap(f::F, w, x, y, z) where {F} = fmap(f, tuple_len(w, x, y, z), w, x, y, z)
# @inline fmap(f::F, ::TupleLength{1}, w, x, y, z) where {F} = (f(tuplefirst(w), tuplefirst(x), tuplefirst(y), tuplefirst(z)),)
# @inline fmap(f::F, ::TupleLength, w, x, y, z) where {F} = (f(tuplefirst(w), tuplefirst(x), tuplefirst(y), tuplefirst(z)), fmap(f, tupletail(w), tupletail(x), tupletail(y), tupletail(z))...)

# @inline fmap(f::F, w) where {F} = VecUnroll(fmapt(f, unrolleddata(w)))
# @inline fmap(f::F, w, x) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x)))
# @inline fmap(f::F, w, x, y) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x), unrolleddata(y)))
# @inline fmap(f::F, w, x, y, z) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x), unrolleddata(y), unrolleddata(z)))

for op ∈ [:(-), :abs, :floor, :ceil, :trunc, :round, :sqrt, :!, :(~), :leading_zeros, :trailing_zeros]
    @eval @inline Base.$op(v1::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, v1.data))
end
@inline Base.inv(v::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(vdiv, vbroadcast(Val{W}(), one(T)), v.data))
@inline Base.reinterpret(::Type{T}, v::VecUnroll) where {T<:Number} = VecUnroll(fmap(reinterpret, T, v.data))
for op ∈ [:(Base.:(+)),:(Base.:(-)),:(Base.:(*)),:(Base.:(/)),:(Base.:(%)),:(Base.:(<<)),:(Base.:(>>)),:(Base.:(>>>)),:(Base.:(&)),:(Base.:(|)),:(Base.:(⊻)),
          :(Base.:(÷)),:(Base.max),:(Base.min),:(Base.copysign),:(Base.:(<)),:(Base.:(≤)),:(Base.:(>)),:(Base.:(≥)),:(Base.:(==)),:(Base.:(≠)),:vadd,:vsub,:vmul]
    @eval begin
        # @inline $op(v1::VecUnroll{N,W,T}, v2::Real) where {N,W,T} = VecUnroll(fmap($op, v1.data, vbroadcast(Val{W}(), T, v2)))
        # @inline $op(v1::Real, v2::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, vbroadcast(Val{W}(), T, v1), v2.data))
        @inline $op(v1::VecUnroll{N,W,T}, v2::S) where {N,W,T,S<:Real} = VecUnroll(fmap($op, v1.data, vbroadcast(Val{W}(), v2)))
        @inline $op(v1::S, v2::VecUnroll{N,W,T}) where {N,W,T,S<:Real} = VecUnroll(fmap($op, vbroadcast(Val{W}(), v1), v2.data))
        # @inline $op(v1::VecUnroll{N,W,T}, v2::AbstractSIMD{W,S}) where {N,W,T,S<:Real} = VecUnroll(fmap($op, v1.data, promote_type(Vec{W,T}, Vec{W,S})(v2)))
        # @inline $op(v1::AbstractSIMD{W,S}, v2::VecUnroll{N,W,T}) where {N,W,T,S<:Real} = VecUnroll(fmap($op, promote_type(Vec{W,T}, Vec{W,S})(v1), v2.data))
        # @inline $op(v1::Real, v2::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, Vec{W,T}(v1), v2.data))
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
        @inline $op(v1::VecUnroll{N,W,T}, ::StaticInt{M}) where {N,W,T,M} = VecUnroll(fmap($op, v1.data, vbroadcast(Val{W}(), T(M))))
        @inline $op(::StaticInt{M}, v1::VecUnroll{N,W,T}) where {N,W,T,M} = VecUnroll(fmap($op, vbroadcast(Val{W}(), T(M)), v1.data))
    end
end
for op ∈ [:%, :&, :|, :⊻, :>>, :>>>, :<<,:(<),:(≤),:(>),:(≥),:(==),:(≠)]
    @eval begin
        @inline Base.$op(vu::VecUnroll, i::MM) = $op(vu, Vec(i))
        @inline Base.$op(i::MM, vu::VecUnroll) = $op(Vec(i), vu)
    end
end
for op ∈ [:>>, :>>>, :<<]
    @eval @inline Base.$op(m::Mask, vu::VecUnroll) = $op(Vec(m), vu)
end
# @inline Base.:(^)(v1::VecUnroll{N,W,T}, v2::Rational) where {N,W,T} = VecUnroll(fmap(^, v1.data, Vec{W,T}(v2)))
@inline Base.copysign(v1::Rational, v2::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(copysign, Vec{W,T}(v1), v2.data))
@inline Base.copysign(v1::Signed, v2::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(copysign, Vec{W,T}(v1), v2.data))
for op ∈ [:rotate_left,:rotate_right,:funnel_shift_left,:funnel_shift_right]
    @eval begin
        @inline $op(v1::VecUnroll{N,W,T}, v2::Real) where {N,W,T} = VecUnroll(fmap($op, v1.data, Vec{W,T}(v2)))
        @inline $op(v1::Real, v2::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, Vec{W,T}(v1), v2.data))
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
    end
end

for op ∈ [:(Base.muladd), :(Base.fma), :vfmadd, :vfnmadd, :vfmsub, :vfnmsub, :vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231]
    @eval begin
        @inline $op(v1::VecUnroll{N,W,T}, v2::Real, v3::Real) where {N,W,T} = VecUnroll(fmap($op, v1.data, Vec{W,T}(v2), Vec{W,T}(v3)))
        @inline $op(v1::Real, v2::VecUnroll{N,W,T}, v3::Real) where {N,W,T} = VecUnroll(fmap($op, Vec{W,T}(v1), v2.data, Vec{W,T}(v3)))
        @inline $op(v1::Real, v2::Real, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, Vec{W,T}(v1), Vec{W,T}(v2), v3.data))
        @inline $op(v1::VecUnroll{N,W,T}, v2::VecUnroll{N,W,T}, v3::Real) where {N,W,T} = VecUnroll(fmap($op, v1.data, v2.data, Vec{W,T}(v3)))
        @inline $op(v1::VecUnroll{N,W,T}, v2::Real, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, v1.data, Vec{W,T}(v2), v3.data))
        @inline $op(v1::Real, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, Vec{W,T}(v1), v2.data, v3.data))
        $op(v1::VecUnroll, v2::VecUnroll, v3::Real) = throw("Size mismatch")
        $op(v1::VecUnroll, v2::Real, v3::VecUnroll) = throw("Size mismatch")
        $op(v1::Real, v2::VecUnroll, v3::VecUnroll) = throw("Size mismatch")
        # @inline $op(v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data, v3.data))
        $op(v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) = throw("Size mismatch")
        @inline $op(v1::VecUnroll{N,W,T}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, v1.data, v2.data, v3.data))
    end
end
@inline ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:NativeTypes} = VecUnroll(fmap(ifelse, v1.data, Vec{W,T}(v2), Vec{W,T}(v3)))
@inline ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:Real} = VecUnroll(fmap(ifelse, v1.data, v2, v3))
@inline ifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,T} = VecUnroll(fmap(ifelse, Vec{W,T}(v1), v2.data, Vec{W,T}(v3)))
@inline ifelse(v1::Vec{W,Bool}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(ifelse, Vec{W,T}(v1), Vec{W,T}(v2), v3.data))
@inline ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,T} = VecUnroll(fmap(ifelse, v1.data, v2.data, Vec{W,T}(v3)))
@inline ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(ifelse, v1.data, Vec{W,T}(v2), v3.data))
@inline ifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(ifelse, Vec{W,T}(v1), v2.data, v3.data))
# ifelse(v1::VecUnroll, v2::VecUnroll, v3::Real) = throw("Size mismatch")
# ifelse(v1::VecUnroll, v2::Real, v3::VecUnroll) = throw("Size mismatch")
# ifelse(v1::Real, v2::VecUnroll, v3::VecUnroll) = throw("Size mismatch")
# ifelse(v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) = throw("Size mismatch")
@inline ifelse(v1::VecUnroll{N,W,<:Boolean}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(ifelse, v1.data, v2.data, v3.data))

@inline Base.:(^)(v::VecUnroll, i::Integer) = VecUnroll(fmap(^, v.data, i))

@inline Base.:(==)(v::VecUnroll{N,W,T}, x::AbstractIrrational) where {N,W,T} = v == vbroadcast(Val{W}(), T(x))
@inline Base.:(==)(x::AbstractIrrational, v::VecUnroll{N,W,T}) where {N,W,T} = vbroadcast(Val{W}(), T(x)) == v

@inline Base.convert(::Type{T}, v::VecUnroll) where {T<:Real} = VecUnroll(fmap(convert, T, v.data))
@inline Base.convert(::Type{VU}, v::VU) where {VU <: VecUnroll} = v
@inline Base.unsafe_trunc(::Type{T}, v::VecUnroll) where {T<:Real} = VecUnroll(fmap(unsafe_trunc, T, v.data))
@inline Base.:(%)(v::VecUnroll, ::Type{T}) where {T<:Real} = VecUnroll(fmap(%, v.data, T))

@inline (::Type{VecUnroll{N,W,T,V}})(vu::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = vu
@inline (::Type{VecUnroll{N,W,T,VT}})(vu::VecUnroll{N,W,S,VS})  where {N,W,T,VT<:AbstractSIMDVector{W,T},S,VS<:AbstractSIMDVector{W,S}} = VecUnroll(fmap(convert, Vec{W,T}, vu.data))


function collapse_expr(N, op)
    N += 1
    t = Expr(:tuple); s = Vector{Symbol}(undef, N)
    for n ∈ 1:N
        s_n = s[n] = Symbol(:v_, n)
        push!(t.args, s_n)
    end
    q = quote
        $(Expr(:meta,:inline))
        $t = data(vu)
    end
    while N > 1
        for n ∈ 1:N >>> 1
            push!(q.args, Expr(:(=), s[n], Expr(:call, op, s[n], s[n + (N >>> 1)])))
        end
        isodd(N) && push!(q.args, Expr(:(=), s[1], Expr(:call, op, s[1], s[N])))
        N >>>= 1
    end
    q
end
@generated collapse_add(vu::VecUnroll{N}) where {N} = collapse_expr(N, :vadd)
@generated collapse_mul(vu::VecUnroll{N}) where {N} = collapse_expr(N, :vmul)
@generated collapse_max(vu::VecUnroll{N}) where {N} = collapse_expr(N, :max)
@generated collapse_min(vu::VecUnroll{N}) where {N} = collapse_expr(N, :min)
@inline vsum(vu::VecUnroll) = vsum(collapse_add(vu))
@inline vprod(vu::VecUnroll) = vprod(collapse_mul(vu))
@inline vmaximum(vu::VecUnroll) = vmaximum(collapse_max(vu))
@inline vminimum(vu::VecUnroll) = vminimum(collapse_min(vu))

