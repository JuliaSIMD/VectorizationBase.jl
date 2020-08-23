

@inline tuplefirst(x) = x
@inline tuplefirst(x::Tuple) = first(x)
@inline tupletail(x) = x
@inline tupletail(x::Tuple) = Base.tail(x)

@inline tuple_len(::Any) = nothing
@inline tuple_len(::Tuple{Vararg{Any,N}}) where {N} = N
@inline tuple_len(a, b) = (la = tuple_len(a); isnothing(la) ? tuple_len(b) : la)
@inline tuple_len(a, b, c) = (la = tuple_len(a, b); isnothing(la) ? tuple_len(c) : la)
@inline tuple_len(a, b, c, d) = (la = tuple_len(a, b, c); isnothing(la) ? tuple_len(d) : la)

# unary # 2^2 - 2 = 2 definitions
@inline fmap(f::F, x::Tuple{X}) where {F,X} = (f(first(x)),)
@inline fmap(f::F, x::NTuple) where {F} = (f(first(x)), fmap(f, Base.tail(x))...)

# binary # 2^3 - 2 = 6 definitions
@inline fmap(f::F, x::Tuple{X}, y::Tuple{Y}) where {F,X,Y} = (f(first(x), first(y)),)
@inline fmap(f::F, x::Tuple{X}, y) where {F,X} = (f(first(x), y),)
@inline fmap(f::F, x, y::Tuple{Y}) where {F,Y} = (f(x, first(y)),)
@inline fmap(f::F, x::NTuple, y::NTuple) where {F} = (f(first(x), first(y)), fmap(f, Base.tail(x), Base.tail(y))...)
@inline fmap(f::F, x::NTuple, y) where {F} = (f(first(x), y), fmap(f, Base.tail(x), y)...)
@inline fmap(f::F, x, y::NTuple) where {F} = (f(x, first(y)), fmap(f, x, Base.tail(y))...)

# ternary # 2^4 - 2 = 14 definitions, or 1
@inline function fmap(f::F, x, y, z) where {F}
    if isone(tuple_len(x, y, z))
        (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)),)
    else
        (f(tuplefirst(x), tuplefirst(y), tuplefirst(z)), fmap(f, tupletail(x), tupletail(y), tupletail(z))...)
    end
end

# quaternary # 2^5 - 2 = 30 definitions, or 1
@inline function fmap(f::F, w, x, y, z) where {F}
    if isone(tuple_len(x, y, z))
        (f(tuplefirst(w), tuplefirst(x), tuplefirst(y), tuplefirst(z)),)
    else
        (f(tuplefirst(w), tuplefirst(x), tuplefirst(y), tuplefirst(z)), fmap(f, tupletail(w), tupletail(x), tupletail(y), tupletail(z))...)
    end
end

# @inline fmap(f::F, w) where {F} = VecUnroll(fmapt(f, unrolleddata(w)))
# @inline fmap(f::F, w, x) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x)))
# @inline fmap(f::F, w, x, y) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x), unrolleddata(y)))
# @inline fmap(f::F, w, x, y, z) where {F} = VecUnroll(fmapt(f, unrolleddata(w), unrolleddata(x), unrolleddata(y), unrolleddata(z)))


