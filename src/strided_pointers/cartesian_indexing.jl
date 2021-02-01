
# Overloadable method, e.g to insert OffsetPrecalc's precalculated stride multiples
@inline tdot(ptr::AbstractStridedPointer, ::Tuple{}, ::Tuple{}, ::Tuple{}) = Zero()
@inline tdot(ptr::AbstractStridedPointer{T}, a, b, c) where {T} = tdot(T, a, b, c)

@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B,Vararg}, ::Tuple{False,Vararg}) where {T,A,B} = lazymul(first(a), first(b))
@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B,Vararg}, ::Tuple{True,Vararg}) where {T,A,B} = lazymul_no_promote(T, first(a), first(b))

@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{False,Vararg}) where {T,A1,A2,B1,B2}
    vadd_fast(lazymul(first(a), first(b)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c)))
end
@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{True,Vararg}) where {T,A1,A2,B1,B2}
    vadd_fast(lazymul_no_promote(T,first(a), first(b)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c)))
end

@inline function tdot(::Type{T}, a::Tuple{A}, b::Tuple{B,Vararg}, c::Tuple{C,Vararg}, ::Tuple{False,Vararg}) where {T,A,B,C}
    lazymul(first(a), first(b), first(c))
end
# @inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B,Vararg}, c::Tuple{C,Vararg}, ::Tuple{True,Vararg}) where {T,A,B,C} = lazymul_no_promote(T, first(a), first(b))
@inline function tdot(::Type{T}, a::Tuple{A}, b::Tuple{B,Vararg}, c::Tuple{C,Vararg}, ::Tuple{True,Vararg}) where {T,A,B,C}
    lazymul_no_promote(T,first(a), first(b), first(c))
end

@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{False,Vararg}) where {T,A1,A2,B1,B2,C1,C2}
    vadd_fast(lazymul(first(a), first(b), first(c)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end
# @inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{True,Vararg}) where {T,A1,A2,B1,B2,C1,C2}
#     vadd_fast(lazymul_no_promote(T,first(a), first(b)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
# end
@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{True,Vararg}) where {T,A1,A2,B1,B2,C1,C2}
    vadd_fast(lazymul_no_promote(T,first(a), first(b), first(c)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end


