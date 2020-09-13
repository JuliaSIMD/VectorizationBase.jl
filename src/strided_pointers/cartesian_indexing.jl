


@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B}, ::Tuple{Val{false}}) where {T,A,B} = lazymul(first(a), first(b))
@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B}, ::Tuple{Val{true}}) where {T,A,B} = lazymul_no_promote(T, first(a), first(b))

@inline tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{Val{false},Vararg}) where {T,A1,A2,B1,B2} = vadd(lazymul(first(a), first(b)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{Val{true},Vararg}) where {T,A1,A2,B1,B2} = vadd(lazymul_no_promote(T,first(a), first(b)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c)))


@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B}, c::Tuple{C}, ::Tuple{Val{false}}) where {T,A,B,C} = lazymul(first(a), first(b))
@inline tdot(::Type{T}, a::Tuple{A}, b::Tuple{B}, c::Tuple{C}, ::Tuple{Val{true}}) where {T,A,B,C} = lazymul_no_promote(T,first(a), first(b))

@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{Val{false},Vararg}) where {T,A1,A2,B1,B2,C1,C2}
    vadd(lazymul(first(a), first(b), first(c)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end
@inline function tdot(::Type{T}, a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{Val{true},Vararg}) where {T,A1,A2,B1,B2,C1,C2}
    vadd(lazymul_no_promote(T,first(a), first(b), first(c)), tdot(T,Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end


