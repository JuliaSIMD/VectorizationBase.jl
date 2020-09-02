


@inline tdot(a::Tuple{A}, b::Tuple{B}, ::Tuple{Val{false}}) where {A,B} = lazymul(first(a), first(b))
@inline tdot(a::Tuple{A}, b::Tuple{B}, ::Tuple{Val{true}}) where {A,B} = lazymul_no_promote(first(a), first(b))

@inline tdot(a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{Val{false},Vararg}) where {A1,A2,B1,B2} = lazyadd(lazymul(first(a), first(b)), tdot(Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdot(a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{Val{true},Vararg}) where {A1,A2,B1,B2} = lazyadd(lazymul_no_promote(first(a), first(b)), tdot(Base.tail(a), Base.tail(b), Base.tail(c)))


@inline tdot(a::Tuple{A}, b::Tuple{B}, c::Tuple{C}, ::Tuple{Val{false}}) where {A,B,C} = lazymul(first(a), first(b))
@inline tdot(a::Tuple{A}, b::Tuple{B}, c::Tuple{C}, ::Tuple{Val{true}}) where {A,B,C} = lazymul_no_promote(first(a), first(b))

@inline function tdot(a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{Val{false},Vararg}) where {A1,A2,B1,B2,C1,C2}
    lazyadd(lazymul(first(a), first(b), first(c)), tdot(Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end
@inline function tdot(a::Tuple{A1,A2,Vararg}, b::Tuple{B1,B2,Vararg}, c::Tuple{C1,C2,Vararg}, d::Tuple{Val{true},Vararg}) where {A1,A2,B1,B2,C1,C2}
    lazyadd(lazymul_no_promote(first(a), first(b), first(c)), tdot(Base.tail(a), Base.tail(b), Base.tail(c), Base.tail(d)))
end


