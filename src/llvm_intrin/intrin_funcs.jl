# Dense fmap
@inline fmapt(f::F, x::Tuple{X}, y::Tuple{Y}, z::Tuple{Z}) where {F,X,Y} = (f(first(x), first(y), first(z)),)
@inline fmapt(f::F, x::NTuple, y::NTuple, z::NTuple) where {F} = (f(first(x), first(y), first(z)), fmap(f, Base.tail(x), Base.tail(y), Base.tail(z))...)

@inline fmap(f::F, x::VecUnroll, y::VecUnroll, z::VecUnroll) where {F} = VecUnroll(fmapt(f, x.data, y.data, z.data))
@inline fmap(f::F, x::VecTile, y::VecTile, z::VecTile) where {F} = VecTile(fmapt(f, x.data, y.data, z.data))

# Broadcast fmap
@inline fmapt(f::F, x::Tuple{X}, y, z) where {F,X} = (f(first(x), y, z),)
@inline fmapt(f::F, x, y::Tuple{Y}, z) where {F,Y} = (f(x, first(y), z),)
@inline fmapt(f::F, x, y, z::Tuple{Z}) where {F,Z} = (f(x, y, first(z)),)

@inline fmapt(f::F, x::Tuple{X}, y::Tuple{Y}, z) where {F,X,Y} = (f(first(x), first(y), z),)
@inline fmapt(f::F, x::Tuple{X}, y, z::Tuple{Z}) where {F,X,Z} = (f(first(x), y, first(z)),)
@inline fmapt(f::F, x, y::Tuple{Y}, z::Tuple{Z}) where {F,Y,Z} = (f(x, first(y), first(z)),)

@inline fmapt(f::F, x::NTuple, y, z) where {F} = (f(first(x), y, z), fmap(f, Base.tail(x), y, z)...)
@inline fmapt(f::F, x, y::NTuple, z) where {F} = (f(x, first(y), z), fmap(f, x, Base.tail(y), z)...)
@inline fmapt(f::F, x, y, z::NTuple) where {F} = (f(x, y, first(z)), fmap(f, x, y, Base.tail(z))...)

@inline fmapt(f::F, x::NTuple, y::NTuple, z) where {F} = (f(first(x), first(y), z), fmap(f, Base.tail(x), Base.tail(y), z)...)
@inline fmapt(f::F, x::NTuple, y, z::NTuple) where {F} = (f(first(x), y, first(z)), fmap(f, Base.tail(x), y, Base.tail(z))...)
@inline fmapt(f::F, x, y::NTuple, z::NTuple) where {F} = (f(x, first(y), first(z)), fmap(f, x, Base.tail(y), Base.tail(z))...)

@inline fmap(f::F, x::VecUnroll, y, z) where {F} = VecUnroll(fmapt(f, x.data, y, z))
@inline fmap(f::F, x, y::VecUnroll, z) where {F} = VecUnroll(fmapt(f, x, y.data, z))
@inline fmap(f::F, x, y, z::VecUnroll) where {F} = VecUnroll(fmapt(f, x, y, z.data))

@inline fmap(f::F, x::VecUnroll, y::VecUnroll, z) where {F} = VecUnroll(fmapt(f, x.data, y.data, z))
@inline fmap(f::F, x::VecUnroll, y, z::VecUnroll) where {F} = VecUnroll(fmapt(f, x.data, y, z.data))
@inline fmap(f::F, x, y::VecUnroll, z::VecUnroll) where {F} = VecUnroll(fmapt(f, x, y.data, z.data))

@inline fmap(f::F, x::VecTile, y, z) where {F} = VecTile(fmapt(f, x.data, y, z))
@inline fmap(f::F, x::VecTile, y::VecUnroll, z) where {F} = VecTile(fmapt(f, x.data, y, z))
@inline fmap(f::F, x::VecTile, y, z::VecUnroll) where {F} = VecTile(fmapt(f, x.data, y, z))
@inline fmap(f::F, x::VecTile, y::VecUnroll, z::VecUnroll) where {F} = VecTile(fmapt(f, x.data, y, z))


@inline fmap(f::F, x, y::VecTile, z) where {F} = VecTile(fmapt(f, x, y.data, z))
@inline fmap(f::F, x::VecUnroll, y::VecTile, z) where {F} = VecTile(fmapt(f, x, y.data, z))
@inline fmap(f::F, x, y::VecTile, z::VecUnroll) where {F} = VecTile(fmapt(f, x, y.data, z))
@inline fmap(f::F, x::VecUnroll, y::VecTile, z::VecUnroll) where {F} = VecTile(fmapt(f, x, y.data, z))

@inline fmap(f::F, x, y, z::VecTile) where {F} = VecTile(fmapt(f, x, y, z.data))
@inline fmap(f::F, x::VecUnroll, y, z::VecTile) where {F} = VecTile(fmapt(f, x, y, z.data))
@inline fmap(f::F, x, y::VecUnroll, z::VecTile) where {F} = VecTile(fmapt(f, x, y, z.data))
@inline fmap(f::F, x::VecUnroll, y::VecUnroll, z::VecTile) where {F} = VecTile(fmapt(f, x, y, z.data))

@inline fmap(f::F, x::VecTile, y::VecTile, z) where {F} = VecTile(fmapt(f, x.data, y.data, z))
@inline fmap(f::F, x::VecTile, y, z::VecTile) where {F} = VecTile(fmapt(f, x.data, y, z.data))
@inline fmap(f::F, x, y::VecTile, z::VecTile) where {F} = VecTile(fmapt(f, x, y.data, z.data))

@inline fmap(f::F, x::VecTile, y::VecTile, z::VecUnroll) where {F} = VecTile(fmapt(f, x.data, y.data, z))
@inline fmap(f::F, x::VecTile, y::VecUnroll, z::VecTile) where {F} = VecTile(fmapt(f, x.data, y, z.data))
@inline fmap(f::F, x::VecUnroll, y::VecTile, z::VecTile) where {F} = VecTile(fmapt(f, x, y.data, z.data))



