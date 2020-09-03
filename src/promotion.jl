
pick_promote(::Type{T}, ::Type{Union{}}) where {T} = T
pick_promote(::Type{Union{}}, ::Type{T}) where {T} = T
pick_promote(::Type{T}, ::Type{T}) where {T} = T
symmetric_promote_rule(::Type{T1}, ::Type{T2}) where {T1,T2} = pick_promote(promote_rule(T1,T2), promote_rule(T2,T1))
symmetric_promote_rule(::Type{T}, ::Type{T}) where {T} = T

ff_promote_rule(::Type{T1}, ::Type{T2}) where {T1,T2} = symmetric_promote_rule(T1,T2)
ff_promote_rule(::Type{T}, ::Type{<:Integer}) where {T <: Integer} = T
ff_promote_rule(::Type{T}, ::Type{<:FloatingTypes}) where {T <: FloatingTypes} = T

Base.promote_rule(::Type{<:AbstractSIMDVector{W,T1}}, ::Type{T2}) where {W,T1,T2<:NativeTypes} = Vec{W,ff_promote_rule(T1,T2)}
Base.promote_rule(::Type{<:AbstractSIMDVector{W,T1}}, ::Type{<:AbstractSIMDVector{W,T2}}) where {W,T1,T2} = Vec{W,symmetric_promote_rule(T1,T2)}


Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{T2}) where {N,W,T1,T2<:NativeTypes} = VecUnroll{N,W,ff_promote_rule(T1,T2)}
Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{Vec{W,T2}}) where {N,W,T1,T2} = VecUnroll{N,W,symmetric_promote_rule(T1,T2)}
Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{VecUnroll{N,W,T2}}) where {N,W,T1,T2} = VecUnroll{N,W,symmetric_promote_rule(T1,T2)}

# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{T2}) where {M,N,W,T1,T2<:NativeTypes} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{Vec{W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecUnroll{M,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
# Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecTile{M,N,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}


