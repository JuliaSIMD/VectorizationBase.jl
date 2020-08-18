

Base.promote_rule(::Type{Vec{W,T1}}, ::Type{T2}) where {W,T1,T2<:NativeTypes} = Vec{W,promote_rule(T1,T2)}
Base.promote_rule(::Type{Vec{W,T1}}, ::Type{Vec{W,T2}}) where {W,T1,T2} = Vec{W,promote_rule(T1,T2)}

Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{T2}) where {N,W,T1,T2<:NativeTypes} = VecUnroll{N,W,promote_rule(T1,T2)}
Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{Vec{W,T2}}) where {N,W,T1,T2} = VecUnroll{N,W,promote_rule(T1,T2)}
Base.promote_rule(::Type{VecUnroll{N,W,T1}}, ::Type{VecUnroll{N,W,T2}}) where {N,W,T1,T2} = VecUnroll{N,W,promote_rule(T1,T2)}

Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{T2}) where {M,N,W,T1,T2<:NativeTypes} = VecTile{M,N,W,promote_rule(T1,T2)}
Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{Vec{W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecUnroll{M,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}
Base.promote_rule(::Type{VecTile{M,N,W,T1}}, ::Type{VecTile{M,N,W,T2}}) where {M,N,W,T1,T2} = VecTile{M,N,W,promote_rule(T1,T2)}


