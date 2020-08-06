@generated function vsub(v::_Vec{_W,T}) where {_W,T<:FloatingTypes}
    W = _W + 1
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = "%res = fneg fast $vtyp %0\nret $vtyp %res"
    quote
        $(Expr(:meta, :inline))
        llvmcall( $instrs, Vec{$W,$T}, Tuple{Vec{$W,$T}}, v )
    end
end
@inline Base.:(-)(v::SVec{W,T}) where {W,T} = SVec(vsub(extract_data(v)))
@inline vsub(v::SVec{W,T}) where {W,T} = SVec(vsub(extract_data(v)))

