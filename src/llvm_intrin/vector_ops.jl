
function shufflevector_instrs(W, T, I, two_operands)
    typ = LLVM_TYPES[T]
    vtyp1 = "<$W x $typ>"
    M = length(I)
    vtyp3 = "<$M x i32>"
    vtypr = "<$M x $typ>"
    mask = '<' * join(map(x->string("i32 ", x), I), ", ") * '>'
    v2 = two_operands ? "%1" : "undef"
    M, """
        %res = shufflevector $vtyp1 %0, $vtyp1 $v2, $vtyp3 $mask
        ret $vtypr %res
    """
end
@generated function shufflevector(v1::Vec{W,T}, v2::Vec{W,T}, ::Val{I}) where {W,T,I}
    M, instrs = shufflevector_instrs(W, T, I, true)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}, _Vec{$W,$T}}, data(v1), data(v2)))
    end
end
@generated function shufflevector(v1::Vec{W,T}, ::Val{I}) where {W,T,I}
    M, instrs = shufflevector_instrs(W, T, I, false)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}}, data(v1)))
    end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::Vec{L,T}) where {W,L,T}
    typ = LLVM_TYPES[T]
    mask = '<' * join(map(x->string("i32 ", x ≥ L ? L : x), 0:W-1), ", ") * '>'
    instrs = """
        %res = shufflevector <$L x $typ> %0, <$L x $typ> undef, <$W x i32> $mask
        ret <$W x $typ> %res
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$L,$T}}, data(v)))
    end
end

