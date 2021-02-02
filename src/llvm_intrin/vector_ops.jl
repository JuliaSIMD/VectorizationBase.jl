
function shufflevector_instrs(W, T, I, W2)
    W2 > W && throw(ArgumentError("W for vector 1 must be at least W for vector two, but W₁ = $W < W₂ = $W2."))
    typ = LLVM_TYPES[T]
    vtyp1 = "<$W x $typ>"
    M = length(I)
    vtyp3 = "<$M x i32>"
    vtypr = "<$M x $typ>"
    mask = '<' * join(map(x->string("i32 ", x), I), ", ") * '>'
    if ((W2 == 0) | (W2 == W))
        v2 = W2 == 0 ? "undef" : "%1"
        M, """
            %res = shufflevector $vtyp1 %0, $vtyp1 $v2, $vtyp3 $mask
            ret $vtypr %res
        """
    else
        vtyp0 = "<$W2 x $typ>"
        maskpad = '<' * join(map(w->string("i32 ", w > W2 ? "undef" : string(w-1)), 1:W), ", ") * '>'
        M, """
            %pad = shufflevector $vtyp0 %1, $vtyp0 undef, <$W x i32> $maskpad
            %res = shufflevector $vtyp1 %0, $vtyp1 %pad, $vtyp3 $mask
            ret $vtypr %res    
        """
    end
end
@generated function shufflevector(v1::Vec{W,T}, v2::Vec{W2,T}, ::Val{I}) where {W,W2,T,I}
    W ≥ W2 || throw(ArgumentError("`v1` should be at least as long as `v2`, but `v1` is a `Vec{$W,$T}` and `v2` is a `Vec{$W2,$T}`."))
    M, instrs = shufflevector_instrs(W, T, I, W2)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}, _Vec{$W2,$T}}, data(v1), data(v2)))
    end
end
@generated function shufflevector(v1::Vec{W,T}, ::Val{I}) where {W,T,I}
    M, instrs = shufflevector_instrs(W, T, I, 0)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W,$T}}, data(v1)))
    end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::Vec{L,T}) where {W,L,T}
    typ = LLVM_TYPES[T]
    mask = '<' * join(map(x->string("i32 ", x ≥ L ? "undef" : string(x)), 0:W-1), ", ") * '>'
    instrs = """
        %res = shufflevector <$L x $typ> %0, <$L x $typ> undef, <$W x i32> $mask
        ret <$W x $typ> %res
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$L,$T}}, data(v)))
    end
end
@generated function vresize(::Union{StaticInt{W},Val{W}}, v::T) where {W,T<:NativeTypes}
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        ret $vtyp %ie
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$T}, v))
    end
end
@generated function shufflevector(i::MM{W,X}, ::Val{I}) where {W,X,I}
    allincr = true
    L = length(I)
    for l ∈ 2:L
        allincr &= (I[l] == I[l-1] + 1)
    end
    allincr || return Expr(:block, Expr(:meta,:inline), :(shufflevector(Vec(i), Val{$I}())))
    Expr(:block, Expr(:meta,:inline), :(MM{$L,$X}( extractelement(i, $(first(I))) )))
end
@generated function Base.vcat(a::Vec{W1,T}, b::Vec{W2,T}) where {W1,W2,T}
    W1 ≥ W2 || throw(ArgumentError("`v1` should be at least as long as `v2`, but `v1` is a `Vec{$W1,$T}` and `v2` is a `Vec{$W2,$T}`."))
    mask = Vector{String}(undef, 2W1)
    for w ∈ 0:W1+W2-1
        mask[w+1] = string(w)
    end
    for w ∈ W1+W2:2W1-1
        mask[w+1] = "undef"
    end
    M, instrs = shufflevector_instrs(W1, T, mask, W2)
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$M,$T}, Tuple{_Vec{$W1,$T}, _Vec{$W2,$T}}, data(a), data(b)))
    end
end

