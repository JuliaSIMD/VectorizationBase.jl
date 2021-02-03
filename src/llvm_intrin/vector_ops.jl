
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



function transpose_vecunroll_quote(W)
    @assert VectorizationBase.ispow2(W)
    log2W = VectorizationBase.intlog2(W)
    q = Expr(:block, Expr(:meta, :inline), :(vud = data(vu)))
    N = W # N vectors of length W
    vectors1 = [Symbol(:v_, n) for n ∈ 0:N-1]
    vectors2 = [Symbol(:v_, n+N) for n ∈ 0:N-1]
    # z = Expr(:call, Expr(:curly, Expr(:(.), :VectorizationBase, QuoteNode(:MM)), W), 0)
    # for n ∈ 1:N
    #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vload)), :ptrA, Expr(:tuple, z, n-1))))
    # end
    for n ∈ 1:N
        push!(q.args, Expr(:(=), vectors1[n], Expr(:ref, :vud, n)))
    end
    Nhalf = N >>> 1
    shuffinstr = Expr(:(.), :VectorizationBase, QuoteNode(:shufflevector))
    vecstride = 1
    partition_stride = 2
    for nsplits in 0:log2W-1
        shuffle0 = VectorizationBase.transposeshuffle(nsplits, W, false)
        shuffle1 = VectorizationBase.transposeshuffle(nsplits, W, true)
        for partition ∈ 0:(W >>> (nsplits+1))-1
            for _n1 ∈ 1:vecstride
                n1 = partition * partition_stride + _n1
                n2 = n1 + vecstride
                v11 = vectors1[n1]
                v12 = vectors1[n2]
                v21 = vectors2[n1]
                v22 = vectors2[n2]
                shuff1 = Expr(:call, shuffinstr, v11, v12, shuffle0)
                shuff2 = Expr(:call, shuffinstr, v11, v12, shuffle1)
                push!(q.args, Expr(:(=), v21, shuff1))
                push!(q.args, Expr(:(=), v22, shuff2))
            end
        end
        vectors1, vectors2 = vectors2, vectors1
        vecstride <<= 1
        partition_stride <<= 1
        # @show vecstride <<= 1
    end
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, vectors1[n])
    end
    # for n ∈ 1:N
    #     push!(q.args, Expr(:(=), vectors1[n], Expr(:call, Expr(:(.), :VectorizationBase, QuoteNode(:vstore!)), :ptrB, vectors1[n], Expr(:tuple, z, n-1))))
    # end
    push!(q.args, Expr(:call, :VecUnroll, t))
    q
end
@generated function transpose_vecunroll(vu::VecUnroll{N,W}) where {N,W}
    N+1 == W || throw(ArgumentError("Transposing is currently only supported for sets of vectors of size equal to their length, but received $(N+1) vectors of length $W."))
    W == 1 && return :vu
    transpose_vecunroll_quote(W)
    # code below lets LLVM do it.
    # q = Expr(:block, Expr(:meta,:inline), :(vud = data(vu)))
    # S = W
    # syms = Vector{Symbol}(undef, W)
    # for w ∈ 1:W
    #     syms[w] = v = Symbol(:v_, w)
    #     push!(q.args, Expr(:(=), v, Expr(:ref, :vud, w)))
    # end
    # while S > 1
    #     S >>>= 1
    #     for s ∈ 1:S
    #         v1 = syms[2s-1]
    #         v2 = syms[2s  ]
    #         vc = Symbol(v1,:_,v2)
    #         push!(q.args, Expr(:(=), vc, Expr(:call, :vcat, v1, v2)))
    #         syms[s] = vc
    #     end        
    # end
    # t = Expr(:tuple)
    # v1 = syms[1];# v2 = syms[2]
    # for w1 ∈ 0:N
    #     shufftup = Expr(:tuple)
    #     for w2 ∈ 0:N
    #         push!(shufftup.args, w2*W + w1)
    #     end
    #     push!(t.args, Expr(:call, :shufflevector, v1, Expr(:call, Expr(:curly, :Val, shufftup))))
    #     # push!(t.args, Expr(:call, :shufflevector, v1, v2, Expr(:call, Expr(:curly, :Val, shufftup))))
    # end
    # push!(q.args, Expr(:call, :VecUnroll, t))
    # q
end

