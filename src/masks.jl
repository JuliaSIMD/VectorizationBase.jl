


@inline extract_data(m::Mask) = m.u
@inline Base.:(&)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(&)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(&)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(&)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? m.u : zero(m.u))
@inline Base.:(&)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? m.u : zero(m.u))

@inline Base.:(|)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(|)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(|)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(|)(m::Mask{W,U}, b::Bool) where {W,U} = b ? max_mask(Mask{W,U}) : m
@inline Base.:(|)(b::Bool, m::Mask{W,U}) where {W,U} = b ? max_mask(Mask{W,U}) : m
@inline Base.:(|)(m::Mask{16,UInt16}, b::Bool) where {W} = Mask{W}(b ? 0xffff : m.u)
@inline Base.:(|)(b::Bool, m::Mask{16,UInt16}) where {W} = Mask{W}(b ? 0xffff : m.u)
@inline Base.:(|)(m::Mask{8,UInt8}, b::Bool) where {W} = Mask{W}(b ? 0xff : m.u)
@inline Base.:(|)(b::Bool, m::Mask{8,UInt8}) where {W} = Mask{W}(b ? 0xff : m.u)
@inline Base.:(|)(m::Mask{4,UInt8}, b::Bool) where {W} = Mask{W}(b ? 0x0f : m.u)
@inline Base.:(|)(b::Bool, m::Mask{4,UInt8}) where {W} = Mask{W}(b ? 0x0f : m.u)
@inline Base.:(|)(m::Mask{2,UInt8}, b::Bool) where {W} = Mask{W}(b ? 0x03 : m.u)
@inline Base.:(|)(b::Bool, m::Mask{2,UInt8}) where {W} = Mask{W}(b ? 0x03 : m.u)

@inline Base.:(⊻)(m1::Mask{W}, m2::Mask{W}) where {W} = Mask{W}(m1.u & m2.u)
@inline Base.:(⊻)(m::Mask{W}, u::Unsigned) where {W} = Mask{W}(m.u & u)
@inline Base.:(⊻)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u & m.u)

@inline Base.:(⊻)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? ~m.u : m.u)
@inline Base.:(⊻)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? ~m.u : m.u)

@inline Base.:(<<)(m::Mask{W}, i) where {W} = Mask{W}(m.u << i)
@inline Base.:(>>)(m::Mask{W}, i) where {W} = Mask{W}(m.u >> i)
@inline Base.:(>>>)(m::Mask{W}, i) where {W} = Mask{W}(m.u >>> i)

@inline Base.:(~)(m::Mask{W}) where {W} = Mask{W}( ~m.u )
@inline Base.:(!)(m::Mask{W}) where {W} = Mask{W}( ~m.u )

@inline Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u == m2.u
@inline Base.:(==)(m::Mask{W}, u::Unsigned) where {W} = m.u == u
@inline Base.:(==)(u::Unsigned, m::Mask{W}) where {W} = u == m.u
@inline Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u != m2.u
@inline Base.:(!=)(m::Mask{W}, u::Unsigned) where {W} = m.u != u
@inline Base.:(!=)(u::Unsigned, m::Mask{W}) where {W} = u != m.u

@inline Base.count_ones(m::Mask) = count_ones(m.u)

function mask_type(W)
    if W <= 8
        return UInt8
    elseif W <= 16
        return UInt16
    elseif W <= 32
        return UInt32
    elseif W <= 64
        return UInt64
    else#if W <= 128
        return UInt128
    end
end

@generated function mask_type(::Type{T}, ::Val{P}) where {T,P}
    mask_type(pick_vector_width(P, T))
end
@generated function mask_type(::Type{T}) where {T}
    mask_type(pick_vector_width(T))
end

@generated function max_mask(::Type{T}) where {T}
    W = pick_vector_width(T)
    U = mask_type(W)
    Mask{W,U}(one(U)<<W - one(U))
end
@generated max_mask(::Type{Mask{W,U}}) where {W,U} = Mask{W,U}(one(U)<<W - one(U))

@generated function mask(::Type{T}, l::Integer) where {T}
    M = mask_type(T)
    W = pick_vector_width(T)
    # tup = Expr(:tuple, [Base.unsafe_trunc(M, 1 << w - 1) for w in 0:W]...) 
    quote
        $(Expr(:meta,:inline))
        rem = valrem(Val{$W}(), l - 1) + 1
        # @inbounds $tup[rem+1]
        Mask{$W,$M}(one($M) << (rem & $(typemax(M))) - $(one(M)))
    end
end

@generated function mask(::Val{W}, l::Integer) where {W}
    M = mask_type(W)
#    W = pick_vector_width(T)
    tup = Expr(:tuple, [Base.unsafe_trunc(M, 1 << w - 1) for w in 0:W]...) 
    quote
        $(Expr(:meta,:inline))
        # @inbounds $tup[rem+1]
        rem = valrem(Val{$W}(), l - 1) + 1
        Mask{$W,$M}(one($M) << (rem & $(typemax(M))) - $(one(M)))
    end
end
@generated mask(::Val{W}, ::Static{L}) where {W, L} = mask(Val(W), L)

unstable_mask(W, rem) = mask(Val(W), rem)

@generated function masktable(::Val{W}, rem::Integer) where {W}
    masks = Expr(:tuple)
    for w ∈ 0:W-1
        push!(masks.args, extract_data(unstable_mask(W, w == 0 ? W : w)))
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(:call, Expr(:curly, :Mask, W), Expr(
            :macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)),
            Expr(:call, :getindex, masks, Expr(:call, :+, 1, Expr(:call, :valrem, Expr(:call, Expr(:curly, W)), :rem)))
        ))
    )
end


@inline getindexzerobased(m::Mask, i) = (m.u >>> i) % Bool
@inline function Base.getindex(m::Mask{W}, i::Integer) where {W}
    @boundscheck i > W && throw(BoundsError(m, i))
    getindexzerobased(m, i - 1)
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{2})
    Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >> 1
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{4})
    Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >> 2
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{8})
    Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >> 3
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{16})
    Base.unsafe_convert(Ptr{UInt16}, ptr.ptr), i.i >> 4
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{32})
    Base.unsafe_convert(Ptr{UInt32}, ptr.ptr), i.i >> 5
end
@inline function ptr_index(ptr::AbstractBitPointer, i::_MM{64})
    Base.unsafe_convert(Ptr{UInt64}, ptr.ptr), i.i >> 6
end

@inline function bitload(ptr::AbstractBitPointer, i::_MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    Mask{W}(vload(ptr, ind))
end
@inline bitload(ptr::AbstractBitPointer, i, mask::Union{Unsigned,Mask}) = bitload(ptr, i)
@inline bitload(ptr::AbstractBitPointer, i::Integer) = getindexzerobased(bitload(ptr, _MM{8}(i)), i & 7)

# @inline function vstore!(ptr::AbstractBitPointer, m::Mask{8}, i::Integer)
    # vstore!(Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), (m.u % Bool), i)
# end
@inline function bitstore!(ptr::AbstractBitPointer, m::Mask{W}, i::_MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end
@inline function bitstore!(ptr::AbstractBitPointer, m::Mask{W}, i::_MM{W}, mask::Mask{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end

@generated function bitstore!(
    ptr::Ptr{T}, v::Mask{W,U}, mask::Mask{W,U}
) where {W,T,U<:Unsigned}
    @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    mtyp_input = llvmtype(U)
    mtyp_trunc = "i$W"
    decls = String[]
    instrs = String[]
    align = sizeof(U)
    push!(instrs, "%ptr = inttoptr $ptyp %0 to <$W x i1>*")
    if mtyp_input == mtyp_trunc
        push!(instrs, "%v = bitcast $mtyp_input %1 to <$W x i1>")
        push!(instrs, "%mask = bitcast $mtyp_input %2 to <$W x i1>")
    else
        push!(instrs, "%vtrunc = trunc $mtyp_input %1 to $mtyp_trunc")
        push!(instrs, "%masktrunc = trunc $mtyp_input %2 to $mtyp_trunc")
        push!(instrs, "%v = bitcast $mtyp_input %1 to <$W x i1>")
        push!(instrs, "%mask = bitcast $mtyp_trunc %masktrunc to <$W x i1>")
    end
    push!(decls,
        "declare void @llvm.masked.store.v$(W)i1(<$W x i1>, <$W x i1>*, i32, <$W x i1>)"
    )
    push!(instrs,
        "call void @llvm.masked.store.v$(W)i1(<$W x i1> %v, <$W x i1>* %ptr, i32 $align, <$W x i1> %mask)"
    )
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid, Tuple{Ptr{$T}, $U, $U},
            ptr, v.u, mask.u)
    end
end
@generated function bitstore!(
    ptr::Ptr{T}, v::Mask{W,U}, ind::I, mask::Mask{W,U}
) where {W,T,I<:Integer,U<:Unsigned}
    @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    mtyp_input = llvmtype(U)
    mtyp_trunc = "i$W"
    decls = String[]
    instrs = String[]
    align = sizeof(U)
    if mtyp_input == mtyp_trunc
        push!(instrs, "%ptr = inttoptr $ptyp %0 to <$W x i1>*")
        push!(instrs, "%offsetptr = getelementptr inbounds <$W x i1>, <$W x i1>* %ptr, i$(8sizeof(I)) %2")
        push!(instrs, "%v = bitcast $mtyp_input %1 to <$W x i1>")
        push!(instrs, "%mask = bitcast $mtyp_input %3 to <$W x i1>")
    else
        push!(instrs, "%ptr = inttoptr $ptyp %0 to i$(8align)*")
        push!(instrs, "%tempptr = getelementptr inbounds i$(8align),  i$(8align)* %ptr, i$(8sizeof(I)) %2")
        push!(instrs, "%offsetptr = bitcast i$(8align) %tempptr to <$W x i1>*")
        push!(instrs, "%vtrunc = trunc $mtyp_input %1 to $mtyp_trunc")
        push!(instrs, "%masktrunc = trunc $mtyp_input %3 to $mtyp_trunc")
        push!(instrs, "%v = bitcast $mtyp_input %1 to <$W x i1>")
        push!(instrs, "%mask = bitcast $mtyp_trunc %masktrunc to <$W x i1>")
    end
    push!(decls,
        "declare void @llvm.masked.store.v$(W)i1(<$W x i1>, <$W x i1>*, i32, <$W x i1>)"
    )
    push!(instrs,
        "call void @llvm.masked.store.v$(W)i1(<$W x i1> %v, <$W x i1>* %offsetptr, i32 $align, <$W x i1> %mask)"
    )
    push!(instrs, "ret void")
    quote
        $(Expr(:meta, :inline))
        Base.llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid, Tuple{Ptr{$T}, $U, $I, $U},
            ptr, v.u, ind, mask.u)
    end
end

@inline vload(ptr::AbstractBitPointer, i::Tuple) = bitload(ptr, offset(ptr, i))
@inline vload(ptr::AbstractBitPointer, i::Tuple, u::Mask) = bitload(ptr, offset(ptr, i), u)
@inline vstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple) = bitstore!(ptr, v, offset(ptr, i))
@inline vstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple, u::Mask) = bitstore!(ptr, v, offset(ptr, i), u)



