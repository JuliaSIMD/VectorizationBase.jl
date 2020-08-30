
#
# We use these definitions because when we have other SIMD operations with masks
# LLVM optimizes the masks better.
function truncate_mask!(instrs, input, W, suffix)
    mtyp_input = "i$(max(8,W))"
    mtyp_trunc = "i$(W)"
    if mtyp_input == mtyp_trunc
        "%mask.$(suffix) = bitcast $mtyp_input %$input to <$W x i1>"
    else
        "%masktrunc.$(suffix) = trunc $mtyp_input %$input to $mtyp_trunc\n%mask.$(suffix) = bitcast $mtyp_trunc %masktrunc.$(suffix) to <$W x i1>"
    end
end
function zext_mask!(instrs, input, W, suffix)
    mtyp_input = "i$(max(8,W))"
    mtyp_trunc = "i$(W)"
    if mtyp_input == mtyp_trunc
        "%res.$(suffix) = bitcast <$W x i1> %$input to $mtyp_input"
    else
        "%restrunc.$(suffix) = bitcast <$W x i1> %$input to $mtyp_trunc\n%res.$(suffix) = zext $mtyp_trunc %restrunc.$(suffix) to $mtyp_input"
    end
end
function binary_mask_op_instrs(W, op)
    mtyp_input = "i$(max(8,W))"
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    truncate_mask!(instrs, '1', W, 1)
    push!(instrs, "%combinedmask = $op <$W x i1> %mask.0, %mask.1")
    zext_mask!(instrs, "combinedmask", W, 1)
    push!(instrs, "ret $mtyp_input %res.1")
    join(instrs, "\n")
end
function binary_mask_op(W, U, op)
    instrs = binary_mask_op_instrs(W, op)
    quote
        $(Expr(:meta,:inline))
        Mask{$W}(llvmcall($instrs, $U, Tuple{$U, $U}, m1.u, m2.u))
    end    
end

@inline Base.zero(::Mask{W,U}) where {W,U} = Mask{W}(zero(U))

@inline data(m::Mask) = m.u
@generated function Base.:(&)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U}
    binary_mask_op(W, U, "and")
end
@generated function Base.:(|)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U}
    binary_mask_op(W, U, "or")
end
@generated function Base.:(⊻)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U}
    binary_mask_op(W, U, "xor")
end
@generated function Base.:(==)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U}
    binary_mask_op(W, U, "icmp eq")
end
@generated function Base.:(!=)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U}
    binary_mask_op(W, U, "icmp ne")
end

function vadd_expr(W,U)
    instrs = String[]
    truncate_mask!(instrs, truncate_mask('0', W, 0))
    truncate_mask!(instrs, truncate_mask('1', W, 1))
    push!(instrs, """%uv.0 = zext <$W x i1> %mask.0 to <$W x i8>
    %uv.1 = zext <$W x i1> %mask.1 to <$W x i8>
    %res = add <$W x i8> %uv.0, %uv.1
    ret <$W x i8> %res""")
    :(Vec(llvmcall($(join(instrs, "\n")), Vec{$W,UInt8}, Tuple{$U, $U}, m1.u, m2.u)))
end
for (W,U) in [(2,UInt8),(4,UInt8),(8,UInt8),(16,UInt16),(32,UInt32),(64,UInt64)] # Julia 1.1 bug
    @eval @inline Base.:(+)(m1::Mask{$W,$U}, m2::Mask{$W,$U}) = $(vadd_expr(W, U))
end
# @generated function vadd(m1::Mask{W,U}, m2::Mask{W,U}) where {W, U <: Unsigned}
#     Expr(:block, Expr(:meta, :inline), vadd_expr(W,U))
# end
@inline Base.:(+)(m1::Mask, m2::Mask) = vadd(m1,m2)

# @inline Base.:(&)(m1::Mask{W}, m2::Mask{W}) where {W} = andmask(m1, m2)
@inline Base.:(&)(m::Mask{W}, u::UIntTypes) where {W} = m & Mask{W}(u)
@inline Base.:(&)(u::UIntTypes, m::Mask{W}) where {W} = Mask{W}(u) & m

@inline Base.:(&)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? m.u : zero(m.u))
@inline Base.:(&)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? m.u : zero(m.u))

@inline Base.:(|)(m1::Mask{W}, m2::Mask{W}) where {W} = ormask(m1, m2)
@inline Base.:(|)(m::Mask{W}, u::UIntTypes) where {W} = ormask(m, Mask{W}(u))
@inline Base.:(|)(u::UIntTypes, m::Mask{W}) where {W} = ormask(Mask{W}(u), m)

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

@inline Base.:(⊻)(m1::Mask{W}, m2::Mask{W}) where {W} = xormask(m1, m2)
@inline Base.:(⊻)(m::Mask{W}, u::UIntTypes) where {W} = xormask(m, Mask{W}(u))
@inline Base.:(⊻)(u::UIntTypes, m::Mask{W}) where {W} = xormask(Mask{W}(u), m)

@inline Base.:(⊻)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? ~m.u : m.u)
@inline Base.:(⊻)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? ~m.u : m.u)

@inline Base.:(<<)(m::Mask{W}, i) where {W} = Mask{W}(shl(m.u, i))
@inline Base.:(>>)(m::Mask{W}, i) where {W} = Mask{W}(shr(m.u, i))
@inline Base.:(>>>)(m::Mask{W}, i) where {W} = Mask{W}(shr(m.u, i))

for (U,W) in [(UInt8,8), (UInt16,16), (UInt32,32), (UInt64,64)]
    @eval @inline Base.any(m::Mask{$W,$U}) = m.u != $(zero(U))
    @eval @inline Base.all(m::Mask{$W,$U}) = m.u == $(typemax(U))
end
@inline Base.any(m::Mask{W}) where {W} = (m.u & max_mask(Val{W}()).u) != zero(m.u)
@inline Base.all(m::Mask{W}) where {W} = (m.u & max_mask(Val{W}()).u) == (max_mask(Val{W}()).u)

@generated function Base.:(!)(m::Mask{W,U}) where {W,U}
    mtyp_input = "i$(8sizeof(U))"
    mtyp_trunc = "i$(W)"
    instrs = String[]
    suffix = truncate_mask!(instrs, '0', W, 0)
    resv = "resvec.$(suffix)"
    push!(instrs, '%' * resv * " = xor <$W x i1> %mask.$(suffix), <$(join(("i1 true" for i in 1:W), ", "))>")
    suffix = zext_mask!(instrs, resv, W, suffix)
    push!(instrs, "ret $mtyp_input %res.$(suffix)")
    quote
        $(Expr(:meta,:inline))
        Mask{$W}(llvmcall($(join(instrs,"\n")), $U, Tuple{$U}, m.u))
    end
end
@inline Base.:(~)(m::Mask) = !m
#@inline Base.:(!)(m::Mask{W}) where {W} = Mask{W}( ~m.u )


@inline Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u == m2.u
@inline Base.:(==)(m::Mask{W}, u::UIntTypes) where {W} = m.u == u
@inline Base.:(==)(u::UIntTypes, m::Mask{W}) where {W} = u == m.u
@inline Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u != m2.u
@inline Base.:(!=)(m::Mask{W}, u::UIntTypes) where {W} = m.u != u
@inline Base.:(!=)(u::UIntTypes, m::Mask{W}) where {W} = u != m.u
# @inline Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = equalmask(m1, m2)
# @inline Base.:(==)(m::Mask{W}, u::UIntTypes) where {W} = equalmask(m1, Mask{W}(m2))
# @inline Base.:(==)(u::UIntTypes, m::Mask{W}) where {W} = equalmask(Mask{W}(m1), m2)
# @inline Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = notequalmask(m1, m2)
# @inline Base.:(!=)(m::Mask{W}, u::UIntTypes) where {W} = notequalmask(m1, Mask{W}(m2))
# @inline Base.:(!=)(u::UIntTypes, m::Mask{W}) where {W} = notequalmask(Mask{W}(m1), m2)

@inline Base.count_ones(m::Mask) = count_ones(m.u)
@inline Base.:(+)(m::Mask, i::Integer) = i + count_ones(m)
@inline Base.:(+)(i::Integer, m::Mask) = i + count_ones(m)

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
mask_type(::Val{4}) = UInt8
mask_type(::Val{8}) = UInt8
mask_type(::Val{16}) = UInt16
mask_type(::Val{32}) = UInt32
mask_type(::Val{64}) = UInt64

@generated function mask_type(::Type{T}, ::Val{P}) where {T,P}
    mask_type(pick_vector_width(P, T))
end
@generated function mask_type(::Type{T}) where {T}
    mask_type(pick_vector_width(T))
end
@generated function Base.zero(::Type{<:Mask{W}}) where {W}
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :Mask), zero(mask_type(W))))
end

@generated function max_mask(::Val{W}) where {W}
    U = mask_type(W)
    Mask{W,U}(one(U)<<W - one(U))
end
@inline max_mask(::Type{T}) where {T} = max_mask(pick_vector_width_val(T))
@generated max_mask(::Type{Mask{W,U}}) where {W,U} = Mask{W,U}(one(U)<<W - one(U))

@generated function mask(::Val{W}, l::Integer) where {W}
    M = mask_type(W)
    quote
        $(Expr(:meta,:inline))
        rem = valrem(Val{$W}(), vsub((l % $M), one($M)))
        Mask{$W,$M}($(typemax(M)) >>> ($(M(8sizeof(M))-1) - rem))
    end
end
@generated mask(::Val{W}, ::Static{L}) where {W, L} = mask(Val(W), L)
@inline mask(::Type{T}, l::Integer) where {T} = mask(pick_vector_width_val(T), l)

# @generated function masktable(::Val{W}, rem::Integer) where {W}
#     masks = Expr(:tuple)
#     for w ∈ 0:W-1
#         push!(masks.args, data(mask(Val(W), w == 0 ? W : w)))
#     end
#     Expr(
#         :block,
#         Expr(:meta,:inline),
#         Expr(:call, Expr(:curly, :Mask, W), Expr(
#             :macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)),
#             Expr(:call, :getindex, masks, Expr(:call, :+, 1, Expr(:call, :valrem, Expr(:call, Expr(:curly, W)), :rem)))
#         ))
#     )
# end

@inline tomask(m::Unsigned) = m
@inline tomask(m::Mask) = m
@generated function tomask(v::Vec{W,Bool}) where {W}
    usize = W > 8 ? nextpow2(W) : 8
    utyp = "i$(usize)"
    U = mask_type(W)
    instrs = String[]
    push!(instrs, "%bitvec = trunc <$W x i8> %0 to <$W x i1>")
    if usize == W
        push!(instrs, "%mask = bitcast <$W x i1> %bitvec to i$(W)")
    else
        push!(instrs, "%maskshort = bitcast <$W x i1> %bitvec to i$(W)")
        push!(instrs, "%mask = zext i$(W) %maskshort to i$(usize)")
    end
    push!(instrs, "ret i$(usize) %mask")
    quote
        $(Expr(:meta, :inline))
        Mask{$W}(llvmcall(
            $(join(instrs, "\n")), $U, Tuple{Vec{$W,Bool}}, v
        ))
    end
end
@inline tomask(v::AbstractSIMDVector{<:Any,Bool}) = tomask(data(v))


@inline getindexzerobased(m::Mask, i) = (m.u >>> i) % Bool
@inline function Base.getindex(m::Mask{W}, i::Integer) where {W}
    @boundscheck i > W && throw(BoundsError(m, i))
    getindexzerobased(m, i - 1)
end
# @inline function ptr_index(ptr::AbstractBitPointer, i::MM{1})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
# @inline function ptr_index(ptr::AbstractBitPointer, i::MM{2})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
# @inline function bitload(ptr::AbstractBitPointer, i::MM{4})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
@inline function ptr_index(ptr::AbstractBitPointer, i::MM{8})
    Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractBitPointer, i::MM{16})
    Base.unsafe_convert(Ptr{UInt16}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractBitPointer, i::MM{32})
    Base.unsafe_convert(Ptr{UInt32}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractBitPointer, i::MM{64})
    Base.unsafe_convert(Ptr{UInt64}, ptr.ptr), i.i >>> 3
end
@inline function bitload(ptr::AbstractBitPointer, i::MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    Mask{W}(vload(ptr, ind))
end
@inline bitload(ptr::AbstractBitPointer, i, ::Union{UIntTypes,Mask}) = bitload(ptr, i)
@inline bitload(ptr::AbstractBitPointer, i::Integer) = getindexzerobased(bitload(ptr, MM{8}(i)), i & 7)

# @inline function vstore!(ptr::AbstractBitPointer, m::Mask{8}, i::Integer)
    # vstore!(Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), (m.u % Bool), i)
# end
@inline function bitstore!(ptr::AbstractBitPointer, m::Mask{W}, i::MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end
@inline function bitstore!(ptr::AbstractBitPointer, m::Mask{W}, i::MM{W}, mask::Mask{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end

@generated function bitstore!(
    ptr::Ptr{T}, v::Mask{W,U}, mask::Mask{W,U}
) where {W,T,U<:UIntTypes}
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
        llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid, Tuple{Ptr{$T}, $U, $U},
            ptr, v.u, mask.u)
    end
end
@generated function bitstore!(
    ptr::Ptr{T}, v::Mask{W,U}, ind::I, mask::Mask{W,U}
) where {W,T,I<:Integer,U<:UIntTypes}
    @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    mtyp_input = llvmtype(U)
    mtyp_trunc = "i$W"
    decls = String[]
    instrs = String[]
    align = sizeof(U)
    push!(instrs, "%ptr = inttoptr $ptyp %0 to i8*")
    push!(instrs, "%offsetptri8 = getelementptr inbounds i8, i8* %ptr, i$(8sizeof(I)) %2")
    push!(instrs, "offsetptr = bitcast i* %offsetptri8 to <$W x i1>*")
    if mtyp_input == mtyp_trunc
        push!(instrs, "%v = bitcast $mtyp_input %1 to <$W x i1>")
        push!(instrs, "%mask = bitcast $mtyp_input %3 to <$W x i1>")
    else
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
        llvmcall($((join(decls, "\n"), join(instrs, "\n"))),
            Cvoid, Tuple{Ptr{$T}, $U, $I, $U},
            ptr, v.u, ind, mask.u)
    end
end

@inline vload(ptr::AbstractBitPointer, i::Tuple) = bitload(ptr, offset(ptr, vadd(i, ptr.offsets)))
@inline vload(ptr::AbstractBitPointer, i::Tuple, ::Mask) = vload(ptr, i)
@inline function vload(bptr::PackedStridedBitPointer{1}, (i,j)::Tuple{MM{W},<:Any}) where {W}
    j = vadd(j, bptr.offsets[2])
    s = bptr.strides[1]
    # shift = vmul(s, j) & (W - 1)
    U = mask_type(Val{W}())
    UW = widen(U)
    indbits = vadd(vadd(i.i, bptr.offsets[1]), vmul(j,s))
    ptr, ind = ptr_index(bptr, MM{W}(indbits))
    u = vload(Base.unsafe_convert(Ptr{UW}, gepbyte(ptr, ind)))
    shift = indbits & 7
    # @show ind, shift, u
    Mask{W}((u >>> shift) % U)
end

@inline getind(a::PackedStridedBitPointer{0}) = a.offsets[1]
@inline getind(a::PackedStridedBitPointer{1}) = vadd(a.offsets[1], vmul(a.offsets[2],a.strides[1]))
@inline Base.:(≥)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) ≥ getind(b)
@inline Base.:(≤)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) ≤ getind(b)
@inline Base.:(>)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) > getind(b)
@inline Base.:(<)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) < getind(b)
@inline Base.:(==)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) == getind(b)
# @inline function vstore!(bptr::PackedStridedBitPointer{1}, v::Mask{W}, (i,j)::Tuple{MM{W},<:Integer}) where {W}
#     j -= 1
#     s = bptr.strides[1]
#     shift = (s * j) & (W - 1)
#     U = mask_type(Val{W}())
#     UW = widen(U)
#     ptr, ind = ptr_index(bptr, MM{W}(i.i - 1 + j*s))
#     um = ((vload(Base.unsafe_convert(Ptr{UW}, gep(ptr, ind))) )
#     u = (v.u % UW) << shift
#     # @show ind, shift, u
#     vstore!(Base.unsafe_convert(Ptr{UW}, gep(ptr, ind)), u | um)

# end
# @inline vstore!(bptr::PackedStridedBitPointer{1}, v::Vec{W,Bool}, i::Tuple{MM{W},<:Integer}) where {W} = vstore!(bptr, tomask(v), i)
# @inline vstore!(bptr::PackedStridedBitPointer{1}, v::Mask{W}, i::Tuple{MM{W},<:Integer}, ::AbstractMask) where {W} = vstore!(bptr, v, i)
# @inline vstore!(bptr::PackedStridedBitPointer{1}, v::Vec{W,Bool}, i::Tuple{MM{W},<:Integer}, ::AbstractMask) where {W} = vstore!(bptr, tomask(v), i)
# @inline vnoaliasstore!(bptr::PackedStridedBitPointer{1}, v::Mask{W}, i::Tuple{MM{W},<:Integer}) where {W} = vstore!(bptr, v, i)
# @inline vnoaliasstore!(bptr::PackedStridedBitPointer{1}, v::Mask{W}, i::Tuple{MM{W},<:Integer}, ::AbstractMask) where {W} = vstore!(bptr, v, i)
# @inline vnoaliasstore!(bptr::PackedStridedBitPointer{1}, v::Vec{W,Bool}, i::Tuple{MM{W},<:Integer}) where {W} = vstore!(bptr, tomask(v), i)
# @inline vnoaliasstore!(bptr::PackedStridedBitPointer{1}, v::Vec{W,Bool}, i::Tuple{MM{W},<:Integer}, ::AbstractMask) where {W} = vstore!(bptr, tomask(v), i)

@inline vstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)))
@inline vstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple, u::AbstractMask) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vnoaliasstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)))
@inline vnoaliasstore!(ptr::AbstractBitPointer, v::Mask, i::Tuple, u::AbstractMask) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vstore!(ptr::AbstractBitPointer, v::Vec{<:Any,Bool}, i::Tuple) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)))
@inline vstore!(ptr::AbstractBitPointer, v::Vec{<:Any,Bool}, i::Tuple, u::AbstractMask) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vnoaliasstore!(ptr::AbstractBitPointer, v::Vec{<:Any,Bool}, i::Tuple) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)))
@inline vnoaliasstore!(ptr::AbstractBitPointer, v::Vec{<:Any,Bool}, i::Tuple, u::AbstractMask) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)), tomask(u))

@generated function Base.isodd(i::MM{W}) where {W}
    U = mask_type(W)
    evenfirst = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa % U
    # Expr(:block, Expr(:meta, :inline), :(isodd(i.i) ? Mask{$W}($oddfirst) : Mask{$W}($evenfirst)))
    Expr(:block, Expr(:meta, :inline), :(Mask{$W}($evenfirst >> (i.i & 0x03))))
end
@generated function Base.iseven(i::MM{W}) where {W}
    U = mask_type(W)
    oddfirst = 0x55555555555555555555555555555555 % U
    evenfirst = oddfirst << 1
    # Expr(:block, Expr(:meta, :inline), :(isodd(i.i) ? Mask{$W}($evenfirst) : Mask{$W}($oddfirst)))
    Expr(:block, Expr(:meta, :inline), :(Mask{$W}($oddfirst >> (i.i & 0x03))))
end

function cmp_quote(W, cond, vtyp, T)
    intrs = String["%m = $cond $vtyp %0, %1"]
    zext_mask!(instrs, 'm', W, '0')
    push!(insrts, "ret i$(max(8,W)) res.0")
    U = mask_type(W); 
    :(Mask{$W}(llvmcall($(join(instrs, "\n")), $U, Tuple{_Vec{$W,$T},_Vec{$W,$T}}, data(v1), data(v2))))
end
function icmp_quote(W, cond, bytes, T)
    vtyp = vtype(W, "i$(8bytes)");
    cmp_quote(W, "icmp " * cond, vtyp, T)
end
function fcmp_quote(W, cond, T)
    vtyp = vtype(W, T === :Float32 ? "float" : "double");
    cmp_quote(W, "fcmp nsz arcp contract reassoc " * cond, vtyp, T)
end
# @generated function compare(::Val{cond}, v1::Vec{W,I}, v2::Vec{W,I}) where {cond, W, I}
    # cmp_quote(W, cond, sizeof(I), I)
# end
# for (f,cond) ∈ [(:(==), :eq), (:(!=), :ne), (:(>), :ugt), (:(≥), :uge), (:(<), :ult), (:(≤), :ule)]
for (f,cond) ∈ [(:(==), "eq"), (:(!=), "ne"), (:(>), "ugt"), (:(≥), "uge"), (:(<), "ult"), (:(≤), "ule")]
    bytes = 1
    while bytes ≤ 8
        T = Symbol(:UInt, 8bytes)
        W = 2
        while W * bytes ≤ REGISTER_SIZE
            @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(icmp_quote(W, cond, bytes, T))
            W += W
        end
        bytes += bytes
    end
    # @eval @inline Base.$f(v1::Vec{W,I}, v2::Vec{W,I}) where {W, I <: Signed} = compare(Val{$(QuoteNode(cond))}(), v1, v2)
end
for (f,cond) ∈ [(:(==), "eq"), (:(≠), "ne"), (:(>), "sgt"), (:(≥), "sge"), (:(<), "slt"), (:(≤), "sle")]
    bytes = 1
    while bytes ≤ 8
        T = Symbol(:Int, 8bytes)
        W = 2
        while W * bytes ≤ REGISTER_SIZE
            @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(cmp_quote(W, cond, bytes, T))
            W += W
        end
        bytes += bytes
    end
end

for (f,cond) ∈ [(:(==), "oeq"), (:(>), "ogt"), (:(≥), "oge"), (:(<), "olt"), (:(≤), "ole"), (:(≠), "one")]
    for (T,bytes) ∈ [(:Float32,4), (:Float64,8)]
        W = 2
        while W * bytes ≤ REGISTER_SIZE
            @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(fcmp_quote(W, cond, T))
            W += W
        end
    end
end

@generated function vifelse(m::Mask{W,U}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,U,T}
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    selty = vtype(W, "i1")
    f = "select"
    if Base.libllvm_version ≥ v"9" && ((T === Float32) || (T === Float64))
        f *= " nsz arcp contract reassoc"
    end
    instrs = "%res = select $f $selty %0, $vtyp %1, $vtyp %2\nret $vtyp %res"
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$U,_Vec{$W,$T},_Vec{$W,$T}}, data(m), data(v1), data(v2)))
    end
end
