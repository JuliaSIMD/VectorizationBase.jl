
abstract type AbstractStridedBitPointer{N,C,B,R,X,O} <: AbstractStridedPointer{Bool,N,C,B,R,X,O} end
struct StridedBitPointer{N,C,B,R,X,O} <: AbstractStridedBitPointer{N,C,B,R,X,O}
    p::Ptr{UInt8}
    strd::X
    offsets::O
end

# @inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{1})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
# @inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{2})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
# @inline function bitload(ptr::AbstractStridedBitPointer, i::MM{4})
#     Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
# end
@inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{8})
    Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{16})
    Base.unsafe_convert(Ptr{UInt16}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{32})
    Base.unsafe_convert(Ptr{UInt32}, ptr.ptr), i.i >>> 3
end
@inline function ptr_index(ptr::AbstractStridedBitPointer, i::MM{64})
    Base.unsafe_convert(Ptr{UInt64}, ptr.ptr), i.i >>> 3
end
@inline function bitload(ptr::AbstractStridedBitPointer, i::MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    Mask{W}(vload(ptr, ind))
end
@inline bitload(ptr::AbstractStridedBitPointer, i, ::Union{Unsigned,Mask}) = bitload(ptr, i)
@inline bitload(ptr::AbstractStridedBitPointer, i::Integer) = getindexzerobased(bitload(ptr, MM{8}(i)), i & 7)

# @inline function vstore!(ptr::AbstractStridedBitPointer, m::Mask{8}, i::Integer)
    # vstore!(Base.unsafe_convert(Ptr{UInt8}, ptr.ptr), (m.u % Bool), i)
# end
@inline function bitstore!(ptr::AbstractStridedBitPointer, m::Mask{W}, i::MM{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end
@inline function bitstore!(ptr::AbstractStridedBitPointer, m::Mask{W}, i::MM{W}, mask::Mask{W}) where {W}
    ptr, ind = ptr_index(ptr, i)
    vstore!(ptr, m.u, ind)
end

@generated function bitstore!(
    ptr::Ptr{T}, v::Mask{W,U}, mask::Mask{W,U}
) where {W,T,U<:Unsigned}
    @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    mtyp_input = LLVM_TYPES[U]
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
) where {W,T,I<:Integer,U<:Unsigned}
    @assert isa(Aligned, Bool)
    ptyp = JuliaPointerType
    mtyp_input = LLVM_TYPES[U]
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

@inline vload(ptr::AbstractStridedBitPointer, i::Tuple) = bitload(ptr, offset(ptr, vadd(i, ptr.offsets)))
@inline vload(ptr::AbstractStridedBitPointer, i::Tuple, ::Mask) = vload(ptr, i)
# @inline function vload(bptr::PackedStridedBitPointer{1}, (i,j)::Tuple{MM{W},<:Any}) where {W}
#     j = vadd(j, bptr.offsets[2])
#     s = bptr.strides[1]
#     # shift = vmul(s, j) & (W - 1)
#     U = mask_type(Val{W}())
#     UW = widen(U)
#     indbits = vadd(vadd(i.i, bptr.offsets[1]), vmul(j,s))
#     ptr, ind = ptr_index(bptr, MM{W}(indbits))
#     u = vload(Base.unsafe_convert(Ptr{UW}, gepbyte(ptr, ind)))
#     shift = indbits & 7
#     # @show ind, shift, u
#     Mask{W}((u >>> shift) % U)
# end

# @inline getind(a::PackedStridedBitPointer{0}) = a.offsets[1]
# @inline getind(a::PackedStridedBitPointer{1}) = vadd(a.offsets[1], vmul(a.offsets[2],a.strides[1]))
# @inline Base.:(≥)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) ≥ getind(b)
# @inline Base.:(≤)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) ≤ getind(b)
# @inline Base.:(>)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) > getind(b)
# @inline Base.:(<)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) < getind(b)
# @inline Base.:(==)(a::PackedStridedBitPointer, b::PackedStridedBitPointer) = getind(a) == getind(b)


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

@inline vstore!(ptr::AbstractStridedBitPointer, v::Mask, i::Tuple) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)))
@inline vstore!(ptr::AbstractStridedBitPointer, v::Mask, i::Tuple, u::AbstractMask) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vnoaliasstore!(ptr::AbstractStridedBitPointer, v::Mask, i::Tuple) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)))
@inline vnoaliasstore!(ptr::AbstractStridedBitPointer, v::Mask, i::Tuple, u::AbstractMask) = bitstore!(ptr, v, offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vstore!(ptr::AbstractStridedBitPointer, v::Vec{<:Any,Bool}, i::Tuple) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)))
@inline vstore!(ptr::AbstractStridedBitPointer, v::Vec{<:Any,Bool}, i::Tuple, u::AbstractMask) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)), tomask(u))
@inline vnoaliasstore!(ptr::AbstractStridedBitPointer, v::Vec{<:Any,Bool}, i::Tuple) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)))
@inline vnoaliasstore!(ptr::AbstractStridedBitPointer, v::Vec{<:Any,Bool}, i::Tuple, u::AbstractMask) = bitstore!(ptr, tomask(v), offset(ptr, vadd(i, ptr.offsets)), tomask(u))

