
const SCOPE_METADATA = """
!1 = !{!\"noaliasdomain\"}
!2 = !{!\"noaliasscope\", !1}
!3 = !{!2}
"""
const LOAD_SCOPE_TBAA = SCOPE_METADATA * """
!4 = !{!"jtbaa"}
!5 = !{!6, !6, i64 0, i64 0}
!6 = !{!"jtbaa_arraybuf", !4, i64 0}
"""
const STORE_TBAA = """
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!"jtbaa_data", !4, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"jtbaa_arraybuf", !6, i64 0}
"""


"""
An omnibus offset constructor.

The general motivation for generating the memory addresses as LLVM IR rather than combining multiple lllvmcall Julia functions is
that we want to minimize the `inttoptr` and `ptrtoint` calculations as we go back and fourth. These can get in the way of some
optimizations, such as memory address calculations.
It is particulary import for `gather` and `scatter`s, as these functions take a `Vec{W,Ptr{T}}` argument to load/store a
`Vec{W,T}` to/from. If `sizeof(T) < sizeof(Int)`, converting the `<W x \$(typ)*` vectors of pointers in LLVM to integer
vectors as they're represented in Julia will likely make them too large to fit in a single register, splitting the operation
into multiple operations, forcing a corresponding split of the `Vec{W,T}` vector as well.
This would all be avoided by not promoting/widenting the `<W x \$(typ)>` into a vector of `Int`s.

For this last issue, an alternate workaround would be to wrap a `Vec` of 32-bit integers with a type that defines it as a pointer for use with
internal llvmcall functions, but I haven't really explored this optimization.
"""
function offset_ptr(
    ::Type{T}, ind_type::Symbol, indargname, ibits::Int, W::Int = 1, X::Int = 1, M::Int = 1, O::Int = 0, forgep::Bool = false
) where {T}
    i = 0
    sizeof_T = sizeof(T)
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ) # vtyp is dest type
    instrs = String[]
    if M == 0
        ind_type = :StaticInt
    elseif ind_type === :StaticInt
        M = 0
    end
    if isone(W)
        X = 1
    end
    
    tz = min(trailing_zeros(M), 3)
    tzf = 1 << tz
    index_gep_typ = ((tzf == sizeof_T) | iszero(M)) ? vtyp : "i$(tzf << 3)"
    M >>= tz
    # after this block, we will have a index_gep_typ pointer
    if iszero(O)
        push!(instrs, "%ptr.$(i) = inttoptr $(JULIAPOINTERTYPE) %0 to $(index_gep_typ)*"); i += 1
    else # !iszero(O)
        if iszero(O & (tzf - 1)) # then index_gep_typ works for the constant offset
            offset_gep_typ = "i8"
            offset = O
        else # then we need another intermediary
            offset_gep_typ = index_gep_typ
            offset = O >>> tz
        end
        push!(instrs, "%ptr.$(i) = inttoptr $(JULIAPOINTERTYPE) %0 to $(offset_gep_typ)*"); i += 1
        push!(instrs, "%ptr.$(i) = getelementptr inbounds $(offset_gep_typ), $(offset_gep_typ)* %ptr.$(i-1), i32 $(offset)"); i += 1
        if forgep && iszero(M) && (iszero(X) || isone(X))
            push!(instrs, "%ptr.$(i) = ptrtoint $(offset_gep_typ)* %ptr.$(i-1) to $(JULIAPOINTERTYPE)"); i += 1
            return instrs, i
        elseif offset_gep_typ != index_gep_typ
            push!(instrs, "%ptr.$(i) = bitcast $(offset_gep_typ)* %ptr.$(i-1) to $(index_gep_typ)*"); i += 1
        end
    end
    # will do final type conversion
    if ind_type === :Vec
        if isone(M)
            indname = indargname
        else
            indname = "indname"
            constmul = llvmconst(W, "i$(ibits) $M")
            push!(instrs, "%$(indname) = mul nsw <$W x i$(ibits)> %$(indargname), <i$(ibits) $(constmul)>")
        end
        push!(instrs, "%ptr.$(i) = getelementptr inbounds $(index_gep_typ), $(index_gep_typ)* %ptr.$(i-1), <$W x i$(ibits)> %$(indname)"); i += 1
        if forgep
            push!(instrs, "%ptr.$(i) = ptrtoint <$W x $index_gep_typ*> %ptr.$(i-1) to <$W x $JULIAPOINTERTYPE>"); i += 1
        elseif index_gep_typ != vtyp
            push!(instrs, "%ptr.$(i) = bitcast <$W x $index_gep_typ*> %ptr.$(i-1) to <$W x $typ*>"); i += 1
        end
        return instrs, i
    end
    if ind_type === :Integer
        if isone(M)
            indname = indargname
        else
            indname = "indname"
            push!(instrs, "%$(indname) = mul nsw i$(ibits) %$(indargname), $M")
        end
        # TODO: if X != 1 and X != 0, check if it is better to gep -> gep, or broadcast -> add -> gep
        push!(instrs, "%ptr.$(i) = getelementptr inbounds $(index_gep_typ), $(index_gep_typ)* %ptr.$(i-1), i$(ibits) %$(indname)"); i += 1
    end
    # ind_type === :Integer || ind_type === :StaticInt
    if forgep && (isone(X) | iszero(X)) # if forgep, just return now
        push!(instrs, "%ptr.$(i) = ptrtoint $(index_gep_typ)* %ptr.$(i-1) to $JULIAPOINTERTYPE"); i += 1
    elseif index_gep_typ != vtyp
        push!(instrs, "%ptr.$(i) = bitcast $(index_gep_typ)* %ptr.$(i-1) to $vtyp*"); i += 1
    end
    if !(isone(X) | iszero(X)) # vec
        vibytes = min(4, REGISTER_SIZE ÷ W)
        vityp = "i$(8vibytes)"
        vi = join((X*w for w ∈ 0:W-1), ", $vityp")
        push!(insrts, "%ptr.$(i) = getelementptr inbounds $(vtyp), $(vtyp)* %ptr.$(i-1), <$W x $(vityp)> <$vityp $vi>"); i += 1
        if forgep
            push!(instrs, "%ptr.$(i) = ptrtoint <$W x $vtyp*> %ptr.$(i-1) to <$W x $JULIAPOINTERTYPE>"); i += 1
        end
    end
    instrs, i
end
gep_returns_vector(W::Int, X::Int, M::Int, ind_type::Symbol) = (!isone(W) && ((ind_type === :Vec) || !(isone(X) | iszero(X))))
#::Type{T}, ::Type{I}, W::Int = 1, ivec::Bool = false, constmul::Int = 1) where {T <: NativeTypes, I <: Integer}
function gep_quote(
    ::Type{T}, ind_type::Symbol, ::Type{I}, W::Int = 1, X::Int = sizeof(T), M::Int = 1, O::Int = 0, forgep::Bool = false
) where {T, I}
    if W > 1 && ind_type !== :Vec
        X, Xr = divrem(X, sizeof(T))
        @assert iszero(Xr)
    end
    if iszero(O) && (iszero(X) | isone(X)) && (iszero(M) || ind_type === :StaticInt)
        return Expr(:block, Expr(:meta, :inline), :ptr)
    end
    ibits = 8sizeof(I)
    # ::Type{T}, ind_type::Symbol, indargname = '1', ibytes::Int, W::Int = 1, X::Int = 1, M::Int = 1, O::Int = 0, forgep::Bool = false
    instrs, i = offset_ptr(T, ind_type, '1', ibits, W, X, M, O, true)
    ret = Expr(:curly, :Ptr, T)
    lret = JULIAPOINTERTYPE
    if gep_returns_vector(W, X, M, ind_type)
        ret = Expr(:curly, :_Vec, W, ret)
        lret = "<$W x $lret>"
    end

    args = Expr(:curly, :Tuple, Expr(:curly, :Ptr, T))
    largs = String[JULIAPOINTERTYPE]
    arg_syms = Union{Symbol,Expr}[:ptr]
    
    if !(iszero(M) || ind_type === :StaticInt)
        push!(arg_syms, Expr(:call, :data, :i))
        if ind_type === :Integer
            push!(args.args, I)
            push!(largs, "i$(ibits)")
        else
            push!(args.args, Expr(:curly, :_Vec, W, I))
            push!(largs, "<$W x i$(ibits)>")
        end
    end
    push!(instrs, "ret $lret %ptr.$(i-1)")
    llvmcall_expr("", join(instrs, "\n"), ret, args, lret, largs, arg_syms)
end

@generated function gep(ptr::Ptr{T}, i::I) where {T <: NativeTypes, I <: Integer}
    gep_quote(T, :Integer, I, 1, 1, 1, 0, true)
end
@generated function gep(ptr::Ptr{T}, ::StaticInt{N}) where {T <: NativeTypes, N}
    gep_quote(T, :StaticInt, Int, 1, 1, 0, N, true)
end
@generated function gep(ptr::Ptr{T}, i::LazyMulAdd{M,I}) where {T <: NativeTypes, I <: Integer, M}
    gep_quote(T, :Integer, I, 1, 1, M, 0, true)
end
@generated function gep(ptr::Ptr{T}, i::Vec{W,I}) where {W, T <: NativeTypes, I <: Integer}
    gep_quote(T, :Vec, I, W, 1, 1, 0, true)
end
@generated function gep(ptr::Ptr{T}, i::LazyMulAdd{M,Vec{W,I}}) where {W, T <: NativeTypes, I <: Integer, M}
    gep_quote(T, :Vec, I, W, 1, M, 0, true)
end
@inline gesp(ptr::AbstractStridedPointer, i) = similar_no_offset(ptr, gep(ptr, i))

function vload_quote(
    ::Type{T}, ::Type{I}, ind_type::Symbol, W::Int, X::Int, M::Int, O::Int, mask::Bool, align::Bool = false
) where {T <: NativeTypes, I <: Integer}
    ibits = 8sizeof(I)
    if W > 1 && ind_type !== :Vec
        X, Xr = divrem(X, sizeof(T))
        @assert iszero(Xr)
    end
    instrs, i = offset_ptr(T, ind_type, '1', ibits, W, X, M, O, false)
    
    grv = gep_returns_vector(W, X, M, ind_type)
    jtyp = isone(W) ? T : _Vec{W,T}
    
    alignment = (align & (!grv)) ? Base.datatype_alignment(jtyp) : Base.datatype_alignment(T)
    
    decl = LOAD_SCOPE_TBAA
    dynamic_index = !(iszero(M) || ind_type === :StaticInt)

    typ = LLVM_TYPES[T]
    lret = vtyp = vtype(W, typ)
    
    mask && truncate_mask!(instrs, '1' + dynamic_index, W, 0)
    if grv
        loadinstr = "$vtyp @llvm.masked.gather." * suffix(W, T) * '.' * suffix(W, Ptr{T})
        decl *= "declare $loadinstr(<$W x $typ*>, i32, <$W x i1>, $vtyp)"
        m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
        passthrough = mask ? "zeroinitializer" : "undef"
        push!(instrs, "%res = call $loadinstr(<$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m, $vtyp $passthrough), !alias.scope !3, !tbaa !5")
    elseif mask
        suff = suffix(W, T)
        loadinstr = "$vtyp @llvm.masked.load." * suff * ".p0" * suff
        decl *= "declare $loadinstr($vtyp*, i32, <$W x i1>, $vtyp)"
        push!(instrs, "%res = call $loadinstr($vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0, $vtyp zeroinitializer), !alias.scope !3, !tbaa !5")
    else
        push!(instrs, "%res = load $vtyp, $vtyp* %ptr.$(i-1), align $alignment, !alias.scope !3, !tbaa !5")
    end
    push!(instrs, "ret $vtyp %res")

    ret = jtyp
    args = Expr(:curly, :Tuple, Expr(:curly, :Ptr, T))
    largs = String[JULIAPOINTERTYPE]
    arg_syms = Union{Symbol,Expr}[:ptr]
    if dynamic_index
        push!(arg_syms, :(data(i)))
        if ind_type === :Integer
            push!(args.args, I)
            push!(largs, "i$(ibits)")
        else
            push!(args.args, :(_Vec{$W,$I}))
            push!(largs, "<$W x i$(ibits)>")
        end
    end
    if mask
        push!(arg_syms, :(data(m)))
        push!(args.args, mask_type(W))
        push!(largs, "i$(max(8,nextpow2(W)))")
    end
    llvmcall_expr(decl, join(instrs, "\n"), ret, args, lret, largs, arg_syms)
end
# vload_quote(T, ::Type{I}, ind_type::Symbol, W::Int, X, M, O, mask, align = false)
@generated function vload(ptr::Ptr{T}) where {T <: NativeTypes}
    vload_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, false)
end
@generated function vload(ptr::Ptr{T}, i::I) where {T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, 1, 1, 1, 0, false, false)
end
@generated function vload(ptr::Ptr{T}, ::StaticInt{N}) where {T <: NativeTypes, N}
    vload_quote(T, Int, :StaticInt, 1, 1, 0, N, false, false)
end
@generated function vload(ptr::Ptr{T}, i::Vec{W,I}) where {W, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Vec, W, 1, 1, 0, false, false)
end
@generated function vload(ptr::Ptr{T}, i::MM{W,X,I}) where {W, X, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, W, X, 1, 0, false, false)
end
@generated function vload(ptr::Ptr{T}, ::MM{W,X,StaticInt{N}}) where {W, X, T <: NativeTypes, N}
    vload_quote(T, Int, :StaticInt, W, X, 1, N, false, false)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,I}) where {M, O, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, 1, 1, M, O, false, false)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,Vec{W,I}}) where {W, M, O, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Vec, W, 1, M, O, false, false)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,I}}) where {W, M, O, X, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, W, X, M, O, false, false)
end

@generated function vload(ptr::Ptr{T}, i::Vec{W,I}, m::Mask{W}) where {W, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Vec, W, 1, 1, 0, true, false)
end
@generated function vload(ptr::Ptr{T}, i::MM{W,X,I}, m::Mask{W}) where {W, X, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, W, X, 1, 0, true, false)
end
@generated function vload(ptr::Ptr{T}, ::MM{W,X,StaticInt{N}}, m::Mask{W}) where {W, X, T <: NativeTypes, N}
    vload_quote(T, Int, :StaticInt, W, X, 1, N, true, false)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,Vec{W,I}}, m::Mask{W}) where {W, M, O, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Vec, W, 1, M, O, true, false)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,I}}, m::Mask{W}) where {W, M, O, X, T <: NativeTypes, I <: Integer}
    vload_quote(T, I, :Integer, W, X, M, O, true, false)
end


# @generated function vload(ptr_base::Ptr{T}, i::Unroll{W,U}) where {W,U,T}
#     t = Expr(:tuple)
#     for u ∈ 0:U-1
#         push!(t.args, :(vload(ptr, MM{$W}(lazymul(StaticInt{$u}(), x)))))
#     end
#     quote
#         $(Expr(:meta,:inline))
#         x = stride(i)
#         ptr = gep(ptr_base, base(i))
#         VecUnroll($t)
#     end
# end

function vstore_quote(
    ::Type{T}, ::Type{I}, ind_type::Symbol, W::Int, X::Int, M::Int, O::Int, mask::Bool, align = false, noalias::Bool = false
) where {T <: NativeTypes, I <: Integer}
    ibits = 8sizeof(I)
    if W > 1 && ind_type !== :Vec
        X, Xr = divrem(X, sizeof(T))
        @assert iszero(Xr)
    end
    instrs, i = offset_ptr(T, ind_type, '2', ibits, W, X, M, O, false)
    
    grv = gep_returns_vector(W, X, M, ind_type)
    jtyp = isone(W) ? T : _Vec{W,T}
    
    alignment = (align & (!grv)) ? Base.datatype_alignment(jtyp) : Base.datatype_alignment(T)

    decl = noalias ? SCOPE_METADATA * STORE_TBAA : STORE_TBAA
    dynamic_index = !(iszero(M) || ind_type === :StaticInt)

    typ = LLVM_TYPES[T]
    lret = vtyp = vtype(W, typ)
    metadata = noalias ? "!alias.scope !3, !tbaa !7" : "!tbaa !7"
    mask && truncate_mask!(instrs, '2' + dynamic_index, W, 0)
    if grv
        storeinstr = "void @llvm.masked.scatter." * suffix(W, T) * '.' * suffix(W, Ptr{T})
        decl *= "declare $storeinstr($vtyp, <$W x $typ*>, i32, <$W x i1>)"
        m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
        push!(instrs, "call $storeinstr($vtyp %1, <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m), $metadata")
        # push!(instrs, "call $storeinstr($vtyp %1, <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m)")
    elseif mask
        suff = suffix(W, T)
        storeinstr = "void @llvm.masked.store." * suff * ".p0" * suff
        decl *= "declare $storeinstr($vtyp, $vtyp*, i32, <$W x i1>)"
        push!(instrs, "call $storeinstr($vtyp %1, $vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0), $metadata")
    else
        push!(instrs, "store $vtyp %1, $vtyp* %ptr.$(i-1), align $alignment, $metadata")
    end
    push!(instrs, "ret void")
    ret = :Cvoid; lret = "void"
    args = Expr(:curly, :Tuple, Expr(:curly, :Ptr, T), isone(W) ? T : Expr(:curly, :NTuple, W, Expr(:curly, :VecElement, T)))
    largs = String[JULIAPOINTERTYPE, vtyp]
    arg_syms = Union{Symbol,Expr}[:ptr, Expr(:call, :data, :v)]
    if dynamic_index
        push!(arg_syms, :(data(i)))
        if ind_type === :Integer
            push!(args.args, I)
            push!(largs, "i$(ibits)")
        else
            push!(args.args, :(_Vec{$W,$I}))
            push!(largs, "<$W x i$(ibits)>")
        end
    end
    if mask
        push!(arg_syms, :(data(m)))
        push!(args.args, mask_type(W))
        push!(largs, "i$(max(8,nextpow2(W)))")
    end
    llvmcall_expr(decl, join(instrs, "\n"), ret, args, lret, largs, arg_syms)
end

@generated function vstore!(ptr::Ptr{T}, v::T) where {T <: NativeTypes}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::T, i::I) where {T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, 1, 1, 1, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::T, ::StaticInt{N}) where {T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, N, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}) where {W, X, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, 1, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::I) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}) where {W, T <: NativeTypes}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, 0, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}) where {W, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}) where {W, X, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, false, false, false)
end

@generated function vstore!(ptr::Ptr{T}, v::T, i::LazyMulAdd{M,O,I}) where {T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, 1, 1, M, O, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, M, O, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}) where {W, X, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, M, O, false, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, false, false, false)
end

@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}, m::Mask{W}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}, m::Mask{W}) where {W, X, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, 1, 0, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::I, m::Mask{W}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}, m::Mask{W}) where {W, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}, m::Mask{W}) where {W, X, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}, m::Mask{W}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, M, O, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}, m::Mask{W}) where {W, X, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, M, O, true, false, false)
end
@generated function vstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}, m::Mask{W}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, true, false, false)
end

@generated function vnoaliasstore!(ptr::Ptr{T}, v::T) where {T <: NativeTypes}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::T, i::I) where {T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, 1, 1, 1, 0, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::T, ::StaticInt{N}) where {T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, N, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}) where {W, X, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, 1, 0, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::I) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}) where {W, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}) where {W, X, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, false, false, true)
end

@generated function vnoaliasstore!(ptr::Ptr{T}, v::T, i::LazyMulAdd{M,O,I}) where {T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, 1, 1, M, O, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, M, O, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}) where {W, X, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, M, O, false, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, false, false, true)
end

@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}, m::Mask{W}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}, m::Mask{W}) where {W, X, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, 1, 0, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::I, m::Mask{W}) where {W, T <: NativeTypes, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}, m::Mask{W}) where {W, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}, m::Mask{W}) where {W, X, T <: NativeTypes, N}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}, m::Mask{W}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Vec, W, 1, M, O, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}, m::Mask{W}) where {W, X, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, X, M, O, true, false, true)
end
@generated function vnoaliasstore!(ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}, m::Mask{W}) where {W, T <: NativeTypes, M, O, I <: Integer}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, true, false, true)
end



"""
AU - Unrolled axis
F - Factor, step size per unroll
N - How many times is it unrolled
AV - Vectorized axis
W - vector width
M - bitmask indicating whether each factor is masked
i::I - index
"""
struct Unroll{AU,F,N,AV,W,M,I}
    i::I
end
@inline Unroll{AU,F,N,AV,W,M}(i::I) where {AU,F,N,AV,W,M,I} = Unroll{AU,F,N,AV,W,M,I}(i)

function unrolled_indicies(D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int)
    baseind = Expr(:tuple)
    for d in 1:D
        i = Expr(:call, :Zero)
        if d == AV && W > 1
            i = Expr(:call, Expr(:curly, :MM, W), i)
        end
        push!(baseind.args, i)
    end
    WF = AU == AV ? W : 1
    inds = Vector{Expr}(undef, N)
    inds[1] = baseind
    for n in 1:N-1
        ind = copy(baseind)
        i = Expr(:call, Expr(:curly, :StaticInt, n*F*WF))
        if AU == AV && W > 1
            i = Expr(:call, Expr(:curly, :MM, W), i)
        end
        ind.args[AU] = i
        inds[n+1] = ind
    end
    inds
end

function vload_unroll_quote(D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, mask::Bool)
    t = Expr(:tuple)
    inds = unrolled_indicies(D, AU, F, N, AV, W)
    for n in 1:N
        l = Expr(:call, :vload, :gptr, inds[n])
        (mask && (M % Bool)) && push!(l.args, :m)
        M >>= 1
        push!(t.args, l)
    end
    quote
        $(Expr(:meta, :inline))
        gptr = gesp(ptr, u.i)
        VecUnroll($t)
    end
end

@generated function vload(ptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,I}) where {AU,F,N,AV,W,M,I,T,D}
    vload_unroll_quote(D, AU, F, N, AV, W, M, false)
end
@generated function vload(ptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,I}, m::Mask{W}) where {AU,F,N,AV,W,M,I,T,D}
    vload_unroll_quote(D, AU, F, N, AV, W, M, true)
end

function vstore_unroll_quote(D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, mask::Bool, noalias::Bool)
    t = Expr(:tuple)
    f = noalias ? :vnoaliasstore! : :vstore!
    inds = unrolled_indicies(D, AU, F, N, AV, W)
    q = quote
        $(Expr(:meta, :inline))
        gptr = gesp(ptr, u.i)
        t = data(v)
    end
    for n in 1:N
        l = Expr(:call, f, :gptr, Expr(:ref, :t, n), inds[n])
        (mask && (M % Bool)) && push!(l.args, :m)
        M >>= 1
        push!(q.args, l)
    end
    q
end
@generated function vstore!(ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}) where {AU,F,N,AV,W,M,I,T,D,Nm1}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, false, false)
end
@generated function vstore!(ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}, m::Mask{W}) where {AU,F,N,AV,W,M,I,T,D,Nm1}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, true, false)
end
@generated function vnoaliasstore!(ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}) where {AU,F,N,AV,W,M,I,T,D,Nm1}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, false, true)
end
@generated function vnoaliasstore!(ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}, m::Mask{W}) where {AU,F,N,AV,W,M,I,T,D,Nm1}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, true, true)
end

# @inline vstore!(::typeof(identity), ptr, v, u) = vstore!(ptr, v, u)
# @inline vstore!(::typeof(identity), ptr, v, u, m) = vstore!(ptr, v, u, m)
# @inline vnoaliasstore!(::typeof(identity), ptr, v, u) = vnoaliasstore!(ptr, v, u)
# @inline vnoaliasstore!(::typeof(identity), ptr, v, u, m) = vnoaliasstore!(ptr, v, u, m)

@generated function vstore!(
    ::Function, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}, m
) where {T,D,C,U,AU,F,N,W,M,I,AV}
    @assert N == U + 1
    # mask means it isn't vectorized
    @assert AV > 0 "AV ≤ 0, but masking what, exactly?"
    Expr(:block, Expr(:meta, :inline), :(vstore!(ptr, vu, u, m)))
end
@generated function vnoaliasstore!(
    ::Function, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}, m
) where {T,D,C,U,AU,F,N,W,M,I,AV}
    @assert N == U + 1
    # mask means it isn't vectorized
    @assert AV > 0 "AV ≤ 0, but masking what, exactly?"
    Expr(:block, Expr(:meta, :inline), :(vnoaliasstore!(ptr, vu, u, m)))
end

function transposeshuffle(split, W, offset::Bool)
    tup = Expr(:tuple)
    w = 0
    S = 1 << split
    i = offset ? S : 0
    while w < W
        for s ∈ 0:S-1
            push!(tup.args, w + s + i)
        end
        for s ∈ 0:S-1
            # push!(tup.args, w + W + s)
            push!(tup.args, w + W + s + i)
        end
        w += 2S
    end
    Expr(:call, Expr(:curly, :Val, tup))
end

function horizonal_reduce_store_expr(W, Ntotal, (C,D,AU,F), op = :+, reduct = :vsum, noalias::Bool = false)
    N = ((C == AU) && isone(F)) ? prevpow2(Ntotal) : 0
    q = Expr(:block, Expr(:meta, :inline), :(v = data(vu)))
    store = noalias ? :vnoaliasstore! : :vstore!
    @assert ispow2(W)
    if N > 1
        if N < Ntotal
            push!(q.args, :(gptr = gesp(ptr, u.i)))
            push!(q.args, :(bptr = pointer(gptr)))
        else
            push!(q.args, :(bptr = gep(ptr, u.i)))
        end
        extractblock = Expr(:block)
        vectors = [Symbol(:v_, n) for n ∈ 0:N-1]
        for n ∈ 1:N
            push!(extractblock.args, Expr(:(=), vectors[n], Expr(:ref, :v, n)))
        end
        push!(q.args, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), extractblock))
        ncomp = 0
        minWN = min(W,N)
        while ncomp < N
            Nt = minWN;
            Wt = W
            splits = 0
            while Nt > 1
                Nt >>>= 1
                shuffle0 = transposeshuffle(splits, Wt, false)
                shuffle1 = transposeshuffle(splits, Wt, true)
                splits += 1
                for nh ∈ 1:Nt
                    n1 = 2nh
                    n0 = n1 - 1
                    v0 = vectors[n0 + ncomp]; v1 = vectors[n1 + ncomp]; vh = vectors[nh + ncomp];
                    # combine n0 and n1
                    push!(q.args, Expr(
                        :(=), vh, Expr(
                            :call, op,
                            Expr(:call, :shufflevector, v0, v1, shuffle0),
                            Expr(:call, :shufflevector, v0, v1, shuffle1))
                    ))
                end
            end
            # v0 is now the only vector
            v0 = vectors[ncomp + 1]
            while Wt > minWN
                Wh = Wt >>> 1
                v0new = Symbol(v0, Wt)
                push!(q.args, Expr(
                    :(=), v0new, Expr(
                        :call, op,
                        Expr(:call, :shufflevector, v0, Expr(:call, Expr(:curly, :Val, Expr(:tuple, [w for w ∈ 0:Wh-1]...)))),
                        Expr(:call, :shufflevector, v0, Expr(:call, Expr(:curly, :Val, Expr(:tuple, [w for w ∈ Wh:Wt-1]...)))))
                )
                      )
                v0 = v0new
                Wt = Wh
            end
            push!(q.args, Expr(:call, store, :bptr, v0))
            ncomp += minWN
        end
    else
        push!(q.args, :(gptr = gesp(ptr, u.i)))
    end
    if N < Ntotal
        zeroexpr = Expr(:call, Expr(:curly, :StaticInt, 0))
        ind = Expr(:tuple); foreach(_ -> push!(ind.args, zeroexpr), 1:D)
        for n ∈ N+1:Ntotal
            (n > N+1) && (ind = copy(ind)) # copy to avoid overwriting old
            ind.args[AU] = Expr(:call, Expr(:curly, :StaticInt, F*(n-1)))
            scalar = Expr(:call, reduct, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :v, n)))
            push!(q.args, Expr(:call, store, :gptr, scalar, ind))
        end
    end
    q
end
@generated function vstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV}
    @assert N == U + 1
    if G === typeof(identity) || AV > 0
        return Expr(:block, Expr(:meta, :inline), :(vstore!(ptr, vu, u)))
    elseif G === typeof(vsum)
        op = :+; reduct = :vsum
    elseif G === typeof(vprod)
        op = :*; reduct = :vprod
    else
        throw("Function $f not recognized.")
    end
    horizonal_reduce_store_expr(W, N, (C,D,AU,F), op, reduct, false)
end
@generated function vnoaliasstore!(
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV}
    @assert N == U + 1
    if G === typeof(identity) || AV > 0
        return Expr(:block, Expr(:meta, :inline), :(vnoaliasstore!(ptr, vu, u)))
    elseif G === typeof(vsum)
        op = :+; reduct = :vsum
    elseif G === typeof(vprod)
        op = :*; reduct = :vprod
    else
        throw("Function $f not recognized.")
    end
    horizonal_reduce_store_expr(W, N, (C,D,AU,F), op, reduct, true)
end


# """
# O - An `NTuple{M,NTuple{N,Int}}` tuple of tuples, specifies offsets of `N`-dim array for each of `M` loads.
# u::U - the base unrolled description.
# """
# struct MultiLoad{O,U,V,F,W,M}
#     u::Unroll{U,V,F,W,M}
# end


for locality ∈ 0:3, readorwrite ∈ 0:1
    instrs = """
        %addr = inttoptr $JULIAPOINTERTYPE %0 to i8*
        call void @llvm.prefetch(i8* %addr, i32 $readorwrite, i32 $locality, i32 1)
        ret void
    """
    @eval @inline function prefetch(ptr::Ptr{Cvoid}, ::Val{$locality}, ::Val{$readorwrite})
        llvmcall(("declare void @llvm.prefetch(i8*, i32, i32, i32)",$instrs), Cvoid, Tuple{Ptr{Cvoid}}, ptr)
    end
end
@inline prefetch(ptr::Ptr{T}, ::Val{L}, ::Val{R}) where {T,L,R} = prefetch(Base.unsafe_convert(Ptr{Cvoid}, ptr), Val{L}(), Val{R}())

@inline function prefetch(ptr::Union{AbstractStridedPointer,Ptr}, i, ::Val{Locality}, ::Val{ReadOrWrite}) where {Locality, ReadOrWrite}
    prefetch(gep(ptr, i), Val{Locality}(), Val{ReadOrWrite}())
end
@inline prefetch(ptr::Ptr) = prefetch(ptr, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, ::Val{L}) where {L} = prefetch(ptr, Val{L}(), Val{0}())
@inline prefetch(ptr::Ptr, i) = prefetch(ptr, i, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, i, ::Val{L}) where {L} = prefetch(ptr, i, Val{L}(), Val{0}())


@inline prefetch0(x, i) = prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{3}(), Val{0}())
@inline prefetch0(x, I::Tuple) = prefetch(gep(stridedpointer(x), extract_data.(I)), Val{3}(), Val{0}())
@inline prefetch0(x, i, j) = prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{3}(), Val{0}())
@inline prefetch1(x, i) = prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{2}(), Val{0}())
@inline prefetch1(x, i, j) = prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{2}(), Val{0}())
@inline prefetch2(x, i) = prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) = prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{1}(), Val{0}())

@generated function lifetime_start!(ptr::Ptr{T}, ::Val{L}) where {L,T}
    decl = "declare void @llvm.lifetime.start(i64, i8* nocapture)"
    instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to i8*\ncall void @llvm.lifetime.start(i64 $(L*sizeof(T)), i8* %ptr)\nret void"
    llvmcall_expr(decl, instrs, :Cvoid, :(Tuple{Ptr{$T}}), "void", [JULIAPOINTERTYPE], [:ptr])
end
@generated function lifetime_end!(ptr::Ptr{T}, ::Val{L}) where {L,T}
    decl = "declare void @llvm.lifetime.end(i64, i8* nocapture)"
    instrs = "%ptr = inttoptr $JULIAPOINTERTYPE %0 to i8*\ncall void @llvm.lifetime.end(i64 $(L*sizeof(T)), i8* %ptr)\nret void"
    llvmcall_expr(decl, instrs, :Cvoid, :(Tuple{Ptr{$T}}), "void", [JULIAPOINTERTYPE], [:ptr])
end

@inline lifetime_start!(ptr::Ptr) = lifetime_start!(ptr, Val{-1}())
@inline lifetime_end!(ptr::Ptr) = lifetime_end!(ptr, Val{-1}())
# Fallback is to do nothing. Intention is (e.g.) for PaddedMatrices/StackPointers.
@inline lifetime_start!(::Any) = nothing
@inline lifetime_end!(::Any) = nothing

@generated function compressstore!(ptr::Ptr{T}, v::Vec{W,T}, mask::Mask{W,U}) where {W,T <: NativeTypes, U<:Unsigned}
    @assert 8sizeof(U) >= W
    typ = LLVM_TYPES[T]
    vtyp = "<$W x $typ>"
    mtyp_input = LLVM_TYPES[U]
    mtyp_trunc = "i$W"
    instrs = String["%ptr = inttoptr $JULIAPOINTERTYPE %1 to $typ*"]
    truncate_mask!(instrs, '2', W, 0)
    decl = "declare void @llvm.masked.compressstore.$(suffix(W,T))($vtyp, $typ*, <$W x i1>)"
    push!(instrs, "call void @llvm.masked.compressstore.$(suffix(W,T))($vtyp %0, $typ* %ptr, <$W x i1> %mask.0)\nret void")
    llvmcall_expr(decl, join(instrs,"\n"), :Cvoid, :(Tuple{NTuple{$W,VecElement{$T}}, Ptr{$T}, $U}), "void", [vtyp, JULIAPOINTERTYPE, "i$(8sizeof(U))"], [:(data(v)), :ptr, :(data(mask))])
end

@generated function expandload(ptr::Ptr{T}, mask::Mask{W,U}) where {W, T <: NativeTypes, U<:Unsigned}
    @assert 8sizeof(U) >= W
    typ = LLVM_TYPES[T]
    vtyp = "<$W x $typ>"
    vptrtyp = "<$W x $typ*>"
    mtyp_input = LLVM_TYPES[U]
    mtyp_trunc = "i$W"
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $JULIAPOINTERTYPE %0 to $typ*")
    if mtyp_input == mtyp_trunc
        push!(instrs, "%mask = bitcast $mtyp_input %1 to <$W x i1>")
    else
        push!(instrs, "%masktrunc = trunc $mtyp_input %1 to $mtyp_trunc")
        push!(instrs, "%mask = bitcast $mtyp_trunc %masktrunc to <$W x i1>")
    end
    decl = "declare $vtyp @llvm.masked.expandload.$(suffix(W,T))($typ*, <$W x i1>, $vtyp)"
    push!(instrs, "%res = call $vtyp @llvm.masked.expandload.$(suffix(W,T))($typ* %ptr, <$W x i1> %mask, $vtyp zeroinitializer)\nret $vtyp %res")
    llvmcall_expr(decl, join(instrs,"\n"), :(NTuple{$W,VecElement{$T}}), :(Tuple{Ptr{$T}, $U}), vtyp, [JULIAPOINTERTYPE, "i$(8sizeof(U))"], [:ptr, :(data(mask))])
end



