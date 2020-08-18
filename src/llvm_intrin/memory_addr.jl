
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

function vload_quote(::Type{T}, W = 1) where {T <: NativeTypes}
    typ = vtype(W, LLVM_TYPE[T])
    decl = LOAD_SCOPE_TBAA
    alignment = Base.datatype_alignment(T)
    instrs = """
        %ptr = inttoptr $JULIAPOINTERTYPE %0 to $typ*
        %res = load $typ, $typ* %ptr, align $alignment, !alias.scope !3, !tbaa !5
        ret $typ %res
    """
    :(llvmcall(($decl,$instrs), $T, Tuple{Ptr{$T}}, ptr))
end
function vload_quote(::Type{T}, ::Type{I}, W = 1, ivec = false) where {T <: NativeTypes, I <: Integer}
    ityp = vtype(ivec ? W : 1, 'i' * string(8sizeof(I)))
    typ = LLVM_TYPE[T]
    vtyp = vtype(W, typ)
    decl = LOAD_SCOPE_TBAA
    alignment = Base.datatype_alignment(T)
    if ivec
        instr = "@llvm.masked.gather." * suffix(W, T)
        decl *= "\ndeclare $typ $instr(<$W x $typ*>, i32, <$W x i1>, $vtyp)"
        
    else
    end
    # push!(flags, "!noalias !0")
    instrs = """
        %typptr = inttoptr $JULIAPOINTERTYPE %0 to i8*
        %iptr = getelementptr inbounds i8, i8* %typptr, $ityp %1
        %ptr = bitcast i8* %iptr to $typ*
        %res = load $typ, $typ* %ptr, align $alignment, !alias.scope !3, !tbaa !5
        ret $typ %res
    """
    :(llvmcall($((decl, join(instrs, "\n"))), $T, Tuple{Ptr{$T}, $I}, ptr, i))
end
function vstore_quote(::Type{T}, alias) where {T <: NativeTypes}
    ptyp = JULIAPOINTERTYPE
    typ = LLVM_TYPE[T]
    instrs = String[]
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    decl = alias ? STORE_TBAA : SCOPE_METADATA * STORE_TBAA
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    aliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !tbaa !7"
    noaliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !noalias !3, !tbaa !7"
    push!(instrs, alias ? aliasstoreinstr : noaliasstoreinstr)
    push!(instrs, "ret void")
    :(llvmcall($((decl, join(instrs, "\n"))), Cvoid, Tuple{Ptr{$T}, $T}, ptr, v))
end
function vstore_quote(::Type{T}, ::Type{I}, alias) where {T <: NativeTypes, I <: Integer}
    ityp = 'i' * string(8sizeof(I))
    ptyp = JULIAPOINTERTYPE
    typ = LLVM_TYPE[T]
    instrs = String[]
    decl = alias ? STORE_TBAA : SCOPE_METADATA * STORE_TBAA
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%typptr = inttoptr $ptyp %0 to i8*")
    push!(instrs, "%iptr = getelementptr inbounds i8, i8* %typptr, $ityp %2")
    push!(instrs, "%ptr = bitcast i8* %iptr to $typ*")
    aliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !tbaa !7"
    noaliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !noalias !3, !tbaa !7"
    push!(instrs, alias ? aliasstoreinstr : noaliasstoreinstr)
    push!(instrs, "ret void")
    :(llvmcall($((decl,join(instrs, "\n"))), Cvoid, Tuple{Ptr{$T}, $T, $I}, ptr, v, i))
end

function gepquote(::Type{T}, ::Type{I}, byte::Bool) where {T <: NativeTypes, I <: Integer}
    ptyp = JULIAPOINTERTYPE
    ityp = llvmtype(I)
    typ = byte ? "i8" : LLVM_TYPE[T]
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%offsetptr = getelementptr inbounds $typ, $typ* %ptr, $ityp %1")
    push!(instrs, "%iptr = ptrtoint $typ* %offsetptr to $ptyp")
    push!(instrs, "ret $ptyp %iptr")
    quote
        llvmcall(
            $(join(instrs, "\n")),
            Ptr{$T}, Tuple{Ptr{$T}, $I},
            ptr, i
        )
    end    
end





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
@inline prefetch(ptr::Ptr{T}, ::Val{L}, ::Val{R}) = where {T,L,R} = prefetch(Base.unsafe_convert(Ptr{Cvoid}, ptr), Val{L}(), Val{R}())

@inline function prefetch(ptr::Union{VectorizationBase.AbstractStridedPointer,Ptr}, i, ::Val{Locality}, ::Val{ReadOrWrite}) where {Locality, ReadOrWrite}
    prefetch(gep(ptr, i), Val{Locality}(), Val{ReadOrWrite}())
end
@inline prefetch(ptr::Ptr) = prefetch(ptr, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, ::Val{L}) where {L} = prefetch(ptr, Val{L}(), Val{0}())
@inline prefetch(ptr::Ptr, i) = prefetch(ptr, i, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, i, ::Val{L}) where {L} = prefetch(ptr, i, Val{L}(), Val{0}())


@inline prefetch0(x, i) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{3}(), Val{0}())
@inline prefetch0(x, I::Tuple) = SIMDPirates.prefetch(gep(stridedpointer(x), extract_data.(I)), Val{3}(), Val{0}())
@inline prefetch0(x, i, j) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j, oi, oj) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{3}(), Val{0}())
@inline prefetch1(x, i) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{2}(), Val{0}())
@inline prefetch1(x, i, j) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j, oi, oj) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{2}(), Val{0}())
@inline prefetch2(x, i) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i),)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i), extract_data(j))), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j, oi, oj) = SIMDPirates.prefetch(gep(stridedpointer(x), (extract_data(i) + extract_data(oi) - 1, extract_data(j) + extract_data(oj) - 1)), Val{1}(), Val{0}())

