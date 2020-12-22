
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

const VectorIndexCore{W} = Union{Vec{W},MM{W},Unroll{<:Any,<:Any,<:Any,<:Any,W}}
const VectorIndex{W} = Union{VectorIndexCore{W},LazyMulAdd{<:Any,<:Any,<:VectorIndexCore{W}}}

# const BoolVec = Union{Mask,VecUnroll{<:Any, <:Any, Bool, <: Mask}}

const SCOPE_METADATA = """
!1 = !{!\"noaliasdomain\"}
!2 = !{!\"noaliasscope\", !1}
!3 = !{!2}
"""
const SCOPE_FLAGS = ", !alias.scope !3";

const USE_TBAA = false
# use TBAA?
# define: LOAD_SCOPE_TBAA, LOAD_SCOPE_TBAA_FLAGS, SCOPE_METADATA, STORE_TBAA, SCOPE_FLAGS, STORE_TBAA_FLAGS
let 
    LOAD_TBAA = """
    !4 = !{!"jtbaa"}
    !5 = !{!6, !6, i64 0, i64 0}
    !6 = !{!"jtbaa_arraybuf", !4, i64 0}
    """;
    LOAD_TBAA_FLAGS = ", !tbaa !5";
    global const LOAD_SCOPE_TBAA = USE_TBAA ? SCOPE_METADATA * LOAD_SCOPE_TBAA : SCOPE_METADATA;
    global const LOAD_SCOPE_TBAA_FLAGS = USE_TBAA ? SCOPE_FLAGS * LOAD_TBAA_FLAGS : SCOPE_FLAGS
        
    global const STORE_TBAA = USE_TBAA ? """
    !4 = !{!"jtbaa", !5, i64 0}
    !5 = !{!"jtbaa"}
    !6 = !{!"jtbaa_data", !4, i64 0}
    !7 = !{!8, !8, i64 0}
    !8 = !{!"jtbaa_arraybuf", !6, i64 0}
    """ : ""
    global const STORE_TBAA_FLAGS = USE_TBAA ? ", !tbaa !7" : ""
end

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
    isbit = T === Bit
    Morig = M
    if isbit
        sizeof_T = 1
        typ = "i1"
        vtyp = isone(W) ? typ : "<$W x i1>"
        M = max(1, M >> 3)
        O >>= 3
        @assert ((isone(X) | iszero(X)) && (ind_type !== :Vec)) "indexing BitArrays with a vector not currently supported."
    else
        sizeof_T = sizeof(T)
        typ = LLVM_TYPES[T]
        vtyp = vtype(W, typ) # vtyp is dest type
    end
    instrs = String[]
    if M == 0
        ind_type = :StaticInt
    elseif ind_type === :StaticInt
        M = 0
    end
    if isone(W)
        X = 1
    end
    if iszero(M)
        tz = intlog2(sizeof_T)
        tzf = sizeof_T
        index_gep_typ = typ
    else
        tz = min(trailing_zeros(M), 3)
        tzf = 1 << tz
        index_gep_typ = ((tzf == sizeof_T) | iszero(M)) ? typ : "i$(tzf << 3)"
        M >>= tz
    end
    # after this block, we will have a index_gep_typ pointer
    if iszero(O)
        push!(instrs, "%ptr.$(i) = inttoptr $(JULIAPOINTERTYPE) %0 to $(index_gep_typ)*"); i += 1
    else # !iszero(O)
        if !iszero(O & (tzf - 1)) # then index_gep_typ works for the constant offset
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
            push!(instrs, "%$(indname) = mul nsw <$W x i$(ibits)> %$(indargname), $(constmul)")
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
        if isbit
            scale = 8
            if ispow2(Morig)
                if abs(Morig) ≥ 8
                    M = Morig >> 3
                    if M != 1
                        indname = "indname"
                        push!(instrs, "%$(indname) = mul nsw i$(ibits) %$(indargname), $M")
                    else
                        indname = indargname
                    end
                else
                    shifter = 3 - intlog2(Morig)
                    push!(instrs, "%shiftedind = ashr i$(ibits) %$(indargname), $shifter")
                    if Morig > 0
                        indname = "shiftedind"
                    else
                        indname = "indname"
                        push!(instrs, "%$(indname) = mul i$(ibits) %shiftedind, -1")
                    end
                end
            else
                @assert iszero(Morig) "Scale factors on bit accesses must be 0 or a power of 2."
                indname = "0"
            end
        else
            if isone(M)
                indname = indargname
            elseif iszero(M)
                indname = "0"
            else
                indname = "indname"
                push!(instrs, "%$(indname) = mul nsw i$(ibits) %$(indargname), $M")
            end
            # TODO: if X != 1 and X != 0, check if it is better to gep -> gep, or broadcast -> add -> gep
        end
        push!(instrs, "%ptr.$(i) = getelementptr inbounds $(index_gep_typ), $(index_gep_typ)* %ptr.$(i-1), i$(ibits) %$(indname)"); i += 1
    end
    # ind_type === :Integer || ind_type === :StaticInt
    if !(isone(X) | iszero(X)) # vec
        vibytes = min(4, REGISTER_SIZE ÷ W)
        vityp = "i$(8vibytes)"
        vi = join((X*w for w ∈ 0:W-1), ", $vityp ")
        if typ !== index_gep_typ
            push!(instrs, "%ptr.$(i) = bitcast $(index_gep_typ)* %ptr.$(i-1) to $(typ)*"); i += 1
        end
        push!(instrs, "%ptr.$(i) = getelementptr inbounds $(typ), $(typ)* %ptr.$(i-1), <$W x $(vityp)> <$vityp $vi>"); i += 1
        if forgep
            push!(instrs, "%ptr.$(i) = ptrtoint <$W x $typ*> %ptr.$(i-1) to <$W x $JULIAPOINTERTYPE>"); i += 1
        end
        return instrs, i
    end
    if forgep # if forgep, just return now
        push!(instrs, "%ptr.$(i) = ptrtoint $(index_gep_typ)* %ptr.$(i-1) to $JULIAPOINTERTYPE"); i += 1
    elseif index_gep_typ != vtyp
        push!(instrs, "%ptr.$(i) = bitcast $(index_gep_typ)* %ptr.$(i-1) to $(vtyp)*"); i += 1
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

@generated function gep(ptr::Ptr{T}, i::I) where {I <: Integer, T <: NativeTypes}
    gep_quote(T, :Integer, I, 1, 1, 1, 0, true)
end
@generated function gep(ptr::Ptr{T}, ::StaticInt{N}) where {N, T <: NativeTypes}
    gep_quote(T, :StaticInt, Int, 1, 1, 0, N, true)
end
@generated function gep(ptr::Ptr{T}, i::LazyMulAdd{M,O,I}) where {T <: NativeTypes, I <: Integer, O, M}
    gep_quote(T, :Integer, I, 1, 1, M, O, true)
end
@generated function gep(ptr::Ptr{T}, i::Vec{W,I}) where {W, T <: NativeTypes, I <: Integer}
    gep_quote(T, :Vec, I, W, 1, 1, 0, true)
end
@generated function gep(ptr::Ptr{T}, i::LazyMulAdd{M,O,Vec{W,I}}) where {W, T <: NativeTypes, I <: Integer, M, O}
    gep_quote(T, :Vec, I, W, 1, M, O, true)
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
    # considers booleans to only occupy 1 bit in memory, so they must be handled specially
    isbit = T === Bit
    if isbit
        # @assert !grv "gather's not are supported with `BitArray`s."
        mask = false # TODO: not this?
        jtyp = isone(W) ? Bool : mask_type(W)
        # typ = "i$W"
        typ = "i1"
    else
        jtyp = isone(W) ? T : _Vec{W,T}
        typ = LLVM_TYPES[T]
    end
    alignment = (align & (!grv)) ? Base.datatype_alignment(jtyp) : Base.datatype_alignment(T)
        
    decl = LOAD_SCOPE_TBAA
    dynamic_index = !(iszero(M) || ind_type === :StaticInt)

    vtyp = vtype(W, typ)
    mask && truncate_mask!(instrs, '1' + dynamic_index, W, 0)
    if grv
        loadinstr = "$vtyp @llvm.masked.gather." * suffix(W, T) * '.' * suffix(W, Ptr{T})
        decl *= "declare $loadinstr(<$W x $typ*>, i32, <$W x i1>, $vtyp)"
        m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
        passthrough = mask ? "zeroinitializer" : "undef"
        push!(instrs, "%res = call $loadinstr(<$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m, $vtyp $passthrough)" * LOAD_SCOPE_TBAA_FLAGS)
    elseif mask
        suff = suffix(W, T)
        loadinstr = "$vtyp @llvm.masked.load." * suff * ".p0" * suff
        decl *= "declare $loadinstr($vtyp*, i32, <$W x i1>, $vtyp)"
        push!(instrs, "%res = call $loadinstr($vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0, $vtyp zeroinitializer)" * LOAD_SCOPE_TBAA_FLAGS)
    else
        push!(instrs, "%res = load $vtyp, $vtyp* %ptr.$(i-1), align $alignment" * LOAD_SCOPE_TBAA_FLAGS)
    end
    if isbit
        lret = string('i', max(8,W))
        if W > 1
            if W < 8
                push!(instrs, "%resint = bitcast <$W x i1> %res to i$(W)")
                push!(instrs, "%resfinal = zext i$(W) %resint to i8")
            else
                push!(instrs, "%resfinal = bitcast <$W x i1> %res to i$(W)")
            end
        else
            push!(instrs, "%resfinal = zext i1 %res to i8")
        end
        push!(instrs, "ret $lret %resfinal")
    else
        lret = vtyp
        push!(instrs, "ret $vtyp %res")
    end
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
    if isbit && W > 1
        quote
            $(Expr(:meta,:inline))
            Mask{$W}($(llvmcall_expr(decl, join(instrs, "\n"), ret, args, lret, largs, arg_syms)))
        end
    else
        llvmcall_expr(decl, join(instrs, "\n"), ret, args, lret, largs, arg_syms)
    end
end
# vload_quote(T, ::Type{I}, ind_type::Symbol, W::Int, X, M, O, mask, align = false)
@generated function vload(ptr::Ptr{T}, ::Val{A}) where {T <: NativeTypes, A}
    vload_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, A)
end
@generated function vload(ptr::Ptr{T}, i::I, ::Val{A}) where {A, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, 1, 1, 1, 0, false, A)
end
@generated function vload(ptr::Ptr{T}, ::StaticInt{N}, ::Val{A}) where {A, N, T <: NativeTypes}
    vload_quote(T, Int, :StaticInt, 1, 1, 0, N, false, A)
end
@generated function vload(ptr::Ptr{T}, i::Vec{W,I}, ::Val{A}) where {A, W, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Vec, W, 1, 1, 0, false, A)
end
@generated function vload(ptr::Ptr{T}, i::MM{W,X,I}, ::Val{A}) where {A, W, X, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, W, X, 1, 0, false, A)
end
@generated function vload(ptr::Ptr{T}, ::MM{W,X,StaticInt{N}}, ::Val{A}) where {A, W, X, T <: NativeTypes, N}
    vload_quote(T, Int, :StaticInt, W, X, 1, N, false, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,I}, ::Val{A}) where {A, M, O, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, 1, 1, M, O, false, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,Vec{W,I}}, ::Val{A}) where {A, W, M, O, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Vec, W, 1, M, O, false, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,I}}, ::Val{A}) where {A, W, M, O, X, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, W, X*M, M, O, false, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, ::Val{A}) where {A, W, M, O, X, T <: NativeTypes, I}
    vload_quote(T, Int, :StaticInt, W, X*M, M, O + I*M, false, A)
end

@generated function vload(ptr::Ptr{T}, i::Vec{W,I}, m::Mask{W}, ::Val{A}) where {A, W, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Vec, W, 1, 1, 0, true, A)
end
@generated function vload(ptr::Ptr{T}, i::MM{W,X,I}, m::Mask{W}, ::Val{A}) where {A, W, X, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, W, X, 1, 0, true, A)
end
@generated function vload(ptr::Ptr{T}, ::MM{W,X,StaticInt{N}}, m::Mask{W}, ::Val{A}) where {A, W, X, T <: NativeTypes, N}
    vload_quote(T, Int, :StaticInt, W, X, 1, N, true, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,Vec{W,I}}, m::Mask{W}, ::Val{A}) where {A, W, M, O, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Vec, W, 1, M, O, true, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,I}}, m::Mask{W}, ::Val{A}) where {A, W, M, O, X, T <: NativeTypes, I <: IntegerTypes}
    vload_quote(T, I, :Integer, W, X*M, M, O, true, A)
end
@generated function vload(ptr::Ptr{T}, i::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, m::Mask{W}, ::Val{A}) where {A, W, M, O, X, T <: NativeTypes, I}
    vload_quote(T, Int, :StaticInt, W, X*M, M, O + I*M, true, A)
end


@inline function _vload_scalar(ptr::Ptr{Bit}, i::Integer, ::Val{A}) where {A}
    d = i >> 3; r = i & 7;
    u = vload(Base.unsafe_convert(Ptr{UInt8}, ptr), d, Val{A}())
    (u >> r) % Bool
end
@inline vload(ptr::Ptr{Bit}, i::IntegerTypesHW, ::Val{A}) where {A} = _vload_scalar(ptr, i, Val{A}())
# avoid ambiguities
@inline vload(ptr::Ptr{Bit}, ::StaticInt{N}, ::Val{A}) where {A,N} = _vload_scalar(ptr, StaticInt{N}(), Val{A}())


@inline vload(ptr::Union{Ptr,AbstractStridedPointer}) = vload(ptr, Val{false}())
@inline vloada(ptr::Union{Ptr,AbstractStridedPointer}) = vload(ptr, Val{true}())
@inline vload(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}) = vload(ptr, i, Val{false}())
@inline vloada(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}) = vload(ptr, i, Val{true}())
@inline vload(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}, m::Mask) = vload(ptr, i, m, Val{false}())
@inline vloada(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}, m::Mask) = vload(ptr, i, m, Val{true}())
@inline vload(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}, b::Bool) = vload(ptr, i, b, Val{false}())
@inline vloada(ptr::Union{Ptr,AbstractStridedPointer}, i::Union{Number,Tuple,Unroll}, b::Bool) = vload(ptr, i, b, Val{true}())

@inline function vload(ptr::Ptr{T}, i::Number, b::Bool, ::Val{A}) where {T,A}
    if b
        vload(ptr, i, Val{A}())
    else
        zero(T)
    end
end
@inline vwidth_from_ind(i::Tuple) = vwidth_from_ind(i, StaticInt(1), StaticInt(0))
@inline vwidth_from_ind(i::Tuple{}, ::StaticInt{W}, ::StaticInt{U}) where {W,U} = (StaticInt{W}(), StaticInt{U}())
@inline function vwidth_from_ind(i::Tuple{<:AbstractSIMDVector{W},Vararg}, ::Union{StaticInt{1},StaticInt{W}}, ::StaticInt{U}) where {W,U}
    vwidth_from_ind(Base.tail(i), StaticInt{W}(), StaticInt{W}(U))
end
@inline function vwidth_from_ind(
    i::Tuple{<:VecUnroll{U,W},Vararg}, ::Union{StaticInt{1},StaticInt{W}}, ::Union{StaticInt{0},StaticInt{U}}
) where {U,W}
    vwidth_from_ind(Base.tail(i), StaticInt{W}(), StaticInt{W}(U))
end
@inline zero_init(::Type{T}, ::StaticInt{1}, ::StaticInt{0}) where {T} = zero(T)
@inline zero_init(::Type{T}, ::StaticInt{W}, ::StaticInt{0}) where {W,T} = vzero(Val(W), T)
@inline zero_init(::Type{T}, ::StaticInt{W}, ::StaticInt{U}) where {W,U,T} = zero(VecUnroll{U,W,T,Vec{W,T}})

@inline zero_init(::Type{T}, ::Tuple{StaticInt{W},StaticInt{U}}) where {W,U,T} = zero_init(T, StaticInt{W}(), StaticInt{U}())

@inline function vload(ptr::Ptr{T}, i::Tuple, b::Bool, ::Val{A}) where {T,A}
    if b
        vload(ptr, i, Val{A}())
    else
        zero_init(T, vwidth_from_ind(i))
    end
end
@generated function zero_vecunroll(::Val{N}, ::Val{W}, ::Type{T}) where {N,W,T}
    Expr(:block, Expr(:meta, :inline), :(zero(VecUnroll{$(N-1),$W,$T,Vec{$W,$T}})))
end
@inline function vload(ptr::Ptr{T}, i::Unroll{AU,F,N,AV,W,M,I}, b::Bool, ::Val{A}) where {T,AU,F,N,AV,W,M,I,A}
    if b
        vload(ptr, i, Val{A}())
    else
        zero_vecunroll(Val{N}(), Val{W}(), T)
        # VecUnroll(ntuple(@inline(_ -> vzero(Val{W}(), T)), Val{N}()))
    end
end

function vstore_quote(
    ::Type{T}, ::Type{I}, ind_type::Symbol, W::Int, X::Int, M::Int, O::Int, mask::Bool, align::Bool, noalias::Bool, nontemporal::Bool
) where {T <: NativeTypes, I <: Integer}
    ibits = 8sizeof(I)
    if W > 1 && ind_type !== :Vec
        X, Xr = divrem(X, sizeof(T))
        @assert iszero(Xr)
    end
    instrs, i = offset_ptr(T, ind_type, '2', ibits, W, X, M, O, false)
    
    grv = gep_returns_vector(W, X, M, ind_type)
    jtyp = isone(W) ? T : _Vec{W,T}

    align != nontemporal # should I do this?
    alignment = (align & (!grv)) ? Base.datatype_alignment(jtyp) : Base.datatype_alignment(T)

    decl = noalias ? SCOPE_METADATA * STORE_TBAA : STORE_TBAA
    metadata = noalias ? SCOPE_FLAGS * STORE_TBAA_FLAGS : STORE_TBAA_FLAGS
    dynamic_index = !(iszero(M) || ind_type === :StaticInt)

    typ = LLVM_TYPES[T]
    lret = vtyp = vtype(W, typ)
    mask && truncate_mask!(instrs, '2' + dynamic_index, W, 0)
    if grv
        storeinstr = "void @llvm.masked.scatter." * suffix(W, T) * '.' * suffix(W, Ptr{T})
        decl *= "declare $storeinstr($vtyp, <$W x $typ*>, i32, <$W x i1>)"
        m = mask ? m = "%mask.0" : llvmconst(W, "i1 1")
        push!(instrs, "call $storeinstr($vtyp %1, <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m)" * metadata)
        # push!(instrs, "call $storeinstr($vtyp %1, <$W x $typ*> %ptr.$(i-1), i32 $alignment, <$W x i1> $m)")
    elseif mask
        suff = suffix(W, T)
        storeinstr = "void @llvm.masked.store." * suff * ".p0" * suff
        decl *= "declare $storeinstr($vtyp, $vtyp*, i32, <$W x i1>)"
        push!(instrs, "call $storeinstr($vtyp %1, $vtyp* %ptr.$(i-1), i32 $alignment, <$W x i1> %mask.0)" * metadata)
    elseif nontemporal
        push!(instrs, "store $vtyp %1, $vtyp* %ptr.$(i-1), align $alignment, !nontemporal !{i32 1}" * metadata)
    else
        push!(instrs, "store $vtyp %1, $vtyp* %ptr.$(i-1), align $alignment" * metadata)
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


@generated function vstore!(
    ptr::Ptr{T}, v::T, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T <: NativeTypes, A, S, NT}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::T, i::I, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, 1, 1, 1, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::T, ::StaticInt{N}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T <: NativeTypes, N, A, S, NT}
    vstore_quote(T, Int, :StaticInt, 1, 1, 0, N, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, X, 1, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::I, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, 0, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, N, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, N, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::T, i::LazyMulAdd{M,O,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, 1, 1, M, O, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Vec, W, 1, M, O, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, X*M, M, O, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, M, O, I, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, X*M, M, O + M*I, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, false, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::Vec{W,I}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Vec, W, 1, 1, 0, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::MM{W,X,I}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, X, 1, 0, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::I, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, sizeof(T), 1, 0, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, ::StaticInt{N}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, N, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, sizeof(T), 0, N, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, ::MM{W,X,StaticInt{N}}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, N, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, X, 0, N, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,Vec{W,I}}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Vec, W, 1, M, O, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,I}}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, X*M, M, O, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,MM{W,X,StaticInt{I}}}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, X, T <: NativeTypes, M, O, I, A, S, NT}
    vstore_quote(T, Int, :StaticInt, W, X*M, M, O + M*I, true, A, S, NT)
end
@generated function vstore!(
    ptr::Ptr{T}, v::Vec{W,T}, i::LazyMulAdd{M,O,I}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T <: NativeTypes, M, O, I <: IntegerTypes, A, S, NT}
    vstore_quote(T, I, :Integer, W, sizeof(T), M, O, true, A, S, NT)
end

# broadcasting scalar stores
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::VectorIndex{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T, A, S, NT}
    vstore!(ptr, convert(Vec{W,T}, v), i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::VectorIndex{W}, m::Mask, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T, A, S, NT}
    vstore!(ptr, convert(Vec{W,T}, v), i, m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::MM{1}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT}
    vstore!(ptr, convert(T, v), i.i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::MM{1}, m::Mask{1}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT}
    Bool(m) && vstore!(ptr, convert(T, v), i.i, Val{A}(), Val{S}(), Val{NT}())
    nothing
end
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::LazyMulAdd{M,O,MM{1,X,I}}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT, M,O,X,I}
    vstore!(ptr, convert(T, v), _materialize(i).i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v::Base.HWReal, i::LazyMulAdd{M,O,MM{1,X,I}}, m::Mask{1}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT, M,O,X,I}
    Bool(m) && vstore!(ptr, convert(T, v), _materialize(i).i, Val{A}(), Val{S}(), Val{NT}())
    nothing
end



# @inline function vstore!(ptr::Ptr{Bit}, m::Mask{W,U}, i, ::Val{A}, ::Val{S}, ::Val{NT}) where {W,U,A,S,NT}
#     @assert W == 8sizeof(U)
#     vstore!(Base.unsafe_convert(Ptr{U}, ptr), data(m), i, Val{A}(), Val{S}(), Val{NT}())
# end

# BitArray stores
@inline function vstore!(ptr::Ptr{Bit}, v::Mask{W,U}, ::Val{A}, ::Val{S}, ::Val{NT}) where {W, U, A, S, NT}
    vstore!(Base.unsafe_convert(Ptr{U}, ptr), data(v), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{Bit}, v::Mask{W,U}, i::VectorIndex{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, U, A, S, NT}
    vstore!(Base.unsafe_convert(Ptr{U}, ptr), data(v), data(i) >> 3, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{Bit}, v::Mask{W,U}, i::VectorIndex{W}, m::Mask, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, U, A, S, NT}
    vstore!(Base.unsafe_convert(Ptr{U}, ptr), data(v), data(i) >> 3, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(f::F, ptr::Ptr{Bit}, v::Mask{W,U}, ::Val{A}, ::Val{S}, ::Val{NT}) where {W, U, A, S, NT, F<:Function}
    vstore!(f, Base.unsafe_convert(Ptr{U}, ptr), data(v), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    f::F, ptr::Ptr{Bit}, v::Mask{W,U}, i::VectorIndex{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, U, A, S, NT, F<:Function}
    vstore!(f, Base.unsafe_convert(Ptr{U}, ptr), data(v), data(i) >> 3, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    f::F, ptr::Ptr{Bit}, v::Mask{W,U}, i::VectorIndex{W}, m::Mask, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, U, A, S, NT, F<:Function}
    vstore!(f, Base.unsafe_convert(Ptr{U}, ptr), data(v), data(i) >> 3, Val{A}(), Val{S}(), Val{NT}())
end



@inline function vstore!(ptr::Ptr{T}, v, ::Val{A}, ::Val{S}, ::Val{NT}) where {T, A, S, NT}
    vstore!(ptr, convert(T, v), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v, i, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT}
    vstore!(ptr, convert(T, v), i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v, i::VectorIndex{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W,T, A, S, NT}
    vstore!(ptr, convert(Vec{W,T}, v), i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    ptr::Ptr{T}, v, i, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T, A, S, NT}
    vstore!(ptr, convert(Vec{W,T}, v), i, m, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(f::F, ptr::Ptr{T}, v, ::Val{A}, ::Val{S}, ::Val{NT}) where {T, A, S, NT, F<:Function}
    vstore!(f, ptr, convert(T, v), Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    f::F, ptr::Ptr{T}, v, i, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T, A, S, NT, F<:Function}
    vstore!(f, ptr, convert(T, v), i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    f::F, ptr::Ptr{T}, v, i::VectorIndex{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T, A, S, NT, F<:Function}
    vstore!(f, ptr, convert(Vec{W,T}, v), i, Val{A}(), Val{S}(), Val{NT}())
end
@inline function vstore!(
    f::F, ptr::Ptr{T}, v, i, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {W, T, A, S, NT, F<:Function}
    vstore!(f, ptr, convert(Vec{W,T}, v), i, m, Val{A}(), Val{S}(), Val{NT}())
end


for (store,align,alias,nontemporal) ∈ [
    (:vstore!,false,false,false),
    (:vstorea!,true,false,false),
    (:vstorent!,true,false,true),
    (:vnoaliasstore!,false,true,false),
    (:vnoaliasstorea!,true,true,false),
    (:vnoaliasstorent!,true,true,true)
]
    @eval begin
        @inline function $store(ptr::Union{Ptr,AbstractStridedPointer}, v::Number)
            vstore!(ptr, v, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll})
            vstore!(ptr, v, i, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll}, m::Mask)
            vstore!(ptr, v, i, m, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll}, b::Bool)
            b && vstore!(ptr, v, i, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        
        @inline function $store(f::F, ptr::Union{Ptr,AbstractStridedPointer}, v::Number) where {F<:Function}
            vstore!(f, ptr, v, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(f::F, ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll}) where {F<:Function}
            vstore!(f, ptr, v, i, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(f::F, ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll}, m::Mask) where {F<:Function}
            vstore!(f, ptr, v, i, m, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
        @inline function $store(f::F, ptr::Union{Ptr,AbstractStridedPointer}, v::Number, i::Union{Number,Tuple,Unroll}, b::Bool) where {F<:Function}
            b && vstore!(f, ptr, v, i, Val{$align}(), Val{$alias}(), Val{$nontemporal}())
        end
    end
end



# unroll
@inline Base.Broadcast.broadcastable(u::Unroll) = (u,)


"""
Returns a vector of expressions for a set of unrolled indices.


"""
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

function vload_unroll_quote(D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, mask::Bool, align::Bool)
    t = Expr(:tuple)
    inds = unrolled_indicies(D, AU, F, N, AV, W)
    # TODO: Consider doing some alignment checks before accepting user's `align`?
    alignval = Expr(:call, Expr(:curly, :Val, align))
    for n in 1:N
        l = Expr(:call, :vload, :gptr, inds[n], alignval)
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

@generated function vload(ptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,I}, ::Val{A}) where {A,AU,F,N,AV,W,M,I,T,D}
    vload_unroll_quote(D, AU, F, N, AV, W, M, false, A)
end
@generated function vload(ptr::AbstractStridedPointer{T,D}, u::Unroll{AU,F,N,AV,W,M,I}, m::Mask{W}, ::Val{A}) where {A,AU,F,N,AV,W,M,I,T,D}
    vload_unroll_quote(D, AU, F, N, AV, W, M, true, A)
end

function vstore_unroll_quote(
    D::Int, AU::Int, F::Int, N::Int, AV::Int, W::Int, M::UInt, mask::Bool, align::Bool, noalias::Bool, nontemporal::Bool
)
    t = Expr(:tuple)
    inds = unrolled_indicies(D, AU, F, N, AV, W)
    q = quote
        $(Expr(:meta, :inline))
        gptr = gesp(ptr, u.i)
        t = data(v)
    end
    alignval = Expr(:call, Expr(:curly, :Val, align))
    noaliasval = Expr(:call, Expr(:curly, :Val, noalias))
    nontemporalval = Expr(:call, Expr(:curly, :Val, nontemporal))
    for n in 1:N
        l = Expr(:call, :vstore!, :gptr, Expr(:ref, :t, n), inds[n], alignval, noaliasval, nontemporalval)
        (mask && (M % Bool)) && push!(l.args, :m)
        M >>= 1
        push!(q.args, l)
    end
    q
end
@generated function vstore!(
    ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {AU,F,N,AV,W,M,I,T,D,Nm1,S,A,NT}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, false, A, S, NT)
end
@generated function vstore!(
    ptr::AbstractStridedPointer{T,D}, v::VecUnroll{Nm1,W}, u::Unroll{AU,F,N,AV,W,M,I}, m::Mask{W}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {AU,F,N,AV,W,M,I,T,D,Nm1,S,A,NT}
    @assert Nm1+1 == N
    vstore_unroll_quote(D, AU, F, N, AV, W, M, true, A, S, NT)
end

# @inline vstore!(::typeof(identity), ptr, v, u) = vstore!(ptr, v, u)
# @inline vstore!(::typeof(identity), ptr, v, u, m) = vstore!(ptr, v, u, m)
# @inline vnoaliasstore!(::typeof(identity), ptr, v, u) = vnoaliasstore!(ptr, v, u)
# @inline vnoaliasstore!(::typeof(identity), ptr, v, u, m) = vnoaliasstore!(ptr, v, u, m)


# If `::Function` vectorization is masked, then it must not be reduced by `::Function`.
@generated function vstore!(
    ::Function, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}, m, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,D,C,U,AU,F,N,W,M,I,AV,A,S,NT}
    @assert N == U + 1
    # mask means it isn't vectorized
    @assert AV > 0 "AV ≤ 0, but masking what, exactly?"
    Expr(:block, Expr(:meta, :inline), :(vstore!(ptr, vu, u, m, Val{$A}(), Val{$S}(), Val{$NT}())))
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
    ::G, ptr::AbstractStridedPointer{T,D,C}, vu::VecUnroll{U,W}, u::Unroll{AU,F,N,AV,W,M,I}, ::Val{A}, ::Val{S}, ::Val{NT}
) where {T,D,C,U,AU,F,N,W,M,I,G<:Function,AV,A,S,NT}
    @assert N == U + 1
    if G === typeof(identity) || AV > 0
        return Expr(:block, Expr(:meta, :inline), :(vstore!(ptr, vu, u, Val{$A}(), Val{$S}(), Val{$NT}())))
    elseif G === typeof(vsum)
        op = :+; reduct = :vsum
    elseif G === typeof(vprod)
        op = :*; reduct = :vprod
    else
        throw("Function $f not recognized.")
    end
    horizonal_reduce_store_expr(W, N, (C,D,AU,F), op, reduct, S)
end



function lazymulunroll_load_quote(M,O,N,mask,align)
    t = Expr(:tuple)
    alignval = Expr(:call, Expr(:curly, :Val, align))
    for n in 1:N+1
        call = if mask
            Expr(:call, :vload, :ptr, :(LazyMulAdd{$M,$O}(u[$n])), m, alignval)
        else
            Expr(:call, :vload, :ptr, :(LazyMulAdd{$M,$O}(u[$n])), alignval)
        end
        push!(t.args, call)
    end
    Expr(:block, Expr(:meta, :inline), :(u = um.data.data), Expr(:call, :VecUnroll, t))
end
@generated function vload(ptr::Ptr{T}, um::LazyMulAdd{M,O,VecUnroll{N,W,I,V}}, ::Val{A}) where {T,M,O,N,W,I,V,A}
    lazymulunroll_load_quote(M,O,N,false,A)
end
@generated function vload(ptr::Ptr{T}, um::LazyMulAdd{M,O,VecUnroll{N,W,I,V}}, m::Mask{W}, ::Val{A}) where {T,M,O,N,W,I,V,A}
    lazymulunroll_load_quote(M,O,N,true,A)
end
function lazymulunroll_store_quote(M,O,N,mask,align,noalias,nontemporal)
    q = Expr(:block, Expr(:meta, :inline), :(u = um.data.data), :(v = vm.data.data))
    alignval = Expr(:call, Expr(:curly, :Val, align))
    noaliasval = Expr(:call, Expr(:curly, :Val, noalias))
    nontemporalval = Expr(:call, Expr(:curly, :Val, nontemporal))
    for n in 1:N+1
        push!(q.args, Expr(:call, :vstore!, :ptr, Expr(:ref, :v, n), :(LazyMulAdd{$M,$O}(u[$n])), alignval, noaliasval, nontemporalval))
    end
    q
end

@generated function prefetch(ptr::Ptr{Cvoid}, ::Val{L}, ::Val{R}) where {L, R}
    @assert L ∈ (0,1,2,3)
    @assert R ∈ (0, 1)
    decl = "declare void @llvm.prefetch(i8*, i32, i32, i32)"
    instrs = """
        %addr = inttoptr $JULIAPOINTERTYPE %0 to i8*
        call void @llvm.prefetch(i8* %addr, i32 $R, i32 $L, i32 1)
        ret void
    """
    llvmcall_expr(decl, instrs, Cvoid, :(Tuple{Ptr{Cvoid}}), "void", [JULIAPOINTERTYPE], [:ptr])
end
@inline prefetch(ptr::Ptr{T}, ::Val{L}, ::Val{R}) where {T,L,R} = prefetch(Base.unsafe_convert(Ptr{Cvoid}, ptr), Val{L}(), Val{R}())

@inline function prefetch(ptr::Union{AbstractStridedPointer,Ptr}, i, ::Val{Locality}, ::Val{ReadOrWrite}) where {Locality, ReadOrWrite}
    prefetch(gep(ptr, i), Val{Locality}(), Val{ReadOrWrite}())
end
@inline prefetch(ptr::Ptr) = prefetch(ptr, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, ::Val{L}) where {L} = prefetch(ptr, Val{L}(), Val{0}())
@inline prefetch(ptr::Ptr, i) = prefetch(ptr, i, Val{3}(), Val{0}())
@inline prefetch(ptr::Ptr, i, ::Val{L}) where {L} = prefetch(ptr, i, Val{L}(), Val{0}())


@inline prefetch0(x, i) = prefetch(gep(stridedpointer(x), (data(i),)), Val{3}(), Val{0}())
@inline prefetch0(x, I::Tuple) = prefetch(gep(stridedpointer(x), data.(I)), Val{3}(), Val{0}())
@inline prefetch0(x, i, j) = prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{3}(), Val{0}())
# @inline prefetch0(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{3}(), Val{0}())
@inline prefetch1(x, i) = prefetch(gep(stridedpointer(x), (data(i),)), Val{2}(), Val{0}())
@inline prefetch1(x, i, j) = prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{2}(), Val{0}())
# @inline prefetch1(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{2}(), Val{0}())
@inline prefetch2(x, i) = prefetch(gep(stridedpointer(x), (data(i),)), Val{1}(), Val{0}())
@inline prefetch2(x, i, j) = prefetch(gep(stridedpointer(x), (data(i), data(j))), Val{1}(), Val{0}())
# @inline prefetch2(x, i, j, oi, oj) = prefetch(gep(stridedpointer(x), (data(i) + data(oi) - 1, data(j) + data(oj) - 1)), Val{1}(), Val{0}())

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

@inline vload(::StaticInt{N}, args...) where {N} = StaticInt{N}()
@inline stridedpointer(::StaticInt{N}) where {N} = StaticInt{N}()
@inline zero_offsets(::StaticInt{N}) where {N} = StaticInt{N}()


