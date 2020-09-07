
#
# We use these definitions because when we have other SIMD operations with masks
# LLVM optimizes the masks better.
function truncate_mask!(instrs, input, W, suffix)
    mtyp_input = "i$(max(8,W))"
    mtyp_trunc = "i$(W)"
    str = if mtyp_input == mtyp_trunc
        "%mask.$(suffix) = bitcast $mtyp_input %$input to <$W x i1>"
    else
        "%masktrunc.$(suffix) = trunc $mtyp_input %$input to $mtyp_trunc\n%mask.$(suffix) = bitcast $mtyp_trunc %masktrunc.$(suffix) to <$W x i1>"
    end
    push!(instrs, str)
end
function zext_mask!(instrs, input, W, suffix)
    mtyp_input = "i$(max(8,W))"
    mtyp_trunc = "i$(W)"
    str = if mtyp_input == mtyp_trunc
        "%res.$(suffix) = bitcast <$W x i1> %$input to $mtyp_input"
    else
        "%restrunc.$(suffix) = bitcast <$W x i1> %$input to $mtyp_trunc\n%res.$(suffix) = zext $mtyp_trunc %restrunc.$(suffix) to $mtyp_input"
    end
    push!(instrs, str)
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
    truncate_mask!(instrs, '0', W, 0)
    truncate_mask!(instrs, '1', W, 1)
    push!(instrs, """%uv.0 = zext <$W x i1> %mask.0 to <$W x i8>
    %uv.1 = zext <$W x i1> %mask.1 to <$W x i8>
    %res = add <$W x i8> %uv.0, %uv.1
    ret <$W x i8> %res""")
    Expr(:block, Expr(:meta, :inline), :(Vec(llvmcall($(join(instrs, "\n")), _Vec{$W,UInt8}, Tuple{$U, $U}, m1.u, m2.u))))
end
@generated Base.:(+)(m1::Mask{W,U}, m2::Mask{W,U}) where {W,U} = vadd_expr(W,U)

# @inline Base.:(+)(m1::Mask, m2::Mask) = vadd(m1,m2)

# @inline Base.:(&)(m1::Mask{W}, m2::Mask{W}) where {W} = andmask(m1, m2)
# @inline Base.:(&)(m::Mask{W}, u::Unsigned) where {W} = m & Mask{W}(u)
# @inline Base.:(&)(u::Unsigned, m::Mask{W}) where {W} = Mask{W}(u) & m

@inline Base.:(&)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? m.u : zero(m.u))
@inline Base.:(&)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? m.u : zero(m.u))

# @inline Base.:(|)(m1::Mask{W}, m2::Mask{W}) where {W} = ormask(m1, m2)
# @inline Base.:(|)(m::Mask{W}, u::Unsigned) where {W} = ormask(m, Mask{W}(u))
# @inline Base.:(|)(u::Unsigned, m::Mask{W}) where {W} = ormask(Mask{W}(u), m)

@inline Base.:(|)(m::Mask{W,U}, b::Bool) where {W,U} = b ? max_mask(Mask{W,U}) : m
@inline Base.:(|)(b::Bool, m::Mask{W,U}) where {W,U} = b ? max_mask(Mask{W,U}) : m
# @inline Base.:(|)(m::Mask{16,UInt16}, b::Bool) = Mask{16}(b ? 0xffff : m.u)
# @inline Base.:(|)(b::Bool, m::Mask{16,UInt16}) = Mask{16}(b ? 0xffff : m.u)
# @inline Base.:(|)(m::Mask{8,UInt8}, b::Bool) = Mask{8}(b ? 0xff : m.u)
# @inline Base.:(|)(b::Bool, m::Mask{8,UInt8}) = Mask{8}(b ? 0xff : m.u)
# @inline Base.:(|)(m::Mask{4,UInt8}, b::Bool) = Mask{4}(b ? 0x0f : m.u)
# @inline Base.:(|)(b::Bool, m::Mask{4,UInt8}) = Mask{4}(b ? 0x0f : m.u)
# @inline Base.:(|)(m::Mask{2,UInt8}, b::Bool) = Mask{2}(b ? 0x03 : m.u)
# @inline Base.:(|)(b::Bool, m::Mask{2,UInt8}) = Mask{2}(b ? 0x03 : m.u)

# @inline Base.:(⊻)(m1::Mask{W}, m2::Mask{W}) where {W} = xormask(m1, m2)
# @inline Base.:(⊻)(m::Mask{W}, u::Unsigned) where {W} = xormask(m, Mask{W}(u))
# @inline Base.:(⊻)(u::Unsigned, m::Mask{W}) where {W} = xormask(Mask{W}(u), m)

@inline Base.:(⊻)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? ~m.u : m.u)
@inline Base.:(⊻)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? ~m.u : m.u)

@inline Base.:(<<)(m::Mask{W}, i) where {W} = Mask{W}(shl(m.u, i))
@inline Base.:(>>)(m::Mask{W}, i) where {W} = Mask{W}(shr(m.u, i))
@inline Base.:(>>>)(m::Mask{W}, i) where {W} = Mask{W}(shr(m.u, i))

for (U,W) in [(UInt8,8), (UInt16,16), (UInt32,32), (UInt64,64)]
    @eval @inline vany(m::Mask{$W,$U}) = m.u != $(zero(U))
    @eval @inline vall(m::Mask{$W,$U}) = m.u == $(typemax(U))
end
@inline vany(m::Mask{W}) where {W} = (m.u & max_mask(Val{W}()).u) !== zero(m.u)
@inline vall(m::Mask{W}) where {W} = (m.u & max_mask(Val{W}()).u) === (max_mask(Val{W}()).u)

@generated function Base.:(!)(m::Mask{W,U}) where {W,U}
    mtyp_input = "i$(8sizeof(U))"
    mtyp_trunc = "i$(W)"
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    mask = llvmconst(W, "i1 true")
    push!(instrs, "%resvec.0 = xor <$W x i1> %mask.0, $mask")
    zext_mask!(instrs, "resvec.0", W, 1)
    push!(instrs, "ret $mtyp_input %res.1")
    quote
        $(Expr(:meta,:inline))
        Mask{$W}(llvmcall($(join(instrs,"\n")), $U, Tuple{$U}, m.u))
    end
end
@inline Base.:(~)(m::Mask) = !m
#@inline Base.:(!)(m::Mask{W}) where {W} = Mask{W}( ~m.u )


# @inline Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u == m2.u
# @inline Base.:(==)(m::Mask{W}, u::Unsigned) where {W} = m.u == u
# @inline Base.:(==)(u::Unsigned, m::Mask{W}) where {W} = u == m.u
# @inline Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = m1.u != m2.u
# @inline Base.:(!=)(m::Mask{W}, u::Unsigned) where {W} = m.u != u
# @inline Base.:(!=)(u::Unsigned, m::Mask{W}) where {W} = u != m.u

# @inline Base.@pure Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = m1 == m2
# @inline Base.:(==)(m::Mask{W}, u::Unsigned) where {W} = m.u == u
# @inline Base.:(==)(u::Unsigned, m::Mask{W}) where {W} = u == m.u
# @inline Base.@pure Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = m1 != m2
# @inline Base.:(!=)(m::Mask{W}, u::Unsigned) where {W} = m.u != u
# @inline Base.:(!=)(u::Unsigned, m::Mask{W}) where {W} = u != m.u


# @inline Base.:(==)(m1::Mask{W}, m2::Mask{W}) where {W} = equalmask(m1, m2)
# @inline Base.:(==)(m::Mask{W}, u::Unsigned) where {W} = equalmask(m1, Mask{W}(m2))
# @inline Base.:(==)(u::Unsigned, m::Mask{W}) where {W} = equalmask(Mask{W}(m1), m2)
# @inline Base.:(!=)(m1::Mask{W}, m2::Mask{W}) where {W} = notequalmask(m1, m2)
# @inline Base.:(!=)(m::Mask{W}, u::Unsigned) where {W} = notequalmask(m1, Mask{W}(m2))
# @inline Base.:(!=)(u::Unsigned, m::Mask{W}) where {W} = notequalmask(Mask{W}(m1), m2)

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
    zext_mask!(instrs, "bitvec", W, 0)
    push!(instrs, "ret i$(usize) %res.0")
    quote
        $(Expr(:meta, :inline))
        Mask{$W}(llvmcall($(join(instrs, "\n")), $U, Tuple{_Vec{$W,Bool}}, data(v)))
    end
end
@inline tomask(v::AbstractSIMDVector{<:Any,Bool}) = tomask(data(v))

@generated function toboolvec(m::Mask{W,U}) where {W,U}
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%res = zext <$W x i1> %mask.0 to <$W x i8>\nret <$W x i8> %res")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($(join(instrs, "\n")), _Vec{$W,Bool}, Tuple{$U}, data(m)))
    end
end

@inline getindexzerobased(m::Mask, i) = (m.u >>> i) % Bool
@inline function getelement(m::Mask{W}, i::Integer) where {W}
    @boundscheck i > W && throw(BoundsError(m, i))
    getindexzerobased(m, i - 1)
end

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

function cmp_quote(W, cond, vtyp, T1, T2 = T1)
    instrs = String["%m = $cond $vtyp %0, %1"]
    zext_mask!(instrs, 'm', W, '0')
    push!(instrs, "ret i$(max(8,W)) %res.0")
    U = mask_type(W);
    quote
        $(Expr(:meta,:inline))
        Mask{$W}(llvmcall($(join(instrs, "\n")), $U, Tuple{_Vec{$W,$T1},_Vec{$W,$T2}}, data(v1), data(v2)))
    end
end
function icmp_quote(W, cond, bytes, T1, T2 = T1)
    vtyp = vtype(W, "i$(8bytes)");
    cmp_quote(W, "icmp " * cond, vtyp, T1, T2)
end
function fcmp_quote(W, cond, T)
    vtyp = vtype(W, T === :Float32 ? "float" : "double");
    cmp_quote(W, "fcmp nsz arcp contract reassoc " * cond, vtyp, T)
end
# @generated function compare(::Val{cond}, v1::Vec{W,I}, v2::Vec{W,I}) where {cond, W, I}
    # cmp_quote(W, cond, sizeof(I), I)
# end
# for (f,cond) ∈ [(:(==), :eq), (:(!=), :ne), (:(>), :ugt), (:(≥), :uge), (:(<), :ult), (:(≤), :ule)]
for (f,cond) ∈ [(:(==), "eq"), (:(!=), "ne")]
    @eval @generated function Base.$f(v1::Vec{W,T1}, v2::Vec{W,T2}) where {W,T1<:Integer,T2<:Integer}
        @assert sizeof(T1) == sizeof(T2)
        icmp_quote(W, $cond, sizeof(T1), T1, T2)
    end
end
for (f,cond) ∈ [(:(>), "ugt"), (:(≥), "uge"), (:(<), "ult"), (:(≤), "ule")]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Unsigned}
        icmp_quote(W, $cond, sizeof(T), T)
    end
    # bytes = 1
    # while bytes ≤ 8
    #     T = Symbol(:UInt, 8bytes)
    #     W = 2
    #     while W * bytes ≤ REGISTER_SIZE
    #         @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(icmp_quote(W, cond, bytes, T))
    #         W += W
    #     end
    #     bytes += bytes
    # end
    # @eval @inline Base.$f(v1::Vec{W,I}, v2::Vec{W,I}) where {W, I <: Signed} = compare(Val{$(QuoteNode(cond))}(), v1, v2)
end
for (f,cond) ∈ [(:(>), "sgt"), (:(≥), "sge"), (:(<), "slt"), (:(≤), "sle")]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Signed}
        icmp_quote(W, $cond, sizeof(T), T)
    end
    # bytes = 1
    # while bytes ≤ 8
    #     T = Symbol(:Int, 8bytes)
    #     W = 2
    #     while W * bytes ≤ REGISTER_SIZE
    #         @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(cmp_quote(W, cond, bytes, T))
    #         W += W
    #     end
    #     bytes += bytes
    # end
end

for (f,cond) ∈ [(:(==), "oeq"), (:(>), "ogt"), (:(≥), "oge"), (:(<), "olt"), (:(≤), "ole"), (:(≠), "one")]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        fcmp_quote(W, $cond, T)
    end
    # for (T,bytes) ∈ [(:Float32,4), (:Float64,8)]
    #     W = 2
    #     while W * bytes ≤ REGISTER_SIZE
    #         @eval @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(fcmp_quote(W, cond, T))
    #         W += W
    #     end
    # end
end

@generated function IfElse.ifelse(m::Mask{W,U}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,U,T}
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

