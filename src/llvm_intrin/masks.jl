
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

@generated function splitint(i::S, ::Type{T}) where {S <: Base.BitInteger, T <: Union{Bool,Base.BitInteger}}
    sizeof_S = sizeof(S)
    sizeof_T = sizeof(T)
    if sizeof_T > sizeof_S
        return :(i % T)
    elseif sizeof_T == sizeof_S
        return :i
    end
    W, r = divrem(sizeof_S, sizeof_T)
    @assert iszero(r)
    vtyp = "<$W x i$(8sizeof_T)>"
    instrs = """
        %split = bitcast i$(8sizeof_S) %0 to $vtyp
        ret $vtyp %split
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$S}, i))
    end
end
@generated function fuseint(v::Vec{W,I}) where {W, I <: Union{Bool,Base.BitInteger}}
    @assert ispow2(W)
    bytes = W * sizeof(I)
    bits = 8bytes
    @assert bytes ≤ 16
    T = (I <: Signed) ? Symbol(:Int, bits) : Symbol(:UInt, bits)
    vtyp = "<$W x i$(8sizeof(I))>"
    styp = "i$(bits)"
    instrs = """
        %fused = bitcast $vtyp %0 to $styp
        ret $styp %fused
    """
    quote
        $(Expr(:meta,:inline))
        llvmcall($instrs, $T, Tuple{_Vec{$W,$I}}, data(v))
    end            
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

@inline Base.:(&)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? m.u : zero(m.u))
@inline Base.:(&)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? m.u : zero(m.u))

@inline Base.:(|)(m::Mask{W,U}, b::Bool) where {W,U} = b ? max_mask(Mask{W,U}) : m
@inline Base.:(|)(b::Bool, m::Mask{W,U}) where {W,U} = b ? max_mask(Mask{W,U}) : m

@inline Base.:(⊻)(m::Mask{W}, b::Bool) where {W} = Mask{W}(b ? ~m.u : m.u)
@inline Base.:(⊻)(b::Bool, m::Mask{W}) where {W} = Mask{W}(b ? ~m.u : m.u)

@inline Base.:(<<)(m::Mask{W}, i::IntegerTypesHW) where {W} = Mask{W}(shl(m.u, i))
@inline Base.:(>>)(m::Mask{W}, i::IntegerTypesHW) where {W} = Mask{W}(shr(m.u, i))
@inline Base.:(>>>)(m::Mask{W}, i::IntegerTypesHW) where {W} = Mask{W}(shr(m.u, i))

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

@inline Base.count_ones(m::Mask) = count_ones(m.u)
@inline Base.:(+)(m::Mask, i::Integer) = i + count_ones(m)
@inline Base.:(+)(i::Integer, m::Mask) = i + count_ones(m)

function mask_type_symbol(W)
    if W <= 8
        return :UInt8
    elseif W <= 16
        return :UInt16
    elseif W <= 32
        return :UInt32
    elseif W <= 64
        return :UInt64
    else#if W <= 128
        return :UInt128
    end
end
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
mask_type(::Union{Val{1},StaticInt{1}}) = UInt8#Bool
mask_type(::Union{Val{2},StaticInt{2}}) = UInt8
mask_type(::Union{Val{4},StaticInt{4}}) = UInt8
mask_type(::Union{Val{8},StaticInt{8}}) = UInt8
mask_type(::Union{Val{16},StaticInt{16}}) = UInt16
mask_type(::Union{Val{32},StaticInt{32}}) = UInt32
mask_type(::Union{Val{64},StaticInt{64}}) = UInt64

@generated function mask_type(::Type{T}, ::Union{Val{P},StaticInt{P}}) where {T,P}
    mask_type_symbol(pick_vector_width(P, T))
end
@generated function mask_type(::Type{T}) where {T}
    W = max(1, register_size(T) >>> intlog2(T))
    mask_type_symbol(W)
    # mask_type_symbol(pick_vector_width(T))
end
@generated function Base.zero(::Type{<:Mask{W}}) where {W}
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :Mask, W), Expr(:call, :zero, mask_type_symbol(W))))
end

@generated function max_mask(::Union{Val{W},StaticInt{W}}) where {W}
    U = mask_type(W)
    Mask{W,U}(one(U)<<W - one(U))
end
@inline max_mask(::Type{T}) where {T} = max_mask(pick_vector_width_val(T))
@generated max_mask(::Type{Mask{W,U}}) where {W,U} = Mask{W,U}(one(U)<<W - one(U))

@generated function valrem(::Union{Val{W},StaticInt{W}}, l) where {W}
    ex = ispow2(W) ? :(l & $(W - 1)) : :(l % $W)
    Expr(:block, Expr(:meta, :inline), ex)
end
@generated function mask(::Union{Val{W},StaticInt{W}}, l::Integer) where {W}
    M = mask_type(W)
    if HAS_OPMASK_REGISTERS
        quote # If the arch has opmask registers, we can generate a bitmask and then move it into the opmask register
            $(Expr(:meta,:inline))
            rem = valrem(Val{$W}(), vsub((l % $M), one($M)))
            Mask{$W,$M}($(typemax(M)) >>> ($(M(8sizeof(M))-1) - rem))
        end
    else
        quote # Otherwise, it's probably more efficient to use a comparison, as this will probably create some type that can be used directly for masked moves/blends/etc
            $(Expr(:meta,:inline))
            rem = valrem(Val{$W}(), vsub((l % $M), one($M)))
            rem ≥ MM{$W}(0)
        end
    end
end
@generated mask(::Union{Val{W},StaticInt{W}}, ::StaticInt{L}) where {W, L} = mask(Val(W), L)
@inline mask(::Type{T}, l::Integer) where {T} = mask(pick_vector_width_val(T), l)


# @generated function masktable(::Union{Val{W},StaticInt{W}}, rem::Integer) where {W}
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
    U = mask_type_symbol(W)
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

@generated function Base.:(%)(m::Mask{W,U}, ::Type{I}) where {W,U,I<:Integer}
    bits = 8sizeof(I)
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%res = zext <$W x i1> %mask.0 to <$W x i$(bits)>\nret <$W x i$(bits)> %res")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($(join(instrs, "\n")), _Vec{$W,$I}, Tuple{$U}, data(m)))
    end
end
Vec(m::Mask{W}) where {W} = m % int_type(Val{W}())

# @inline getindexzerobased(m::Mask, i) = (m.u >>> i) % Bool
# @inline function extractelement(m::Mask{W}, i::Integer) where {W}
#     @boundscheck i > W && throw(BoundsError(m, i))
#     getindexzerobased(m, i)
# end
@generated function extractelement(v::Mask{W,U}, i::I) where {W,U,I}
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%res1 = extractelement <$W x i1> %mask.0, i$(8sizeof(I)) %1")
    push!(instrs, "%res8 = zext i1 %res1 to i8\nret i8 %res8")
    instrs_string = join(instrs, "\n")
    call = :(llvmcall($instrs_string, Bool, Tuple{$U,$I}, data(v), i))
    Expr(:block, Expr(:meta, :inline), call)
end
@generated function insertelement(v::Mask{W,U}, x::T, i::I) where {W, T, U, I <: Union{Bool,IntegerTypesHW}}
    mtyp_input = "i$(max(8,W))"
    instrs = String["%bit = trunc i$(8sizeof(T)) %1 to i1"]
    truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%bitvec = insertelement <$W x i1> %mask.0, i1 %bit, i$(8sizeof(I)) %2")
    zext_mask!(instrs, "bitvec", W, 1)
    push!(instrs, "ret $(mtyp_input) %res.1")
    instrs_string = join(instrs, "\n")
    call = :(Mask{$W}(llvmcall($instrs_string, $U, Tuple{$U,$T,$I}, data(v), x, i)))
    Expr(:block, Expr(:meta, :inline), call)
end


# @generated function Base.isodd(i::MM{W,1}) where {W}
#     U = mask_type(W)
#     evenfirst = 0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa % U
#     # Expr(:block, Expr(:meta, :inline), :(isodd(i.i) ? Mask{$W}($oddfirst) : Mask{$W}($evenfirst)))
#     Expr(:block, Expr(:meta, :inline), :(Mask{$W}($evenfirst >> (i.i & 0x03))))
# end
# @generated function Base.iseven(i::MM{W,1}) where {W}
#     U = mask_type(W)
#     oddfirst = 0x55555555555555555555555555555555 % U
#     # evenfirst = oddfirst << 1
#     # Expr(:block, Expr(:meta, :inline), :(isodd(i.i) ? Mask{$W}($evenfirst) : Mask{$W}($oddfirst)))
#     Expr(:block, Expr(:meta, :inline), :(Mask{$W}($oddfirst >> (i.i & 0x03))))
# end
@inline Base.isodd(i::MM{W,1}) where {W} = Mask{W}((0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa % mask_type(Val{W}())) >>> (i.i & 0x03))
@inline Base.iseven(i::MM{W,1}) where {W} = Mask{W}((0x55555555555555555555555555555555 % mask_type(Val{W}())) >>> (i.i & 0x03))

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
    vtyp = vtype(W, T === Float32 ? "float" : "double");
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

# for (f,cond) ∈ [(:(==), "oeq"), (:(>), "ogt"), (:(≥), "oge"), (:(<), "olt"), (:(≤), "ole"), (:(≠), "one")]
for (f,cond) ∈ [(:(==), "ueq"), (:(>), "ugt"), (:(≥), "uge"), (:(<), "ult"), (:(≤), "ule"), (:(≠), "une")]
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


# import IfElse: ifelse
@generated function ifelse(m::Mask{W,U}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,U<:Unsigned,T}
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    selty = vtype(W, "i1")
    f = "select"
    if Base.libllvm_version ≥ v"9" && ((T === Float32) || (T === Float64))
        f *= " nsz arcp contract reassoc"
    end
    instrs = String[]
    truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%res = $f $selty %mask.0, $vtyp %1, $vtyp %2\nret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($(join(instrs,"\n")), _Vec{$W,$T}, Tuple{$U,_Vec{$W,$T},_Vec{$W,$T}}, data(m), data(v1), data(v2)))
    end
end
# @inline ifelse(m::Mask, v::Vec, s) = ((x,y) = promote(v,s); ifelse(m,x,y))
# @inline ifelse(m::Mask, s, v::Vec) = ((x,y) = promote(s,v); ifelse(m,x,y))
@inline ifelse(m::Mask{W}, s1::T, s2::T) where {W,T<:NativeTypes} = ifelse(m, Vec{W,T}(s1), Vec{W,T}(s2))
@inline ifelse(m::Mask{W}, s1, s2) where {W} = ((x1,x2) = promote(s1,s2); ifelse(m, x1, x2))

@inline Base.Bool(m::Mask{1,UInt8}) = (m.u & 0x01) === 0x01
@inline Base.convert(::Type{Bool}, m::Mask{1,UInt8}) = (m.u & 0x01) === 0x01
@inline ifelse(m::Mask{1}, s1::T, s2::T) where {T<:NativeTypes} = ifelse(Bool(m), s1, s2)

@inline Base.isnan(v::AbstractSIMD) = v != v

@inline Base.isfinite(x::AbstractSIMD) = iszero(x - x)

@inline Base.flipsign(x::AbstractSIMD, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
for T ∈ [:Float32, :Float64]
    @eval begin
        @inline Base.flipsign(x::AbstractSIMD, y::$T) = ifelse(y > zero(y), x, -x)
        @inline Base.flipsign(x::$T, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
    end
end
@inline Base.flipsign(x::AbstractSIMD, y::Real) = ifelse(y > zero(y), x, -x)
@inline Base.flipsign(x::Real, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
@inline Base.flipsign(x::Signed, y::AbstractSIMD) = ifelse(y > zero(y), x, -x)
@inline Base.isodd(x::AbstractSIMD{W,T}) where {W,T<:Integer} = (x & one(T)) != zero(T)


@generated function ifelse(m::Vec{W,Bool}, v1::Vec{W,T}, v2::Vec{W,T}) where {W,T}
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    selty = vtype(W, "i1")
    f = "select"
    if Base.libllvm_version ≥ v"9" && ((T === Float32) || (T === Float64))
        f *= " nsz arcp contract reassoc"
    end
    instrs = String["%mask.0 = trunc <$W x i8> %0 to <$W x i1>"]
    # truncate_mask!(instrs, '0', W, 0)
    push!(instrs, "%res = $f $selty %mask.0, $vtyp %1, $vtyp %2\nret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($(join(instrs,"\n")), _Vec{$W,$T}, Tuple{_Vec{$W,Bool},_Vec{$W,$T},_Vec{$W,$T}}, data(m), data(v1), data(v2)))
    end
end
@inline ifelse(b::Bool, s::NativeTypes, v::V) where {V <: AbstractSIMD} = ifelse(b, convert(V, s), v)
@inline ifelse(b::Bool, v::V, s::NativeTypes) where {V <: AbstractSIMD} = ifelse(b, v, convert(V, s))

@generated function Base.convert(::Type{Bit}, v::Vec{W,Bool}) where {W,Bool}
    instrs = String[]
    push!(instrs, "%m = trunc <$W x i8> %0 to <$W x i1>")
    zext_mask!(instrs, 'm', W, '0')
    push!(instrs, "ret i$(max(8,W)) %res.0")
    U = mask_type(W);
    quote
        $(Expr(:meta,:inline))
        Mask{$W}(llvmcall($(join(instrs, "\n")), $U, Tuple{_Vec{$W,Bool}}, data(v)))
    end    
end
@inline Base.convert(::Type{Vec{W,Bit}}, v::Vec{W,Bool}) where {W,Bool} = convert(Bit, v)

                            
