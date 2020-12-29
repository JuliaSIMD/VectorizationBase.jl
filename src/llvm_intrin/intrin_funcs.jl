
@generated function saturated_add(x::I, y::I) where {I <: IntegerTypesHW}
    typ = "i$(8sizeof(I))"
    s = I <: Signed ? 's' : 'u'
    f = "@llvm.$(s)add.sat.$typ"
    decl = "declare $typ $f($typ, $typ)"
    instrs = """
        %res = call $typ $f($typ %0, $typ %1)
        ret $typ %res
    """
    llvmcall_expr(decl, instrs, I, :(Tuple{$I,$I}), typ, [typ,typ], [:x,:y])
end
@generated function saturated_add(x::Vec{W,I}, y::Vec{W,I}) where {W,I}
    
end

@eval @inline function assume(b::Bool)
    $(llvmcall_expr("declare void @llvm.assume(i1)", "%b = trunc i8 %0 to i1\ncall void @llvm.assume(i1 %b)\nret void", :Cvoid, :(Tuple{Bool}), "void", ["i8"], [:b]))
end

@eval @inline function expect(b::Bool)
    $(llvmcall_expr("declare i1 @llvm.expect.i1(i1, i1)", """
    %b = trunc i8 %0 to i1
    %actual = call i1 @llvm.expect.i1(i1 %b, i1 true)
    %byte = zext i1 %actual to i8
    ret i8 %byte""", :Bool, :(Tuple{Bool}), "i8", ["i8"], [:b]))
end
@generated function expect(i::I, ::Val{N}) where {I <: Integer, N}
    ityp = 'i' * string(8sizeof(I))
    llvmcall_expr("declare i1 @llvm.expect.$ityp($ityp, i1)", """
    %actual = call $ityp @llvm.expect.$ityp($ityp %0, $ityp $N)
    ret $ityp %actual""", I, :(Tuple{$I}), ityp, [ityp], [:i])
end

# for (op,f) ∈ [("abs",:abs)]
# end
if Base.libllvm_version ≥ v"12"
    for (op,f,S) ∈ [("smax",:max,:Signed),("smin",:min,:Signed),("umax",:max,:Unsigned),("umin",:min,:Unsigned)]
        @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T <: $S} = llvmcall_expr($op, W, T, (W, W), (T, T))
    end
else
    @inline Base.max(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = ifelse(v1 > v2, v1, v2)
    @inline Base.min(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = ifelse(v1 < v2, v1, v2)
end
@inline Base.max(s::NativeTypes, v::Vec{W}) where {W} = max(vbroadcast(Val{W}(), s), v)
@inline Base.max(v::Vec{W}, s::NativeTypes) where {W} = max(v, vbroadcast(Val{W}(), s))
@inline Base.min(s::NativeTypes, v::Vec{W}) where {W} = min(vbroadcast(Val{W}(), s), v)
@inline Base.min(v::Vec{W}, s::NativeTypes) where {W} = min(v, vbroadcast(Val{W}(), s))
# for T ∈ [Float32, Float64]
#     W = 2
#     while W * sizeof(T) ≤ REGISTER_SIZE
# floating point
for (op,f) ∈ [("sqrt",:sqrt),("fabs",:abs),("floor",:floor),("ceil",:ceil),("trunc",:trunc),("nearbyint",:round)
              ]
    # @eval @generated Base.$f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = llvmcall_expr($op, W, T, (W,), (T,), "nsz arcp contract afn reassoc")
    @eval @generated Base.$f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = llvmcall_expr($op, W, T, (W,), (T,), "fast")
end


@generated function Base.round(::Type{Int64}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    llvmcall_expr("lrint", W, Int64, (W,), (T,), "nsz arcp contract afn reassoc")
end
@generated function Base.round(::Type{Int32}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    llvmcall_expr("lrint", W, Int32, (W,), (T,), "nsz arcp contract afn reassoc")
end
@inline Base.trunc(::Type{I}, v::AbstractSIMD{W,T}) where {W, I<:IntegerTypesHW, T <: NativeTypes} = convert(I, v)

# """
#    setbits(x::Unsigned, y::Unsigned, mask::Unsigned)

# If you have AVX512, setbits of vector-arguments will select bits according to mask `m`, selecting from `y` if 0 and from `x` if `1`.
# For scalar arguments, or vector arguments without AVX512, `setbits` requires the additional restrictions on `y` that all bits for
# which `m` is 1, `y` must be 0.
# That is for scalar arguments or vector arguments without AVX512, it requires the restriction that
# ((y ⊻ m) & m) == m
# """
# @inline setbits(x, y, m) = (x & m) | y

"""
   bitselect(m::Unsigned, x::Unsigned, y::Unsigned)

If you have AVX512, setbits of vector-arguments will select bits according to mask `m`, selecting from `x` if 0 and from `y` if `1`.
For scalar arguments, or vector arguments without AVX512, `setbits` requires the additional restrictions on `y` that all bits for
which `m` is 1, `y` must be 0.
That is for scalar arguments or vector arguments without AVX512, it requires the restriction that
((y ⊻ m) & m) == m
"""
@inline bitselect(m, x, y) = ((~m) & x) | (m & y)
if AVX512F
    # AVX512 lets us use 1 instruction instead of 2 dependent instructions to set bits
    @generated function vpternlog(m::Vec{W,UInt64}, x::Vec{W,UInt64}, y::Vec{W,UInt64}, ::Val{L}) where {W, L}
        @assert W ∈ (2,4,8)
        bits = 64W
        decl64 = "declare <$W x i64> @llvm.x86.avx512.mask.pternlog.q.$(bits)(<$W x i64>, <$W x i64>, <$W x i64>, i32, i8)"
        instr64 = """
            %res = call <$W x i64> @llvm.x86.avx512.mask.pternlog.q.$(bits)(<$W x i64> %0, <$W x i64> %1, <$W x i64> %2, i32 $L, i8 -1)
            ret <$W x i64> %res
        """
        arg_syms = [:(data(m)), :(data(x)), :(data(y))]
        llvmcall_expr(decl64, instr64, :(_Vec{$W,UInt64}), :(Tuple{_Vec{$W,UInt64},_Vec{$W,UInt64},_Vec{$W,UInt64}}), "<$W x i64>", ["<$W x i64>", "<$W x i64>", "<$W x i64>"], arg_syms)
    end
    @generated function vpternlog(m::Vec{W,UInt32}, x::Vec{W,UInt32}, y::Vec{W,UInt32}, ::Val{L}) where {W, L}
        if W ∉ (4,8,16)
            return Expr(:block, Expr(:meta, :inline), :(((~m) & x) | (m & y)))
        end
        bits = 32W
        decl32 = "declare <$W x i32> @llvm.x86.avx512.mask.pternlog.d.$(bits)(<$W x i32>, <$W x i32>, <$W x i32>, i32, i16)"
        instr32 = """
            %res = call <$W x i32> @llvm.x86.avx512.mask.pternlog.d.$(bits)(<$W x i32> %0, <$W x i32> %1, <$W x i32> %2, i32 $L, i16 -1)
            ret <$W x i32> %res
        """
        arg_syms = [:(data(m)), :(data(x)), :(data(y))]
        llvmcall_expr(decl32, instr32, :(_Vec{$W,UInt32}), :(Tuple{_Vec{$W,UInt32},_Vec{$W,UInt32},_Vec{$W,UInt32}}), "<$W x i32>", ["<$W x i32>", "<$W x i32>", "<$W x i32>"], arg_syms)
    end
    # @eval @generated function setbits(x::Vec{W,T}, y::Vec{W,T}, m::Vec{W,T}) where {W,T <: Union{UInt32,UInt64}}
    #     ex = if W*sizeof(T) ∈ (16,32,64)
    #         :(vpternlog(x, y, m, Val{216}()))
    #     else
    #         :((x & m) | y)
    #     end
    #     Expr(:block, Expr(:meta, :inline), ex)
    # end
    @generated function bitselect(m::Vec{W,T}, x::Vec{W,T}, y::Vec{W,T}) where {W,T <: Union{UInt32,UInt64}}
        ex = if W*sizeof(T) ∈ (16,32,64)
            :(vpternlog(m, x, y, Val{172}()))
        else
            :(((~m) & x) | (m & y))
        end
        Expr(:block, Expr(:meta, :inline), ex)
    end
    @inline Base.copysign(v1::Vec{W,Float64}, v2::Vec{W,Float64}) where {W} = reinterpret(Float64, bitselect(Vec{W,UInt64}(0x8000000000000000), reinterpret(UInt64, v1), reinterpret(UInt64, v2)))
    @inline Base.copysign(v1::Vec{W,Float32}, v2::Vec{W,Float32}) where {W} = reinterpret(Float32, bitselect(Vec{W,UInt32}(0x80000000), reinterpret(UInt32, v1), reinterpret(UInt32, v2)))
end


for (op,f) ∈ [("minnum",:min),("maxnum",:max),("copysign",:copysign),
              ]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, W, T, (W for _ in 1:2), (T for _ in 1:2), "nsz arcp contract afn reassoc")
    end
end
@inline _signbit(v::Vec{W, I}) where {W, I<:Signed} = v & Vec{W,I}(typemin(I))
@inline Base.copysign(v1::Vec{W,I}, v2::Vec{W,I}) where {W, I <: Signed} = ifelse(_signbit(v1) == _signbit(v2), v1, -v1)

@inline Base.copysign(x::Float32, v::Vec{W}) where {W} = copysign(vbroadcast(Val{W}(), x), v)
@inline Base.copysign(x::Float64, v::Vec{W}) where {W} = copysign(vbroadcast(Val{W}(), x), v)
@inline Base.copysign(x::Float32, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = copysign(vbroadcast(Val{W}(), x), v)
@inline Base.copysign(x::Float64, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = copysign(vbroadcast(Val{W}(), x), v)
@inline Base.copysign(v::Vec, u::VecUnroll) = VecUnroll(fmap(copysign, v, u.data))
@inline Base.copysign(v::Vec{W,T}, x::NativeTypes) where {W,T} = copysign(v, Vec{W,T}(x))
@inline Base.copysign(v1::Vec{W,T}, v2::Vec{W}) where {W,T} = copysign(v1, convert(Vec{W,T}, v2))
@inline Base.copysign(v1::Vec{W,T}, ::Vec{W,<:Unsigned}) where {W,T} = abs(v1)
@inline Base.copysign(s::IntegerTypesHW, v::Vec{W}) where {W} = copysign(vbroadcast(Val{W}(), s), v)

# ternary
for (op,f) ∈ [("fma",:fma),("fmuladd",:muladd)]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, W, T, (W for _ in 1:3), (T for _ in 1:3), $(f === :fma ? nothing : "nsz arcp contract afn reassoc"))
    end
end
# floating vector, integer scalar
# @generated function Base.:(^)(v1::Vec{W,T}, v2::Int32) where {W, T <: Union{Float32,Float64}}
#     llvmcall_expr("powi", W, T, (W, 1), (T, Int32), "nsz arcp contract afn reassoc")
# end
for (op,f) ∈ [
    ("experimental.vector.reduce.v2.fadd",:vsum),
    ("experimental.vector.reduce.v2.fmul",:vprod)
]
    @eval @generated function $f(v1::T, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, -1, T, (1, W), (T, T), "nsz arcp contract afn reassoc")
    end
end
vsum(s::T, v::Vec{W,T}) where {W,T} = Base.FastMath.add_fast(s, vsum(v))
vprod(s::T, v::Vec{W,T}) where {W,T} = Base.FastMath.mul_fast(s, vprod(v))
for (op,f) ∈ [
    ("experimental.vector.reduce.fmax",:vmaximum),
    ("experimental.vector.reduce.fmin",:vminimum)
]
    @eval @generated function $f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, -1, T, (W,), (T,), "nsz arcp contract afn reassoc")
    end
end
for (op,f,S) ∈ [
    ("experimental.vector.reduce.add",:vsum,:Integer),
    ("experimental.vector.reduce.mul",:vprod,:Integer),
    ("experimental.vector.reduce.and",:vall,:Integer),
    ("experimental.vector.reduce.or",:vany,:Integer),
    ("experimental.vector.reduce.xor",:vxorreduce,:Integer),
    ("experimental.vector.reduce.smax",:vmaximum,:Signed),
    ("experimental.vector.reduce.smin",:vminimum,:Signed),
    ("experimental.vector.reduce.umax",:vmaximum,:Unsigned),
    ("experimental.vector.reduce.umin",:vminimum,:Unsigned)
]
    @eval @generated function $f(v1::Vec{W,T}) where {W, T <: $S}
        llvmcall_expr($op, -1, T, (W,), (T,))
    end
end

#         W += W
#     end
# end
@inline vsum(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = vsum(-zero(T), v)
@inline vprod(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = vprod(one(T), v)
@inline vsum(x, v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = vsum(convert(T, x), v)
@inline vprod(x, v::Vec{W,T}) where {W,T<:Union{Float32,Float64}} = vprod(convert(T, x), v)

for (f,f_to,op,reduce,twoarg) ∈ [
    (:reduced_add,:reduce_to_add,:+,:vsum,true),(:reduced_prod,:reduce_to_prod,:*,:vprod,true),
    (:reduced_max,:reduce_to_max,:max,:vmaximum,false),(:reduced_min,:reduce_to_min,:min,:vminimum,false)
]
    @eval begin
        @inline $f_to(x::NativeTypes, y::NativeTypes) = x
        @inline $f_to(x::AbstractSIMD, y::AbstractSIMD) = x
        @inline $f_to(x::AbstractSIMD, y::NativeTypes) = $reduce(x)
        @inline $f(x::NativeTypes, y::NativeTypes) = $op(x,y)
        @inline $f(x::AbstractSIMD, y::AbstractSIMD) = $op(x,y)
    end
    if twoarg
        # @eval @inline $f(y::T, x::AbstractSIMD{W,T}) where {W,T} = $reduce(y, x)
        @eval @inline $f(x::AbstractSIMD, y::NativeTypes) = $reduce(y, x)
        # @eval @inline $f(x::AbstractSIMD, y::NativeTypes) = ((y2,x2,r) = @show (y, x, $reduce(y, x)); r)
    else
        # @eval @inline $f(y::T, x::AbstractSIMD{W,T}) where {W,T} = $op(y, $reduce(x))
        @eval @inline $f(x::AbstractSIMD, y::NativeTypes) = $op(y, $reduce(x))
    end
end


@inline roundint(x::Float32) = round(Int32, x)
if AVX512DQ
    @inline roundint(x::Float64) = round(Int, x)
    @inline roundint(v::Vec{W,Float32}) where {W} = round(Int32, v)
    @inline roundint(v::Vec{W,Float64}) where {W} = round(Int64, v)
else
    @inline roundint(x::Float64) = round(Int32, x)
    @inline roundint(v::Vec{W}) where {W} = round(Int32, v)
end
# binary

function count_zeros_func(W, I, op, tf = 1)
    typ = "i$(8sizeof(I))"
    vtyp = "<$W x $typ>"
    instr = "@llvm.$op.v$(W)$(typ)"
    decl = "declare $vtyp $instr($vtyp, i1)"
    instrs = "%res = call $vtyp $instr($vtyp %0, i1 $tf)\nret $vtyp %res"
    llvmcall_expr(decl, instrs, _Vec{W,I}, Tuple{_Vec{W,I}}, vtyp, (vtyp,), (:(data(v)),))
end
# @generated Base.abs(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "abs", 0)
@generated Base.leading_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "ctlz")
@generated Base.trailing_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "cttz")



for (op,f) ∈ [("ctpop", :count_ones)]
    @eval @generated Base.$f(v1::Vec{W,T}) where {W,T} = llvmcall_expr($op, W, T, (W,), (T,))
end

for (op,f) ∈ [("fshl",:funnel_shift_left),("fshr",:funnel_shift_right)
              ]
    @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W,T}
        llvmcall_expr($op, W, T, (W for _ in 1:3), (T for _ in 1:3))
    end
end
@inline function funnel_shift_left(a::T, b::T, c::T) where {T}
    _T = eltype(a)
    S = 8sizeof(_T) % _T
    (a << c) | (b >>> (S - c))
end
@inline function funnel_shift_right(a::T, b::T, c::T) where {T}
    _T = eltype(a)
    S = 8sizeof(_T) % _T
    (a >>> c) | (b << (S - c))
end
@inline function funnel_shift_left(_a, _b, _c)
    a, b, c = promote(_a, _b, _c)
    funnel_shift_left(a, b, c)
end
@inline function funnel_shift_right(_a, _b, _c)
    a, b, c = promote(_a, _b, _c)
    funnel_shift_right(a, b, c)
end
@inline rotate_left(a::T, b::T) where {T} = funnel_shift_left(a, a, b)
@inline rotate_right(a::T, b::T) where {T} = funnel_shift_right(a, a, b)
@inline function rotate_left(_a, _b)
    a, b = promote_div(_a, _b)
    funnel_shift_left(a, a, b)
end
@inline function rotate_right(_a, _b)
    a, b = promote_div(_a, _b)
    funnel_shift_right(a, a, b)
end

# for T ∈ [UInt8,UInt16,UInt32,UInt64]
#     bytes = sizeof(T)
#     W = 2
#     while W * bytes ≤ REGISTER_SIZE

#         for (op,f) ∈ [("ctpop", :count_ones)]
#             @eval @inline Base.$f(v1::Vec{$W,$T}) = $(llvmcall_expr(op, W, T, (W,), (T,)))
#         end
        
#         for (op,f) ∈ [("fshl",:funnel_shift_left),("fshr",:funnel_shift_right)
#                       ]
#             @eval @inline function $f(v1::Vec{$W,$T}, v2::Vec{$W,$T})
#                 $(llvmcall_expr(op, W, T, (W for _ in 1:3), (T for _ in 1:3)))
#             end
#         end

#         W += W
#     end
# end

@inline vfmadd(a, b, c) = muladd(a, b, c)
@inline vfnmadd(a, b, c) = muladd(-a, b, c)
@inline vfmsub(a, b, c) = muladd(a, b, -c)
@inline vfnmsub(a, b, c) = -muladd(a, b, c)

@inline vfmadd231(a, b, c) = vfmadd(a, b, c)
@inline vfnmadd231(a, b, c) = vfnmadd(a, b, c)
@inline vfmsub231(a, b, c) = vfmsub(a, b, c)
@inline vfnmsub231(a, b, c) = vfnmsub(a, b, c)

if FMA
    @eval begin
        @generated function vfmadd231(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
            if !(ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE) && (W ≥ (T === Float32 ? 4 : 2)))
                return Expr(:block, Expr(:meta, :inline), :(vfmadd(a, b, c)))
            end
            typ = LLVM_TYPES[T]
            suffix = T == Float32 ? "ps" : "pd"
            vfmadd_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            quote
                $(Expr(:meta, :inline))
                Vec(llvmcall($vfmadd_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
            end
        end
        @generated function vfnmadd231(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
            if !(ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE) && (W ≥ (T === Float32 ? 4 : 2)))
                return Expr(:block, Expr(:meta, :inline), :(vfnmadd(a, b, c)))
            end
            typ = LLVM_TYPES[T]
            suffix = T == Float32 ? "ps" : "pd"
            vfnmadd_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            quote
                $(Expr(:meta, :inline))
                Vec(llvmcall($vfnmadd_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
            end
        end
        @generated function vfmsub231(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
            if !(ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE) && (W ≥ (T === Float32 ? 4 : 2)))
                return Expr(:block, Expr(:meta, :inline), :(vfmsub(a, b, c)))
            end
            typ = LLVM_TYPES[T]
            suffix = T == Float32 ? "ps" : "pd"
            vfmsub_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            quote
                $(Expr(:meta, :inline))
                Vec(llvmcall($vfmsub_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
            end
        end
        @generated function vfnmsub231(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
            if !(ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE) && (W ≥ (T === Float32 ? 4 : 2)))
                return Expr(:block, Expr(:meta, :inline), :(vfnmsub(a, b, c)))
            end
            typ = LLVM_TYPES[T]
            suffix = T == Float32 ? "ps" : "pd"
            vfnmsub_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            quote
                $(Expr(:meta, :inline))
                Vec(llvmcall($vfnmsub_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
            end
        end
    end
    if AVX512BW
        @eval begin
            @generated function ifelse(::typeof(vfmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(ifelse(vfmadd, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfmaddmask_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                                ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfmaddmask_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function ifelse(::typeof(vfnmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(ifelse(vfmmadd, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfnmaddmask_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfnmaddmask_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function ifelse(::typeof(vfmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(ifelse(vfmsub, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfmsubmask_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfmsubmask_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function ifelse(::typeof(vfnmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(ifelse(vfnmsub, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfnmsubmask_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfnmsubmask_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
        end
    end
end
    # for T ∈ [Float32,Float64]
    #     W = 16 ÷ sizeof(T)
    #     local suffix = T == Float32 ? "ps" : "pd"
    #     typ = LLVM_TYPES[T]
    #     while W <= VectorizationBase.REGISTER_SIZE ÷ sizeof(T)
    #         vfmadd_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
    #             ret <$W x $(typ)> %res"""
    #         vfnmadd_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
    #             ret <$W x $(typ)> %res"""
    #         vfmsub_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
    #             ret <$W x $(typ)> %res"""
    #         vfnmsub_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
    #             ret <$W x $(typ)> %res"""
    #         @eval begin
    #             @inline function vfmadd231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                 Vec(llvmcall($vfmadd_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
    #             end
    #             @inline function vfnmadd231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                 Vec(llvmcall($vfnmadd_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
    #             end
    #             @inline function vfmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                 Vec(llvmcall($vfmsub_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
    #             end
    #             @inline function vfnmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                 Vec(llvmcall($vfnmsub_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
    #             end
    #         end
    #         if VectorizationBase.AVX512BW && W ≥ 8
    #             vfmaddmask_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
    #                 ret <$W x $(typ)> %res"""
    #             vfnmaddmask_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
    #                 ret <$W x $(typ)> %res"""
    #             vfmsubmask_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
    #                 ret <$W x $(typ)> %res"""
    #             vfnmsubmask_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
    #                 ret <$W x $(typ)> %res"""
    #             U = VectorizationBase.mask_type(W)
    #             @eval begin
    #                 @inline function ifelse(::typeof(vfmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function ifelse(::typeof(vfnmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfnmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function ifelse(::typeof(vfmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function ifelse(::typeof(vfnmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfnmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #             end
    #         end
    #         W += W
    #     end
# end

@inline ifelse(f::F, m::Mask, a::Vararg{Any,K}) where {F<:Function,K} = ifelse(m, f(a...), a[K])

"""
Fast approximate reciprocal.

Guaranteed accurate to at least 2^-14 ≈ 6.103515625e-5.

Useful for special funcion implementations.
"""
@inline inv_approx(x) = Base.FastMath.inv_fast(x)
@inline inv_approx(v::VecUnroll) = VecUnroll(fmap(inv_approx, v.data))
@generated function inv_approx(v::Vec{W,T}) where {W, T <: Union{Float32, Float64}}
    ((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)) || return Expr(:block, Expr(:meta, :inline), :(inv(v)))
    bits = 8sizeof(T) * W
    if (AVX512F && (bits === 512)) || (AVX512VL && (bits ∈ (128, 256)))
        typ = T === Float64 ? "double" : "float"
        vtyp = "<$W x $(typ)>"
        dors = T === Float64 ? "d" : "s"
        f = "@llvm.x86.avx512.rcp14.p$(dors).$(bits)"
        decl = "declare $(vtyp) $f($(vtyp), $(vtyp), i$(min(8,W))) nounwind readnone"
        instrs = "%res = call $(vtyp) $f($vtyp %0, $vtyp zeroinitializer, i$(min(8,W)) -1)\nret $(vtyp) %res"
        return llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], [:(data(v))])
    end
    if (AVX && (W == 8)) && (T === Float32)
        decl = "declare <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float>) nounwind readnone"
        instrs = "%res = call <8 x float> @llvm.x86.avx.rcp.ps.256(<8 x float> %0)\nret <8 x float> %res"
        return llvmcall_expr(decl, instrs, :(_Vec{8,Float32}), :(Tuple{_Vec{8,Float32}}), "<8 x float>", ["<8 x float>"], [:(data(v))])
    elseif W == 4
        decl = "declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>) nounwind readnone"
        instrs = "%res = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %0)\nret <4 x float> %res"
        if T === Float32
            return llvmcall_expr(decl, instrs, :(_Vec{4,Float32}), :(Tuple{_Vec{4,Float32}}), "<4 x float>", ["<4 x float>"], [:(data(v))])
        else#if T === Float64
            argexpr = [:(data(convert(Float32, v)))]
            call = llvmcall_expr(decl, instrs, :(_Vec{4,Float32}), :(Tuple{_Vec{4,Float32}}), "<4 x float>", ["<4 x float>"], argexpr, true)
            return Expr(:block, Expr(:meta, :inline), :(convert(Float64, $call)))
        end
    end
    Expr(:block, Expr(:meta, :inline), :(inv(v)))
end

