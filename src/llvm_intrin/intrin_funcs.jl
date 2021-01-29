
@generated function saturated_add(x::I, y::I) where {I <: IntegerTypesHW}
    typ = "i$(8sizeof(I))"
    s = I <: Signed ? 's' : 'u'
    f = "@llvm.$(s)add.sat.$typ"
    decl = "declare $typ $f($typ, $typ)"
    instrs = """
        %res = call $typ $f($typ %0, $typ %1)
        ret $typ %res
    """
    llvmcall_expr(decl, instrs, JULIA_TYPES[I], :(Tuple{$I,$I}), typ, [typ,typ], [:x,:y])
end
@generated function saturated_add(x::Vec{W,I}, y::Vec{W,I}) where {W,I}
    typ = "i$(8sizeof(I))"
    vtyp = "<$W x i$(typ)>"
    s = I <: Signed ? 's' : 'u'
    f = "@llvm.$(s)add.sat.$(suffix(W,typ))"
    decl = "declare $vtyp $f($typ, $typ)"
    instrs = """
        %res = call $vtyp $f($vtyp %0, $vtyp %1)
        ret $vtyp %res
    """
    llvmcall_expr(decl, instrs, :(_Vec{$W,$I}), :(Tuple{_Vec{$W,$I},_Vec{$W,$I}}), vtyp, [vtyp,vtyp], [:(data(x)),:(data(y))])
end

@eval @inline function assume(b::Bool)
    $(llvmcall_expr("declare void @llvm.assume(i1)", "%b = trunc i8 %0 to i1\ncall void @llvm.assume(i1 %b)\nret void", :Cvoid, :(Tuple{Bool}), "void", ["i8"], [:b]))
end

# Doesn't work, presumably because the `noalias` doesn't propagate outside the function boundary.
# @generated function noalias!(ptr::Ptr{T}) where {T <: NativeTypes}
#     Base.libllvm_version ≥ v"11" || return :ptr
#     typ = LLVM_TYPES[T]
#     # if Base.libllvm_version < v"10"
#     #     funcname = "noalias" * typ
#     #     decls = "define noalias $typ* @$(funcname)($typ *%a) willreturn noinline { ret $typ* %a }"
#     #     instrs = """
#     #         %ptr = inttoptr $ptyp %0 to $typ*
#     #         %naptr = call $typ* @$(funcname)($typ* %ptr)
#     #         %jptr = ptrtoint $typ* %naptr to $ptyp
#     #         ret $ptyp %jptr
#     #     """
#     # else
#     decls = "declare void @llvm.assume(i1)"
#     instrs = """
#                 %ptr = inttoptr $(JULIAPOINTERTYPE) %0 to $typ*
#                 call void @llvm.assume(i1 true) ["noalias"($typ* %ptr)]
#                 %int = ptrtoint $typ* %ptr to $(JULIAPOINTERTYPE)
#                 ret $(JULIAPOINTERTYPE) %int
#             """
#     llvmcall_expr(decls, instrs, :(Ptr{$T}), :(Tuple{Ptr{$T}}), JULIAPOINTERTYPE, [JULIAPOINTERTYPE], [:ptr])
# end
# @inline noalias!(x) = x

# @eval @inline function expect(b::Bool)
#     $(llvmcall_expr("declare i1 @llvm.expect.i1(i1, i1)", """
#     %b = trunc i8 %0 to i1
#     %actual = call i1 @llvm.expect.i1(i1 %b, i1 true)
#     %byte = zext i1 %actual to i8
#     ret i8 %byte""", :Bool, :(Tuple{Bool}), "i8", ["i8"], [:b]))
# end
# @generated function expect(i::I, ::Val{N}) where {I <: Integer, N}
#     ityp = 'i' * string(8sizeof(I))
#     llvmcall_expr("declare i1 @llvm.expect.$ityp($ityp, i1)", """
#     %actual = call $ityp @llvm.expect.$ityp($ityp %0, $ityp $N)
#     ret $ityp %actual""", I, :(Tuple{$I}), ityp, [ityp], [:i])
# end

# for (op,f) ∈ [("abs",:abs)]
# end
if Base.libllvm_version ≥ v"12"
    for (op,f,S) ∈ [("smax",:max,:Signed),("smin",:min,:Signed),("umax",:max,:Unsigned),("umin",:min,:Unsigned)]
        vf = Symbol(:v,f)
        @eval @generated $vf(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T <: $S} = (TS = JULIA_TYPES[T]; build_llvmcall_expr($op, W, TS, [W, W], [TS, TS]))
    end
else
    @inline vmax(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = vifelse(v1 > v2, v1, v2)
    @inline vmin(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = vifelse(v1 < v2, v1, v2)
end
@inline vmax_fast(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = vmax(v1, v2)
@inline vmin_fast(v1::Vec{W,<:Integer}, v2::Vec{W,<:Integer}) where {W} = vmin(v1, v2)
     
# floating point
for (op,f) ∈ [("sqrt",:vsqrt),("fabs",:vabs),("floor",:vfloor),("ceil",:vceil),("trunc",:vtrunc),("nearbyint",:vround)
              ]
    # @eval @generated Base.$f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = llvmcall_expr($op, W, T, (W,), (T,), "nsz arcp contract afn reassoc")
    @eval @generated $f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = (TS = T === Float32 ? :Float32 : :Float64; build_llvmcall_expr($op, W, TS, [W], [TS], "fast"))
end
@inline vsqrt(v::AbstractSIMD{W,T}) where {W,T<:IntegerTypes} = vsqrt(float(v))

# @generated function Base.round(::Type{Int64}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
#     llvmcall_expr("lrint", W, Int64, (W,), (T,), "")
# end
# @generated function Base.round(::Type{Int32}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
#     llvmcall_expr("lrint", W, Int32, (W,), (T,), "")
# end
@inline vtrunc(::Type{I}, v::AbstractSIMD{W,T}) where {W, I<:IntegerTypesHW, T <: NativeTypes} = vconvert(I, v)
for f ∈ [:vround, :vfloor, :vceil]
    @eval @inline $f(::Type{I}, v::AbstractSIMD{W,T}) where {W,I<:IntegerTypesHW,T <: NativeTypes} = vconvert(I, $f(v))
end
for f ∈ [:vtrunc, :vround, :vfloor, :vceil]
    @eval @inline $f(v::AbstractSIMD{W,I}) where {W,I<:IntegerTypesHW} = v
end

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

# AVX512 lets us use 1 instruction instead of 2 dependent instructions to set bits
@generated function vpternlog(m::Vec{W,UInt64}, x::Vec{W,UInt64}, y::Vec{W,UInt64}, ::Val{L}) where {W, L}
    @assert has_feature("x86_64_avx512f")
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
    @assert has_feature("x86_64_avx512f")
    @assert W ∈ (4,8,16)
    # if W ∉ (4,8,16)
    #     return Expr(:block, Expr(:meta, :inline), :(((~m) & x) | (m & y)))
    # end
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
    ex = if !has_feature("x86_64_avx512f")
        :(((~m) & x) | (m & y))
    elseif W*sizeof(T) ∈ (16,32,64)
        :(vpternlog(m, x, y, Val{172}()))
    else
        :(((~m) & x) | (m & y))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end
@generated function vcopysign(v1::Vec{W,Float64}, v2::Vec{W,Float64}) where {W}
    ex = if !has_feature("x86_64_avx512f")
        :(llvm_copysign(v1, v2))
    else
        :(reinterpret(Float64, bitselect(Vec{W,UInt64}(0x8000000000000000), reinterpret(UInt64, v1), reinterpret(UInt64, v2))))
    end
    Expr(:block, Expr(:meta,:inline), ex)
end
@generated function vcopysign(v1::Vec{W,Float32}, v2::Vec{W,Float32}) where {W}
    ex = if !has_feature("x86_64_avx512f")
        :(llvm_copysign(v1, v2))
    else
        :(reinterpret(Float32, bitselect(Vec{W,UInt32}(0x80000000), reinterpret(UInt32, v1), reinterpret(UInt32, v2))))
    end
    Expr(:block, Expr(:meta,:inline), ex)
end

for (op,f,fast) ∈ [
    ("minnum",:vmin,false),("minnum",:vmin_fast,true),
    ("maxnum",:vmax,false),("maxnum",:vmax_fast,true),
    ("copysign",:llvm_copysign,true)
]
    ff = fast_flags(fast)
    fast && (ff *= " nnan")
    @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        TS = T === Float32 ? :Float32 : :Float64
        build_llvmcall_expr($op, W, TS, [W, W], [TS, TS], $ff)
    end
end
@inline _signbit(v::Vec{W, I}) where {W, I<:Signed} = v & Vec{W,I}(typemin(I))
@inline vcopysign(v1::Vec{W,I}, v2::Vec{W,I}) where {W, I <: Signed} = vifelse(_signbit(v1) == _signbit(v2), v1, -v1)

@inline vcopysign(x::Float32, v::Vec{W}) where {W} = vcopysign(vbroadcast(Val{W}(), x), v)
@inline vcopysign(x::Float64, v::Vec{W}) where {W} = vcopysign(vbroadcast(Val{W}(), x), v)
@inline vcopysign(x::Float32, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = vcopysign(vbroadcast(Val{W}(), x), v)
@inline vcopysign(x::Float64, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = vcopysign(vbroadcast(Val{W}(), x), v)
@inline vcopysign(v::Vec, u::VecUnroll) = VecUnroll(fmap(vcopysign, v, u.data))
@inline vcopysign(v::Vec{W,T}, x::NativeTypes) where {W,T} = vcopysign(v, Vec{W,T}(x))
@inline vcopysign(v1::Vec{W,T}, v2::Vec{W}) where {W,T} = vcopysign(v1, convert(Vec{W,T}, v2))
@inline vcopysign(v1::Vec{W,T}, ::Vec{W,<:Unsigned}) where {W,T} = vabs(v1)
@inline vcopysign(s::IntegerTypesHW, v::Vec{W}) where {W} = vcopysign(vbroadcast(Val{W}(), s), v)
@inline vcopysign(v::Vec, s::UnsignedHW) = vabs(v)
@inline vcopysign(v::VecUnroll, s::UnsignedHW) = vabs(v)
@inline vcopysign(v::VecUnroll{N,W,T}, s::NativeTypes) where {N,W,T} = VecUnroll(fmap(vcopysign, v.data, vbroadcast(Val{W}(), s)))

for f ∈ [:vmax, :vmax_fast, :vmin, :vmin_fast]
    @eval begin
        @inline function $f(a::Union{FloatingTypes,Vec{<:Any,<:FloatingTypes}}, b::Union{FloatingTypes,Vec{<:Any,<:FloatingTypes}})
            c, d = promote(a, b)
            $f(c, d)
        end
        @inline function $f(a::Union{FloatingTypes,Vec{<:Any,<:FloatingTypes}}, b::Union{NativeTypes,Vec{<:Any,<:NativeTypes}})
            c, d = promote(a, b)
            $f(c, d)
        end
        @inline function $f(a::Union{NativeTypes,Vec{<:Any,<:NativeTypes}}, b::Union{FloatingTypes,Vec{<:Any,<:FloatingTypes}})
            c, d = promote(a, b)
            $f(c, d)
        end
        @inline $f(v::Vec{W,<:IntegerTypesHW}, s::IntegerTypesHW) where {W} = $f(v, vbroadcast(Val{W}(), s))
        @inline $f(s::IntegerTypesHW, v::Vec{W,<:IntegerTypesHW}) where {W} = $f(vbroadcast(Val{W}(), s), v)
    end
end

# ternary
for (op,f,fast) ∈ [
    ("fma",:vfma,false),("fma",:vfma_fast,true),
    ("fmuladd",:vmuladd,false),("fmuladd",:vmuladd_fast,true)
]
    @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W, T <: FloatingTypes}
        TS = T === Float32 ? :Float32 : :Float64
        # TS = JULIA_TYPES[T]
        build_llvmcall_expr($op, W, TS, [W, W, W], [TS, TS, TS], $(fast_flags(fast)))
    end
end
# @inline Base.fma(a::Vec, b::Vec, c::Vec) = vfma(a,b,c)
# @inline Base.muladd(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,T<:FloatingTypes} = vmuladd(a,b,c)
# Generic fallbacks
@inline vfma(a::NativeTypes, b::NativeTypes, c::NativeTypes) = fma(a,b,c)
@inline vmuladd(a::NativeTypes, b::NativeTypes, c::NativeTypes) = muladd(a,b,c)
@inline vfma_fast(a::NativeTypes, b::NativeTypes, c::NativeTypes) = fma(a,b,c)
@inline vmuladd_fast(a::NativeTypes, b::NativeTypes, c::NativeTypes) = muladd(a,b,c)
@inline vfma(a, b, c) = fma(a,b,c)
@inline vmuladd(a, b, c) = muladd(a,b,c)
@inline vfma_fast(a, b, c) = fma(a,b,c)
@inline vmuladd_fast(a, b, c) = muladd(a,b,c)
for f ∈ [:vfma, :vmuladd, :vfma_fast, :vmuladd_fast]
    @eval @inline function $f(v1::AbstractSIMD{W,T}, v2::AbstractSIMD{W,T}, v3::AbstractSIMD{W,T}) where {W,T <: IntegerTypesHW}
        vadd(vmul(v1, v2), v3)
    end
end

# vfmadd -> muladd -> promotes arguments to hit definitions from VectorizationBase
# const vfmadd = FMA_FAST ? vfma : vmuladd
@generated function vfmadd(a, b, c)
    ex = if fma_fast()
        :(vfma(a, b, c))
    else
        :(vmuladd(a, b, c))
    end
    Expr(:block, Expr(:meta,:inline), ex)
end
@inline vfnmadd(a, b, c) = vfmadd(-a, b, c)
@inline vfmsub(a, b, c) = vfmadd(a, b, -c)
@inline vfnmsub(a, b, c) = -vfmadd(a, b, c)
# const vfmadd_fast = FMA_FAST ? vfma_fast : vmuladd_fast
@generated function vfmadd_fast(a, b, c)
    ex = if fma_fast()
        :(vfma_fast(a, b, c))
    else
        :(vmuladd_fast(a, b, c))
    end
    Expr(:block, Expr(:meta,:inline), ex)
end
@inline vfnmadd_fast(a, b, c) = vfmadd_fast(Base.FastMath.sub_fast(a), b, c)
@inline vfmsub_fast(a, b, c) = vfmadd_fast(a, b, Base.FastMath.sub_fast(c))
@inline vfnmsub_fast(a, b, c) = Base.FastMath.sub_fast(vfmadd_fast(a, b, c))
# floating vector, integer scalar
# @generated function Base.:(^)(v1::Vec{W,T}, v2::Int32) where {W, T <: Union{Float32,Float64}}
#     llvmcall_expr("powi", W, T, (W, 1), (T, Int32), "nsz arcp contract afn reassoc")
# end
for (op,f) ∈ [
    ("experimental.vector.reduce.v2.fadd",:vsum),
    ("experimental.vector.reduce.v2.fmul",:vprod)
]
    @eval @generated function $f(v1::T, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        # TS = JULIA_TYPES[T]
        TS = T === Float32 ? :Float32 : :Float64
        build_llvmcall_expr($op, -1, TS, [1, W], [TS, TS], "nsz arcp contract afn reassoc")
    end
end
@inline vsum(s::S, v::Vec{W,T}) where {W,T,S} = Base.FastMath.add_fast(s, vsum(v))
@inline vprod(s::S, v::Vec{W,T}) where {W,T,S} = Base.FastMath.mul_fast(s, vprod(v))
for (op,f) ∈ [
    ("experimental.vector.reduce.fmax",:vmaximum),
    ("experimental.vector.reduce.fmin",:vminimum)
]
    @eval @generated function $f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        # TS = JULIA_TYPES[T]
        TS = T === Float32 ? :Float32 : :Float64
        build_llvmcall_expr($op, -1, TS, [W], [TS], "nsz arcp contract afn reassoc")
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
        TS = JULIA_TYPES[T]
        build_llvmcall_expr($op, -1, TS, [W], [TS])
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
@generated function roundint(x::Float64)
    ex = if has_feature("x86_64_avx512dq")
        :(round(Int, x))
    else
        :(round(Int32, x))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end
@generated function roundint(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
    ex = if T === Float64 && has_feature("x86_64_avx512dq") && Sys.WORD_SIZE ≥ 64
        :(round(Int64, v))
    else
        :(round(Int32, v))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end    

# binary

function count_zeros_func(W, I, op, tf = 1)
    typ = "i$(8sizeof(I))"
    vtyp = "<$W x $typ>"
    instr = "@llvm.$op.v$(W)$(typ)"
    decl = "declare $vtyp $instr($vtyp, i1)"
    instrs = "%res = call $vtyp $instr($vtyp %0, i1 $tf)\nret $vtyp %res"
    rettypexpr = :(_Vec{$W,$I})
    llvmcall_expr(decl, instrs, rettypexpr, :(Tuple{$rettypexpr}), vtyp, [vtyp], [:(data(v))])
end
# @generated Base.abs(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "abs", 0)
@generated vleading_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "ctlz")
@generated vtrailing_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "cttz")



for (op,f) ∈ [("ctpop", :vcount_ones)]
    @eval @generated $f(v1::Vec{W,T}) where {W,T} = (TS = JULIA_TYPES[T]; build_llvmcall_expr($op, W, TS, [W], [TS]))
end

for (op,f) ∈ [("fshl",:funnel_shift_left),("fshr",:funnel_shift_right)
              ]
    @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W,T}
        TS = JULIA_TYPES[T]
        build_llvmcall_expr($op, W, TS, [W,W,W], [TS,TS,TS])
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
@inline funnel_shift_left(a::MM, b::MM, c::MM) = funnel_shift_left(Vec(a), Vec(b), Vec(c))
@inline funnel_shift_right(a::MM, b::MM, c::MM) = funnel_shift_right(Vec(a), Vec(b), Vec(c))
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

@inline vfmadd231(a, b, c) = vfmadd(a, b, c)
@inline vfnmadd231(a, b, c) = vfnmadd(a, b, c)
@inline vfmsub231(a, b, c) = vfmsub(a, b, c)
@inline vfnmsub231(a, b, c) = vfnmsub(a, b, c)
@generated function vfmadd231(a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    if !(has_feature("x86_64_fma") && ispow2(W) && (W * sizeof(T) ≤ register_size()) && (W ≥ (T === Float32 ? 4 : 2)))
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
    if !(has_feature("x86_64_fma") && ispow2(W) && (W * sizeof(T) ≤ register_size()) && (W ≥ (T === Float32 ? 4 : 2)))
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
    if !(has_feature("x86_64_fma") && ispow2(W) && (W * sizeof(T) ≤ register_size()) && (W ≥ (T === Float32 ? 4 : 2)))
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
    if !(has_feature("x86_64_fma") && ispow2(W) && (W * sizeof(T) ≤ register_size()) && (W ≥ (T === Float32 ? 4 : 2)))
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

@generated function vifelse(::typeof(vfmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
    if !(has_feature("x86_64_avx512bw") && (W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ register_size()))
        return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmadd, m, a, b, c)))
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
@generated function vifelse(::typeof(vfnmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
    if !(has_feature("x86_64_avx512bw") && (W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ register_size()))
        return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmmadd, m, a, b, c)))
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
@generated function vifelse(::typeof(vfmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
    if !(has_feature("x86_64_avx512bw") && (W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ register_size()))
        return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmsub, m, a, b, c)))
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
@generated function vifelse(::typeof(vfnmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U<:Unsigned,T<:Union{Float32,Float64}}
    if !(has_feature("x86_64_avx512bw") && (W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ register_size()))
        return Expr(:block, Expr(:meta, :inline), :(vifelse(vfnmsub, m, a, b, c)))
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


"""
    Fast approximate reciprocal.

    Guaranteed accurate to at least 2^-14 ≈ 6.103515625e-5.

    Useful for special funcion implementations.
    """
@inline inv_approx(x) = inv(x)
@inline inv_approx(v::VecUnroll) = VecUnroll(fmap(inv_approx, v.data))

function inv_approx_expr(W, @nospecialize(T), vector::Bool=true)
    ((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)) || return :(inv(v))
    bits = 8sizeof(T) * W
    pors = (vector | has_feature("x86_64_avx512f")) ? 'p' : 's'
    if (has_feature("x86_64_avx512f") && (bits === 512)) || (has_feature("x86_64_avx512vl") && (bits ∈ (128, 256)))
        typ = T === Float64 ? "double" : "float"
        vtyp = "<$W x $(typ)>"
        dors = T === Float64 ? "d" : "s"
        f = "@llvm.x86.avx512.rcp14.$(pors)$(dors).$(bits)"
        decl = "declare $(vtyp) $f($(vtyp), $(vtyp), i$(max(8,W))) nounwind readnone"
        instrs = "%res = call $(vtyp) $f($vtyp %0, $vtyp zeroinitializer, i$(max(8,W)) -1)\nret $(vtyp) %res"
        return llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], [:(data(v))], true)
    end
    if (has_feature("x86_64_avx") && (W == 8)) && (T === Float32)
        decl = "declare <8 x float> @llvm.x86.avx.rcp.$(pors)s.256(<8 x float>) nounwind readnone"
        instrs = "%res = call <8 x float> @llvm.x86.avx.rcp.$(pors)s.256(<8 x float> %0)\nret <8 x float> %res"
        return llvmcall_expr(decl, instrs, :(_Vec{8,Float32}), :(Tuple{_Vec{8,Float32}}), "<8 x float>", ["<8 x float>"], [:(data(v))], true)
    elseif W == 4
        decl = "declare <4 x float> @llvm.x86.sse.rcp.$(pors)s(<4 x float>) nounwind readnone"
        instrs = "%res = call <4 x float> @llvm.x86.sse.rcp.$(pors)s(<4 x float> %0)\nret <4 x float> %res"
        if T === Float32
            return llvmcall_expr(decl, instrs, :(_Vec{4,Float32}), :(Tuple{_Vec{4,Float32}}), "<4 x float>", ["<4 x float>"], [:(data(v))])
        else#if T === Float64
            argexpr = [:(data(convert(Float32, v)))]
            call = llvmcall_expr(decl, instrs, :(_Vec{4,Float32}), :(Tuple{_Vec{4,Float32}}), "<4 x float>", ["<4 x float>"], argexpr, true)
            return :(convert(Float64, $call))
        end
    elseif (has_feature("x86_64_avx512f") || (T === Float32)) && bits < 128
        L = 16 ÷ sizeof(T)
        inv_expr = inv_approx_expr(L, T, W > 1)
        resize_expr = W < 1 ? :(extractelement(v⁻¹, 0)) : :(vresize(Val{$W}(), v⁻¹))
        return quote
            v⁻¹ = let v = vresize(Val{$L}(), v)
                $inv_expr
            end
            $resize_expr
        end
    end
    :(inv(v))
end

@generated function inv_approx(v::Vec{W,T}) where {W, T <: Union{Float32, Float64}}
    Expr(:block, Expr(:meta, :inline), inv_approx_expr(W, T))
end
@generated function inv_approx(v::T) where {T <: Union{Float32, Float64}}
    Expr(:block, Expr(:meta, :inline), inv_approx_expr(0, T))
end

"""
vinv_fast(x)

More accurate version of inv_approx, using 1 (`Float32`) or 2 (`Float64`) Newton iterations to achieve reasonable accuracy.
Requires x86 CPUs for `Float32` support, and `AVX512F` for `Float64`. Otherwise, it falls back on `vinv(x)`.

y = 1 / x
Use a Newton iteration:
yₙ₊₁ = yₙ - f(yₙ)/f′(yₙ)
f(yₙ) = 1/yₙ - x
f′(yₙ) = -1/yₙ²
yₙ₊₁ = yₙ + (1/yₙ - x) * yₙ² = yₙ + yₙ - x * yₙ² = 2yₙ - x * yₙ² = yₙ * ( 2 - x * yₙ ) 
yₙ₊₁ = yₙ * ( 2 - x * yₙ )
"""
@inline vinv_fast(v) = vinv(v)
if ((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686))
    @inline function vinv_fast(v::AbstractSIMD{W,Float32}) where {W}
        v⁻¹ = inv_approx(v)
        vmul_fast(v⁻¹, vfnmadd_fast(v, v⁻¹, 2f0))
    end
    @generated function vinv_fast(v::AbstractSIMD{W,Float64}) where {W}
        ex = if !(has_feature("x86_64_avx512f"))
            :(vinv(v))
        else
            quote
                v⁻¹₁ = inv_approx(v)
                v⁻¹₂ = vmul_fast(v⁻¹₁, vfnmadd_fast(v, v⁻¹₁, 2.0))
                vmul_fast(v⁻¹₂, vfnmadd_fast(v, v⁻¹₂, 2.0))
            end
        end
        Expr(:block, Expr(:meta,:inline), ex)
    end
end
@inline vinv_fast(v::AbstractSIMD{<:Any,<:Integer}) = vinv_fast(float(v))

