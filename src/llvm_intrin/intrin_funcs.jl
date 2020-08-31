

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
for (op,f,S) ∈ [("smax",:max,:Signed),("smin",:min,:Signed)]
    @eval @generated Base.$f(v1::Vec{W,T}) where {W, T <: $S} = llvmcall_expr($op, W, T, (W,), (T,))
end
# for T ∈ [Float32, Float64]
#     W = 2
#     while W * sizeof(T) ≤ REGISTER_SIZE
# floating point
for (op,f) ∈ [("sqrt",:sqrt),("fabs",:abs),("floor",:floor),("ceil",:ceil),("trunc",:trunc),("round",:round)
              ]
    @eval @generated Base.$f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}} = llvmcall_expr($op, W, T, (W,), (T,), "fast")
end

@generated function Base.round(::Type{Int64}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    llvmcall_expr("lrint", W, Int64, (W,), (T,), "fast")
end
@generated function Base.round(::Type{Int32}, v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    llvmcall_expr("lrint", W, Int32, (W,), (T,), "fast")
end

for (op,f) ∈ [("minnum",:min),("maxnum",:max),("copysign",:copysign),
              ]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W, T}
        llvmcall_expr($op, W, T, (W for _ in 1:2), (T for _ in 1:2), "fast")
    end
end
# ternary
for (op,f) ∈ [("fma",:fma),("fmuladd",:muladd)]
    @eval @generated function Base.$f(v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, W, T, (W for _ in 1:3), (T for _ in 1:3), $(f === :fma ? nothing : "fast"))
    end
end
# floating vector, integer scalar
@generated function Base.:(^)(v1::Vec{W,T}, v2::Int32) where {W, T <: Union{Float32,Float64}}
    llvmcall_expr("powi", W, T, (W, 1), (T, Int32), "fast")
end
for (op,f) ∈ [
    ("experimental.vector.reduce.v2.fadd",:vsum),
    ("experimental.vector.reduce.v2.fmul",:vprod)
]
    @eval @generated function $f(v1::T, v2::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, 1, T, (1, W), (T, T), "fast")
    end
end
for (op,f) ∈ [
    ("experimental.vector.reduce.v2.fmax",:vmaximum),
    ("experimental.vector.reduce.v2.fmin",:vminimum)
]
    @eval @generated function $f(v1::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
        llvmcall_expr($op, 1, T, (W,), (T,), "fast")
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
        llvmcall_expr($op, 1, T, (W,), (T,))
    end
end
        
#         W += W
#     end
# end

@inline vsum(v::Vec{W,T}) where {W,T} = vsum(-zero(T), v)
@inline vprod(v::Vec{W,T}) where {W,T} = vprod(one(T), v)

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
@generated vleading_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "ctlz")
@generated vtrailing_zeros(v::Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "cttz")



for (op,f) ∈ [("ctpop", :count_ones)]
    @eval @generated Base.$f(v1::Vec{W,T}) where {W,T} = llvmcall_expr($op, W, T, (W,), (T,))
end

for (op,f) ∈ [("fshl",:funnel_shift_left),("fshr",:funnel_shift_right)
              ]
    @eval @generated function $f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T}
        llvmcall_expr($op, W, T, (W for _ in 1:3), (T for _ in 1:3))
    end
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
            @generated function vifelse(::typeof(vfmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U,T}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmadd, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfmaddmask_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                                ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function vifelse(::typeof(vfnmadd231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U,T}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmmadd, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfnmaddmask_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfnmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function vifelse(::typeof(vfmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U,T}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(vifelse(vfmsub, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfmsubmask_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                end
            end
            @generated function vifelse(::typeof(vfnmsub231), m::Mask{W,U}, a::Vec{W,T}, b::Vec{W,T}, c::Vec{W,T}) where {W,U,T}
                if !((W ≥ 8) && ispow2(W) && (W * sizeof(T) ≤ REGISTER_SIZE))
                    return Expr(:block, Expr(:meta, :inline), :(vifelse(vfnmsub, m, a, b, c)))
                end
                typ = LLVM_TYPES[T]
                suffix = T == Float32 ? "ps" : "pd"                    
                vfnmsubmask_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                            ret <$W x $(typ)> %res"""
                quote
                    $(Expr(:meta,:inline))
                    Vec(llvmcall($vfnmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
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
    #                 @inline function vifelse(::typeof(vfmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function vifelse(::typeof(vfnmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfnmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function vifelse(::typeof(vfmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #                 @inline function vifelse(::typeof(vfnmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
    #                     Vec(llvmcall($vfnmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
    #                 end
    #             end
    #         end
    #         W += W
    #     end
    # end
@inline vifelse(f::F, m::Mask, a::Vararg{<:Any,K}) where {F,K} = vifelse(m, f(a...), a[K])
