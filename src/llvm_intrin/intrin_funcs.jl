

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

# for [("abs",:abs)]
# end
# for [("smax",:max),("smin",:min)]
    
# end
for T ∈ [Float32, Float64]
    W = 2
    while W * sizeof(T) ≤ REGISTER_SIZE
        # floating point
        for (op,f) ∈ [("sqrt",:sqrt),("fabs",:abs),("floor",:floor),("ceil",:ceil),("trunc",:trunc),("round",:round)
                      ]
            @eval @inline Base.$f(v1::Vec{$W,$T}) = $(llvmcall_expr(op, W, T, (W,), (T,), "fast"))
        end
        if W * sizeof(Int64) ≤ REGISTER_SIZE
            @eval @inline Base.round(::Type{Int64}, v1::Vec{$W,$T}) = $(llvmcall_expr("lrint", W, Int64, (W,), (T,), "fast"))
        end
        @eval @inline Base.round(::Type{Int32}, v1::Vec{$W,$T}) = $(llvmcall_expr("lrint", W, Int32, (W,), (T,), "fast"))

        for (op,f) ∈ [("minnum",:min),("maxnum",:max),("copysign",:copysign),
                  ]
            @eval @inline function Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T})
                $(llvmcall_expr(op, W, T, (W for _ in 1:2), (T for _ in 1:2), "fast"))
            end
        end
        # ternary
        for (op,f) ∈ [("fma",:fma),("fmuladd",:muladd)]
            @eval @inline function Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T})
                $(llvmcall_expr(op, W, T, (W for _ in 1:3), (T for _ in 1:3), f === :fma ? nothing : "fast"))
            end
        end
        # floating vector, integer scalar
        let op = "powi", f = :(^)
            @eval @inline function Base.$f(v1::Vec{$W,$T}, v2::Int32)
                $(llvmcall_expr(op, W, T, (W, 1), (T, Int32), "fast"))
            end
        end
        for (op,f) ∈ [
            ("experimental.vector.reduce.v2.fadd",:sum),
            ("experimental.vector.reduce.v2.fmul",:prod)
        ]
            @eval @inline function Base.$f(v1::$T, v1::Vec{$W,$T})
                $(llvmcall_expr(op, 1, T, (1, W), (T, T), "fast"))
            end
        end
        for (op,f) ∈ [
            ("experimental.vector.reduce.v2.fmax",:maximum),
            ("experimental.vector.reduce.v2.fmin",:minimum)
        ]
            @eval @inline function Base.$f(v1::Vec{$W,$T}, v2::Int32)
                $(llvmcall_expr(op, 1, T, (W,), (T,), "fast"))
            end
        end
        
        W += W
    end
end

@inline sum(v::Vec{W,T}) where {W,T} = sum(-zero(T), v)
@inline prod(v::Vec{W,T}) where {W,T} = sum(one(T), v)

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



function count_zeros_func(W, I, op)
    typ = "i$(8sizeof(I))"
    vtyp = "<$W x $typ>"
    instr = "@llvm.$op.v$(W)$(typ)"
    decl = "declare $vtyp $instr($vtyp, i1)"
    instrs = "%res = call $vtyp $instr($vtyp %0, i1 1)\nret $vtyp %res"
    llvmcall_expr(decl, instrs, _Vec{W,I}, Tuple{_Vec{W,I}}, vtyp, (vtyp,), (:(data(v)),))    
end
@generated vleading_zeros(v::_Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "ctlz")
@generated vtrailing_zeros(v::_Vec{W,I}) where {W, I <: Integer} = count_zeros_func(W, I, "cttz")


for T ∈ [UInt8,UInt16,UInt32,UInt64]
    bytes = sizeof(T)
    W = 2
    while W * bytes ≤ REGISTER_SIZE

        for (op,f) ∈ [("ctpop", :count_ones)]
            @eval @inline Base.$f(v1::Vec{$W,$T}) = $(llvmcall_expr(op, W, T, (W,), (T,)))
        end
        
        for (op,f) ∈ [("fshl",:funnel_shift_left),("fshr",:funnel_shift_right)
                      ]
            @eval @inline function $f(v1::Vec{$W,$T}, v2::Vec{$W,$T})
                $(llvmcall_expr(op, W, T, (W for _ in 1:3), (T for _ in 1:3)))
            end
        end

        W += W
    end
end


if FMA
    for T ∈ [Float32,Float64]
        W = 16 ÷ sizeof(T)
        local suffix = T == Float32 ? "ps" : "pd"
        typ = llvmtype(T)
        while W <= VectorizationBase.REGISTER_SIZE ÷ sizeof(T)
            vfmadd_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            vfnmadd_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            vfmsub_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            vfnmsub_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1", "=v,0,v,v"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0)
                ret <$W x $(typ)> %res"""
            @eval begin
                @inline function vfmadd231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    Vec(llvmcall($vfmadd_str, _Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
                end
                @inline function vfnmadd231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    Vec(llvmcall($vfnmadd_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
                end
                @inline function vfmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    Vec(llvmcall($vfmsub_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
                end
                @inline function vfnmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    Vec(llvmcall($vfnmsub_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T}}, data(a), data(b), data(c)))
                end
            end
            if VectorizationBase.AVX512BW && W ≥ 8
                vfmaddmask_str = """%res = call <$W x $(typ)> asm "vfmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                    ret <$W x $(typ)> %res"""
                vfnmaddmask_str = """%res = call <$W x $(typ)> asm "vfnmadd231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                    ret <$W x $(typ)> %res"""
                vfmsubmask_str = """%res = call <$W x $(typ)> asm "vfmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                    ret <$W x $(typ)> %res"""
                vfnmsubmask_str = """%res = call <$W x $(typ)> asm "vfnmsub231$(suffix) \$3, \$2, \$1 {\$4}", "=v,0,v,v,^Yk"(<$W x $(typ)> %2, <$W x $(typ)> %1, <$W x $(typ)> %0, i$W %3)
                    ret <$W x $(typ)> %res"""
                U = VectorizationBase.mask_type(W)
                @eval begin
                    @inline function vifelse(::typeof(vfmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                        Vec(llvmcall($vfmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                    end
                    @inline function vifelse(::typeof(vfnmadd231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                        Vec(llvmcall($vfnmaddmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                    end
                    @inline function vifelse(::typeof(vfmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                        Vec(llvmcall($vfmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                    end
                    @inline function vifelse(::typeof(vfnmsub231), m::Mask{$W,$U}, a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                        Vec(llvmcall($vfnmsubmask_str, Vec{$W,$T}, Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$U}, data(a), data(b), data(c), data(m)))
                    end
                end
            end
            W += W
        end
    end
end
@inline vifelse(f::F, m::Mask, a::Vararg{<:Any,K}) where {F,K} = vifelse(m, f(a...), a[K])
