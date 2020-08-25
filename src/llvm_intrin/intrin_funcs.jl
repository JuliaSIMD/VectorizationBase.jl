

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

