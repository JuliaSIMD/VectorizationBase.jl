


if VectorizationBase.FMA3
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
                    llvmcall($vfmadd_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T}}, a, b, c)
                end
                @inline function vfnmadd231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    llvmcall($vfnmadd_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T}}, a, b, c)
                end
                @inline function vfmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    llvmcall($vfmsub_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T}}, a, b, c)
                end
                @inline function vfnmsub231(a::Vec{$W,$T}, b::Vec{$W,$T}, c::Vec{$W,$T})
                    llvmcall($vfnmsub_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T}}, a, b, c)
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
                    @inline function vifelse(::typeof(vfmadd231), m::Mask{$W,$U}, a::SVec{$W,$T}, b::SVec{$W,$T}, c::SVec{$W,$T})
                        SVec(llvmcall($vfmaddmask_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T},$U}, extract_data(a), extract_data(b), extract_data(c), extract_data(m)))
                    end
                    @inline function vifelse(::typeof(vfnmadd231), m::Mask{$W,$U}, a::SVec{$W,$T}, b::SVec{$W,$T}, c::SVec{$W,$T})
                        SVec(llvmcall($vfnmaddmask_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T},$U}, extract_data(a), extract_data(b), extract_data(c), extract_data(m)))
                    end
                    @inline function vifelse(::typeof(vfmsub231), m::Mask{$W,$U}, a::SVec{$W,$T}, b::SVec{$W,$T}, c::SVec{$W,$T})
                        SVec(llvmcall($vfmsubmask_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T},$U}, extract_data(a), extract_data(b), extract_data(c), extract_data(m)))
                    end
                    @inline function vifelse(::typeof(vfnmsub231), m::Mask{$W,$U}, a::SVec{$W,$T}, b::SVec{$W,$T}, c::SVec{$W,$T})
                        SVec(llvmcall($vfnmsubmask_str, Vec{$W,$T}, Tuple{Vec{$W,$T},Vec{$W,$T},Vec{$W,$T},$U}, extract_data(a), extract_data(b), extract_data(c), extract_data(m)))
                    end
                end
            end
            W += W
        end
    end
end

