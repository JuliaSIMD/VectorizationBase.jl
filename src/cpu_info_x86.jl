
fma_fast() = has_feature(Val(:x86_64_fma)) | has_feature(Val(:x86_64_fma4))
register_size() = ifelse(
    has_feature(Val(:x86_64_avx512f)),
    StaticInt{64}(),
    ifelse(
        has_feature(Val(:x86_64_avx)),
        StaticInt{32}(),
        StaticInt{16}()
    )
)
simd_integer_register_size() = ifelse(
    has_feature(Val(:x86_64_avx2)),
    register_size(),
    ifelse(
        has_feature(Val(:x86_64_sse2)),
        StaticInt{16}(),
        StaticInt{8}()
    )
)
if Sys.ARCH === :i686
    register_count() = StaticInt{8}()
elseif Sys.ARCH === :x86_64
    register_count() = ifelse(has_feature(Val(:x86_64_avx512f)), StaticInt{32}(), StaticInt{16}())
end
has_opmask_registers() = has_feature(Val(:x86_64_avx512f))

reset_extra_features!() = nothing

fast_int64_to_double() = has_feature(Val(:x86_64_avx512dq))


