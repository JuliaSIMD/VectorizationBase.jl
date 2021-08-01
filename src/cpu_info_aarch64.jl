
_has_aarch64_sve() = (Base.libllvm_version ≥ v"11") && (Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_AArch64_sve))

if Int === Int64
    @noinline vscale() = ccall("llvm.vscale.i64", llvmcall, Int64, ())
else
    @noinline vscale() = ccall("llvm.vscale.i32", llvmcall, Int32, ())
end

# TODO: find actually support SVE
# _dynamic_register_size() = _has_aarch64_sve() ? 16vscale() : 16
_dynamic_register_size() = 16

function _set_sve_vector_width!(bytes = _dynamic_register_size())
    @eval begin
        register_size() = StaticInt{$bytes}()
        simd_integer_register_size() = StaticInt{$bytes}()
    end
    nothing
end


if _has_aarch64_sve()# && !(Bool(has_feature(Val(:aarch64_sve))))
    has_feature(::Val{:aarch64_sve_cpuid}) = True()
    _set_sve_vector_width!()
else
    # has_feature(::Val{:aarch64_svejl}) = False()
    register_size() = StaticInt{16}()
    simd_integer_register_size() = StaticInt{16}()
end

function reset_extra_features!()
    drs = _dynamic_register_size()
    register_size() ≠ drs && _set_sve_vector_width!(drs)
    hassve = _has_aarch64_sve()
    if hassve ≠ has_feature(Val(:aarch64_sve_cpuid))
        @eval has_feature(::Val{:aarch64_sve_cpuid}) = $(Expr(:call, hassve ? :True : :False))
    end
end

fma_fast() = True()
register_count() = StaticInt{32}()
has_opmask_registers() = has_feature(Val(:aarch64_sve_cpuid))

fast_int64_to_double() = True()

fast_half() = False()

