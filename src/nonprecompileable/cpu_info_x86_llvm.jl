_llvmlib = if VERSION ≥ v"1.6.0-DEV.1429"
    Libdl.dlopen(Base.libllvm_path())
else
    Libdl.dlopen(only(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist())))
end

const _gethostcpufeatures = Libdl.dlsym(_llvmlib, :LLVMGetHostCPUFeatures)
const _features_cstring = ccall(_gethostcpufeatures, Cstring, ())
const _features = deepcopy(filter(ext -> (m = match(r"\d", ext); isnothing(m) ? true : m.offset != 2 ), split(unsafe_string(_features_cstring), ',')))

Libc.free(_features_cstring)

include(joinpath(@__DIR__, "_feature_lines_cpu_info_x86_llvm.jl"))

const REGISTER_SIZE = AVX512F ? 64 : (AVX ? 32 : 16)
const REGISTER_COUNT = Sys.ARCH === :i686 ? 8 : (AVX512F ? 32 : 16)

const SIMD_INTEGER_REGISTER_SIZE = if AVX2
    REGISTER_SIZE
elseif SSE2
    16
else
    8
end

const HAS_OPMASK_REGISTERS = AVX512F

const FMA_FAST = FMA || FMA4
