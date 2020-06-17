using CpuId, LLVM

features = split(unsafe_string(LLVM.API.LLVMGetHostCPUFeatures()), ',')
avx512f = any(isequal("+avx512f"), features)

register_size = avx512f ? 64 : 32
register_count = avx512f ? 32 : 16
cache_size = CpuId.cachesize()
num_cores = CpuId.cpucores()

avx2 = any(isequal("+avx2"), features)

# Should I just add all the flags in features?
cpu_info_string = """
const REGISTER_SIZE = $register_size
const REGISTER_COUNT = $register_count
const FP256 = $(cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = $(cachelinesize())
const CACHE_SIZE = $cache_size
const NUM_CORES = $num_cores
const FMA3 = $(any(isequal("+fma"), features))
const AVX2 = $(avx2)
const AVX512F = $(avx512f)
const AVX512ER = $(any(isequal("+avx512er"), features))
const AVX512PF = $(any(isequal("+avx512pf"), features))
const AVX512VL = $(any(isequal("+avx512vl"), features))
const AVX512BW = $(any(isequal("+avx512bw"), features))
const AVX512DQ = $(any(isequal("+avx512dq"), features))
const AVX512CD = $(any(isequal("+avx512cd"), features))
const SIMD_NATIVE_INTEGERS = $(avx2)

"""


