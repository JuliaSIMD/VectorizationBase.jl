using CpuId, LLVM

features = split(unsafe_string(LLVM.API.LLVMGetHostCPUFeatures()), ',')
avx512f = any(isequal("+avx512f"), features)
features2 = map(ext -> Base.Unicode.uppercase(ext[2:end]), features)
present = map(ext -> , features)

setfeatures = join(map(ext -> "const " * Base.Unicode.uppercase(ext[2:end]) * '=' * string(first(ext) == '+'), features), "\n")

register_size = avx512f ? 64 : 32
register_count = avx512f ? 32 : 16
cache_size = CpuId.cachesize()
num_cores = CpuId.cpucores()

avx2 = any(isequal("+avx2"), features)

# Should I just add all the flags in features?
cpu_info_string = setfeatures * """
const REGISTER_SIZE = $register_size
const REGISTER_COUNT = $register_count
const FP256 = $(cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = $(cachelinesize())
const CACHE_SIZE = $cache_size
const NUM_CORES = $num_cores
const SIMD_NATIVE_INTEGERS = $(avx2)

"""


