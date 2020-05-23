using CpuId

register_size = simdbytes()
register_count = cpufeature(CpuId.AVX512F) ? 32 : 16
register_capacity = register_size * register_count
cache_size = CpuId.cachesize()
num_cores = CpuId.cpucores()

cpu_info_string = """
const REGISTER_SIZE = $register_size
const REGISTER_COUNT = $register_count
const REGISTER_CAPACITY = $register_capacity
const FP256 = $(cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = $(cachelinesize())
const CACHE_SIZE = $cache_size
const NUM_CORES = $num_cores
const FMA3 = $(cpufeature(CpuId.FMA3))
const AVX2 = $(cpufeature(CpuId.AVX2))
const AVX512F = $(cpufeature(CpuId.AVX512F))
const AVX512ER = $(cpufeature(CpuId.AVX512ER))
const AVX512PF = $(cpufeature(CpuId.AVX512PF))
const AVX512VL = $(cpufeature(CpuId.AVX512VL))
const AVX512BW = $(cpufeature(CpuId.AVX512BW))
const AVX512DQ = $(cpufeature(CpuId.AVX512DQ))
const AVX512CD = $(cpufeature(CpuId.AVX512CD))
const SIMD_NATIVE_INTEGERS = $(cpufeature(CpuId.AVX2))

"""


