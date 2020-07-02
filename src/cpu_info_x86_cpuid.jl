
import CpuId

const REGISTER_SIZE = CpuId.simdbytes()
const REGISTER_COUNT = CpuId.cpufeature(CpuId.AVX512F) ? 32 : 16
const FP256 = (CpuId.cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = (CpuId.cachelinesize())
const CACHE_SIZE = CpuId.cachesize()
const NUM_CORES = CpuId.cpucores()
const FMA3 = (CpuId.cpufeature(CpuId.FMA3))
const AVX2 = (CpuId.cpufeature(CpuId.AVX2))
const AVX512F = (CpuId.cpufeature(CpuId.AVX512F))
const AVX512ER = (CpuId.cpufeature(CpuId.AVX512ER))
const AVX512PF = (CpuId.cpufeature(CpuId.AVX512PF))
const AVX512VL = (CpuId.cpufeature(CpuId.AVX512VL))
const AVX512BW = (CpuId.cpufeature(CpuId.AVX512BW))
const AVX512DQ = (CpuId.cpufeature(CpuId.AVX512DQ))
const AVX512CD = (CpuId.cpufeature(CpuId.AVX512CD))
const SIMD_NATIVE_INTEGERS = (CpuId.cpufeature(CpuId.AVX2))


