
@warn "Running the generic build script for unrecognized architectures. If you would like contribute to improving performance on your CPU architecture, please see https://github.com/chriselrod/VectorizationBase.jl/issues/9."

cpu_info_string = """
const REGISTER_SIZE = 16
const REGISTER_COUNT = 16
const REGISTER_CAPACITY = 256
const FP256 = true
const CACHELINE_SIZE = 64
const CACHE_SIZE = $((1<<15,1<<20,0))
const NUM_CORES = $(Sys.CPU_THREADS)
const FMA3 = false
const AVX2 = false 
const AVX512F = false
const AVX512ER = false
const AVX512PF = false
const AVX512VL = false
const AVX512BW = false
const AVX512DQ = false
const AVX512CD = false
const SIMD_NATIVE_INTEGERS = true

"""

