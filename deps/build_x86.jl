using CpuId, LLVM

features = split(unsafe_string(LLVM.API.LLVMGetHostCPUFeatures()), ',')
offsetnottwo(::Nothing) = true
offsetnottwo(m::RegexMatch) = m.offset != 2
features = filter(ext -> offsetnottwo(match(r"\d", ext)), features)
avx512f = any(isequal("+avx512f"), features)
avx2 = any(isequal("+avx2"), features)
avx = any(isequal("+avx"), features)

extension_name(ext) = replace(Base.Unicode.uppercase(ext[2:end]), r"\." => "_")
present = map(ext -> first(ext) == '+', features)

setfeatures = join(map(ext -> "const " * extension_name(ext) * '=' * string(first(ext) == '+'), features), "\n")

register_size = avx512f ? 64 : (avx ? 32 : 16)
register_count = avx512f ? 32 : 16
cache_size = CpuId.cachesize()
num_cores = CpuId.cpucores()

# Should I just add all the flags in features?
cpu_info_string = setfeatures * """

const FMA3 = FMA
const REGISTER_SIZE = $register_size
const REGISTER_COUNT = $register_count
const FP256 = $(cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = $(cachelinesize())
const CACHE_SIZE = $cache_size
const NUM_CORES = $num_cores
const SIMD_NATIVE_INTEGERS = $(avx2)

"""


