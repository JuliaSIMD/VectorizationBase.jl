
import CpuId, LLVM

let features = filter(ext -> (m = match(r"\d", ext); isnothing(m) ? true : m.offset != 2 ) , split(unsafe_string(LLVM.API.LLVMGetHostCPUFeatures()), ','))
    offsetnottwo(::Nothing) = true
    offsetnottwo(m::RegexMatch) = m.offset != 2

    avx512f = any(isequal("+avx512f"), features)
    avx2 = any(isequal("+avx2"), features)
    avx = any(isequal("+avx"), features)

    register_size = avx512f ? 64 : (avx ? 32 : 16)
    register_count = avx512f ? 32 : 16
    cache_size = CpuId.cachesize()
    num_cores = CpuId.cpucores()

    @eval const REGISTER_SIZE = $register_size
    @eval const REGISTER_COUNT = $register_count
    @eval const FP256 = $(CpuId.cpufeature(CpuId.FP256)) # Is AVX2 fast?
    @eval const CACHELINE_SIZE = $(CpuId.cachelinesize())
    @eval const CACHE_SIZE = $cache_size
    @eval const NUM_CORES = $num_cores
    @eval const SIMD_NATIVE_INTEGERS = $(avx2)

    for ext ∈ features
        @eval const $(Symbol(replace(Base.Unicode.uppercase(ext[2:end]), r"\." => "_"))) = $(first(ext) == '+')
    end
end

const FMA3 = FMA

