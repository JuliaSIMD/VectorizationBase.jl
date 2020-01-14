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
const AVX512F = $(cpufeature(CpuId.AVX512F))
"""

open(joinpath(@__DIR__, "..", "src", "cpu_info.jl"), "w") do f
    write(f, cpu_info_string)
end
