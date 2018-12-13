using CpuId

register_size = simdbytes()
register_count = cpufeature(CpuId.AVX512F) ? 32 : 16
register_capacity = register_size * register_count

if cpuvendor() == :Intel
    cache_size = CpuId.cachesize()
    num_cores = CpuId.cpucores()
elseif cpuvendor() == :AMD
    if Sys.CPU_NAME == "znver1"
        num_cores = Sys.CPU_THREADS > 4 ? Sys.CPU_THREADS ÷ 2 : 4
        cache_size = (32768, 524288, num_cores > 8 ? 524288 << 5 : 524288 << 4 )
    else
        throw("""Architecture not yet supported. Please file an issue!
                You can fix immediately by supplying the correct number of cores
                and the cache sizes (L1 data per core, L2 per core, L3).
                If you do, please file an issue with the information or a PR
                adding it to the build script, so your architecture will be
                supported for all future releases.""")
    end
else
    throw("This vendor not currently supported! Please file an issue.")
end

cpu_info_string = """
const REGISTER_SIZE = $register_size
const REGISTER_COUNT = $register_count
const REGISTER_CAPACITY = $register_capacity
const FP256 = $(cpufeature(CpuId.FP256)) # Is AVX2 fast?
const CACHELINE_SIZE = $(cachelinesize())
const CACHE_SIZE = $cache_size
const NUM_CORES = $num_cores
"""

open(joinpath(@__DIR__, "..", "src", "cpu_info.jl"), "w") do f
    write(f, cpu_info_string)
end
