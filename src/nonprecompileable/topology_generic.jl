const NUM_CORES = Sys.CPU_THREADS >>> 1

const CACHE_COUNT = (
    NUM_CORES,
    NUM_CORES,
    1,
    0
)

const CACHE_LEVELS = 3

"""
L₁, L₂, L₃, L₄ cache size

Warning: this is a generic fallback
Assuming
Split: 32 KiB L₁/core, 0.5 MiB L₂ /core
Shared 1 MiB L₃/core
"""
const CACHE_SIZE = ( 32 * (1 << 10), (1 << 20) >>> 1, (1 << 20) * NUM_CORES )

const L₁CACHE = (size = CACHE_SIZE[1], depth = 1, linesize = 64, associativity = 8, type = :Data)
const L₂CACHE = (size = CACHE_SIZE[2], depth = 2, linesize = 64, associativity = 8, type = :Unified)
const L₃CACHE = (size = CACHE_SIZE[3], depth = 3, linesize = 64, associativity = 8, type = :Unified)
const L₄CACHE = (size = nothing, depth = nothing, linesize = nothing, associativity = nothing, type = nothing)
