const COUNTS = Hwloc.histmap(TOPOLOGY);
# TODO: Makes topological assumptions that aren't right for
# multiple nodes or with >3 or <3 levels of Cache.

const CACHE_COUNT_L1 = if has_override("CACHE_COUNT_L1")
    @info("Using the overrided value for CACHE_COUNT_L1")
    get_override(Int, "CACHE_COUNT_L1")
else
    COUNTS[:L1Cache]
end
const CACHE_COUNT_L2 = if has_override("CACHE_COUNT_L2")
    @info("Using the overrided value for CACHE_COUNT_L2")
    get_override(Int, "CACHE_COUNT_L2")
else
    COUNTS[:L2Cache]
end
const CACHE_COUNT_L3 = if has_override("CACHE_COUNT_L3")
    @info("Using the overrided value for CACHE_COUNT_L3")
    get_override(Int, "CACHE_COUNT_L3")
else
    COUNTS[:L3Cache]
end
const CACHE_COUNT_L4 = if has_override("CACHE_COUNT_L4")
    @info("Using the overrided value for CACHE_COUNT_L4")
    get_override(Int, "CACHE_COUNT_L4")
else
    COUNTS[:L4Cache]
end

"""
L₁, L₂, L₃, L₄ cache count
"""
const CACHE_COUNT = (
    CACHE_COUNT_L1,
    CACHE_COUNT_L2,
    CACHE_COUNT_L3,
    CACHE_COUNT_L4,
)

const NUM_CORES = COUNTS[:Core]

const CACHE_LEVELS = something(findfirst(isequal(0), CACHE_COUNT), length(CACHE_COUNT) + 1) - 1

function define_cache(N)
    if N > CACHE_LEVELS
        return (
            size = nothing,
            depth = nothing,
            linesize = nothing,
            associativity = nothing,
            type = nothing
        )
    end
    cache_name = (:L1Cache, :L2Cache, :L3Cache, :L4Cache)[N]
    c = first(t for t in TOPOLOGY if t.type_ == cache_name && t.attr.depth == N).attr
    (
        size = c.size,
        depth = c.depth,
        linesize = c.linesize,
        associativity = c.associativity,
        type = c.type_
    )
end


const L₁CACHE = define_cache(1)
const L₂CACHE = define_cache(2)
const L₃CACHE = define_cache(3)
const L₄CACHE = define_cache(4)

const CACHE_SIZE_L1 = if has_override("CACHE_SIZE_L1")
    @info("Using the overrided value for CACHE_SIZE_L1")
    get_override(Int, "CACHE_SIZE_L1")
else
    L₁CACHE.size
end
const CACHE_SIZE_L2 = if has_override("CACHE_SIZE_L2")
    @info("Using the overrided value for CACHE_SIZE_L2")
    get_override(Int, "CACHE_SIZE_L2")
else
    L₂CACHE.size
end
const CACHE_SIZE_L3 = if has_override("CACHE_SIZE_L3")
    @info("Using the overrided value for CACHE_SIZE_L3")
    get_override(Int, "CACHE_SIZE_L3")
else
    L₃CACHE.size
end
const CACHE_SIZE_L4 = if has_override("CACHE_SIZE_L4")
    @info("Using the overrided value for CACHE_SIZE_L4")
    get_override(Int, "CACHE_SIZE_L4")
else
    L₄CACHE.size
end

"""
L₁, L₂, L₃, L₄ cache size
"""
const CACHE_SIZE = (
    CACHE_SIZE_L1,
    CACHE_SIZE_L2,
    CACHE_SIZE_L3,
    CACHE_SIZE_L4,
)
