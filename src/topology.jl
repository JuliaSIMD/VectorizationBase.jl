
const COUNTS = Hwloc.histmap(TOPOLOGY);
# TODO: Makes topological assumptions that aren't right for
# multiple nodes or with >3 or <3 levels of Cache.

const CACHE_COUNT = (
    COUNTS[:L1Cache],
    COUNTS[:L2Cache],
    COUNTS[:L3Cache],
    COUNTS[:L4Cache]
)
const NUM_CORES = COUNTS[:Core]

const CACHE_LEVELS = something(findfirst(isequal(0), CACHE_COUNT) - 1, length(CACHE_COUNT) + 1)

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
"""
L₁, L₂, L₃, L₄ cache size
"""
const CACHE_SIZE = (
    L₁CACHE.size,
    L₂CACHE.size,
    L₃CACHE.size,
    L₄CACHE.size
)

