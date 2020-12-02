

const TOPOLOGY = Hwloc.topology_load();
const CACHE = TOPOLOGY.children[1].children[1];
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
    c = CACHE
    for n ∈ 1:CACHE_LEVELS-N
        c = first(c.children)
    end
    (
        size = c.attr.size,
        depth = c.attr.depth,
        linesize = c.attr.linesize,
        associativity = c.attr.associativity,
        type = c.attr.type_
    )
end


const L₁CACHE = define_cache(1)
const L₂CACHE = define_cache(2)
const L₃CACHE = define_cache(3)
const L₄CACHE = define_cache(4)
"""
L₁, L₂, L₃ cache size
"""
const CACHE_SIZE = (
    L₁CACHE.size,
    L₂CACHE.size,
    L₃CACHE.size,
    L₄CACHE.size
)
# const CACHE_NEST_COUNT = (
#     length(CACHE.children[1].children),
#     length(CACHE.children),
#     length(TOPOLOGY.children[1].children)
# )


