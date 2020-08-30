

const TOPOLOGY = Hwloc.topology_load();
const CACHE = TOPOLOGY.children[1].children[1];
const COUNTS = Hwloc.histmap(TOPOLOGY);
# TODO: Makes topological assumptions that aren't right for
# multiple nodes or with >3 or <3 levels of Cache.
const L₁CACHE = (
    size = CACHE.children[1].children[1].attr.size,
    depth = CACHE.children[1].children[1].attr.depth,
    linesize = CACHE.children[1].children[1].attr.linesize,
    associativity = CACHE.children[1].children[1].attr.associativity,
    type = CACHE.children[1].children[1].attr.type_
)
const L₂CACHE = (
    size = CACHE.children[1].attr.size,
    depth = CACHE.children[1].attr.depth,
    linesize = CACHE.children[1].attr.linesize,
    associativity = CACHE.children[1].attr.associativity,
    type = CACHE.children[1].attr.type_
)
const L₃CACHE = (
    size = CACHE.attr.size,
    depth = CACHE.attr.depth,
    linesize = CACHE.attr.linesize,
    associativity = CACHE.attr.associativity,
    type = CACHE.attr.type_
)
"""
L₁, L₂, L₃ cache size
"""
const CACHE_SIZE = (
    L₁CACHE.size,
    L₂CACHE.size,
    L₃CACHE.size
)
# const CACHE_NEST_COUNT = (
#     length(CACHE.children[1].children),
#     length(CACHE.children),
#     length(TOPOLOGY.children[1].children)
# )
const CACHE_COUNT = (
    COUNTS[:L1Cache],
    COUNTS[:L2Cache],
    COUNTS[:L3Cache]
)
const NUM_CORES = COUNTS[:Core]


