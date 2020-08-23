
struct LazyMul{N,T} data::T end
@inline data(lm::LazyMul) = data(lm.data)

