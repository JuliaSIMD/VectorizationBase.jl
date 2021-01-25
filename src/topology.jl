
mutable struct Topology
    topology::Union{Nothing,Hwloc.Object}
end
Topology() = Topology(nothing)

function count_attr(topology::Hwloc.Object, attr)
    count = 0
    for t ∈ topology
        count += t.type_ == attr
    end
    count
end

@generated function sattr_count(::Val{attr}) where {attr}
    assert_init_has_finished()
    topology = TOPOLOGY.topology
    topology === nothing && return nothing
    count = count_attr(topology, attr)
    Expr(:call, Expr(:curly, :StaticInt, count))
end

scache_count(::Union{StaticInt{1},Val{1}}) = sattr_count(Val{:L1Cache}())
scache_count(::Union{StaticInt{2},Val{2}}) = sattr_count(Val{:L2Cache}())
scache_count(::Union{StaticInt{3},Val{3}}) = sattr_count(Val{:L3Cache}())
scache_count(::Union{StaticInt{4},Val{4}}) = sattr_count(Val{:L4Cache}())
cache_count(::Union{StaticInt{N},Val{N}}) where {N} = convert(Int, scache_count(Val(N)))

snum_machines() = sattr_count(Val{:Machine}())
num_machines() = convert(Int, snum_machines())

snum_sockets() = sattr_count(Val{:Package}())
num_sockets() = convert(Int, snum_sockets())

snum_cores() = sattr_count(Val{:Core}())
num_cores() = convert(Int, snum_cores())

snum_threads() = sattr_count(Val{:PU}())
num_threads() = convert(Int, snum_threads())

function snum_cache_levels()
    l4s = scache_count(Val(4))
    l4s === nothing && return nothing
    if l4s === Zero()
        if scache_count(Val(3)) === Zero()
            if scache_count(Val(2)) === Zero()
                if scache_count(Val(1)) === Zero()
                    return StaticInt{0}()
                else
                    return StaticInt{1}()
                end
            else
                return StaticInt{2}()
            end
        else
            return StaticInt{3}()
        end
    else
        return StaticInt{4}()
    end
end
num_cache_levels() = convert(Int, snum_cache_levels())

function define_cache(N)
    topology = TOPOLOGY.topology
    if (topology === nothing) || (N > num_cache_levels())
        return (
            size = nothing,
            depth = nothing,
            linesize = nothing,
            associativity = nothing,
            type = nothing
        )
    end
    cache_name = (:L1Cache, :L2Cache, :L3Cache, :L4Cache)[N]
    c = first(t for t in topology if t.type_ == cache_name && t.attr.depth == N).attr
    (
        size = c.size,
        depth = c.depth,
        linesize = c.linesize,
        associativity = c.associativity,
        type = c.type_
    )
end

@generated function cache_description(::Union{Val{N},StaticInt{N}}) where {N}
    assert_init_has_finished()
    return define_cache(N)
end

@generated function scache_size(::Union{Val{N},StaticInt{N}}) where {N}
    assert_init_has_finished()
    return Expr(:call, Expr(:curly, :StaticInt, something(define_cache(N).size, 0)))
end
                         
cache_size(::Union{Val{N},StaticInt{N}}) where {N} = convert(Int, scache_size(Val(N)))

@generated function scacheline_size(::Union{Val{N},StaticInt{N}}) where {N}
    assert_init_has_finished()
    return Expr(:call, Expr(:curly, :StaticInt, something(define_cache(N).linesize, 64)))
end

cacheline_size(::Union{Val{N},StaticInt{N}}) where {N} = convert(Int, scacheline_size(Val(N)))
scacheline_size() = scacheline_size(Val(1))
cacheline_size() = cacheline_size(Val(1))
