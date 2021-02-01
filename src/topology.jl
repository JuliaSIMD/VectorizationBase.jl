mutable struct Topology
    topology::Union{Nothing,Hwloc.Object}
end
Topology() = Topology(nothing)
function safe_topology_load!()
    try
        TOPOLOGY.topology = Hwloc.topology_load();
    catch e
        @warn e
        @warn """
            Using Hwloc failed. Please file an issue with the above warning at: https://github.com/JuliaParallel/Hwloc.jl
            Proceeding with generic topology assumptions. This may result in reduced performance.
        """
    end
end

const TOPOLOGY = Topology()
safe_topology_load!()

function count_attr(topology::Hwloc.Object, attr::Symbol)
    count = 0
    for t ∈ topology
        count += t.type_ == attr
    end
    count
end

function count_attr(attr::Symbol)
    topology = TOPOLOGY.topology
    topology === nothing && return nothing
    count_attr(topology, attr)
end

function define_attr_count(fname::Symbol, v)
    if v === nothing
        @eval $fname() = nothing
        return
    elseif v isa Integer
        @eval $fname() = StaticInt{$(convert(Int,v))}()
    elseif v isa Bool
        if v
            @eval $fname() = True()
        else
            @eval $fname() = False()
        end
    else
        @eval $fname() = $v
    end
    nothing
end
# define_attr(fname::Symbol, attr::Symbol) = define_attr_count(fname, )

for (f, attr) ∈ [
    (:num_l1cache, :L1Cache),
    (:num_l2cache, :L2Cache),
    (:num_l3cache, :L3Cache),
    (:num_l4cache, :L4Cache),
    (:num_machines, :Machine),
    (:num_sockets, :Package),
    (:num_cores, :Core),
    (:num_threads, :PU)
]
    define_attr_count(f, count_attr(attr))
end

function redefine_attr_count()
    iter = [
        (num_l1cache(), :num_l1cache, :L1Cache),
        (num_l2cache(), :num_l2cache, :L2Cache),
        (num_l3cache(), :num_l3cache, :L3Cache),
        (num_l4cache(), :num_l4cache, :L4Cache),
        (num_machines(), :num_machines, :Machine),
        (num_sockets(), :num_sockets, :Package),
        (num_cores(), :num_cores, :Core),
        (num_threads(), :num_threads, :PU)
    ]
    for (v, f, attr) ∈ iter
        ref = count_attr(attr)
        ref == v || define_attr_count(f, ref)
    end
    nothing
end

num_cache(::Union{Val{1},StaticInt{1}}) = num_l1cache()
num_cache(::Union{Val{2},StaticInt{2}}) = num_l2cache()
num_cache(::Union{Val{3},StaticInt{3}}) = num_l3cache()
num_cache(::Union{Val{4},StaticInt{4}}) = num_l4cache()

function num_cache_levels()
    numl4 = num_l4cache()
    numl4 === nothing && return nothing
    ifelse(
        eq(numl4, Zero()),
        ifelse(
            eq(num_l3cache(), Zero()),
            ifelse(
                eq(num_l2cache(), Zero()),
                ifelse(
                    eq(num_l1cache(), Zero()),
                    Zero(),
                    One()
                ),
                StaticInt{2}()
            ),
            StaticInt{3}()
        ),
        StaticInt{4}()
    )
end


function dynamic_cache_inclusivity()::NTuple{4,Bool}
    if !((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686))
        return (false,false,false,false)
    end
    function get_cache_edx(subleaf)
        # source: https://github.com/m-j-w/CpuId.jl/blob/401b638cb5a020557bce7daaf130963fb9c915f0/src/CpuInstructions.jl#L38
        # credit Markus J. Weber, copyright: https://github.com/m-j-w/CpuId.jl/blob/master/LICENSE.md
        Base.llvmcall(
            """
            ; leaf = %0, subleaf = %1, %2 is some label
            ; call 'cpuid' with arguments loaded into registers EAX = leaf, ECX = subleaf
            %2 = tail call { i32, i32, i32, i32 } asm sideeffect "cpuid",
                "={ax},={bx},={cx},={dx},{ax},{cx},~{dirflag},~{fpsr},~{flags}"
                (i32 4, i32 %0) #2
            ; retrieve the result values and return eax and edx contents
            %3 = extractvalue { i32, i32, i32, i32 } %2, 0
            %4 = extractvalue { i32, i32, i32, i32 } %2, 3
            %5  = insertvalue [2 x i32] undef, i32 %3, 0
            %6  = insertvalue [2 x i32]   %5 , i32 %4, 1
            ; return the value
            ret [2 x i32] %6
            """
            # llvmcall requires actual types, rather than the usual (...) tuple
            , Tuple{UInt32,UInt32}, Tuple{UInt32}, subleaf % UInt32
        )
    end
    # eax0, edx1 = get_cache_edx(0x00000000)
    t = (false,false,false,false)
    i = zero(UInt32)
    j = 0
    while (j < 4)
        eax, edx = get_cache_edx(i)
        i += one(UInt32)
        iszero(eax & 0x1f) && break
        iszero(eax & 0x01) && continue
        ci = ((edx & 0x00000002) != 0x00000000) & (eax & 0x1f != 0x00000000)
        t = Base.setindex(t, ci, (j += 1))
    end
    t
end


nothing_cache_summary() = (size = nothing, linesize = nothing, associativity = nothing, type = nothing, inclusive = nothing)
function dynamic_cache_summary(N)
    topology = TOPOLOGY.topology
    cache_name = (:L1Cache, :L2Cache, :L3Cache, :L4Cache)[N]
    if (topology === nothing) || (iszero(count_attr(cache_name)))
        return nothing_cache_summary()
    end
    c = first(t for t in topology if t.type_ == cache_name && t.attr.depth == N).attr
    (
        size = c.size,
        linesize = c.linesize,
        associativity = c.associativity,
        type = c.type_,
        inclusive = dynamic_cache_inclusivity()[N]
    )
end
cache_size(_) = nothing
cache_linesize(_) = StaticInt{64}() # assume...
cache_associativity(_) = nothing
cache_type(_) = nothing
cache_inclusive(_) = nothing
function define_cache(N, c = dynamic_cache_summary(N))
    c === nothing_cache_summary() && return
    @eval begin
        cache_size(::Union{Val{$N},StaticInt{$N}}) = StaticInt{$(c.size)}()
        cache_linesize(::Union{Val{$N},StaticInt{$N}}) = StaticInt{$(c.linesize)}()
        cache_associativity(::Union{Val{$N},StaticInt{$N}}) = StaticInt{$(c.associativity)}()
        cache_type(::Union{Val{$N},StaticInt{$N}}) = Val{$(c.type === nothing ? nothing : QuoteNode(c.type))}()
        cache_inclusive(::Union{Val{$N},StaticInt{$N}}) = $(c.inclusive ? :True : :False)()
    end
    nothing
end
function redefine_cache(N)
    c = (
        size = cache_size(StaticInt(N)),
        linesize = cache_linesize(StaticInt(N)),
        associativity = cache_associativity(StaticInt(N)),
        type = cache_type(StaticInt(N)),
        inclusive = cache_inclusive(StaticInt(N))
    )
    correct = dynamic_cache_summary(N)
    c === correct || define_cache(N, correct)
    nothing
end
foreach(define_cache, 1:4)

cache_linesize() = cache_linesize(Val(1))

