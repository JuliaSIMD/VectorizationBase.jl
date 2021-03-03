
using Libdl
function feature_string()
    llvmlib_path = VERSION ≥ v"1.6.0-DEV.1429" ? Base.libllvm_path() : only(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist()))
    libllvm = Libdl.dlopen(llvmlib_path)
    gethostcpufeatures = Libdl.dlsym(libllvm, :LLVMGetHostCPUFeatures)
    features_cstring = ccall(gethostcpufeatures, Cstring, ())
    features = filter(ext -> (m = match(r"\d", ext); isnothing(m) ? true : m.offset != 2 ), split(unsafe_string(features_cstring), ','))
    features, features_cstring
end

# const FEATURE_DICT = Dict{String,Bool}()
# has_feature(str) = get(FEATURE_DICT, str, false)

archstr() = Sys.ARCH === :i686 ? "x86_64_" : string(Sys.ARCH) * '_'

# feature_name(ext) = archstr() * replace(ext[2:end], r"\." => "_")
feature_name(ext) = Symbol(archstr() * ext[2:end])
process_feature(ext) = (feature_name(ext), first(ext) == '+')

has_feature(_) = False()
function set_featue(feature::Symbol, has::Bool)
    featqn = QuoteNode(feature)
    if has
        @eval has_feature(::Val{$featqn}) = True()
    else
        @eval has_feature(::Val{$featqn}) = False()
    end
end
function set_features!()
    features, features_cstring = feature_string()
    for ext ∈ features
        feature, has = process_feature(ext)
        set_featue(feature, has)
    end
    Libc.free(features_cstring)
end
set_features!()

function reset_features!()
    features, features_cstring = feature_string()
    for ext ∈ features
        feature, has = process_feature(ext)
        if Bool(has_feature(Val(feature)))::Bool !== has
            @info "Defining $(has ? "presence" : "absense") of feature $feature."
            set_featue(feature, has)
        end
    end
    Libc.free(features_cstring)
end

register_size() = ifelse(
    has_feature(Val(:x86_64_avx512f)),
    StaticInt{64}(),
    ifelse(
        has_feature(Val(:x86_64_avx)),
        StaticInt{32}(),
        StaticInt{16}()
    )
)
simd_integer_register_size() = ifelse(
    has_feature(Val(:x86_64_avx2)),
    register_size(),
    ifelse(
        has_feature(Val(:x86_64_sse2)),
        StaticInt{16}(),
        StaticInt{8}()
    )
)
fma_fast() = has_feature(Val(:x86_64_fma)) | has_feature(Val(:x86_64_fma4))
if Sys.ARCH === :i686
    register_count() = StaticInt{8}()
else
    register_count() = ifelse(has_feature(Val(:x86_64_avx512f)), StaticInt{32}(), StaticInt{16}())
end
has_opmask_registers() = has_feature(Val(:x86_64_avx512f))

register_size(::Type{T}) where {T} = register_size()
register_size(::Type{T}) where {T<:Union{Signed,Unsigned}} = simd_integer_register_size()

fast_int64_to_double() = has_feature(Val(:x86_64_avx512dq))

function define_cpu_name()
    cpu = QuoteNode(Symbol(Sys.CPU_NAME::String))
    @eval cpu_name() = Val{$cpu}()
end
define_cpu_name()

