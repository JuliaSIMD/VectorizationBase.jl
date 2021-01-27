
using Libdl
function feature_string()
    llvmlib_path = VERSION ≥ v"1.6.0-DEV.1429" ? Base.libllvm_path() : only(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist()))
    libllvm = Libdl.dlopen(llvmlib_path)
    gethostcpufeatures = Libdl.dlsym(libllvm, :LLVMGetHostCPUFeatures)
    features_cstring = ccall(gethostcpufeatures, Cstring, ())
    features = filter(ext -> (m = match(r"\d", ext); isnothing(m) ? true : m.offset != 2 ), split(unsafe_string(features_cstring), ','))
    features, features_cstring
end

const FEATURE_DICT = Dict{String,Bool}()
has_feature(str) = get(FEATURE_DICT, str, false)

archstr() = Sys.ARCH === :i686 ? "x86_64_" : string(Sys.ARCH) * '_'
# feature_name(ext) = archstr() * replace(ext[2:end], r"\." => "_")
feature_name(ext) = archstr() * ext[2:end]
process_feature(ext) = (feature_name(ext), first(ext) == '+')

let (features, features_cstring) = feature_string()
    sizehint!(FEATURE_DICT, length(features))
    for ext ∈ features
        FEATURE_DICT[feature_name(ext)] = false
    end
    Libc.free(features_cstring)
end

function set_features!()
    features, features_cstring = feature_string()
    for ext ∈ features
        extname, hasfeature = process_feature(ext)
        FEATURE_DICT[extname] = hasfeature
    end
    Libc.free(features_cstring)
end

dynamic_register_size() = has_feature("x86_64_avx512f") ? 64 : has_feature("x86_64_avx") ? 32 : 16
dynamic_integer_register_size() = has_feature("x86_64_avx2") ? dynamic_register_size() : (has_feature("x86_64_sse2") ? 16 : 8)
dynamic_fma_fast() = has_feature("x86_64_fma") | has_feature("x86_64_fma4")
dynamic_register_count() = Sys.ARCH === :i686 ? 8 : (has_feature("x86_64_avx512f") ? 32 : 16)
dynamic_has_opmask_registers() = has_feature("x86_64_avx512f")

# This is terrible, I know. Please let me know if you have a better solution
@generated function fma_fast()
    assert_init_has_finished()
    return dynamic_fma_fast()
end

@generated function has_opmask_registers()
    assert_init_has_finished()
    return dynamic_has_opmask_registers()
end
@generated function sregister_size() 
    assert_init_has_finished()
    return Expr(:call, Expr(:curly, :StaticInt, dynamic_register_size()))
end

@generated function sregister_count()
    assert_init_has_finished()
    return Expr(:call, Expr(:curly, :StaticInt, dynamic_register_count()))
end

@generated function ssimd_integer_register_size()
    assert_init_has_finished()
    return Expr(:call, Expr(:curly, :StaticInt, dynamic_integer_register_size()))
end

register_size() = convert(Int, sregister_size())
register_count() = convert(Int, sregister_count())
simd_integer_register_size() = convert(Int, ssimd_integer_register_size())

register_size(::Type{T}) where {T} = register_size()
register_size(::Type{T}) where {T<:Union{Signed,Unsigned}} = simd_integer_register_size()
sregister_size(::Type{T}) where {T} = sregister_size()
sregister_size(::Type{T}) where {T<:Union{Signed,Unsigned}} = ssimd_integer_register_size()


