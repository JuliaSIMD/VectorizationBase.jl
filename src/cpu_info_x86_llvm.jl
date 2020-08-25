

let llvmlib = Libdl.dlopen(only(filter(lib->occursin(r"LLVM\b", basename(lib)), Libdl.dllist()))),
    gethostcpufeatures = Libdl.dlsym(llvmlib, :LLVMGetHostCPUFeatures),
    features_cstring = ccall(gethostcpufeatures, Cstring, ()),
    features = filter(ext -> (m = match(r"\d", ext); isnothing(m) ? true : m.offset != 2 ), split(unsafe_string(features_cstring), ','))

    avx512f = any(isequal("+avx512f"), features)
    avx2 = any(isequal("+avx2"), features)
    avx = any(isequal("+avx"), features)

    register_size = avx512f ? 64 : (avx ? 32 : 16)
    register_count = avx512f ? 32 : 16

    @eval const REGISTER_SIZE = $register_size
    @eval const REGISTER_COUNT = $register_count
    @eval const CACHE_SIZE = $cache_size
    @eval const SIMD_NATIVE_INTEGERS = $(avx2)

    for ext ∈ features
        @eval const $(Symbol(replace(Base.Unicode.uppercase(ext[2:end]), r"\." => "_"))) = $(first(ext) == '+')
    end
    Libc.free(features_cstring)
end

const FMA_FAST = FMA | FMA4

