@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    include(joinpath(@__DIR__, "cpu_info_x86_llvm.jl"))
else
    include(joinpath(@__DIR__, "cpu_info_generic.jl"))
end

const DYNAMIC_REGISTER_SIZE = @static if Sys.ARCH === :x86_64
    if AVX512F
        64
    else
        if AVX
            32
        else
            16
        end
    end
else
    16
end

const DYNAMIC_INTEGER_REGISTER_SIZE = @static if Sys.ARCH === :x86_64
    if AVX2 || SSE2
        16
    else
        8
    end
else
    8
end

const DYNAMIC_FMA_FAST = @static if Sys.ARCH === :x86_64
    FMA || FMA4
else
    false
end

const DYNAMIC_REGISTER_COUNT = @static if Sys.ARCH === :i686
    8
else
    @static if Sys.ARCH === :x86_64
        if AVX512F
            32
        else
            16
        end
    else
        16
    end
end

const DYNAMIC_HAS_OPMASK_REGISTERS = @static if Sys.ARCH === :x86_64
    AVX512F
else
    false
end

const SDYNAMIC_REGISTER_SIZE = StaticInt{DYNAMIC_REGISTER_SIZE}()
const SDYNAMIC_INTEGER_REGISTER_SIZE = StaticInt{DYNAMIC_INTEGER_REGISTER_SIZE}()
# const SDYNAMIC_FMA_FAST = StaticBool{DYNAMIC_FMA_FAST}()
const SDYNAMIC_REGISTER_COUNT = StaticInt{DYNAMIC_REGISTER_COUNT}()
# const SDYNAMIC_HAS_OPMASK_REGISTERS = StaticBool{DYNAMIC_HAS_OPMASK_REGISTERS}()
