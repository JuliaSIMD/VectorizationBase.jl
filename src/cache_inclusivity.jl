
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

@generated function cache_inclusivity()
    assert_init_has_finished()
    return dynamic_cache_inclusivity()
end

# @generated cache_inclusivity(::Union{Val{N},StaticInt{N}}) where {N} = dynamic_cache_inclusivity()
