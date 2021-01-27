import InteractiveUtils
InteractiveUtils.versioninfo(stdout; verbose=true)

include("testsetup.jl")

@time @testset "VectorizationBase.jl" begin
    @testset "_Generate" begin
        VectorizationBase._Generate._print_feature_lines(
            devnull,
            VectorizationBase._features,
        )
    end

    W = @inferred(VectorizationBase.pick_vector_width(Float64))
    @test @inferred(VectorizationBase.pick_integer(Val(W))) == (VectorizationBase.register_size() == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE ? Int64 : Int32)


    @test first(A) === A[1]
    @test W64S == W64
    @time @testset "Struct-Wrapped Vec" begin
        @test data(zero(Vec{4,Float64})) === (VE(0.0),VE(0.0),VE(0.0),VE(0.0)) === data(Vec{4,Float64}(0.0))
        @test data(one(Vec{4,Float64})) === (VE(1.0),VE(1.0),VE(1.0),VE(1.0)) === data(Vec{4,Float64}(1.0)) === data(data(Vec{4,Float64}(1.0)))
        v = Vec((VE(1.0),VE(2.0),VE(3.0),VE(4.0)))
        @test v === Vec{4,Float64}(1, 2, 3, 4) === conj(v) === v' === Vec{4,Float64}(v)
        @test length(v) == 4 == first(size(v))
        @test eltype(v) == Float64
        for i in 1:4
            @test i == VectorizationBase.extractelement(v, i-1)
            # @test i === Vec{4,Int}(v)[i] # should use fptosi (ie, vconvert defined in SIMDPirates).
        end
        @test zero(v) === zero(typeof(v))
        @test one(v) === one(typeof(v))
        # @test Vec{W32,Float32}(one(Vec{W32,Float64})) === Vec(one(Vec{W32,Float32})) === one(Vec{W32,Float32}) # conversions should be tested in SIMDPirates
        @test Vec{1,Int}(1) === 1

        vu = Vec(collect(1.0:16.0)...) + 2
        @test vu(1,1) === vu.data[1](1)
        @test vu(2,1) === vu.data[1](2)
        @test vu(1,2) === vu.data[2](1)
        @test vu(2,2) === vu.data[2](2)
        if W64 == 8
            @test vu.data[1] === Vec(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
            @test vu.data[2] === Vec(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
        elseif W64 == 4
            @test vu.data[1] === Vec(3.0, 4.0, 5.0, 6.0)
            @test vu.data[2] === Vec(7.0, 8.0, 9.0, 10.0)
            @test vu.data[3] === Vec(11.0, 12.0, 13.0, 14.0)
            @test vu.data[4] === Vec(15.0, 16.0, 17.0, 18.0)
            @test Vec(1.2, 3.4, 3.4) === Vec(1.2, 3.4, 3.4, 0.0)
        elseif W64 == 2
            @test vu.data[1] === Vec(3.0, 4.0)
            @test vu.data[2] === Vec(5.0, 6.0)
            @test vu.data[3] === Vec(7.0, 8.0)
            @test vu.data[4] === Vec(9.0, 10.0)
            @test vu.data[5] === Vec(11.0, 12.0)
            @test vu.data[6] === Vec(13.0, 14.0)
            @test vu.data[7] === Vec(15.0, 16.0)
            @test vu.data[8] === Vec(17.0, 18.0)
        end

    end

    @time @testset "alignment.jl" begin

        @test all(i -> VectorizationBase.align(i) == VectorizationBase.register_size(), 1:VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i) == 2VectorizationBase.register_size(), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i) == 10VectorizationBase.register_size(), (1:VectorizationBase.register_size()) .+ 9VectorizationBase.register_size())

        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},   VectorizationBase.register_size()), 1:VectorizationBase.DYNAMIC_REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},  2VectorizationBase.register_size()), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid}, 20VectorizationBase.register_size()), (1:VectorizationBase.register_size()) .+ 19VectorizationBase.register_size())

        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), 1:VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size())

        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), 1:VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size())

        @test reinterpret(Int, VectorizationBase.align(pointer(A))) % VectorizationBase.register_size() === 0

        @test all(i -> VectorizationBase.aligntrunc(i) == 0, 0:VectorizationBase.register_size()-1)
        @test all(i -> VectorizationBase.aligntrunc(i) == VectorizationBase.register_size(), VectorizationBase.register_size():2VectorizationBase.register_size()-1)
        @test all(i -> VectorizationBase.aligntrunc(i) == 9VectorizationBase.register_size(), (0:VectorizationBase.register_size()-1) .+ 9VectorizationBase.register_size())

        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), 1:VectorizationBase.register_size())
        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size())

        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), 1:VectorizationBase.register_size())
        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), 1+VectorizationBase.register_size():2VectorizationBase.register_size())
        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size())

        a = Vector{Float64}(undef, 0)
        ptr = pointer(a)
        @test UInt(VectorizationBase.align(ptr, 1 << 12)) % (1 << 12) == 0
    end

    @time @testset "masks.jl" begin
        # @test Mask{8,UInt8}(0x0f) === @inferred Mask(0x0f)
        # @test Mask{16,UInt16}(0x0f0f) === @inferred Mask(0x0f0f)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 0)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 8)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 16)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), VectorizationBase.StaticInt(0))
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 0)
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 16)
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 32)
        @test all(w -> VectorizationBase.mask_type(w) === UInt8, 1:8)
        @test all(w -> VectorizationBase.mask_type(w) === UInt16, 9:16)
        @test all(w -> VectorizationBase.mask_type(w) === UInt32, 17:32)
        @test all(w -> VectorizationBase.mask_type(w) === UInt64, 33:64)
        @test all(w -> VectorizationBase.mask_type(w) === UInt128, 65:128)
        if VectorizationBase.register_size() == 64 # avx512
            # @test VectorizationBase.mask_type(Float16) === UInt32
            @test VectorizationBase.mask_type(Float32) === UInt16
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xffffffff # 32
            @test data(VectorizationBase.max_mask(Float32)) === 0xffff     # 16
            @test data(VectorizationBase.max_mask(Float64)) === 0xff       # 8
        elseif VectorizationBase.register_size() == 32 # avx or avx2
            # @test VectorizationBase.mask_type(Float16) === UInt16
            @test VectorizationBase.mask_type(Float32) === UInt8
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xffff     # 16
            @test data(VectorizationBase.max_mask(Float32)) === 0xff       # 8
            @test data(VectorizationBase.max_mask(Float64)) === 0x0f       # 4
        elseif VectorizationBase.register_size() == 16 # sse
            # @test VectorizationBase.mask_type(Float16) === UInt8
            @test VectorizationBase.mask_type(Float32) === UInt8
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xff       # 8
            @test data(VectorizationBase.max_mask(Float32)) === 0x0f       # 4
            @test data(VectorizationBase.max_mask(Float64)) === 0x03       # 2
        end
        @test all(w -> bitstring(VectorizationBase.mask(Val( 8), w)) == reduce(*, ( 8 - i < w ? "1" : "0" for i in 1:8 )), 1:8 )
        @test all(w -> bitstring(VectorizationBase.mask(Val(16), w)) == reduce(*, (16 - i < w ? "1" : "0" for i in 1:16)), 1:16)
        @test all(w -> VectorizationBase.mask(Float64, w) === VectorizationBase.mask(@inferred(VectorizationBase.pick_vector_width_val(Float64)), w), 1:W64)

        @test VectorizationBase.vbroadcast(Val(8), true) === Vec(true, true, true, true, true, true, true, true)

        @test !VectorizationBase.vall(Mask{8}(0xfc))
        @test !VectorizationBase.vall(Mask{4}(0xfc))
        @test VectorizationBase.vall(Mask{8}(0xff))
        @test VectorizationBase.vall(Mask{4}(0xcf))

        @test VectorizationBase.vany(Mask{8}(0xfc))
        @test VectorizationBase.vany(Mask{4}(0xfc))
        @test !VectorizationBase.vany(Mask{8}(0x00))
        @test !VectorizationBase.vany(Mask{4}(0xf0))

        @test VectorizationBase.vall(Mask{8}(0xfc) + Mask{8}(0xcf) == Vec(0x01,0x01,0x02,0x02,0x01,0x01,0x02,0x02))
        @test VectorizationBase.vall(Mask{4}(0xfc) + Mask{4}(0xcf) == Vec(0x01,0x01,0x02,0x02))

        @test VectorizationBase.vall(Mask{8}(0xec) != Mask{8}(0x13))
        @test VectorizationBase.vall((!Mask{8}(0xac) & Mask{8}(0xac)) == Mask{8}(0x00))
        @test !VectorizationBase.vany((!Mask{8}(0xac) & Mask{8}(0xac)))
        @test VectorizationBase.vall((!Mask{8}(0xac) | Mask{8}(0xac)) == Mask{8}(0xff))
        @test VectorizationBase.vall((!Mask{8}(0xac) | Mask{8}(0xac)))

        @test VectorizationBase.vall(VectorizationBase.splitint(0xb53a5d6426a9d29d, Int8) == Vec{8,Int8}(-99, -46, -87, 38, 100, 93, 58, -75))
        # other splitint tests for completeness sake
        @test VectorizationBase.splitint(0xb53a5d6426a9d29d, Int64) === 0xb53a5d6426a9d29d
        @test VectorizationBase.splitint(0xff, UInt16) === 0x00ff
        @test !VectorizationBase.vany(
        VectorizationBase.splitint(0x47766b9a9509d175acd77ff497236795, Int8) != Vec{16,Int8}(-107, 103, 35, -105, -12, 127, -41, -84, 117, -47, 9, -107, -102, 107, 118, 71)
        )


        @test (Mask{8}(0xac) | false) === Mask{8}(0xac)
        @test (Mask{8}(0xac) | true) === Mask{8}(0xff)
        @test (false | Mask{8}(0xac)) === Mask{8}(0xac)
        @test (true | Mask{8}(0xac)) === Mask{8}(0xff)
        @test (Mask{8}(0xac) & false) === Mask{8}(0x00)
        @test (Mask{8}(0xac) & true) === Mask{8}(0xac)
        @test (false & Mask{8}(0xac)) === Mask{8}(0x00)
        @test (true & Mask{8}(0xac)) === Mask{8}(0xac)
        @test (Mask{8}(0xac) ⊻ false) === Mask{8}(0xac)
        @test (Mask{8}(0xac) ⊻ true) === Mask{8}(0x53)
        @test (false ⊻ Mask{8}(0xac)) === Mask{8}(0xac)
        @test (true ⊻ Mask{8}(0xac)) === Mask{8}(0x53)

        @test (Mask{4}(0x05) | true) === Mask{4}(0x0f)
        @test (Mask{4}(0x05) | false) === Mask{4}(0x05)
        @test (true | Mask{4}(0x05)) === Mask{4}(0x0f)
        @test (false | Mask{4}(0x05)) === Mask{4}(0x05)
        for T ∈ [UInt8, UInt16, UInt32]
            Ws = T === UInt8 ? [2,4,8] : [8sizeof(T)]
            for W ∈ Ws
                u1 = rand(T)
                u2 = rand(T)
                m = ~(typemax(T) << W)
                @test (Mask{W}(u1) & Mask{W}(u2)) === Mask{W}( (u1 & u2) & m )
                @test (Mask{W}(u1) | Mask{W}(u2)) === Mask{W}( (u1 | u2) & m )
                @test (Mask{W}(u1) ⊻ Mask{W}(u2)) === Mask{W}( (u1 ⊻ u2) & m )
            end
        end
        @test convert(Bool, Mask{8}(0xec)) === Vec(false,false,true,true,false,true,true,true) === VectorizationBase.ifelse(convert(Bool, Mask{8}(0xec)), vbroadcast(Val(8),true), vbroadcast(Val(8),false))

        @test (MM{8}(2) ∈ 3:8) === Mask{8}(0x7e)

        fbitvector1 = falses(20);
        fbitvector2 = falses(20);
        mu = VectorizationBase.VecUnroll((Mask{4}(0x0f),Mask{4}(0x0f)))
        GC.@preserve fbitvector begin
            vstore!(stridedpointer(fbitvector1), mu, (VectorizationBase.MM(StaticInt{8}(), 1),))
            vstore!(stridedpointer(fbitvector2), mu, (VectorizationBase.MM(StaticInt{8}(), 1),), Mask{8}(0x7e))
        end
        @test all(fbitvector1[1:8])
        @test !any(fbitvector1[9:end])
        @test !fbitvector2[1]
        @test all(fbitvector2[2:7])
        @test !any(fbitvector2[8:end])
    end

    # @testset "number_vectors.jl" begin
    # # eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :(size(A)), 8)) # doesn't work?
    # @test VectorizationBase.length_loads(A, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13*17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13*17, 8)) == divrem(length(A), 8)
    # @test VectorizationBase.size_loads(A,1, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13   )()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13   , 8)) == divrem(size(A,1), 8)
    # @test VectorizationBase.size_loads(A,2, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() ->    17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__,    17, 8)) == divrem(size(A,2), 8)
    # end


    @time @testset "vector_width.jl" begin
        for T ∈ (Float32,Float64)
            @test @inferred(VectorizationBase.pick_vector_width(T)) * sizeof(T) == @inferred(VectorizationBase.pick_vector_width_val(T)) * sizeof(T) == @inferred(VectorizationBase.register_size()) == DYNAMIC_REGISTER_SIZE
            @test @inferred(VectorizationBase.pick_vector_width(T)) * sizeof(T) === @inferred(VectorizationBase.register_size()) === DYNAMIC_REGISTER_SIZE
            @test @inferred(VectorizationBase.pick_vector_width_val(T)) * @inferred(VectorizationBase.static_sizeof(T)) === DYNAMIC_REGISTER_SIZE
        end
        for T ∈ (Int8,Int16,Int32,Int64)
            @test @inferred(VectorizationBase.pick_vector_width(T)) * sizeof(T) == @inferred(VectorizationBase.pick_vector_width_val(T)) * sizeof(T) == VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE
            UT = unsigned(T)
            @test @inferred(VectorizationBase.pick_vector_width(UT)) * sizeof(UT) == @inferred(VectorizationBase.pick_vector_width_val(UT)) * sizeof(UT) == VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE
        end

        @test @inferred(VectorizationBase.pick_vector_width_val(Float64, Int32, Float64, Float32, Float64)) * VectorizationBase.static_sizeof(Float64) === DYNAMIC_REGISTER_SIZE
        @test @inferred(VectorizationBase.pick_vector_width_val(Float64, Int64, Float64, Float32, Float64)) * VectorizationBase.static_sizeof(Float64) === VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE
        @test @inferred(VectorizationBase.pick_vector_width_val(Float64, Int32)) * VectorizationBase.static_sizeof(Float64) === DYNAMIC_REGISTER_SIZE
        @test @inferred(VectorizationBase.pick_vector_width_val(Float64, Int64)) * VectorizationBase.static_sizeof(Float64) === VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE
        @test @inferred(VectorizationBase.pick_vector_width_val(Float32, Float32)) * VectorizationBase.static_sizeof(Float32) === DYNAMIC_REGISTER_SIZE
        @test @inferred(VectorizationBase.pick_vector_width_val(Float32, Int32)) * VectorizationBase.static_sizeof(Float32) === VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE

        @test all(VectorizationBase.ispow2, 0:1)
        @test all(i -> !any(VectorizationBase.ispow2, 1+(1 << (i-1)):(1 << i)-1 ) && VectorizationBase.ispow2(1 << i), 2:9)
        @test all(i ->  VectorizationBase.intlog2(1 << i) == i, 0:(Int == Int64 ? 53 : 30))
        FTypes = (Float32, Float64)
        Wv = ntuple(i -> VectorizationBase.register_size() >> (i+1), Val(2))
        for (T, N) in zip(FTypes, Wv)
            W = VectorizationBase.pick_vector_width(T)
            @test Vec{W,T} == VectorizationBase.pick_vector(Val(W), T) == VectorizationBase.pick_vector(T)
            @test W == VectorizationBase.pick_vector_width(Val(W), T)
            @test StaticInt(W) === VectorizationBase.pick_vector_width_val(Val(W), T) == VectorizationBase.pick_vector_width_val(T)
            while true
                W >>= 1
                W == 0 && break
                W2, Wshift2 = VectorizationBase.pick_vector_width_shift(W, T)
                @test W2 == 1 << Wshift2 == VectorizationBase.pick_vector_width(W, T) == VectorizationBase.pick_vector_width(Val(W),T)  == W
                @test StaticInt(W) === VectorizationBase.pick_vector_width_val(Val(W), T)
                for n in W+1:2W
                    W3, Wshift3 = VectorizationBase.pick_vector_width_shift(n, T)
                    @test W2 << 1 == W3 == 1 << (Wshift2+1) == 1 << Wshift3 == VectorizationBase.pick_vector_width(n, T) == VectorizationBase.pick_vector_width(Val(n),T) == W << 1
                    @test VectorizationBase.pick_vector(Val(W), T) == VectorizationBase.pick_vector(W, T) == Vec{W,T}
                end
            end
        end

        # @test VectorizationBase.nextpow2(0) == 1
        @test all(i -> VectorizationBase.nextpow2(i) == i, 0:2)
        for j in 1:10
            l, u = (1<<j)+1, 1<<(j+1)
            @test all(i -> VectorizationBase.nextpow2(i) == u, l:u)
        end

    end

    @time @testset "Memory" begin
        C = rand(40,20,10) .> 0;
        mtest = vload(stridedpointer(C), ((MM{16})(9), 2, 3));
        v1 = C[9:24,2,3];
        @test tovector(mtest) == v1
        @test [vload(stridedpointer(C), (1+w, 2+w, 3)) for w ∈ 1:W64] == getindex.(Ref(C), 1 .+ (1:W64), 2 .+ (1:W64), 3)
        vstore!(stridedpointer(C), !mtest, ((MM{16})(17), 3, 4))
        @test .!v1 == C[17:32,3,4] == tovector(vload(stridedpointer(C), ((MM{16})(17), 3, 4)))

        dims = (41,42,43) .* 3;
        # dims = (41,42,43);
        A = reshape(collect(Float64(0):Float64(prod(dims)-1)), dims);

        P = PermutedDimsArray(A, (3,1,2));
        O = OffsetArray(P, (-4, -2, -3));
        indices = (
            StaticInt{1}(), StaticInt{2}(), 2, MM{W64}(2), MM{W64,2}(3), Vec(ntuple(i -> 2i + 1, Val(W64))...),
            VectorizationBase.LazyMulAdd{2,-1}(MM{W64}(3)), VectorizationBase.LazyMulAdd{2,-2}(Vec(ntuple(i -> 2i + 1, Val(W64))...))
        )
        # for i ∈ indices, j ∈ indices, k ∈ indices, B ∈ [A, P, O]
        for _i ∈ indices, _j ∈ indices, _k ∈ indices, im ∈ 1:3, jm ∈ 1:3, km ∈ 1:3, B ∈ (A, P, O)
            i = VectorizationBase.lazymul(im, _i)
            j = VectorizationBase.lazymul(jm, _j)
            k = VectorizationBase.lazymul(km, _k)
            iv = tovector(i); jv = tovector(j); kv = tovector(k)
            if B === C
                off = 9 - iv[1] % 8
                iv += off
                i += off
            end
            # @show typeof(B), i, j, k (im, _i), (jm, _j), (km, _k)
            x = getindex.(Ref(B), iv, jv, kv)
            GC.@preserve B begin
                # @show i,j,k, typeof(B)
                v = @inferred(vload(stridedpointer(B), (i, j, k)))
            end
            @test x == tovector(v)
            if length(x) > 1
                m = Mask{W64}(rand(UInt8))
                mv = tovector(m)
                x .*= mv
                GC.@preserve B begin
                    v = @inferred(vload(stridedpointer(B), (i, j, k), m))
                end
                @test x == tovector(v)
            end
            for store! ∈ (vstore!, VectorizationBase.vnoaliasstore!)
                y = isone(length(x)) ? randn() : randnvec(length(x))
                GC.@preserve B store!(stridedpointer(B), y, (i, j, k))
                x = getindex.(Ref(B), iv, jv, kv)
                # @show i, j, k typeof.((i, j, k)), store!, typeof(B) y
                @test x == tovector(y)
                if length(x) > 1
                    z = Vec(ntuple(_ -> Core.VecElement(randn()), length(x)))
                    GC.@preserve B store!(stridedpointer(B), z, (i, j, k), m)
                    y = getindex.(Ref(B), iv, jv, kv)
                    @test y == ifelse.(mv, tovector(z), x)
                end
            end
        end
        for AU ∈ 1:3, B ∈ (A, P, O)
            i, j, k = 2, 3, 4
            for AV ∈ 1:3
                v1 = randnvec(); v2 = randnvec(); v3 = randnvec();
                GC.@preserve B begin
                    vstore!(stridedpointer(B), VectorizationBase.VecUnroll((v1,v2,v3)), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k)))
                    vu = @inferred(vload(stridedpointer(B), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k))))
                end
                @test v1 === vu.data[1]
                @test v2 === vu.data[2]
                @test v3 === vu.data[3]

                ir = 0:(AV == 1 ? W64-1 : 0); jr = 0:(AV == 2 ? W64-1 : 0); kr = 0:(AV == 3 ? W64-1 : 0)
                x1 = getindex.(Ref(B), i .+ ir, j .+ jr, k .+ kr)
                if AU == 1
                    ir = ir .+ length(ir)
                elseif AU == 2
                    jr = jr .+ length(jr)
                elseif AU == 3
                    kr = kr .+ length(kr)
                end
                x2 = getindex.(Ref(B), i .+ ir, j .+ jr, k .+ kr)
                if AU == 1
                    ir = ir .+ length(ir)
                elseif AU == 2
                    jr = jr .+ length(jr)
                elseif AU == 3
                    kr = kr .+ length(kr)
                end
                x3 = getindex.(Ref(B), i .+ ir, j .+ jr, k .+ kr)

                @test x1 == tovector(vu.data[1])
                @test x2 == tovector(vu.data[2])
                @test x3 == tovector(vu.data[3])

            end
            v1 = randnvec(); v2 = randnvec(); v3 = randnvec(); v4 = randnvec(); v5 = randnvec()
            GC.@preserve B begin
                vstore!(VectorizationBase.vsum, stridedpointer(B), VectorizationBase.VecUnroll((v1,v2,v3,v4,v5)), VectorizationBase.Unroll{AU,1,5,0,W64,zero(UInt)}((i, j, k)))
            end
            ir = 0:(AU == 1 ? 4 : 0); jr = 0:(AU == 2 ? 4 : 0); kr = 0:(AU == 3 ? 4 : 0)
            xvs = getindex.(Ref(B), i .+ ir, j .+ jr, k .+ kr)
            @test xvs ≈ map(VectorizationBase.vsum, [v1,v2,v3,v4,v5])
        end
        x = Vector{Int}(undef, 100);
        i = MM{1}(0)
        for j ∈ 1:25
            vstore!(pointer(x), j, (i * VectorizationBase.static_sizeof(Int)))
            i += 1
        end
        for j ∈ 26:50
            vstore!(pointer(x), j, (VectorizationBase.static_sizeof(Int) * i), Mask{1}(0xff))
            i += 1
        end
        for j ∈ 51:75
            vstore!(pointer(x), j, VectorizationBase.lazymul(i, VectorizationBase.static_sizeof(Int)))
            i += 1
        end
        for j ∈ 76:100
            vstore!(pointer(x), j, VectorizationBase.lazymul(VectorizationBase.static_sizeof(Int), i), Mask{1}(0xff))
            i += 1
        end
        @test x == 1:100

        xf64 = rand(100);
        vxtu = @inferred(vload(stridedpointer(xf64), (MM{4W64}(1),)));
        @test vxtu isa VectorizationBase.VecUnroll
        vxtutv = tovector(vxtu);
        vxtutvmult = 3.5 .* vxtutv;
        @inferred(vstore!(stridedpointer(xf64), 3.5 * vxtu, (MM{4W64}(1),)));
        @test tovector(@inferred(vload(stridedpointer(xf64), (MM{4W64}(1),)))) == vxtutvmult
        mbig = Mask{4W64}(rand(UInt32)); # TODO: update if any arches support >512 bit vectors
        mbigtv = tovector(mbig);
        @test tovector(@inferred(vload(stridedpointer(xf64), (MM{4W64}(1),), mbig))) == ifelse.(mbigtv, vxtutvmult, 0.0)
        @inferred(vstore!(stridedpointer(xf64), -11 * vxtu, (MM{4W64}(1),), mbig));
        @test tovector(@inferred(vload(stridedpointer(xf64), (MM{4W64}(1),)))) == ifelse.(mbigtv, -11 .* vxtutv, vxtutvmult)
    end

    @time @testset "Grouped Strided Pointers" begin
        M, K, N = 4, 5, 6
        A = rand(M, K); B = rand(K, N); C = rand(M, N);
        fs = [identity, adjoint]
        for fA ∈ fs, fB ∈ fs, fC ∈ fs
            At = fA === identity ? A : copy(A')'
            Bt = fB === identity ? B : copy(B')'
            Ct = fC === identity ? C : copy(C')'
            gsp = @inferred(VectorizationBase.grouped_strided_pointer((At,Bt,Ct), Val{(((1,1),(3,1)),((1,2),(2,1)),((2,2),(3,2)))}()))
            if fA === fC
                @test sizeof(gsp.strides) == 2sizeof(Int)
            end
            @test sizeof(gsp.offsets) == 0
            pA, pB, pC = VectorizationBase.stridedpointers(gsp)
            @test pA === stridedpointer(At)
            @test pB === stridedpointer(Bt)
            @test pC === stridedpointer(Ct)
        end
    end

    @time @testset "Unary Functions" begin
        for T ∈ (Float32,Float64)
            v = let W = VectorizationBase.pick_vector_width_val(T)
                VectorizationBase.VecUnroll((
                    Vec(ntuple(_ -> (randn(T)), W)...),
                    Vec(ntuple(_ -> (randn(T)), W)...),
                    Vec(ntuple(_ -> (randn(T)), W)...)
                ))
            end
            x = tovector(v)
            for f ∈ [
                -, abs, inv, floor, ceil, trunc, round, VectorizationBase.relu, abs2,
                Base.FastMath.abs2_fast, Base.FastMath.sub_fast, sign
            ]
                # @show T, f
                @test tovector(@inferred(f(v))) == map(f, x)
            end
            # Don't require exact, but `eps(T)` seems like a reasonable `rtol`, at least on AVX512 systems:
            # function relapprox(x::AbstractVector{T},y) where {T}
            #     t = max(norm(x),norm(y)) * eps(T)
            #     n = norm(x .- y)
            #     n / t
            # end
            # function randapprox(::Type{T}) where {T}
            #     x = Vec(ntuple(_ -> 10randn(T), VectorizationBase.pick_vector_width_val(T))...)
            #     via = @fastmath inv(x)
            #     vir = inv(x)
            #     relapprox(tovector(via), tovector(vir))
            # end
            # f32t = map(_ -> randapprox(Float32), 1:1_000_000);
            # f64t = map(_ -> randapprox(Float64), 1:1_000_000);
            # summarystats(f64t)
            # summarystats(f32t)
            # for now, I'll use `4eps(T)` if the systems don't have AVX512, but should check to set a stricter bound.
            # also put `sqrt ∘ abs` in here
            let rtol = eps(T) * (VectorizationBase.AVX512F ? 1 : 4) # more accuracte
                @test isapprox(tovector(@inferred(Base.FastMath.inv_fast(v))), map(Base.FastMath.inv_fast, x), rtol = rtol)
                let f = sqrt ∘ abs
                    if T === Float32
                        @test isapprox(tovector(@inferred(f(v))), map(f, x), rtol = rtol)
                    elseif T === Float64 # exact with `Float64`
                        @test tovector(@inferred(f(v))) == map(f, x)
                    end
                end
            end
            for f ∈ [floor, ceil, trunc, round]
                @test tovector(@inferred(f(Int32, v))) == map(y -> f(Int32,y), x)
                @test tovector(@inferred(f(Int64, v))) == map(y -> f(Int64,y), x)
            end
            invtol = VectorizationBase.AVX512F ? 2^-14 : 1.5*2^-12 # moreaccurate with AVX512
            @test isapprox(tovector(@inferred(VectorizationBase.inv_approx(v))), map(VectorizationBase.inv_approx, x), rtol = invtol)
        end

        int = VectorizationBase.AVX512DQ ? Int : Int32
        vi = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> rand(int), Val(W64))...),
            Vec(ntuple(_ -> rand(int), Val(W64))...),
            Vec(ntuple(_ -> rand(int), Val(W64))...)
        )) % Int
        xi = tovector(vi)
        for f ∈ [-, abs, inv, floor, ceil, trunc, round, sqrt ∘ abs, sign]
            @test tovector(@inferred(f(vi))) == map(f, xi)
        end
        let rtol = eps(Float64) * (VectorizationBase.AVX512F ? 1 : 4) # more accuracte
            @test isapprox(tovector(@inferred(Base.FastMath.inv_fast(vi))), map(Base.FastMath.inv_fast, xi), rtol = rtol)
        end
        # vpos = VectorizationBase.VecUnroll((
        #     Vec(ntuple(_ -> Core.VecElement(rand()), Val(W64))),
        #     Vec(ntuple(_ -> Core.VecElement(rand()), Val(W64))),
        #     Vec(ntuple(_ -> Core.VecElement(rand()), Val(W64)))
        # ))
        # for f ∈ [sqrt]
        #     @test tovector(f(vpos)) == map(f, tovector(vpos))
        # end
    end
    @time @testset "Binary Functions" begin
        # TODO: finish getting these tests to pass
        # for I1 ∈ (Int32,Int64,UInt32,UInt64), I2 ∈ (Int32,Int64,UInt32,UInt64)
        for I1 ∈ (Int32,Int64), I2 ∈ (Int32,Int64,UInt32)
            # TODO: No longer skip these either.
            sizeof(I1) > sizeof(I2) && continue
            vi1 = VectorizationBase.VecUnroll((
                Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(W64)))
            ))
            srange = one(I2):(VectorizationBase.AVX512DQ ? I2(8sizeof(I1)-1) : I2(31))
            vi2 = VectorizationBase.VecUnroll((
                Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(W64))),
                Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(W64)))
            ))
            i = rand(srange); j = rand(I1);
            m1 = VectorizationBase.VecUnroll((MM{W64}(I1(7)), MM{W64}(I1(1)), MM{W64}(I1(13)), MM{W64}(I1(32%last(srange)))));
            m2 = VectorizationBase.VecUnroll((MM{W64,2}(I2(3)), MM{W64,2}(I2(8)), MM{W64,2}(I2(39%last(srange))), MM{W64,2}(I2(17))));
            xi1 = tovector(vi1); xi2 = tovector(vi2);
            xi3 =  mapreduce(tovector, vcat, m1.data);
            xi4 =  mapreduce(tovector, vcat, m2.data);
            I3 = promote_type(I1,I2);
            # I4 = sizeof(I1) < sizeof(I2) ? I1 : (sizeof(I1) > sizeof(I2) ? I2 : I3)
            for f ∈ [
                +, -, *, ÷, /, %, <<, >>, >>>, ⊻, &, |, fld, mod,
                VectorizationBase.rotate_left, VectorizationBase.rotate_right, copysign, maxi, mini, maxi_fast, mini_fast
            ]
            # for f ∈ [+, -, *, div, ÷, /, rem, %, <<, >>, >>>, ⊻, &, |, fld, mod, VectorizationBase.rotate_left, VectorizationBase.rotate_right, copysign, max, min]
                # @show f, I1, I2
                # if (!VectorizationBase.AVX512DQ) && (f === /) && sizeof(I1) === sizeof(I2) === 8
                #     continue
                # end
                check_within_limits(tovector(@inferred(f(vi1, vi2))),  trunc_int.(f.(size_trunc_int.(xi1, I3), size_trunc_int.(xi2, I3)), I3));
                check_within_limits(tovector(@inferred(f(j, vi2))), trunc_int.(f.(size_trunc_int.(j, I3), size_trunc_int.(xi2, I3)), I3));
                check_within_limits(tovector(@inferred(f(vi1, i))), trunc_int.(f.(size_trunc_int.(xi1, I3), size_trunc_int.(i, I3)), I3));
                check_within_limits(tovector(@inferred(f(m1, i))), trunc_int.(f.(size_trunc_int.(xi3, I3), size_trunc_int.(i, I3)), I3));
                check_within_limits(tovector(@inferred(f(m1, vi2))), trunc_int.(f.(size_trunc_int.(xi3, I3), size_trunc_int.(xi2, I3)), I3));
                check_within_limits(tovector(@inferred(f(m1, m2))), trunc_int.(f.(size_trunc_int.(xi3, I3), size_trunc_int.(xi4, I3)), I3));
                check_within_limits(tovector(@inferred(f(m1, m1))), trunc_int.(f.(size_trunc_int.(xi3, I1), size_trunc_int.(xi3, I1)), I1));
                check_within_limits(tovector(@inferred(f(m2, i))), trunc_int.(f.(size_trunc_int.(xi4, I3), size_trunc_int.(i, I3)), I2));
                check_within_limits(tovector(@inferred(f(m2, vi2))), trunc_int.(f.(size_trunc_int.(xi4, I3), size_trunc_int.(xi2, I3)), I2));
                check_within_limits(tovector(@inferred(f(m2, m2))), trunc_int.(f.(size_trunc_int.(xi4, I3), size_trunc_int.(xi4, I3)), I2));
                check_within_limits(tovector(@inferred(f(m2, m1))), trunc_int.(f.(size_trunc_int.(xi4, I3), size_trunc_int.(xi3, I3)), I3));
                if !((f === VectorizationBase.rotate_left) || (f === VectorizationBase.rotate_right))
                    check_within_limits(tovector(@inferred(f(j, m1))), trunc_int.(f.(j, xi3), I1));
                    check_within_limits(tovector(@inferred(f(j, m2))), trunc_int.(f.(size_trunc_int.(j, I1), size_trunc_int.(xi4, I1)), I1));
                end
            end
            @test tovector(@inferred(vi1 ^ i)) ≈ xi1 .^ i
            @test @inferred(VectorizationBase.vall(@inferred(1 - MM{W64}(1)) == (1 - Vec(ntuple(identity, Val(W64))...)) ))
        end
        vf1 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        ))
        vf2 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        xf1 = tovector(vf1); xf2 = tovector(vf2); xf22 = vcat(xf2,xf2)
        a = randn();
        for f ∈ [+, -, *, /, %, max, min, copysign, rem, Base.FastMath.max_fast, Base.FastMath.min_fast, Base.FastMath.div_fast, Base.FastMath.rem_fast]
            # @show f
            @test tovector(@inferred(f(vf1, vf2))) ≈ f.(xf1, xf22)
            @test tovector(@inferred(f(a, vf1))) ≈ f.(a, xf1)
            @test tovector(@inferred(f(a, vf2))) ≈ f.(a, xf2)
            @test tovector(@inferred(f(vf1, a))) ≈ f.(xf1, a)
            @test tovector(@inferred(f(vf2, a))) ≈ f.(xf2, a)
        end

        vi2 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64)))
        ))
        vones, vi2f, vtwos = promote(1.0, vi2, 2f0); # promotes a binary function, right? Even when used with three args?
        @test vones === VectorizationBase.VecUnroll((vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0)));
        @test vtwos === VectorizationBase.VecUnroll((vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0)));
        @test VectorizationBase.vall(vi2f == vi2)
        W32 = StaticInt(W64)*StaticInt(2)
        vf2 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(randn(Float32)), W32)),
            Vec(ntuple(_ -> Core.VecElement(randn(Float32)), W32))
        ))
        vones32, v2f32, vtwos32 = promote(1.0, vf2, 2f0); # promotes a binary function, right? Even when used with three args?
        @test vones32 === VectorizationBase.VecUnroll((vbroadcast(W32, 1f0),vbroadcast(W32, 1f0)))
        @test vtwos32 === VectorizationBase.VecUnroll((vbroadcast(W32, 2f0),vbroadcast(W32, 2f0)))
        @test vf2 === v2f32

        i = rand(1:31)
        m1 = VectorizationBase.VecUnroll((MM{W64}(7), MM{W64}(1), MM{W64}(13), MM{W64}(18)))
        @test tovector(clamp(m1, 2:i)) == clamp.(tovector(m1), 2, i)
        @test tovector(mod(m1, 1:i)) == mod1.(tovector(m1), i)

        @test VectorizationBase.vdivrem.(1:30, 1:30') == divrem.(1:30, 1:30')
        @test VectorizationBase.vcld.(1:30, 1:30') == cld.(1:30, 1:30')
        @test VectorizationBase.vrem.(1:30, 1:30') == rem.(1:30, 1:30')
    end
    @time @testset "Ternary Functions" begin
        for T ∈ (Float32, Float64)
            v1, v2, v3, m = let W = @inferred(VectorizationBase.pick_vector_width_val(T))
                v1 = VectorizationBase.VecUnroll((
                    Vec(ntuple(_ -> randn(T), W)...),
                    Vec(ntuple(_ -> randn(T), W)...)
                ))
                v2 = VectorizationBase.VecUnroll((
                    Vec(ntuple(_ -> randn(T), W)...),
                    Vec(ntuple(_ -> randn(T), W)...)
                ))
                v3 = VectorizationBase.VecUnroll((
                    Vec(ntuple(_ -> randn(T), W)...),
                    Vec(ntuple(_ -> randn(T), W)...)
                ))
                _W = @inferred(VectorizationBase.pick_vector_width(T))
                m = VectorizationBase.VecUnroll((Mask{_W}(rand(UInt16)),Mask{_W}(rand(UInt16))))
                v1, v2, v3, m
            end
            x1 = tovector(v1); x2 = tovector(v2); x3 = tovector(v3);
            a = randn(T); b = randn(T)
            a64 = Float64(a); b64 = Float64(b); # test promotion
            mv = tovector(m)
            for f ∈ [
                muladd, fma, clamp, VectorizationBase.vmuladd_fast, VectorizationBase.vfma_fast,
                VectorizationBase.vfmadd, VectorizationBase.vfnmadd, VectorizationBase.vfmsub, VectorizationBase.vfnmsub,
                VectorizationBase.vfmadd_fast, VectorizationBase.vfnmadd_fast, VectorizationBase.vfmsub_fast, VectorizationBase.vfnmsub_fast,
                VectorizationBase.vfmadd231, VectorizationBase.vfnmadd231, VectorizationBase.vfmsub231, VectorizationBase.vfnmsub231
            ]
                @test tovector(@inferred(f(v1, v2, v3))) ≈ map(f, x1, x2, x3)
                @test tovector(@inferred(f(v1, v2, a64))) ≈ f.(x1, x2, a)
                @test tovector(@inferred(f(v1, a64, v3))) ≈ f.(x1, a, x3)
                @test tovector(@inferred(f(a64, v2, v3))) ≈ f.(a, x2, x3)
                @test tovector(@inferred(f(v1, a64, b64))) ≈ f.(x1, a, b)
                @test tovector(@inferred(f(a64, v2, b64))) ≈ f.(a, x2, b)
                @test tovector(@inferred(f(a64, b64, v3))) ≈ f.(a, b, x3)

                @test tovector(@inferred(VectorizationBase.ifelse(f, m, v1, v2, v3))) ≈ ifelse.(mv, f.(x1, x2, x3), x3)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, v1, v2, a64))) ≈ ifelse.(mv, f.(x1, x2, a), a)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, v1, a64, v3))) ≈ ifelse.(mv, f.(x1, a, x3), x3)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, a64, v2, v3))) ≈ ifelse.(mv, f.(a, x2, x3), x3)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, v1, a64, b64))) ≈ ifelse.(mv, f.(x1, a, b), b)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, a64, v2, b64))) ≈ ifelse.(mv, f.(a, x2, b), b)
                @test tovector(@inferred(VectorizationBase.ifelse(f, m, a64, b64, v3))) ≈ ifelse.(mv, f.(a, b, x3), x3)
            end
        end
        vi64 = VectorizationBase.VecUnroll((
           Vec(ntuple(_ -> rand(Int64), Val(W64))...),
           Vec(ntuple(_ -> rand(Int64), Val(W64))...),
           Vec(ntuple(_ -> rand(Int64), Val(W64))...)
        ))
        vi32 = VectorizationBase.VecUnroll((
           Vec(ntuple(_ -> rand(Int32), Val(W64))...),
           Vec(ntuple(_ -> rand(Int32), Val(W64))...),
           Vec(ntuple(_ -> rand(Int32), Val(W64))...)
        ))
        xi64 = tovector(vi64); xi32 = tovector(vi32);
        @test tovector(@inferred(VectorizationBase.ifelse(vi64 > vi32, vi64, vi32))) == ifelse.(xi64 .> xi32, xi64, xi32)
        @test tovector(@inferred(VectorizationBase.ifelse(vi64 < vi32, vi64, vi32))) == ifelse.(xi64 .< xi32, xi64, xi32)
        @test tovector(@inferred(VectorizationBase.ifelse(true, vi64, vi32))) == ifelse.(true, xi64, xi32)
        @test tovector(@inferred(VectorizationBase.ifelse(false, vi64, vi32))) == ifelse.(false, xi64, xi32)
        vu64_1 = VectorizationBase.VecUnroll((
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...)
        ))
        vu64_2 = VectorizationBase.VecUnroll((
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...)
        ))
        vu64_3 = VectorizationBase.VecUnroll((
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...),
           Vec(ntuple(_ -> rand(UInt64), Val(W64))...)
        ))
        xu1 = tovector(vu64_1); xu2 = tovector(vu64_2); xu3 = tovector(vu64_3);
        for f ∈ [clamp, muladd, VectorizationBase.ifmalo, VectorizationBase.ifmahi, VectorizationBase.vfmadd, VectorizationBase.vfnmadd, VectorizationBase.vfmsub, VectorizationBase.vfnmsub]
            @test tovector(@inferred(f(vu64_1,vu64_2,vu64_3))) == f.(xu1, xu2, xu3)
        end
    end
    @time @testset "Special functions" begin
        if VERSION ≥ v"1.6.0-DEV.674" && VectorizationBase.SSE4_1
            erfs = [0.1124629160182849, 0.22270258921047847, 0.3286267594591274, 0.42839235504666845, 0.5204998778130465, 0.6038560908479259, 0.6778011938374184, 0.7421009647076605, 0.7969082124228322, 0.8427007929497149, 0.8802050695740817, 0.9103139782296353, 0.9340079449406524, 0.9522851197626487, 0.9661051464753108, 0.976348383344644, 0.9837904585907745, 0.9890905016357308, 0.9927904292352575, 0.9953222650189527, 0.997020533343667, 0.9981371537020182, 0.9988568234026434, 0.999311486103355, 0.999593047982555, 0.9997639655834707, 0.9998656672600594, 0.9999249868053346, 0.9999589021219005, 0.9999779095030014, 0.9999883513426328, 0.9999939742388483]
            if VectorizationBase.AVX512F
                v = VectorizationBase.verf(Vec{8, Float64}(0.1:0.1:0.8...,))
                @test [v(i) for i in 1:8] ≈ erfs[1:8]
                v = VectorizationBase.verf(Vec{16, Float32}(0.1:0.1:1.6...,))
                @test [v(i) for i in 1:16] ≈ erfs[1:16]
            end
            if VectorizationBase.AVX
                v = VectorizationBase.verf(Vec{4, Float64}(0.1:0.1:0.4...,))
                @test [v(i) for i in 1:4] ≈ erfs[1:4]
                v = VectorizationBase.verf(Vec{8, Float32}(0.1:0.1:0.8...,))
                @test [v(i) for i in 1:8] ≈ erfs[1:8]
            end
            if VectorizationBase.SSE4_1
                v = VectorizationBase.verf(Vec{2, Float64}(0.1:0.1:0.2...,))
                @test [v(i) for i in 1:2] ≈ erfs[1:2]
            end
        end
    end
    @time @testset "Non-broadcasting operations" begin
        v1 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))); vu1 = VectorizationBase.VecUnroll((v1, Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))));
        v2 = Vec(ntuple(_ -> Core.VecElement(rand(-100:100)), Val(W64))); vu2 = VectorizationBase.VecUnroll((v2, Vec(ntuple(_ -> Core.VecElement(rand(-100:100)), Val(W64)))));
        @test @inferred(VectorizationBase.vsum(2.3, v1)) ≈ @inferred(VectorizationBase.vsum(v1)) + 2.3 ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(v1, 2.3))) ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(2.3, v1)))
        @test @inferred(VectorizationBase.vsum(vu1)) + 2.3 ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(vu1, 2.3))) ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(2.3, vu1)))
        @test @inferred(VectorizationBase.vsum(v2)) + 3 == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(v2, 3))) == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(3, v2)))
        @test @inferred(VectorizationBase.vsum(vu2)) + 3 == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(vu2, 3))) == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(3, vu2)))
        @test @inferred(VectorizationBase.vprod(v1)) * 2.3 ≈ @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(v1, 2.3))) ≈ @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(2.3, v1)))
        @test @inferred(VectorizationBase.vprod(v2)) * 3 == @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(3, v2)))
        @test @inferred(VectorizationBase.vall(v1 + v2 == VectorizationBase.addscalar(v1, v2)))
        @test 4.0 == @inferred(VectorizationBase.addscalar(2.0, 2.0))

        v3 = Vec(0, 1, 2, 3); vu3 = VectorizationBase.VecUnroll((v3, v3 - 1))
        v4 = Vec(0.0, 1.0, 2.0, 3.0)
        v5 = Vec(0f0, 1f0, 2f0, 3f0, 4f0, 5f0, 6f0, 7f0)
        @test @inferred(VectorizationBase.vmaximum(v3)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3, 2)))
        @test @inferred(VectorizationBase.vmaximum(v3 % UInt)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3 % UInt, 2 % UInt)))
        @test @inferred(VectorizationBase.vmaximum(v4)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v4, prevfloat(3.0))))
        @test @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v4, nextfloat(3.0)))) == nextfloat(3.0)
        @test @inferred(VectorizationBase.vmaximum(v5)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v5, prevfloat(7f0)))) === VectorizationBase.vmaximum(VectorizationBase.maxscalar(prevfloat(7f0), v5))
        @test @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v5, nextfloat(7f0)))) == @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(nextfloat(7f0), v5))) == nextfloat(7f0)

        @test VectorizationBase.maxscalar(v3, 2) === Vec(2, 1, 2, 3)
        @test VectorizationBase.maxscalar(v3, -1) === v3
        @test VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3 % UInt, -1 % UInt)) === -1 % UInt
        @test VectorizationBase.maxscalar(v4, 1e-16) === Vec(1e-16, 1.0, 2.0, 3.0)
        @test VectorizationBase.maxscalar(v4, -1e-16) === v4
        @test VectorizationBase.vmaximum(vu3) == 3
        @test VectorizationBase.vmaximum(VectorizationBase.maxscalar(vu3,2)) == 3
        @test VectorizationBase.vmaximum(VectorizationBase.maxscalar(vu3,4)) == 4
        @test VectorizationBase.vminimum(vu3) == -1
        @test VectorizationBase.vminimum(VectorizationBase.minscalar(vu3,0)) == -1
        @test VectorizationBase.vminimum(VectorizationBase.minscalar(vu3,-2)) == VectorizationBase.vminimum(VectorizationBase.minscalar(-2,vu3)) == -2
    end
    @time @testset "broadcasting" begin
        @test VectorizationBase.vzero(Val(1), UInt32) === VectorizationBase.vzero(StaticInt(1), UInt32) === VectorizationBase.vzero(UInt32) === zero(UInt32)
        @test VectorizationBase.vzero(Val(1), Int) === VectorizationBase.vzero(StaticInt(1), Int) === VectorizationBase.vzero(Int) === 0
        @test VectorizationBase.vzero(Val(1), Float32) === VectorizationBase.vzero(StaticInt(1), Float32) === VectorizationBase.vzero(Float32) === 0f0
        @test VectorizationBase.vzero(Val(1), Float64) === VectorizationBase.vzero(StaticInt(1), Float64) === VectorizationBase.vzero(Float64) === 0.0
        @test VectorizationBase.vzero() === VectorizationBase.vzero(W64S, Float64)
        @test VectorizationBase.vbroadcast(StaticInt(2)*W64S, one(Int64)) === VectorizationBase.vbroadcast(StaticInt(2)*W64S, one(Int32))
        @test VectorizationBase.vbroadcast(StaticInt(2)*W64S, one(UInt64)) === VectorizationBase.vbroadcast(StaticInt(2)*W64S, one(UInt32)) === one(VectorizationBase.Vec{2W64,UInt64}) === oneunit(VectorizationBase.Vec{2W64,UInt64})

        @test VectorizationBase.vall(VectorizationBase.vbroadcast(W64S, pointer(A)) == vbroadcast(W64S, first(A)))
        @test VectorizationBase.vbroadcast(W64S, pointer(A,2)) === Vec{W64}(A[2]) === Vec(A[2])

        @test zero(VectorizationBase.VecUnroll((VectorizationBase.vbroadcast(W64S, pointer(A)), VectorizationBase.vbroadcast(W64S, pointer(A,2))))) === VectorizationBase.VecUnroll((VectorizationBase.vzero(W64S, Float64), VectorizationBase.vzero()))

        @test VectorizationBase.VecUnroll{2,W64,Float64}(first(A)) === VectorizationBase.VecUnroll{2,W64,Float64}(VectorizationBase.vbroadcast(W64S, pointer(A))) === VectorizationBase.VecUnroll((VectorizationBase.vbroadcast(W64S, pointer(A)),VectorizationBase.vbroadcast(W64S, pointer(A)),VectorizationBase.vbroadcast(W64S, pointer(A)))) === VectorizationBase.VecUnroll{2}(VectorizationBase.vbroadcast(W64S, pointer(A)))
    end
    @time @testset "CartesianVIndex" begin
        @test VectorizationBase.maybestaticfirst(CartesianIndices(A)) === VectorizationBase.CartesianVIndex(ntuple(_ -> VectorizationBase.One(), ndims(A)))
        @test VectorizationBase.maybestaticlast(CartesianIndices(A)) === VectorizationBase.CartesianVIndex(size(A))
        @test VectorizationBase.CartesianVIndex((StaticInt(1),2,VectorizationBase.CartesianVIndex((StaticInt(4), StaticInt(7))), CartesianIndex(12,14), StaticInt(2), 1)) === VectorizationBase.CartesianVIndex((StaticInt(1),2,StaticInt(4),StaticInt(7),12,14,StaticInt(2),1))
        @test Tuple(VectorizationBase.CartesianVIndex((StaticInt(1),2,VectorizationBase.CartesianVIndex((StaticInt(4), StaticInt(7))), CartesianIndex(12,14), StaticInt(2), 1))) === (StaticInt(1),2,StaticInt(4),StaticInt(7),12,14,StaticInt(2),1)
        @test length(VectorizationBase.CartesianVIndex((StaticInt(1),2,VectorizationBase.CartesianVIndex((StaticInt(4), StaticInt(7))), CartesianIndex(12,14), StaticInt(2), 1))) === 8
        @test VectorizationBase.static_length(VectorizationBase.CartesianVIndex((StaticInt(1),2,VectorizationBase.CartesianVIndex((StaticInt(4), StaticInt(7))), CartesianIndex(12,14), StaticInt(2), 1))) === StaticInt{8}()
        @test VectorizationBase.CartesianVIndex((StaticInt(-4), StaticInt(7))):VectorizationBase.CartesianVIndex((StaticInt(14), StaticInt(73))) === CartesianIndices((StaticInt(-4):StaticInt(14),StaticInt(7):StaticInt(73)))
        @test VectorizationBase.maybestaticfirst(CartesianIndices(A)):VectorizationBase.maybestaticlast(CartesianIndices(A)) == CartesianIndices(A)
        @test VectorizationBase.maybestaticfirst(CartesianIndices(A)):VectorizationBase.maybestaticlast(CartesianIndices(A)) === CartesianIndices(map(i -> VectorizationBase.One():i, size(A)))
    end
    @time @testset "Promotion" begin
        vi2 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(W64)))
        ))
        vones, vi2f, vtwos = @inferred(promote(1.0, vi2, 2f0)); # promotes a binary function, right? Even when used with three args?
        @test vones === VectorizationBase.VecUnroll((vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0),vbroadcast(Val(W64), 1.0)));
        @test vtwos === VectorizationBase.VecUnroll((vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0),vbroadcast(Val(W64), 2.0)));
        @test @inferred(VectorizationBase.vall(vi2f == vi2))
        vf2 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(randn(Float32)), StaticInt(W32))),
            Vec(ntuple(_ -> Core.VecElement(randn(Float32)), StaticInt(W32)))
        ))
        vones32, v2f32, vtwos32 = @inferred(promote(1.0, vf2, 2f0)); # promotes a binary function, right? Even when used with three args?
        @test vones32 === VectorizationBase.VecUnroll((vbroadcast(StaticInt(W32), 1f0),vbroadcast(StaticInt(W32), 1f0)))
        @test vtwos32 === VectorizationBase.VecUnroll((vbroadcast(StaticInt(W32), 2f0),vbroadcast(StaticInt(W32), 2f0)))
        @test vf2 === v2f32


        vm = if VectorizationBase.AVX512DQ
            VectorizationBase.VecUnroll((
                MM{W64}(rand(Int)),MM{W64}(rand(Int)),MM{W64}(rand(Int)),MM{W64}(rand(Int))
            ))
        else
            VectorizationBase.VecUnroll((
                MM{W64}(rand(Int32)),MM{W64}(rand(Int32)),MM{W64}(rand(Int32)),MM{W64}(rand(Int32))
            ))
        end
        @test tovector(@inferred(vm > vi2)) == (tovector(vm) .> tovector(vi2))

        m = Mask{2W64}(rand(UInt));
        v64f = Vec(ntuple(_ -> randn(), Val{2W64}())...)
        v32i = Vec(ntuple(_ -> rand(Int32), Val{2W64}())...)
        mtv = tovector(m); v64ftv = tovector(v64f); v32itv = tovector(v32i);
        vum = @inferred(muladd(v64f, v32i, m))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(v64ftv, v32itv, mtv)

        vum = @inferred(muladd(v64f, m, v32i))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(v64ftv, mtv, v32itv)

        vum = @inferred(muladd(v32i, v64f, m))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(v32itv, v64ftv, mtv)

        vum = @inferred(muladd(v32i, m, v64f))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(v32itv, mtv, v64ftv)

        vum = @inferred(muladd(m, v64f, v32i))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(mtv, v64ftv, v32itv)

        vum = @inferred(muladd(m, v32i, v64f))
        @test vum isa VectorizationBase.VecUnroll
        @test tovector(vum) ≈ muladd.(mtv, v32itv, v64ftv)
    end
    @time @testset "Lazymul" begin
        # partially covered in memory
        for i ∈ (-5, -1, 0, 1, 4, 8), j ∈ (-5, -1, 0, 1, 4, 8)
            @test VectorizationBase.lazymul(StaticInt(i), StaticInt(j)) === VectorizationBase.lazymul_no_promote(StaticInt(i), StaticInt(j)) === StaticInt(i*j)
        end
        fi = VectorizationBase.LazyMulAdd{8,0}(MM{8}(StaticInt(16)))
        si = VectorizationBase.LazyMulAdd{2}(240)
        @test @inferred(VectorizationBase.vadd_fast(fi, si)) === VectorizationBase.LazyMulAdd{2,128}(MM{8,4}(240))
    end
    @time @testset "Arch Functions" begin
        @test VectorizationBase.DYNAMIC_REGISTER_SIZE() == @inferred(VectorizationBase.register_size()) == DYNAMIC_REGISTER_SIZE
        @test VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.DYNAMIC_INTEGER_REGISTER_SIZE == VectorizationBase.SDYNAMIC_INTEGER_REGISTER_SIZE
        @test VectorizationBase.dynamic_register_count() == @inferred(VectorizationBase.register_count()) == @inferred(VectorizationBase.sregister_count())
        @test VectorizationBase.dynamic_fma_fast() == VectorizationBase.fma_fast()
        @test VectorizationBase.dynamic_has_opmask_registers() == VectorizationBase.has_opmask_registers()

        @test VectorizationBase.dynamic_cache_inclusivity() === VectorizationBase.cache_inclusivity()

        @test VectorizationBase.Hwloc.histmap(VectorizationBase.Hwloc.topology_load())[Symbol("L", convert(Int, @inferred(VectorizationBase.snum_cache_levels())), "Cache")] > 0



    end

    @time Aqua.test_all(VectorizationBase)
end

            # ptr_A = pointer(A)
            # vA = VectorizationBase.stridedpointer(A)
            # Att = copy(A')'
            # vAtt = VectorizationBase.stridedpointer(Att)
            # @test eltype(vA) == Float64
            # @test Base.unsafe_convert(Ptr{Float64}, vA) === ptr_A === pointer(vA)
            # @test vA == VectorizationBase.stridedpointer(vA)
            # @test all(i -> A[i+1] === VectorizationBase.vload(ptr_A + 8i) === VectorizationBase.vload(vA, (i,)) === Float64(i), 0:15)
            # VectorizationBase.vstore!(vA, 99.9, (3,))
            # @test 99.9 === VectorizationBase.vload(ptr_A + 8*3) === VectorizationBase.vload(vA, (VectorizationBase.StaticInt(3),)) === VectorizationBase.vload(vA, (3,0)) === A[4,1]
            # VectorizationBase.vstore!(vAtt, 99.9, (3,1))
            # @test 99.9 === VectorizationBase.vload(vAtt, (3,1)) === VectorizationBase.vload(vAtt, (VectorizationBase.StaticInt(3),1)) === Att[4,2]
            # VectorizationBase.vnoaliasstore!(ptr_A+8*4, 999.9)
            # @test 999.9 === VectorizationBase.vload(ptr_A + 8*4) === VectorizationBase.vload(pointer(vA), 4*sizeof(eltype(A))) === VectorizationBase.vload(vA, (4,))
            # @test vload(vA, (7,2)) == vload(vAtt, (7,2)) == A[8,3]
            # @test vload(VectorizationBase.subsetview(vA, Val(1), 7), (2,)) == vload(VectorizationBase.subsetview(vAtt, Val(1), 7), (2,)) == A[8,3]
            # @test vload(VectorizationBase.subsetview(vA, Val(2), 2), (7,)) == vload(VectorizationBase.subsetview(vAtt, Val(2), 2), (7,)) == A[8,3]
            #     @test vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (2,)) == vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (VectorizationBase.StaticInt(2),)) == A[3,3]
            #     @test vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (1,)) == vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (VectorizationBase.StaticInt(1),)) == A[2,2]
            #     B = rand(5, 5)
            # vB = VectorizationBase.stridedpointer(B)
            # @test vB[1, 2] == B[2, 3] == vload(VectorizationBase.stridedpointer(B, 2, 3))
            # @test vB[3] == B[4] == vload(VectorizationBase.stridedpointer(B, 4))
            # @test vload(Vec{4,Float64}, vB) == Vec{4,Float64}(ntuple(i->B[i], Val(4)))
