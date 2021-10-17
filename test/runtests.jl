import InteractiveUtils, Aqua
InteractiveUtils.versioninfo(stdout; verbose=true)

include("testsetup.jl")

@time @testset "VectorizationBase.jl" begin
  # Write your own tests here.
  # Aqua.test_all(VectorizationBase, ambiguities = VERSION < v"1.6-DEV")
  println("Aqua.test_all")
  @time Aqua.test_all(VectorizationBase)
  # @test isempty(detect_unbound_args(VectorizationBase))
  # @test isempty(detect_ambiguities(VectorizationBase))

  W = Int(@inferred(VectorizationBase.pick_vector_width(Float64)))
  @test @inferred(VectorizationBase.pick_integer(Val(W))) == (VectorizationBase.register_size() == VectorizationBase.simd_integer_register_size() ? Int64 : Int32)


  @test first(A) === A[1]
  @test W64S == W64
  println("Struct-Wrapped Vec")
  @time @testset "Struct-Wrapped Vec" begin
    @test data(zero(Vec{W64,Float64})) === ntuple(VE ∘ zero ∘ float, Val(W64)) === data(Vec{W64,Float64}(0.0))
    @test data(one(Vec{W64,Float64})) === ntuple(VE ∘ one ∘ float, Val(W64)) === data(Vec{W64,Float64}(1.0)) === data(data(Vec{W64,Float64}(1.0)))
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
    @test_throws ErrorException vu.data

    @test vu(1,1) === VectorizationBase.data(vu)[1](1)
    @test vu(2,1) === VectorizationBase.data(vu)[1](2)
    @test vu(1,2) === VectorizationBase.data(vu)[2](1)
    @test vu(2,2) === VectorizationBase.data(vu)[2](2)
    if W64 == 8
      @test VectorizationBase.data(vu)[1] === Vec(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
      @test VectorizationBase.data(vu)[2] === Vec(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
    elseif W64 == 4
      @test VectorizationBase.data(vu)[1] === Vec(3.0, 4.0, 5.0, 6.0)
      @test VectorizationBase.data(vu)[2] === Vec(7.0, 8.0, 9.0, 10.0)
      @test VectorizationBase.data(vu)[3] === Vec(11.0, 12.0, 13.0, 14.0)
      @test VectorizationBase.data(vu)[4] === Vec(15.0, 16.0, 17.0, 18.0)
      @test Vec(1.2, 3.4, 3.4) === Vec(1.2, 3.4, 3.4, 0.0)
    elseif W64 == 2
      @test VectorizationBase.data(vu)[1] === Vec(3.0, 4.0)
      @test VectorizationBase.data(vu)[2] === Vec(5.0, 6.0)
      @test VectorizationBase.data(vu)[3] === Vec(7.0, 8.0)
      @test VectorizationBase.data(vu)[4] === Vec(9.0, 10.0)
      @test VectorizationBase.data(vu)[5] === Vec(11.0, 12.0)
      @test VectorizationBase.data(vu)[6] === Vec(13.0, 14.0)
      @test VectorizationBase.data(vu)[7] === Vec(15.0, 16.0)
      @test VectorizationBase.data(vu)[8] === Vec(17.0, 18.0)
    end

  end

  println("alignment.jl")
  @time @testset "alignment.jl" begin
    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.align(i) == VectorizationBase.register_size()
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.align(i) == 2VectorizationBase.register_size()
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 9VectorizationBase.register_size()
      @test VectorizationBase.align(i) == 10VectorizationBase.register_size()
    end
    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},   Int(VectorizationBase.register_size()))
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},  2Int(VectorizationBase.register_size()))
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 19VectorizationBase.register_size()
      @test VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid}, 20Int(VectorizationBase.register_size()))
    end
    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32)
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32)
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size()
      @test VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32)
    end

    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64)
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64)
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size()
      @test VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64)
    end

    @test reinterpret(Int, VectorizationBase.align(pointer(A))) % VectorizationBase.register_size() === 0

    for i ∈ 0:VectorizationBase.register_size()-1
      @test VectorizationBase.aligntrunc(i) == 0
    end
    for i ∈ VectorizationBase.register_size():2VectorizationBase.register_size()-1
      @test VectorizationBase.aligntrunc(i) == VectorizationBase.register_size()
    end
    for i ∈ (0:VectorizationBase.register_size()-1) .+ 9VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i) == 9VectorizationBase.register_size()
    end

    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32)
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32)
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32)
    end

    for i ∈ 1:VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64)
    end
    for i ∈ 1+VectorizationBase.register_size():2VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64)
    end
    for i ∈ (1:VectorizationBase.register_size()) .+ 29VectorizationBase.register_size()
      @test VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64)
    end

    a = Vector{Float64}(undef, 0)
    ptr = pointer(a)
    @test UInt(VectorizationBase.align(ptr, 1 << 12)) % (1 << 12) == 0
  end

  println("masks.jl")
  @time @testset "masks.jl" begin
    # @test Mask{8,UInt8}(0x0f) === @inferred Mask(0x0f)
    # @test Mask{16,UInt16}(0x0f0f) === @inferred Mask(0x0f0f)
    @test EVLMask{8,UInt8}(0xff,8) === mask(Val(8), 0)
    @test EVLMask{8,UInt8}(0xff,8) === mask(Val(8), 8)
    @test EVLMask{8,UInt8}(0xff,8) === mask(Val(8), 16)
    @test EVLMask{8,UInt8}(0xff,8) === mask(Val(8), VectorizationBase.StaticInt(0))
    @test EVLMask{16,UInt16}(0xffff,16) === mask(Val(16), 0)
    @test EVLMask{16,UInt16}(0xffff,16) === mask(Val(16), 16)
    @test EVLMask{16,UInt16}(0xffff,16) === mask(Val(16), 32)
    @test EVLMask{12,UInt16}(0x01ff, 9) === mask(Val(12), 117)
    @test VectorizationBase.data(mask(Val(128),253)) == 0x1fffffffffffffffffffffffffffffff
    @test mask(Val(128),253) === EVLMask{128,UInt128}(0x1fffffffffffffffffffffffffffffff, 125)
    @test EVLMask{1}(true, 1) === true
    @test    Mask{1}(true)    === true
    @test EVLMask{1}(false, 1) === false
    @test    Mask{1}(false)    === false
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
    @test all(w -> VectorizationBase.mask(Float64, w) === VectorizationBase.mask(@inferred(VectorizationBase.pick_vector_width(Float64)), w), 1:W64)

    @test VectorizationBase.vbroadcast(Val(8), true) === Vec(true, true, true, true, true, true, true, true)

    @test !VectorizationBase.vall(Mask{8}(0xfc))
    @test !VectorizationBase.vall(Mask{4}(0xfc))
    @test VectorizationBase.vall(EVLMask{8}(0xff,8))
    @test !VectorizationBase.vall(EVLMask{8}(0x1f,5))
    @test VectorizationBase.vall(Mask{4}(0xcf))

    @test VectorizationBase.vany(Mask{8}(0xfc))
    @test VectorizationBase.vany(Mask{4}(0xfc))
    @test !VectorizationBase.vany(Mask{8}(0x00))
    @test !VectorizationBase.vany(Mask{4}(0xf0))

    @test VectorizationBase.vall(Mask{8}(0xfc) + Mask{8}(0xcf) == Vec(0x01,0x01,0x02,0x02,0x01,0x01,0x02,0x02))
    @test VectorizationBase.vall(Mask{4}(0xfc) + Mask{4}(0xcf) == Vec(0x01,0x01,0x02,0x02))
    @test VectorizationBase.vall(Mask{8}(0xcf) + EVLMask{8}(0x1f,5) == Vec(0x02, 0x02, 0x02, 0x02, 0x01, 0x00, 0x01, 0x01))
    

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

    @test (EVLMask{8}(0x1f,5) | EVLMask{8}(0x03,3)) === EVLMask{8}(0x1f,5)
    @test (Mask{8}(0x1f) | EVLMask{8}(0x03,3)) === Mask{8}(0x1f)
    @test (EVLMask{8}(0x1f,5) | Mask{8}(0x03)) === Mask{8}(0x1f)
    @test (Mask{8}(0x1f) | Mask{8}(0x03)) === Mask{8}(0x1f)

    @test (EVLMask{8}(0x1f,5) & EVLMask{8}(0x03,3)) === EVLMask{8}(0x03,3)
    @test (Mask{8}(0x1f) & EVLMask{8}(0x03,3)) === Mask{8}(0x03)
    @test (EVLMask{8}(0x1f,5) & Mask{8}(0x03)) === Mask{8}(0x03)
    @test (Mask{8}(0x1f) & Mask{8}(0x03)) === Mask{8}(0x03)
    
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
    GC.@preserve fbitvector1 fbitvector2 begin
      vstore!(stridedpointer(fbitvector1), mu, (VectorizationBase.MM(StaticInt{8}(), 1),))
      vstore!(stridedpointer(fbitvector2), mu, (VectorizationBase.MM(StaticInt{8}(), 1),), Mask{8}(0x7e))
      vstore!(stridedpointer(fbitvector1), mu, Unroll{1,4,2,1,4,zero(UInt),1}((9,)))
      vstore!(stridedpointer(fbitvector2), mu, Unroll{1,4,2,1,4,2%UInt,1}((9,)), Mask{4}(0x03))
    end
    @test all(fbitvector1[1:16])
    @test !any(fbitvector1[17:end])
    @test !fbitvector2[1]
    @test all(fbitvector2[2:7])
    @test !fbitvector2[8]
    @test all(fbitvector2[9:14])
    @test !any(fbitvector2[15:end])
  end

  # @testset "number_vectors.jl" begin
  # # eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :(size(A)), 8)) # doesn't work?
  # @test VectorizationBase.length_loads(A, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13*17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13*17, 8)) == divrem(length(A), 8)
  # @test VectorizationBase.size_loads(A,1, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13   )()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13   , 8)) == divrem(size(A,1), 8)
  # @test VectorizationBase.size_loads(A,2, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() ->    17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__,    17, 8)) == divrem(size(A,2), 8)
  # end


  println("vector_width.jl")
  @time @testset "vector_width.jl" begin
    for T ∈ (Float32,Float64)
      @test @inferred(VectorizationBase.pick_vector_width(T)) * @inferred(VectorizationBase.static_sizeof(T)) === @inferred(VectorizationBase.register_size(T)) === @inferred(VectorizationBase.register_size())
    end
    for T ∈ (Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64)
      @test @inferred(VectorizationBase.pick_vector_width(T)) * @inferred(VectorizationBase.static_sizeof(T)) === @inferred(VectorizationBase.register_size(T)) === @inferred(VectorizationBase.simd_integer_register_size())
    end
    @test VectorizationBase.static_sizeof(BigFloat) === VectorizationBase.static_sizeof(Int)
    @test VectorizationBase.static_sizeof(Float32) === VectorizationBase.static_sizeof(Int32) === VectorizationBase.StaticInt(4)
    @test @inferred(VectorizationBase.pick_vector_width(Float16)) === @inferred(VectorizationBase.pick_vector_width(Float32))
    @test @inferred(VectorizationBase.pick_vector_width(Float64, Int32, Float64, Float32, Float64)) * VectorizationBase.static_sizeof(Float64) === @inferred(VectorizationBase.register_size())
    @test @inferred(VectorizationBase.pick_vector_width(Float64, Int32)) * VectorizationBase.static_sizeof(Float64) === @inferred(VectorizationBase.register_size())

    @test @inferred(VectorizationBase.pick_vector_width(Float32, Float32)) * VectorizationBase.static_sizeof(Float32) === @inferred(VectorizationBase.register_size())
    @test @inferred(VectorizationBase.pick_vector_width(Float32, Int32)) * VectorizationBase.static_sizeof(Float32) === @inferred(VectorizationBase.simd_integer_register_size())

    @test all(VectorizationBase._ispow2, 0:1)
    @test all(i -> !any(VectorizationBase._ispow2, 1+(1 << (i-1)):(1 << i)-1 ) && VectorizationBase._ispow2(1 << i), 2:9)
    @test all(i ->  VectorizationBase.intlog2(1 << i) == i, 0:(Int == Int64 ? 53 : 30))
    FTypes = (Float32, Float64)
    Wv = ntuple(i -> @inferred(VectorizationBase.register_size()) >> (i+1), Val(2))
    for (T, N) in zip(FTypes, Wv)
      W = @inferred(VectorizationBase.pick_vector_width(T))
      # @test Vec{Int(W),T} == VectorizationBase.pick_vector(W, T) == VectorizationBase.pick_vector(T)
      @test W == @inferred(VectorizationBase.pick_vector_width(W, T))
      @test W === @inferred(VectorizationBase.pick_vector_width(W, T)) == @inferred(VectorizationBase.pick_vector_width(T))
      while true
        W >>= VectorizationBase.One()
        W == 0 && break
        W2, Wshift2 = @inferred(VectorizationBase.pick_vector_width_shift(W, T))
        @test W2 == VectorizationBase.One() << Wshift2 == @inferred(VectorizationBase.pick_vector_width(W, T)) == VectorizationBase.pick_vector_width(Val(Int(W)),T)  == W
        @test StaticInt(W) === VectorizationBase.pick_vector_width(Val(Int(W)), T) === VectorizationBase.pick_vector_width(W, T)
        for n in W+1:2W
          W3, Wshift3 = VectorizationBase.pick_vector_width_shift(StaticInt(n), T)
          @test W2 << 1 == W3 == 1 << (Wshift2+1) == 1 << Wshift3 == VectorizationBase.pick_vector_width(StaticInt(n), T) == VectorizationBase.pick_vector_width(Val(n),T) == W << 1
          # @test VectorizationBase.pick_vector(W, T) == VectorizationBase.pick_vector(W, T) == Vec{Int(W),T}
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

  println("Memory")
  @time @testset "Memory" begin
    C = rand(40,20,10) .> 0;
    mtest = vload(stridedpointer(C), ((MM{16})(9), 2, 3, 1));
    @test VectorizationBase.offsetprecalc(stridedpointer(C), Val((5,5))) === VectorizationBase.offsetprecalc(VectorizationBase.offsetprecalc(stridedpointer(C), Val((5,5))), Val((3,3)))
    @test VectorizationBase.bytestrides(VectorizationBase.offsetprecalc(stridedpointer(C), Val((5,5)))) === VectorizationBase.bytestrides(stridedpointer(C))
    @test VectorizationBase.bytestrides(C) === VectorizationBase.bytestrides(stridedpointer(C))
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
      StaticInt{1}(), StaticInt{2}(), 2, MM{W64}(2), MM{W64,2}(3), MM{W64,-1}(W64+2), Vec(ntuple(i -> 2i + 1, Val(W64))...)#,
      # VectorizationBase.LazyMulAdd{2,-1}(MM{W64}(3))#, VectorizationBase.LazyMulAdd{2,-2}(Vec(ntuple(i -> 2i + 1, Val(W64))...))
    )
    println("LazyMulAdd Loads/Stores")
    @time @testset "LazyMulAdd Loads/Stores" begin
      max_const = 2
      for _i ∈ indices, _j ∈ indices, _k ∈ indices, im ∈ 1:max_const, jm ∈ 1:max_const, km ∈ 1:max_const, B ∈ (A, P, O)
        i = @inferred(VectorizationBase.lazymul(StaticInt(im), _i))
        j = @inferred(VectorizationBase.lazymul(StaticInt(jm), _j))
        k = @inferred(VectorizationBase.lazymul(StaticInt(km), _k))
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
    end
    println("VecUnroll Loads/Stores")
    @time @testset "VecUnroll Loads/Stores" begin
      for AU ∈ 1:3, B ∈ (A, P, O), i ∈ (StaticInt(1),2,StaticInt(2)), j ∈ (StaticInt(1),3,StaticInt(3)), k ∈ (StaticInt(1),4,StaticInt(4))
        # @show AU, typeof(B), i, j, k
        for AV ∈ 1:3
          v1 = randnvec(); v2 = randnvec(); v3 = randnvec();
          GC.@preserve B begin
            if AU == AV
              vstore!(VectorizationBase.offsetprecalc(stridedpointer(B), Val((5,5,5))), VectorizationBase.VecUnroll((v1,v2,v3)), VectorizationBase.Unroll{AU,W64,3,AV,W64,zero(UInt)}((i, j, k)))
              vu = @inferred(vload(stridedpointer(B), VectorizationBase.Unroll{AU,W64,3,AV,W64,zero(UInt)}((i, j, k))))
            else
              vstore!(stridedpointer(B), VectorizationBase.VecUnroll((v1,v2,v3)), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k)))
              vu = @inferred(vload(stridedpointer(B), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k))))
            end
          end
          @test v1 === VectorizationBase.data(vu)[1]
          @test v2 === VectorizationBase.data(vu)[2]
          @test v3 === VectorizationBase.data(vu)[3]

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

          @test x1 == tovector(VectorizationBase.data(vu)[1])
          @test x2 == tovector(VectorizationBase.data(vu)[2])
          @test x3 == tovector(VectorizationBase.data(vu)[3])

        end
        v1 = randnvec(); v2 = randnvec(); v3 = randnvec(); v4 = randnvec(); v5 = randnvec()
        GC.@preserve B begin
          vstore!(VectorizationBase.vsum, stridedpointer(B), VectorizationBase.VecUnroll((v1,v2,v3,v4,v5)), VectorizationBase.Unroll{AU,1,5,0,1,zero(UInt)}((i, j, k)))
        end
        ir = 0:(AU == 1 ? 4 : 0); jr = 0:(AU == 2 ? 4 : 0); kr = 0:(AU == 3 ? 4 : 0)
        xvs = getindex.(Ref(B), i .+ ir, j .+ jr, k .+ kr)
        @test xvs ≈ map(VectorizationBase.vsum, [v1,v2,v3,v4,v5])
      end
    end
    x = Vector{Int}(undef, 100);
    i = MM{1}(0)
    for j ∈ 1:25
      VectorizationBase.__vstore!(pointer(x), j, (i * VectorizationBase.static_sizeof(Int)), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.register_size())
      i += 1
    end
    for j ∈ 26:50
      VectorizationBase.__vstore!(pointer(x), j, (VectorizationBase.static_sizeof(Int) * i), Mask{1}(0xff), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.register_size())
      i += 1
    end
    for j ∈ 51:75
      VectorizationBase.__vstore!(pointer(x), j, VectorizationBase.lazymul(i, VectorizationBase.static_sizeof(Int)), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.register_size())
      i += 1
    end
    for j ∈ 76:100
      VectorizationBase.__vstore!(pointer(x), j, VectorizationBase.lazymul(VectorizationBase.static_sizeof(Int), i), Mask{1}(0xff), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.False(), VectorizationBase.register_size())
      i += 1
    end
    @test x == 1:100

    
    let ind4 = VectorizationBase.Unroll{1,Int(W64),4,1,Int(W64),zero(UInt)}((1,)),
      xf64 = rand(100), xf16 = rand(Float16, 32)
      # indu = VectorizationBase.VecUnroll((MM{W64}(1,), MM{W64}(1+W64,), MM{W64}(1+2W64,), MM{W64}(1+3W64,)))
      GC.@preserve xf64 begin
        vxtu = @inferred(vload(stridedpointer(xf64), ind4));
        @test vxtu isa VectorizationBase.VecUnroll{3,Int(W64),Float64,Vec{Int(W64),Float64}}
        vxtutv = tovector(vxtu);
        vxtutvmult = 3.5 .* vxtutv;
        @inferred(vstore!(stridedpointer(xf64), 3.5 * vxtu, ind4));
        @test tovector(@inferred(vload(stridedpointer(xf64), ind4))) == vxtutvmult
        mbig = Mask{4W64}(rand(UInt32)); # TODO: update if any arches support >512 bit vectors
        mbigtv = tovector(mbig);
        
        ubig = VectorizationBase.VecUnroll(VectorizationBase.splitvectortotuple(StaticInt(4), W64S, mbig))
        # @test tovector(@inferred(vload(stridedpointer(xf64), indu, ubig))) == ifelse.(mbigtv, vxtutvmult, 0.0)
        @test tovector(@inferred(vload(stridedpointer(xf64), ind4, ubig))) == ifelse.(mbigtv, vxtutvmult, 0.0)
        # @inferred(vstore!(stridedpointer(xf64), -11 * vxtu, indu, ubig));
        # @test tovector(@inferred(vload(stridedpointer(xf64), ind4))) == ifelse.(mbigtv, -11 .* vxtutv, vxtutvmult)
        @inferred(vstore!(stridedpointer(xf64), -77 * vxtu, ind4, ubig));
        @test tovector(@inferred(vload(stridedpointer(xf64), ind4))) == ifelse.(mbigtv, -77 .* vxtutv, vxtutvmult)

        vxf16 = @inferred(vload(stridedpointer(xf16), ind4))
        @test vxf16 isa VectorizationBase.VecUnroll{3,Int(W64),Float16,Vec{Int(W64),Float16}}
        @test tovector(vxf16) == view(xf16, 1:(4*W64))
      end
    end
    colors = [(R = rand(), G = rand(), B = rand()) for i ∈ 1:100];
    colormat = reinterpret(reshape, Float64, colors)
    sp = stridedpointer(colormat)
    GC.@preserve colors begin
      @test tovector(@inferred(vload(sp, VectorizationBase.Unroll{1,1,3,2,8,zero(UInt)}((1,9))))) == vec(colormat[:,9:16]')
      vu = @inferred(vload(sp, VectorizationBase.Unroll{1,1,3,2,8,zero(UInt)}((1,41))))
      @inferred(vstore!(sp, vu, VectorizationBase.Unroll{1,1,3,2,8,zero(UInt)}((1,1))))
    end
    @test vec(colormat[:,41:48]) == vec(colormat[:,1:8])
  end

  println("Grouped Strided Pointers")
  @time @testset "Grouped Strided Pointers" begin
    M, K, N = 4, 5, 6
    A = Matrix{Float64}(undef, M, K);
    B = Matrix{Float64}(undef, K, N);
    C = Matrix{Float64}(undef, M, N);
    struct SizedWrapper{M,N,T,AT<:AbstractMatrix{T}} <: AbstractMatrix{T} ; A::AT; end
    SizedWrapper{M,N}(A::AT) where {M,N,T,AT<:AbstractMatrix{T}} = SizedWrapper{M,N,T,AT}(A) 
    Base.size(::SizedWrapper{M,N}) where {M,N} = (M,N);
    VectorizationBase.size(::SizedWrapper{M,N}) where {M,N} = (StaticInt(M),StaticInt(N));
    Base.getindex(A::SizedWrapper, i...) = getindex(parent(A), i...)
    Base.parent(dw::SizedWrapper) = dw.A
    VectorizationBase.ArrayInterface.parent_type(::Type{SizedWrapper{M,N,T,AT}}) where {M,N,T,AT} = AT
    VectorizationBase.memory_reference(dw::SizedWrapper) = VectorizationBase.memory_reference(parent(dw))
    VectorizationBase.contiguous_axis(::Type{A}) where {A<:SizedWrapper} = VectorizationBase.contiguous_axis(VectorizationBase.ArrayInterface.parent_type(A))
    VectorizationBase.contiguous_batch_size(dw::SizedWrapper) = VectorizationBase.contiguous_batch_size(parent(dw))
    VectorizationBase.stride_rank(::Type{A}) where {A<:SizedWrapper} = VectorizationBase.stride_rank(VectorizationBase.ArrayInterface.parent_type(A))
    VectorizationBase.offsets(dw::SizedWrapper) = VectorizationBase.offsets(parent(dw))
    VectorizationBase.val_dense_dims(dw::SizedWrapper{T,N}) where {T,N} = VectorizationBase.val_dense_dims(parent(dw))
    function VectorizationBase.strides(dw::SizedWrapper{M,N,T}) where {M,N,T}
      x1 = StaticInt(1)
      if VectorizationBase.val_stride_rank(dw) === Val((1,2))
        return x1, x1 * StaticInt{M}()
      else#if VectorizationBase.val_stride_rank(dw) === Val((2,1))
        return x1 * StaticInt{N}(), x1
      end
    end
    
    GC.@preserve A B C begin
      fs = (false,true)#[identity, adjoint]
      for ai ∈ fs, bi ∈ fs, ci ∈ fs
        At = ai ? A : (similar(A')');
        Bt = bi ? B : (similar(B')');
        Ct = ci ? C : (similar(C')');
        spdw = VectorizationBase.DensePointerWrapper{(true,true)}(VectorizationBase.stridedpointer(At))
        gsp, pres = @inferred(VectorizationBase.grouped_strided_pointer((spdw,Bt,Ct), Val{(((1,1),(3,1)),((1,2),(2,1)),((2,2),(3,2)))}()))
        if ai === ci
          @test sizeof(gsp.strides) == 2sizeof(Int)
        end
        # Test to confirm that redundant strides are not stored in the grouped strided pointer
        @test sizeof(gsp) == sizeof(Int) * (6 - (ai & ci) - ((!ai) & bi) - ((!bi) & (!ci)))
        @test sizeof(gsp.offsets) == 0
        pA, pB, pC = @inferred(VectorizationBase.stridedpointers(gsp))
        @test pA === stridedpointer(At)
        @test pB === stridedpointer(Bt)
        @test pC === stridedpointer(Ct)
        Btsw = SizedWrapper{K,N}(Bt)
        gsp2, pres2 = @inferred(VectorizationBase.grouped_strided_pointer((At,Btsw,Ct), Val{(((1,1),(3,1)),((1,2),(2,1)),((2,2),(3,2)))}()));
        @test sizeof(gsp2) == sizeof(Int) * (5 - (ai & ci) - ((!ai) & bi) - ((!bi) & (!ci)))

        pA2, pB2, pC2 = @inferred(VectorizationBase.stridedpointers(gsp2))
        @test pointer(pA2) == pointer(At)
        @test pointer(pB2) == pointer(Bt)
        @test pointer(pC2) == pointer(Ct)
        @test strides(pA2) == strides(pA)
        @test strides(pB2) == strides(pB)
        @test strides(pC2) == strides(pC)
      end
    end

    data_in_large = Array{Float64}(undef, 4, 4, 4, 4, 1);
    data_in = view(data_in_large, :, 1, :, :, 1);
    tmp1= Array{Float64}(undef, 4, 4, 4);
    sp_data_in, sp_tmp1 = VectorizationBase.stridedpointers(VectorizationBase.grouped_strided_pointer((data_in,tmp1), Val((((1,1),(2,1)),)))[1])
    @test sp_data_in === stridedpointer(data_in)
    @test sp_tmp1 === stridedpointer(tmp1)
  end

  println("Adjoint VecUnroll")
  @time @testset "Adjoint VecUnroll" begin
    W = W64
    while W > 1
      A = rand(W,W); B = similar(A);
      GC.@preserve A B begin
        vut = @inferred(vload(stridedpointer(A), VectorizationBase.Unroll{2,1,W,1,W}((1,1))))
        vu = @inferred(VectorizationBase.transpose_vecunroll(vut))
        @test vu === @inferred(vload(stridedpointer(A'), VectorizationBase.Unroll{2,1,W,1,W}((1,1))))
        @test vu === @inferred(vload(stridedpointer(A), VectorizationBase.Unroll{1,1,W,2,W}((1,1))))
        vstore!(stridedpointer(B), vu, VectorizationBase.Unroll{2,1,W,1,W}((1,1)))
      end
      @test A == B'
      W >>= 1
    end
    W = 2W64
    while W > 1
      A = rand(Float32,W,W); B = similar(A);
      GC.@preserve A B begin
        vut = @inferred(vload(stridedpointer(A), VectorizationBase.Unroll{2,1,W,1,W}((1,1))))
        vu = @inferred(VectorizationBase.transpose_vecunroll(vut))
        @test vu === @inferred(vload(stridedpointer(A'), VectorizationBase.Unroll{2,1,W,1,W}((1,1))))
        @test vu === @inferred(vload(stridedpointer(A), VectorizationBase.Unroll{1,1,W,2,W}((1,1))))
        vstore!(stridedpointer(B), vu, VectorizationBase.Unroll{2,1,W,1,W}((1,1)))
      end
      @test A == B'
      W >>= 1
    end
  end
  
  println("Unary Functions")
  @time @testset "Unary Functions" begin
    for T ∈ (Float32,Float64)
      for f ∈ [floatmin,floatmax,typemin,typemax]
        @test f(Vec{Int(pick_vector_width(T)),T}) === Vec(ntuple(_ -> f(T), pick_vector_width(T))...)
      end
      v = let W = VectorizationBase.pick_vector_width(T)
        VectorizationBase.VecUnroll((
          Vec(ntuple(_ -> (randn(T)), W)...),
          Vec(ntuple(_ -> (randn(T)), W)...),
          Vec(ntuple(_ -> (randn(T)), W)...)
        ))
      end
      x = tovector(v);
      for f ∈ [
        -, abs, inv, floor, ceil, trunc, round, VectorizationBase.relu, abs2,
        Base.FastMath.abs2_fast, Base.FastMath.sub_fast, sign
        ]
        # @show T, f
        @test tovector(@inferred(f(v))) == map(f, x)
      end
      # test fallbacks
      for (vf,bf) ∈ [(VectorizationBase.vinv,inv),(VectorizationBase.vabs,abs),(VectorizationBase.vround,round),(VectorizationBase.vsub,-),(VectorizationBase.vsub_fast,Base.FastMath.sub_fast)]
        for i ∈ -5:5
          @test vf(i) == bf(i)
        end
        for i ∈ -3.0:0.1:3.0
          @test vf(i) == bf(i)
        end
      end
      vxabs = abs(v * 1000);
      vxabsvec = tovector(vxabs);
      @test tovector(exponent(vxabs)) == exponent.(vxabsvec)
      @test tovector(significand(vxabs)) == significand.(vxabsvec)
      @test tovector(exponent(inv(vxabs))) == exponent.(inv.(vxabsvec))
      @test tovector(significand(inv(vxabs))) == significand.(inv.(vxabsvec))
      @test v ^ 2 === v ^ StaticInt(2) === v*v
      @test v ^ 3 === v ^ StaticInt(3) === (v*v)*v
      @test v ^ 4 === v ^ StaticInt(4) === abs2(abs2(v))
      @test v ^ 5 === v ^ StaticInt(5) === abs2(abs2(v))*v
      @test v ^ 6 === v ^ StaticInt(6) === abs2(abs2(v))*abs2(v)
      # Don't require exact, but `eps(T)` seems like a reasonable `rtol`, at least on AVX512 systems:
      # function relapprox(x::AbstractVector{T},y) where {T}
      #     t = max(norm(x),norm(y)) * eps(T)
      #     n = norm(x .- y)
      #     n / t
      # end
      # function randapprox(::Type{T}) where {T}
      #     x = Vec(ntuple(_ -> 10randn(T), VectorizationBase.pick_vector_width(T))...)
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
      let rtol = eps(T) * (Bool(VectorizationBase.has_feature(Val(:x86_64_avx512f))) ? 1 : 4) # more accuracte
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
      invtol = Bool(VectorizationBase.has_feature(Val(:x86_64_avx512f))) ? 2^-14 : 1.5*2^-12 # moreaccurate with AVX512
      @test isapprox(tovector(@inferred(VectorizationBase.inv_approx(v))), map(VectorizationBase.inv_approx, x), rtol = invtol)
    end

    int = Bool(VectorizationBase.has_feature(Val(:x86_64_avx512dq))) ? Int : Int32
    int2 = Bool(VectorizationBase.has_feature(Val(:x86_64_avx2))) ? Int : Int32
    vi = VectorizationBase.VecUnroll((
      Vec(ntuple(_ -> rand(int), Val(W64))...),
      Vec(ntuple(_ -> rand(int), Val(W64))...),
      Vec(ntuple(_ -> rand(int), Val(W64))...)
    )) % int2
    xi = tovector(vi)
    for f ∈ [-, abs, inv, floor, ceil, trunc, round, sqrt ∘ abs, sign]
      @test tovector(@inferred(f(vi))) == map(f, xi)
    end
    let rtol = eps(Float64) * (Bool(VectorizationBase.has_feature(Val(:x86_64_avx512f))) ? 1 : 4) # more accuracte
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
  println("Binary Functions")
  @time @testset "Binary Functions" begin
    # TODO: finish getting these tests to pass
    # for I1 ∈ (Int32,Int64,UInt32,UInt64), I2 ∈ (Int32,Int64,UInt32,UInt64)
    for (vf,bf,testfloat) ∈ [(VectorizationBase.vadd,+,true),(VectorizationBase.vadd_fast,Base.FastMath.add_fast,true),(VectorizationBase.vadd_nsw,+,false),#(VectorizationBase.vadd_nuw,+,false),(VectorizationBase.vadd_nw,+,false),
                             (VectorizationBase.vsub,-,true),(VectorizationBase.vsub_fast,Base.FastMath.sub_fast,true),(VectorizationBase.vsub_nsw,-,false),#(VectorizationBase.vsub_nuw,-,false),(VectorizationBase.vsub_nw,-,false),
                             (VectorizationBase.vmul,*,true),(VectorizationBase.vmul_fast,Base.FastMath.mul_fast,true),(VectorizationBase.vmul_nsw,*,false),#(VectorizationBase.vmul_nuw,*,false),(VectorizationBase.vmul_nw,*,false),
                             (VectorizationBase.vrem,%,false),(VectorizationBase.vrem_fast,%,false)]
      for i ∈ -10:10, j ∈ -6:6
        ((j == 0) && (bf === %)) && continue
        @test vf(i%Int8,j%Int8) == bf(i%Int8,j%Int8)
        @test vf(i%UInt8,j%UInt8) == bf(i%UInt8,j%UInt8)
        @test vf(i%Int16,j%Int16) == bf(i%Int16,j%Int16)
        @test vf(i%UInt16,j%UInt16) == bf(i%UInt16,j%UInt16)
        @test vf(i%Int32,j%Int32) == bf(i%Int32,j%Int32)
        @test vf(i%UInt32,j%UInt32) == bf(i%UInt32,j%UInt32)
        @test vf(i%Int64,j%Int64) == bf(i%Int64,j%Int64)
        @test vf(i%UInt64,j%UInt64) == bf(i%UInt64,j%UInt64)
        @test vf(i%Int128,j%Int128) == bf(i%Int128,j%Int128)
        @test vf(i%UInt128,j%UInt128) == bf(i%UInt128,j%UInt128)
      end
      if testfloat
        for i ∈ -1.5:0.39:1.8, j ∈ -3:0.09:3.0
          # `===` for `NaN` to pass
          @test vf(i,j) === bf(i,j)
          @test vf(Float32(i),Float32(j)) === bf(Float32(i),Float32(j))
        end
      end
    end
    for i ∈ -1.5:0.39:1.8, j ∈ -3:0.09:3.0
      for i ∈ -1.5:0.379:1.8, j ∈ -3:0.089:3.0
        @test VectorizationBase.vdiv(i,j) == VectorizationBase.vdiv_fast(i,j) == 1e2i ÷ 1e2j
        @test VectorizationBase.vdiv(Float32(i),Float32(j)) == VectorizationBase.vdiv_fast(Float32(i),Float32(j)) == Float32(1f2i) ÷ Float32(1f2j)
        vr64_ref = 1e-2*(1e2i % 1e2j)
        @test VectorizationBase.vrem(i,j) ≈ vr64_ref atol = 1e-16 rtol=1e-13
        @test VectorizationBase.vrem_fast(i,j) ≈ vr64_ref atol = 1e-16 rtol=1e-13
        vr32_ref = 1f-2*(Float32(1f2i) % Float32(1f2j))
        @test VectorizationBase.vrem(Float32(i),Float32(j)) ≈ vr32_ref atol=1f-7 rtol=2f-5
        @test VectorizationBase.vrem_fast(Float32(i),Float32(j)) ≈ vr32_ref atol=1f-7 rtol=2f-5
      end
    end
    let WI = Int(VectorizationBase.pick_vector_width(Int64))
      for I1 ∈ (Int32,Int64), I2 ∈ (Int32,Int64,UInt32)
        # TODO: No longer skip these either.
        sizeof(I1) > sizeof(I2) && continue
        vi1 = VectorizationBase.VecUnroll((
          Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(I1)), Val(WI)))
        ))
        srange = one(I2):(Bool(VectorizationBase.has_feature(Val(:x86_64_avx512dq))) ? I2(8sizeof(I1)-1) : I2(31))
        vi2 = VectorizationBase.VecUnroll((
          Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(WI))),
          Vec(ntuple(_ -> Core.VecElement(rand(srange)), Val(WI)))
        ))
        i = rand(srange); j = rand(I1);
        m1 = VectorizationBase.VecUnroll((MM{WI}(I1(7)), MM{WI}(I1(1)), MM{WI}(I1(13)), MM{WI}(I1(32%last(srange)))));
        m2 = VectorizationBase.VecUnroll((MM{WI,2}(I2(3)), MM{WI,2}(I2(8)), MM{WI,2}(I2(39%last(srange))), MM{WI,2}(I2(17))));
        @test typeof(m1 + I1(11)) === typeof(m1)
        @test typeof(m2 + I2(11)) === typeof(m2)
        xi1 = tovector(vi1); xi2 = tovector(vi2);
        xi3 =  mapreduce(tovector, vcat, VectorizationBase.data(m1));
        xi4 =  mapreduce(tovector, vcat, VectorizationBase.data(m2));
        I3 = promote_type(I1,I2);
        # I4 = sizeof(I1) < sizeof(I2) ? I1 : (sizeof(I1) > sizeof(I2) ? I2 : I3)
        for f ∈ [
          +, -, *, ÷, /, %, <<, >>, >>>, ⊻, &, |, fld, mod,
          VectorizationBase.rotate_left, VectorizationBase.rotate_right, copysign, maxi, mini, maxi_fast, mini_fast
          ]
          # for f ∈ [+, -, *, div, ÷, /, rem, %, <<, >>, >>>, ⊻, &, |, fld, mod, VectorizationBase.rotate_left, VectorizationBase.rotate_right, copysign, max, min]
          # @show f, I1, I2
          # if (!Bool(VectorizationBase.has_feature(Val(:x86_64_avx512dq)))) && (f === /) && sizeof(I1) === sizeof(I2) === 8
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
            # @show 12
            # check_within_limits(tovector(@inferred(f(j, m2))), trunc_int.(f.(size_trunc_int.(j, I1), size_trunc_int.(xi4, I1)), I1));
          end
        end
        @test tovector(@inferred(vi1 ^ i)) ≈ Float64.(xi1) .^ i
        @test @inferred(VectorizationBase.vall(@inferred(1 - MM{WI}(1)) == (1 - Vec(ntuple(identity, Val(WI))...)) ))
      end
      vf1 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> Core.VecElement(randn()), Val(WI))),
        Vec(ntuple(_ -> Core.VecElement(randn()), Val(WI)))
      ))
      vf2 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(WI)))
      @test vf2 * 1//2 === vf2 * 0.5 === vf2 / 2
      xf1 = tovector(vf1); xf2 = tovector(vf2); xf22 = vcat(xf2,xf2)
      a = randn();
      for f ∈ [+, -, *, /, %, max, min, copysign, rem, Base.FastMath.max_fast, Base.FastMath.min_fast, Base.FastMath.div_fast, Base.FastMath.rem_fast, Base.FastMath.hypot_fast]
        # @show f
        @test tovector(@inferred(f(vf1, vf2))) ≈ f.(xf1, xf22)
        @test tovector(@inferred(f(a, vf1))) ≈ f.(a, xf1)
        @test tovector(@inferred(f(a, vf2))) ≈ f.(a, xf2)
        @test tovector(@inferred(f(vf1, a))) ≈ f.(xf1, a)
        @test tovector(@inferred(f(vf2, a))) ≈ f.(xf2, a)
      end

      vi2 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(WI))),
        Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(WI))),
        Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(WI))),
        Vec(ntuple(_ -> Core.VecElement(rand(1:M-1)), Val(WI)))
      ))
      vones, vi2f, vtwos = promote(1.0, vi2, 2f0); # promotes a binary function, right? Even when used with three args?
      @test vones === VectorizationBase.VecUnroll((vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0),vbroadcast(Val(WI), 1.0)));
      @test vtwos === VectorizationBase.VecUnroll((vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0),vbroadcast(Val(WI), 2.0)));
      @test VectorizationBase.vall(VectorizationBase.collapse_and(vi2f == vi2))
      W32 = StaticInt(WI)*StaticInt(2)
      vf2 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> Core.VecElement(randn(Float32)), W32)),
        Vec(ntuple(_ -> Core.VecElement(randn(Float32)), W32))
      ))
      vones32, v2f32, vtwos32 = promote(1.0, vf2, 2f0); # promotes a binary function, right? Even when used with three args?
      if VectorizationBase.simd_integer_register_size() == VectorizationBase.register_size()
        @test vones32 === VectorizationBase.VecUnroll((vbroadcast(W32, 1f0),vbroadcast(W32, 1f0))) === VectorizationBase.VecUnroll((vbroadcast(W32, Float16(1)),vbroadcast(W32, Float16(1))))
        @test vtwos32 === VectorizationBase.VecUnroll((vbroadcast(W32, 2f0),vbroadcast(W32, 2f0))) === VectorizationBase.VecUnroll((vbroadcast(W32, Float16(2)),vbroadcast(W32, Float16(2))))
        @test vf2 === v2f32
      else
        @test vones32 === VectorizationBase.VecUnroll((vbroadcast(W32, 1.0),vbroadcast(W32, 1.0)))
        @test vtwos32 === VectorizationBase.VecUnroll((vbroadcast(W32, 2.0),vbroadcast(W32, 2.0)))
        @test convert(Float64, vf2) === v2f32
      end
      vtwosf16 = convert(Float16, vtwos32)
      @test vtwosf16 isa VectorizationBase.VecUnroll{1,Int(W32),Float16,Vec{Int(W32),Float16}}
      vtwosf32 = convert(Float32, vtwos32)
      @test vtwosf32 isa VectorizationBase.VecUnroll{1,Int(W32),Float32,Vec{Int(W32),Float32}}
      @test promote(vtwosf16,vtwosf16) === (vtwosf32,vtwosf32)
      @test vtwosf16 + vtwosf16 === vtwosf32 + vtwosf32
      i = rand(1:31)
      m1 = VectorizationBase.VecUnroll((MM{WI}(7), MM{WI}(1), MM{WI}(13), MM{WI}(18)))
      @test tovector(clamp(m1, 2:i)) == clamp.(tovector(m1), 2, i)
      @test tovector(mod(m1, 1:i)) == mod1.(tovector(m1), i)

      @test VectorizationBase.vdivrem.(1:30, 1:30') == divrem.(1:30, 1:30')
      @test VectorizationBase.vcld.(1:30, 1:30') == cld.(1:30, 1:30')
      @test VectorizationBase.vrem.(1:30, 1:30') == rem.(1:30, 1:30')

      @test gcd(Vec(42,64,0,-37), Vec(18,96,-38,0)) === Vec(6,32,38,37)
      @test lcm(Vec(24,16,42,0),Vec(18,12,18,17)) === Vec(72, 48, 126, 0)
    end
    if VectorizationBase.simd_integer_register_size() ≥ 16
      @test VecUnroll((Vec(ntuple(Int32,Val(4))...),Vec(ntuple(Int32 ∘ Base.Fix2(+,4), Val(4))...))) << Vec(0x01,0x02,0x03,0x04) === VecUnroll((Vec(map(Int32,(2,8,24,64))...), Vec(map(Int32,(10,24,56,128))...)))
    end

    @test VectorizationBase.vdiv_fast(VecUnroll((11,12,13,14)),3) === VecUnroll((11,12,13,14)) ÷ 3 === VecUnroll((3,4,4,4))
    @test VectorizationBase.vand(true,true) == true
    @test VectorizationBase.vand(false,false) == VectorizationBase.vand(false,true) == VectorizationBase.vand(true,false) == false
    @test VectorizationBase.vor(true,true) == VectorizationBase.vor(false,true) == VectorizationBase.vor(true,false) == true
    @test VectorizationBase.vor(false,false) == false
    @test VectorizationBase.vxor(false,true) == VectorizationBase.vxor(true,false) == true
    @test VectorizationBase.vxor(false,false) == VectorizationBase.vxor(true,true) == false
    
  end
  println("Ternary Functions")
  @time @testset "Ternary Functions" begin
    for T ∈ (Float32, Float64)
      v1, v2, v3, m = let W = @inferred(VectorizationBase.pick_vector_width(T))
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
        _W = Int(@inferred(VectorizationBase.pick_vector_width(T)))
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
      @test tovector(@inferred(VectorizationBase.vfmaddsub(v1,v2,v3))) ≈ muladd.(x1, x2, x3 .* ifelse.(iseven.(eachindex(x1)), 1,-1) )
      @test tovector(@inferred(VectorizationBase.vfmsubadd(v1,v2,v3))) ≈ muladd.(x1, x2, x3 .* ifelse.(iseven.(eachindex(x1)),-1, 1) )
    end
    let WI = Int(VectorizationBase.pick_vector_width(Int64))
      vi64 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> rand(Int64), Val(WI))...),
        Vec(ntuple(_ -> rand(Int64), Val(WI))...),
        Vec(ntuple(_ -> rand(Int64), Val(WI))...)
      ))
      vi32 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> rand(Int32), Val(WI))...),
        Vec(ntuple(_ -> rand(Int32), Val(WI))...),
        Vec(ntuple(_ -> rand(Int32), Val(WI))...)
      ))
      xi64 = tovector(vi64); xi32 = tovector(vi32);
      @test tovector(@inferred(VectorizationBase.ifelse(vi64 > vi32, vi64, vi32))) == ifelse.(xi64 .> xi32, xi64, xi32)
      @test tovector(@inferred(VectorizationBase.ifelse(vi64 < vi32, vi64, vi32))) == ifelse.(xi64 .< xi32, xi64, xi32)
      @test tovector(@inferred(VectorizationBase.ifelse(true, vi64, vi32))) == ifelse.(true, xi64, xi32)
      @test tovector(@inferred(VectorizationBase.ifelse(false, vi64, vi32))) == ifelse.(false, xi64, xi32)
      vu64_1 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      vu64_2 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      vu64_3 = VectorizationBase.VecUnroll((
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...),
        Vec(ntuple(_ -> rand(UInt64), Val(WI))...)
      ))
      xu1 = tovector(vu64_1); xu2 = tovector(vu64_2); xu3 = tovector(vu64_3);
      for f ∈ [clamp, muladd, VectorizationBase.ifmalo, VectorizationBase.ifmahi, VectorizationBase.vfmadd, VectorizationBase.vfnmadd, VectorizationBase.vfmsub, VectorizationBase.vfnmsub]
        @test tovector(@inferred(f(vu64_1,vu64_2,vu64_3))) == f.(xu1, xu2, xu3)
      end
    end
  end
  println("Special functions")
  @time @testset "Special functions" begin
    if VERSION ≥ v"1.6.0-DEV.674" && Bool(VectorizationBase.has_feature(Val(Symbol("x86_64_sse4.1"))))
      erfs = [0.1124629160182849, 0.22270258921047847, 0.3286267594591274, 0.42839235504666845, 0.5204998778130465, 0.6038560908479259, 0.6778011938374184, 0.7421009647076605, 0.7969082124228322, 0.8427007929497149, 0.8802050695740817, 0.9103139782296353, 0.9340079449406524, 0.9522851197626487, 0.9661051464753108, 0.976348383344644, 0.9837904585907745, 0.9890905016357308, 0.9927904292352575, 0.9953222650189527, 0.997020533343667, 0.9981371537020182, 0.9988568234026434, 0.999311486103355, 0.999593047982555, 0.9997639655834707, 0.9998656672600594, 0.9999249868053346, 0.9999589021219005, 0.9999779095030014, 0.9999883513426328, 0.9999939742388483]
      
      if Bool(VectorizationBase.has_feature(Val(:x86_64_avx512f)))
        v = VectorizationBase.verf(Vec{8, Float64}(0.1:0.1:0.8...,))
        @test [v(i) for i in 1:8] ≈ erfs[1:8]
        v = VectorizationBase.verf(Vec{16, Float32}(0.1:0.1:1.6...,))
        @test [v(i) for i in 1:16] ≈ erfs[1:16]
      end
      if Bool(VectorizationBase.has_feature(Val(:x86_64_avx)))
        v = VectorizationBase.verf(Vec{4, Float64}(0.1:0.1:0.4...,))
        @test [v(i) for i in 1:4] ≈ erfs[1:4]
        v = VectorizationBase.verf(Vec{8, Float32}(0.1:0.1:0.8...,))
        @test [v(i) for i in 1:8] ≈ erfs[1:8]
      end
      if Bool(VectorizationBase.has_feature(Val(Symbol("x86_64_sse4.1"))))
        v = VectorizationBase.verf(Vec{2, Float64}(0.1:0.1:0.2...,))
        @test [v(i) for i in 1:2] ≈ erfs[1:2]
      end
    end
  end
  println("Non-broadcasting operations")
  @time @testset "Non-broadcasting operations" begin
    v1 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))); vu1 = VectorizationBase.VecUnroll((v1, Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))));
    v2 = Vec(ntuple(_ -> Core.VecElement(rand(-100:100)), Val(W64))); vu2 = VectorizationBase.VecUnroll((v2, Vec(ntuple(_ -> Core.VecElement(rand(-100:100)), Val(W64)))));
    @test @inferred(VectorizationBase.vsum(2.3, v1)) ≈ @inferred(VectorizationBase.vsum(v1)) + 2.3 ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(v1, 2.3))) ≈ @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(2.3, v1)))
    @test @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(vu1))) + 2.3 ≈ @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(VectorizationBase.addscalar(vu1, 2.3)))) ≈ @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(VectorizationBase.addscalar(2.3, vu1))))
    @test @inferred(VectorizationBase.vsum(v2)) + 3 == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(v2, 3))) == @inferred(VectorizationBase.vsum(VectorizationBase.addscalar(3, v2)))
    @test @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(vu2))) + 3 == @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(VectorizationBase.addscalar(vu2, 3)))) == @inferred(VectorizationBase.vsum(VectorizationBase.collapse_add(VectorizationBase.addscalar(3, vu2))))
    @test @inferred(VectorizationBase.vprod(v1)) * 2.3 ≈ @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(v1, 2.3))) ≈ @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(2.3, v1)))
    @test @inferred(VectorizationBase.vprod(v2)) * 3 == @inferred(VectorizationBase.vprod(VectorizationBase.mulscalar(3, v2)))
    @test @inferred(VectorizationBase.vall(v1 + v2 == VectorizationBase.addscalar(v1, v2)))
    @test 4.0 == @inferred(VectorizationBase.addscalar(2.0, 2.0))

    
    v3 = Vec(ntuple(Base.Fix2(-,1), W64)...)
    vu3 = VectorizationBase.VecUnroll((v3, v3 - 1))
    v4 = Vec(ntuple(Base.Fix2(-,1.0), W64)...)
    v5 = Vec(ntuple(Base.Fix2(-,1f0), W32)...)
    @test @inferred(VectorizationBase.vmaximum(v3)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3, W64-2)))
    @test @inferred(VectorizationBase.vmaximum(v3 % UInt)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3 % UInt, (W64-2) % UInt)))
    @test @inferred(VectorizationBase.vmaximum(v4)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v4, prevfloat(W64-1.0))))
    @test @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v4, nextfloat(W64-1.0)))) == nextfloat(W64-1.0)
    @test @inferred(VectorizationBase.vmaximum(v5)) === @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v5, prevfloat(W32-1f0)))) === VectorizationBase.vmaximum(VectorizationBase.maxscalar(prevfloat(W32 - 1f0), v5))
    @test @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(v5, nextfloat(W32-1f0)))) == @inferred(VectorizationBase.vmaximum(VectorizationBase.maxscalar(nextfloat(W32-1f0), v5))) == nextfloat(W32-1f0)
    
    @test VectorizationBase.maxscalar(v3, 2)(1) == 2
    @test (VectorizationBase.maxscalar(v3, 2) ≠ v3) === Mask{W64}(0x01)
    @test VectorizationBase.maxscalar(v3, -1) === v3
    @test VectorizationBase.vmaximum(VectorizationBase.maxscalar(v3 % UInt, -1 % UInt)) === -1 % UInt
    @test VectorizationBase.maxscalar(v4, 1e-16) === VectorizationBase.insertelement(v4, 1e-16, 0)
    @test VectorizationBase.maxscalar(v4, -1e-16) === v4
    @test VectorizationBase.vmaximum(VectorizationBase.collapse_max(vu3)) == W64-1
    @test VectorizationBase.vmaximum(VectorizationBase.collapse_max(VectorizationBase.maxscalar(vu3,W64-2))) == W64-1
    @test VectorizationBase.vmaximum(VectorizationBase.collapse_max(VectorizationBase.maxscalar(vu3,W64))) == W64
    @test VectorizationBase.vminimum(VectorizationBase.collapse_min(vu3)) == -1
    @test VectorizationBase.vminimum(VectorizationBase.collapse_min(VectorizationBase.minscalar(vu3,0))) == -1
    @test VectorizationBase.vminimum(VectorizationBase.collapse_min(VectorizationBase.minscalar(vu3,-2))) == VectorizationBase.vminimum(VectorizationBase.collapse_min(VectorizationBase.minscalar(-2,vu3))) == -2
  end
  println("broadcasting")
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
  println("CartesianVIndex")
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
  println("Promotion")
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
    @test @inferred(VectorizationBase.vall(VectorizationBase.collapse_and(vi2f == vi2)))
    vf2 = VectorizationBase.VecUnroll((
      Vec(ntuple(_ -> Core.VecElement(randn(Float32)), StaticInt(W32))),
      Vec(ntuple(_ -> Core.VecElement(randn(Float32)), StaticInt(W32)))
    ))
    vones32, v2f32, vtwos32 = @inferred(promote(1.0, vf2, 2f0)); # promotes a binary function, right? Even when used with three args?
    @test vones32 === VectorizationBase.VecUnroll((vbroadcast(StaticInt(W32), 1f0),vbroadcast(StaticInt(W32), 1f0)))
    @test vtwos32 === VectorizationBase.VecUnroll((vbroadcast(StaticInt(W32), 2f0),vbroadcast(StaticInt(W32), 2f0)))
    @test vf2 === v2f32
    @test isequal(tovector(@inferred(bswap(vf2))), map(bswap, tovector(vf2)))

    vm = if Bool(VectorizationBase.has_feature(Val(:x86_64_avx512dq)))
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

    vx = convert(Vec{16,Int64}, 1)
    @test typeof(vx) === typeof(zero(vx)) === Vec{16,Int64}

    vxf32 = Vec(ntuple(_ -> randn(Float32), VectorizationBase.pick_vector_width(Float32))...)
    xf32, yf32 = promote(vxf32, 1.0)
    @test xf32 === vxf32
    @test yf32 === vbroadcast(VectorizationBase.pick_vector_width(Float32), 1f0)
    vxi32 = Vec(ntuple(_ -> rand(Int32), VectorizationBase.pick_vector_width(Int32))...)
    xi32, yi32 = promote(vxi32, one(Int64))
    @test xi32 === vxi32
    @test yi32 === vbroadcast(VectorizationBase.pick_vector_width(Int32), one(Int32))
    @test ntoh(vxi32) === Vec(map(ntoh, Tuple(vxi32))...)

    @test VecUnroll((1.0,2.0,3.0)) * VecUnroll((1f0,2f0,3f0)) === VecUnroll((1.0, 4.0, 9.0))
  end
  println("Lazymul")
  @time @testset "Lazymul" begin
    # partially covered in memory
    for i ∈ (-5, -1, 0, 1, 4, 8), j ∈ (-5, -1, 0, 1, 4, 8)
      @test @inferred(VectorizationBase.lazymul(StaticInt(i), StaticInt(j))) === StaticInt(i*j)
    end
    fi = VectorizationBase.LazyMulAdd{8,0}(MM{8}(StaticInt(16)))
    si = VectorizationBase.LazyMulAdd{2}(240)
    @test @inferred(VectorizationBase.vadd_nsw(fi, si)) === VectorizationBase.LazyMulAdd{2,128}(MM{8,4}(240))
  end
  # TODO: Put something here.
  # @time @testset "Arch Functions" begin
  #     @test VectorizationBase.dynamic_register_size() == @inferred(VectorizationBase.register_size()) == @inferred(VectorizationBase.register_size())
  #     @test VectorizationBase.dynamic_integer_register_size() == @inferred(VectorizationBase.simd_integer_register_size()) == @inferred(VectorizationBase.ssimd_integer_register_size())
  #     @test VectorizationBase.dynamic_register_count() == @inferred(VectorizationBase.register_count()) == @inferred(VectorizationBase.sregister_count())
  #     @test VectorizationBase.dynamic_fma_fast() == VectorizationBase.fma_fast()
  #     @test VectorizationBase.dynamic_has_opmask_registers() == VectorizationBase.has_opmask_registers()
  # end
  println("Static Zero and One")
  @time @testset "Static Zero and One" begin
    vx = randnvec(W64)
    vu = VectorizationBase.VecUnroll((vx,randnvec(W64)))
    vm = MM{16}(24);
    for f ∈ [+,Base.FastMath.add_fast]
      @test f(vx, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vx) === vx
      @test f(vu, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vu) === vu
      @test f(vm, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vm) === vm
    end
    for f ∈ [-,Base.FastMath.sub_fast]
      @test f(vx, VectorizationBase.Zero()) ===  vx
      @test f(VectorizationBase.Zero(), vx) === -vx
      @test f(vu, VectorizationBase.Zero()) ===  vu
      @test f(VectorizationBase.Zero(), vu) === -vu
      @test f(vm, VectorizationBase.Zero()) ===  vm
      @test f(VectorizationBase.Zero(), vm) === -vm
    end
    for f ∈ [*,Base.FastMath.mul_fast]
      @test f(vx, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vx) === VectorizationBase.Zero()
      @test f(vu, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vu) === VectorizationBase.Zero()
      @test f(vm, VectorizationBase.Zero()) === f(VectorizationBase.Zero(), vm) === VectorizationBase.Zero()
      @test f(vx, VectorizationBase.One()) === f(VectorizationBase.One(), vx) === vx
      @test f(vu, VectorizationBase.One()) === f(VectorizationBase.One(), vu) === vu
      @test f(vm, VectorizationBase.One()) === f(VectorizationBase.One(), vm) === vm
    end
    vnan = NaN * vx
    for f ∈ [fma, muladd, VectorizationBase.vfma_fast, VectorizationBase.vmuladd_fast]
      @test f(vnan, VectorizationBase.Zero(), vx) === vx
      @test f(VectorizationBase.Zero(), vnan, vx) === vx
    end
  end

  @inline function vlog(x1::VectorizationBase.AbstractSIMD{W,Float64}) where {W} # Testing if an assorted mix of operations
    x2 = reinterpret(UInt64, x1)
    x3 = x2 >>> 0x0000000000000020
    greater_than_zero = x1 > zero(x1)
    alternative = VectorizationBase.ifelse(x1 == zero(x1), -Inf, NaN)
    isinf = x1 == Inf
    x5 = x3 + 0x0000000000095f62
    x6 = x5 >>> 0x0000000000000014
    x7 = x6 - 0x00000000000003ff
    x8 = convert(Float64, x7 % Int)
    x9 = x5 << 0x0000000000000020
    x10 = x9 & 0x000fffff00000000
    x11 = x10 + 0x3fe6a09e00000000
    x12 = x2 & 0x00000000ffffffff
    x13 = x11 | x12
    x14 = reinterpret(Float64, x13)
    
    x15 = x14 - 1.0
    x16 = x15 * x15
    x17 = 0.5 * x16
    x18 = x14 + 1.0
    x19 = x15 / x18
    x20 = x19 * x19
    x21 = x20 * x20
    x22 = vfmadd(x21, 0.15313837699209373, 0.22222198432149784)
    x23 = vfmadd(x21, x22, 0.3999999999940942)
    x24 = x23 * x21
    x25 = vfmadd(x21, 0.14798198605116586, 0.1818357216161805)
    x26 = vfmadd(x21, x25, 0.2857142874366239)
    x27 = vfmadd(x21, x26, 0.6666666666666735)
    x28 = x27 * x20
    x29 = x24 + x17
    x30 = x29 + x28
    x31 = x8 * 1.9082149292705877e-10
    x32 = vfmadd(x19, x30, x31)
    x33 = x15 - x17
    x34 = x33 + x32
    x35 = vfmadd(x8, 0.6931471803691238, x34)
    x36 = VectorizationBase.ifelse(greater_than_zero, x35, alternative)
    VectorizationBase.ifelse(isinf, Inf, x36)
  end

  println("Defining log")
  @time @testset "Defining log." begin
    vx = Vec(ntuple(_ -> rand(), VectorizationBase.StaticInt(3) * VectorizationBase.pick_vector_width(Float64))...)
    check_within_limits(tovector(@inferred(vlog(vx))), log.(tovector(vx)))
  end

  println("Saturated add")
  @time @testset "Saturated add" begin
    @test VectorizationBase.saturated_add(0xf0, 0xf0) === 0xff
    @test VectorizationBase.saturated_add(2_000_000_000 % Int32, 1_000_000_000 % Int32) === typemax(Int32)
    v = Vec(ntuple(_ -> rand(typemax(UInt)>>1+one(UInt):typemax(UInt)), VectorizationBase.pick_vector_width(UInt))...)
    @test VectorizationBase.saturated_add(v, v) === vbroadcast(VectorizationBase.pick_vector_width(UInt), typemax(UInt))
  end

  println("Special Functions")
  using SpecialFunctions
  @time @testset "Special Functions" begin
    for T ∈ [Float32,Float64]
      min_non_denormal = nextfloat(abs(reinterpret(T, typemax(Base.uinttype(T)) & (~Base.exponent_mask(T)))))
      l2mnd = log2(min_non_denormal)
      xx = collect(range(T(0.8)*l2mnd, T(0.8)*abs(l2mnd), length = 2^21));
      test_acc(exp2, exp2, T, xx, 3)

      lemnd = log(min_non_denormal)
      xx .= range(T(0.8)*lemnd, T(0.8)*abs(lemnd), length = 2^21);
      test_acc(exp, exp, T, xx, 3)
      
      l10mnd = log10(min_non_denormal)
      xx .= range(T(0.8)*l10mnd, T(0.8)*abs(l10mnd), length = 2^21);
      test_acc(exp10, exp10, T, xx, 3)

      if T === Float32
        xx .= range(-4f0, 4f0, length = 2^21);
        erftol = 3
      else
        xx .= range(-6.0, 6.0, length = 2^21);
        erftol = 7
      end;
      test_acc(VectorizationBase.verf, erf, T, xx, erftol)
      # xx .= exp2.(range(T(0.8)*l2mnd, T(0.8)*abs(l2mnd), length = 2^20));
      # test_acc(VectorizationBase.vlog2, log2, T, xx, 7)
    end
    @test exp(VecUnroll((1.1,2.3))) === VecUnroll((3.0041660239464334, 9.97418245481472))
    @test exp(VecUnroll((1,2))) === VecUnroll((2.7182818284590455,7.3890560989306495))
  end

  # fix the stackoverflow error in `vmax_fast`, `vmax`, `vmin` and `vmin_fast` for floating types
  @time @testset "fix stackoverflow for `vmax_fast` et al." begin
    @test VectorizationBase.vmax_fast(1.0,3.0) === 3.0
    @test VectorizationBase.vmax_fast(1,3) === 3
    @test VectorizationBase.vmin_fast(1,3) === 1
    @test VectorizationBase.vmin_fast(1.0,3.0) === 1.0
    @test VectorizationBase.vmax(1.0,3.0) === 3.0
    @test VectorizationBase.vmax(1,3) === 3
    @test VectorizationBase.vmin(1,3) === 1
    @test VectorizationBase.vmin(1.0,3.0) === 1.0
  end

  @time @testset "Generic strided pointer" begin
    A = rand(ComplexF64, 3, 4);
    x = ["hi" "howdy"; "greetings" "hello"];
    GC.@preserve A x begin
      @test A[2,3] === vload(stridedpointer(A), (2,3))
      c = 123.0 - 456.0im
      vstore!(stridedpointer(A), c, (3,2))
      @test A[3,2] == c
      @test x[1] === vload(stridedpointer(x), (0,))
      @test x[3] === vload(stridedpointer(x), (2,))
      w = "welcome!"
      vstore!(stridedpointer(x), w, (1,))
      @test w === x[2]
      h = "hallo"
      vstore!(stridedpointer(x), h, (2,2))
      @test x[2,2] === h
      vload(stridedpointer(x), (1,2)) === x[1,2]
    end
  end
  @testset "NullStep" begin
    A = rand(4,5);
    GC.@preserve A begin
      @test @inferred(vload(VectorizationBase.gesp(VectorizationBase.stridedpointer(A), (VectorizationBase.NullStep(),VectorizationBase.NullStep())), (1,2))) == A[1,2]
      @test @inferred(vload(VectorizationBase.gesp(VectorizationBase.stridedpointer(A), (StaticInt(0),VectorizationBase.NullStep())), (2,3))) == A[2,3]
      @test @inferred(vload(VectorizationBase.gesp(VectorizationBase.stridedpointer(A), (VectorizationBase.NullStep(),StaticInt(0))), (3,4))) == A[3,4]
    end
    B = A .> 0.5;
    spb = stridedpointer(B)
    @test VectorizationBase.gesp(spb, (3,)) === VectorizationBase.gesp(spb, (3,0))
  end
  # end
end # @testset VectorizationBase

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
