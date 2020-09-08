using VectorizationBase, OffsetArrays
using VectorizationBase: data
using Test

const W64 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float64)
const W32 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float32)
const VE = Core.VecElement

function tovector(u::VectorizationBase.VecUnroll{_N,W,T}) where {_N,W,T}
    N = _N + 1; i = 0
    x = Vector{T}(undef, N * W)
    for n ∈ 1:N
        v = u.data[n]
        for w ∈ 1:W
            x[(i += 1)] = VectorizationBase.getelement(v, w)
        end
    end
    x
end
tovector(v::VectorizationBase.AbstractSIMDVector{W}) where {W} = [VectorizationBase.getelement(v,w) for w ∈ 1:W]
tovector(v::VectorizationBase.LazyMulAdd) = tovector(convert(Vec, v))
tovector(x) = x
A = randn(13, 17); L = length(A); M, N = size(A);

@testset "VectorizationBase.jl" begin
    # Write your own tests here.
    @test isempty(detect_unbound_args(VectorizationBase))

    W = VectorizationBase.pick_vector_width(Float64)
    @test @inferred(VectorizationBase.pick_integer(Val(W))) == (VectorizationBase.AVX512DQ ? Int64 : Int32)

    
    @test first(A) === A[1]
    @testset "Struct-Wrapped Vec" begin
        @test data(zero(Vec{4,Float64})) === (VE(0.0),VE(0.0),VE(0.0),VE(0.0)) === data(Vec{4,Float64}(0.0))
        @test data(one(Vec{4,Float64})) === (VE(1.0),VE(1.0),VE(1.0),VE(1.0)) === data(Vec{4,Float64}(1.0)) === data(data(Vec{4,Float64}(1.0)))
        v = Vec((VE(1.0),VE(2.0),VE(3.0),VE(4.0)))
        @test v === Vec{4,Float64}(1, 2, 3, 4) === conj(v) === v'
        @test length(v) == 4 == first(size(v))
        @test eltype(v) == Float64
        for i in 1:4
            @test i == VectorizationBase.getelement(v, i)
            # @test i === Vec{4,Int}(v)[i] # should use fptosi (ie, vconvert defined in SIMDPirates).
        end
        @test zero(v) === zero(typeof(v))
        @test one(v) === one(typeof(v))
        # @test Vec{W32,Float32}(one(Vec{W32,Float64})) === Vec(one(Vec{W32,Float32})) === one(Vec{W32,Float32}) # conversions should be tested in SIMDPirates
        @test Vec{1,Int}(1) === 1
    end

    @testset "alignment.jl" begin

        @test all(i -> VectorizationBase.align(i) == VectorizationBase.REGISTER_SIZE, 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i) == 2VectorizationBase.REGISTER_SIZE, 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i) == 10VectorizationBase.REGISTER_SIZE, (1:VectorizationBase.REGISTER_SIZE) .+ 9VectorizationBase.REGISTER_SIZE)

        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},   VectorizationBase.REGISTER_SIZE), 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},  2VectorizationBase.REGISTER_SIZE), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid}, 20VectorizationBase.REGISTER_SIZE), (1:VectorizationBase.REGISTER_SIZE) .+ 19VectorizationBase.REGISTER_SIZE)

        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == W32*cld(i,W32), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == W64*cld(i,W64), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

        @test reinterpret(Int, VectorizationBase.align(pointer(A))) % VectorizationBase.REGISTER_SIZE === 0

        @test all(i -> VectorizationBase.aligntrunc(i) == 0, 0:VectorizationBase.REGISTER_SIZE-1)
        @test all(i -> VectorizationBase.aligntrunc(i) == VectorizationBase.REGISTER_SIZE, VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE-1)
        @test all(i -> VectorizationBase.aligntrunc(i) == 9VectorizationBase.REGISTER_SIZE, (0:VectorizationBase.REGISTER_SIZE-1) .+ 9VectorizationBase.REGISTER_SIZE)

        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.aligntrunc(i,W32) == VectorizationBase.aligntrunc(i,Float32) == VectorizationBase.aligntrunc(i,Int32) == W32*div(i,W32), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), 1:VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
        @test all(i -> VectorizationBase.aligntrunc(i,W64) == VectorizationBase.aligntrunc(i,Float64) == VectorizationBase.aligntrunc(i,Int64) == W64*div(i,W64), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

        a = Vector{Float64}(undef, 0)
        ptr = pointer(a)
        @test UInt(VectorizationBase.align(ptr, 1 << 12)) % (1 << 12) == 0
    end

    @testset "masks.jl" begin
        # @test Mask{8,UInt8}(0x0f) === @inferred Mask(0x0f)
        # @test Mask{16,UInt16}(0x0f0f) === @inferred Mask(0x0f0f)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 0)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 8)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), 16)
        @test Mask{8,UInt8}(0xff) === mask(Val(8), VectorizationBase.Static(0))
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 0)
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 16)
        @test Mask{16,UInt16}(0xffff) === mask(Val(16), 32)
        @test all(w -> VectorizationBase.mask_type(w) === UInt8, 1:8)
        @test all(w -> VectorizationBase.mask_type(w) === UInt16, 9:16)
        @test all(w -> VectorizationBase.mask_type(w) === UInt32, 17:32)
        @test all(w -> VectorizationBase.mask_type(w) === UInt64, 33:64)
        @test all(w -> VectorizationBase.mask_type(w) === UInt128, 65:128)
        if VectorizationBase.REGISTER_SIZE == 64 # avx512
            # @test VectorizationBase.mask_type(Float16) === UInt32
            @test VectorizationBase.mask_type(Float32) === UInt16
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xffffffff # 32
            @test data(VectorizationBase.max_mask(Float32)) === 0xffff     # 16
            @test data(VectorizationBase.max_mask(Float64)) === 0xff       # 8
        elseif VectorizationBase.REGISTER_SIZE == 32 # avx or avx2
            # @test VectorizationBase.mask_type(Float16) === UInt16
            @test VectorizationBase.mask_type(Float32) === UInt8
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xffff     # 16
            @test data(VectorizationBase.max_mask(Float32)) === 0xff       # 8
            @test data(VectorizationBase.max_mask(Float64)) === 0x0f       # 4
        elseif VectorizationBase.REGISTER_SIZE == 16 # sse
            # @test VectorizationBase.mask_type(Float16) === UInt8
            @test VectorizationBase.mask_type(Float32) === UInt8
            @test VectorizationBase.mask_type(Float64) === UInt8
            # @test VectorizationBase.max_mask(Float16) === 0xff       # 8
            @test data(VectorizationBase.max_mask(Float32)) === 0x0f       # 4
            @test data(VectorizationBase.max_mask(Float64)) === 0x03       # 2
        end
        @test all(w -> bitstring(VectorizationBase.mask(Val( 8), w)) == reduce(*, ( 8 - i < w ? "1" : "0" for i in 1:8 )), 1:8 )
        @test all(w -> bitstring(VectorizationBase.mask(Val(16), w)) == reduce(*, (16 - i < w ? "1" : "0" for i in 1:16)), 1:16)
        @test all(w -> VectorizationBase.mask(Float64, w) === VectorizationBase.mask(VectorizationBase.pick_vector_width_val(Float64), w), 1:W64)

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
    end

    # @testset "number_vectors.jl" begin
    # # eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :(size(A)), 8)) # doesn't work?
    # @test VectorizationBase.length_loads(A, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13*17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13*17, 8)) == divrem(length(A), 8)
    # @test VectorizationBase.size_loads(A,1, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13   )()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13   , 8)) == divrem(size(A,1), 8)
    # @test VectorizationBase.size_loads(A,2, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() ->    17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__,    17, 8)) == divrem(size(A,2), 8)
    # end

    
    @testset "vector_width.jl" begin
        @test all(VectorizationBase.ispow2, 0:1)
        @test all(i -> !any(VectorizationBase.ispow2, 1+(1 << (i-1)):(1 << i)-1 ) && VectorizationBase.ispow2(1 << i), 2:9)
        @test all(i ->  VectorizationBase.intlog2(1 << i) == i, 0:(Int == Int64 ? 53 : 30))
        FTypes = (Float32, Float64)
        Wv = ntuple(i -> VectorizationBase.REGISTER_SIZE >> (i+1), Val(2))
        for (T, N) in zip(FTypes, Wv)
            W = VectorizationBase.pick_vector_width(T)
            @test Vec{W,T} == VectorizationBase.pick_vector(Val(W), T) == VectorizationBase.pick_vector(T)
            @test W == VectorizationBase.pick_vector_width(Val(W), T)
            @test Val(W) === VectorizationBase.pick_vector_width_val(Val(W), T) == VectorizationBase.pick_vector_width_val(T)
            while true
                W >>= 1
                W == 0 && break
                W2, Wshift2 = VectorizationBase.pick_vector_width_shift(W, T)
                @test W2 == 1 << Wshift2 == VectorizationBase.pick_vector_width(W, T) == VectorizationBase.pick_vector_width(Val(W),T)  == W
                @test Val(W) === VectorizationBase.pick_vector_width_val(Val(W), T)
                for n in W+1:2W
                    W3, Wshift3 = VectorizationBase.pick_vector_width_shift(n, T)
                    @test W2 << 1 == W3 == 1 << (Wshift2+1) == 1 << Wshift3 == VectorizationBase.pick_vector_width(n, T) == VectorizationBase.pick_vector_width(Val(n),T) == W << 1
                    @test VectorizationBase.pick_vector(Val(W), T) == VectorizationBase.pick_vector(W, T) == Vec{W,T}
                end
            end
        end

        @test VectorizationBase.nextpow2(0) == 1
        @test all(i -> VectorizationBase.nextpow2(i) == i, 1:2)
        for j in 1:10
            l, u = (1<<j)+1, 1<<(j+1)
            @test all(i -> VectorizationBase.nextpow2(i) == u, l:u)
        end

    end

    @testset "StridedPointer" begin
        dims = (41,42,43) .* 3;
        # dims = (41,42,43);
        A = reshape(collect(Float64(0):Float64(prod(dims)-1)), dims);
        P = PermutedDimsArray(A, (3,1,2));
        O = OffsetArray(P, (-4, -2, -3));

        indices = [
            2, MM{W64}(1), Vec(ntuple(i -> Core.VecElement(2i + 1), Val(W64))),
            VectorizationBase.LazyMulAdd{2,-1}(MM{W64}(3)), VectorizationBase.LazyMulAdd{2,-2}(Vec(ntuple(i -> Core.VecElement(2i + 1), Val(W64))))
        ]
        for i ∈ indices, j ∈ indices, k ∈ indices, B ∈ [A, P, O]
            # @show typeof(B), i, j, k
            x = getindex.(Ref(B), tovector(i), tovector(j), tovector(k))
            GC.@preserve B begin
                v = vload(stridedpointer(B), (i, j, k))
            end
            @test x == tovector(v)
            m = Mask{W64}(rand(UInt8))
            x .*= tovector(m)
            GC.@preserve B begin
                v = vload(stridedpointer(B), (i, j, k), m)
            end
            @test x == tovector(v)
        end
        for AU ∈ 1:3, AV ∈ 1:3, B ∈ [A, P, O]
            i, j, k = 2, 3, 4
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
            GC.@preserve B begin
                vu = vload(stridedpointer(B), VectorizationBase.Unroll{AU,1,3,AV,W64,zero(UInt)}((i, j, k)))
            end
            @test x1 == tovector(vu.data[1])
            @test x2 == tovector(vu.data[2])
            @test x3 == tovector(vu.data[3])
        end
    end

    @testset "Unary Functions" begin
        v = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        ))
        x = tovector(v)
        for f ∈ [-, abs, floor, ceil, trunc, round, sqrt ∘ abs]
            @test tovector(f(v)) == map(f, x)
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
    @testset "Binary Functions" begin
        vi1 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(rand(Int)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(Int)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(Int)), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(Int)), Val(W64)))
        ))
        vi2 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(rand(1:8sizeof(Int))), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:8sizeof(Int))), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:8sizeof(Int))), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(rand(1:8sizeof(Int))), Val(W64)))
        ))
        i = rand(1:8sizeof(Int)); j = rand(Int);
        xi1 = tovector(vi1); xi2 = tovector(vi2);
        for f ∈ [+, -, *, ÷, /, %, <<, >>, >>>, ⊻, &, |, VectorizationBase.rotate_left, VectorizationBase.rotate_right, copysign, max, min]
            @show f
            @test tovector(f(vi1, vi2)) ≈ f.(xi1, xi2)
            @test tovector(f(j, vi2)) ≈ f.(j, xi2)
            @test tovector(f(vi1, i)) ≈ f.(xi1, i)
        end
        
        vf1 = VectorizationBase.VecUnroll((
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64))),
            Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        ))
        vf2 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        xf1 = tovector(vf1); xf2 = tovector(vf2); xf22 = vcat(xf2,xf2)
        a = randn();
        for f ∈ [+, -, *, /, %, max, min, copysign]
            # @show f
            @test tovector(f(vf1, vf2)) ≈ f.(xf1, xf22)
            @test tovector(f(a, vf1)) ≈ f.(a, xf1)
            @test tovector(f(a, vf2)) ≈ f.(a, xf2)
            @test tovector(f(vf1, a)) ≈ f.(xf1, a)
            @test tovector(f(vf2, a)) ≈ f.(xf2, a)
        end
    end
    @testset "Ternary Functions" begin
        v1 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        v2 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        v3 = Vec(ntuple(_ -> Core.VecElement(randn()), Val(W64)))
        x1 = tovector(v1); x2 = tovector(v2); x3 = tovector(v3);
        a = randn(); b = randn()
        for f ∈ [
            muladd, fma,
            VectorizationBase.vfmadd, VectorizationBase.vfnmadd, VectorizationBase.vfmsub, VectorizationBase.vfnmsub,
            VectorizationBase.vfmadd231, VectorizationBase.vfnmadd231, VectorizationBase.vfmsub231, VectorizationBase.vfnmsub231
        ]
            @test tovector(f(v1, v2, v3)) ≈ map(f, x1, x2, x3)
            @test tovector(f(v1, v2, a)) ≈ f.(x1, x2, a)
            @test tovector(f(v1, a, v3)) ≈ f.(x1, a, x3)
            @test tovector(f(a, v2, v3)) ≈ f.(a, x2, x3)
            @test tovector(f(v1, a, b)) ≈ f.(x1, a, b)
            @test tovector(f(a, v2, b)) ≈ f.(a, x2, b)
            @test tovector(f(a, b, v3)) ≈ f.(a, b, x3)
        end
    end
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
            # @test 99.9 === VectorizationBase.vload(ptr_A + 8*3) === VectorizationBase.vload(vA, (VectorizationBase.Static(3),)) === VectorizationBase.vload(vA, (3,0)) === A[4,1]
            # VectorizationBase.vstore!(vAtt, 99.9, (3,1))
            # @test 99.9 === VectorizationBase.vload(vAtt, (3,1)) === VectorizationBase.vload(vAtt, (VectorizationBase.Static(3),1)) === Att[4,2]
            # VectorizationBase.vnoaliasstore!(ptr_A+8*4, 999.9)
            # @test 999.9 === VectorizationBase.vload(ptr_A + 8*4) === VectorizationBase.vload(pointer(vA), 4*sizeof(eltype(A))) === VectorizationBase.vload(vA, (4,))
            # @test vload(vA, (7,2)) == vload(vAtt, (7,2)) == A[8,3]
            # @test vload(VectorizationBase.subsetview(vA, Val(1), 7), (2,)) == vload(VectorizationBase.subsetview(vAtt, Val(1), 7), (2,)) == A[8,3]
            # @test vload(VectorizationBase.subsetview(vA, Val(2), 2), (7,)) == vload(VectorizationBase.subsetview(vAtt, Val(2), 2), (7,)) == A[8,3]
            #     @test vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (2,)) == vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (VectorizationBase.Static(2),)) == A[3,3]
            #     @test vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (1,)) == vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (VectorizationBase.Static(1),)) == A[2,2]
            #     B = rand(5, 5)
            # vB = VectorizationBase.stridedpointer(B)
            # @test vB[1, 2] == B[2, 3] == vload(VectorizationBase.stridedpointer(B, 2, 3))
            # @test vB[3] == B[4] == vload(VectorizationBase.stridedpointer(B, 4))
            # @test vload(Vec{4,Float64}, vB) == Vec{4,Float64}(ntuple(i->B[i], Val(4)))

