using VectorizationBase
using Test

W64 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float64)
W32 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float32)

@testset "VectorizationBase.jl" begin
    # Write your own tests here.
@testset "Struct-Wrapped Vec" begin
@test extract_data(zero(SVec{4,Float64})) === (VE(0.0),VE(0.0),VE(0.0),VE(0.0)) === extract_data(SVec{4,Float64}(0.0))
@test extract_data(one(SVec{4,Float64})) === (VE(1.0),VE(1.0),VE(1.0),VE(1.0)) === extract_data(SVec{4,Float64}(1.0))
v = SVec((VE(1.0),VE(2.0),VE(3.0),VE(4.0)))
for i in 1:4
    @test i == v[i]
    @test i === SVec{4,Int}(v)[i]
end
@test zero(v) === zero(typeof(v))
@test one(v) === one(typeof(v))
@test SVec{W32,Float32}(one(SVec{W32,Float64})) === SVec(one(SVec{W32,Float32})) === one(SVec{W32,Float32})
end

@testset "alignment.jl" begin

@test all(i -> VectorizationBase.align(i) == VectorizationBase.REGISTER_SIZE, 1:VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i) == 2VectorizationBase.REGISTER_SIZE, 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i) == 10VectorizationBase.REGISTER_SIZE, (1:VectorizationBase.REGISTER_SIZE) .+ 9VectorizationBase.REGISTER_SIZE)

@test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},   VectorizationBase.REGISTER_SIZE), 1:VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid},  2VectorizationBase.REGISTER_SIZE), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(reinterpret(Ptr{Cvoid}, i)) == reinterpret(Ptr{Cvoid}, 20VectorizationBase.REGISTER_SIZE), (1:VectorizationBase.REGISTER_SIZE) .+ 19VectorizationBase.REGISTER_SIZE)

@test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == 8cld(i,8), 1:VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == 8cld(i,8), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i,W32) == VectorizationBase.align(i,Float32) == VectorizationBase.align(i,Int32) == 8cld(i,8), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

@test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == 4cld(i,4), 1:VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == 4cld(i,4), 1+VectorizationBase.REGISTER_SIZE:2VectorizationBase.REGISTER_SIZE)
@test all(i -> VectorizationBase.align(i,W64) == VectorizationBase.align(i,Float64) == VectorizationBase.align(i,Int64) == 4cld(i,4), (1:VectorizationBase.REGISTER_SIZE) .+ 29VectorizationBase.REGISTER_SIZE)

end

@testset "masks.jl" begin
@test all(w -> VectorizationBase.mask_type(w) == UInt8, 1:8)
@test all(w -> VectorizationBase.mask_type(w) == UInt16, 9:16)
@test all(w -> VectorizationBase.mask_type(w) == UInt32, 17:32)
@test all(w -> VectorizationBase.mask_type(w) == UInt64, 33:64)
@test all(w -> VectorizationBase.mask_type(w) == UInt128, 65:128)
if VectorizationBase.REGISTER_SIZE == 64 # avx512
    @test VectorizationBase.mask_type(Float16) == UInt32
    @test VectorizationBase.mask_type(Float32) == UInt16
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) === 0xffffffff # 32
    @test VectorizationBase.max_mask(Float32) === 0xffff     # 16
    @test VectorizationBase.max_mask(Float64) === 0xff       # 8
elseif VectorizationBase.REGISTER_SIZE == 32 # avx or avx2
    @test VectorizationBase.mask_type(Float16) == UInt16
    @test VectorizationBase.mask_type(Float32) == UInt8
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) === 0xffff     # 16
    @test VectorizationBase.max_mask(Float32) === 0xff       # 8
    @test VectorizationBase.max_mask(Float64) === 0x0f       # 4
elseif VectorizationBase.REGISTER_SIZE == 16 # sse
    @test VectorizationBase.mask_type(Float16) == UInt8
    @test VectorizationBase.mask_type(Float32) == UInt8
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) === 0xff       # 8
    @test VectorizationBase.max_mask(Float32) === 0x0f       # 4
    @test VectorizationBase.max_mask(Float64) === 0x03       # 2
end
@test all(w -> bitstring(VectorizationBase.mask(Val( 8), w)) == reduce(*, ( 8 - i < w ? "1" : "0" for i in 1:8 )), 0:8 )
@test all(w -> bitstring(VectorizationBase.mask(Val(16), w)) == reduce(*, (16 - i < w ? "1" : "0" for i in 1:16)), 0:16)
@test all(w -> VectorizationBase.mask_from_remainder(Float64, w) === VectorizationBase.mask(Float64, w) === VectorizationBase.mask(Val(8), w), 0:W64)
end

@testset "number_vectors.jl" begin
A = randn(13, 17); L = length(A); M, N = size(A);
# eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :(size(A)), 8)) # doesn't work?
@test VectorizationBase.length_loads(A, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13*17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13*17, 8)) == divrem(length(A), 8)
@test VectorizationBase.size_loads(A,1, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13   )()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13   , 8)) == divrem(size(A,1), 8)
@test VectorizationBase.size_loads(A,2, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() ->    17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__,    17, 8)) == divrem(size(A,2), 8)
end

@testset "vector_width.jl" begin
@test all(VectorizationBase.power2check, 0:1)
@test all(i -> !any(VectorizationBase.power2check, 1+(1 << (i-1)):(1 << i)-1 ) && VectorizationBase.power2check(1 << i), 2:9)
@test all(i ->  VectorizationBase.intlog2(1 << i) == i, 0:(Int == Int64 ? 53 : 30))
FTypes = (Float16, Float32, Float64)
Wv = ntuple(i -> VectorizationBase.REGISTER_SIZE >> i, Val(3))
for (T, N) in zip(FTypes,Wv)
    W, Wshift = VectorizationBase.pick_vector_width_shift(:IGNORE_ME, T)
    @test W == 1 << Wshift == VectorizationBase.pick_vector_width(T) == N
    while true
        W >>= 1
        W == 0 && break
        W2, Wshift2 = VectorizationBase.pick_vector_width_shift(W, T)
        @test W2 == 1 << Wshift2 == VectorizationBase.pick_vector_width(W, T) == W
        for n in W+1:2W
            W3, Wshift3 = VectorizationBase.pick_vector_width_shift(n, T)
            @test W2 << 1 == W3 == 1 << (Wshift2+1) == 1 << Wshift3 == VectorizationBase.pick_vector_width(n, T) == W << 1
        end
    end
end

end

@testset "vectorizable.jl" begin
A = Float64.(0:15)
ptr_A = pointer(A)
vA = VectorizationBase.vectorizable(A)
vA == VectorizationBase.vectorizable(ptr_A)
@test all(i -> A[i+1] === VectorizationBase.load(ptr_A + 8i) === VectorizationBase.load(vA + i) === vA[i+1] === (vA+i)[] === Float64(i), 0:15)
VectorizationBase.store!(vA+3, 99.9)
@test (vA + 3)[] === vA[4] ===  99.9 === VectorizationBase.load(ptr_A + 8*3)
VectorizationBase.store!(ptr_A+8*4, 999.9)
@test (vA + 4)[] === vA[5] === 999.9 === VectorizationBase.load(ptr_A + 8*4)
end

end
