using VectorizationBase
using Test

const W64 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float64)
const W32 = VectorizationBase.REGISTER_SIZE ÷ sizeof(Float32)

A = randn(13, 17); L = length(A); M, N = size(A);

@testset "VectorizationBase.jl" begin
    # Write your own tests here.
@test isempty(detect_unbound_args(VectorizationBase))

@test first(A) === A[1]
@testset "Struct-Wrapped Vec" begin
@test extract_data(zero(SVec{4,Float64})) === (VE(0.0),VE(0.0),VE(0.0),VE(0.0)) === extract_data(SVec{4,Float64}(0.0))
@test extract_data(one(SVec{4,Float64})) === (VE(1.0),VE(1.0),VE(1.0),VE(1.0)) === extract_data(SVec{4,Float64}(1.0)) === extract_data(extract_data(SVec{4,Float64}(1.0)))
v = SVec((VE(1.0),VE(2.0),VE(3.0),VE(4.0)))
@test v === SVec{4,Float64}(1, 2, 3, 4) === conj(v) === v'
@test length(v) == 4 == first(size(v))
@test eltype(v) == Float64
for i in 1:4
    @test i == v[i]
    # @test i === SVec{4,Int}(v)[i] # should use fptosi (ie, vconvert defined in SIMDPirates).
end
@test zero(v) === zero(typeof(v))
@test one(v) === one(typeof(v))
# @test SVec{W32,Float32}(one(SVec{W32,Float64})) === SVec(one(SVec{W32,Float32})) === one(SVec{W32,Float32}) # conversions should be tested in SIMDPirates
    @test firstval(v) === firstval(extract_data(v)) === 1.0
    @test SVec{1,Int}(1) === SVec{1,Int}((Core.VecElement(1),))
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
    @test Mask{8,UInt8}(0x0f) === @inferred Mask(0x0f)
    @test Mask{16,UInt16}(0x0f0f) === @inferred Mask(0x0f0f)
    @test Mask{8,UInt8}(0xff) == mask(Val(8), 0)
    @test Mask{8,UInt8}(0xff) == mask(Val(8), 8)
    @test Mask{8,UInt8}(0xff) == mask(Val(8), 16)
    @test Mask{8,UInt8}(0xff) == mask(Val(8), VectorizationBase.Static(0))
    @test Mask{16,UInt16}(0xffff) == mask(Val(16), 0)
    @test Mask{16,UInt16}(0xffff) == mask(Val(16), 16)
    @test Mask{16,UInt16}(0xffff) == mask(Val(16), 32)
@test all(w -> VectorizationBase.mask_type(w) == UInt8, 1:8)
@test all(w -> VectorizationBase.mask_type(w) == UInt16, 9:16)
@test all(w -> VectorizationBase.mask_type(w) == UInt32, 17:32)
@test all(w -> VectorizationBase.mask_type(w) == UInt64, 33:64)
@test all(w -> VectorizationBase.mask_type(w) == UInt128, 65:128)
if VectorizationBase.REGISTER_SIZE == 64 # avx512
    @test VectorizationBase.mask_type(Float16) == UInt32
    @test VectorizationBase.mask_type(Float32) == UInt16
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) == 0xffffffff # 32
    @test VectorizationBase.max_mask(Float32) == 0xffff     # 16
    @test VectorizationBase.max_mask(Float64) == 0xff       # 8
elseif VectorizationBase.REGISTER_SIZE == 32 # avx or avx2
    @test VectorizationBase.mask_type(Float16) == UInt16
    @test VectorizationBase.mask_type(Float32) == UInt8
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) == 0xffff     # 16
    @test VectorizationBase.max_mask(Float32) == 0xff       # 8
    @test VectorizationBase.max_mask(Float64) == 0x0f       # 4
elseif VectorizationBase.REGISTER_SIZE == 16 # sse
    @test VectorizationBase.mask_type(Float16) == UInt8
    @test VectorizationBase.mask_type(Float32) == UInt8
    @test VectorizationBase.mask_type(Float64) == UInt8
    @test VectorizationBase.max_mask(Float16) == 0xff       # 8
    @test VectorizationBase.max_mask(Float32) == 0x0f       # 4
    @test VectorizationBase.max_mask(Float64) == 0x03       # 2
end
@test all(w -> bitstring(VectorizationBase.mask(Val( 8), w)) == reduce(*, ( 8 - i < w ? "1" : "0" for i in 1:8 )), 1:8 )
@test all(w -> bitstring(VectorizationBase.mask(Val(16), w)) == reduce(*, (16 - i < w ? "1" : "0" for i in 1:16)), 1:16)
        @test all(w -> VectorizationBase.mask(Float64, w) === VectorizationBase.mask(VectorizationBase.pick_vector_width_val(Float64), w), 1:W64)

        @test VectorizationBase.vbroadcast(Val(8), true) === true
end

@testset "number_vectors.jl" begin
# eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :(size(A)), 8)) # doesn't work?
@test VectorizationBase.length_loads(A, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13*17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13*17, 8)) == divrem(length(A), 8)
@test VectorizationBase.size_loads(A,1, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() -> 13   )()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, 13   , 8)) == divrem(size(A,1), 8)
@test VectorizationBase.size_loads(A,2, Val(8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__, :((() ->    17)()), 8)) == eval(VectorizationBase.num_vector_load_expr(@__MODULE__,    17, 8)) == divrem(size(A,2), 8)
end

@testset "vector_width.jl" begin
@test all(VectorizationBase.ispow2, 0:1)
@test all(i -> !any(VectorizationBase.ispow2, 1+(1 << (i-1)):(1 << i)-1 ) && VectorizationBase.ispow2(1 << i), 2:9)
@test all(i ->  VectorizationBase.intlog2(1 << i) == i, 0:(Int == Int64 ? 53 : 30))
FTypes = (Float16, Float32, Float64)
Wv = ntuple(i -> VectorizationBase.REGISTER_SIZE >> i, Val(3))
for (T, N) in zip(FTypes, Wv)
    W, Wshift = VectorizationBase.pick_vector_width_shift(:IGNORE_ME, T)
    @test W == 1 << Wshift == VectorizationBase.pick_vector_width(T) == N == VectorizationBase.pick_vector_width(:IGNORE_ME, T)
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
A = reshape(collect(Float64(0):Float64(63)), (16, 4))
ptr_A = pointer(A)
vA = VectorizationBase.stridedpointer(A)
Att = copy(A')'
vAtt = VectorizationBase.stridedpointer(Att)
@test eltype(vA) == Float64
@test Base.unsafe_convert(Ptr{Float64}, vA) === ptr_A === pointer(vA)
@test vA == VectorizationBase.stridedpointer(vA)
@test all(i -> A[i+1] === VectorizationBase.vload(ptr_A + 8i) === VectorizationBase.vload(vA, (i,)) === Float64(i), 0:15)
VectorizationBase.vstore!(vA, 99.9, (3,))
@test 99.9 === VectorizationBase.vload(ptr_A + 8*3) === VectorizationBase.vload(vA, (VectorizationBase.Static(3),)) === VectorizationBase.vload(vA, (3,0)) === A[4,1]
VectorizationBase.vstore!(vAtt, 99.9, (3,1))
@test 99.9 === VectorizationBase.vload(vAtt, (3,1)) === VectorizationBase.vload(vAtt, (VectorizationBase.Static(3),1)) === Att[4,2]
VectorizationBase.vnoaliasstore!(ptr_A+8*4, 999.9)
@test 999.9 === VectorizationBase.vload(ptr_A + 8*4) === VectorizationBase.vload(pointer(vA), 4*sizeof(eltype(A))) === VectorizationBase.vload(vA, (4,))
@test vload(vA, (7,2)) == vload(vAtt, (7,2)) == A[8,3]
@test vload(VectorizationBase.subsetview(vA, Val(1), 7), (2,)) == vload(VectorizationBase.subsetview(vAtt, Val(1), 7), (2,)) == A[8,3]
@test vload(VectorizationBase.subsetview(vA, Val(2), 2), (7,)) == vload(VectorizationBase.subsetview(vAtt, Val(2), 2), (7,)) == A[8,3]
    @test vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (2,)) == vload(VectorizationBase.double_index(vA, Val(0), Val(1)), (VectorizationBase.Static(2),)) == A[3,3]
    @test vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (1,)) == vload(VectorizationBase.double_index(vAtt, Val(0), Val(1)), (VectorizationBase.Static(1),)) == A[2,2]
    B = rand(5, 5)
vB = VectorizationBase.stridedpointer(B)
@test vB[1, 2] == B[2, 3] == vload(VectorizationBase.stridedpointer(B, 2, 3))
@test vB[3] == B[4] == vload(VectorizationBase.stridedpointer(B, 4))
@test vload(SVec{4,Float64}, vB) == SVec{4,Float64}(ntuple(i->B[i], Val(4)))
end

end
