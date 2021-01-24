# Seperate file to make it easier to include separately from REPL for running pieces
using VectorizationBase, OffsetArrays, Aqua
using VectorizationBase: data
using Test

const W64S = VectorizationBase.pick_vector_width_val(Float64)
const W64 = VectorizationBase.register_size() ÷ sizeof(Float64)
const W32 = VectorizationBase.register_size() ÷ sizeof(Float32)
const VE = Core.VecElement
randnvec(N = Val{W64}()) = Vec(ntuple(_ -> Core.VecElement(randn()), N))
function tovector(u::VectorizationBase.VecUnroll{_N,W,_T}) where {_N,W,_T}
    T = _T === VectorizationBase.Bit ? Bool : _T
    N = _N + 1; i = 0
    x = Vector{T}(undef, N * W)
    for n ∈ 1:N
        v = u.data[n]
        for w ∈ 0:W-1
            x[(i += 1)] = VectorizationBase.extractelement(v, w)
        end
    end
    x
end
tovector(v::VectorizationBase.AbstractSIMDVector{W}) where {W} = [VectorizationBase.extractelement(v,w) for w ∈ 0:W-1]
tovector(v::VectorizationBase.LazyMulAdd) = tovector(VectorizationBase._materialize(v))
tovector(x) = x
tovector(i::MM{W,X}) where {W,X} = collect(range(i.i, step = X, length = W))
tovector(i::MM{W,X,I}) where {W,X,I<:Union{Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64}} = collect(range(i.i, step = I(X), length = I(W)))
A = randn(13, 17); L = length(A); M, N = size(A);

trunc_int(x::Integer, ::Type{T}) where {T} = x % T
trunc_int(x, ::Type{T}) where {T} = x
size_trunc_int(x::Signed, ::Type{T}) where {T} = signed(x % T)
size_trunc_int(x::Unsigned, ::Type{T}) where {T} = unsigned(x % T)
size_trunc_int(x, ::Type{T}) where {T} = x

check_within_limits(x, y) = @test x ≈ y
function check_within_limits(x::Vector{T}, y) where {T <: Integer}
    if VectorizationBase.has_feature("x86_64_avx512dq")
        return @test x ≈ y
    end
    r = typemin(Int32) .≤ y .≤ typemax(Int32)
    xs = x[r]; ys = y[r]
    @test xs ≈ ys
end

maxi(a,b) = max(a,b)
mini(a,b) = min(a,b)
function maxi(a::T1,b::T2) where {T1<:Base.BitInteger,T2<:Base.BitInteger}
    T = promote_type(T1,T2)
    T(a > b ? a : b)
end
function mini(a::T1,b::T2) where {T1<:Base.BitInteger,T2<:Base.BitInteger}
    _T = promote_type(T1,T2)
    T = if T1 <: Signed || T2 <: Signed
        signed(_T)
    else
        _T
    end
    T(a < b ? a : b)
end
maxi_fast(a,b) = Base.FastMath.max_fast(a,b)
mini_fast(a,b) = Base.FastMath.min_fast(a,b)
maxi_fast(a::Base.BitInteger, b::Base.BitInteger) = maxi(a, b)
mini_fast(a::Base.BitInteger, b::Base.BitInteger) = mini(a, b)

