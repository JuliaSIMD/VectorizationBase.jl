

function intlog2(N::I) where {I <: Integer}
    # This version may be slightly faster when vectorized?
    # u = 0x4330000000000000 + N
    # d = reinterpret(Float64,u) - 4503599627370496.0
    # u = (reinterpret(Int64, d) >> 52) - 1023
    # This version is easier to read and understand how it works,
    # and probably faster when evaluated serially
    u = 8sizeof(I) - 1 - leading_zeros(N)
    Base.unsafe_trunc(I, u)
end
intlog2(::Type{T}) where {T} = intlog2(sizeof(T))
ispow2(x::Integer) = (x & (x - 1)) == zero(x)
nextpow2(W) = vleft_bitshift(one(W), (8sizeof(W) - leading_zeros(W-1)))
prevpow2(W) = vleft_bitshift(one(W), (8sizeof(W) - leading_zeros(W)-1))
    # W -= 1
    # W |= W >> 1
    # W |= W >> 2
    # W |= W >> 4
    # W |= W >> 8
    # W |= W >> 16
    # W + 1

@generated function T_shift(::Type{T}) where {T}
    intlog2(sizeof(T))
end
@generated function pick_vector_width(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    max(1, REGISTER_SIZE >>> shift)
end
@generated function pick_vector_width_shift(::Type{T} = Float64) where {T}
    shift = T_shift(T)
    W = max(1, REGISTER_SIZE >>> shift)
    Wshift = intlog2(W)
    W, Wshift
end
function downadjust_W_and_Wshift(N, W, Wshift)
    N > W && return W, Wshift
    TwoN = N << 1
    while W >= TwoN
        W >>>= 1
        Wshift -= 1
    end
    W, Wshift
end
function pick_vector_width_shift(N::Integer, ::Type{T} = Float64) where {T}
    W, Wshift = pick_vector_width_shift(T)
    downadjust_W_and_Wshift(N, W, Wshift)
end
function pick_vector_width(N::Integer, ::Type{T} = Float64) where {T}
    W = pick_vector_width(T)
    first(downadjust_W_and_Wshift(N, W, 0))
end
function pick_vector_width_shift(N::Integer, size_T::Integer)
    W = max(1, REGISTER_SIZE ÷ size_T)
    Wshift = intlog2(W)
    downadjust_W_and_Wshift(N, W, Wshift)
end
function pick_vector_width(N::Integer, size_T::Integer)
    W = max(1, REGISTER_SIZE ÷ size_T)
    first(downadjust_W_and_Wshift(N, W, 0))
end

pick_vector_width(::Symbol, T) = pick_vector_width(T)
pick_vector_width_shift(::Symbol, T) = pick_vector_width_shift(T)

@generated pick_vector_width(::Val{N}, ::Type{T} = Float64) where {N,T} = pick_vector_width(N, T)

@generated pick_vector_width_val(::Type{T} = Float64) where {T} = Val{pick_vector_width(T)}()
pick_vector_width_val(::Type{Bool}) = Val{16}()
@generated pick_vector_width_val(::Val{N}, ::Type{T} = Float64) where {N,T} = Val{pick_vector_width(Val(N), T)}()

@generated function pick_vector_width_val(vargs...)
    sT = 1
    has_bool = false
    for v ∈ vargs
        T = v.parameters[1]
        if T === Bool
            has_bool = true#; sT = min(sT, sTv)
        elseif !SIMD_NATIVE_INTEGERS && T <: Integer
            sT = REGISTER_SIZE
        else
            # sT = min(sT, sizeof(T))
            sT = max(sT, sizeof(T))
        end
    end
    W = REGISTER_SIZE ÷ sT
    W = max(ifelse(has_bool, 8, 1), W)
    Val{W}()
end
@generated function adjust_W(::Val{N}, ::Val{W}) where {N,W}
    W1 = first(downadjust_W_and_Wshift(N, W, 0))
    Val{W1}()
end
pick_vector_width_val(::Val{N}, vargs...) where {N} = adjust_W(Val{N}(), pick_vector_width_val(vargs...))

@inline Base.@pure vadd(a::Int64, b::Int64) = llvmcall("%res = add nsw i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b)
@inline Base.@pure vsub(a::Int64, b::Int64) = llvmcall("%res = sub nsw i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b)
@inline Base.@pure vmul(a::Int64, b::Int64) = llvmcall("%res = mul nsw i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b)
@inline Base.@pure vadd(a::Int32, b::Int32) = llvmcall("%res = add nsw i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b)
@inline Base.@pure vsub(a::Int32, b::Int32) = llvmcall("%res = sub nsw i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b)
@inline Base.@pure vmul(a::Int32, b::Int32) = llvmcall("%res = mul nsw i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b)

@inline Base.@pure vleft_bitshift(a::Int64, b::Int64) = llvmcall("%res = shl nsw i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b)
@inline Base.@pure vleft_bitshift(a::Int32, b::Int32) = llvmcall("%res = shl nsw i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b)

# @inline Base.@pure vright_bitshift(a::Int64, b::Int64) = llvmcall("%res = ashr nsw i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b)
# @inline Base.@pure vright_bitshift(a::Int32, b::Int32) = llvmcall("%res = ashr nsw i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b)

@inline vadd(::Static{i}, j) where {i} = vadd(i, j)
@inline vadd(i, ::Static{j}) where {j} = vadd(i, j)
@inline vsub(::Static{i}, j) where {i} = vsub(i, j)
@inline vsub(i, ::Static{j}) where {j} = vsub(i, j)

@inline staticmul(::Val{W}, i) where {W} = vmul(Static(W), i)
@inline staticmuladd(::Val{W}, b, c) where {W} = vadd(vmul(Static(W), b), c)
@inline valmul(::Val{W}, i) where {W} = vmul(W, i)
@inline valadd(::Val{W}, i) where {W} = vadd(W, i)
@inline valsub(::Val{W}, i) where {W} = vsub(W, i)
@inline valsub(i, ::Val{W}) where {W} = vsub(i, W)
@inline valrem(::Val{W}, i) where {W} = i & (W - 1)
@inline valmuladd(::Val{W}, b, c) where {W} = vadd(vmul(W, b), c)
@inline valmulsub(::Val{W}, b, c) where {W} = vsub(vmul(W, b), c)
@inline valmul(::Val{W}, i::T) where {W,T<:Integer} = vmul((W % T), i)
@inline valadd(::Val{W}, i::T) where {W,T<:Integer} = vadd((W % T), i)
@inline valsub(::Val{W}, i::T) where {W,T<:Integer} = vsub((W % T), i)
@inline valrem(::Val{W}, i::T) where {W,T<:Integer} = i & ((W % T) - one(T))
@inline valmuladd(::Val{W}, b::T, c::T) where {W,T<:Integer} = vadd(vmul(W % T, b), c)
@inline valmulsub(::Val{W}, b::T, c::T) where {W,T<:Integer} = vsub(vmul(W % T, b), c)

@generated pick_vector(::Type{T}) where {T} = Vec{pick_vector_width(T),T}
pick_vector(N, T) = Vec{pick_vector_width(N, T),T}
@generated pick_vector(::Val{N}, ::Type{T}) where {N, T} =  pick_vector(N, T)

@inline _MM(::Val{W}) where {W} = _MM{W}(0)
@inline _MM(::Val{W}, i) where {W} = _MM{W}(i)
@inline _MM{W}(a::LazyStaticMul) where {W} = _MM{W}(extract_data(a))
@inline gep(ptr::Ptr, i::_MM) = gep(ptr, i.i)

@inline staticm1(i::_MM{W,I}) where {W,I} = _MM{W}(vsub(i.i, one(I)))
@inline staticp1(i::_MM{W,I}) where {W,I} = _MM{W}(vadd(i.i, one(I)))
@inline vadd(i::_MM{W}, j::Integer) where {W} = _MM{W}(vadd(i.i, j))
@inline vadd(i::Integer, j::_MM{W}) where {W} = _MM{W}(vadd(i, j.i))
@inline vadd(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(vadd(i.i, j))
@inline vadd(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(vadd(i, j.i))
@inline vsub(i::_MM{W}, j::Integer) where {W} = _MM{W}(vsub(i.i, j))
@inline vsub(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(vsub(i.i, j))
@inline vmulnp(i, j) = vmul(i,j)
@inline vmulnp(i, j::_MM{W}) where {W} = _MM{W}(vmul(i, j.i))
@inline vmuladdnp(a, b, c) = vadd(vmul(a,b), c)
@inline vmuladdnp(a, b::_MM{W}, c) where {W} = vadd(_MM{W}(vmul(a,b.i)), c)
@inline vmuladdnp(a, b::_MM{W}, c::_MM{W}) where {W} = vadd(vmul(a,b), c)
@inline vmuladdnp(a, b::_MM{W}, c::SVec) where {W} = vadd(vmul(a,b), c)
@inline vmuladdnp(a, b::_MM{W}, c::_Vec) where {W} = vadd(vmul(a,b), c)
@inline Base.:(+)(i::_MM{W}, j::Integer) where {W} = _MM{W}(vadd(i.i, j))
@inline Base.:(+)(i::Integer, j::_MM{W}) where {W} = _MM{W}(vadd(i, j.i))
@inline Base.:(+)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(vadd(i.i, j))
@inline Base.:(+)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(vadd(i, j.i))
# @inline Base.:(+)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i + j.i)
@inline Base.:(-)(i::_MM{W}, j::Integer) where {W} = _MM{W}(vsub(i.i, j))
# @inline Base.:(-)(i::Integer, j::_MM{W}) where {W} = _MM{W}(i - j.i)
@inline Base.:(-)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(vsub(i.i, j))
# @inline Base.:(-)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i - j.i)
# @inline Base.:(-)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i - j.i)
# @inline Base.:(*)(i::_MM{W}, j) where {W} = _MM{W}(i.i * j)
# @inline Base.:(*)(i, j::_MM{W}) where {W} = _MM{W}(i * j.i)
# @inline Base.:(*)(i::_MM{W}, ::Static{j}) where {W,j} = _MM{W}(i.i * j)
# @inline Base.:(*)(::Static{i}, j::_MM{W}) where {W,i} = _MM{W}(i * j.i)
# @inline Base.:(*)(i::_MM{W}, j::_MM{W}) where {W} = _MM{W}(i.i * j.i)
@inline scalar_less(i, j) = i < j
@inline scalar_less(i::_MM, j::Integer) = i.i < j
@inline scalar_less(i::Integer, j::_MM) = i < j.i
@inline scalar_less(i::_MM, ::Static{j}) where {j} = i.i < j
@inline scalar_less(::Static{i}, j::_MM) where {i} = i < j.i
@inline scalar_less(i::_MM, j::_MM) = i.i < j.i
@inline scalar_greater(i, j) = i > j
@inline scalar_greater(i::_MM, j::Integer) = i.i > j
@inline scalar_greater(i::Integer, j::_MM) = i > j.i
@inline scalar_greater(i::_MM, ::Static{j}) where {j} = i.i > j
@inline scalar_greater(::Static{i}, j::_MM) where {i} = i > j.i
@inline scalar_greater(i::_MM, j::_MM) = i.i > j.i
@inline scalar_equal(i, j) = i == j
@inline scalar_equal(i::_MM, j::Integer) = i.i == j
@inline scalar_equal(i::Integer, j::_MM) = i == j.i
@inline scalar_equal(i::_MM, ::Static{j}) where {j} = i.i == j
@inline scalar_equal(::Static{i}, j::_MM) where {i} = i == j.i
@inline scalar_equal(i::_MM, j::_MM) = i.i == j.i
@inline scalar_notequal(i, j) = i != j
@inline scalar_notequal(i::_MM, j::Integer) = i.i != j
@inline scalar_notequal(i::Integer, j::_MM) = i != j.i
@inline scalar_notequal(i::_MM, ::Static{j}) where {j} = i.i != j
@inline scalar_notequal(::Static{i}, j::_MM) where {i} = i != j.i
@inline scalar_notequal(i::_MM, j::_MM) = i.i != j.i

@inline extract_data(i::_MM) = i.i



for T ∈ [Float32,Float64,Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64]#, Float16]
    maxW = pick_vector_width(T)
    typ = llvmtype(T)
    for log2W ∈ 0:intlog2(maxW)
        W = 1 << log2W
        instrs = "ret <$W x $typ> zeroinitializer"
        @eval @inline vzero(::Val{$W}, ::Type{$T}) = SVec(llvmcall($instrs, Vec{$W,$T}, Tuple{}, ))
        vtyp = "<$W x $typ>"
        instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        ret $vtyp %v
        """
        @eval @inline vbroadcast(::Val{$W}, s::$T) = SVec(llvmcall($instrs, Vec{$W,$T}, Tuple{$T}, s))
    end
end

# @inline _vload(ptr::Ptr{T}, i::Integer) where {T} = vload(ptr + vmul(sizeof(T), i))
# @inline _vload(ptr::Ptr, v::SVec{<:Any,<:Integer}) = vload(ptr, v.data)
# @inline _vload(ptr::Ptr, v::Vec{<:Any,<:Integer}) = vload(ptr, v)
# @inline _vload(ptr::Ptr{T}, i::_MM{W}) where {W,T} = vload(Val{W}(), ptr + vmul(sizeof(T), i.i))
# @inline _vload(ptr::AbstractPointer, i) = _vload(ptr.ptr, offset(ptr, i))
# @inline vload(ptr::AbstractPointer{T}, i::Tuple) where {T} = _vload(ptr.ptr, offset(ptr, i))


