

@inline function Base.log(x1::AbstractSIMD{W,Float64}) where {W}
    x2 = reinterpret(UInt64, x1)
    x3 = x2 >>> 0x0000000000000020
    notzero = x1 != zero(x1)
    greater_than_zero = x1 > zero(x1)
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
    x36 = ifelse(notzero, x35, -Inf)
    x37 = ifelse(greater_than_zero, x36, NaN)
    ifelse(isinf, Inf, x37)
end
# @inline Base.log(v::AbstractSIMD) = log(float(v))

# @inline function vlog_fast(x1::AbstractSIMD{W,Float32}) where {W}
#     notzero = x1 != zero(x1)
#     greater_than_zero = x1 > zero(x1)
#     x3 = x1 < 1.1754944f-38#3.4332275f-5
#     # x6 = true if x3 entirely false
#     x7 = x1 * 8.388608f6#14.0f0
#     x8 = ifelse(x3, x7, x1)
#     x10 = reinterpret(UInt32, x8)
#     x11 = x10 + 0x004afb0d
#     x12 = x11 >>> 0x00000017
#     x13 = ifelse(x3, 0xffffff6a, 0xffffff81)
#     x15 = x12 + x13
#     x16 = x11 & 0x007fffff
#     x17 = x16 + 0x3f3504f3
#     x18 = reinterpret(Float32, x17)
#     x19 = x18 - 1f0
#     x20 = x18 + 1f0
#     x21 = x19 / x20
#     x22 = x21 * x21
#     x23 = x22 * x22
#     x24 = vfmadd(x23, 0.24279079f0, 0.40000972f0)
#     x25 = x24 * x23
#     x26 = vfmadd(x23, 0.6666666f0, 0.6666666f0)
#     x27 = x26 * x22
#     x28 = x19 * x19
#     x29 = x28 * 5f-1
#     x30 = convert(Float32, x15 % Int32)
#     x31 = x27 + x29
#     x32 = x31 + x25
#     x33 = x30 * -0.00021219444f0
#     x34 = vfmadd(x21, x32, x33)
#     x35 = x19 - x29
#     x36 = x35 + x34
#     x37 = vfmadd(x30, 0.6933594f0, x36)
#     # x37 = vfmadd(x30, 0.6931472f0, x36)
#     x39 = ifelse(x1 == Inf32, Inf32, x37)
#     x40 = ifelse(notzero, x39, -Inf32)
#     ifelse(x1 < zero(x1), NaN32, x40)
# end
# @inline Base.FastMath.log_fast(v::AbstractSIMD) = vlog_fast(float(v))


