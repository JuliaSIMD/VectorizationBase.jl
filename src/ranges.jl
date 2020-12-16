

function pick_integer_bytes(W, preferred)
    # SIMD quadword integer support requires AVX512DQ
    if !AVX512DQ
        preferred = min(4, preferred)
    end
    max(1,min(preferred, prevpow2(REGISTER_SIZE ÷ W)))
end
function integer_of_bytes(bytes)
    if bytes == 8
        Int64
    elseif bytes == 4
        Int32
    elseif bytes == 2
        Int16
    elseif bytes == 1
        Int8
    else
        throw("$bytes is an invalid number of bytes for integers.")
    end
end
function pick_integer(W, pref)
    integer_of_bytes(pick_integer_bytes(W, pref))
end
pick_integer(::Val{W}) where {W} = pick_integer(W, sizeof(Int))

@generated function vrange(::Val{W}, ::Type{T}, ::Val{O}, ::Val{F}) where {W,T,O,F}
    if T <: Integer
        _T2 = pick_integer(W, sizeof(T))
        T2 = T <: Signed ? _T2 : unsigned(_T2)
    else
        T2 = T
    end
    t = Expr(:tuple)
    foreach(w -> push!(t.args, Expr(:call, :(Core.VecElement), T2(F*w + O))), 0:W-1)
    Expr(:block, Expr(:meta, :inline), Expr(:call, :Vec, t))
end

"""
  vrange(::Val{W}, i::I, ::Val{O}, ::Val{F})

W - Vector width
i::I - dynamic offset
O - static offset
F - static multiplicative factor
"""
@generated function vrangeincr(::Val{W}, i::I, ::Val{O}, ::Val{F}) where {W,I<:Integer,O,F}
    isone(W) && return Expr(:block, Expr(:meta,:inline), :(vadd(i, $(O % I))))
    bytes = pick_integer_bytes(W, sizeof(I))
    bits = 8bytes
    jtypesym = Symbol(:Int, bits)
    iexpr = bytes == sizeof(I) ? :i : Expr(:call, :%, :i, jtypesym)
    typ = "i$(bits)"
    vtyp = vtype(W, typ)
    rangevec = join(("$typ $(F*w + O)" for w ∈ 0:W-1), ", ")
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        %res = add nsw $vtyp %v, <$rangevec>
        ret $vtyp %res
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$jtypesym}, Tuple{$jtypesym}, $iexpr))
    end
end
@generated function vrangeincr(::Val{W}, i::T, ::Val{O}, ::Val{F}) where {W,T<:FloatingTypes,O,F}
    isone(W) && return Expr(:block, Expr(:meta,:inline), :(Base.FastMath.add_fast(i, $(T(O)))))
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    rangevec = join(("$typ $(F*w+O).0" for w ∈ 0:W-1), ", ")
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        %res = fadd fast $vtyp %v, <$rangevec>
        ret $vtyp %res
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$T}, i))
    end
end
@generated function vrangemul(::Val{W}, i::I, ::Val{O}, ::Val{F}) where {W,I<:Integer,O,F}
    isone(W) && return Expr(:block, Expr(:meta,:inline), :(vmul(i, $(O % I))))
    bytes = pick_integer_bytes(W, sizeof(T))
    bits = 8bytes
    jtypesym = Symbol(:Int, bits)
    iexpr = bytes == sizeof(I) ? :i : Expr(:call, :%, :i, jtypesym)
    typ = "i$(bits)"
    vtyp = vtype(W, typ)
    rangevec = join(("$typ $(F*w+O)" for w ∈ 0:W-1), ", ")
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        %res = mul nsw $vtyp %v, <$rangevec>
        ret $vtyp %res
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall(instrs, _Vec{$W,$jtypesym}, Tuple{$jtypesym}, $iexpr))
    end
end
@generated function vrangemul(::Val{W}, i::T, ::Val{O}, ::Val{F}) where {W,T<:FloatingTypes,O,F}
    isone(W) && return Expr(:block, Expr(:meta,:inline), :(Base.FastMath.mul_fast(i, $(T(O)))))
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    rangevec = join(("$typ $(F*w+O).0" for w ∈ 0:W-1), ", ")
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        %res = fmul fast $vtyp %v, <$rangevec>
        ret $vtyp %res
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall(instrs, _Vec{$W,$T}, Tuple{$T}, i))
    end
end


@inline Vec(i::MM{W,X}) where {W,X} = vrangeincr(Val{W}(), data(i), Val{0}(), Val{X}())
@inline Vec(i::MM{W,X,StaticInt{N}}) where {W,X,N} = vrange(Val{W}(), Int, Val{N}(), Val{X}())
@inline Vec(i::MM{1}) = data(i)
@inline Vec(i::MM{1,<:Any,StaticInt{N}}) where {N} = N
@inline Base.convert(::Type{Vec{W,T}}, i::MM{W,X}) where {W,X,T} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())
@inline Base.convert(::Type{T}, i::MM{W,X}) where {W,X,T<:NativeTypes} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())

# Addition
# @inline Base.:(+)(i::MM{W}, j::MM{W}) where {W} = vadd(vrange(i), vrange(j))
# @inline Base.:(+)(i::MM{W,X}, j::MM{W,Y}) where {W} = vrange(vrangeincr(Val{W}(), data(i) + data(j), Val{0}(), StaticInt{X}() + StaticInt{Y}()))
@inline Base.:(+)(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vadd(data(i), data(j)), StaticInt{X}() + StaticInt{Y}())
@inline Base.:(+)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd(Vec(i), j)
@inline Base.:(+)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd(i, Vec(j))

# @inline vadd(i::MM{W,X}, j::Integer) where {W,X} = MM{W,X}(vadd(i.i, j))
# @inline vadd(j::Integer, i::MM{W,X}) where {W,X} = MM{W,X}(vadd(i.i, j))
@inline vadd(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vadd(data(i), data(j)), StaticInt{X}() + StaticInt{Y}())
@inline vadd(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd(Vec(i), j)
@inline vadd(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd(i, Vec(j))
# Subtraction
@inline Base.:(-)(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vsub(data(i), data(j)), StaticInt{X}() - StaticInt{Y}())
# @inline Base.:(-)(i::MM{W,X}, j::MM{W,X}) where {W,X} = vsub(data(i), data(j))
@inline Base.:(-)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub(Vec(i), j)
@inline Base.:(-)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub(i, Vec(j))
@inline vsub(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vsub(data(i), data(j)), StaticInt{X}() + StaticInt{Y}())
@inline vsub(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub(Vec(i), j)
@inline vsub(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub(i, Vec(j))
# Multiplication
@inline Base.:(*)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vmul(Vec(i), j)
@inline Base.:(*)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vmul(i, Vec(j))
@inline Base.:(*)(i::MM{W}, j::MM{W}) where {W} = vmul(Vec(i), Vec(j))
@inline vmul(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vmul(Vec(i), j)
@inline vmul(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vmul(i, Vec(j))
@inline vmul(i::MM{W}, j::MM{W}) where {W} = vmul(Vec(i), Vec(j))
@inline vmul(i::MM, j::Integer) = vmul(Vec(i), j)
@inline vmul(j::Integer, i::MM) = vmul(j, Vec(i))

# Multiplication without promotion
@inline vmul_no_promote(a, b) = vmul(a, b)
@inline vmul_no_promote(a::MM{W}, b) where {W} = MM{W}(vmul(a.i, b))
@inline vmul_no_promote(a, b::MM{W}) where {W} = MM{W}(vmul(a, b.i))
@inline vmul_no_promote(a::MM{W}, b::MM{W}) where {W} = vmul(a, b) # must promote
vmul_no_promote(a::MM, b::MM) = throw("Dimension mismatch.")

# Division
@generated function floattype(::Val{W}) where {W}
    (REGISTER_SIZE ÷ W) ≥ 8 ? :Float64 : :Float32
end
@inline Base.float(i::MM{W,X}) where {W,X} = Vec(MM{W,X}(floattype(Val{W}())(i.i)))
@inline Base.:(/)(i::MM, j::T) where {T<:Real} = float(i) / j
@inline Base.:(/)(j::T, i::MM) where {T<:Real} = j / float(i)
@inline Base.:(/)(i::MM, j::MM) = float(i) / float(j)
@inline Base.inv(i::MM) = inv(float(i))
@inline Base.:(/)(vu::VecUnroll, m::MM) = vu * inv(m)
@inline Base.:(/)(m::MM, vu::VecUnroll) = Vec(m) / vu

# @inline Base.:(<<)(i::MM, j::IntegerTypes) = Vec(i) << j
# @inline Base.:(>>)(i::MM, j::IntegerTypes) = Vec(i) >> j
# @inline Base.:(>>>)(i::MM, j::IntegerTypes) = Vec(i) >>> j
# @inline Base.:(<<)(i::MM, j::Vec) = Vec(i) << j
# @inline Base.:(>>)(i::MM, j::Vec) = Vec(i) >> j
# @inline Base.:(>>>)(i::MM, j::Vec) = Vec(i) >>> j

@inline Base.:(<<)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:IntegerTypes} = MM{W}(i.i << j, StaticInt{X}() << j)
@inline Base.:(>>)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:IntegerTypes} = MM{W}(i.i >> j, StaticInt{X}() >> j)
@inline Base.:(>>>)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:IntegerTypes} = MM{W}(i.i >>> j, StaticInt{X}() >>> j)

@inline Base.:(<<)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:StaticInt} = MM{W}(i.i << j, StaticInt{X}() << j)
@inline Base.:(>>)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:StaticInt} = MM{W}(i.i >> j, StaticInt{X}() >> j)
@inline Base.:(>>>)(i::MM{W,X,T}, j::StaticInt) where {W,X,T<:StaticInt} = MM{W}(i.i >>> j, StaticInt{X}() >>> j)


for (f,op) ∈ [
    (:scalar_less, :(<)), (:scalar_greater,:(>)), (:scalar_greaterequal,:(≥)), (:scalar_lessequal,:(≤)), (:scalar_equal,:(==)), (:scalar_notequal,:(!=))
]
    @eval @inline $f(i::MM, j::Real) = $op(data(i), j)
    @eval @inline $f(i::Real, j::MM) = $op(i, data(j))
    @eval @inline $f(i::MM, ::StaticInt{j}) where {j} = $op(data(i), j)
    @eval @inline $f(::StaticInt{i}, j::MM) where {i} = $op(i, data(j))
    @eval @inline $f(i::MM, j::MM) = $op(data(i), data(j))
    @eval @inline $f(i, j) = $op(i, j)
end

for f ∈ [:(<<), :(>>>), :(>>)]
    @eval begin
        @inline Base.$f(i::MM{W,X,T}, v::SignedHW) where {W,X,T<:SignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T}, v::SignedHW) where {W,X,T<:UnsignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T}, v::UnsignedHW) where {W,X,T<:SignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T}, v::UnsignedHW) where {W,X,T<:UnsignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T}, v::IntegerTypes) where {W,X,T<:StaticInt} = $f(Vec(i), v)

        @inline Base.$f(v::SignedHW, i::MM{W,X,T}) where {W,X,T<:SignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::UnsignedHW, i::MM{W,X,T}) where {W,X,T<:SignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::SignedHW, i::MM{W,X,T}) where {W,X,T<:UnsignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::UnsignedHW, i::MM{W,X,T}) where {W,X,T<:UnsignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::IntegerTypes, i::MM{W,X,T}) where {W,X,T<:StaticInt} = $f(v, Vec(i))

        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:SignedHW,T2<:SignedHW} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:UnsignedHW,T2<:SignedHW} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:SignedHW,T2<:UnsignedHW} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:UnsignedHW,T2<:UnsignedHW} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:StaticInt,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:StaticInt} = $f(Vec(i), Vec(j))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:StaticInt,T2<:StaticInt} = $f(Vec(i), Vec(j))

        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:SignedHW,T2<:SignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:UnsignedHW,T2<:SignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:SignedHW,T2<:UnsignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:UnsignedHW,T2<:UnsignedHW} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:StaticInt,T2<:IntegerTypes} = $f(Vec(i), v)

        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:SignedHW,T2<:SignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:UnsignedHW,T2<:SignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:SignedHW,T2<:UnsignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:UnsignedHW,T2<:UnsignedHW} = $f(v, Vec(i))
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:StaticInt} = $f(v, Vec(i))
    end
end
for f ∈ [ :(÷), :(&), :(|), :(⊻), :(%), :(<), :(>), :(≥), :(≤), :(==), :(!=), :min, :max, :copysign]
    @eval begin
        @inline Base.$f(i::MM{W,X,T}, v::IntegerTypes) where {W,X,T<:IntegerTypes} = $f(Vec(i), v)
        @inline Base.$f(v::IntegerTypes, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:IntegerTypes,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:IntegerTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
    end
end
for f ∈ [:(<), :(>), :(≥), :(≤), :(==), :(!=), :min, :max, :copysign]
    
    @eval begin
        # left floating
        @inline Base.$f(i::MM{W,X,T}, v::IntegerTypes) where {W,X,T<:FloatingTypes} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:FloatingTypes,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:FloatingTypes,T2<:IntegerTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:FloatingTypes,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
        # right floating
        @inline Base.$f(i::MM{W,X,T}, v::FloatingTypes) where {W,X,T<:IntegerTypes} = $f(Vec(i), v)
        @inline Base.$f(v::IntegerTypes, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:IntegerTypes,T2<:FloatingTypes} = $f(Vec(i), v)
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:FloatingTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:FloatingTypes} = $f(Vec(i), Vec(j))
        # both floating
        @inline Base.$f(i::MM{W,X,T}, v::FloatingTypes) where {W,X,T<:FloatingTypes} = $f(Vec(i), v)
        @inline Base.$f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:FloatingTypes,T2<:FloatingTypes} = $f(Vec(i), v)
        @inline Base.$f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:FloatingTypes,T2<:FloatingTypes} = $f(v, Vec(i))
        @inline Base.$f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:FloatingTypes,T2<:FloatingTypes} = $f(Vec(i), Vec(j))
    end
    if f === :copysign
        @eval begin
            @inline Base.$f(v::Float32, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline Base.$f(v::Float32, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
            @inline Base.$f(v::Float64, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline Base.$f(v::Float64, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        end
    else
        @eval begin
            @inline Base.$f(v::FloatingTypes, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline Base.$f(v::FloatingTypes, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        end
    end
end

# @inline vadd(::MM{W,Zero}, v::AbstractSIMDVector{W,T}) where {W,T} = vadd(vrange(Val{W}(), T, Val{0}(), Val{1}()), v)
# @inline vadd(v::AbstractSIMDVector{W,T}, ::MM{W,Zero}) where {W,T} = vadd(vrange(Val{W}(), T, Val{0}(), Val{1}()), v)
@inline vadd(i::MM{W,Zero}, j::MM{W,Zero}) where {W} = vrange(Val{W}(), Int, Val{0}(), Val{2}())
# @inline vadd(a::MM, ::Zero) = a
# @inline vadd(::Zero, a::MM) = a
# @inline Base.:(+)(a::MM, ::Zero) = a
# @inline Base.:(+)(::Zero, a::MM) = a
# # @inline vmul(::MM{W,Zero}, i) where {W} = svrangemul(Val{W}(), i, Val{0}())
# @inline vmul(i, ::MM{W,Zero}) where {W} = svrangemul(Val{W}(), i, Val{0}())


