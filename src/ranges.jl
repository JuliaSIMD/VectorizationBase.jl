



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
    isone(W) && return Expr(:block, Expr(:meta,:inline), :(vadd_fast(i, $(O % I))))
    bytes = pick_integer_bytes(W, sizeof(I))
    bits = 8bytes
    jtypesym = Symbol(I <: Signed  ? :Int : :UInt, bits)
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
# @generated function vrangemul(::Val{W}, i::I, ::Val{O}, ::Val{F}) where {W,I<:Integer,O,F}
#     isone(W) && return Expr(:block, Expr(:meta,:inline), :(vmul(i, $(O % I))))
#     bytes = pick_integer_bytes(W, sizeof(T))
#     bits = 8bytes
#     jtypesym = Symbol(I <: Signed  ? :Int : :UInt, bits)
#     iexpr = bytes == sizeof(I) ? :i : Expr(:call, :%, :i, jtypesym)
#     typ = "i$(bits)"
#     vtyp = vtype(W, typ)
#     rangevec = join(("$typ $(F*w+O)" for w ∈ 0:W-1), ", ")
#     instrs = """
#         %ie = insertelement $vtyp undef, $typ %0, i32 0
#         %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
#         %res = mul nsw $vtyp %v, <$rangevec>
#         ret $vtyp %res
#     """
#     quote
#         $(Expr(:meta,:inline))
#         Vec(llvmcall(instrs, _Vec{$W,$jtypesym}, Tuple{$jtypesym}, $iexpr))
#     end
# end
# @generated function vrangemul(::Val{W}, i::T, ::Val{O}, ::Val{F}) where {W,T<:FloatingTypes,O,F}
#     isone(W) && return Expr(:block, Expr(:meta,:inline), :(Base.FastMath.mul_fast(i, $(T(O)))))
#     typ = LLVM_TYPES[T]
#     vtyp = vtype(W, typ)
#     rangevec = join(("$typ $(F*w+O).0" for w ∈ 0:W-1), ", ")
#     instrs = """
#         %ie = insertelement $vtyp undef, $typ %0, i32 0
#         %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
#         %res = fmul fast $vtyp %v, <$rangevec>
#         ret $vtyp %res
#     """
#     quote
#         $(Expr(:meta,:inline))
#         Vec(llvmcall(instrs, _Vec{$W,$T}, Tuple{$T}, i))
#     end
# end


@inline Vec(i::MM{W,X}) where {W,X} = vrangeincr(Val{W}(), data(i), Val{0}(), Val{X}())
@inline Vec(i::MM{W,X,StaticInt{N}}) where {W,X,N} = vrange(Val{W}(), Int, Val{N}(), Val{X}())
@inline Vec(i::MM{1}) = data(i)
@inline Vec(i::MM{1,<:Any,StaticInt{N}}) where {N} = N
@inline vconvert(::Type{Vec{W,T}}, i::MM{W,X}) where {W,X,T} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())
@inline vconvert(::Type{T}, i::MM{W,X}) where {W,X,T<:NativeTypes} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())

# Addition
@inline vadd_fast(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vadd_fast(data(i), data(j)), StaticInt{X}() + StaticInt{Y}())
@inline vadd_fast(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd_fast(Vec(i), j)
@inline vadd_fast(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd_fast(i, Vec(j))
@inline vadd(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = vadd_fast(i, j)
@inline vadd(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd_fast(i, j)
@inline vadd(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd_fast(i, j)

# Subtraction
@inline vsub(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = MM{W}(vsub_fast(data(i), data(j)), StaticInt{X}() - StaticInt{Y}())
@inline vsub(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub_fast(Vec(i), j)
@inline vsub(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub_fast(i, Vec(j))
@inline vsub_fast(i::MM{W,X}, j::MM{W,Y}) where {W,X,Y} = vsub(i, j)
@inline vsub_fast(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub(i, j)
@inline vsub_fast(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub(i, j)
# Multiplication
@inline vmul(i::MM{W}, j::Mask{W}) where {W} = Vec(i) * j
@inline vmul(i::Mask{W}, j::MM{W}) where {W} = i * Vec(j)
@inline vmul(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = Vec(i) * j
@inline vmul(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = i * Vec(j)
@inline vmul(i::MM{W}, j::MM{W}) where {W} = vmul_fast(Vec(i), Vec(j))
@inline vmul_fast(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vmul_fast(Vec(i), j)
@inline vmul_fast(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vmul_fast(i, Vec(j))
@inline vmul_fast(i::MM{W}, j::MM{W}) where {W} = vmul_fast(Vec(i), Vec(j))
@inline vmul_fast(i::MM, j::Integer) = vmul_fast(Vec(i), j)
@inline vmul_fast(j::Integer, i::MM) = vmul_fast(j, Vec(i))

# Multiplication without promotion
@inline vmul_no_promote(a, b) = vmul_fast(a, b)
@inline vmul_no_promote(a::MM{W}, b) where {W} = MM{W}(vmul_fast(a.i, b))
@inline vmul_no_promote(a, b::MM{W}) where {W} = MM{W}(vmul_fast(a, b.i))
@inline vmul_no_promote(a::MM{W}, b::MM{W}) where {W} = vmul_fast(a, b) # must promote
vmul_no_promote(a::MM, b::MM) = throw("Dimension mismatch.")

# Division
@generated function floattype(::Val{W}) where {W}
    (dynamic_register_size() ÷ W) ≥ 8 ? :Float64 : :Float32
end
@inline vfloat(i::MM{W,X,I}) where {W,X,I} = Vec(MM{W,X}(floattype(Val{W}())(i.i % pick_integer(Val{W}(),I))))
@inline vfdiv(i::MM, j::T) where {T<:Real} = float(i) / j
@inline vfdiv(j::T, i::MM) where {T<:Real} = j / float(i)

@inline vfdiv(i::MM, j::VecUnroll{N,W,T,V}) where {N,W,T,V} = float(i) / j
@inline vfdiv(j::VecUnroll{N,W,T,V}, i::MM) where {N,W,T,V} = j / float(i)

@inline vfdiv(i::MM, j::MM) = float(i) / float(j)
@inline vfdiv(vu::VecUnroll, m::MM) = vu * inv(m)
@inline vfdiv(m::MM, vu::VecUnroll) = Vec(m) / vu

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

for f ∈ [:vshl, :vashr, :vlshr]
    @eval begin
        @inline $f(i::MM{W,X,T}, v::SignedHW) where {W,X,T<:SignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T}, v::SignedHW) where {W,X,T<:UnsignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T}, v::UnsignedHW) where {W,X,T<:SignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T}, v::UnsignedHW) where {W,X,T<:UnsignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T}, v::IntegerTypesHW) where {W,X,T<:StaticInt} = $f(Vec(i), v)

        @inline $f(v::SignedHW, i::MM{W,X,T}) where {W,X,T<:SignedHW} = $f(v, Vec(i))
        @inline $f(v::UnsignedHW, i::MM{W,X,T}) where {W,X,T<:SignedHW} = $f(v, Vec(i))
        @inline $f(v::SignedHW, i::MM{W,X,T}) where {W,X,T<:UnsignedHW} = $f(v, Vec(i))
        @inline $f(v::UnsignedHW, i::MM{W,X,T}) where {W,X,T<:UnsignedHW} = $f(v, Vec(i))
        @inline $f(v::IntegerTypesHW, i::MM{W,X,T}) where {W,X,T<:StaticInt} = $f(v, Vec(i))

        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:SignedHW,T2<:SignedHW} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:UnsignedHW,T2<:SignedHW} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:SignedHW,T2<:UnsignedHW} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:UnsignedHW,T2<:UnsignedHW} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:StaticInt,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:StaticInt} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:StaticInt,T2<:StaticInt} = $f(Vec(i), Vec(j))

        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:SignedHW,T2<:SignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:UnsignedHW,T2<:SignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:SignedHW,T2<:UnsignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:UnsignedHW,T2<:UnsignedHW} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:StaticInt,T2<:IntegerTypes} = $f(Vec(i), v)

        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:SignedHW,T2<:SignedHW} = $f(v, Vec(i))
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:UnsignedHW,T2<:SignedHW} = $f(v, Vec(i))
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:SignedHW,T2<:UnsignedHW} = $f(v, Vec(i))
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:UnsignedHW,T2<:UnsignedHW} = $f(v, Vec(i))
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:StaticInt} = $f(v, Vec(i))
    end
end
# for f ∈ [:vand, :vor, :vxor, :vlt, :vle, :vgt, :vge, :veq, :vne, :vmin, :vmax, :vcopysign]
for f ∈ [:vand, :vor, :vxor, :veq, :vne, :vmin, :vmin_fast, :vmax, :vmax_fast, :vcopysign]
    @eval begin
        @inline $f(i::MM{W,X,T}, v::IntegerTypes) where {W,X,T<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::IntegerTypes, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:IntegerTypes,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:IntegerTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
    end
end
for f ∈ [ :vdiv, :vrem ]
    @eval begin
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:SignedHW,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:SignedHW,T2<:IntegerTypes} = $f(v, Vec(i))        
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:SignedHW,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:UnsignedHW,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:UnsignedHW,T2<:IntegerTypes} = $f(v, Vec(i))        
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:UnsignedHW,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
    end
end
for f ∈ [:vlt, :vle, :vgt, :vge, :veq, :vne, :vmin, :vmax, :vcopysign]
    @eval begin
        # left floating
        @inline $f(i::MM{W,X,T}, v::IntegerTypes) where {W,X,T<:FloatingTypes} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:FloatingTypes,T2<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:FloatingTypes,T2<:IntegerTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:FloatingTypes,T2<:IntegerTypes} = $f(Vec(i), Vec(j))
        # right floating
        @inline $f(i::MM{W,X,T}, v::FloatingTypes) where {W,X,T<:IntegerTypes} = $f(Vec(i), v)
        @inline $f(v::IntegerTypes, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:IntegerTypes,T2<:FloatingTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:IntegerTypes,T2<:FloatingTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:IntegerTypes,T2<:FloatingTypes} = $f(Vec(i), Vec(j))
        # both floating
        @inline $f(i::MM{W,X,T}, v::FloatingTypes) where {W,X,T<:FloatingTypes} = $f(Vec(i), v)
        @inline $f(i::MM{W,X,T1}, v::AbstractSIMDVector{W,T2}) where {W,X,T1<:FloatingTypes,T2<:FloatingTypes} = $f(Vec(i), v)
        @inline $f(v::AbstractSIMDVector{W,T1}, i::MM{W,X,T2}) where {W,X,T1<:FloatingTypes,T2<:FloatingTypes} = $f(v, Vec(i))
        @inline $f(i::MM{W,X1,T1}, j::MM{W,X2,T2}) where {W,X1,X2,T1<:FloatingTypes,T2<:FloatingTypes} = $f(Vec(i), Vec(j))
    end
    if f === :copysign
        @eval begin
            @inline $f(v::Float32, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline $f(v::Float32, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
            @inline $f(v::Float64, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline $f(v::Float64, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        end
    else
        @eval begin
            @inline $f(v::FloatingTypes, i::MM{W,X,T}) where {W,X,T<:IntegerTypes} = $f(v, Vec(i))
            @inline $f(v::FloatingTypes, i::MM{W,X,T}) where {W,X,T<:FloatingTypes} = $f(v, Vec(i))
        end
    end
end

@inline vadd_fast(i::MM{W,Zero}, j::MM{W,Zero}) where {W} = vrange(Val{W}(), Int, Val{0}(), Val{2}())
@inline vadd(i::MM{W,Zero}, j::MM{W,Zero}) where {W} = vrange(Val{W}(), Int, Val{0}(), Val{2}())


