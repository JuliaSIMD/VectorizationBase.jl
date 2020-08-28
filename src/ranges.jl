

function pick_integer_bytes(W, preferred)
    # SIMD quadword integer support requires AVX512DQ
    if !AVX512DQ
        preferred = min(4, preferred)
    end
    min(preferred, prevpow2(REGISTER_SIZE ÷ W))
end
function pick_integer(W, pref)
    bytes = pick_integer_bytes(W, pref)
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
    bytes = pick_integer_bytes(W, sizeof(T))
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


@inline Vec(i::MM{W}) where {W} = vrangeincr(Val{W}(), data(i), Val{0}(), Val{1}())
@inline Vec(i::MM{W,Static{N}}) where {W,N} = vrange(Val{W}(), Val{N}(), Val{1}())
@inline Vec(i::MM{1}) = data(i)
@inline Vec(i::MM{1,Static{N}}) where {N} = N
@inline Base.convert(::Type{Vec{W,T}}, i::MM{W}) where {W,T} = vrange(Val{W}(), T, Val{0}(), Val{1}())

# Addition
@inline Base.:(+)(i::MM{W}, j::MM{W}) where {W} = vadd(vrange(i), vrange(j))
@inline Base.:(+)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd(Vec(i), j)
@inline Base.:(+)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd(i, Vec(j))
@inline vadd(i::MM{W}, j::MM{W}) where {W} = vadd(vrange(i), vrange(j))
@inline vadd(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vadd(Vec(i), j)
@inline vadd(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vadd(i, Vec(j))
# Subtraction
@inline Base.:(-)(i::MM{W}, j::MM{W}) where {W} = vsub(vrange(i), vrange(j))
@inline Base.:(-)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub(Vec(i), j)
@inline Base.:(-)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub(i, Vec(j))
@inline vsub(i::MM{W}, j::MM{W}) where {W} = vsub(vrange(i), vrange(j))
@inline vsub(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vsub(Vec(i), j)
@inline vsub(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vsub(i, Vec(j))
# Multiplication
@inline Base.:(*)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vmul(Vec(i), j)
@inline Base.:(*)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vmul(i, Vec(j))
@inline Base.:(*)(i::MM{W}, j::MM{W}) where {W} = vmul(Vec(i), Vec(j))
@inline vmul(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = vmul(Vec(i), j)
@inline vmul(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = vmul(i, Vec(j))
@inline vmul(i::MM{W}, j::MM{W}) where {W} = vmul(Vec(i), Vec(j))


# Multiplication without promotion
@inline vmul_no_promote(a, b) = vmul(a, b)
@inline vmul_no_promote(a::MM{W}, b) where {W} = MM{W}(vmul(a.i, b))
@inline vmul_no_promote(a, b::MM{W}) where {W} = MM{W}(vmul(a, b.i))
@inline vmul_no_promote(a::MM{W}, b::MM{W}) where {W} = vmul(a, b) # must promote

# Division
@generated function floattype(::Val{W}) where {W}
    (REGISTER_SIZE ÷ W) ≥ 8 ? :Float64 : :Float32
end
floatvec(i::MM{W}) where {W} = Vec(MM{W}(floattype(Val{W}())(i.i)))
@inline Base.:(/)(i::MM, j::T) where {T<:Number} = floatvec(i) / j
@inline Base.:(/)(j::T, i::MM) where {T<:Number} = j / floatvec(i)
@inline Base.:(/)(i::MM, j::MM) = floatvec(i) / floatvec(j)
@inline Base.inv(i::MM{W}) = inv(floatvec(i))

@inline Base.:(<<)(i::MM, j::Number) = Vec(i) << j
@inline Base.:(>>)(i::MM, j::Number) = Vec(i) >> j
@inline Base.:(>>>)(i::MM, j::Number) = Vec(i) >>> j

for op ∈ [:(<), :(>), :(≥), :(≤), :(==), :(!=), :(&), :(|), :(⊻), :(%)]
    @eval @inline Base.$op(i::MM, j::Number) = $op(Vec(i), j)
    @eval @inline Base.$op(i::Number, j::MM) = $op(i, Vec(j))
    @eval @inline Base.$op(i::MM, ::Static{j}) where {j} = $op(Vec(i), j)
    @eval @inline Base.$op(::Static{i}, j::MM) where {i} = $op(i, Vec(j))
    @eval @inline Base.$op(i::MM, j::MM) = $op(Vec(i), Vec(j))
end

@inline vadd(::MM{W,Zero}, v::AbstractSIMDVector{W,T}) where {W,T} = vadd(vrange(Val{W}(), T, Val{0}(), Val{1}()), v)
@inline vadd(v::AbstractSIMDVector{W,T}, ::MM{W,Zero}) where {W,T} = vadd(vrange(Val{W}(), T, Val{0}(), Val{1}()), v)
@inline vadd(i::MM{W,Zero}, j::MM{W,Zero}) where {W} = vrange(Val{W}(), Int, Val{0}(), Val{2}())
# @inline vadd(a::MM, ::Zero) = a
# @inline vadd(::Zero, a::MM) = a
# @inline Base.:(+)(a::MM, ::Zero) = a
# @inline Base.:(+)(::Zero, a::MM) = a
# # @inline vmul(::MM{W,Zero}, i) where {W} = svrangemul(Val{W}(), i, Val{0}())
# @inline vmul(i, ::MM{W,Zero}) where {W} = svrangemul(Val{W}(), i, Val{0}())

@inline vmul(::MM{W,Static{N}}, i) where {W,N} = svrangemul(Val{W}(), i, Val{N}())
@inline vmul(i, ::MM{W,Static{N}}) where {W,N} = svrangemul(Val{W}(), i, Val{N}())

