@generated function vrange(::Val{W}, ::Type{T}) where {W,T}
    Expr(:block, Expr(:meta, :inline), Expr(:call, :Vec, Expr(:tuple, [Expr(:call, :(Core.VecElement), T(w)) for w ∈ 0:W-1]...)))
end
@generated function vrangeincr(::Val{W}, i::I, ::Val{O}) where {W,I<:Integer,O}
    bytes = I === Int ? min(8, VectorizationBase.prevpow2(VectorizationBase.REGISTER_SIZE ÷ W)) : sizeof(I)
    # bytes = min(8, VectorizationBase.prevpow2(VectorizationBase.REGISTER_SIZE ÷ W))
    bits = 8bytes
    jtypesym = Symbol(:Int, bits)
    iexpr = bytes == sizeof(I) ? :i : Expr(:call, :%, :i, jtypesym)
    typ = "i$(bits)"
    vtyp = "<$W x $typ>"
    rangevec = join(("$typ $(w+O)" for w ∈ 0:W-1), ", ")
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "%res = add nsw $vtyp %v, <$rangevec>")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        llvmcall(
            $(join(instrs,"\n")), Vec{$W,$jtypesym}, Tuple{$jtypesym}, $iexpr
        )
    end
end
@generated function vrangeincr(::Val{W}, i::T, ::Val{O}) where {W,T<:FloatingTypes,O}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    rangevec = join(("$typ $(w+O).0" for w ∈ 0:W-1), ", ")
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "%res = fadd $vtyp %v, <$rangevec>")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        llvmcall(
            $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, i
        )
    end
end
@generated function vrangemul(::Val{W}, i::I, ::Val{O}) where {W,I<:Integer,O}
    bytes = I === Int ? min(8, VectorizationBase.prevpow2(VectorizationBase.REGISTER_SIZE ÷ W)) : sizeof(I)
    bits = 8bytes
    jtypesym = Symbol(:Int, bits)
    iexpr = bytes == sizeof(I) ? :i : Expr(:call, :%, :i, jtypesym)
    typ = "i$(bits)"
    vtyp = "<$W x $typ>"
    rangevec = join(("$typ $(w+O)" for w ∈ 0:W-1), ", ")
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "%res = mul nsw $vtyp %v, <$rangevec>")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        llvmcall(
            $(join(instrs,"\n")), Vec{$W,$jtypesym}, Tuple{$jtypesym}, $iexpr
        )
    end
end
@generated function vrangemul(::Val{W}, i::T, ::Val{O}) where {W,T<:FloatingTypes,O}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    rangevec = join(("$typ $(w+O).0" for w ∈ 0:W-1), ", ")
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp undef, $typ %0, i32 0")
    push!(instrs, "%v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer")
    push!(instrs, "%res = fmul fast $vtyp %v, <$rangevec>")
    push!(instrs, "ret $vtyp %res")
    quote
        $(Expr(:meta,:inline))
        llvmcall(
            $(join(instrs,"\n")), Vec{$W,$T}, Tuple{$T}, i
        )
    end
end

@inline svrangeincr(::Val{W}, i, ::Val{O}) where {W,O} = Vec(vrangeincr(Val{W}(), i, Val{O}()))
@inline svrangemul(::Val{W}, i, ::Val{O}) where {W,O} = Vec(vrangemul(Val{W}(), i, Val{O}()))


@inline vrange(i::MM{W}) where {W} = vrangeincr(Val{W}(), i.i, Val{0}())
@inline svrange(i::MM{W}) where {W} = Vec(vrangeincr(Val{W}(), i.i, Val{0}()))
@inline Base.:(+)(i::MM{W}, j::MM{W}) where {W} = Vec(vadd(vrange(i), vrange(j)))
@inline Base.:(+)(i::MM{W}, j::Vec{W}) where {W} = vadd(vrange(i), j)
@inline Base.:(+)(i::Vec{W}, j::MM{W}) where {W} = vadd(i, vrange(j))
@inline Base.:(*)(i::MM{W}, j::Vec{W}) where {W} = vmul(vrange(i), j)
@inline Base.:(*)(i::Vec{W}, j::MM{W}) where {W} = vmul(i, vrange(j))
@inline Base.:(+)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = Vec(vadd(vrange(i), extract_data(j)))
@inline Base.:(+)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = Vec(vadd(extract_data(i), vrange(j)))
@inline Base.:(*)(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = Vec(vmul(vrange(i), extract_data(j)))
@inline Base.:(*)(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = Vec(vmul(extract_data(i), vrange(j)))
@inline vadd(i::MM{W}, j::MM{W}) where {W} = Vec(vadd(vrange(i), vrange(j)))
@inline vadd(i::MM{W}, j::Vec{W}) where {W} = vadd(vrange(i), j)
@inline vadd(i::Vec{W}, j::MM{W}) where {W} = vadd(i, vrange(j))
@inline vadd(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = Vec(vadd(vrange(i), extract_data(j)))
@inline vadd(i::AbstractSIMDVector{W}, j::MM{W}) where {W} = Vec(vadd(extract_data(i), vrange(j)))
@inline vmul(i::MM{W}, j::Vec{W}) where {W} = vmul(vrange(i), j)
@inline vmul(i::MM{W}, j::AbstractSIMDVector{W}) where {W} = Vec(vmul(vrange(i), extract_data(j)))
@inline vmul(j::Vec{W}, i::MM{W}) where {W} = vmul(j, vrange(i))
@inline vmul(j::AbstractSIMDVector{W}, i::MM{W}) where {W} = Vec(vmul(extract_data(j), vrange(i)))
@inline Base.:(/)(i::MM, j::T) where {T<:Number} = Vec(vfdiv(vrange(i,T), j))
@inline Base.:(/)(j::T, i::MM) where {T<:Number} = Vec(vfdiv(j, vrange(i,T)))
@inline Base.:(/)(i::MM, j::Vec{W,T}) where {W,T<:Number} = Vec(vfdiv(vrange(i,T), j))
@inline Base.:(/)(j::Vec{W,T}, i::MM) where {W,T<:Number} = Vec(vfdiv(j, vrange(i,T)))
@inline Base.:(/)(i::MM, j::MM) = Vec(vfdiv(vrange(i), vrange(j)))
@inline Base.inv(i::MM) = inv(svrange(i))


@inline vrange(::Val{W}) where {W} = vrange(Val{W}(), Float64)
@inline svrange(::Val{W}) where {W} = svrange(Val{W}(), Float64)

@inline vrange(i::MM{W}, ::Type{T}) where {W,T} = vrangeincr(Val{W}(), T(i.i), Val{0}())
@inline vrange(i::MM{W}, ::Type{T}) where {W,T <: Integer} = vrangeincr(Val{W}(), i.i % T, Val{0}())
@inline svrange(i::MM, ::Type{T}) where {T} = Vec(vrange(i, T))


@inline Base.:(<<)(i::MM, j::Integer) = svrange(i) << j
@inline Base.:(>>)(i::MM, j::Integer) = svrange(i) >> j
@inline Base.:(>>>)(i::MM, j::Integer) = svrange(i) >>> j

@inline Base.:(*)(i::MM{W}, j::T) where {W,T} = vmul(svrange(i), j)
@inline Base.:(*)(j::T, i::MM{W}) where {W,T} = vmul(svrange(i), j)
@inline vmul(i::MM{W}, j::T) where {W,T} = vmul(svrange(i), j)
@inline vmul(j::T, i::MM{W}) where {W,T} = vmul(svrange(i), j)
@inline vmul(i::MM{W}, ::Static{j}) where {W,j} = vmul(svrange(i), j)
@inline vmul(::Static{j}, i::MM{W}) where {W,j} = vmul(svrange(i), j)
@inline vconvert(::Type{Vec{W,T}}, i::MM{W}) where {W,T} = svrange(i, T)




@inline Base.:(-)(i::Integer, j::MM{W}) where {W} = vsub(i, svrange(j))
@inline Base.:(-)(::Static{i}, j::MM{W}) where {W,i} = vsub(i, svrange(j))
@inline Base.:(-)(i::MM{W}, j::MM{W}) where {W} = vsub(svrange(i), svrange(j))
@inline Base.:(-)(i::MM{W}) where {W} = -svrange(i)
@inline vsub(i::Integer, j::MM{W}) where {W} = vsub(i, svrange(j))
@inline vsub(::Static{i}, j::MM{W}) where {W,i} = vsub(i, svrange(j))
@inline vsub(i::MM{W}, j::MM{W}) where {W} = vsub(svrange(i), svrange(j))
@inline vsub(i::MM{W}) where {W} = -svrange(i)


for op ∈ [:(<), :(>), :(≥), :(≤), :(==), :(!=), :(&), :(|), :(⊻), :(%)]
    @eval @inline Base.$op(i::MM, j::Integer) = $op(svrange(i), j)
    @eval @inline Base.$op(i::Integer, j::MM) = $op(i, svrange(j))
    @eval @inline Base.$op(i::MM, ::Static{j}) where {j} = $op(svrange(i), j)
    @eval @inline Base.$op(::Static{i}, j::MM) where {i} = $op(i, svrange(j))
    @eval @inline Base.$op(i::MM, j::MM) = $op(svrange(i), svrange(j))
end
@inline Base.:(*)(i::MM, j::MM) = Vec(vmul(vrange(i), vrange(j)))
@inline vmul(i::MM, j::MM) = Vec(vmul(vrange(i), vrange(j)))


using VectorizationBase: Static, Zero, One
@inline vadd(::MM{W,Zero}, v::AbstractSIMDVector{W,T}) where {W,T} = vadd(vrange(Val{W}(), T), v)
@inline vadd(v::AbstractSIMDVector{W,T}, ::MM{W,Zero}) where {W,T} = vadd(vrange(Val{W}(), T), v)
@inline vadd(::MM{W,Zero}, ::MM{W,Zero}) where {W} = vrangemul(Val{W}(), 2, Val{0}())
# @inline vmul(::MM{W,Zero}, i) where {W} = svrangemul(Val{W}(), i, Val{0}())
# @inline vmul(i, ::MM{W,Zero}) where {W} = svrangemul(Val{W}(), i, Val{0}())

@inline vmul(::MM{W,Static{N}}, i) where {W,N} = svrangemul(Val{W}(), i, Val{N}())
@inline vmul(i, ::MM{W,Static{N}}) where {W,N} = svrangemul(Val{W}(), i, Val{N}())

