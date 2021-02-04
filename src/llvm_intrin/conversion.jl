function convert_func(op, T1, W1, T2, W2 = W1)
    typ1 = LLVM_TYPES[T1]
    typ2 = LLVM_TYPES[T2]
    vtyp1 = vtype(W1, typ1)
    vtyp2 = vtype(W2, typ2)
    instrs = """
    %res = $op $vtyp2 %0 to $vtyp1
    ret $vtyp1 %res
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W1,$T1}, Tuple{_Vec{$W2,$T2}}, data(v)))
    end
end
# For bitcasting between signed and unsigned integers (LLVM does not draw a distinction, but they're separate in Julia)
function identity_func(W, T1, T2)
    vtyp1 = vtype(W, LLVM_TYPES[T1])
    instrs = """
    ret $vtyp1 %0
    """
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T1}, Tuple{_Vec{$W,$T2}}, data(v)))
    end
end

### `vconvert(::Type{<:AbstractSIMDVector}, x)` methods
### These are the critical `vconvert` methods; scalar and `VecUnroll` are implemented with respect to them.
@generated function vconvert(::Type{Vec{W,F}}, v::Vec{W,T}) where {W,F<:Union{Float32,Float64},T<:IntegerTypesHW}
    convert_func(T <: Signed ? "sitofp" : "uitofp", F, W, T)
end

@generated function vconvert(::Type{Vec{W,T}}, v::Vec{W,F}) where {W,F<:Union{Float32,Float64},T<:IntegerTypesHW}
    convert_func(T <: Signed ? "fptosi" : "fptoui", T, W, F)
end
@generated function vconvert(::Type{Vec{W,T1}}, v::Vec{W,T2}) where {W,T1<:IntegerTypesHW,T2<:IntegerTypesHW}
    sz1 = sizeof(T1)::Int; sz2 = sizeof(T2)::Int
    if sz1 < sz2
        convert_func("trunc", T1, W, T2)
    elseif sz1 == sz2
        identity_func(W, T1, T2)
    else
        convert_func(((T1 <: Signed) && (T2 <: Signed)) ? "sext" : "zext", T1, W, T2)
    end
end
@generated function vconvert(::Type{Vec{W,Float32}}, v::Vec{W,Float64}) where {W}
    convert_func("fptrunc", Float32, W, Float64, W)
end
@generated function vconvert(::Type{Vec{W,Float64}}, v::Vec{W,Float32}) where {W}
    convert_func("fpext", Float64, W, Float32, W)
end
@inline vconvert(::Type{<:Mask{W}}, v::Vec{W,Bool}) where {W} = tomask(v)
# @generated function vconvert(::Type{<:Mask{W}}, v::Vec{W,Bool}) where {W}
#     instrs = String[]
#     push!(instrs, "%m = trunc <$W x i8> %0 to <$W x i1>")
#     zext_mask!(instrs, 'm', W, '0')
#     push!(instrs, "ret i$(max(8,W)) %res.0")
#     U = mask_type_symbol(W);
#     quote
#         $(Expr(:meta,:inline))
#         Mask{$W}(llvmcall($(join(instrs, "\n")), $U, Tuple{_Vec{$W,Bool}}, data(v)))
#     end
# end
@inline vconvert(::Type{Vec{W,Bit}}, v::Vec{W,Bool}) where {W,Bool} = vconvert(Mask{W}, v)

@inline vconvert(::Type{Vec{W,T}}, v::Vec{W,T}) where {W,T<:IntegerTypesHW} = v
@inline vconvert(::Type{Vec{W,T}}, v::Vec{W,T}) where {W,T} = v
@inline vconvert(::Type{Vec{W,T}}, s::NativeTypes) where {W,T} = vbroadcast(Val{W}(), T(s))
@inline vconvert(::Type{Vec{W,T}}, s::IntegerTypesHW) where {W,T<:IntegerTypesHW} = _vbroadcast(StaticInt{W}(), s % T, StaticInt{W}() * static_sizeof(T))
@inline vconvert(::Type{V}, u::VecUnroll) where {V<:AbstractSIMDVector} = VecUnroll(fmap(vconvert, V, u.data))
@inline vconvert(::Type{V}, u::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector} = u


@inline vconvert(::Type{<:AbstractSIMDVector{W,T}}, i::MM{W,X}) where {W,X,T} = vrangeincr(Val{W}(), T(data(i)), Val{0}(), Val{X}())
@inline vconvert(::Type{MM{W,X,T}}, i::MM{W,X}) where {W,X,T} = MM{W,X}(T(i.i))

@inline function vconvert(::Type{V}, v::Mask{W}) where {W, T <: Union{Base.HWReal,Bool}, V <: AbstractSIMDVector{W,T}}
    vifelse(v, one(T), zero(T))
end
@inline vconvert(::Type{V}, v::Mask{W}) where {W, V <: AbstractSIMDVector{W,Bit}} = v
@inline function vconvert(::Type{V}, v::Vec{W,Bool}) where {W, T <: Base.HWReal, V <: AbstractSIMDVector{W,T}}
    vifelse(v, one(T), zero(T))
end




### `vconvert(::Type{<:NativeTypes}, x)` methods. These forward to `vconvert(::Type{Vec{W,T}}, x)`
@inline vconvert(::Type{T}, s::T) where {T<:NativeTypes} = s
@inline vconvert(::Type{T}, s::T) where {T<:IntegerTypesHW} = s
@inline vconvert(::Type{T}, s::NativeTypes) where {T<:NativeTypes} = s
@inline vconvert(::Type{T}, s::IntegerTypesHW) where {T<:IntegerTypesHW} = s % T
@inline vconvert(::Type{T}, v::AbstractSIMD{W,T}) where {T<:NativeTypes,W} = v
@inline vconvert(::Type{T}, v::AbstractSIMD{W,S}) where {T<:NativeTypes,S,W} = vconvert(Vec{W,T}, v)

### `vconvert(::Type{<:VecUnroll}, x)` methods
@inline function vconvert(::Type{VecUnroll{N,W,T,V}}, s::NativeTypes) where {N,W,T,V}
    VecUnroll{N}(vconvert(V, s))
end
@inline function _vconvert(::Type{VecUnroll{N,W,T,V}}, v::AbstractSIMDVector{W}) where {N,W,T,V}
    VecUnroll{N}(vconvert(V, v))
end
@inline function vconvert(::Type{VecUnroll{N,W,T,V}}, v::VecUnroll{N,W}) where {N,W,T,V}
    VecUnroll(fmap(vconvert, V, v.data))
end
@inline vconvert(::Type{VecUnroll{N,W,T,V}}, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = v

# @inline vconvert(::Type{T}, v::T) where {T} = v


@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::Mask{L}) where {N,W,L}
    @assert N*W == L "Can't split a vector of length $L into $N pieces of length $W."
    t = Expr(:tuple, :(Mask{$W}(u)))
    s = 0
    for n ∈ 2:N
        push!(t.args, :(Mask{$W}(u >>> $(s += W))))
    end
    # This `vconvert` will dispatch to one of the following two `vconvert` methods
    Expr(:block, Expr(:meta,:inline), :(u = data(v)), t)
end
@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::AbstractSIMDVector{L}) where {N,W,L}
    @assert N*W == L "Can't split a vector of length $L into $N pieces of length $W."
    t = Expr(:tuple);
    j = 0
    for i ∈ 1:N
        val = Expr(:tuple)
        for w ∈ 1:W
            push!(val.args, j)
            j += 1
        end
        push!(t.args, :(shufflevector(v, Val{$val}())))
    end
    Expr(:block, Expr(:meta,:inline), t)
end
@generated function splitvectortotuple(::StaticInt{N}, ::StaticInt{W}, v::LazyMulAdd{M,O}) where {N,W,M,O}
    # LazyMulAdd{M,O}(splitvectortotuple(StaticInt{N}(), StaticInt{W}(), v.data))
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, :(LazyMulAdd{$M,$O}(splitdata[$n])))
    end
    Expr(:block, Expr(:meta,:inline), :(splitdata = splitvectortotuple(StaticInt{$N}(), StaticInt{$W}(), v.data)), t)
end

@generated function vconvert(::Type{VecUnroll{N, W, T, V}}, v::AbstractSIMDVector{L}) where {N, W, T, V, L}
    if W == L # _vconvert will dispatch to one of the two above
        Expr(:block, Expr(:meta,:inline), :(_vconvert(VecUnroll{$N,$W,$T,$V}, v)))
    else
        Expr(:block, Expr(:meta,:inline), :(vconvert(VecUnroll{$N,$W,$T,$V}, VecUnroll(splitvectortotuple(StaticInt{$(N+1)}(), StaticInt{$W}(), v)))))
    end
end

@inline Vec{W,T}(v::Vec{W,S}) where {W,T,S} = vconvert(Vec{W,T}, v)
@inline Vec{W,T}(v::S) where {W,T,S<:NativeTypes} = vconvert(Vec{W,T}, v)


@inline vsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % signed(T)
@inline vunsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % unsigned(T)

@generated function _vfloat(v::Vec{W,I}, ::StaticInt{RS}) where {W, I <: Integer, RS}
    ex = if 8W ≤ RS
        :(vconvert(Vec{$W,Float64}, v))
    else
        :(vconvert(Vec{$W,Float32}, v))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end
@inline vfloat(v::Vec{W,I}) where {W, I <: Integer} = _vfloat(v, register_size())
@inline vfloat(v::AbstractSIMD{W,T}) where {W,T <: Union{Float32,Float64}} = v
@inline vfloat(vu::VecUnroll) = VecUnroll(fmap(vfloat, vu.data))
# @inline vfloat(v::Vec{W,I}) where {W, I <: Union{UInt64, Int64}} = Vec{W,Float64}(v)


@inline vfloat_fast(v::AbstractSIMDVector{W,T}) where {W,T <: Union{Float32,Float64}} = v
@inline vfloat_fast(vu::VecUnroll{W,T}) where {W,T<:Union{Float32,Float64}} = vu
@inline vfloat_fast(vu::VecUnroll) = VecUnroll(fmap(vfloat_fast, vu.data))

@generated function __vfloat_fast(v::Vec{W,I}, ::StaticInt{RS}) where {W, I <: Integer, RS}
    arg = if (2W*sizeof(I) ≤ RS) || sizeof(I) ≤ 4
        :v
    elseif I <: Signed
        :(v % Int32)
    else
        :(v % UInt32)
    end
    ex = if 8W ≤ RS
        :(Vec{$W,Float64}($arg))
    else
        :(Vec{$W,Float32}($arg))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end
@inline _vfloat_fast(v, ::False) = __vfloat_fast(v, register_size())
@inline _vfloat_fast(v, ::True) = vfloat(v)

@inline vfloat_fast(v::Vec) = _vfloat_fast(v, has_feature(Val(:x86_64_avx512dq)))

@generated function vreinterpret(::Type{T1}, v::Vec{W2,T2}) where {W2, T1 <: NativeTypes, T2}
    W1 = W2 * sizeof(T2) ÷ sizeof(T1)
    Expr(:block, Expr(:meta, :inline), :(vreinterpret(Vec{$W1,$T1}, v)))
end
@inline vreinterpret(::Type{Vec{1,T1}}, v::Vec{W,T2}) where {W,T1,T2<:Base.BitInteger} = reinterpret(T1, fuseint(v))
@generated function vreinterpret(::Type{Vec{W1,T1}}, v::Vec{W2,T2}) where {W1, W2, T1, T2}
    @assert sizeof(T1) * W1 == W2 * sizeof(T2)
    convert_func("bitcast", T1, W1, T2, W2)
end

@inline vunsafe_trunc(::Type{I}, v::Vec{W,T}) where {W,I,T} = vconvert(Vec{W,I}, v)
@inline vrem(v::AbstractSIMDVector{W,T}, ::Type{I}) where {W,I,T} = vconvert(Vec{W,I}, v)
@inline vrem(v::AbstractSIMDVector{W,T}, ::Type{V}) where {W,I,T,V<:AbstractSIMD{W,I}} = vconvert(V, v)
@inline vrem(r::IntegerTypesHW, ::Type{V}) where {W, I, V <: AbstractSIMD{W,I}} = convert(V, r % I)
