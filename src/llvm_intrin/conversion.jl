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

@generated function Vec{W,F}(v::Vec{W,T}) where {W,F<:Union{Float32,Float64},T<:Integer}
    convert_func(T <: Signed ? "sitofp" : "uitofp", F, W, T)
end

@generated function Vec{W,T}(v::Vec{W,F}) where {W,F<:Union{Float32,Float64},T<:Integer}
    convert_func(T <: Signed ? "fptosi" : "fptoui", T, W, F)
end
@generated function Vec{W,T1}(v::Vec{W,T2}) where {W,T1<:Integer,T2<:Integer}
    sz1 = sizeof(T1)::Int; sz2 = sizeof(T2)::Int
    if sz1 < sz2
        convert_func("trunc", T1, W, T2)
    elseif sz1 == sz2
        identity_func(W, T1, T2)
    else
        convert_func(((T1 <: Signed) && (T2 <: Signed)) ? "sext" : "zext", T1, W, T2)
    end
end

@inline vsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % signed(T)
@inline vunsigned(v::AbstractSIMD{W,T}) where {W,T <: Base.BitInteger} = v % unsigned(T)

@inline vfloat(v::Vec{W,I}) where {W, I <: Union{UInt64, Int64}} = Vec{W,Float64}(v)
@generated function vfloat(v::Vec{W,I}) where {W, I <: Integer}
    ex = if 8W ≤ REGISTER_SIZE
        :(Vec{$W,Float64}(v))
    else
        :(Vec{$W,Float32}(v))
    end
    Expr(:block, Expr(:meta, :inline), ex)
end
@inline vfloat(v::AbstractSIMD{W,T}) where {W,T <: Union{Float32,Float64}} = v
@inline vfloat(vu::VecUnroll) = VecUnroll(fmap(float, vu.data))

@generated function Vec{W,Float32}(v::Vec{W,Float64}) where {W}
    convert_func("fptrunc", Float32, W, Float64, W)
end
@generated function Vec{W,Float64}(v::Vec{W,Float32}) where {W}
    convert_func("fpext", Float64, W, Float32, W)
end

@inline vconvert(::Type{Vec{W,T}}, s::T) where {W, T <: NativeTypes} = vbroadcast(Val{W}(), s)
@inline vconvert(::Type{Vec{W,T}}, s::T) where {W, T <: Integer} = vbroadcast(Val{W}(), s)
@inline vconvert(::Type{Vec{W,T}}, s::NativeTypes) where {W, T} = vbroadcast(Val{W}(), convert(T, s))
@inline vconvert(::Type{Vec{W,T}}, s::IntegerTypesHW) where {W, T <: IntegerTypesHW} = vbroadcast(Val{W}(), s % T)
@generated function vconvert(::Type{T}, v::Vec{W,S}) where {T<:Number,S,W}
    if S <: T
        Expr(:block, Expr(:meta,:inline), :v)
    else
        Expr(:block, Expr(:meta,:inline), :(Vec{$W,$T}(v)))
    end
end
@inline function vconvert(::Type{U}, v::Vec{W,S}) where {N,W,T,U<:VecUnroll{N,W,T},S}
    VecUnroll{N}(vconvert(T, v))
end
@inline vconvert(::Type{Vec{W,T}}, v::Vec{W,S}) where {T<:Number,S,W} = Vec{W,T}(v)
@inline vconvert(::Type{Vec{W,T}}, v::Vec{W,T}) where {T<:Number,W} = v

@inline vconvert(::Type{T}, v::Union{Mask,VecUnroll{<:Any, <:Any, Bool, <: Mask}}) where {T <: Union{Base.HWReal,Bool}} = ifelse(v, one(T), zero(T))
@inline vconvert(::Type{<:AbstractSIMD{W,T}}, v::Union{Mask{W},VecUnroll{<:Any, W, Bool, <: Mask}}) where {W, T <: Union{Base.HWReal,Bool}} = ifelse(v, one(T), zero(T))
@inline vconvert(::Type{Bit}, v::Union{Mask,VecUnroll{<:Any, <:Any, Bool, <: Mask}}) = v
@inline vconvert(::Type{<:AbstractSIMD{W,Bit}}, v::Union{Mask{W},VecUnroll{<:Any, W, Bool, <: Mask}}) where {W} = v
# @inline vconvert(::typeof{T}, v::T)

@generated function vreinterpret(::Type{T1}, v::Vec{W2,T2}) where {W2, T1 <: NativeTypes, T2}
    W1 = W2 * sizeof(T2) ÷ sizeof(T1)
    Expr(:block, Expr(:meta, :inline), :(vreinterpret(Vec{$W1,$T1}, v)))
end
@generated function vreinterpret(::Type{Vec{W1,T1}}, v::Vec{W2,T2}) where {W1, W2, T1, T2}
    @assert sizeof(T1) * W1 == W2 * sizeof(T2)
    convert_func("bitcast", T1, W1, T2, W2)
end



@inline Base.unsafe_trunc(::Type{I}, v::Vec{W,T}) where {W,I,T} = vconvert(Vec{W,I}, v)
@inline Base.:(%)(v::AbstractSIMDVector{W,T}, ::Type{I}) where {W,I,T} = vconvert(Vec{W,I}, v)
@inline Base.:(%)(v::AbstractSIMDVector{W,T}, ::Type{V}) where {W,I,T,V<:AbstractSIMD{W,I}} = vconvert(V, v)
@inline Base.:(%)(r::Integer, ::Type{V}) where {W, I, V <: AbstractSIMD{W,I}} = vbroadcast(Val{W}(), r % I)

