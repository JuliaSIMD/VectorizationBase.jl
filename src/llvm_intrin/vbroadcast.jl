
@inline vzero(::Val{1}, ::Type{T}) where {T<:NativeTypes} = zero(T)
@inline vzero(::StaticInt{1}, ::Type{T}) where {T<:NativeTypes} = zero(T)
@generated function _vzero(::StaticInt{W}, ::Type{T}, ::StaticInt{RS}) where {W,T<:NativeTypes,RS}
    # isone(W) && return Expr(:block, Expr(:meta,:inline), Expr(:call, :zero, T))
    if W * sizeof(T) > RS
        d, r1 = divrem(sizeof(T) * W, RS)
        Wnew, r2 = divrem(W, d)
        (iszero(r1) & iszero(r2)) || throw(ArgumentError("If broadcasting to greater than 1 vector length, should make it an integer multiple of the number of vectors."))
        t = Expr(:tuple)
        for i ∈ 1:d
            push!(t.args, :v)
        end
        return Expr(:block, Expr(:meta,:inline), :(v = vzero(StaticInt{$Wnew}(), $T)), :(VecUnroll($t)))
    end
    typ = LLVM_TYPES[T]
    instrs = "ret <$W x $typ> zeroinitializer"
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{}))
    end
end
@generated function _vbroadcast(::StaticInt{W}, s::_T, ::StaticInt{RS}) where {W,_T<:NativeTypes,RS}
    isone(W) && return :s
    if _T <: Integer && sizeof(_T) * W > RS
        intbytes = max(4, RS ÷ W)
        T = integer_of_bytes(intbytes)
        if _T <: Unsigned
            T = unsigned(T)
        end
        # ssym = :(s % $T)
        ssym = :(convert($T, s))
    elseif sizeof(_T) * W > RS
        d, r1 = divrem(sizeof(_T) * W, RS)
        Wnew, r2 = divrem(W, d)
        (iszero(r1) & iszero(r2)) || throw(ArgumentError("If broadcasting to greater than 1 vector length, should make it an integer multiple of the number of vectors."))
        t = Expr(:tuple)
        for i ∈ 1:d
            push!(t.args, :v)
        end
        return Expr(:block, Expr(:meta,:inline), :(v = vbroadcast(StaticInt{$Wnew}(), s)), :(VecUnroll($t)))
    else
        T = _T
        ssym = :s
    end
    typ = LLVM_TYPES[T]
    vtyp = vtype(W, typ)
    instrs = """
        %ie = insertelement $vtyp undef, $typ %0, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        ret $vtyp %v
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{$T}, $ssym))
    end
end
@inline vzero(::Union{Val{W},StaticInt{W}}, ::Type{T}) where {W,T} = _vzero(StaticInt{W}(), T, register_size(T))
@inline vbroadcast(::Union{Val{W},StaticInt{W}}, s::T) where {W,T} = _vbroadcast(StaticInt{W}(), s, register_size(T))

@generated function vbroadcast(::Union{Val{W},StaticInt{W}}, ptr::Ptr{T}) where {W, T}
    isone(W) && return Expr(:block, Expr(:meta, :inline), :(vload(ptr)))
    typ = LLVM_TYPES[T]
    ptyp = JULIAPOINTERTYPE
    vtyp = "<$W x $typ>"
    alignment = Base.datatype_alignment(T)
    instrs = """
        %ptr = inttoptr $ptyp %0 to $typ*
        %res = load $typ, $typ* %ptr, align $alignment
        %ie = insertelement $vtyp undef, $typ %res, i32 0
        %v = shufflevector $vtyp %ie, $vtyp undef, <$W x i32> zeroinitializer
        ret $vtyp %v
    """
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $instrs, _Vec{$W,$T}, Tuple{Ptr{$T}}, ptr ))
    end
end

@inline vbroadcast(::Union{Val{W},StaticInt{W}}, v::AbstractSIMDVector{W}) where {W} = v

@generated function vbroadcast(::Union{Val{W},StaticInt{W}}, v::V) where {W,L,T,V<:AbstractSIMDVector{L,T}}
    N, r = divrem(L, W)
    @assert iszero(r)
    V = if T === Bit
        :(Mask{$W,$(mask_type_symbol(W))})
    else
        :(Vec{$W,$T})
    end
    Expr(:block, Expr(:meta,:inline), :(vconvert(VecUnroll{$(N-1),$W,$T,$V}, v)))
end

@inline Vec{W,T}(v::Vec{W,T}) where {W,T} = v

@inline Base.zero(::Type{Vec{W,T}}) where {W,T} = vzero(Val{W}(), T)
@inline Base.zero(::Vec{W,T}) where {W,T} = zero(Vec{W,T})
@inline Base.one(::Vec{W,T}) where {W,T} = vbroadcast(Val{W}(), one(T))

@inline Base.one(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Val{W}(), one(T))
@inline Base.oneunit(::Type{Vec{W,T}}) where {W,T} = vbroadcast(Val{W}(), one(T))
@inline vzero(::Type{T}) where {T<:Number} = zero(T)
@inline vzero() = vzero(pick_vector_width(Float64), Float64)

@inline Vec{W,T}(s::Real) where {W,T} = vbroadcast(Val{W}(), T(s))
@inline Vec{W}(s::T) where {W,T<:NativeTypes} = vbroadcast(Val{W}(), s)
@inline Vec(s::T) where {T<:NativeTypes} = vbroadcast(pick_vector_width(T), s)

@generated function Base.zero(::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V}
    t = Expr(:tuple); foreach(_ -> push!(t.args, :(zero(Vec{$W,$T}))), 0:N)
    Expr(:block, Expr(:meta, :inline), :(VecUnroll($t)))
end
@inline Base.zero(::VecUnroll{N,W,T,V}) where {N,W,T,V} = zero(VecUnroll{N,W,T,V})

@generated function VecUnroll{N,W,T,V}(x::S) where {N,W,T,V<:AbstractSIMDVector{W,T},S<:Real}
    t = Expr(:tuple)
    for n ∈ 0:N
        push!(t.args, :(convert($V, x)))
    end
    Expr(:block, Expr(:meta,:inline), :(VecUnroll($t)))
end
@inline VecUnroll{N,W,T}(x::NativeTypesV) where {N,W,T} = VecUnroll{N,W,T,Vec{W,T}}(x)
@inline VecUnroll{N}(x::V) where {N,W,T,V <: AbstractSIMDVector{W,T}} = VecUnroll{N,W,T,V}(x)


