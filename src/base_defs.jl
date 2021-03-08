
const FASTDICT = Dict{Symbol,Expr}([
    :(+) => :(Base.FastMath.add_fast),
    :(-) => :(Base.FastMath.sub_fast),
    :(*) => :(Base.FastMath.mul_fast),
    :(/) => :(Base.FastMath.div_fast),
    :(÷) => :(VectorizationBase.vdiv_fast), # VectorizationBase.vdiv == integer, VectorizationBase.vfdiv == float
    :(%) => :(Base.FastMath.rem_fast),
    :abs2 => :(Base.FastMath.abs2_fast),
    # :inv => :(Base.FastMath.inv_fast), # this is slower in most benchmarks
    :hypot => :(Base.FastMath.hypot_fast),
    :max => :(Base.FastMath.max_fast),
    :min => :(Base.FastMath.min_fast),
    :muladd => :(VectorizationBase.vmuladd_fast),
    :fma => :(VectorizationBase.vfma_fast),
    :vfmadd => :(VectorizationBase.vfmadd_fast),
    :vfnmadd => :(VectorizationBase.vfnmadd_fast),
    :vfmsub => :(VectorizationBase.vfmsub_fast),
    :vfnmsub => :(VectorizationBase.vfnmsub_fast),
    :log => :(Base.FastMath.log_fast),
    :log2 => :(Base.FastMath.log2_fast),
    :log10 => :(Base.FastMath.log10_fast)
])

for (op,f) ∈ [
    (:(Base.:-),:vsub),
    (:(Base.FastMath.sub_fast), :vsub_fast),
    # (:(Base.FastMath.abs2_fast),:vabs2_fast),
    (:(Base.inv),:vinv),
    (:(Base.FastMath.inv_fast),:vinv_fast),
    (:(Base.abs),:vabs),
    (:(Base.round),:vround),
    (:(Base.floor),:vfloor),
    (:(Base.ceil),:vceil),
    (:(Base.trunc),:vtrunc),
    (:(Base.unsafe_trunc),:vtrunc),
    (:(Base.signed),:vsigned),
    (:(Base.unsigned),:vunsigned),
    (:(Base.float),:vfloat),
    (:(Base.sqrt),:vsqrt),
    (:(Base.leading_zeros),:vleading_zeros),
    (:(Base.trailing_zeros),:vtrailing_zeros),
    (:(Base.count_ones),:vcount_ones),
]
    @eval begin
        @inline $op(a::AbstractSIMD) = $f(a)
    end
end
@inline Base.:(~)(v::AbstractSIMD{W,T}) where {W,T<:IntegerTypesHW} = v ⊻ vbroadcast(Val(W), -1 % T)
@inline Base.FastMath.abs2_fast(v::AbstractSIMD) = vmul_fast(v,v)

@inline no_promote(a,b) = (a,b)
for (op, f, promote) ∈ [
    (:(Base.:+),:vadd,:promote),
    (:(Base.FastMath.add_fast), :vadd_fast,:promote),
    (:(Base.:-),:vsub,:promote),
    (:(Base.FastMath.sub_fast), :vsub_fast,:promote),
    (:(Base.:*),:vmul,:promote),
    (:(Base.FastMath.mul_fast), :vmul_fast,:promote),
    (:(Base.:/),:vfdiv,:promote_div),
    (:(Base.FastMath.div_fast),:vfdiv_fast,:promote_div),
    (:(Base.:%),:vrem,:promote_div),
    (:(Base.FastMath.rem_fast),:vrem_fast,:promote_div),
    (:(Base.:÷),:vdiv,:promote_div),
    (:(Base.:<<),:vshl,:promote_div),
    (:(Base.:>>),:vashr,:promote_div),
    (:(Base.:>>>),:vlshr,:promote_div),
    (:(Base.:&),:vand,:promote),
    (:(Base.:|),:vor,:promote),
    (:(Base.:⊻),:vxor,:promote),
    (:(Base.max),:vmax,:no_promote),
    (:(Base.min),:vmin,:no_promote),
    (:(Base.FastMath.max_fast),:vmax_fast,:no_promote),
    (:(Base.FastMath.min_fast),:vmin_fast,:no_promote),
    # (:(Base.copysign),:vcopysign,:no_promote),
    (:(Base.:(==)), :veq, :no_promote),
    (:(Base.:(≠)), :vne, :no_promote),
    (:(Base.:(>)), :vgt, :no_promote),
    (:(Base.:(≥)), :vge, :no_promote),
    (:(Base.:(<)), :vlt, :no_promote),
    (:(Base.:(≤)), :vle, :no_promote),
]
    @eval begin
        # @inline $op(a::AbstractSIMD,b::AbstractSIMD) = ((c,d) = $promote(a,b); $f(c,d))
        @inline $op(a::AbstractSIMD,b::AbstractSIMD) = ((c,d) = $promote(a,b); $f(c,d))
        @inline $op(a::NativeTypes,b::AbstractSIMD) = ((c,d) = $promote(a,b); $f(c,d))
        @inline $op(a::AbstractSIMD,b::NativeTypes) = ((c,d) = $promote(a,b); $f(c,d))
    end
end
for op ∈ [:(Base.:(*)), :(Base.FastMath.mul_fast)]
    @eval begin
        @inline $op(m::AbstractSIMD{W,B1}, v::AbstractSIMD{W,B2}) where {W,B1<:Union{Bool,Bit},B2<:Union{Bool,Bit}} = m & v
        @inline $op(m::AbstractSIMD{W,B}, v::AbstractSIMD{W}) where {W,B<:Union{Bool,Bit}} = ifelse(m, v, zero(v))
        @inline $op(v::AbstractSIMD{W}, m::AbstractSIMD{W,B}) where {W,B<:Union{Bool,Bit}} = ifelse(m, v, zero(v))
    end
end
# copysign needs a heavy hand to avoid ambiguities
@inline Base.copysign(a::VecUnroll,b::AbstractSIMDVector) = VecUnroll(fmap(vcopysign, getfield(a, :data), b))
@inline Base.copysign(a::VecUnroll,b::VecUnroll) = VecUnroll(fmap(vcopysign, getfield(a, :data), getfield(b, :data)))
@inline Base.copysign(a::AbstractSIMDVector,b::VecUnroll) = VecUnroll(fmap(vcopysign, a, getfield(b, :data)))
@inline Base.copysign(a::AbstractSIMDVector,b::AbstractSIMDVector) = vcopysign(a,b)
@inline Base.copysign(a::NativeTypes,b::VecUnroll{N,W}) where {N,W} = VecUnroll(fmap(vcopysign, vbroadcast(Val{W}(), a), getfield(b, :data)))
@inline Base.copysign(a::VecUnroll{N,W},b::Base.HWReal) where {N,W} = VecUnroll(fmap(vcopysign, getfield(a, :data), vbroadcast(Val{W}(), b)))
@inline Base.copysign(a::IntegerTypesHW,b::AbstractSIMDVector) = vcopysign(a,b)
@inline Base.copysign(a::AbstractSIMDVector,b::Base.HWReal) = vcopysign(a,b)
for T ∈ [:Rational, :SignedHW, :Float32, :Float64]
    @eval begin
        @inline function Base.copysign(a::$T, b::AbstractSIMDVector{W,T}) where {W,T <: Union{Float32,Float64,SignedHW}}
            v1, v2 = promote(a, b)
            vcopysign(v1, v2)
        end
        @inline Base.copysign(a::$T, b::AbstractSIMDVector{W,T}) where {W,T <: UnsignedHW} = vbroadcast(Val{W}(), abs(a))
        @inline Base.copysign(a::$T, b::VecUnroll) = VecUnroll(fmap(copysign, a, getfield(b, :data)))
    end
end
for (op, f) ∈ [
    (:(Base.:+),:vadd),
    (:(Base.FastMath.add_fast), :vadd_fast),
    (:(Base.:-),:vsub),
    (:(Base.FastMath.sub_fast), :vsub_fast),
    (:(Base.:*),:vmul),
    (:(Base.FastMath.mul_fast), :vmul_fast)
]
    @eval begin
        @inline $op(m::MM, j::NativeTypes) = $f(m, j)
        @inline $op(j::NativeTypes, m::MM) = $f(j, m)
        @inline $op(m::MM, ::StaticInt{N}) where {N} = $f(m, StaticInt{N}())
        @inline $op(::StaticInt{N}, m::MM) where {N} = $f(StaticInt{N}(), m)
    end
end

for (op,c) ∈ [(:(>), :(&)), (:(≥), :(&)), (:(<), :(|)), (:(≤), :(|))]
    @eval begin
        @inline function Base.$op(v1::AbstractSIMD{W,I}, v2::AbstractSIMD{W,U}) where {W,I<:Signed,U<:Unsigned}
            m1 = $op(v1,  zero(I))
            m2 = $op(v1 % U,  v2)
            $c(m1, m2)
        end
    end
end
for (f,vf) ∈ [
    (:convert,:vconvert),(:reinterpret,:vreinterpret),(:trunc,:vtrunc),(:unsafe_trunc,:vtrunc),(:round,:vround),(:floor,:vfloor),(:ceil,:vceil)
]
    @eval begin
        @inline Base.$f(::Type{T}, x::NativeTypes) where {T <: AbstractSIMD} = $vf(T, x)
        @inline Base.$f(::Type{T}, v::AbstractSIMD) where {T <: NativeTypes} = $vf(T, v)
        @inline Base.$f(::Type{T}, v::AbstractSIMD) where {T <: AbstractSIMD} = $vf(T, v)
    end
end
for (f,vf) ∈ [
    (:(Base.rem),:vrem), (:(Base.FastMath.rem_fast),:vrem_fast)
]
    @eval begin
        @inline $f(x::NativeTypes, ::Type{T}) where {T <: AbstractSIMD} = $vf(x, T)
        @inline $f(v::AbstractSIMD, ::Type{T}) where {T <: NativeTypes} = $vf(v, T)
        @inline $f(v::AbstractSIMD, ::Type{T}) where {T <: AbstractSIMD} = $vf(v, T)
    end
end

# These are defined here on `Base` functions to avoid `promote`
@inline function Base.:(<<)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:SignedHW,T2<:UnsignedHW}
    convert(T1, vshl(convert(T2, v1), v2))
end
@inline function Base.:(<<)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:UnsignedHW,T2<:SignedHW}
    convert(T1, vshl(convert(T2, v1), v2))
end
@inline function Base.:(>>)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:SignedHW,T2<:UnsignedHW}
    vashr(v1, (v2 % T1))
end
@inline function Base.:(>>)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:UnsignedHW,T2<:SignedHW}
    vashr(v1, (v2 % T1))
end
@inline function Base.:(>>>)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:SignedHW,T2<:UnsignedHW}
    convert(T1, vlshr(convert(T2, v1), v2))
end
@inline function Base.:(>>>)(v1::AbstractSIMD{W,T1}, v2::AbstractSIMD{W,T2}) where {W,T1<:UnsignedHW,T2<:SignedHW}
    convert(T2, vlshr(v1, convert(T1, v2)))
end

@inline function promote_except_first(a,b,c)
    d, e = promote(b, c)
    a, d, e
end
for (op, f, promotef) ∈ [
    (:(Base.fma), :vfma, :promote),
    (:(Base.muladd), :vmuladd, :promote),
    (:(IfElse.ifelse), :vifelse, :promote_except_first)
]
    @eval begin
        @inline function $op(a::AbstractSIMD, b::AbstractSIMD, c::AbstractSIMD)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::AbstractSIMD, b::AbstractSIMD, c::NativeTypes)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::AbstractSIMD, b::NativeTypes, c::AbstractSIMD)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::NativeTypes, b::AbstractSIMD, c::AbstractSIMD)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::AbstractSIMD, b::NativeTypes, c::NativeTypes)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::NativeTypes, b::AbstractSIMD, c::NativeTypes)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end
        @inline function $op(a::NativeTypes, b::NativeTypes, c::AbstractSIMD)
            x, y, z = $promotef(a, b, c)
            $f(x, y, z)
        end        
    end
end
@inline IfElse.ifelse(f::Function, m::AbstractSIMD{W,B}, args::Vararg{NativeTypesV,K}) where {W,K,B<:Union{Bool,Bit}} = vifelse(f, m, args...)
@inline IfElse.ifelse(f::Function, m::Bool, args::Vararg{NativeTypesV,K}) where {K} = vifelse(f, m, args...)
for (f,add,mul) ∈ [
    (:fma,:(+),:(*)), (:muladd,:(+),:(*)),
    (:vfma,:(+),:(*)), (:vmuladd,:(+),:(*)),
    (:vfma_fast,:(Base.FastMath.add_fast),:(Base.FastMath.mul_fast)),
    (:vmuladd_fast,:(Base.FastMath.add_fast),:(Base.FastMath.mul_fast))
]
    if (f !== :fma) && (f !== :muladd)
        @eval begin
            @inline function $f(a::NativeTypesV, b::NativeTypesV, c::NativeTypesV)
                x, y, z = promote(a, b, c)
                $f(x, y, z)
            end
        end
    else
        f = :(Base.$f)
    end
    @eval begin
        @inline $f(a::Zero, b::NativeTypesV, c::NativeTypesV) = c
        @inline $f(a::NativeTypesV, b::Zero, c::NativeTypesV) = c
        @inline $f(a::Zero, b::Zero, c::NativeTypesV) = c
        @inline $f(a::One, b::Zero, c::NativeTypesV) = c
        @inline $f(a::Zero, b::One, c::NativeTypesV) = c

        @inline $f(a::One, b::NativeTypesV, c::NativeTypesV) = $add(b, c)
        @inline $f(a::NativeTypesV, b::One, c::NativeTypesV) = $add(a, c)
        @inline $f(a::One, b::One, c::NativeTypesV) = $add(one(c), c)

        @inline $f(a::NativeTypesV, b::NativeTypesV, c::Zero) = $mul(a,b)
        @inline $f(a::Zero, b::NativeTypesV, c::Zero) = Zero()
        @inline $f(a::NativeTypesV, b::Zero, c::Zero) = Zero()
        @inline $f(a::Zero, b::Zero, c::Zero) = Zero()
        @inline $f(a::One, b::Zero, c::Zero) = Zero()
        @inline $f(a::Zero, b::One, c::Zero) = Zero()

        @inline $f(a::One, b::NativeTypesV, c::Zero) = b
        @inline $f(a::NativeTypesV, b::One, c::Zero) = a
        @inline $f(a::One, b::One, c::Zero) = One()
    end
end

# masks
for (vf,f) ∈ [
    (:vnot,:(!)),
    (:vnot,:(~)),
]
    @eval begin
        @inline Base.$f(m::AbstractSIMD{<:Any,<:Union{Bool,Bit}}) = $vf(m)
    end
end
