
# unary # 2^2 - 2 = 2 definitions
@inline fmap(f::F, x::Tuple{X}) where {F,X} = (f(first(x)),)
@inline fmap(f::F, x::NTuple) where {F} = (f(first(x)), fmap(f, Base.tail(x))...)

# binary # 2^3 - 2 = 6 definitions
@inline fmap(f::F, x::Tuple{X}, y::Tuple{Y}) where {F,X,Y} = (f(first(x), first(y)),)
@inline fmap(f::F, x::Tuple{X}, y) where {F,X} = (f(first(x), y),)
@inline fmap(f::F, x, y::Tuple{Y}) where {F,Y} = (f(x, first(y)),)
@inline fmap(f::F, x::Tuple{Vararg{Any,N}}, y::Tuple{Vararg{Any,N}}) where {F,N} = (f(first(x), first(y)), fmap(f, Base.tail(x), Base.tail(y))...)
@inline fmap(f::F, x::Tuple, y) where {F} = (f(first(x), y), fmap(f, Base.tail(x), y)...)
@inline fmap(f::F, x, y::Tuple) where {F} = (f(x, first(y)), fmap(f, x, Base.tail(y))...)


fmap(f::F, x::Tuple{X}, y::Tuple) where {F,X} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple{Y}) where {F,Y} = throw("Dimension mismatch.")
fmap(f::F, x::Tuple, y::Tuple) where {F} = throw("Dimension mismatch.")

@generated function fmap(f::F, x::Vararg{Any,N}) where {F,N}
    q = Expr(:block, Expr(:meta, :inline))
    t = Expr(:tuple)
    U = 1
    call = Expr(:call, :f)
    syms = Vector{Symbol}(undef, N)
    istup = Vector{Bool}(undef, N)
    for n ∈ 1:N
        syms[n] = xₙ = Symbol(:x_, n)
        push!(q.args, Expr(:(=), xₙ, Expr(:ref, :x, n)))
        istup[n] = ist = (x[n] <: Tuple)
        if ist
            U = length(x[n].parameters)
            push!(call.args, Expr(:ref, xₙ, 1))
        else
            push!(call.args, xₙ)
        end
    end
    push!(t.args, call)
    for u ∈ 2:U
        call = Expr(:call, :f)
        for n ∈ 1:N
            xₙ = syms[n]
            if istup[n]
                push!(call.args, Expr(:ref, xₙ, u))
            else
                push!(call.args, xₙ)
            end
        end
        push!(t.args, call)
    end
    push!(q.args, t); q
end

for op ∈ [:vsub, :vabs, :vfloor, :vceil, :vtrunc, :vround, :vsqrt, :vnot, :vleading_zeros, :vtrailing_zeros, :vsub_fast]
    @eval @inline $op(v1::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap($op, v1.data))
end
# only for `Float32` and `Float64`
@inline inv_approx(v::VecUnroll{N,W,T}) where {N,W,T<:Union{Float32,Float64}} = VecUnroll(fmap(inv_approx, v.data))

@inline vreinterpret(::Type{T}, v::VecUnroll) where {T<:Number} = VecUnroll(fmap(vreinterpret, T, v.data))
for op ∈ [:vadd,:vsub,:vmul,:vand,:vor,:vxor,:vlt,:vle,:vgt,:vge,:veq,:vne,:vadd_fast,:vsub_fast,:vmul_fast]
    @eval begin
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
        @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{M}) where {N,W,T,V,M} = VecUnroll(fmap($op, v1.data, vbroadcast(Val{W}(), T(M))))
        @inline $op(::StaticInt{M}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V,M} = VecUnroll(fmap($op, vbroadcast(Val{W}(), T(M)), v1.data))

        @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{1}) where {N,W,T,V} = VecUnroll(fmap($op, v1.data, One()))
        @inline $op(::StaticInt{1}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V} = VecUnroll(fmap($op, One(), v1.data))

        @inline $op(v1::VecUnroll{N,W,T,V}, ::StaticInt{0}) where {N,W,T,V} = VecUnroll(fmap($op, v1.data, Zero()))
        @inline $op(::StaticInt{0}, v1::VecUnroll{N,W,T,V}) where {N,W,T,V} = VecUnroll(fmap($op, Zero(), v1.data))
    end
end
for op ∈ [:vmax,:vmax_fast,:vmin,:vmin_fast,:vcopysign]
    @eval begin
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
    end
end
for op ∈ [:vgt,:vge,:vlt,:vle,:veq,:vne,:vmax,:vmax_fast,:vmin,:vmin_fast]
    @eval begin
        @inline $op(v::VecUnroll{N,W,T,V}, s::Union{NativeTypes,AbstractSIMDVector}) where {N,W,T,V} = VecUnroll(fmap($op, v.data, vbroadcast(Val{W}(), s)))
        @inline $op(s::Union{NativeTypes,AbstractSIMDVector}, v::VecUnroll{N,W,T,V}) where {N,W,T,V} = VecUnroll(fmap($op, vbroadcast(Val{W}(), s), v.data))
    end
end

for op ∈ [:vrem, :vshl, :vashr, :vlshr, :vdiv, :vfdiv, :vrem_fast, :vfdiv_fast]
    @eval begin
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
    end
end
for op ∈ [:vrem, :vand, :vor, :vxor, :vshl, :vashr, :vlshr, :vlt, :vle,:vgt,:vge,:veq,:vne]
    @eval begin
        @inline $op(vu::VecUnroll, i::MM) = $op(vu, Vec(i))
        @inline $op(i::MM, vu::VecUnroll) = $op(Vec(i), vu)
    end
end
for op ∈ [:vshl, :vashr, :vlshr]
    @eval @inline $op(m::Mask, vu::VecUnroll) = $op(Vec(m), vu)
end
for op ∈ [:rotate_left,:rotate_right,:funnel_shift_left,:funnel_shift_right]
    @eval begin
        @inline $op(v1::VecUnroll{N,W,T,V}, v2::R) where {N,W,T,V,R<:IntegerTypes} = VecUnroll(fmap($op, v1.data, promote_type(V, R)(v2)))
        @inline $op(v1::R, v2::VecUnroll{N,W,T,V}) where {N,W,T,V,R<:IntegerTypes} = VecUnroll(fmap($op, promote_type(V, R)(v1), v2.data))
        @inline $op(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap($op, v1.data, v2.data))
    end
end

for op ∈ [:vfma, :vmuladd, :vfma_fast, :vmuladd_fast, :vfnmadd, :vfmsub, :vfnmsub, :vfnmadd_fast, :vfmsub_fast, :vfnmsub_fast,
          :vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231, :ifmahi, :ifmalo]
    @eval begin
        @inline $op(v1::VecUnroll{N,W}, v2::VecUnroll{N,W}, v3::VecUnroll{N,W}) where {N,W} = VecUnroll(fmap($op, v1.data, v2.data, v3.data))
    end
end
@inline vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:NativeTypes} = VecUnroll(fmap(vifelse, v1.data, Vec{W,T}(v2), Vec{W,T}(v3)))
@inline vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::T, v3::T) where {N,W,T<:Real} = VecUnroll(fmap(vifelse, v1.data, v2, v3))
@inline vifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,T} = VecUnroll(fmap(vifelse, Vec{W,T}(v1), v2.data, Vec{W,T}(v3)))
@inline vifelse(v1::Vec{W,Bool}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(vifelse, Vec{W,T}(v1), Vec{W,T}(v2), v3.data))
@inline vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::VecUnroll{N,W,T}, v3::Union{NativeTypes,AbstractSIMDVector,StaticInt}) where {N,W,T} = VecUnroll(fmap(vifelse, v1.data, v2.data, Vec{W,T}(v3)))
@inline vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::Union{NativeTypes,AbstractSIMDVector,StaticInt}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(vifelse, v1.data, Vec{W,T}(v2), v3.data))
@inline vifelse(v1::Vec{W,Bool}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(vifelse, Vec{W,T}(v1), v2.data, v3.data))
@inline vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::VecUnroll{N,W,T}, v3::VecUnroll{N,W,T}) where {N,W,T} = VecUnroll(fmap(vifelse, v1.data, v2.data, v3.data))
@inline function vifelse(v1::VecUnroll{N,W,<:Boolean}, v2::VecUnroll{N,W}, v3::VecUnroll{N,W}) where {N,W}
    v4, v5 = promote(v2, v3)
    VecUnroll(fmap(vifelse, v1.data, v4.data, v5.data))
end


@inline veq(v::VecUnroll{N,W,T}, x::AbstractIrrational) where {N,W,T} = v == vbroadcast(Val{W}(), T(x))
@inline veq(x::AbstractIrrational, v::VecUnroll{N,W,T}) where {N,W,T} = vbroadcast(Val{W}(), T(x)) == v

@inline vconvert(::Type{T}, v::VecUnroll) where {T<:Real} = VecUnroll(fmap(vconvert, T, v.data))
@inline vconvert(::Type{U}, v::VecUnroll) where {N,W,T,U<:VecUnroll{N,W,T}} = VecUnroll(fmap(vconvert, T, v.data))
@inline vconvert(::Type{VU}, v::VU) where {N,W,T,VU <: VecUnroll{N,W,T}} = v
@inline vunsafe_trunc(::Type{T}, v::VecUnroll) where {T<:Real} = VecUnroll(fmap(vunsafe_trunc, T, v.data))
@inline vrem(v::VecUnroll, ::Type{T}) where {T<:Real} = VecUnroll(fmap(vrem, v.data, T))
@inline vrem(v::VecUnroll{N,W}, ::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V} = VecUnroll(fmap(vrem, v.data, V))

@inline (::Type{VecUnroll{N,W,T,V}})(vu::VecUnroll{N,W,T,V}) where {N,W,T,V<:AbstractSIMDVector{W,T}} = vu
@inline (::Type{VecUnroll{N,W,T,VT}})(vu::VecUnroll{N,W,S,VS})  where {N,W,T,VT<:AbstractSIMDVector{W,T},S,VS<:AbstractSIMDVector{W,S}} = VecUnroll(fmap(convert, Vec{W,T}, vu.data))


function collapse_expr(N, op)
    N += 1
    t = Expr(:tuple); s = Vector{Symbol}(undef, N)
    for n ∈ 1:N
        s_n = s[n] = Symbol(:v_, n)
        push!(t.args, s_n)
    end
    q = quote
        $(Expr(:meta,:inline))
        $t = data(vu)
    end
    while N > 1
        for n ∈ 1:N >>> 1
            push!(q.args, Expr(:(=), s[n], Expr(:call, op, s[n], s[n + (N >>> 1)])))
        end
        isodd(N) && push!(q.args, Expr(:(=), s[1], Expr(:call, op, s[1], s[N])))
        N >>>= 1
    end
    q
end
@generated collapse_add(vu::VecUnroll{N}) where {N} = collapse_expr(N, :vadd)
@generated collapse_mul(vu::VecUnroll{N}) where {N} = collapse_expr(N, :vmul)
@generated collapse_max(vu::VecUnroll{N}) where {N} = collapse_expr(N, :max)
@generated collapse_min(vu::VecUnroll{N}) where {N} = collapse_expr(N, :min)
@generated collapse_and(vu::VecUnroll{N}) where {N} = collapse_expr(N, :&)
@generated collapse_or(vu::VecUnroll{N}) where {N} = collapse_expr(N, :|)
@inline vsum(vu::VecUnroll) = vsum(collapse_add(vu))
@inline vprod(vu::VecUnroll) = vprod(collapse_mul(vu))
@inline vmaximum(vu::VecUnroll) = vmaximum(collapse_max(vu))
@inline vminimum(vu::VecUnroll) = vminimum(collapse_min(vu))
@inline vall(vu::VecUnroll) = vall(collapse_and(vu))
@inline vany(vu::VecUnroll) = vany(collapse_or(vu))

