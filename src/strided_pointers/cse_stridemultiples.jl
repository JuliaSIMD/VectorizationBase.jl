"""

Basically:

if I ∈ [3,5,7,9]
    c[(I - 1) >> 1]
else
    b * I
end

because

c = b .* [3, 5, 7, 9]


"""
@generated function vmul_maybe_cached(::Static{I}, b, c::NTuple{N,Int}) where {I,N}
    Is = (I - 1) >> 1
    if isodd(I) && 1 ≤ Is ≤ N
        Expr(:block, Expr(:meta,:inline), Expr(:call, :getindex, :c, Is))
    elseif I ∈ (6, 10)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :vmul, Expr(:call, Expr(:curly, :Static, 2)), Expr(:call, :getindex, :c, I >> 2)))
    else
        Expr(:block, Expr(:meta,:inline), Expr(:call, :vmul, Expr(:call, Expr(:curly, :Static, I)), :b))
    end
end
@inline vmul_maybe_cached(a, b, c) = vmul(a, b)
@inline vmul_maybe_cached(a::Static, b, ::Nothing) = vmul(a, b)
@inline tdot(a::Tuple{Static{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = vmul_maybe_cached(Static{I1}(),b[1],c[1])
@inline tdot(a::Tuple{Static{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{Nothing,Vararg}) where {I1,I2} = vmul(Static{I1}(),b[1])
@inline tdot(a::Tuple{I1,Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = vmul(a[1],b[1])

@inline tdot(a::Tuple{Static{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = vadd(vmul_maybe_cached(Static{I1}(),b[1],c[1]), tdot(Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdot(a::Tuple{Static{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{Nothing,I6,Vararg}) where {I1,I2,I4,I5,I6} = vadd(vmul(Static{I1}(),b[1]), tdot(Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdot(a::Tuple{I1,I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = vadd(vmul(a[1],b[1]), tdot(Base.tail(a), Base.tail(b), Base.tail(c)))

struct OffsetPrecalc{T,P<:AbstractStridedPointer{T},I} <: AbstractStridedPointer{T}
    ptr::P
    precalc::I
end
@inline Base.pointer(ptr::OffsetPrecalc) = pointer(ptr.ptr)
@inline Base.similar(ptr::OffsetPrecalc, p::Ptr) = OffsetPrecalc(similar(ptr.ptr, p), ptr.precalc)
@inline pointerforcomparison(p::OffsetPrecalc) = pointerforcomparison(p.ptr)
@inline pointerforcomparison(p::OffsetPrecalc, i::Tuple) = pointerforcomparison(p.ptr, i)
@inline offsetprecalc(x, ::Any) = x
@inline offsetprecalc(x::PackedStridedBitPointer, ::Val{<:Any}) = x
@inline offsetprecalc(x::RowMajorStridedBitPointer, ::Val{<:Any}) = x
# descript is a tuple of (unrollfactor) for ech ind; if it shouldn't preallocate, unrollfactor may be set to 1
function precalc_quote_from_descript(descript)
    precalc = Expr(:tuple)
    anyprecalcs = false
    pstrides = Expr(:block, Expr(:(=), :pstride, Expr(:(.), :p, QuoteNode(:strides))))
    for (i,uf) ∈ enumerate(descript)
        if uf < 3
            push!(precalc.args, nothing)
        else
            t = Expr(:tuple)
            anyprecalcs = true
            pstride_i = Symbol(:pstride_, i)
            push!(pstrides.args, Expr(:(=), pstride_i, Expr(:ref, :pstride, i)))
            for u ∈ 3:uf
                if isodd(u)
                    push!(t.args, Expr(:call, :vmul, u, pstride_i))
                end
            end
            push!(precalc.args, t)
        end        
    end
    if anyprecalcs
        Expr(:block, Expr(:meta,:inline), pstrides, Expr(:call, :OffsetPrecalc, :p, precalc))
    else
        Expr(:block, Expr(:meta,:inline), :p)
    end
end
@inline offsetprecalc(p::AbstractColumnMajorStridedPointer{T,0}, ::Val{<:Any}) where {T} = p
@generated function offsetprecalc(p::AbstractColumnMajorStridedPointer, ::Val{descript}) where {descript}
    precalc_quote_from_descript(Base.tail(descript))
end

@inline stride1offset(ptr::OffsetPrecalc, i) = stride1offset(ptr.ptr, i)

@inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer}, ::Tuple{}) where {T} = Zero()
@inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Tuple{I}) where {I,T} = Zero()
# @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Integer) where {T} = vmulnp(static_sizeof(T), i)
@inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Tuple{I1,I2,Vararg}) where {I1,I2,T}
    @inbounds tdot(Base.tail(i), ptr.ptr.strides, ptr.precalc)
end


@inline offsetprecalc(p::AbstractRowMajorStridedPointer{T,0}, ::Val{<:Any}) where {T} = p
@generated function offsetprecalc(p::AbstractRowMajorStridedPointer, ::Val{descript}) where {descript}
    precalc_quote_from_descript(Base.tail(reverse(descript)))
end

@inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractRowMajorStridedPointer{T,1}}, i::Tuple{I1,I2}) where {I1,I2,T}
    @inbounds vmul_maybe_cached(i[1], ptr.ptr.strides[1], ptr.precalc[1])
end
@inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractRowMajorStridedPointer{T}}, i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3,T}
    ri = reverse(i)
    @inbounds tdot(Base.tail(ri), ptr.ptr.strides, ptr.precalc)
end

@generated function offsetprecalc(p::AbstractSparseStridedPointer, ::Val{descript}) where {descript}
    precalc_quote_from_descript(descript)
end
@inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Tuple{I}) where {I,T} = @inbounds vmul_maybe_cached(i[1], ptr.ptr.strides[1], ptr.precalc[1])
# @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Integer) where {T} = @inbounds vmul_maybe_cached(i, ptr.ptr.strides[1], ptr.precalc[1])
@inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Tuple{I1,I2,Vararg}) where {I1,I2,T}
    @inbounds tdot(i, ptr.ptr.strides, ptr.precalc)
end


