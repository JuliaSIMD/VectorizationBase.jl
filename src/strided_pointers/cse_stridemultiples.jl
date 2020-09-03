struct OffsetPrecalc{T,N,C,B,R,X,M,P<:AbstractStridedPointer{T,N,C,B,R,X,M},I} <: AbstractStridedPointer{T,N,C,B,R,X,M}
    ptr::P
    precalc::I
end
@inline Base.pointer(ptr::OffsetPrecalc) = pointer(ptr.ptr)
@inline Base.similar(ptr::OffsetPrecalc, p::Ptr) = OffsetPrecalc(similar(ptr.ptr, p), ptr.precalc)
@inline pointerforcomparison(p::OffsetPrecalc) = pointerforcomparison(p.ptr)
@inline pointerforcomparison(p::OffsetPrecalc, i::Tuple) = pointerforcomparison(p.ptr, i)
@inline offsetprecalc(x, ::Any) = x

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
@generated function lazymul_maybe_cached(::Static{I}, b, c::NTuple{N,Int}) where {I,N}
    Is = (I - 1) >> 1
    if isodd(I) && 1 ≤ Is ≤ N
        Expr(:block, Expr(:meta,:inline), Expr(:call, :getindex, :c, Is))
    elseif I ∈ (6, 10)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul, Expr(:call, Expr(:curly, :Static, 2)), Expr(:call, :getindex, :c, I >> 2)))
    else
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul, Expr(:call, Expr(:curly, :Static, I)), :b))
    end
end
@inline lazymul_maybe_cached(a, b, c) = lazymul(a, b)
@inline lazymul_maybe_cached(a::Static, b, ::Nothing) = lazymul(a, b)
@inline tdotc(a::Tuple{Static{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = lazymul_maybe_cached(Static{I1}(),b[1],c[1])
@inline tdotc(a::Tuple{Static{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{Nothing,Vararg}) where {I1,I2} = lazymul(Static{I1}(),b[1])
@inline tdotc(a::Tuple{I1,Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = lazymul(a[1],b[1])

@inline tdotc(a::Tuple{Static{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = lazyadd(lazymul_maybe_cached(Static{I1}(),b[1],c[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdotc(a::Tuple{Static{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{Nothing,I6,Vararg}) where {I1,I2,I4,I5,I6} = lazyadd(lazymul(Static{I1}(),b[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))
@inline tdotc(a::Tuple{I1,I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = lazyadd(lazymul(a[1],b[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))

# descript is a tuple of (unrollfactor) for ech ind; if it shouldn't preallocate, unrollfactor may be set to 1
function precalc_quote_from_descript(descript, contig, X)
    precalc = Expr(:tuple)
    anyprecalcs = anydynamicprecals = false
    pstrideextracts = Expr(:block)
    for (i,uf) ∈ enumerate(descript)
        i == contig && continue
        if uf < 3
            push!(precalc.args, nothing)
        else
            t = Expr(:tuple)
            Xᵢ = X[i]
            anyprecalcs = true
            if Xᵢ == -1
                anydynamicprecals = true
                pstride_i = Symbol(:pstride_, i)
                push!(pstrideextracts.args, Expr(:(=), pstride_i, Expr(:ref, :pstride, i)))
                foreach(u -> push!(t.args, Expr(:call, :vmul, u, pstride_i)), 3:2:uf)
            else
                foreach(u -> push!(t.args, u * Xᵢ), 3:2:uf)
            end
            push!(precalc.args, t)
        end        
    end
    q = Expr(:block, Expr(:meta,:inline))
    if anydynamicprecals
        push!(q.args, :(pstride = strides(p)))
        push!(q.args, pstrideextracts)
    end
    if anyprecalcs
        push!(q.args, Expr(:call, :OffsetPrecalc, :p, precalc))
    else
        push!(q.args, :p)
    end
    q
end


# @inline stride1offset(ptr::OffsetPrecalc, i) = stride1offset(ptr.ptr, i)

# @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer}, ::Tuple{}) where {T} = Zero()
# @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Tuple{I}) where {I,T} = Zero()
# # @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Integer) where {T} = vmulnp(static_sizeof(T), i)
# @inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractColumnMajorStridedPointer{T}}, i::Tuple{I1,I2,Vararg}) where {I1,I2,T}
#     @inbounds tdotc(Base.tail(i), ptr.ptr.strides, ptr.precalc)
# end


# @inline offsetprecalc(p::AbstractRowMajorStridedPointer{T,0}, ::Val{<:Any}) where {T} = p
# @generated function offsetprecalc(p::AbstractRowMajorStridedPointer, ::Val{descript}) where {descript}
#     precalc_quote_from_descript(Base.tail(reverse(descript)))
# end

# @inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractRowMajorStridedPointer{T,1}}, i::Tuple{I1,I2}) where {I1,I2,T}
#     @inbounds vmul_maybe_cached(i[1], ptr.ptr.strides[1], ptr.precalc[1])
# end
# @inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractRowMajorStridedPointer{T}}, i::Tuple{I1,I2,I3,Vararg}) where {I1,I2,I3,T}
#     ri = reverse(i)
#     @inbounds tdotc(Base.tail(ri), ptr.ptr.strides, ptr.precalc)
# end

# @generated function offsetprecalc(p::AbstractSparseStridedPointer, ::Val{descript}) where {descript}
#     precalc_quote_from_descript(descript)
# end
# @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Tuple{I}) where {I,T} = @inbounds vmul_maybe_cached(i[1], ptr.ptr.strides[1], ptr.precalc[1])
# # @inline stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Integer) where {T} = @inbounds vmul_maybe_cached(i, ptr.ptr.strides[1], ptr.precalc[1])
# @inline function stridedoffset(ptr::OffsetPrecalc{T,<:AbstractSparseStridedPointer{T}}, i::Tuple{I1,I2,Vararg}) where {I1,I2,T}
#     @inbounds tdotc(i, ptr.ptr.strides, ptr.precalc)
# end


