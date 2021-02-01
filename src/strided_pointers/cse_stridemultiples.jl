struct OffsetPrecalc{T,N,C,B,R,X,M,P<:AbstractStridedPointer{T,N,C,B,R,X,M},I} <: AbstractStridedPointer{T,N,C,B,R,X,M}
    ptr::P
    precalc::I
end
@inline Base.pointer(ptr::OffsetPrecalc) = pointer(ptr.ptr)
@inline Base.similar(ptr::OffsetPrecalc, p::Ptr) = OffsetPrecalc(similar(ptr.ptr, p), ptr.precalc)
@inline pointerforcomparison(p::OffsetPrecalc) = pointerforcomparison(p.ptr)
# @inline pointerforcomparison(p::OffsetPrecalc, i::Tuple) = pointerforcomparison(p.ptr, i)
@inline offsetprecalc(x, ::Any) = x
@inline pointerforcomparison(p::AbstractStridedPointer) = pointer(p)
@inline pointerforcomparison(p::AbstractStridedPointer, i) = gep(p, i)
@inline ArrayInterface.offsets(p::OffsetPrecalc) = offsets(p.ptr)

@inline Base.strides(p::OffsetPrecalc) = strides(p.ptr)

@inline function similar_no_offset(sptr::OffsetPrecalc{T}, ptr::Ptr{T}) where {T}
    OffsetPrecalc(similar_no_offset(sptr.ptr, ptr), sptr.precalc)
end


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
@generated function lazymul(::StaticInt{I}, b, c::Tuple{Vararg{Any,N}}) where {I,N}
    Is = (I - 1) >> 1
    if (isodd(I) && 1 ≤ Is ≤ N) && (c.parameters[Is] !== nothing)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :getindex, :c, Is))
    elseif (I ∈ (6, 10) && (I >> 2 ≤ N)) && (c.parameters[I >> 2] !== nothing)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul, Expr(:call, Expr(:curly, :StaticInt, 2)), Expr(:call, :getindex, :c, I >> 2)))
    else
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul, Expr(:call, Expr(:curly, :StaticInt, I)), :b))
    end
end
@inline lazymul(a, b, c) = lazymul(a, b)
@inline lazymul(a::StaticInt, b, ::Nothing) = lazymul(a, b)

@generated function lazymul_no_promote(::Type{T}, ::StaticInt{I}, b, c::Tuple{Vararg{Any,N}}) where {T, I, N}
    Is = (I - 1) >> 1
    if (isodd(I) && 1 ≤ Is ≤ N) && (c.parameters[Is] !== nothing)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :getindex, :c, Is))
    elseif (I ∈ (6, 10) && (I >> 2 ≤ N)) && (c.parameters[I >> 2] !== nothing)
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul_no_promote, Expr(:call, Expr(:curly, :StaticInt, 2)), Expr(:call, :getindex, :c, I >> 2)))
    else
        Expr(:block, Expr(:meta,:inline), Expr(:call, :lazymul_no_promote, Expr(:call, Expr(:curly, :StaticInt, I)), :b))
    end
end
@inline lazymul_no_promote(::Type{T}, a, b, c) where {T} = lazymul_no_promote(T, a, b)
@inline lazymul_no_promote(::Type{T}, a::StaticInt, b, ::Nothing) where {T} = lazymul_no_promote(T, a, b)

# @inline tdotc(a::Tuple{StaticInt{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = lazymul_maybe_cached(StaticInt{I1}(),b[1],c[1])
# @inline tdotc(a::Tuple{StaticInt{I1},Vararg}, b::Tuple{I2,Vararg}, c::Tuple{Nothing,Vararg}) where {I1,I2} = lazymul(StaticInt{I1}(),b[1])
# @inline tdotc(a::Tuple{I1,Vararg}, b::Tuple{I2,Vararg}, c::Tuple{I3,Vararg}) where {I1,I2,I3} = lazymul(a[1],b[1])

# @inline tdotc(a::Tuple{StaticInt{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = lazyadd(lazymul_maybe_cached(StaticInt{I1}(),b[1],c[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))
# @inline tdotc(a::Tuple{StaticInt{I1},I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{Nothing,I6,Vararg}) where {I1,I2,I4,I5,I6} = lazyadd(lazymul(StaticInt{I1}(),b[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))
# @inline tdotc(a::Tuple{I1,I4,Vararg}, b::Tuple{I2,I5,Vararg}, c::Tuple{I3,I6,Vararg}) where {I1,I2,I3,I4,I5,I6} = lazyadd(lazymul(a[1],b[1]), tdotc(Base.tail(a), Base.tail(b), Base.tail(c)))


_unwrap(::Type{StaticInt{N}}) where {N} = N
_unwrap(_) = nothing
# descript is a tuple of (unrollfactor) for each ind; if it shouldn't preallocate, unrollfactor may be set to 1
function precalc_quote_from_descript(descript, contig, X)
    precalc = Expr(:tuple)
    anyprecalcs = anydynamicprecals = false
    pstrideextracts = Expr(:block)
    for (i,uf) ∈ enumerate(descript)
        if i == contig || uf < 3
            push!(precalc.args, nothing)
        else
            t = Expr(:tuple)
            Xᵢ = unwrap(X[i])
            anyprecalcs = true
            if Xᵢ === nothing
                anydynamicprecals = true
                pstride_i = Symbol(:pstride_, i)
                push!(pstrideextracts.args, Expr(:(=), pstride_i, Expr(:ref, :pstride, i)))
                foreach(u -> push!(t.args, Expr(:call, :vmul_fast, u, pstride_i)), 3:2:uf)
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
@generated function offsetprecalc(p::AbstractStridedPointer{T,N,C,B,R,X,O}, ::Val{descript}) where {T,N,C,B,R,X,O,descript}
    precalc_quote_from_descript(descript, C, X.parameters)
end

# offsetprecalc(p::StridedBitPointer) = p


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


@inline tdot(ptr::OffsetPrecalc{T}, a, b, c) where {T} = tdot(T, a, b, ptr.precalc, c)
@inline tdot(ptr::OffsetPrecalc, ::Tuple{}, ::Tuple{}, ::Tuple{}) = Zero()

