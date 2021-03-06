struct OffsetPrecalc{T,N,C,B,R,X,M,P<:AbstractStridedPointer{T,N,C,B,R,X,M},I} <: AbstractStridedPointer{T,N,C,B,R,X,M}
    ptr::P
    precalc::I
end
@inline Base.pointer(ptr::OffsetPrecalc) = pointer(getfield(ptr, :ptr))
@inline llvmptr(ptr::OffsetPrecalc) = llvmptr(getfield(ptr, :ptr))
@inline Base.similar(ptr::OffsetPrecalc, p::Pointer) = OffsetPrecalc(similar(getfield(ptr, :ptr), p), getfield(ptr, :precalc))
# @inline pointerforcomparison(p::OffsetPrecalc) = pointerforcomparison(getfield(p, :ptr))
# @inline pointerforcomparison(p::OffsetPrecalc, i::Tuple) = pointerforcomparison(p.ptr, i)
@inline offsetprecalc(x, ::Any) = x
# @inline pointerforcomparison(p::AbstractStridedPointer) = pointer(p)
# @inline pointerforcomparison(p::AbstractStridedPointer, i) = gep(p, i)
@inline ArrayInterface.offsets(p::OffsetPrecalc) = offsets(getfield(p, :ptr))

@inline Base.strides(p::OffsetPrecalc) = strides(getfield(p, :ptr))
@inline ArrayInterface.strides(p::OffsetPrecalc) = strides(getfield(p, :ptr))

@inline function similar_no_offset(sptr::OffsetPrecalc{T}, ptr::Pointer{T}) where {T}
    OffsetPrecalc(similar_no_offset(getfield(sptr, :ptr), ptr), getfield(sptr, :precalc))
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
    ex = if (isodd(I) && 1 ≤ Is ≤ N) && (c.parameters[Is] !== nothing)
        Expr(:call, GlobalRef(Core, :getfield), :c, Is, false)
    elseif (I ∈ (6, 10) && (I >> 2 ≤ N)) && (c.parameters[I >> 2] !== nothing)
        Expr(:call, :lazymul, Expr(:call, Expr(:curly, :StaticInt, 2)), Expr(:call, GlobalRef(Core, :getfield), :c, I >> 2, false))
    else
        Expr(:call, :lazymul, Expr(:call, Expr(:curly, :StaticInt, I)), :b)
    end
    Expr(:block, Expr(:meta,:inline), ex)
end
@inline lazymul(a, b, c) = lazymul(a, b)
@inline lazymul(a::StaticInt, b, ::Nothing) = lazymul(a, b)

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
            Xᵢ = _unwrap(X[i])
            anyprecalcs = true
            if Xᵢ === nothing
                anydynamicprecals = true
                pstride_i = Symbol(:pstride_, i)
                push!(pstrideextracts.args, Expr(:(=), pstride_i, Expr(:call, GlobalRef(Core, :getfield), :pstride, i, false)))
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

@inline tdot(ptr::OffsetPrecalc{T}, a, b) where {T} = tdot(llvmptr(ptr), a, b, getfield(ptr, :precalc))
@inline tdot(ptr::OffsetPrecalc, ::Tuple{}, ::Tuple{}) = (llvmptr(ptr), Zero())

