
@generated function static_tuple(t::SDTuple{N,X,P}) where {N,X,P}
    i = 0
    sdt = Expr(:tuple)
    Xv = tuple_type_to_value_tuple(X)
    for n ∈ 1:N
        push!(sdt.args, Xv[n] == -1 ? Expr(:ref, :x, (i += 1)) : Expr(:call, Expr(:curly, :Static, Xv[n])))
    end
    q = Expr(:block, Expr(:meta, :inline))
    i > 0 && push!(q.args, :(x = t.x))
    push!(q.args, sdt)
    q
end

@inline mulsizeof(::Type{T}, ::Tuple{}) where {T,X} = ()
@inline mulsizeof(::Type{T}, x::Tuple{X}) where {T,X} = (mulsizeof(T, first(x)), )
@inline mulsizeof(::Type{T}, x::Tuple) where {T} = (mulsizeof(T, first(x)), mulsizeof(T, Base.tail(x))...)
@generated function mulsizeof(::Type{T}, ::Type{S}) where {T, N, S <: Tuple{Vararg{Any,N}}}
    Smul = Expr(:curly, :Tuple)
    for n in 1:N
        Sₙ = S.parameters[n]
        push!(Smul.args, Sₙ == -1 ? -1 : sizeof(T) * Sₙ)
    end
    Smul
end
@inline function mulsizeof(::Type{T}, t::SDTuple{N,X,P}) where {X,N,P,T}
    SDTuple{N,mulsizeof(T, X),P}(mulsizeof(T, t.x))
end


abstract type AbstractStridedPointer{T,N,C,B,R,X,P} end

struct StridedPointer{T,N,C,B,R,X,P} <: AbstractStridedPointer{T,N,C,B,R,X,P}
    p::Ptr{T}
    st::SDTuple{N,X,P}
end
@inline stridedpointer(A::AbstractArray) = stridedpointer(device(A), A)
@inline function stridedpointer(::CPUPointer, A::AbstractArray{T}) where {T}
    stridedpointer(pointer(A), contiguous_axis(A), contiguous_batch_size(A), striderank(A), mulsizeof(T, sdstrides(A)))
end
@inline function stridedpointer(ptr::Ptr{T}, ::Contiguous{C}, ::ContiguousBatch{B}, ::StrideRank{R}, st::SDTuple{N,X,P}) where {T,C,B,R,X,N,P}
    StridedPointer{T,N,C,B,R,X,P}(ptr, st)
end
@inline strides(ptr::StridedPointer) = static_tuple(ptr.st)
@inline ArrayInterface.contiguous_axis_indicator(ptr::StridedPointer{T,N,C}) where {T,N,C} = contiguous_axis_indicator(Contiguous{C}(), Val{N}())

# Shouldn't need to special case Array
# function stridedpointer(A::Array{T,N}) where {T,N}
#     StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
# end

@inline vload(ptr::AbstractStridedPointer, i) = vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
@inline vload(ptr::AbstractStridedPointer, i, m) = vload(pointer(ptr), tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)
@inline vstore!(ptr::AbstractStridedPointer, v, i) = vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)))
@inline vstore!(ptr::AbstractStridedPointer, v, i, m) = vstore!(pointer(ptr), v, tdot(i, strides(ptr), contiguous_axis_indicator(ptr)), m)

