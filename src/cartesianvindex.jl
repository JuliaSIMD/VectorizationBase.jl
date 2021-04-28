
struct CartesianVIndex{N,T<:Tuple{Vararg{Integer,N}}} <: Base.AbstractCartesianIndex{N}
    I::T
    @inline CartesianVIndex(I::T) where {N, T <: Tuple{Vararg{Integer,N}}} = new{N,T}(I)
end
Base.length(::CartesianVIndex{N}) where {N} = N
ArrayInterface.known_length(::Type{<:CartesianVIndex{N}}) where {N} = N
Base.Tuple(i::CartesianVIndex) = getfield(i, :I)
function Base.:(:)(I::CartesianVIndex{N}, J::CartesianVIndex{N}) where {N} 
   CartesianIndices(map((i,j) -> i:j, getfield(I, :I), getfield(J,:I)))
end
Base.@propagate_inbounds Base.getindex(I::CartesianVIndex, i) = getfield(I,:I)[i]
_ndim(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
@inline gesp(p::AbstractStridedPointer{T,N}, i::Tuple{CartesianVIndex{N}}) where {T,N} = gesp(p, getfield(getfield(i,1,false),:I))
# _ndim(::Type{<:AbstractArray{N}}) where {N} = N
@generated function CartesianVIndex(I::T) where {T <: Tuple{Vararg{Union{Integer,CartesianIndex,CartesianVIndex}}}}
    iexpr = Expr(:tuple)
    Tp = T.parameters
    q = Expr(:block)
    for i in eachindex(Tp)
        I_i = Symbol(:I_, i)
        push!(q.args, Expr(:(=), I_i, Expr(:ref, :I, i)))
        if Tp[i] <: Base.AbstractCartesianIndex
            for n in 1:_ndim(Tp[i])
                push!(iexpr.args, Expr(:ref, I_i, n))
            end
        else
            push!(iexpr.args, I_i)
        end
    end
    push!(q.args, Expr(:call, :CartesianVIndex, iexpr))
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), q))
end

# @inline Base.CartesianIndex(I::Tuple{Vararg{Union{Integer,CartesianIndex,CartesianVIndex,StaticInt}}}) = CartesianVIndex(I)

@generated function _maybestaticfirst(a::Tuple{Vararg{Any,N}}) where {N}
    quote
        $(Expr(:meta,:inline))
        Base.Cartesian.@ntuple $N n -> maybestaticfirst(a[n])
    end
end
@generated function _maybestaticlast(a::Tuple{Vararg{Any,N}}) where {N}
    quote
        $(Expr(:meta,:inline))
        Base.Cartesian.@ntuple $N n -> maybestaticlast(a[n])
    end
end
@inline maybestaticfirst(A::CartesianIndices) = CartesianVIndex(_maybestaticfirst(A.indices))
@inline maybestaticlast(A::CartesianIndices) = CartesianVIndex(_maybestaticlast(A.indices))

for (op,f) ∈ [(:(+),:vadd_fast), (:(-),:vsub_fast), (:(*),:vmul_fast)]
  @eval begin
    @inline Base.$op(a::CartesianVIndex, b) = CartesianVIndex(fmap($f, getfield(a,:I), b))
    @inline Base.$op(a, b::CartesianVIndex) = CartesianVIndex(fmap($f, a, getfield(b,:I)))
    @inline Base.$op(a::CartesianVIndex, b::CartesianVIndex) = CartesianVIndex(fmap($f, getfield(a,:I), getfield(b,:I)))
  end
end



