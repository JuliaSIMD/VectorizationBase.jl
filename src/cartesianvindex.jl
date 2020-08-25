
struct CartesianVIndex{N,T} <: Base.AbstractCartesianIndex{N}
    I::T
    @inline CartesianVIndex(I::T) where {N, T <: Tuple{Vararg{Union{Integer,Static},N}}} = new{N,T}(I)
end
Base.length(::CartesianVIndex{N}) where {N} = N
Base.length(::Type{<:CartesianVIndex{N}}) where {N} = N
Base.Tuple(i::CartesianVIndex) = i.I
function Base.:(:)(I::CartesianVIndex{N}, J::CartesianVIndex{N}) where {N} 
   CartesianIndices(map((i,j) -> i:j, I.I, J.I))
end
ndim(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
ndim(::Type{<:AbstractArray{N}}) where {N} = N
@generated function CartesianVIndex(I::T) where {N, T <: Tuple{Vararg{Integer,Static,CartesianIndex,CartesianVIndex}}}
    iexp = Expr(:tuple)
    Tp = T.paramters
    q = Expr(:block)
    for i in eachindex(Tp)
        I_i = Symbol(:I_, i)
        if Tp[i] <: Base.AbstractCartesianIndex
            push!(q.args, Expr(:(=), I_i, Expr(:ref, :I, i)))
            for n in 1:ndim(Tp[i])
                push!(iexp.args, Expr(:ref, I_i, n))
            end
        else
            push!(iexpr.args, I_i)
        end
        
    end
    Expr(:block, Expr(:meta, :inline), Expr(:macrocall, Symobl("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), q), Expr(:call, :CartesianVIndex, iexpr))
end


