
struct CartesianVIndex{N,T} <: Base.AbstractCartesianIndex{N}
    I::T
    @inline CartesianVIndex(I::T) where {N, T <: Tuple{Vararg{<:Any,N}}} = new{N,T}(I)
end
Base.length(::CartesianVIndex{N}) where {N} = N
Base.length(::Type{<:CartesianVIndex{N}}) where {N} = N
Base.Tuple(i::CartesianVIndex) = i.I
function Base.:(:)(I::CartesianVIndex{N}, J::CartesianVIndex{N}) where {N} 
   CartesianIndices(map((i,j) -> i:j, I.I, J.I))
end



