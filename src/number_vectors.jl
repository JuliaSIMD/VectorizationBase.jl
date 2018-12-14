
num_vector_load_expr(N::Symbol, W) = :(divrem($N, $W))
function num_vector_load_expr(mod, N::Expr, W)
    if N.args[1] == :length
        q = :(($mod).VectorizationBase.length_loads($(N.args[2]), Val{$W}()))
    elseif N.args[1] == :size && length(N.args) == 3
        q = :(($mod).VectorizationBase.size_loads($(N.args[2]), $(N.args[3]), Val{$W}()))
    else
        q = :(divrem($N, $W))
    end
    q
end
### Generic fallbacks
### PaddedMatrices provide methods that determinalistically provide r = 0
@inline function length_loads(A, ::Val{W}) where W
    divrem(length(A), W)
end
@inline function size_loads(A, dim, ::Val{W}) where W
    divrem(size(A, dim), W)
end
