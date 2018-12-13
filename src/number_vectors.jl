
num_vector_load_expr(N::Symbol, W) = :(divrem($N, $W))
function num_vector_load_expr(N::Expr, W)
    if N.args[1] == :length
        q = :(length_loads($(N.args[2]), Val{$W}()))
    elseif N.args[1] == :size && length(N.args) == 3
        q = :(size_loads($(N.args[2]), $(N.args[3]), Val{$W}()))
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
@inlie function size_loads(A, dim, ::Val{W}) where W
    divrem(size(A, dim), W)
end
