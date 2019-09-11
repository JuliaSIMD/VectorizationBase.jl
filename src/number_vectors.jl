
# num_vector_load_expr(N::Symbol, W) = :(divrem($N, $W))
function num_vector_load_expr(mod, N::Expr, W::Integer)
    if N.args[1] == :length
        :(($mod).VectorizationBase.length_loads($(N.args[2]), Val{$W}()))
    elseif N.args[1] == :size && length(N.args) == 3
        :(($mod).VectorizationBase.size_loads($(N.args[2]), $(N.args[3]), Val{$W}()))
    else
        Wshift = intlog2(W)
        :($N >> $Wshift, $N & $(W-1))
    end
end
function num_vector_load_expr(mod, N, W::Integer)
    Wshift = intlog2(W)
    :($N >> $Wshift, $N & $(W-1))
end
### Generic fallbacks
### PaddedMatrices provide methods that determinalistically provide r = 0
@generated function length_loads(A, ::Val{W}) where W
    Wshift = intlog2(W)
    quote
        $(Expr(:meta, :inline))
        N = length(A)
        N >> $Wshift, N & $(W - 1)
    end
end
@generated function size_loads(A, dim, ::Val{W}) where W
    Wshift = intlog2(W)
    quote
        $(Expr(:meta, :inline))
        N = size(A, dim)
        N >> $Wshift, N & $(W - 1)
    end
end
