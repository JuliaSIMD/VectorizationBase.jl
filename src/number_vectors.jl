
num_vector_load_expr(N::Symbol, W) = :(divrem($N, $W))
function num_vector_load_expr(mod, N::Expr, W)
    if N.args[1] == :length
        q = :(($mod).VectorizationBase.length_loads($(N.args[2]), Val{$W}()))
    elseif N.args[1] == :size && length(N.args) == 3
        q = :(($mod).VectorizationBase.size_loads($(N.args[2]), $(N.args[3]), Val{$W}()))
    else
        Wshift = intlog2(W)
        q = :($N >> $Wshift, $N & $(W-1))
    end
    q
end
function num_vector_load_expr(mod, N::Symbol, W::Integer)
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
