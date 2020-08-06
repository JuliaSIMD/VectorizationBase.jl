
@generated function inlinetuplemap(f::F, args::Vararg{<:NTuple{U},N}) where {F,U,N}
    tup = Expr(:tuple)
    for u ∈ 1:U
        call = Expr(:call, :f)
        for n ∈ 1:N
            push!(call.args, Expr(:ref, Expr(:ref, :args, n), :u))
        end
        push!(tup.args, call)
    end
    Expr(
        :block,
        Expr(:meta, :inline),
        tup
    )
end

function binary_op(op, W, @nospecialize(_::Type{T}), ty) where {T}
    if isone(W)
        V = T
        v1 = :(v1[1])
        v2 = :(v1[2])
    else
        ty = "<$W x $ty>"
        V = NTuple{W,Core.VecElement{T}}
        v1 = :(extract_data(v1))
        v2 = :(extract_data(v2))
    end
    instrs = """
        %res = $op $ty %0, %1
        ret $ty %res
    """
    quote
        $(Expr(:meta, :inline))
        llvmcall($instrs, $V, Tuple{$V,$V}, $v1, $v2)
    end
end
function integer_binary_op(op, W, @nospecialize(_::Type{T})) where {T}
    ty = 'i' * string(8*sizeof(T))
    binary_op(op, W, T, ty)
end

for (op,f,ff) ∈ [("add",:+,:vadd),("sub",:-,:vsub),]
    nswop = op * " nsw"
    nuwop = op * " nuw"
    for ST ∈ [Int8,Int16,Int32,Int64]
        UT = unsigned(ST)
        st = sizeof(ST)
        W = 1
        while W * st ≤ REGISTER_SIZE
            @eval begin
                Base.@pure @inline Base.$f(v1, v2) = $(integer_binary_op(op, W, ST))
                Base.@pure @inline Base.$f(v1, v2) = $(integer_binary_op(op, W, UT))
            end
            W += W
        end
    end
end

for (op,f,ff) ∈ [("fadd",:+,:vadd),("fsub",:-,:vsub),]
    
end

