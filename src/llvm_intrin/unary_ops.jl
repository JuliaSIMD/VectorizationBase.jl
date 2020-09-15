
@generated function Base.:(-)(v::Vec{W,T}) where {W, T <: Union{Float32,Float64}}
    vtyp = vtype(W, T)
    instrs = "%res = fneg nsz arcp contract afn reassoc $vtyp %0\nret $vtyp %res"
    quote
        $(Expr(:meta, :inline))
        Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T}}, data(v)))
    end
end
# for T ∈ [Float32,Float64]
#     W = 2
#     typ = LLVM_TYPES[T]
#     while W ≤ pick_vector_width(T)
#         vtyp = "<$W x $typ>"
#         instrs = "%res = fneg fast $vtyp %0\nret $vtyp %res"
#         @eval @inline Base.:(-)(v::Vec{$W,$T}) = Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T}}, data(v)))
#         W += W
#     end
# end

# @inline Base.:(-)(v::VecUnroll) = fmap(-, v)
# @inline Base.:(-)(v::VecTile) = fmap(-, v)
# @inline Base.:(-)(v::Vec{<:Any,<:FloatingTypes}) = -v
@inline Base.:(-)(v::Vec{1,<:FloatingTypes}) = -first(v)
@inline Base.:(-)(v::Vec{<:Any,<:NativeTypes}) = zero(v) - v

@inline Base.inv(v::Vec) = vdiv(one(v), v)


