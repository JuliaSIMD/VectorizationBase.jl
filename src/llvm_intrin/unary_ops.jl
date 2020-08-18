
@inline fmapt(f::F, x::Tuple{X}) where {F,X} = (f(first(x)),)
@inline fmapt(f::F, x::NTuple) where {F} = (f(first(x)), fmap(f, Base.tail(x))...)

@inline fmap(f::F, x::VecUnroll) where {F} = VecUnroll(fmapt(f, x.data))
@inline fmap(f::F, x::VecTile) where {F} = VecTile(fmapt(f, x.data))

for T ∈ [Float32,Float64]
    W = 2
    typ = LLVM_TYPES[T]
    while W ≤ pick_vector_width(T)
        vtyp = "<$W x $typ>"
        instrs = "%res = fneg fast $vtyp %0\nret $vtyp %res"
        @eval @inline Base.:(-)(v::Vec{$W,$T}) = Vec(llvmcall($instrs, _Vec{$W,$T}, Tuple{_Vec{$W,$T}}, data(v)))
        W += W
    end
end

@inline vsub(v::Vec{<:Any,<:FloatingTypes}) = -v
@inline vsub(v::Vec{1,<:FloatingTypes}) = -first(v)
@inline vsub(v::NativeTypes) = -v
@inline vsub(v::Vec{<:Any,<:NativeTypes}) = zero(v) - v


