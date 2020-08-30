# Scalar operations

# @inline Base.@pure vshr(a::Int64, b) = llvmcall("%res = ashr i64 %0, %1\nret i64 %res", Int64, Tuple{Int64,Int64}, a, b % Int64)
# @inline Base.@pure vshr(a::Int32, b) = llvmcall("%res = ashr i32 %0, %1\nret i32 %res", Int32, Tuple{Int32,Int32}, a, b % Int32)

# @inline Base.@pure vshr(a::Int16, b) = llvmcall("%res = ashr i16 %0, %1\nret i16 %res", Int16, Tuple{Int16,Int16}, a, b % Int16)
# @inline Base.@pure vshr(a::Int8, b) = llvmcall("%res = ashr i8 %0, %1\nret i8 %res", Int8, Tuple{Int8,Int8}, a, b % Int8)

# @inline Base.@pure vshr(a::UInt64, b) = llvmcall("%res = lshr i64 %0, %1\nret i64 %res", UInt64, Tuple{UInt64,UInt64}, a, b % UInt64)
# @inline Base.@pure vshr(a::UInt32, b) = llvmcall("%res = lshr i32 %0, %1\nret i32 %res", UInt32, Tuple{UInt32,UInt32}, a, b % UInt32)

# @inline Base.@pure vshr(a::UInt16, b) = llvmcall("%res = lshr i16 %0, %1\nret i16 %res", UInt16, Tuple{UInt16,UInt16}, a, b % UInt16)
# @inline Base.@pure vshr(a::UInt8, b) = llvmcall("%res = lshr i8 %0, %1\nret i8 %res", UInt8, Tuple{UInt8,UInt8}, a, b % UInt8)



function binary_op(op, W, @nospecialize(_::Type{T})) where {T}
    ty = LLVM_TYPES[T]
    if isone(W)
        V = T
    else
        ty = "<$W x $ty>"
        V = NTuple{W,VecElement{T}}
    end
    instrs = "%res = $op $ty %0, %1\nret $ty %res"
    call = :(llvmcall($instrs, $V, Tuple{$V,$V}, data(v1), data(v2)))
    W > 1 && (call = Expr(:call, :Vec, call))
    Expr(:block, Expr(:meta, :inline), call)
end
# @generated function binary_operation(::Val{op}, v1::V1, v2::V2) where {op, V1, V2}
#     M1, N1, W1, T1 = description(V1)
#     M2, N2, W2, T2 = description(V2)
    
#     lc = Expr(:call, :llvmcall, join(instrs, "\n"), )
# end
# function binary_op(op, W, @nospecialize(_::Type{T})) where {T}
#     ty = 'i' * string(8*sizeof(T))
#     binary_op(op, W, T, ty)
# end

# Integer
    # vop = Symbol('v', op)
for (op,f) ∈ [("add",:+),("sub",:-),("mul",:*),("shl",:<<)]
    nswop = op * " nsw"
    nuwop = op * " nuw"
    ff = Symbol('v', op)

    @eval @generated $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op * (T <: Signed ? " nsw" : " nuw"), W, T)
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op($op, W, T)
    
    for Ts ∈ [Int8,Int16,Int32,Int64]
        Tu = unsigned(Ts)
        st = sizeof(Ts)
        @eval begin
            Base.@pure @inline $ff(v1::$Ts, v2::$Ts) = $(binary_op(nswop, 1, Ts))
            Base.@pure @inline $ff(v1::$Tu, v2::$Tu) = $(binary_op(nuwop, 1, Tu))
        end
        # W = 2
        # while W ≤ pick_vector_width(Ts)
        #     @eval begin
        #         Base.@pure @inline $ff(v1::Vec{$W,$Ts}, v2::Vec{$W,$Ts}) = $(binary_op(nswop, W, Ts))
        #         Base.@pure @inline $ff(v1::Vec{$W,$Tu}, v2::Vec{$W,$Tu}) = $(binary_op(nuwop, W, Tu))
        #         Base.@pure @inline Base.$f(v1::Vec{$W,$Ts}, v2::Vec{$W,$Ts}) = $(binary_op(op, W, Ts))
        #         Base.@pure @inline Base.$f(v1::Vec{$W,$Tu}, v2::Vec{$W,$Tu}) = $(binary_op(op, W, Tu))
        #     end
        #     W += W
        # end
    end
end
for (op,f) ∈ [("div",:÷),("rem",:%)]
    uop = 'u' * op
    sop = 's' * op
    ff = Symbol('v', op)

    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Integer} = binary_op((T <: Signed ? 's' : 'u') * $op, W, Ts)
    
    for Ts ∈ [Int8,Int16,Int32,Int64]
        Tu = unsigned(Ts)
        st = sizeof(Ts)
        @eval begin
            Base.@pure @inline $ff(v1::$Ts, v2::$Ts) = $(binary_op(sop, 1, Ts))
            Base.@pure @inline $ff(v1::$Tu, v2::$Tu) = $(binary_op(uop, 1, Tu))
        end
        # W = 2
        # while W ≤ pick_vector_width(Ts)
        #     @eval begin
        #         Base.@pure @inline Base.$f(v1::Vec{$W,$Ts}, v2::Vec{$W,$Ts}) = $(binary_op(sop, W, Ts))
        #         Base.@pure @inline Base.$f(v1::Vec{$W,$Tu}, v2::Vec{$W,$Tu}) = $(binary_op(uop, W, Tu))
        #     end
        #     W += W
        # end
    end
    # @eval @inline $ff(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T} = $f(v1, v2)
    # @eval const $ff = $f
end
@inline vcld(x, y) = vadd(vdiv(vsub(x,one(x)), y), one(x))
@inline function vdivrem(x, y)
    d = vdiv(x, y)
    r = vsub(x, vmul(d, y))
    d, r
end
for (op,f,s) ∈ [("lshr",:>>,0x01),("ashr",:>>,0x02),("ashr",:>>>,0x03),("and",:&,0x03),("or",:|,0x03),("xor",:⊻,0x03)]
    ff = Symbol('v', op)
    fdef = Expr(:where, :(Base.$f(v1::Vec{W,T}, v2::Vec{W,T})), :W)
    if iszero(s & 0x01)
        push!(fdef.args, :(T <: Unsigned))
    elseif iszero(s & 0x02)
        push!(fdef.args, :(T <: Signed))
    else
        push!(fdef.args, :T)
    end
    @eval @generated $fdef = binary_op($op, W, T)
    (op === "lshr" || op === "ashr") || continue
    for Ts ∈ [Int8,Int16,Int32,Int64]
        Tu = unsigned(Ts)
        # W = 2
        # while W ≤ pick_vector_width(Ts)
        #     if !iszero(s & 0x01) # signed def
        #         @eval begin
        #             Base.@pure @inline Base.$f(v1::Vec{$W,$Ts}, v2::Vec{$W,$Ts}) = $(binary_op(op, W, Ts))
        #             # @inline $ff(v1::Vec{$W,$Ts}, v2::Vec{$W,$Ts}) = $f(v1, v2)
        #         end
        #     end
        #     if !iszero(s & 0x02) # unsigend def
        #         @eval begin
        #             Base.@pure @inline Base.$f(v1::Vec{$W,$Tu}, v2::Vec{$W,$Tu}) = $(binary_op(op, W, Tu))
        #             # @inline $ff(v1::Vec{$W,$Tu}, v2::Vec{$W,$Tu}) = $f(v1, v2)
        #         end
        #     end
        #     W += W
        # end
        if !iszero(s & 0x01) # signed def
            @eval begin
                Base.@pure @inline $ff(v1::$Ts, v2::$Ts) = $(binary_op(op, 1, Ts))
                # Base.@pure $ff(v1::Vec{W,$Ts}, v2::Vec{W,$Ts}) where {W} = $f(v1, v2)
            end
        end
        if !iszero(s & 0x02) && !((s === 0x03) && (op === "ashr")) # unsigend def
            @eval begin
                Base.@pure @inline $ff(v1::$Tu, v2::$Tu) = $(binary_op(op, 1, Tu))
                # Base.@pure $ff(v1::Vec{W,$Tu}, v2::Vec{W,$Tu}) where {W} = $f(v1, v2)
            end
        end
    end
end
# Bitwise
# for Ts ∈ [Int8, Int16, Int32, Int64]
#     bits = 8sizeof(Ts)
#     Tu = unsigned(Ts)
#     ashr_instr = "%res = ashr i$bits %0, %1\nret i$bits %res"
#     lshr_instr = "%res = lshr i$bits %0, %1\nret i$bits %res"
#     @eval @inline Base.@pure shr(a::$Ts, b) = llvmcall($ashr_instr, $Ts, Tuple{$Ts,$Ts}, a, b % $Ts)
#     @eval @inline Base.@pure shr(a::$Tu, b) = llvmcall($lshr_instr, $Tu, Tuple{$Tu,$Tu}, a, b % $Tu)
# end

for (op,f,ff) ∈ [("fadd",:+,:vadd),("fsub",:-,:vsub),("fmul",:*,:vmul),("fdiv",:/,:vfdiv),("frem",:%,:vrem)]
    @eval @generated Base.$f(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T} = binary_op($(op * " fast"), W, T)
    # for T ∈ [Float32, Float64]
    #     W = 2
    #     while W ≤ pick_vector_width(T)
    #         @eval begin
    #             Base.@pure @inline Base.$f(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(binary_op(op * " fast", W, T))
    #             Base.@pure @inline $ff(v1::Vec{$W,$T}, v2::Vec{$W,$T}) = $(binary_op(op, W, T))
    #         end
    #         W += W
    #     end
    # end
end
@inline Base.inv(v::Vec) = vdiv(one(v), v)

@inline Base.:(/)(a::Vec{W,<:Integer}, b::Vec{W,<:Integer}) where {W} = float(a) / float(b)

    # for op ∈ [:+,:vadd,:-,:vsub,:*,:vmul,:/,:vfdiv,:%,:vrem,:<<,:vshl,:>>,:vlshr,:>>>,:vashr,:&,:vand,:|,:vor,:⊻,:vxor,:÷,:vdiv]
for op ∈ [:+,:-,:*,:/,:%,:<<,:>>,:>>>,:&,:|,:⊻,:÷]
    @eval begin
        @inline Base.$op(v1::VecUnroll, v2) = fmap($op, v1.data, v2)
        @inline Base.$op(v1, v2::VecUnroll) = fmap($op, v1, v2.data)
        @inline Base.$op(v1::VecUnroll, v2::VecUnroll) = fmap($op, v1.data, v2.data)
    end
end

