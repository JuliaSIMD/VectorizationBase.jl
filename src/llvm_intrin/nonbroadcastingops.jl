
@generated function addscalar(v::Vec{W,T}, s::T) where {W, T <: Integer}
    typ = "i$(8sizeof(T))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp zeroinitializer, $typ %1, i32 0")
    push!(instrs, "%v = add $vtyp %0, %ie")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end
@generated function addscalar(v::Vec{W,T}, s::T) where {W, T <: Union{Float16,Float32,Float64}}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp zeroinitializer, $typ %1, i32 0")
    push!(instrs, "%v = fadd $(fastflags(T)) $vtyp %0, %ie")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end

@generated function mulscalar(v::Vec{W,T}, s::T) where {W, T <: Integer}
    typ = "i$(8sizeof(T))"
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp $(llvmconst(W, T, 1)), $typ %1, i32 0")
    push!(instrs, "%v = mul $vtyp %0, %ie")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end
@generated function mulscalar(v::Vec{W,T}, s::T) where {W, T <: Union{Float16,Float32,Float64}}
    typ = llvmtype(T)
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp $(llvmconst(W, T, 1.0)), $typ %1, i32 0")
    push!(instrs, "%v = fmul $(fastflags(T)) $vtyp %0, %ie")
    push!(instrs, "ret $vtyp %v")
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end

function scalar_maxmin(W, ::Type{T}, ismax) where {T}
    comp = if T <: Signed
        ismax ? "icmp sgt" : "icmp slt"
    elseif T <: Unsigned
        ismax ? "icmp ugt" : "icmp ult"
    elseif ismax
        "fcmp ogt"
    else
        "fcmp olt"
    end
    if T <: Integer
        typ = "i$(8sizeof(T))"
        basevalue = llvmconst(W, T, ismax ? typemin(T) : typemax(T))
    else
        opzero = ismax ? -Inf : Inf
        if T === Float64
            typ = "double"
            basevalue = llvmconst(W, T, repr(reinterpret(UInt64, opzero)))
        elseif T === Float32
            typ = "float"
            basevalue = llvmconst(W, T, repr(reinterpret(UInt32, Float32(opzero))))
        elseif T === Float16
            typ = "half"
            basevalue = llvmconst(W, T, repr(reinterpret(UInt16, Float16(opzero))))
        else
            throw("T === $T not currently supported.")
        end
    end
    vtyp = "<$W x $typ>"
    instrs = String[]
    push!(instrs, "%ie = insertelement $vtyp $(basevalue), $typ %1, i32 0")
    push!(instrs, "%selection = $comp $vtyp %0, %ie")
    push!(instrs, "%v = select <$W x i1> %selection, $vtyp %0, $vtyp %ie")
    push!(instrs, "ret $vtyp %v")
    instrs
end
@generated function maxscalar(v::Vec{W,T}, s::T) where {W, T}
    instrs = scalar_maxmin(W, T, true)
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end
@generated function minscalar(v::Vec{W,T}, s::T) where {W, T}
    instrs = scalar_maxmin(W, T, false)
    quote
        $(Expr(:meta,:inline))
        Vec(llvmcall( $(join(instrs,"\n")), NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}},$T}, data(v), s ))
    end
end
for (f,op) ∈ [(:addscalar,:(+)), (:mulscalar,:(*)), (:maxscalar,:max), (:minscalar,:min)]
    @eval begin
        @inline $f(v::VecUnroll, s) = VecUnroll(fmap($f, v.data, s))
        @inline $f(s::T, v::AbstractSIMD{W,T}) where {W,T} = $f(v, s)
        @inline $f(a, b) = $op(a, b)
    end
end

