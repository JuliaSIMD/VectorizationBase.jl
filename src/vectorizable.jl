##


# Convert Julia types to LLVM types
const LLVMTYPE = Dict{DataType,String}(
    Bool => "i8",   # Julia represents Tuple{Bool} as [1 x i8]
    Int8 => "i8",
    Int16 => "i16",
    Int32 => "i32",
    Int64 => "i64",
    Int128 => "i128",
    UInt8 => "i8",
    UInt16 => "i16",
    UInt32 => "i32",
    UInt64 => "i64",
    UInt128 => "i128",
    Float16 => "half",
    Float32 => "float",
    Float64 => "double",
    Nothing => "void"
)
llvmtype(x)::String = LLVMTYPE[x]
const JuliaPointerType = LLVMTYPE[Int]

const NativeTypes = Union{Bool, Base.HWReal}
# The `@eval` loop is to avoid type ambiguities on Julia 1.0. Later versions did not have this problem

# function vload_quote(::Type{T})
# end

const SCOPE_METADATA = """
!1 = !{!\"noaliasdomain\"}
!2 = !{!\"noaliasscope\", !1}
!3 = !{!2}
"""
const LOAD_SCOPE_TBAA = SCOPE_METADATA * """
!4 = !{!"jtbaa"}
!5 = !{!6, !6, i64 0, i64 0}
!6 = !{!"jtbaa_arraybuf", !4, i64 0}
"""
const STORE_TBAA = """
!4 = !{!"jtbaa", !5, i64 0}
!5 = !{!"jtbaa"}
!6 = !{!"jtbaa_data", !4, i64 0}
!7 = !{!8, !8, i64 0}
!8 = !{!"jtbaa_arraybuf", !6, i64 0}
"""

function vload_quote(::Type{T}) where {T <: NativeTypes}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    decl = LOAD_SCOPE_TBAA
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%res = load $typ, $typ* %ptr, align $alignment, !alias.scope !3, !tbaa !5")
    push!(instrs, "ret $typ %res")
    :(llvmcall($((decl,join(instrs, "\n"))), $T, Tuple{Ptr{$T}}, ptr))
end
function vload_quote(::Type{T}, ::Type{I}) where {T <: NativeTypes, I <: Integer}
    ityp = 'i' * string(8sizeof(I))
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    decl = LOAD_SCOPE_TBAA
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%typptr = inttoptr $ptyp %0 to i8*")
    push!(instrs, "%iptr = getelementptr inbounds i8, i8* %typptr, $ityp %1")
    push!(instrs, "%ptr = bitcast i8* %iptr to $typ*")
    push!(instrs, "%res = load $typ, $typ* %ptr, align $alignment, !alias.scope !3, !tbaa !5")
    push!(instrs, "ret $typ %res")
    :(llvmcall($((decl, join(instrs, "\n"))), $T, Tuple{Ptr{$T}, $I}, ptr, i))
end
function vstore_quote(::Type{T}, alias) where {T <: NativeTypes}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    decl = alias ? STORE_TBAA : SCOPE_METADATA * STORE_TBAA
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    aliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !tbaa !7"
    noaliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !noalias !3, !tbaa !7"
    push!(instrs, alias ? aliasstoreinstr : noaliasstoreinstr)
    push!(instrs, "ret void")
    :(llvmcall($((decl, join(instrs, "\n"))), Cvoid, Tuple{Ptr{$T}, $T}, ptr, v))
end
function vstore_quote(::Type{T}, ::Type{I}, alias) where {T <: NativeTypes, I <: Integer}
    ityp = 'i' * string(8sizeof(I))
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    instrs = String[]
    decl = alias ? STORE_TBAA : SCOPE_METADATA * STORE_TBAA
    alignment = Base.datatype_alignment(T)
    # push!(flags, "!noalias !0")
    push!(instrs, "%typptr = inttoptr $ptyp %0 to i8*")
    push!(instrs, "%iptr = getelementptr inbounds i8, i8* %typptr, $ityp %2")
    push!(instrs, "%ptr = bitcast i8* %iptr to $typ*")
    aliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !tbaa !7"
    noaliasstoreinstr = "store $typ %1, $typ* %ptr, align $alignment, !noalias !3, !tbaa !7"
    push!(instrs, alias ? aliasstoreinstr : noaliasstoreinstr)
    push!(instrs, "ret void")
    :(llvmcall($((decl,join(instrs, "\n"))), Cvoid, Tuple{Ptr{$T}, $T, $I}, ptr, v, i))
end

function gepquote(::Type{T}, ::Type{I}, byte::Bool) where {T <: NativeTypes, I <: Integer}
    ptyp = JuliaPointerType
    ityp = llvmtype(I)
    typ = byte ? "i8" : llvmtype(T)
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%offsetptr = getelementptr inbounds $typ, $typ* %ptr, $ityp %1")
    push!(instrs, "%iptr = ptrtoint $typ* %offsetptr to $ptyp")
    push!(instrs, "ret $ptyp %iptr")
    quote
        llvmcall(
            $(join(instrs, "\n")),
            Ptr{$T}, Tuple{Ptr{$T}, $I},
            ptr, i
        )
    end    
end

@inline vload(x::Number, args...) = x
for T ∈ [Bool,Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64,Float32,Float64]
    @eval @inline vload(ptr::Ptr{$T}) = $(vload_quote(T))
    @eval @inline vstore!(ptr::Ptr{$T}, v::$T) = $(vstore_quote(T, true))
    @eval @inline vnoaliasstore!(ptr::Ptr{$T}, v::$T) = $(vstore_quote(T, false))
    for I ∈ [Int32,UInt32,Int64,UInt64]
        @eval @inline vload(ptr::Ptr{$T}, i::$I) = $(vload_quote(T, I))
        @eval @inline vstore!(ptr::Ptr{$T}, v::$T, i::$I) = $(vstore_quote(T, I, true))
        @eval @inline vnoaliasstore!(ptr::Ptr{$T}, v::$T, i::$I) = $(vstore_quote(T, I, false))
        # @eval @inline gep(ptr::Ptr{$T}, i::$I) = $(gepquote(T, I, false))
        @eval @inline Base.@pure gep(ptr::Ptr{$T}, i::$I) = $(gepquote(T, I, false))
        @eval @inline Base.@pure gepbyte(ptr::Ptr{$T}, i::$I) = $(gepquote(T, I, true))
        # @eval @inline gep(ptr::Ptr{$T}, i::$I) = $(gepquote(T, I, false))
        # @eval @inline gepbyte(ptr::Ptr{$T}, i::$I) = $(gepquote(T, I, true))
    end    
end
@inline vload(ptr, ::Static{N}) where {N} = vload(ptr, N)
@inline vstore!(ptr, v, ::Static{N}) where {N} = vstore!(ptr, v, N)

# Fall back definitions
@inline vload(ptr::Ptr) = Base.unsafe_load(ptr)
# @inline vload(ptr::Ptr, i::Int) = vload(gep(ptr, i))
@inline vstore!(ptr::Ptr{T}, v::T) where {T} = Base.unsafe_store!(ptr, v)
@inline vload(::Type{T1}, ptr::Ptr{T2}) where {T1, T2} = vload(Base.unsafe_convert(Ptr{T1}, ptr))
@inline vstore!(ptr::Ptr{T1}, v::T2) where {T1,T2} = vstore!(ptr, convert(T1, v))
@inline vstore!(ptr::Ptr{T1}, v::T2) where {T1<:Integer,T2<:Integer} = vstore!(ptr, v % T1)
# @inline vstore!(ptr::Ptr{T1}, v::T2) where {T1,T2} = vstore!(ptr, convert(T1, v))

@inline gep(ptr::Ptr{Mask{8,UInt8}}, i) = Base.unsafe_convert(Ptr{Mask{8,UInt8}}, gep(Base.unsafe_convert(Ptr{UInt8}, ptr), i >> 3))
@inline gepbyte(ptr::Ptr{Mask{8,UInt8}}, i::Integer) = Base.unsafe_convert(Ptr{Mask{8,UInt8}}, gep(Base.unsafe_convert(Ptr{UInt8}, ptr), i))
@inline vload(ptr::Ptr{Mask{8,UInt8}}, i) = (vload(Base.unsafe_convert(Ptr{UInt8}, ptr), i >> 3) >> (i & 7)) % Bool
@inline vload(ptr::Ptr{Mask{8,UInt8}}, i::_MM{8}) = Mask{8}(vload(Base.unsafe_convert(Ptr{UInt8}, ptr), i.i >> 3))
@inline vload(ptr::Ptr{Mask{8,UInt8}}, i::_MM{16}) = Mask{16}(vload(Base.unsafe_convert(Ptr{UInt16}, gep(ptr, i.i))))
@inline vload(ptr::Ptr{Mask{8,UInt8}}, i::_MM{32}) = Mask{32}(vload(Base.unsafe_convert(Ptr{UInt32}, gep(ptr, i.i))))
@inline vload(ptr::Ptr{Mask{8,UInt8}}, i::_MM{64}) = Mask{64}(vload(Base.unsafe_convert(Ptr{UInt64}, gep(ptr, i.i))))
@inline function vstore!(ptr::Ptr{Mask{8,UInt8}}, v::Bool, i)
    p = gep(Base.unsafe_convert(Ptr{UInt8}, ptr), i >> 3)
    u = vload(p)
    m = 0x01 << (i & 7)
    u = ifelse(v, u | m, u ⊻ m)
    vstore!(p, u); v
end
@inline vstore!(ptr::Ptr{Mask{8,UInt8}}, v::UInt8, i::_MM{8}) = Mask{8}(vstore!(Base.unsafe_convert(Ptr{UInt8}, ptr), i.i >> 3))
@inline vstore!(ptr::Ptr{Mask{8,UInt8}}, v::UInt16, i::_MM{16}) = Mask{16}(vstore!(Base.unsafe_convert(Ptr{UInt16}, gep(ptr, i.i >> 3))))
@inline vstore!(ptr::Ptr{Mask{8,UInt8}}, v::UInt32, i::_MM{32}) = Mask{32}(vstore!(Base.unsafe_convert(Ptr{UInt32}, gep(ptr, i.i >> 3))))
@inline vstore!(ptr::Ptr{Mask{8,UInt8}}, v::UInt64, i::_MM{64}) = Mask{64}(vstore!(Base.unsafe_convert(Ptr{UInt64}, gep(ptr, i.i >> 3))))


@inline tdot(::Tuple{}, ::Tuple{}) = Zero()
@inline tdot(a::Tuple{I1}, b::Tuple{I2}) where {I1,I2} = @inbounds vmul(a[1], b[1])
@inline tdot(a::Tuple{I1,I3}, b::Tuple{I2,I4}) where {I1,I2,I3,I4} = @inbounds vadd(vmul(a[1],b[1]), vmul(a[2],b[2]))
@inline tdot(a::Tuple{I1,I3,I5}, b::Tuple{I2,I4,I6}) where {I1,I2,I3,I4,I5,I6} = @inbounds vadd(vadd(vmul(a[1],b[1]), vmul(a[2],b[2])), vmul(a[3],b[3]))
# @inline tdot(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = @inbounds a[1]*b[1] + tdot(Base.tail(a), Base.tail(b))

@inline tdot(a::Tuple{I}, ::Tuple{}) where {I} = Zero() #@inbounds a[1]
@inline tdot(::Tuple{}, b::Tuple{I}) where {I} = Zero() #@inbounds b[1]
@inline tdot(a::Tuple{I1,Vararg}, b::Tuple{I2}) where {I1,I2} = @inbounds vmul(a[1],b[1])
@inline tdot(a::Tuple{I1}, b::Tuple{I2,Vararg}) where {I1,I2} = @inbounds vmul(a[1],b[1])
@inline tdot(a::Tuple{I1,Vararg}, b::Tuple{I2,Vararg}) where {I1,I2} = @inbounds vadd(vmul(a[1],b[1]), tdot(Base.tail(a), Base.tail(b)))


"""
A wrapper to the base pointer type, that supports pointer arithmetic.
Note that `VectorizationBase.vload` and `VectorizationBase.vstore!` are 0-indexed,
while `Base.unsafe_load` and `Base.unsafe_store!` are 1-indexed.
x = [1, 2, 3, 4, 5, 6, 7, 8];
ptrx = Pointer(x);
vload(ptrx)
# 1
vload(ptrx + 1)
# 2
ptrx[]
# 1
(ptrx+1)[]
# 2
ptrx[1]
# 1
ptrx[2]
# 2
"""
abstract type AbstractPointer{T} end

@generated function gepbyte(ptr::Ptr{T}, i::I) where {T, I <: Integer}
    q = gepquote(T, I, true)
    pushfirst!(q.args, Expr(:meta, :inline))
    q
end
@generated function gep(ptr::Ptr{T}, i::I) where {T <: NativeTypes, I <: Integer}
    q = gepquote(T, I, false)
    pushfirst!(q.args, Expr(:meta, :inline))
    q
end

@generated function gep(ptr::Ptr{T}, i::_Vec{_W,I}) where {_W, T <: NativeTypes, I <: Integer}
    W = _W + 1
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    ityp = llvmtype(I)
    vityp = "<$W x $ityp>"
    vptyp = "<$W x $ptyp>"
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $typ*")
    push!(instrs, "%offsetptr = getelementptr inbounds $typ, $typ* %ptr, $vityp %1")
    push!(instrs, "%iptr = ptrtoint <$W x $typ*> %offsetptr to $vptyp")
    push!(instrs, "ret $vptyp %iptr")
    quote
        $(Expr(:meta, :inline))
        llvmcall(
            $(join(instrs, "\n")),
            NTuple{$W,Core.VecElement{Ptr{$T}}}, Tuple{Ptr{$T}, NTuple{$W,Core.VecElement{$I}}},
            ptr, i
        )
    end
end
@inline gep(ptr::Ptr{T}, i::_Vec{_W,I}, ::Val{0}) where {T <: NativeTypes, _W, I <: Integer} = ptr
@generated function gep(ptr::Ptr{T}, i::_Vec{_W,I}, ::Val{N}) where {T <: NativeTypes, _W, I <: Integer, N}
    if N ∉ [1,2,4,8]
        tz = trailing_zeros(N)
        if iszero(tz) || isone(count_ones(N))
            return :(Base.@_inline_meta; gepbyte(ptr, vmul(i,N)))
        else
            return :(Base.@_inline_meta; gep(ptr, vmul(i, $(N >> tz)), Val{$(1 << tz)}()))
        end
    end
    W = _W + 1
    ptyp = "i$(8 * sizeof(Int))"
    # We can remove a bitcast from the optimized IR
    styp = N == sizeof(T) ? llvmtype(T) : "i$(8 * N)"
    ityp = "i$(8 * sizeof(I))"
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $styp*")
    push!(instrs, "%offsetptr = getelementptr inbounds $styp, $styp* %ptr, <$W x $ityp> %1")
    push!(instrs, "%iptr = ptrtoint <$W x $styp*> %offsetptr to <$W x $ptyp>")
    push!(instrs, "ret <$W x $ptyp> %iptr")
    quote
        Base.@_inline_meta
        llvmcall(
            $(join(instrs, "\n")),
            NTuple{$W,Core.VecElement{Ptr{$T}}}, Tuple{Ptr{$T}, NTuple{$W,Core.VecElement{$I}}},
            ptr, i
        )
    end
end
@inline gep(ptr::Ptr{T}, i::I, ::Val{0}) where {T <: NativeTypes, I <:Integer} = ptr
@generated function gep(ptr::Ptr{T}, i::I, ::Val{N}) where {T <: NativeTypes, I <: Integer, N}
    if N ∉ [1,2,4,8]
        tz = trailing_zeros(N)
        if iszero(tz) || isone(count_ones(N))
            return :(Base.@_inline_meta; gepbyte(ptr, vmul(i,N)))
        else
            return :(Base.@_inline_meta; gep(ptr, vmul(i, $(N >> tz)), Val{$(1 << tz)}()))
        end
    end
    ptyp = "i$(8 * sizeof(Int))"
    # We can remove a bitcast from the optimized IR
    styp = N == sizeof(T) ? llvmtype(T) : "i$(8 * N)"
    ityp = "i$(8 * sizeof(I))"
    instrs = String[]
    push!(instrs, "%ptr = inttoptr $ptyp %0 to $styp*")
    push!(instrs, "%offsetptr = getelementptr inbounds $styp, $styp* %ptr, $ityp %1")
    push!(instrs, "%iptr = ptrtoint $styp* %offsetptr to $ptyp")
    push!(instrs, "ret $ptyp %iptr")
    quote
        Base.@_inline_meta
        llvmcall(
            $(join(instrs, "\n")),
            Ptr{$T}, Tuple{Ptr{$T}, $I},
            ptr, i
        )
    end
end

@inline gep(ptr::Ptr, ::Static{0}) = ptr
@inline gepbyte(ptr::Ptr, ::Static{0}) = ptr
# @inline gepbyte(ptr::AbstractPointer, ::Static{0}) = pointer(ptr)
@inline gep(ptr::Ptr, ::Static{N}) where {N} = gep(ptr, N)
@inline gepbyte(ptr::Ptr, ::Static{N}) where {N} = gepbyte(ptr, N)
@inline gepbyte(ptr::Ptr, i::LazyStaticMul{N}) where {N} = gep(ptr, i.data, Val{N}())
@inline gep(ptr::Ptr, v::SVec) = gep(ptr, extract_data(v))
@inline gep(ptr::Ptr{Cvoid}, i::Integer) = gepbyte(ptr, i)

# struct Reference{T} <: AbstractPointer{T}
    # ptr::Ptr{T}
    # @inline Reference(ptr::Ptr{T}) where {T} = new{T}(ptr)
# end
@inline Base.eltype(::AbstractPointer{T}) where {T} = T
# @inline gep(ptr::Pointer, i::Tuple{<:Integer}) = gep(ptr, first(i))
# @inline vstore!(ptr::AbstractPointer{T1}, v::T2, args...) where {T1,T2} = vstore!(ptr, convert(T1, v), args...)



@inline vload(::Type{SVec{W,T}}, ptr::AbstractPointer) where {W,T} = vload(SVec{W,T}, pointer(ptr))
@inline vloadstride(ptr::AbstractPointer{T}, str1, strd::Union{_MM,AbstractSIMDVector}) where {T} = vload(pointer(ptr), vadd(vmul(static_sizeof(T), str1), strd))
@inline vloadstride(ptr::AbstractPointer{T}, str1, strd::Union{_MM,AbstractSIMDVector}, u::Union{AbstractMask,Unsigned}) where {T} = vload(pointer(ptr), vadd(vmul(static_sizeof(T), str1), strd), u)

@inline vloadstride(ptr::AbstractPointer{T}, str1, strd) where {T} = vload(gepbyte(pointer(ptr), strd), vmulnp(static_sizeof(T), str1))
@inline vloadstride(ptr::AbstractPointer{T}, str1, strd, u::Union{AbstractMask,Unsigned}) where {T} = vload(gepbyte(pointer(ptr), strd), vmulnp(static_sizeof(T), str1), u)

# @inline vloadstride(ptr::AbstractPointer{T}, str1::Integer, strd) where {T} = vload(gep(gepbyte(pointer(ptr), strd), str1))
# @inline vloadstride(ptr::AbstractPointer{T}, str1::Integer, strd, u::Union{AbstractMask,Unsigned}) where {T} = vload(gep(gepbyte(pointer(ptr), strd), str1), u)
# @inline vloadstride(ptr::AbstractPointer{T}, str1::SVec{<:Any,<:Integer}, strd) where {T} = vload(gep(gepbyte(pointer(ptr), strd), str1))
# @inline vloadstride(ptr::AbstractPointer{T}, str1::SVec{<:Any,<:Integer}, strd, u::Union{AbstractMask,Unsigned}) where {T} = vload(gep(gepbyte(pointer(ptr), strd), str1), u)

@inline vload(ptr::AbstractPointer{T}, i::Tuple) where {T} = vloadstride(ptr, stride1offset(ptr, i), stridedoffset(ptr, i))
@inline vload(ptr::AbstractPointer{T}, i::Tuple, u::Union{AbstractMask,Unsigned}) where {T} = vloadstride(ptr, stride1offset(ptr, i), stridedoffset(ptr, i), u)


for (fs, f) ∈ [(:vstorestride!, :vstore!), (:vnoaliasstorestride!, :vnoaliasstore!)]
    @eval begin
        @inline function $fs(ptr::AbstractPointer{T}, v, str1, strd) where {T}
            $f(gepbyte(pointer(ptr), strd), v, vmulnp(static_sizeof(T), str1))
        end
        @inline function $fs(ptr::AbstractPointer{T}, v, str1, strd::Union{_MM,AbstractSIMDVector}) where {T}
            $f(pointer(ptr), v, vadd(vmul(static_sizeof(T), str1), strd))
        end
        @inline function $fs(ptr::AbstractPointer{T}, v, str1, strd, u::Union{AbstractMask,Unsigned}) where {T}
            $f(gepbyte(pointer(ptr), strd), v, vmulnp(static_sizeof(T), str1), u)
        end
        @inline function $fs(ptr::AbstractPointer{T}, v, str1, strd::Union{_MM,AbstractSIMDVector}, u::Union{AbstractMask,Unsigned}) where {T}
            $f(pointer(ptr), v, vadd(vmul(static_sizeof(T), str1), strd), u)
        end
    end
end

@inline vstore!(ptr::AbstractPointer, v, i::Tuple) = vstorestride!(ptr, v, stride1offset(ptr, i), stridedoffset(ptr, i))
@inline vstore!(ptr::AbstractPointer, v, i::Tuple, u::Union{AbstractMask,Unsigned}) = vstorestride!(ptr, v, stride1offset(ptr, i), stridedoffset(ptr, i), u)
@inline vnoaliasstore!(ptr::AbstractPointer, v, i::Tuple) = vnoaliasstorestride!(ptr, v, stride1offset(ptr, i), stridedoffset(ptr, i))
@inline vnoaliasstore!(ptr::AbstractPointer, v, i::Tuple, u::Union{AbstractMask,Unsigned}) = vnoaliasstorestride!(ptr, v, stride1offset(ptr, i), stridedoffset(ptr, i), u)

@inline vnoaliasstore!(args...) = vstore!(args...) # generic fallback

@inline vstore!(ptr::Ptr{T}, v::Number, i::Integer) where {T <: Number} = vstore!(ptr, convert(T, v), i)
@inline vstore!(ptr::Ptr{T}, v::Integer, i::Integer) where {T <: Integer} = vstore!(ptr, v % T, i)
# @inline vstore!(ptr::AbstractPointer{T}, v::T, i...) where {T<:Integer} = vstore!(pointer(ptr), v, offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T1}, v::T2, i...) where {T1<:Integer,T2<:Integer} = vstore!(pointer(ptr), v % T1, offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T1}, v::T2, i...) where {T1<:Number,T2<:Number} = vstore!(pointer(ptr), convert(T1,v), offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T}, v::T, i...) where {T<:Number} = vstore!(pointer(ptr), v, offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T}, v::T, i...) where {T<:Integer} = vstore!(pointer(ptr), v, offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T1}, v::T2, i...) where {T1<:Integer,T2<:Integer} = vstore!(pointer(ptr), v % T1, offset(ptr, i...))
# @inline vstore!(ptr::AbstractPointer{T1}, v::T2, i...) where {T1<:Number,T2<:Number} = vstore!(pointer(ptr), convert(T1,v), offset(ptr, i...))

abstract type AbstractStridedPointer{T} <: AbstractPointer{T} end
abstract type AbstractColumnMajorStridedPointer{T,N} <: AbstractStridedPointer{T} end
struct PackedStridedPointer{T,N} <: AbstractColumnMajorStridedPointer{T,N}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
struct PackedStridedBitPointer{Nm1,N} <: AbstractColumnMajorStridedPointer{Bool,Nm1}
    ptr::Ptr{UInt64}
    strides::NTuple{Nm1,Int}
    offsets::NTuple{N,Int}
end
@inline function gesp(ptr::PackedStridedBitPointer, i::Tuple)
    PackedStridedBitPointer(ptr.ptr, ptr.strides, vadd(i, ptr.offsets))
end
struct PaddedStridedBitPointer{N} <: AbstractColumnMajorStridedPointer{Bool,N}
    ptr::Ptr{Mask{8,UInt8}}#use Mask for dispatch
    strides::NTuple{N,Int}
end
PaddedStridedBitPointer(p::Ptr{<:Unsigned}, s) = PaddedStridedBitPointer(Base.unsafe_convert(Ptr{Mask{8,UInt8}}, p), s)
PaddedStridedBitPointer(A::StridedArray{T}) where {T <: Unsigned} = PaddedStridedBitPointer(pointer(A), staticmul(T,tailstrides(A)))

abstract type AbstractRowMajorStridedPointer{T,N} <: AbstractStridedPointer{T} end
struct RowMajorStridedPointer{T,N} <: AbstractRowMajorStridedPointer{T,N}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
struct RowMajorStridedBitPointer{N} <: AbstractRowMajorStridedPointer{Bool,N}
    ptr::Ptr{UInt64}
    strides::NTuple{N,Int}
end

struct PermutedDimsStridedPointer{S1,S2,T,P<:AbstractStridedPointer{T}} <: AbstractStridedPointer{T}
    ptr::P
end
@inline PermutedDimsStridedPointer{S1,S2}(ptr::P) where {S1,S2, T, P <: AbstractStridedPointer{T}} = PermutedDimsStridedPointer{S1,S2,T,P}(ptr)
@inline function resort_tuple(i::Tuple{Vararg{<:Any,N}}, ::Val{S}) where {N,S}
    ntuple(Val{N}()) do n
        i[S[n]]
    end
end
@inline Base.pointer(ptr::PermutedDimsStridedPointer) = pointer(ptr.ptr)
@inline function stridedpointer(A::PermutedDimsArray{T,N,S1,S2}) where {T,N,S1,S2}
    PermutedDimsStridedPointer{S1,S2}(stridedpointer(parent(A)))
end
@inline pointerforcomparison(ptr::AbstractStridedPointer, i::Tuple) = gep(ptr, i)
@inline pointerforcomparison(ptr::AbstractStridedPointer) = pointer(ptr)


@generated function subsetview(ptr::PermutedDimsStridedPointer{S1,S2}, ::Val{I}, i) where {S1,S2,I}
    s1new = Expr(:tuple)
    s2new = Expr(:tuple)
    resize!(s2new.args, length(S1)-1)
    Iinner = S1[I]
    i = 0
    for s ∈ S1
        if s < Iinner
            i += 1
            push!(s1new.args, s)
            s2new.args[s] = i
        elseif s > Iinner
            i += 1
            push!(s1new.args, s - 1)
            s2new.args[s-1] = i
        end
    end
    sv = :(subsetview(ptr.ptr, Val{$Iinner}(), i))
    quote
        $(Expr(:meta,:inline))
        PermutedDimsStridedPointer{$s1new,$s2new}($sv)
    end
end

# @inline function stridedpointer(A::PermutedDimsArray{T,2,(2,1),(2,1)}) where {T}
#     RowMajorStridedPointer(stridedp
# end

abstract type AbstractSparseStridedPointer{T,N} <: AbstractStridedPointer{T} end
struct SparseStridedPointer{T,N} <: AbstractSparseStridedPointer{T,N}
    ptr::Ptr{T}
    strides::NTuple{N,Int}
end
# struct SparseStridedBitPointer{N} <: AbstractSparseStridedPointer{Bool,N}
#     ptr::Ptr{UInt64}
#     strides::NTuple{N,Int}
# end

abstract type AbstractStaticStridedPointer{T,X} <: AbstractStridedPointer{T} end
struct StaticStridedPointer{T,X} <: AbstractStaticStridedPointer{T,X}
    ptr::Ptr{T}
end
struct StaticStridedBitPointer{X} <: AbstractStaticStridedPointer{Bool,X}
    ptr::Ptr{UInt64}
end
const AbstractBitPointer = Union{PackedStridedBitPointer, RowMajorStridedBitPointer, StaticStridedBitPointer}#, SparseStridedBitPointer
@inline pointerforcomparison(ptr::AbstractBitPointer, i::Tuple) = gesp(ptr, i)
@inline pointerforcomparison(ptr::AbstractBitPointer) = ptr

@inline offset(ptr::AbstractStridedPointer{T}, i::Tuple) where {T} = vmuladdnp(static_sizeof(T), stride1offset(ptr, i), stridedoffset(ptr, i))

@inline stride1offset(::AbstractColumnMajorStridedPointer, i::Tuple{}) = Zero()
@inline stride1offset(::AbstractColumnMajorStridedPointer, i::Tuple) = first(i)
@inline stridedoffset(ptr::AbstractColumnMajorStridedPointer, i::Tuple{I}) where {I} = Zero()
@inline stridedoffset(ptr::AbstractColumnMajorStridedPointer, i::Tuple{I1,I2,Vararg}) where {I1,I2} = tdot(Base.tail(i), ptr.strides)


# @inline offset(ptr::AbstractSparseStridedPointer, i::Integer) = i * @inbounds ptr.strides[1]
# @inline offset(ptr::AbstractStaticStridedPointer{<:Any,<:Tuple{1,Vararg}}, i::Integer) = i
# @inline offset(ptr::AbstractStaticStridedPointer{<:Any,<:Tuple{M,Vararg}}, i::Integer) where {M} = M*i
@inline gep(ptr::AbstractStridedPointer, i::Tuple) = gep(gepbyte(pointer(ptr), stridedoffset(ptr, i)), extract_data(stride1offset(ptr, i)))
# @inline gep(ptr::AbstractStridedPointer, i::Tuple{I}) where {I} = gepbyte(ptr.ptr, first(offset(ptr, i)))
@inline gepbyte(ptr::AbstractStridedPointer, i::Tuple) = gep(gepbyte(pointer(ptr), stridedoffset(ptr, i)), extract_data(stride1offset(ptr, i)))
# @inline gepbyte(ptr::AbstractStridedPointer, i::Tuple{I}) where {I} = gepbyte(ptr.ptr, first(offset(ptr, i)))
@inline gesp(ptr::AbstractStridedPointer, i) = similar(ptr, gep(ptr, i))


@inline Base.similar(p::PackedStridedPointer, ptr::Ptr) = PackedStridedPointer(ptr, p.strides)
@inline Base.similar(p::PaddedStridedBitPointer, ptr::Ptr) = PaddedStridedBitPointer(ptr, p.strides)
@inline Base.similar(p::PackedStridedBitPointer{Nm1,N}, ptr::Ptr) where {Nm1,N} = PackedStridedBitPointer(ptr, p.strides, ntuple(_ -> 0, Val{N}()))
@inline Base.similar(p::RowMajorStridedPointer, ptr::Ptr) = RowMajorStridedPointer(ptr, p.strides)
@inline Base.similar(p::RowMajorStridedBitPointer, ptr::Ptr) = RowMajorStridedBitPointer(ptr, p.strides)
@inline Base.similar(p::SparseStridedPointer, ptr::Ptr) = SparseStridedPointer(ptr, p.strides)
# @inline Base.similar(p::SparseStridedBitPointer, ptr::Ptr) = SparseStridedBitPointer(ptr, p.strides)
@inline Base.similar(::StaticStridedPointer{T,X}, ptr::Ptr) where {T,X} = StaticStridedPointer{T,X}(ptr)
@inline Base.similar(::StaticStridedBitPointer{X}, ptr::Ptr) where {X} = StaticStridedBitPointer{X}(ptr)
@inline Base.similar(p::PermutedDimsStridedPointer{S1,S2}, ptr) where {S1,S2} = PermutedDimsStridedPointer{S1,S2}(similar(p.ptr, ptr))

@inline LinearAlgebra.transpose(ptr::RowMajorStridedPointer) = PackedStridedPointer(ptr.ptr, ptr.strides)
@inline stride1offset(::AbstractRowMajorStridedPointer{T,0}, i::Tuple{I}) where {T,I} = first(i)
@inline stride1offset(::AbstractRowMajorStridedPointer{T,1}, i::Tuple{I1,I2}) where {T,I1,I2} = last(i)
@inline stride1offset(::AbstractRowMajorStridedPointer{T,1}, i::Tuple{I1}) where {T,I1} = Zero()
@inline stride1offset(::AbstractRowMajorStridedPointer{T,N}, i::Tuple{I1,Vararg{<:Any,N}}) where {T,I1,N} = last(i)
# @inline stride1offset(::AbstractRowMajorStridedPointer{T,0}, i::Tuple) where {T} = last(i)

@inline stridedoffset(ptr::AbstractRowMajorStridedPointer{T,0}, i::Tuple) where {T} = Zero()
@inline stridedoffset(ptr::AbstractRowMajorStridedPointer{T,1}, i::Tuple{I1,I2}) where {T,I1,I2} = @inbounds vmul(i[1],ptr.strides[1])
@inline stridedoffset(ptr::AbstractRowMajorStridedPointer{T,2}, i::Tuple{I1,I2,I3}) where {T,I1,I2,I3} = @inbounds vadd(vmul(i[1],ptr.strides[2]), vmul(i[2],ptr.strides[1]))
@inline stridedoffset(ptr::AbstractRowMajorStridedPointer, i::Tuple) = tdot(reverse(ptr.strides), i)

@inline stride1offset(::AbstractSparseStridedPointer, ::Tuple) = Zero()
@inline stridedoffset(ptr::AbstractSparseStridedPointer, i::Tuple) = tdot(i, ptr.strides)

@generated function LinearAlgebra.transpose(ptr::StaticStridedPointer{T,X}) where {T,X}
    tup = Expr(:curly, :Tuple)
    N = length(X.parameters)
    for n ∈ N:-1:1
        push!(tup.args, (X.parameters[n])::Int)
    end
    Expr(:block, Expr(:meta, :inline), Expr(:call, Expr(:curly, :StaticStridedPointer, T, tup), Expr(:(.), :ptr, QuoteNode(:ptr))))
end

# @inline offset(ptr::AbstractStaticStridedPointer{T,<:Tuple{1,Vararg}}, i::Integer) where {T} = vmulnp(static_sizeof(T), i)
# @inline offset(ptr::AbstractStaticStridedPointer{T,<:Tuple{N,Vararg}}, i::Integer) where {N,T} = vmul(N * static_sizeof(T), i)
@inline offset(ptr::AbstractStaticStridedPointer{T,<:Tuple{1,Vararg}}, i::Tuple{I}) where {T,I<:Integer} = vmulnp(static_sizeof(T), first(i))
@inline offset(ptr::AbstractStaticStridedPointer{T,<:Tuple{N,Vararg}}, i::Tuple{I}) where {N,T,I<:Integer} = vmul(vmul(Static(N), static_sizeof(T)), first(i))

# @inline offset(ptr::AbstractStaticStridedPointer{T,Tuple{1,1}}, i::Tuple{<:_MM,<:Any}) where {T} = @inbounds vmuladdnp(static_sizeof(T), i[1], vmul(static_sizeof(T), i[2]))
# @inline offset(ptr::AbstractStaticStridedPointer{T,Tuple{1,1}}, i::Tuple{<:Any,<:_MM}) where {T} = @inbounds vmuladdnp(static_sizeof(T), i[2], vmul(static_sizeof(T), i[1]))
# @inline offset(ptr::AbstractStaticStridedPointer{T,Tuple{1,1}}, i::Tuple{<:Any,<:Any}) where {T} = @inbounds vadd(vmul(static_sizeof(T), i[1]), vmul(static_sizeof(T), i[2]))

function indprod(X, i, st)
    Xᵢ = (X[i])::Int
    iᵢ = Expr(:ref, :i, i)
    Expr(:call, :vmul, st * Xᵢ, iᵢ)
end
@generated function stride1offset(ptr::AbstractStaticStridedPointer{T,X}, i::I) where {T,X,I<:Tuple}
    N = length(I.parameters)
    Xv = X.parameters
    M = min(N, length(X.parameters))
    Xvv = Int[(Xv[m])::Int for m in 1:M]
    mstart = findfirst(isequal(1), Xvv)
    isnothing(mstart) && return Expr(:block, Expr(:meta, :inline), Zero())
    ret = Expr(:ref, :i, mstart)
    for m ∈ 1+mstart:M
        if isone(Xvv[m])
            ret = Expr(:call, :vadd, ret, Expr(:ref, :i, m))
        end
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(
            :macrocall,
            Symbol("@inbounds"),
            LineNumberNode(@__LINE__, Symbol(@__FILE__)), ret
         )
    )
end
@generated function stridedoffset(ptr::AbstractStaticStridedPointer{T,X}, i::I) where {T,X,I<:Tuple}
    N = length(I.parameters)
    Xv = X.parameters
    M = min(N, length(X.parameters))
    Xvv = Int[(Xv[m])::Int for m in 1:M]
    mstart = findfirst(!isequal(1), Xvv)
    isnothing(mstart) && return Expr(:block, Expr(:meta, :inline), Zero())
    st = sizeof(T)
    ind = indprod(Xvv, mstart, st)
    for m ∈ 1+mstart:M
        if !isone(Xvv[m])
            ind = Expr(:call, :vadd, indprod(Xv, m, st), ind)
        end
    end
    Expr(
        :block,
        Expr(:meta,:inline),
        Expr(
            :macrocall,
            Symbol("@inbounds"),
            LineNumberNode(@__LINE__, Symbol(@__FILE__)), ind
         )
    )
end

struct StaticStridedStruct{T,X,S} <: AbstractStaticStridedPointer{T,X}
    ptr::S
    offset::Int # keeps track of offset, incase of nested offset calls
end

const AbstractInitializedStridedPointer{T} = Union{
    PackedStridedPointer{T},
    RowMajorStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T},
    StaticStridedStruct{T}
}
const AbstractInitializedPointer{T} = Union{
    # Pointer{T},
    PackedStridedPointer{T},
    RowMajorStridedPointer{T},
    SparseStridedPointer{T},
    StaticStridedPointer{T},
    StaticStridedStruct{T}
}

# @inline Base.stride(ptr::AbstractColumnMajorStridedPointer, i) = isone(i) ? 1 : @inbounds ptr.strides[i-1]
# @inline Base.stride(ptr::AbstractSparseStridedPointer, i) = @inbounds ptr.strides[i]
# @generated function Base.stride(::AbstractStaticStridedPointer{T,X}, i) where {T,X}
#     Expr(:block, Expr(:meta, :inline), Expr(:getindex, Expr(:tuple, X.parameters...), :i))
# end
# @inline LinearAlgebra.stride1(ptr::AbstractColumnMajorStridedPointer) = 1
# @inline LinearAlgebra.stride1(ptr::AbstractSparseStridedPointer) = @inbounds first(ptr.strides)
# @inline LinearAlgebra.stride1(::AbstractStaticStridedPointer{T,<:Tuple{X,Vararg}}) where {T,X} = X

@inline offset(ptr::AbstractPointer, i::CartesianIndex) = offset(ptr, i.I)

@inline Base.:+(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, gep(pointer(ptr), i))
@inline Base.:+(i, ptr::AbstractPointer{T}) where {T} = similar(ptr, gep(pointer(ptr), i))
@inline Base.:-(ptr::AbstractPointer{T}, i) where {T} = similar(ptr, gep(pointer(ptr), - i))

@inline vload(ptr::AbstractPointer) = vload(pointer(ptr))
@inline Base.unsafe_load(p::AbstractPointer) = vload(pointer(p))
@inline Base.getindex(p::AbstractPointer) = vload(pointer(p))

@inline vload(ptr::AbstractPointer{T}, ::Tuple{}) where {T} = vload(pointer(ptr))
# @inline vload(ptr::AbstractPointer, i::Tuple) = vload(pointer(ptr), offset(ptr, i))
# @inline vload(ptr::AbstractPointer, i::Tuple, u::Unsigned) = vload(pointer(ptr), offset(ptr, i), u)
@inline Base.unsafe_load(ptr::AbstractPointer, i) = vload(pointer(ptr), offset(ptr, vsub(i, 1)))
@inline Base.getindex(ptr::AbstractPointer, i) = vload(ptr, (i, ))
@inline Base.getindex(ptr::AbstractPointer, i, j) = vload(ptr, (i, j))
@inline Base.getindex(ptr::AbstractPointer, i, j, k) = vload(ptr, (i, j, k))
@inline Base.getindex(ptr::AbstractPointer, i, j, k, rests...) = vload(ptr, (i, j, k, rests...))

@inline vstore!(ptr::AbstractPointer{T}, v::T) where {T} = vstore!(pointer(ptr), v)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T) where {T} = vstore!(pointer(ptr), v)
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T) where {T} = vstore!(pointer(ptr), v)

# @inline vstore!(ptr::AbstractPointer{T}, v, i::Tuple) where {T} = vstore!(pointer(ptr), v, offset(ptr, i))
# @inline vstore!(ptr::AbstractPointer{T}, v, i::Tuple, u::Unsigned) where {T} = vstore!(pointer(ptr), v, offset(ptr, i), u)
@inline Base.unsafe_store!(ptr::AbstractPointer{T}, v::T, i) where {T} = vstore!(pointer(ptr), v, offset(ptr, vsub(i, 1)))
@inline Base.setindex!(ptr::AbstractPointer{T}, v::T, i) where {T} = vstore!(pointer(ptr), v, offset(ptr, i))

# @inline Pointer(A) = Pointer(pointer(A))
@inline Base.pointer(ptr::AbstractPointer) = ptr.ptr
@inline Base.unsafe_convert(::Type{Ptr{T}}, ptr::AbstractPointer{T}) where {T} = pointer(ptr)

@inline stridedpointer(x) = x#Pointer(x)

"""
    stridedpointer(x, i)

Gets the stridedpointer pointing to i, using 1-based indexes.
"""
@inline function stridedpointer(x, i)
    gesp(stridedpointer(x), (staticm1(i),))
end
@inline function stridedpointer(x, i1, i2, I...)
    gesp(stridedpointer(x), staticm1((i1, i2, I...)))
end
@inline stridedpointer(x::Ptr) = PackedStridedPointer(x, tuple())
# @inline stridedpointer(x::Union{LowerTriangular,UpperTriangular}) = stridedpointer(parent(x))
# @inline stridedpointer(x::AbstractArray) = stridedpointer(parent(x))
@inline tailstrides(A::AbstractArray) = Base.tail(strides(A))
@inline tailstrides(A::AbstractVector) = tuple()
@inline tailstrides(A::AbstractMatrix) = (stride(A,2),)
@inline tailstrides(A::BitArray{1}) = tuple()
@inline tailstrides(A::BitArray{2}) = (size(A,1),)
@inline tailstrides(A::BitArray{3}) = (size(A,1),size(A,1)*size(A,2))
@generated function tailstrides(A::BitArray{N}) where {N}
    quote
        (Base.Cartesian.@ntuple $(N-1) s) = size(A)
        Base.Cartesian.@nexprs $(N-2) n -> s_{n+1} = vmul(s_n, s_{n+1})
        (Base.Cartesian.@ntuple $(N-1) s)
    end
end
@inline stridedpointer(A::AbstractArray{T}) where {T} = PackedStridedPointer(pointer(A), staticmul(T,tailstrides(A)))
@inline stridedpointer(A::BitArray{N}) where {N} = PackedStridedBitPointer(pointer(A.chunks), tailstrides(A), ntuple(_ -> 0, Val{N}()))
@inline stridedpointer(A::AbstractArray{T,0}) where {T} = pointer(A)
@inline stridedpointer(A::SubArray{T,0,P,S}) where {T,P,S <: Tuple{Int,Vararg}} = pointer(A)
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P,S <: Tuple{<:StepRange,Vararg}} = SparseStridedPointer(pointer(A), staticmul(T, strides(A)))
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P,S <: Tuple{Int,Vararg}} = SparseStridedPointer(pointer(A), staticmul(T, strides(A)))
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P,S} = PackedStridedPointer(pointer(A), staticmul(T, tailstrides(A)))
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P<:Union{Adjoint,Transpose},S} = RowMajorStridedPointer(pointer(A), staticmul(T, Base.tail(Base.reverse(strides(A)))))
# Slow fallback
@inline stridedpointer(A::SubArray{T,N,P,S}) where {T,N,P<:PermutedDimsArray,S} = SparseStridedPointer(pointer(A), staticmul(T, strides(A)))
@inline function stridedpointer(A::SubArray{T,N,P,S}) where {S1,S2,T,N,P<:PermutedDimsArray{<:StridedArray,N,S1,S2},S<:Tuple{Vararg{AbstractUnitRange}}}
    ptr = similar(stridedpointer(parent(parent(A))), pointer(A))
    PermutedDimsStridedPointer{S1,S2}(ptr)
end
@inline stridedpointer(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T, A <: DenseVector{T}} = StaticStridedPointer{T,Tuple{1,1}}(pointer(parent(B)))
@inline stridedpointer(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T, A <: StridedVector{T}} = adjoint_vector_strided_pointer(stridedpointer(parent(B)))
@inline function adjoint_vector_strided_pointer(ptr::SparseStridedPointer)
    x = ptr.strides[1]
    SparseStridedPointer(pointer(ptr), (x,x))
end
@inline adjoint_vector_strided_pointer(ptr::PackedStridedPointer{T}) where {T} = StaticStridedPointer{T,Tuple{1,1}}(pointer(ptr))
@inline adjoint_vector_strided_pointer(ptr::StaticStridedPointer{T,Tuple{X}}) where {T,X} = StaticStridedPointer{T,Tuple{X,X}}(pointer(ptr))


@inline function stridedpointer(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T,N,A <: AbstractArray{T,N}}
    pB = parent(B)
    RowMajorStridedPointer(pointer(pB), staticmul(T, tailstrides(pB)))
end
@inline function stridedpointer(B::Union{Adjoint{Bool,A},Transpose{Bool,A}}) where {N,A <: BitArray{N}}
    pB = parent(B)
    RowMajorStridedBitPointer(pointer(pB.chunks), tailstrides(pB))
end
@inline function stridedpointer(C::Union{Adjoint{T,A},Transpose{T,A}}) where {T, P, B, A <: SubArray{T,2,P,Tuple{Int,Vararg},B}}
    pC = parent(C)
    SparseStridedPointer(pointer(pC), staticmul(T, reverse(strides(pC))))
end

@inline stridedpointer(x::Number) = x
@inline stridedpointer(x::AbstractRange) = x
# @inline stridedpointer(ptr::Pointer) = PackedStridedPointer(pointer(ptr), tuple())
@inline stridedpointer(ptr::AbstractPointer) = ptr

@generated function noalias!(ptr::Ptr{T}) where {T <: NativeTypes}
    ptyp = JuliaPointerType
    typ = llvmtype(T)
    if Base.libllvm_version < v"10"
        funcname = "noalias" * typ
        decls = "define noalias $typ* @$(funcname)($typ *%a) willreturn noinline { ret $typ* %a }"
        instrs = """
            %ptr = inttoptr $ptyp %0 to $typ*
            %naptr = call $typ* @$(funcname)($typ* %ptr)
            %jptr = ptrtoint $typ* %naptr to $ptyp
            ret $ptyp %jptr
        """
    else
        decls = "declare void @llvm.assume(i1)"
        instrs = """
            %ptr = inttoptr $ptyp %0 to $typ*
            call void @llvm.assume(i1 true) ["noalias"($typ* %ptr)]
            %int = ptrtoint $typ* %ptr to $ptyp
            ret $ptyp %int
        """
    end
    quote
        $(Expr(:meta,:inline))
        llvmcall(
            $((decls, instrs)),
            Ptr{$T}, Tuple{Ptr{$T}}, ptr
        )
    end
end
@inline noalias!(ptr::AbstractPointer) = similar(ptr, noalias!(pointer(ptr)))
@inline noalias!(x::Any) = x
@inline noaliasstridedpointer(x) = stridedpointer(x)
@inline noaliasstridedpointer(A::AbstractRange) = stridedpointer(x)
@inline noaliasstridedpointer(A::AbstractArray) = noalias!(stridedpointer(A))
# @inline StaticStridedStruct{T,X}(s::S) where {T,X,S} = StaticStridedStruct{T,X,S}(s, 0)
# @inline StaticStridedStruct{T,X}(s::S, i::Int) where {T,X,S} = StaticStridedStruct{T,X,S}(s, i)
# @inline offset(ptr::StaticStridedStruct{T,X,S}, i::Integer) where {T,X,S} = StaticStridedStruct{T,X,S}(pointer(ptr), ptr.offset + i)
# # Trying to avoid generated functions
# @inline offset(ptr::StaticStridedStruct{T,X,S}, i::Tuple{<:Integer}) where {T,X,S} = @inbounds StaticStridedStruct{T,X,S}(pointer(ptr), ptr.offset + i[1])
# @inline function offset(ptr::StaticStridedStruct{T,Tuple{A},S}, i::Tuple{<:Integer,<:Integer}) where {T,A,S}
#     @inbounds StaticStridedStruct{T,Tuple{A},S}(pointer(ptr), ptr.offset + i[1] + A * i[2])
# end
# @inline function offset(ptr::StaticStridedStruct{T,Tuple{A,B},S}, i::Tuple{<:Integer,<:Integer,<:Integer}) where {T,A,B,S}
#     @inbounds StaticStridedStruct{T,Tuple{A,B},S}(pointer(ptr), ptr.offset + i[1] + A*i[2] + B*i[3])
# end
# @inline function offset(ptr::StaticStridedStruct{T,Tuple{A,B,C},S}, i::Tuple{<:Integer,<:Integer,<:Integer,<:Integer}) where {T,A,B,C,S}
#     @inbounds StaticStridedStruct{T,Tuple{A,B,C},S}(pointer(ptr), ptr.offset + i[1] + A*i[2] + B*i[3] + C*i[4] )
# end

# @generated tupletype_to_tuple(::Type{T}) where {T<:Tuple} = Expr(:block, Expr(:meta,:inline), Expr(:tuple, T.parameters...))
# @inline function offset(ptr::StaticStridedStruct{T,X,S}, i::NTuple{N}) where {T,X,S,N}
#     strides = tupletype_to_tuple(X)
#     StaticStridedStruct{T,X,S}(pointer(ptr), ptr.offset + first(i) + tdot(strides, Base.tail(i)))
# end


struct RangeWrapper{R <: AbstractRange, I}
    r::R
    i::I
end
@inline gesp(r::AbstractRange, i::I) where {I} = RangeWrapper(r, i)
@inline gesp(r::RangeWrapper, i::I) where {I} = RangeWrapper(r.r, r.i .+ i)
@inline vload(rw::RangeWrapper, i, mask) = vload(rw, i)
@inline vload(rw::RangeWrapper, i::Tuple{I}) where {I}  = vload(rw.r, @inbounds (vadd(rw.i[1], i[1]),))
@inline vload(rw::RangeWrapper, i::Tuple{I1,I2}) where {I1,I2}  = vload(rw.r, @inbounds (vadd(rw.i[1], i[1]), vadd(rw.i[2], i[2])))
@inline vload(rw::RangeWrapper, i::Tuple{I1,I2,I3}) where {I1,I2,I3}  = vload(rw.r, @inbounds (vadd(rw.i[1], i[1]), vadd(rw.i[2], i[2]), vadd(rw.i[3], i[3])))
@inline vload(rw::RangeWrapper, i)  = vload(rw.r, vadd.(rw.i, i))

@inline vload(r::AbstractRange, i::Tuple{I}) where {I} = @inbounds r[vadd(i[1], one(I))]
@inline vload(r::LinearIndices, i::Tuple) = @inbounds r[staticp1(i)...]


@inline subsetview(ptr::PackedStridedPointer, ::Val{1}, i::Integer) = SparseStridedPointer(gep(ptr.ptr, i), ptr.strides)
@generated function subsetview(ptr::PackedStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    I > N + 1 && return :ptr
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != I-1]...)
    offset = Expr(:call, :vmul, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, I - 1)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gepbyte, :p, offset)),
        Expr(:call, :PackedStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::RowMajorStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    if N + 1 == I
        return Expr(
            :block, Expr(:meta, :inline),
            Expr(:(=), :p, Expr(:call, :gep, Expr(:(.), :ptr, QuoteNode(:ptr)), :i)),
            Expr(:call, :SparseStridedPointer, :p, Expr(:call, :reverse, Expr(:(.), :ptr, QuoteNode(:strides))))
        )
    elseif N + 1 < I
        return :ptr
    end
    strideind = N + 1 - I
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != strideind]...)
    offset = Expr(:call, :vmul, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, strideind)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gepbyte, :p, offset)),
        Expr(:call, :RowMajorStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::SparseStridedPointer{T, N}, ::Val{I}, i::Integer) where {I, T, N}
    I > N && return :ptr
    strides = Expr(:tuple, [Expr(:ref, :s, n) for n ∈ 1:N if n != I]...)
    offset = Expr(:call, :vmul, :i, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), Expr(:ref, :s, I)))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :s, Expr(:(.), :ptr, QuoteNode(:strides))),
        Expr(:(=), :strides, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__, Symbol(@__FILE__)), strides)),
        Expr(:(=), :gp, Expr(:call, :gepbyte, :p, offset)),
        Expr(:call, :SparseStridedPointer, :gp, :strides)
    )
end

@generated function subsetview(ptr::StaticStridedPointer{T, X}, ::Val{I}, i::Integer) where {I, T, X}
    I > length(X.parameters) && return :ptr
    Xa = Expr(:curly, :Tuple)
    Xparam = X.parameters
    for n ∈ 1:length(Xparam)
        n == I && continue
        push!(Xa.args, Xparam[n])
    end
    Xᵢ = (Xparam[I])::Int
    offset = Expr(:call, isone(Xᵢ) ? :vmulnp : :vmul, :i, Xᵢ * sizeof(T))
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :gp, Expr(:call, :gepbyte, :p, offset)),
        Expr(:call, Expr(:curly, :StaticStridedPointer, T, Xa), :gp)
    )
end

@generated function subsetview(ptr::StaticStridedStruct{T, X}, ::Val{I}, i::Integer) where {I, T, X}
    I > length(X.parameters) && return :ptr
    Xa = Expr(:curly, :Tuple)
    Xparam = X.parameters
    for n ∈ 1:length(Xparam)
        n == I && continue
        push!(Xa.args, Xparam[n])
    end
    offset = Expr(:call, :vmul, :i, Xparam[I])
    Expr(
        :block,
        Expr(:meta, :inline),
        Expr(:(=), :p, Expr(:(.), :ptr, QuoteNode(:ptr))),
        Expr(:(=), :offset, Expr(:call, :vadd, Expr(:(.), :ptr, QuoteNote(:offset)), offset)),
        Expr(:call, Expr(:curly, :StaticStridedStruct, T, Xa), :p, :offset)
    )
end

@inline stridedpointer(A::AbstractArray, indices::Tuple) = stridedpointer(view(A, indices...))
@inline stridedpointer(A::AbstractArray, ::Type{Transpose}) = stridedpointer(transpose(A))
@inline stridedpointer(A::AbstractArray, ::Type{Adjoint}) = stridedpointer(adjoint(A))
@inline stridedpointer(A::AbstractArray, ::Nothing) = stridedpointer(A)

@inline filter_strides_by_dimequal1(sz::NTuple{N,Int}, st::NTuple{N,Int}) where {N} = @inbounds ntuple(n -> isone(sz[n]) ? 0 : st[n], Val{N}())

@inline stridedpointer_for_broadcast(A::AbstractRange) = A
@inline function stridedpointer_for_broadcast(A::AbstractArray{T,N}) where {T,N}
    PackedStridedPointer(pointer(A), staticmul(T, filter_strides_by_dimequal1(Base.tail(size(A)), tailstrides(A))))
end
@inline stridedpointer_for_broadcast(B::Union{Adjoint{T,A},Transpose{T,A}}) where {T,A <: AbstractVector{T}} = stridedpointer_for_broadcast(parent(B))
@inline stridedpointer_for_broadcast(A::SubArray{T,0,P,S}) where {T,P,S <: Tuple{Int,Vararg}} = pointer(A)
@inline function stridedpointer_for_broadcast(A::SubArray{T,N,P,S}) where {T,N,P,S <: Tuple{Int,Vararg}}
    SparseStridedPointer(pointer(A), staticmul(T, filter_strides_by_dimequal1(size(A), strides(A))))
end
@inline function stridedpointer_for_broadcast(A::SubArray{T,N,P,S}) where {T,N,P,S}
    PackedStridedPointer(pointer(A), staticmul(T, filter_strides_by_dimequal1(Base.tail(size(A)), tailstrides(A))))
end
@inline stridedpointer_for_broadcast(A::BitArray) = stridedpointer(A)

struct MappedStridedPointer{F, T, P <: AbstractPointer{T}}
    f::F
    ptr::P
end
@inline vload(ptr::MappedStridedPointer) = ptr.f(vload(ptr.ptr))

"""
These use 0 based indexing for convenience with respect to the packed and row major strided pointers.
Preserves the first index, dropping the second.
"""
@inline function double_index(ptr::PackedStridedPointer{T}, ::Val{0}, ::Val{M}) where {M,T}
    x = ptr.strides
    SparseStridedPointer(pointer(ptr), @inbounds Base.setindex(x, vadd(x[M], sizeof(T)), M))
end
@generated function double_index(ptr::PackedStridedPointer{T,K}, ::Val{N}, ::Val{M}) where {K,N,M,T}
    tup = Expr(:tuple)
    for k ∈ 1:K
        k == N && continue
        if k == M
            push!(tup.args, Expr(:call, :vadd, Expr(:ref, :x, N), Expr(:ref, :x, M)))
        else
            push!(tup.args, Expr(:ref, :x, k))
        end
    end
    quote
        $(Expr(:meta,:inline))
        x = ptr.strides
        xabridged = @inbounds $tup
        PackedStridedPointer(pointer(ptr), xabridged)
    end
end
@inline function double_index(ptr::RowMajorStridedPointer{T,1}, ::Val{0}, ::Val{1}) where {T}
    x = ptr.strides
    @inbounds SparseStridedPointer(pointer(ptr), (vadd(x[1], sizeof(T)),))
end
@generated function double_index(ptr::SparseStridedPointer{T,K}, ::Val{Nm1}, ::Val{Mm1}) where {K,Nm1,Mm1, T}
    tup = Expr(:tuple)
    N = Nm1 + 1; M = Mm1 + 1
    for k ∈ 1:K
        k == N && continue
        if k == M
            push!(tup.args, Expr(:call, :vadd, Expr(:ref, :x, N), Expr(:ref, :x, M)))
        else
            push!(tup.args, Expr(:ref, :x, k))
        end
    end
    quote
        $(Expr(:meta,:inline))
        x = ptr.strides
        xabridged = @inbounds $tup
        SparseStridedPointer(pointer(ptr), xabridged)
    end
end
@generated function double_index(ptr::StaticStridedPointer{T,X}, ::Val{Nm1}, ::Val{Mm1}) where {X,Nm1,Mm1,T}
    tup = Expr(:curly, :Tuple)
    Xparam = X.parameters
    N = Nm1 + 1; M = Mm1 + 1
    for k ∈ eachindex(Xparam)
        k == N && continue
        if k == M
            push!(tup.args, (Xparam[N])::Int + (Xparam[M])::Int)
        else
            push!(tup.args, (Xparam[k])::Int)
        end
    end
    Expr(:block, Expr(:meta,:inline), :(StaticStridedPointer{$T,$tup}(pointer(ptr))))
end

struct ZeroInitializedStridedPointer{T, P <: AbstractStridedPointer{T}} <: AbstractStridedPointer{T}; ptr::P; end
@inline ZeroInitializedStridedPointer(A) = ZeroInitializedStridedPointer(stridedpointer(A))
@inline vload(::ZeroInitializedStridedPointer{T}) where {T} = zero(T)

@inline vec_length() = Val{1}()
@inline vec_length(::Integer) = Val{1}()
@inline vec_length(::Static) = Val{1}()
@inline vec_length(::_MM{W}) where {W} = Val{W}()
@inline vec_length(::Vec{W}) where {W} = Val{W}()
@inline vec_length(::SVec{W}) where {W} = Val{W}()
@inline promote_val_width(::Val{1}, ::Val{1}) = Val{1}()
@inline promote_val_width(::Val{W}, ::Val{1}) where {W} = Val{W}()
@inline promote_val_width(::Val{1}, ::Val{W}) where {W} = Val{W}()
@inline promote_val_width(::Val{W}, ::Val{W}) where {W} = Val{W}()

@inline vec_length(i::Tuple{}) = Val{1}()
@inline vec_length(i::Tuple{I1}) where {I1} = vec_length(first(i))
@inline vec_length(i::Tuple{I1,I2,Vararg}) where {I1,I2} = promote_val_width(vec_length(first(i)), vec_length(Base.tail(i)))

@inline vload(::ZeroInitializedStridedPointer{T}, ::Val{1}) where {T} = zero(T)
@inline vload(::ZeroInitializedStridedPointer{T}, ::Val{W}) where {T,W} = vzero(SVec{W,T})

@inline vload(::Val{W}, ::ZeroInitializedStridedPointer{T}, ::Any) where {W, T} = vzero(SVec{W,T})
@inline vload(::Val{W}, ::ZeroInitializedStridedPointer{T}, ::Any, ::Mask) where {W, T} = vzero(SVec{W,T})

@inline vload(ptr::ZeroInitializedStridedPointer, i::Tuple) = vload(ptr, vec_length(i))
@inline vload(ptr::ZeroInitializedStridedPointer, i::Tuple, ::Mask) = vload(ptr, vec_length(i))

@inline vstore!(ptr::ZeroInitializedStridedPointer, v::Number, i::Tuple) = vstore!(ptr.ptr, v, i)
@inline vstore!(ptr::ZeroInitializedStridedPointer, v::Number, i::Tuple, m::Mask) = vstore!(ptr.ptr, v, i, m)
@inline vstore!(ptr::ZeroInitializedStridedPointer, v::AbstractStructVec, i::Tuple) = vstore!(ptr.ptr, v, i)
@inline vstore!(ptr::ZeroInitializedStridedPointer, v::AbstractStructVec, i::Tuple, m::Mask) = vstore!(ptr.ptr, v, i, m)

@inline vnoaliasstore!(ptr::ZeroInitializedStridedPointer, v::Number, i::Tuple) = vnoaliasstore!(ptr.ptr, v, i)
@inline vnoaliasstore!(ptr::ZeroInitializedStridedPointer, v::Number, i::Tuple, m::Mask) = vnoaliasstore!(ptr.ptr, v, i, m)
@inline vnoaliasstore!(ptr::ZeroInitializedStridedPointer, v::AbstractStructVec, i::Tuple) = vnoaliasstore!(ptr.ptr, v, i)
@inline vnoaliasstore!(ptr::ZeroInitializedStridedPointer, v::AbstractStructVec, i::Tuple, m::Mask) = vnoaliasstore!(ptr.ptr, v, i, m)

@inline stride1offset(ptr::ZeroInitializedStridedPointer, i) = stride1offset(ptr.ptr, i)
@inline stridedoffset(ptr::ZeroInitializedStridedPointer, i) = stridedoffset(ptr.ptr, i)
@inline gep(ptr::ZeroInitializedStridedPointer, i::Tuple) = gep(ptr.ptr, i)
@inline gesp(ptr::ZeroInitializedStridedPointer, i::Tuple) = ZeroInitializedStridedPointer(gesp(ptr.ptr, i))

@inline Base.pointer(ptr::ZeroInitializedStridedPointer) = pointer(ptr.ptr)

