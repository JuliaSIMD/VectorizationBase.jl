
@eval @inline function assume(b::Bool)
    $(llvmcall_expr("declare void @llvm.assume(i1)", "%b = trunc i8 %0 to i1\ncall void @llvm.assume(i1 %b)\nret void", :Cvoid, :(Tuple{Bool}), "void", ["i8"], [:b]))
end

@eval @inline function expect(b::Bool)
    $(llvmcall_expr("declare i1 @llvm.expect.i1(i1, i1)", """
    %b = trunc i8 %0 to i1
    %actual = call i1 @llvm.expect.i1(i1 %b, i1 true)
    %byte = zext i1 %actual to i8
    ret i8 %byte""", :Bool, :(Tuple{Bool}), "i8", ["i8"], [:b]))
end
@generated function expect(i::I, ::Val{N}) where {I <: Integer, N}
    ityp = 'i' * string(8sizeof(I))
    llvmcall_expr("declare i1 @llvm.expect.$ityp($ityp, i1)", """
    %actual = call $ityp @llvm.expect.$ityp($ityp %0, $ityp $N)
    ret $ityp %actual""", I, :(Tuple{$I}), ityp, [ityp], [:i])
end







abstract type AbstractStridedPointer{T,C,B,R,X,N,P} end

struct StridedPointer{T,C,B,R,X,N,P} <: AbstractStridedPointer{T,C,B,R,X,N,P}
    p::Ptr{T}
    st::SDTuple{X,N,P}
end

@inline function stridedpointer(ptr::Ptr{T}, ::Contiguous{C}, ::ContiguousBatch{B}, ::StrideRank{R}, st::SDTuple{X,N,P}) where {T,C,B,R,X,N,P}
    StridedPointer{T,C,B,R,X,N,P}(ptr, st)
end
@inline function stridedpointer(::CPUPointer, A::AbstractArray)
    stridedpointer(pointer(A), contiguous_axis(A), contiguous_batch_size(A), striderank(A), sdstrides(A))
end

@inline stridedpointer(A::AbstractArray) = stridedpointer(device(A), A)



function stridedpointer(A::Array{T,N}) where {T,N}
    StridedPointer{T,1,0,ntuple(identity,Val{N}()),ntuple(n -> isone(n) ? 1 : -1, Val{N}()), N, N-1}(pointer(A), Base.tail(strides(A)))
end

