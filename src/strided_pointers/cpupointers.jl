# i8* Ptr to default address space
const CPUPtr{T} = Core.LLVMPtr{T,0}
const AbstractPtr{T} = Union{Ptr{T},CPUPtr{T}}

@inline cpupointer(A::CPUPtr) = A
@inline cpupointer(A::Ptr{T}) where {T} = reinterpret(CPUPtr{T}, A)
@inline cpupointer(A::AbstractArray{T}) where {T} = reinterpret(CPUPtr{T}, pointer(A))
@inline cpupointer(A) = cpupointer(pointer(A))

