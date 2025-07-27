function conflictquote(W::Int = 16, bits::Int = 32)
  @assert bits == 32 || bits == 64
  s = bits == 32 ? 'd' : 'q'
  typ = "i$(bits)"
  vtyp = "<$W x $(typ)>"
  op = "@llvm.x86.avx512.conflict.$s.$(bits*W)"
  decl = "declare <$W x $(typ)> $op(<$W x $(typ)>)"
  instrs = "%res = call <$W x $(typ)> $op(<$W x $(typ)> %0)\n ret <$W x $(typ)> %res"
  T = Symbol(:UInt, bits)
  llvmcall_expr(
    decl,
    instrs,
    :(_Vec{$W,$T}),
    :(Tuple{_Vec{$W,$T}}),
    vtyp,
    [vtyp],
    [:(data(v))]
  )
end

@generated vpconflict(v::Vec{W,T}) where {W,T} = conflictquote(W, 8sizeof(T))
