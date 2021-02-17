
@generated function vscalef(v1::Vec{W,T}, v2::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
    bits = 8W*sizeof(T)
    bits ∈ (128,256,512) || throw(ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."))
    ltyp = LLVM_TYPES[T]
    vtyp = "<$W x $ltyp>"
    dors = T === Float64 ? 'd' : 's'
    instr = "$vtyp @llvm.x86.avx512.mask.scalef.p$(dors).$bits"
    mtyp = W == 16 ? "i16" : "i8"
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp undef, $mtyp -1, i32 11)\nret $vtyp %res"
    arg_syms = [:(data(v1)), :(data(v2))]
    llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T},_Vec{$W,$T}}), vtyp, fill(vtyp, 2), arg_syms)
end
@generated function vscalef(m::Mask{W}, v1::Vec{W,T}, v2::Vec{W,T}, v3::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
    bits = 8W*sizeof(T)
    bits ∈ (128,256,512) || throw(ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."))
    ltyp = LLVM_TYPES[T]
    vtyp = "<$W x $ltyp>"
    dors = T === Float64 ? 'd' : 's'
    mtyp = W == 16 ? "i16" : "i8"
    mtypj = W == 16 ? :UInt16 : :UInt8
    instr = "$vtyp @llvm.x86.avx512.mask.scalef.p$(dors).$bits"
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp %2, $mtyp %3, i32 11)\nret $vtyp %res"
    arg_syms = [:(data(v1)), :(data(v2)), :(data(v3)), :(data(m))]
    llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$mtypj}), vtyp, [vtyp,vtyp,vtyp,mtyp], arg_syms)
end
@generated function vsreduce(v::Vec{W,T}, ::Val{M}) where {W,T<:Union{Float32,Float64}, M}
    bits = 8W*sizeof(T)
    bits ∈ (128,256,512) || throw(ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."))
    M isa Integer || throw(ArgumentError("M must be an integer, but received $M of type $(typeof(M))."))
    ltyp = LLVM_TYPES[T]
    vtyp = "<$W x $ltyp>"
    dors = T === Float64 ? 'd' : 's'
    instr = "$vtyp @llvm.x86.avx512.mask.reduce.p$(dors).$bits"
    mtyp = W == 16 ? "i16" : "i8"
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 $M, $vtyp undef, $mtyp -1, i32 8)\nret $vtyp %res"
    arg_syms = [:(data(v))]
    llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end

@generated function vpermi2pd(c::Vec{8,UInt64}, v1::Vec{8,Float64}, v2::Vec{8,Float64}) #where {W,T<:Union{Float32,Float64}, M}
    W = 8; T = Float64
    bits = 8W*sizeof(T)
    # bits ∈ (128,256,512) || throw(ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."))
    ityp = "i$(8sizeof(T))"
    vityp = "<$W x $ityp>"
    ltyp = LLVM_TYPES[T]
    vtyp = "<$W x $ltyp>"
    dors = T === Float64 ? 'd' : 's'
    instr = "$vtyp @llvm.x86.avx512.vpermi2var.p$(dors).$bits"
    decl = "declare $instr($vtyp, $vityp, $vtyp)"
    instrs = "%res = call $instr($vtyp %0, $vityp %1, $vtyp %2)\nret $vtyp %res"
    arg_syms = [:(data(v1)), :(data(c)), :(data(v2))]
    jityp = T === Float64 ? :UInt64 : :UInt32
    llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T},_Vec{$W,$jityp},_Vec{$W,$T}}), vtyp, [vtyp,vityp,vtyp], arg_syms)
end

@inline vscalef(v1::VecUnroll, v2::VecUnroll) = VecUnroll(fmap(vscalef, getfield(v1, :data), getfield(v2, :data)))
@inline vsreduce(v::VecUnroll, ::Val{M}) where {M} = VecUnroll(fmap(vsreduce, getfield(v, :data), Val{M}()))
@inline vpermi2pd(v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) = VecUnroll(fmap(vpermi2pd, getfield(v1, :data), getfield(v2, :data), getfield(v3, :data)))
@inline vpermi2pd(v1::VecUnroll, v2::Vec, v3::Vec) = VecUnroll(fmap(vpermi2pd, getfield(v1, :data), v2, v3))


# magic rounding constant: 1.5*2^52 Adding, then subtracting it from a float rounds it to an Int.
MAGIC_ROUND_CONST(::Type{Float64}) = 6.755399441055744e15
MAGIC_ROUND_CONST(::Type{Float32}) = 1.2582912f7

# min and max arguments by base and type
MAX_EXP(::Val{2},  ::Type{Float64}) =  1024.0                   # log2 2^1023*(2-2^-52)
MIN_EXP(::Val{2},  ::Type{Float64}) = -1022.0                   # log2(big(2)^-1023*(2-2^-52))
MAX_EXP(::Val{2},  ::Type{Float32}) =  128f0                    # log2 2^127*(2-2^-52)
MIN_EXP(::Val{2},  ::Type{Float32}) = -126f0                    # log2 2^-1075
MAX_EXP(::Val{ℯ},  ::Type{Float64}) =  709.782712893383996732   # log 2^1023*(2-2^-52)
MIN_EXP(::Val{ℯ},  ::Type{Float64}) = -708.396418532264106335  # log 2^-1075
MAX_EXP(::Val{ℯ},  ::Type{Float32}) =  88.72283905206835f0      # log 2^127 *(2-2^-23)
MIN_EXP(::Val{ℯ},  ::Type{Float32}) = -87.3365448101577555f0#-103.97207708f0           # log 2^-150
MAX_EXP(::Val{10}, ::Type{Float64}) =  308.25471555991675       # log10 2^1023*(2-2^-52)
MIN_EXP(::Val{10}, ::Type{Float64}) = -307.65260000       # log10 2^-1075
MAX_EXP(::Val{10}, ::Type{Float32}) =  38.531839419103626f0     # log10 2^127 *(2-2^-23)
MIN_EXP(::Val{10}, ::Type{Float32}) = -37.9297794795476f0      # log10 2^-127 *(2-2^-23)

# 256/log(base, 2) (For Float64 reductions)
LogBo256INV(::Val{2}, ::Type{Float64})    = 256.
LogBo256INV(::Val{ℯ}, ::Type{Float64})    = 369.3299304675746
LogBo256INV(::Val{10}, ::Type{Float64})   = 850.4135922911647
# -log(base, 2)/256 in upper and lower bits
LogBo256U(::Val{2}, ::Type{Float64})      = -0.00390625
LogBo256U(::Val{ℯ}, ::Type{Float64})      = -0.0027076061740622863
LogBo256U(::Val{10}, ::Type{Float64})     = -0.0011758984205624266
LogBo256L(base::Val{2}, ::Type{Float64})  =  0.
LogBo256L(base::Val{ℯ}, ::Type{Float64})  = -9.058776616587108e-20
LogBo256L(base::Val{10}, ::Type{Float64}) = 1.0952062999160822e-20

# 1/log(base, 2) (For Float32 reductions)
LogBINV(::Val{2}, ::Type{Float32})    =  1f0
LogBINV(::Val{ℯ}, ::Type{Float32})    =  1.442695f0
LogBINV(::Val{10}, ::Type{Float32})   =  3.321928f0
# -log(base, 2) in upper and lower bits
LogBU(::Val{2}, ::Type{Float32})      = -1f0
LogBU(::Val{ℯ}, ::Type{Float32})      = -0.6931472f0
LogBU(::Val{10}, ::Type{Float32})     = -0.30103f0
LogBL(base::Val{2}, ::Type{Float32})  =  0f0
LogBL(base::Val{ℯ}, ::Type{Float32})  =  1.9046542f-9
LogBL(base::Val{10}, ::Type{Float32}) =  1.4320989f-8

const FloatType64 = Union{Float64,AbstractSIMD{<:Any,Float64}}
const FloatType32 = Union{Float32,AbstractSIMD{<:Any,Float32}}
# Range reduced kernels          
@inline function expm1b_kernel(::Val{2}, x::FloatType64)
    # c6 = 0.6931471807284470571335252997834339128744539291358546258980326560263434831636494
    # c5 = 0.2402265119815758621794630410361025063296309075509484445159065872903725193960909
    # c4 = 0.05550410353447979823044149277158612685395896745775325243210607075766620053156177
    # c3 = 0.009618027253668450057706478612143223628979891379942570690446533010539871321541621
    # c2 = 0.001333392256353875413926876917141786686018234585146983223440727245459444740967253
    # c1 = 0.0001546929114168849728971327603158937595919966441732209337930866845915899223829891
    # c0 = 1.520192159457321441849564286267892034534060236471603225598783028117591315796835e-05
    # c5 = 0.6931472067096466099497350107329038640311532915014328403152023514862215769240471
    # c4 = 0.2402265150505520926831534797602254284855354178135282005410994764797061783074115
    # c3 = 0.05550327215766594452554739168596479012109775178907146059734348050730310383358696
    # c2 = 0.00961799451416147891836707565892019722415069604010430702590438969636015825337285
    # c1 = 0.001340043166700788064581996335332499076913713747844545061461125917537819216640726
    # c0 = 0.0001547802227945780278842074081393459334029013917349238368250485753166922523500416
    # x * muladd(muladd(muladd(muladd(muladd(muladd(muladd(muladd(c0,x,c1),x,c2),x,c3),x,c4),x,c5),x,c6),x,c7),x,c8)
    # x * muladd(muladd(muladd(muladd(muladd(muladd(c0,x,c1),x,c2),x,c3),x,c4),x,c5),x,c6)
    # x * muladd(muladd(muladd(muladd(muladd(c0,x,c1),x,c2),x,c3),x,c4),x,c5)
    x * muladd(muladd(muladd(0.009618130135925114, x, 0.055504115022757844), x, 0.2402265069590989), x, 0.6931471805599393)
end
@inline function expm1b_kernel(::Val{ℯ}, x::FloatType64)
    x * muladd(muladd(muladd(0.04166666762124105, x, 0.1666666704849642), x, 0.49999999999999983), x, 0.9999999999999998)
end
@inline function expm1b_kernel(::Val{10}, x::FloatType64)
    x * muladd(muladd(muladd(muladd(0.5393833837413015, x, 1.1712561359457612), x, 2.0346785922926713), x, 2.6509490552382577), x, 2.302585092994046)
end
@inline function expb_kernel(::Val{2}, x::FloatType32)
    muladd(muladd(muladd(muladd(muladd(muladd(0.00015478022f0, x, 0.0013400431f0), x, 0.009617995f0), x, 0.05550327f0), x, 0.24022652f0), x, 0.6931472f0), x, 1.0f0)
end
@inline function expb_kernel(::Val{ℯ}, x::FloatType32)
    muladd(muladd(muladd(muladd(muladd(muladd(0.0013956056f0, x, 0.008375129f0), x, 0.041666083f0), x, 0.16666415f0), x, 0.5f0), x, 1.0f0), x, 1.0f0)
end
@inline function expb_kernel(::Val{10}, x::FloatType32)
    muladd(muladd(muladd(muladd(muladd(muladd(0.20799689f0, x, 0.54208815f0), x, 1.1712388f0), x, 2.034648f0), x, 2.6509492f0), x, 2.3025851f0), x, 1.0f0)
end

const J_TABLE= Float64[2.0^(big(j-1)/256) for j in 1:256];
const TABLE_EXP_64_0= Vec(ntuple(j -> Core.VecElement(Float64(2.0^(big(j-1)/16))), Val(8)))
const TABLE_EXP_64_1= Vec(ntuple(j -> Core.VecElement(Float64(2.0^(big(j+7)/16))), Val(8)))

@inline target_trunc(v, ::VectorizationBase.True) = v
@inline target_trunc(v, ::VectorizationBase.False) = v % UInt32
@inline target_trunc(v) = target_trunc(v, VectorizationBase.has_feature(Val(:x86_64_avx512dq)))

@inline fast_fma(a, b, c, ::True) = fma(a, b, c)
@inline function fast_fma(a, b, c, ::False)
    d = dadd(dmul(Double(a),Double(b)),Double(c))
    add_ieee(d.hi, d.lo)
end

@inline function vexp2_v3(x::AbstractSIMD{W,Float64}) where {W}
    x256 = 256.0*x
    r = vsreduce(x256, Val(0)) * 0.00390625
    N_float = x - r
    inds = convert(UInt, vsreduce(N_float, Val(1))*256.0)
    js = vload(VectorizationBase.zero_offsets(stridedpointer(J_TABLE)), (inds,))
    small_part = vfmadd(js, expm1b_kernel(Val(2), r), js)
    res = vscalef(small_part, N_float)
    return res
end

@inline function vexp2_v1(x::AbstractSIMD{8,Float64})
    x16 = x
    # x16 = 16x
    r =  vsreduce(x16, Val(4))
    m = x16 - r
    mfrac = m
    inds = (reinterpret(UInt64, mfrac) >> 0x000000000000002d) & 0x000000000000000f
    # @show r m mfrac reinterpret(UInt64, m) reinterpret(UInt64, mfrac)
    # js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
    # @show m mfrac r
    small_part = expm1b_kernel(Val(2), r) + 1.0
    # js = 1.0
    # small_part = vfmadd(js, expm1b_kernel(Val(2), r), js)
    vscalef(small_part, mfrac)
end

@inline function expm1b_kernel_5(::Val{2}, x)
    c5 = 0.6931471805599457533351827593325319924753473772859614915719486459595837313933663
    c4 = 0.2402265069591009431089489060897837825648676480621950809556237945205562267112511
    c3 = 0.05550410865663929372911461843767669974316894963735870580154522796380984673567634
    c2 = 0.00961812910613182376367867689426991348318504321185094738294343470767871628697879
    c1 = 0.001333378157683735211078403326604752238340853209789619792858391909299167771871147
    c0 = 0.0001540378851029625623114826398060979330719673345637296642237670082377277446583639

    x * vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(c0,x,c1),x,c2),x,c3),x,c4),x,c5)
end
@inline function expm1b_kernel_4(::Val{2}, x)
    c4 = 0.6931471805599461972549081995383434692316977327912755704234013405443109498729026
    c3 = 0.2402265069131940842333497738928958607054740795078596615709864445611497846077303
    c2 = 0.05550410865300270379171299778517151376504051870524903806295523325435530249981495
    c1 = 0.009618317140648284298097106744730186251913149278152053357630395863210686828434175
    c0 = 0.001333381881551676348461495248002642715422207072457864472267417920610122672570108
    x * vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(c0,x,c1),x,c2),x,c3),x,c4)
end

@inline function vexp2_v2(x::AbstractSIMD{8,Float64})
    x16 = 16.0*x
    # x8 = 8x
    r =  vsreduce(x16, Val(0))
    @fastmath  begin
    m = x16 - r
    # m + r = x16, r ∈ (-0.5,0.5]
    # m/16 + r/16 = x, r ∈ (-1/32, 1/32]
    # we must now vreduce `mfrac`
    # return m
    r *= 0.0625
    m *= 0.0625
    end
    # expr = expm1b_kernel_4(Val(2), r)
    expr = expm1b_kernel_5(Val(2), r)
    inds = convert(UInt64, vsreduce(m, Val(1)) * 16.0)
    # inds = (reinterpret(UInt64, mfrac) >> 0x000000000000002d) & 0x000000000000000f
    # inds = (reinterpret(UInt64, mfrac) >> 0x0000000000000035) & 0x000000000000000f
    # @show r m mfrac reinterpret(UInt64, m) reinterpret(UInt64, mfrac)
    js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
    # return r, mfrac, js
    # @show js m 16mfrac r mfrac inds%Int ((inds>>1)%Int)
    # js = 1.0
    small_part = vfmadd(js, expr, js)
    vscalef(small_part, m)#, r, mfrac, js, inds
end

# for (func, base) in (:vexp2=>Val(2), :vexp=>Val(ℯ), :vexp10=>Val(10))
#     @eval begin
#         @inline function $func(x::AbstractSIMD{W,Float64}) where {W}
#             N_float = muladd(x, LogBo256INV($base, Float64), MAGIC_ROUND_CONST(Float64))
#             N = target_trunc(reinterpret(UInt64, N_float))
#             N_float = N_float - MAGIC_ROUND_CONST(Float64)
#             r = fast_fma(N_float, LogBo256U($base, Float64), x, fma_fast())
#             r = fast_fma(N_float, LogBo256L($base, Float64), r, fma_fast())
#             js = vload(VectorizationBase.zero_offsets(stridedpointer(J_TABLE)), (N & 0x000000ff,))
#             k = N >>> 0x00000008
#             small_part = vfmadd(js, expm1b_kernel($base, r), js)
#             twopk = (k % UInt64) << 0x0000000000000034
#             @show N_float k twopk small_part
#             return small_part
#             # res = reinterpret(Float64, twopk + small_part)
#             # return res
#             vscalef(small_part, N_float)
#         end
#     end
# end
