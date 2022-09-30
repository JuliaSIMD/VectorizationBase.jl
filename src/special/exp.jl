
# `_vscalef` for architectures without `vscalef`.
# magic rounding constant: 1.5*2^52 Adding, then subtracting it from a float rounds it to an Int.
MAGIC_ROUND_CONST(::Type{Float64}) = 6.755399441055744e15
MAGIC_ROUND_CONST(::Type{Float32}) = 1.2582912f7
@inline function vscalef(
  x::Union{T,AbstractSIMD{<:Any,T}},
  y::Union{T,AbstractSIMD{<:Any,T}},
  ::False,
) where {T<:Union{Float32,Float64}}
  _vscalef(x, floor(y))
end
@inline signif_bits(::Type{Float32}) = 0x00000017 # 23
@inline signif_bits(::Type{Float64}) = 0x0000000000000034 # 52
@inline function _vscalef(
  x::Union{T,AbstractSIMD{<:Any,T}},
  y::Union{T,AbstractSIMD{<:Any,T}},
) where {T<:Union{Float32,Float64}}
  N = reinterpret(Base.uinttype(T), y + MAGIC_ROUND_CONST(T))
  k = N# >>> 0x00000008

  small_part = reinterpret(Base.uinttype(T), x)
  twopk = (k % Base.uinttype(T)) << signif_bits(T)
  reinterpret(T, twopk + small_part)
end
@inline vscalef(
  m::AbstractMask,
  v1::AbstractSIMD,
  v2::AbstractSIMD,
  v3::AbstractSIMD,
  ::False,
) = vifelse(m, vscalef(v1, v2, False()), v3)
@inline vscalef(v1::T, v2::T) where {T<:AbstractSIMD} =
  vscalef(v1, v2, has_feature(Val(:x86_64_avx512f)))
@inline vscalef(m::AbstractMask, v1::T, v2::T, v3::T) where {T<:AbstractSIMD} =
  vscalef(m, v1, v2, v3, has_feature(Val(:x86_64_avx512f)))
@inline vscalef(v1::T, v2::T) where {T<:Union{Float32,Float64}} = vscalef(v1, v2, False())
@inline vscalef(b::Bool, v1::T, v2::T, v3::T) where {T<:Union{Float32,Float64}} =
  b ? vscalef(v1, v2, False()) : zero(T)
@inline vscalef(v1, v2) = ((v3, v4) = promote(v1, v2); vscalef(v3, v4))
@generated function vscalef(
  v1::Vec{W,T},
  v2::Vec{W,T},
  ::True,
) where {W,T<:Union{Float32,Float64}}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.scalef.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp, i32)"
    # instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp undef, $mtyp -1, i32 11)\nret $vtyp %res"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp undef, $mtyp -1, i32 8)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v1)), :(data(v2))]
  llvmcall_expr(
    decl,
    instrs,
    :(_Vec{$W,$T}),
    :(Tuple{_Vec{$W,$T},_Vec{$W,$T}}),
    vtyp,
    fill(vtyp, 2),
    arg_syms,
  )
end
@generated function vscalef(
  m::AbstractMask{W},
  v1::Vec{W,T},
  v2::Vec{W,T},
  v3::Vec{W,T},
  ::True,
) where {W,T<:Union{Float32,Float64}}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  mtyp = W == 16 ? "i16" : "i8"
  mtypj = W == 16 ? :UInt16 : :UInt8
  instr = "$vtyp @llvm.x86.avx512.mask.scalef.p$(dors).$bits"
  if bits == 512
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp %2, $mtyp %3, i32 11)\nret $vtyp %res"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp %2, $mtyp %3, i32 8)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, $vtyp, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, $vtyp %1, $vtyp %2, $mtyp %3)\nret $vtyp %res"
  end
  arg_syms = [:(data(v1)), :(data(v2)), :(data(v3)), :(data(m))]
  llvmcall_expr(
    decl,
    instrs,
    :(_Vec{$W,$T}),
    :(Tuple{_Vec{$W,$T},_Vec{$W,$T},_Vec{$W,$T},$mtypj}),
    vtyp,
    [vtyp, vtyp, vtyp, mtyp],
    arg_syms,
  )
end
@generated function vsreduce(v::Vec{W,T}, ::Val{M}) where {W,T<:Union{Float32,Float64},M}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  M isa Integer ||
    throw(ArgumentError("M must be an integer, but received $M of type $(typeof(M))."))
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.reduce.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 $M, $vtyp undef, $mtyp -1, i32 8)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, i32 $M, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end

@generated function vpermi2pd(c::Vec{8,UInt64}, v1::Vec{8,Float64}, v2::Vec{8,Float64}) #where {W,T<:Union{Float32,Float64}, M}
  W = 8
  T = Float64
  bits = (8W * sizeof(T))::Int
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
  llvmcall_expr(
    decl,
    instrs,
    :(_Vec{$W,$T}),
    :(Tuple{_Vec{$W,$T},_Vec{$W,$jityp},_Vec{$W,$T}}),
    vtyp,
    [vtyp, vityp, vtyp],
    arg_syms,
  )
end

@inline vscalef(v1::VecUnroll, v2::VecUnroll) =
  VecUnroll(fmap(vscalef, getfield(v1, :data), getfield(v2, :data)))
@inline vscalef(m::VecUnroll, v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) = VecUnroll(
  fmap(
    vscalef,
    getfield(m, :data),
    getfield(v1, :data),
    getfield(v2, :data),
    getfield(v3, :data),
  ),
)
@inline vsreduce(v::VecUnroll, ::Val{M}) where {M} =
  VecUnroll(fmap(vsreduce, getfield(v, :data), Val{M}()))
@inline vpermi2pd(v1::VecUnroll, v2::VecUnroll, v3::VecUnroll) =
  VecUnroll(fmap(vpermi2pd, getfield(v1, :data), getfield(v2, :data), getfield(v3, :data)))
@inline vpermi2pd(v1::VecUnroll, v2::Vec, v3::Vec) =
  VecUnroll(fmap(vpermi2pd, getfield(v1, :data), v2, v3))


# magic rounding constant: 1.5*2^52 Adding, then subtracting it from a float rounds it to an Int.

# min and max arguments by base and type
MAX_EXP(::Val{2}, ::Type{Float64}) = 1024.0                   # log2 2^1023*(2-2^-52)
MIN_EXP(::Val{2}, ::Type{Float64}) = -1022.0                   # log2(big(2)^-1023*(2-2^-52))
MAX_EXP(::Val{2}, ::Type{Float32}) = 128.0f0                    # log2 2^127*(2-2^-52)
MIN_EXP(::Val{2}, ::Type{Float32}) = -126.0f0                    # log2 2^-1075
MAX_EXP(::Val{ℯ}, ::Type{Float64}) = 709.782712893383996732   # log 2^1023*(2-2^-52)
MIN_EXP(::Val{ℯ}, ::Type{Float64}) = -708.396418532264106335  # log 2^-1075
MAX_EXP(::Val{ℯ}, ::Type{Float32}) = 88.72283905206835f0      # log 2^127 *(2-2^-23)
MIN_EXP(::Val{ℯ}, ::Type{Float32}) = -87.3365448101577555f0#-103.97207708f0           # log 2^-150
MAX_EXP(::Val{10}, ::Type{Float64}) = 308.25471555991675       # log10 2^1023*(2-2^-52)
MIN_EXP(::Val{10}, ::Type{Float64}) = -307.65260000       # log10 2^-1075
MAX_EXP(::Val{10}, ::Type{Float32}) = 38.531839419103626f0     # log10 2^127 *(2-2^-23)
MIN_EXP(::Val{10}, ::Type{Float32}) = -37.9297794795476f0      # log10 2^-127 *(2-2^-23)

# 256/log(base, 2) (For Float64 reductions)
LogBo256INV(::Val{2}, ::Type{Float64}) = 256.0
LogBo256INV(::Val{ℯ}, ::Type{Float64}) = 369.3299304675746
LogBo256INV(::Val{10}, ::Type{Float64}) = 850.4135922911647
LogBo16INV(::Val{2}, ::Type{Float64}) = 16.0
LogBo16INV(::Val{ℯ}, ::Type{Float64}) = 23.083120654223414
LogBo16INV(::Val{10}, ::Type{Float64}) = 53.150849518197795
# -log(base, 2)/256 in upper and lower bits
LogBo256U(::Val{2}, ::Type{Float64}) = -0.00390625
LogBo256U(::Val{ℯ}, ::Type{Float64}) = -0.0027076061740622863
LogBo256U(::Val{10}, ::Type{Float64}) = -0.0011758984205624266
LogBo256L(base::Val{2}, ::Type{Float64}) = 0.0
LogBo256L(base::Val{ℯ}, ::Type{Float64}) = -9.058776616587108e-20
LogBo256L(base::Val{10}, ::Type{Float64}) = 1.0952062999160822e-20

LogBo16U(::Val{2}, ::Type{Float64}) = -0.0625
LogBo16U(::Val{ℯ}, ::Type{Float64}) =
  -0.04332169878499658183857700759113603550471875839751595338254250059333710137310597
LogBo16U(::Val{10}, ::Type{Float64}) =
  -0.01881437472899882470085868092028081417301186759138178383190171632044426182965149
LogBo16L(base::Val{2}, ::Type{Float64}) = Zero()
LogBo16L(base::Val{ℯ}, ::Type{Float64}) = -1.4494042586539372e-18
LogBo16L(base::Val{10}, ::Type{Float64}) = 1.7523300798657315e-19




# 1/log(base, 2) (For Float32 reductions)
LogBINV(::Val{2}, ::Type{Float32}) = 1.0f0
LogBINV(::Val{ℯ}, ::Type{Float32}) = 1.442695f0
LogBINV(::Val{10}, ::Type{Float32}) = 3.321928f0
# -log(base, 2) in upper and lower bits
LogBU(::Val{2}, ::Type{Float32}) = -1.0f0
LogBU(::Val{ℯ}, ::Type{Float32}) = -0.6931472f0
LogBU(::Val{10}, ::Type{Float32}) = -0.30103f0
LogBL(base::Val{2}, ::Type{Float32}) = 0.0f0
LogBL(base::Val{ℯ}, ::Type{Float32}) = 1.9046542f-9
LogBL(base::Val{10}, ::Type{Float32}) = 1.4320989f-8

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
  x * muladd(
    muladd(muladd(0.009618130135925114, x, 0.055504115022757844), x, 0.2402265069590989),
    x,
    0.6931471805599393,
  )
end
@inline function expm1b_kernel(::Val{ℯ}, x::FloatType64)
  x * muladd(
    muladd(muladd(0.04166666762124105, x, 0.1666666704849642), x, 0.49999999999999983),
    x,
    0.9999999999999998,
  )
end
@inline function expm1b_kernel(::Val{10}, x::FloatType64)
  x * muladd(
    muladd(
      muladd(muladd(0.5393833837413015, x, 1.1712561359457612), x, 2.0346785922926713),
      x,
      2.6509490552382577,
    ),
    x,
    2.302585092994046,
  )
end
@inline function expb_kernel(::Val{2}, x::FloatType32)
  muladd(
    muladd(
      muladd(
        muladd(
          muladd(
            muladd(muladd(1.5316464f-5, x, 0.00015478022f0), x, 0.0013400431f0),
            x,
            0.009617995f0,
          ),
          x,
          0.05550327f0,
        ),
        x,
        0.24022652f0,
      ),
      x,
      0.6931472f0,
    ),
    x,
    1.0f0,
  )
end
@inline function expb_kernel(::Val{ℯ}, x::FloatType32)
  muladd(
    muladd(
      muladd(
        muladd(
          muladd(
            muladd(muladd(0.00019924171f0, x, 0.0013956056f0), x, 0.008375129f0),
            x,
            0.041666083f0,
          ),
          x,
          0.16666415f0,
        ),
        x,
        0.5f0,
      ),
      x,
      1.0f0,
    ),
    x,
    1.0f0,
  )
end
@inline function expb_kernel(::Val{10}, x::FloatType32)
  muladd(
    muladd(
      muladd(
        muladd(
          muladd(
            muladd(muladd(0.06837386f0, x, 0.20799689f0), x, 0.54208815f0),
            x,
            1.1712388f0,
          ),
          x,
          2.034648f0,
        ),
        x,
        2.6509492f0,
      ),
      x,
      2.3025851f0,
    ),
    x,
    1.0f0,
  )
end

const J_TABLE = Float64[2.0^(big(j - 1) / 256) for j = 1:256];

@inline fast_fma(a, b, c, ::True) = fma(a, b, c)
@inline function fast_fma(a, b, c, ::False)
  d = dadd(dmul(Double(a), Double(b), False()), Double(c))
  add_ieee(d.hi, d.lo)
end

@static if (Sys.ARCH === :x86_64) | (Sys.ARCH === :i686)
  const TABLE_EXP_64_0 =
    Vec(ntuple(j -> Core.VecElement(Float64(2.0^(big(j - 1) / 16))), Val(8)))
  const TABLE_EXP_64_1 =
    Vec(ntuple(j -> Core.VecElement(Float64(2.0^(big(j + 7) / 16))), Val(8)))

  @inline target_trunc(v, ::VectorizationBase.True) = v
  @inline target_trunc(v, ::VectorizationBase.False) = v % UInt32
  @inline target_trunc(v) =
    target_trunc(v, VectorizationBase.has_feature(Val(:x86_64_avx512dq)))



  # @inline function vexp2_v1(x::AbstractSIMD{8,Float64})
  #     x16 = x
  #     # x16 = 16x
  #     r =  vsreduce(x16, Val(4))
  #     m = x16 - r
  #     mfrac = m
  #     inds = (reinterpret(UInt64, mfrac) >> 0x000000000000002d) & 0x000000000000000f
  #     # @show r m mfrac reinterpret(UInt64, m) reinterpret(UInt64, mfrac)
  #     # js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
  #     # @show m mfrac r
  #     small_part = expm1b_kernel(Val(2), r) + 1.0
  #     # js = 1.0
  #     # small_part = vfmadd(js, expm1b_kernel(Val(2), r), js)
  #     vscalef(small_part, mfrac)
  # end

  @inline function expm1b_kernel_16(::Val{2}, x)
    c5 = 0.6931471805599457533351827593325319924753473772859614915719486459595837313933663
    c4 = 0.2402265069591009431089489060897837825648676480621950809556237945205562267112511
    c3 = 0.05550410865663929372911461843767669974316894963735870580154522796380984673567634
    c2 = 0.00961812910613182376367867689426991348318504321185094738294343470767871628697879
    c1 = 0.001333378157683735211078403326604752238340853209789619792858391909299167771871147
    c0 =
      0.0001540378851029625623114826398060979330719673345637296642237670082377277446583639

    x * vmuladd_fast(
      vmuladd_fast(
        vmuladd_fast(vmuladd_fast(vmuladd_fast(c0, x, c1), x, c2), x, c3),
        x,
        c4,
      ),
      x,
      c5,
    )
  end
  @inline function expm1b_kernel_16(::Val{ℯ}, x)
    c5 = 1.000000000000000640438225946852982701258391604480638275588427888399057119915227
    c4 = 0.5000000000000004803287542332217715766116071510522583538834274474935075866898906
    c3 = 0.1666666666420970554527207281834431226735741940161719645985080160507073865025253
    c2 = 0.04166666666018301921665935823120024420659933010915524936059730214858712998209511
    c1 = 0.008333472974984405879148046293292753978131598285368755170712233472964279745777974
    c0 = 0.001388912162496018623851591184688043066476532389093553363939521815388727152058955
    x * vmuladd_fast(
      vmuladd_fast(
        vmuladd_fast(vmuladd_fast(vmuladd_fast(c0, x, c1), x, c2), x, c3),
        x,
        c4,
      ),
      x,
      c5,
    )
  end
  @inline function expm1b_kernel_16(::Val{10}, x)
    c5 = 2.302585092994047158681503503460480873860999793973827515869365761962430703319599
    c4 = 2.650949055239201551934947671858339219424413336755573194223955910456821842421188
    c3 = 2.034678591993528625083054018110717593492391962926107057414972651492780932119609
    c2 = 1.171255148730010832148986815739840479496404930758611797062000621828706728750671
    c1 = 0.5393919676343165962473794696403862709895176169091768718271240564783020989456929
    c0 = 0.2069993173257113377172910397724414085027323868592924170210484263853229417141011
    x * vmuladd_fast(
      vmuladd_fast(
        vmuladd_fast(vmuladd_fast(vmuladd_fast(c0, x, c1), x, c2), x, c3),
        x,
        c4,
      ),
      x,
      c5,
    )
  end

  # @inline function expm1b_kernel_6(::Val{2}, x)
  #     c6 = 0.6931471805599453094157857128856867906777808839530204388709017060018142940776171
  #     c5 = 0.2402265069591008469523412357777710806331204676211882505524452303635329805655358
  #     c4 = 0.05550410866482161698117481145997657675888637052903928922860089114193615876343007
  #     c3 = 0.009618129106525681209446910869436511674612664593938506025921861852732097374219765
  #     c2 = 0.001333355814485111028814491629997422287837080138262113640339762815224264878390602
  #     c1 = 0.0001540375624549508734984308915404920724081959405205139643458055193131059150331185
  #     c0 = 1.525295744513115862409484203574886089068776710280414076942740530839031792795809e-05
  #     x * vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(c0,x,c1),x,c2),x,c3),x,c4),x,c5),x,c6)
  # end
  # # using Remez
  # # N,D,E,X = ratfn_minimax(x -> (exp2(x) - big"1")/x, [big(nextfloat(-0.03125)),big(0.03125)], 4, 0); @show(E); N
  # @inline function expm1b_kernel_4(::Val{2}, x)
  #     c4 = 0.6931471805599461972549081995383434692316977327912755704234013405443109498729026
  #     c3 = 0.2402265069131940842333497738928958607054740795078596615709864445611497846077303
  #     c2 = 0.05550410865300270379171299778517151376504051870524903806295523325435530249981495
  #     c1 = 0.009618317140648284298097106744730186251913149278152053357630395863210686828434175
  #     c0 = 0.001333381881551676348461495248002642715422207072457864472267417920610122672570108
  #     x * vmuladd_fast(vmuladd_fast(vmuladd_fast(vmuladd_fast(c0,x,c1),x,c2),x,c3),x,c4)
  # end

  # @inline function vexp2_v4(x::AbstractSIMD{W,Float64}) where {W}
  #     # r - VectorizationBase.vroundscale(r, Val(16*(4)))
  #     r = VectorizationBase.vsreduce(x,Val(0))
  #     rscale = VectorizationBase.vroundscale(r, Val(64))
  #     rs = r - rscale
  #     inds = convert(UInt, vsreduce(rscale, Val(1))*16.0)
  #     expr = expm1b_kernel_5(Val(2), rs)
  #     N_float = x - rs
  #     # @show inds rs N_float

  #     js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
  #     small_part = vfmadd(js, expr, js)
  #     res = vscalef(small_part, N_float)
  #     return res

  # end

  ####################################################################################################
  ################################## Non-AVX512 implementation #######################################
  ####################################################################################################

  # With AVX512, we use a tiny look-up table of just 16 numbers for `Float64`,
  # because we can perform the lookup using `vpermi2pd`, which is much faster than gather.
  # To compensate, we need a larger polynomial.

  # Because of the larger polynomial, this implementation works better on systems with 2 FMA units.

  @inline function vexp2(x::AbstractSIMD{8,Float64}, ::True)
    r = vsreduce(16.0x, Val(0)) * 0.0625
    N_float = x - r
    expr = expm1b_kernel_16(Val(2), r)
    inds = convert(UInt64, vsreduce(N_float, Val(1)) * 16.0)
    # inds = ((trunc(Int64, 16.0*N_float)%UInt64)) & 0x000000000000000f
    js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
    small_part = vfmadd(js, expr, js)
    res = vscalef(small_part, N_float)
    return res
  end
  # @inline function vexp2_v2(x::AbstractSIMD{8,Float64}, ::True)#, ::Val{N}) where {N}
  #     r1 = vsreduce(x, Val(0))
  #     m = x - r1
  #     r = vfmsub(vsreduce(r1 * 16.0, Val(1)), 0.0625, 0.5)
  #     j = r1 - r
  #     js = vpermi2pd(convert(UInt, j), TABLE_EXP_64_0, TABLE_EXP_64_1)

  #     expr = expm1b_kernel_5(Val(2), r) # 2^r - 1

  #     small_part = vfmadd(js, expr, js)
  #     # @show r m j
  #     vscalef(small_part, m)
  # end
  # @inline function vexp_v3(x::AbstractSIMD{8,Float64}, ::True)#, ::Val{N}) where {N}
  #     xl2e = mul_ieee(1.4426950408889634, x)
  #     r1 = vsreduce(xl2e, Val(0))
  #     m = xl2e - r1
  #     r = vfmsub(vsreduce(r1 * 16.0, Val(1)), 0.0625, 0.5)
  #     j = r1 - r
  #     js = vpermi2pd(convert(UInt, j), TABLE_EXP_64_0, TABLE_EXP_64_1)
  #     rs = vfnmadd(0.6931471805599453094172321214581765680755001343602552541206800094933936219696955, m+j, x)
  #     expr = expm1b_kernel_5(Val(ℯ), r) # 2^r - 1

  #     small_part = vfmadd(js, expr, js)
  #     @show r m j
  #     vscalef(small_part, m)
  # end

  # _log2(::Val{ℯ}) = 1.4426950408889634
  # invlog2hi(::Val{ℯ}) = 0.6931471805599453094172321214581765680755001343602552541206800094933936219696955
  # invlog2lo(::Val{ℯ}) = -2.319046813846299615494855463875478650412068000949339362196969553467383712860567e-17
  # _log2(::Val{10}) = 3.321928094887362347870319429489390175864831393024580612054756395815934776608624
  # invlog2hi(::Val{10}) = 0.3010299956639811952137388947244930267681898814621085413104274611271081892744238
  # invlog2lo(::Val{10}) = 2.803728127785170339013117338996875833689572538872891810725576172209659522828247e-18

  # Requires two more floating point μops, but 8 less loading μops than the default version.
  # This thus microbenchmarks a little worse, but the theory is that using less cache than the
  # 256 Float64 * 8 bytes/Float64 = 2 KiB table may improve real world performance / reduce
  # random latency.
  @inline function vexp_avx512(x::AbstractSIMD{8,Float64}, ::Val{B}) where {B}
    N_float = round(x * LogBo16INV(Val(B), Float64))
    r = muladd(N_float, LogBo16U(Val(B), Float64), x)
    r = muladd(N_float, LogBo16L(Val(B), Float64), r)
    inds = ((trunc(Int64, N_float) % UInt64)) & 0x000000000000000f
    expr = expm1b_kernel_16(Val(B), r)
    js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
    small_part = vfmadd(js, expr, js)
    res = vscalef(small_part, 0.0625 * N_float)
    return res
  end

  @inline function vexp_avx512(x::AbstractSIMD{W,Float64}, ::Val{B}) where {W,B}
    N_float = muladd(x, LogBo256INV(Val{B}(), Float64), MAGIC_ROUND_CONST(Float64))
    N = target_trunc(reinterpret(UInt64, N_float))
    N_float = N_float - MAGIC_ROUND_CONST(Float64)
    r = fma(N_float, LogBo256U(Val{B}(), Float64), x)
    r = fma(N_float, LogBo256L(Val{B}(), Float64), r)
    # @show (N & 0x000000ff) % Int
    # @show N N & 0x000000ff
    js = vload(VectorizationBase.zero_offsets(stridedpointer(J_TABLE)), (N & 0x000000ff,))
    # k = N >>> 0x00000008
    # small_part = reinterpret(UInt64, vfmadd(js, expm1b_kernel(Val{B}(), r), js))
    small_part = vfmadd(js, expm1b_kernel(Val{B}(), r), js)
    # return reinterpret(Float64, small_part), r, k, N_float, js
    res = vscalef(small_part, 0.00390625 * N_float)
    # twopk = (k % UInt64) << 0x0000000000000034
    # res = reinterpret(Float64, twopk + small_part)
    return res
  end
  @inline function vexp_avx512(
    x::Union{Float32,AbstractSIMD{<:Any,Float32}},
    ::Val{B},
  ) where {B}
    N_float = vfmadd(x, LogBINV(Val{B}(), Float32), MAGIC_ROUND_CONST(Float32))
    N = reinterpret(UInt32, N_float)
    N_float = (N_float - MAGIC_ROUND_CONST(Float32))

    r = fast_fma(N_float, LogBU(Val{B}(), Float32), x, fma_fast())
    r = fast_fma(N_float, LogBL(Val{B}(), Float32), r, fma_fast())

    small_part = expb_kernel(Val{B}(), r)
    res = vscalef(small_part, N_float)
    # twopk = N << 0x00000017
    # res = reinterpret(Float32, twopk + small_part)
    return res
  end


else# if !((Sys.ARCH === :x86_64) | (Sys.ARCH === :i686))
  const target_trunc = identity
end

# @inline function vexp_avx512(vu::VecUnroll{1,8,Float64,Vec{8,Float64}}, ::Val{B}) where {B}
#   x, y = data(vu)
#   N_float₁ = round(x*LogBo16INV(Val(B), Float64))
#   N_float₂ = muladd(y, LogBo256INV(Val{B}(), Float64), MAGIC_ROUND_CONST(Float64))
#   r₁ = muladd(N_float₁, LogBo16U(Val(B), Float64), x)
#   N₂ = target_trunc(reinterpret(UInt64, N_float₂))
#   r₁ = muladd(N_float₁, LogBo16L(Val(B), Float64), r₁)
#   js₂ = vload(VectorizationBase.zero_offsets(stridedpointer(J_TABLE)), (N₂ & 0x000000ff,))
#   N_float₂ = N_float₂ - MAGIC_ROUND_CONST(Float64)
#   inds₁ = ((trunc(Int64, N_float₁)%UInt64)) & 0x000000000000000f
#   r₂ = fma(N_float₂, LogBo256U(Val{B}(), Float64), y)
#   expr₁ = expm1b_kernel_16(Val(B), r₁)
#   r₂ = fma(N_float₂, LogBo256L(Val{B}(), Float64), r₂)
#   js₁ = vpermi2pd(inds₁, TABLE_EXP_64_0, TABLE_EXP_64_1)
#   small_part₁ = vfmadd(js₁, expr₁, js₁)
#   small_part₂ = vfmadd(js₂, expm1b_kernel(Val{B}(), r₂), js₂)
#   res₁ = vscalef(small_part₁, 0.0625*N_float₁)
#   res₂ = vscalef(small_part₂, 0.00390625*N_float₂)
#   return VecUnroll((res₁, res₂))
# end

# @inline _vexp(x, ::True) = vexp2( 1.4426950408889634 * x, True() )
# @inline _vexp(x, ::True) = vexp2( mul_ieee(1.4426950408889634, x), True() )
# @inline _vexp10(x, ::True) = vexp2( 3.321928094887362 * x, True() )
# @inline _vexp(x) = _vexp(x, has_feature(Val(:x86_64_avx512f)))
# @inline _vexp10(x) = _vexp10(x, has_feature(Val(:x86_64_avx512f)))

@inline vexp(x::AbstractSIMD, ::True) = vexp_avx512(x, Val(ℯ))
@inline vexp2(x::AbstractSIMD, ::True) = vexp_avx512(x, Val(2))
@inline vexp10(x::AbstractSIMD, ::True) = vexp_avx512(x, Val(10))
# @inline vexp(x::AbstractSIMD{W,Float32}, ::True) where {W} = vexp_generic(x, Val(ℯ))
# @inline vexp2(x::AbstractSIMD{W,Float32}, ::True) where {W} = vexp_generic(x, Val(2))
# @inline vexp10(x::AbstractSIMD{W,Float32}, ::True) where {W} = vexp_generic(x, Val(10))

@inline Base.exp(v::AbstractSIMD{W}) where {W} = vexp(float(v))
@inline Base.exp2(v::AbstractSIMD{W}) where {W} = vexp2(float(v))
@inline Base.exp10(v::AbstractSIMD{W}) where {W} = vexp10(float(v))
@static if (Base.libllvm_version ≥ v"11") & ((Sys.ARCH === :x86_64) | (Sys.ARCH === :i686))
  @inline vexp(v::AbstractSIMD) = vexp(float(v), has_feature(Val(:x86_64_avx512f)))
  @inline vexp2(v::AbstractSIMD) = vexp2(float(v), has_feature(Val(:x86_64_avx512f)))
  @inline vexp10(v::AbstractSIMD) = vexp10(float(v), has_feature(Val(:x86_64_avx512f)))
else
  @inline vexp(v::AbstractSIMD) = vexp(float(v), False())
  @inline vexp2(v::AbstractSIMD) = vexp2(float(v), False())
  @inline vexp10(v::AbstractSIMD) = vexp10(float(v), False())
end
@inline vexp(v::Union{Float32,Float64}) = vexp(v, False())
@inline vexp2(v::Union{Float32,Float64}) = vexp2(v, False())
@inline vexp10(v::Union{Float32,Float64}) = vexp10(v, False())
@inline vexp(v::AbstractSIMD{2,Float32}) = vexp(v, False())
@inline vexp2(v::AbstractSIMD{2,Float32}) = vexp2(v, False())
@inline vexp10(v::AbstractSIMD{2,Float32}) = vexp10(v, False())




####################################################################################################
################################## Non-AVX512 implementation #######################################
####################################################################################################


@inline function vexp_generic_core(
  x::Union{Float64,AbstractSIMD{<:Any,Float64}},
  ::Val{B},
) where {B}
  N_float = muladd(x, LogBo256INV(Val{B}(), Float64), MAGIC_ROUND_CONST(Float64))
  N = target_trunc(reinterpret(UInt64, N_float))
  N_float = N_float - MAGIC_ROUND_CONST(Float64)
  r = fast_fma(N_float, LogBo256U(Val{B}(), Float64), x, fma_fast())
  r = fast_fma(N_float, LogBo256L(Val{B}(), Float64), r, fma_fast())
  # @show (N & 0x000000ff) % Int
  js = vload(VectorizationBase.zero_offsets(stridedpointer(J_TABLE)), (N & 0x000000ff,))
  k = N >>> 0x00000008
  small_part = reinterpret(UInt64, vfmadd(js, expm1b_kernel(Val{B}(), r), js))
  # return reinterpret(Float64, small_part), r, k, N_float, js
  twopk = (k % UInt64) << 0x0000000000000034
  res = reinterpret(Float64, twopk + small_part)
  return res
end
@inline function vexp_generic(
  x::Union{Float64,AbstractSIMD{<:Any,Float64}},
  ::Val{B},
) where {B}
  res = vexp_generic_core(x, Val{B}())
  res = ifelse(x >= MAX_EXP(Val{B}(), Float64), Inf, res)
  res = ifelse(x <= MIN_EXP(Val{B}(), Float64), 0.0, res)
  res = ifelse(isnan(x), x, res)
  return res
end

@inline function vexp_generic_core(
  x::Union{Float32,AbstractSIMD{<:Any,Float32}},
  ::Val{B},
) where {B}
  N_float = vfmadd(x, LogBINV(Val{B}(), Float32), MAGIC_ROUND_CONST(Float32))
  N = reinterpret(UInt32, N_float)
  N_float = (N_float - MAGIC_ROUND_CONST(Float32))

  r = fast_fma(N_float, LogBU(Val{B}(), Float32), x, fma_fast())
  r = fast_fma(N_float, LogBL(Val{B}(), Float32), r, fma_fast())

  small_part = reinterpret(UInt32, expb_kernel(Val{B}(), r))
  twopk = N << 0x00000017
  res = reinterpret(Float32, twopk + small_part)
  return res
end
@inline function vexp_generic(
  x::Union{Float32,AbstractSIMD{<:Any,Float32}},
  ::Val{B},
) where {B}
  res = vexp_generic_core(x, Val{B}())
  res = ifelse(x >= MAX_EXP(Val{B}(), Float32), Inf32, res)
  res = ifelse(x <= MIN_EXP(Val{B}(), Float32), 0.0f0, res)
  res = ifelse(isnan(x), x, res)
  return res
end
for (func, base) in (:vexp2 => Val(2), :vexp => Val(ℯ), :vexp10 => Val(10))
  @eval @inline $func(x, ::False) = vexp_generic(x, $base)
end



####################################################################################################
#################################### LOG HELPERS ###################################################
####################################################################################################

# TODO: move these back to log.jl when when the log implementations there are good & well tested enough to use.

@generated function vgetexp(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.getexp.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, $vtyp undef, $mtyp -1, i32 12)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end
@generated function vgetmant(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.getmant.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 11, $vtyp undef, $mtyp -1, i32 12)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, i32 11, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end
@generated function vgetmant(
  v::Vec{W,T},
  ::Union{StaticInt{N},Val{N}},
) where {W,T<:Union{Float32,Float64},N}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.getmant.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 $N, $vtyp undef, $mtyp -1, i32 12)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, i32 $N, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end
@generated function vgetmant12(v::Vec{W,T}) where {W,T<:Union{Float32,Float64}}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.getmant.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 12, $vtyp undef, $mtyp -1, i32 8)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, i32 12, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end

@generated function vroundscale(v::Vec{W,T}, ::Union{Val{N},StaticInt{N}}) where {W,T,N}
  bits = (8W * sizeof(T))::Int
  any(==(bits), (128, 256, 512)) || throw(
    ArgumentError("Vectors are $bits bits, but only 128, 256, and 512 bits are supported."),
  )
  ltyp = LLVM_TYPES[T]
  vtyp = "<$W x $ltyp>"
  dors = T === Float64 ? 'd' : 's'
  instr = "$vtyp @llvm.x86.avx512.mask.rndscale.p$(dors).$bits"
  mtyp = W == 16 ? "i16" : "i8"
  if bits == 512
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp, i32)"
    instrs = "%res = call $instr($vtyp %0, i32 $N, $vtyp undef, $mtyp -1, i32 4)\nret $vtyp %res"
  else
    decl = "declare $instr($vtyp, i32, $vtyp, $mtyp)"
    instrs = "%res = call $instr($vtyp %0, i32 $N, $vtyp undef, $mtyp -1)\nret $vtyp %res"
  end
  arg_syms = [:(data(v))]
  llvmcall_expr(decl, instrs, :(_Vec{$W,$T}), :(Tuple{_Vec{$W,$T}}), vtyp, [vtyp], arg_syms)
end

@inline vgetexp(v::VecUnroll) = VecUnroll(fmap(vgetexp, getfield(v, :data)))
@inline vgetmant(v::VecUnroll) = VecUnroll(fmap(vgetmant, getfield(v, :data)))
@inline vgetmant(v::VecUnroll, ::Union{StaticInt{N},Val{N}}) where {N} =
  VecUnroll(fmap(vgetmant, getfield(v, :data), StaticInt{N}()))
@inline Base.significand(v::VecUnroll, ::True) =
  VecUnroll(fmap(vgetmant12, getfield(v, :data)))
@inline vroundscale(v::VecUnroll, ::Union{StaticInt{N},Val{N}}) where {N} =
  VecUnroll(fmap(vroundscale, getfield(v, :data), StaticInt{N}()))
@inline Base.significand(v::Vec, ::True) = vgetmant12(v)
@inline Base.significand(v::AbstractSIMDVector, ::True) = vgetmant12(Vec(v))

mask_exponent(::Val{Float64}) = 0x000f_ffff_ffff_ffff
set_exponent(::Val{Float64}) = 0x3ff0_0000_0000_0000

mask_exponent(::Val{Float32}) = 0x007fffff
set_exponent(::Val{Float32}) = 0x3f800000

mask_exponent(::Val{Float16}) = 0x07ff
set_exponent(::Val{Float16}) = 0x3c00

@inline function Base.significand(v::AbstractSIMD{W,T}, ::False) where {W,T}
  reinterpret(
    T,
    (reinterpret(Base.uinttype(T), v) & mask_exponent(Val(T))) | set_exponent(Val(T)),
  )
end
@inline Base.exponent(v::Vec, ::True) = vgetexp(v)
@inline Base.exponent(v::AbstractSIMDVector, ::True) = vgetexp(Vec(v))
@inline Base.exponent(v::VecUnroll, ::True) = VecUnroll(fmap(vgetexp, getfield(v, :data)))
@static if VERSION ≥ v"1.7.0-beta"
  using Base: inttype
else
  @inline inttype(::Type{T}) where {T} = signed(Base.uinttype(T))
end
@inline function Base.exponent(v::AbstractSIMD{W,T}, ::False) where {W,T}
  I = inttype(T)
  vshift = reinterpret(I, v) >> (Base.significand_bits(T) % I)
  e = ((vshift % Int) & Base.exponent_raw_max(T)) - I(Base.exponent_bias(T))
  convert(T, e % Int32)
end

@inline Base.significand(v::AbstractSIMD{W,T}) where {W,T<:Union{Float32,Float64}} =
  significand(v, has_feature(Val(:x86_64_avx512f)))
@inline Base.exponent(v::AbstractSIMD{W,T}) where {W,T<:Union{Float32,Float64}} =
  exponent(v, has_feature(Val(:x86_64_avx512f)))
@inline Base.significand(v::AbstractSIMD{W,Float16}) where {W} = significand(v, False())
@inline Base.exponent(v::AbstractSIMD{W,Float16}) where {W} = exponent(v, False())
@inline Base.ldexp(v::AbstractSIMD, e::AbstractSIMD) =
  vscalef(v, e, has_feature(Val(:x86_64_avx512f)))



# @inline function vexp2_v2(x::AbstractSIMD{8,Float64})
#     x16 = 16.0*x
#     # x8 = 8x
#     r =  vsreduce(x16, Val(0)) * 0.0625
#     @fastmath  begin
#     m = x - r
#     # m + r = x16, r ∈ (-0.5,0.5]
#     # m/16 + r/16 = x, r ∈ (-1/32, 1/32]
#     # we must now vreduce `mfrac`
#     # return m
#     end
#     # expr = expm1b_kernel_4(Val(2), r)
#     expr = expm1b_kernel_5(Val(2), r)
#     inds = convert(UInt64, vsreduce(m, Val(1)) * 16.0)
#     # inds = (reinterpret(UInt64, mfrac) >> 0x000000000000002d) & 0x000000000000000f
#     # inds = (reinterpret(UInt64, mfrac) >> 0x0000000000000035) & 0x000000000000000f
#     # @show r m mfrac reinterpret(UInt64, m) reinterpret(UInt64, mfrac)
#     js = vpermi2pd(inds, TABLE_EXP_64_0, TABLE_EXP_64_1)
#     # return r, mfrac, js
#     # @show js m 16mfrac r mfrac inds%Int ((inds>>1)%Int)
#     # js = 1.0
#     small_part = vfmadd(js, expr, js)
#     vscalef(small_part, m)#, r, mfrac, js, inds
# end

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
