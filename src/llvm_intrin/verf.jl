# Copyright (c) 2016, Johan Mabille, Sylvain Corlay, Wolf Vollprecht and Martin Renou
# Copyright (c) 2016, QuantStack
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@generated function _verf(v::Vec{W,Float64}, ::True) where {W}
    bits = 64W
    @assert W ∈ (2,4,8)
    fmastr = "@llvm.fma.v$(W)f64"
    rndstr = "@llvm.x86.avx512.mask.rndscale.pd.$(bits)"
    d2istr = "@llvm.x86.avx512.mask.cvttpd2qq.$(bits)"
    str = """
    attributes #0 = { alwaysinline nounwind readnone }
    declare <$W x double> $(fmastr)(<$W x double>, <$W x double>, <$W x double>) #0
    declare <$W x double> $(rndstr)(<$W x double>, i32, <$W x double>, i8$(W == 8 ? ", i32" : "")) #0
    declare <$W x i64> $(d2istr)(<$W x double>, <$W x i64>, i8$(W == 8 ? ", i32" : "")) #0

    define <$W x double> @entry(<$W x double>) #0 {
    top:
      %1 = bitcast <$W x double> %0 to <$W x i64>
      %2 = and <$W x i64> %1, $(llvmconst(W, "i64 9223372036854775807"))
      %3 = bitcast <$W x i64> %2 to <$W x double>
      %4 = fmul <$W x double> %3, %3
      %5 = bitcast <$W x double> %4 to <$W x i64>
      %6 = fcmp olt <$W x double> %3, $(llvmconst(W, "double 6.500000e-01"))
      %7 = bitcast <$W x i1> %6 to i$(W)
      %8 = icmp eq i$(W) %7, 0
      br i1 %8, label %21, label %9
    ; 9:                                               ; preds = %top
      %10 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> $(llvmconst(W, "double 0x3F110512D5B20332")), <$W x double> $(llvmconst(W, "double 0x3F53B7664358865A"))) #0
      %11 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %10, <$W x double> $(llvmconst(W, "double 0x3FA4A59A4F02579C"))) #0
      %12 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %11, <$W x double> $(llvmconst(W, "double 0x3FC16500F106C0A5"))) #0
      %13 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %12, <$W x double> $(llvmconst(W, "double 0x3FF20DD750429B61"))) #0
      %14 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> $(llvmconst(W, "double 0x3F37EA4332348252")), <$W x double> $(llvmconst(W, "double 0x3F8166F75999DBD1"))) #0
      %15 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %14, <$W x double> $(llvmconst(W, "double 0x3FB64536CA92EA2F"))) #0
      %16 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %15, <$W x double> $(llvmconst(W, "double 0x3FDD0A84EB1CA867"))) #0
      %17 = tail call <$W x double> $(fmastr)(<$W x double> %4, <$W x double> %16, <$W x double> $(llvmconst(W, "double 1.000000e+00"))) #0
      %18 = fmul <$W x double> %13, %0
      %19 = fdiv <$W x double> %18, %17
      %20 = icmp eq i$(W) %7, -1
      br i1 %20, label %106, label %21
    ; 21:                                               ; preds = %9, %top
      %22 = phi <$W x double> [ %19, %9 ], [ zeroinitializer, %top ]
      %23 = fcmp olt <$W x double> %3, $(llvmconst(W, "double 2.200000e+00"))
      %24 = bitcast <$W x i1> %23 to i$(W)
      %25 = xor i$(W) %7, -1
      %26 = and i$(W) %25, %24
      %27 = xor <$W x i64> %5, $(llvmconst(W, "i64 -9223372036854775808"))
      %28 = bitcast <$W x i64> %27 to <$W x double>
      %29 = fmul <$W x double> %28, $(llvmconst(W, "double 0x3FF71547652B82FE"))
      %30 = tail call <$W x double> $(rndstr)(<$W x double> %29, i32 0, <$W x double> zeroinitializer, i8 -1$(W == 8 ? ", i32 4" : ""))
      %31 = tail call <$W x double> $(fmastr)(<$W x double> %30, <$W x double> $(llvmconst(W, "double 0xBFE62E42FEE00000")), <$W x double> %28) #0
      %32 = fmul <$W x double> %30, $(llvmconst(W, "double 0x3DEA39EF35793C76"))
      %33 = fsub <$W x double> %31, %32
      %34 = fmul <$W x double> %33, %33
      %35 = tail call <$W x double> $(fmastr)(<$W x double> %34, <$W x double> $(llvmconst(W, "double 0x3E66376972BEA4D0")), <$W x double> $(llvmconst(W, "double 0xBEBBBD41C5D26BF1"))) #0
      %36 = tail call <$W x double> $(fmastr)(<$W x double> %34, <$W x double> %35, <$W x double> $(llvmconst(W, "double 0x3F11566AAF25DE2C"))) #0
      %37 = tail call <$W x double> $(fmastr)(<$W x double> %34, <$W x double> %36, <$W x double> $(llvmconst(W, "double 0xBF66C16C16BEBD93"))) #0
      %38 = tail call <$W x double> $(fmastr)(<$W x double> %34, <$W x double> %37, <$W x double> $(llvmconst(W, "double 0x3FC555555555553E"))) #0
      %39 = fneg <$W x double> %38
      %40 = tail call <$W x double> $(fmastr)(<$W x double> %34, <$W x double> %39, <$W x double> %33) #0
      %41 = fmul <$W x double> %40, %33
      %42 = fsub <$W x double> $(llvmconst(W, "double 2.000000e+00")), %40
      %43 = fdiv <$W x double> %41, %42
      %44 = fsub <$W x double> $(llvmconst(W, "double 1.000000e+00")), %32
      %45 = fadd <$W x double> %44, %31
      %46 = fadd <$W x double> %45, %43
      %47 = fcmp ole <$W x double> %28, $(llvmconst(W, "double 0xC086232BDD7ABCD2"))
      %48 = tail call <$W x i64> $(d2istr)(<$W x double> %30, <$W x i64> zeroinitializer, i8 -1$(W == 8 ? ", i32 4" : ""))  #0
      %49 = trunc <$W x i64> %48 to <$W x i32>
      %50 = shl <$W x i64> %48, $(llvmconst(W, "i64 52"))
      %51 = add <$W x i64> %50, $(llvmconst(W, "i64 4607182418800017408"))
      %52 = bitcast <$W x i64> %51 to <$W x double>
      %53 = fmul <$W x double> %46, %52
      %54 = select <$W x i1> %47, <$W x double> zeroinitializer, <$W x double> %53
      %55 = fcmp oge <$W x double> %28, $(llvmconst(W, "double 0x40862E42FEFA39EF"))
      %56 = select <$W x i1> %55, <$W x double> $(llvmconst(W, "double 0x7FF0000000000000")), <$W x double> %54
      %57 = icmp eq i$(W) %26, 0
      br i1 %57, label %83, label %58
    ; 58:                                               ; preds = %21
      %59 = tail call fast <$W x double> $(fmastr)(<$W x double> %3, <$W x double> zeroinitializer, <$W x double> $(llvmconst(W, "double 0x3F7CF4CFE0AACBB4"))) #0
      %60 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %59, <$W x double> $(llvmconst(W, "double 0x3FB2488A6B5CB5E5"))) #0
      %61 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %60, <$W x double> $(llvmconst(W, "double 0x3FD53DD7A67C7E9F"))) #0
      %62 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %61, <$W x double> $(llvmconst(W, "double 0x3FEC1986509E687B"))) #0
      %63 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %62, <$W x double> $(llvmconst(W, "double 0x3FF54DFE9B258A60"))) #0
      %64 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %63, <$W x double> $(llvmconst(W, "double 0x3FEFFFFFFBBB552B"))) #0
      %65 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> $(llvmconst(W, "double 0x3F89A996639B0D00")), <$W x double> $(llvmconst(W, "double 0x3FC033C113A7DEEE"))) #0
      %66 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %65, <$W x double> $(llvmconst(W, "double 0x3FE307622FCFF772"))) #0
      %67 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %66, <$W x double> $(llvmconst(W, "double 0x3FF9E677C2777C3C"))) #0
      %68 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %67, <$W x double> $(llvmconst(W, "double 0x40053B1052DCA8BD"))) #0
      %69 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %68, <$W x double> $(llvmconst(W, "double 0x4003ADEAE79B9708"))) #0
      %70 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %69, <$W x double> $(llvmconst(W, "double 1.000000e+00"))) #0
      %71 = fmul <$W x double> %56, %64
      %72 = fdiv <$W x double> %71, %70
      %73 = fsub <$W x double> $(llvmconst(W, "double 1.000000e+00")), %72
      %74 = bitcast <$W x double> %73 to <$W x i64>
      %75 = fcmp olt <$W x double> %0, zeroinitializer
      %76 = xor <$W x i64> %74, $(llvmconst(W, "i64 -9223372036854775808"))
      %77 = bitcast <$W x i64> %76 to <$W x double>
      %78 = select <$W x i1> %75, <$W x double> %77, <$W x double> %73
      %79 = select <$W x i1> %6, <$W x double> %22, <$W x double> %78
      %80 = or <$W x i1> %23, %6
      %81 = bitcast <$W x i1> %80 to i$(W)
      %82 = icmp eq i$(W) %81, -1
      br i1 %82, label %106, label %83
    ; 83:                                               ; preds = %58, %21
      %84 = phi <$W x double> [ %79, %58 ], [ %22, %21 ]
      %85 = tail call fast <$W x double> $(fmastr)(<$W x double> %3, <$W x double> zeroinitializer, <$W x double> $(llvmconst(W, "double 0x3F971D0907EA7A92")))
      %86 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %85, <$W x double> $(llvmconst(W, "double 0x3FC42210F88B9D43"))) #0
      %87 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %86, <$W x double> $(llvmconst(W, "double 0x3FE29BE1CFF90D94"))) #0
      %88 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %87, <$W x double> $(llvmconst(W, "double 0x3FF44744306832AE"))) #0
      %89 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %88, <$W x double> $(llvmconst(W, "double 0x3FF9FA202DEB88E5"))) #0
      %90 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %89, <$W x double> $(llvmconst(W, "double 0x3FEFFF5A9E697AE2"))) #0
      %91 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> $(llvmconst(W, "double 0x3FA47BD61BBB3843")), <$W x double> $(llvmconst(W, "double 0x3FD1D7AB774BB837"))) #0
      %92 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %91, <$W x double> $(llvmconst(W, "double 0x3FF0CFD4CB6CDE9F"))) #0
      %93 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %92, <$W x double> $(llvmconst(W, "double 0x400315FFDFD5CE91"))) #0
      %94 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %93, <$W x double> $(llvmconst(W, "double 0x400AFD487397568F"))) #0
      %95 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %94, <$W x double> $(llvmconst(W, "double 0x400602F24BF3FDB6"))) #0
      %96 = tail call <$W x double> $(fmastr)(<$W x double> %3, <$W x double> %95, <$W x double> $(llvmconst(W, "double 1.000000e+00"))) #0
      %97 = fmul <$W x double> %56, %90
      %98 = fdiv <$W x double> %97, %96
      %99 = fsub <$W x double> $(llvmconst(W, "double 1.000000e+00")), %98
      %100 = bitcast <$W x double> %99 to <$W x i64>
      %101 = fcmp olt <$W x double> %0, zeroinitializer
      %102 = xor <$W x i64> %100, $(llvmconst(W, "i64 -9223372036854775808"))
      %103 = bitcast <$W x i64> %102 to <$W x double>
      %104 = select <$W x i1> %101, <$W x double> %103, <$W x double> %99
      %105 = select <$W x i1> %23, <$W x double> %84, <$W x double> %104
      br label %106
    ; 106:                                              ; preds = %9, %58, %83
      %107 = phi <$W x double> [ %19, %9 ], [ %105, %83 ], [ %79, %58 ]
      ret <$W x double> %107
    }
    """
    return quote
        $(Expr(:meta,:inline))
        Vec(Base.llvmcall(($str, "entry"), _Vec{$W,Float64}, Tuple{_Vec{$W,Float64}}, data(v)))
    end
end
@inline _verf(v::Vec{4,Float64}, ::False) = __verf(v, has_feature(Val(:x86_64_avx)))
__verf(v::Vec{4,Float64}, ::False) = throw("`__verf` with `Vec{4,Float64}` requires a CPU that supports AVX instructions.")
@inline function __verf(v::Vec{4,Float64}, ::True)
    Vec(Base.llvmcall(("""
    attributes #0 = { alwaysinline nounwind readnone }
    declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0
    declare i32 @llvm.x86.avx.vtestz.pd.256(<4 x double>, <4 x double>) #0
    declare i32 @llvm.x86.avx.vtestc.pd.256(<4 x double>, <4 x double>) #0
    declare <4 x double> @llvm.x86.avx.round.pd.256(<4 x double>, i32) #0
    declare <4 x i32> @llvm.x86.avx.cvtt.pd2dq.256(<4 x double>) #0

    define <4 x double> @entry(<4 x double>) #0 {
    top:
      %1 = bitcast <4 x double> %0 to <4 x i64>
      %2 = and <4 x i64> %1, <i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807, i64 9223372036854775807>
      %3 = bitcast <4 x i64> %2 to <4 x double>
      %4 = fmul <4 x double> %3, %3
      %5 = bitcast <4 x double> %4 to <4 x i64>
      %6 = fcmp olt <4 x double> %3, <double 6.500000e-01, double 6.500000e-01, double 6.500000e-01, double 6.500000e-01>
      %7 = sext <4 x i1> %6 to <4 x i64>
      %8 = bitcast <4 x i64> %7 to <4 x double>
      %9 = tail call i32 @llvm.x86.avx.vtestz.pd.256(<4 x double> %8, <4 x double> %8) #16
      %10 = icmp eq i32 %9, 0
      br i1 %10, label %11, label %24
    ; 11:                                               ; preds = %top
      %12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> <double 0x3F110512D5B20332, double 0x3F110512D5B20332, double 0x3F110512D5B20332, double 0x3F110512D5B20332>, <4 x double> <double 0x3F53B7664358865A, double 0x3F53B7664358865A, double 0x3F53B7664358865A, double 0x3F53B7664358865A>) #16
      %13 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %12, <4 x double> <double 0x3FA4A59A4F02579C, double 0x3FA4A59A4F02579C, double 0x3FA4A59A4F02579C, double 0x3FA4A59A4F02579C>) #16
      %14 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %13, <4 x double> <double 0x3FC16500F106C0A5, double 0x3FC16500F106C0A5, double 0x3FC16500F106C0A5, double 0x3FC16500F106C0A5>) #16
      %15 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %14, <4 x double> <double 0x3FF20DD750429B61, double 0x3FF20DD750429B61, double 0x3FF20DD750429B61, double 0x3FF20DD750429B61>) #16
      %16 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> <double 0x3F37EA4332348252, double 0x3F37EA4332348252, double 0x3F37EA4332348252, double 0x3F37EA4332348252>, <4 x double> <double 0x3F8166F75999DBD1, double 0x3F8166F75999DBD1, double 0x3F8166F75999DBD1, double 0x3F8166F75999DBD1>) #16
      %17 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %16, <4 x double> <double 0x3FB64536CA92EA2F, double 0x3FB64536CA92EA2F, double 0x3FB64536CA92EA2F, double 0x3FB64536CA92EA2F>) #16
      %18 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %17, <4 x double> <double 0x3FDD0A84EB1CA867, double 0x3FDD0A84EB1CA867, double 0x3FDD0A84EB1CA867, double 0x3FDD0A84EB1CA867>) #16
      %19 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %4, <4 x double> %18, <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>) #16
      %20 = fmul <4 x double> %15, %0
      %21 = fdiv <4 x double> %20, %19
      %22 = tail call i32 @llvm.x86.avx.vtestc.pd.256(<4 x double> %8, <4 x double> <double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF>) #16
      %23 = icmp eq i32 %22, 0
      br i1 %23, label %24, label %114
    ; 24:                                               ; preds = %11, %0
      %25 = phi <4 x double> [ %21, %11 ], [ zeroinitializer, %top ]
      %26 = fcmp olt <4 x double> %3, <double 2.200000e+00, double 2.200000e+00, double 2.200000e+00, double 2.200000e+00>
      %27 = xor <4 x i1> %6, <i1 true, i1 true, i1 true, i1 true>
      %28 = and <4 x i1> %26, %27
      %29 = sext <4 x i1> %28 to <4 x i64>
      %30 = bitcast <4 x i64> %29 to <4 x double>
      %31 = xor <4 x i64> %5, <i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808>
      %32 = bitcast <4 x i64> %31 to <4 x double>
      %33 = fmul <4 x double> %32, <double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE, double 0x3FF71547652B82FE>
      %34 = tail call <4 x double> @llvm.x86.avx.round.pd.256(<4 x double> %33, i32 0)
      %35 = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %34
      %36 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %35, <4 x double> <double 0x3FE62E42FEE00000, double 0x3FE62E42FEE00000, double 0x3FE62E42FEE00000, double 0x3FE62E42FEE00000>, <4 x double> %32) #16
      %37 = fmul <4 x double> %34, <double 0x3DEA39EF35793C76, double 0x3DEA39EF35793C76, double 0x3DEA39EF35793C76, double 0x3DEA39EF35793C76>
      %38 = fsub <4 x double> %36, %37
      %39 = fmul <4 x double> %38, %38
      %40 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %39, <4 x double> <double 0x3E66376972BEA4D0, double 0x3E66376972BEA4D0, double 0x3E66376972BEA4D0, double 0x3E66376972BEA4D0>, <4 x double> <double 0xBEBBBD41C5D26BF1, double 0xBEBBBD41C5D26BF1, double 0xBEBBBD41C5D26BF1, double 0xBEBBBD41C5D26BF1>) #16
      %41 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %39, <4 x double> %40, <4 x double> <double 0x3F11566AAF25DE2C, double 0x3F11566AAF25DE2C, double 0x3F11566AAF25DE2C, double 0x3F11566AAF25DE2C>) #16
      %42 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %39, <4 x double> %41, <4 x double> <double 0xBF66C16C16BEBD93, double 0xBF66C16C16BEBD93, double 0xBF66C16C16BEBD93, double 0xBF66C16C16BEBD93>) #16
      %43 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %39, <4 x double> %42, <4 x double> <double 0x3FC555555555553E, double 0x3FC555555555553E, double 0x3FC555555555553E, double 0x3FC555555555553E>) #16
      %44 = fsub <4 x double> <double -0.000000e+00, double -0.000000e+00, double -0.000000e+00, double -0.000000e+00>, %39
      %45 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %44, <4 x double> %43, <4 x double> %38) #16
      %46 = fmul <4 x double> %45, %38
      %47 = fsub <4 x double> <double 2.000000e+00, double 2.000000e+00, double 2.000000e+00, double 2.000000e+00>, %45
      %48 = fdiv <4 x double> %46, %47
      %49 = fsub <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %37
      %50 = fadd <4 x double> %49, %36
      %51 = fadd <4 x double> %50, %48
      %52 = fcmp ole <4 x double> %32, <double 0xC086232BDD7ABCD2, double 0xC086232BDD7ABCD2, double 0xC086232BDD7ABCD2, double 0xC086232BDD7ABCD2>
      %53 = tail call <4 x i32> @llvm.x86.avx.cvtt.pd2dq.256(<4 x double> %34) #16
      %54 = zext <4 x i32> %53 to <4 x i64>
      %55 = shl <4 x i64> %54, <i64 52, i64 52, i64 52, i64 52>
      %56 = add <4 x i64> %55, <i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408, i64 4607182418800017408>
      %57 = bitcast <4 x i64> %56 to <4 x double>
      %58 = fmul <4 x double> %51, %57
      %59 = select <4 x i1> %52, <4 x double> zeroinitializer, <4 x double> %58
      %60 = fcmp oge <4 x double> %32, <double 0x40862E42FEFA39EF, double 0x40862E42FEFA39EF, double 0x40862E42FEFA39EF, double 0x40862E42FEFA39EF>
      %61 = select <4 x i1> %60, <4 x double> <double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000, double 0x7FF0000000000000>, <4 x double> %59
      %62 = tail call i32 @llvm.x86.avx.vtestz.pd.256(<4 x double> %30, <4 x double> %30) #16
      %63 = icmp eq i32 %62, 0
      br i1 %63, label %64, label %91
    ; 64:                                               ; preds = %24
      %65 = tail call fast <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> zeroinitializer, <4 x double> <double 0x3F7CF4CFE0AACBB4, double 0x3F7CF4CFE0AACBB4, double 0x3F7CF4CFE0AACBB4, double 0x3F7CF4CFE0AACBB4>) #16
      %66 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %65, <4 x double> <double 0x3FB2488A6B5CB5E5, double 0x3FB2488A6B5CB5E5, double 0x3FB2488A6B5CB5E5, double 0x3FB2488A6B5CB5E5>) #16
      %67 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %66, <4 x double> <double 0x3FD53DD7A67C7E9F, double 0x3FD53DD7A67C7E9F, double 0x3FD53DD7A67C7E9F, double 0x3FD53DD7A67C7E9F>) #16
      %68 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %67, <4 x double> <double 0x3FEC1986509E687B, double 0x3FEC1986509E687B, double 0x3FEC1986509E687B, double 0x3FEC1986509E687B>) #16
      %69 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %68, <4 x double> <double 0x3FF54DFE9B258A60, double 0x3FF54DFE9B258A60, double 0x3FF54DFE9B258A60, double 0x3FF54DFE9B258A60>) #16
      %70 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %69, <4 x double> <double 0x3FEFFFFFFBBB552B, double 0x3FEFFFFFFBBB552B, double 0x3FEFFFFFFBBB552B, double 0x3FEFFFFFFBBB552B>) #16
      %71 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> <double 0x3F89A996639B0D00, double 0x3F89A996639B0D00, double 0x3F89A996639B0D00, double 0x3F89A996639B0D00>, <4 x double> <double 0x3FC033C113A7DEEE, double 0x3FC033C113A7DEEE, double 0x3FC033C113A7DEEE, double 0x3FC033C113A7DEEE>) #16
      %72 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %71, <4 x double> <double 0x3FE307622FCFF772, double 0x3FE307622FCFF772, double 0x3FE307622FCFF772, double 0x3FE307622FCFF772>) #16
      %73 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %72, <4 x double> <double 0x3FF9E677C2777C3C, double 0x3FF9E677C2777C3C, double 0x3FF9E677C2777C3C, double 0x3FF9E677C2777C3C>) #16
      %74 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %73, <4 x double> <double 0x40053B1052DCA8BD, double 0x40053B1052DCA8BD, double 0x40053B1052DCA8BD, double 0x40053B1052DCA8BD>) #16
      %75 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %74, <4 x double> <double 0x4003ADEAE79B9708, double 0x4003ADEAE79B9708, double 0x4003ADEAE79B9708, double 0x4003ADEAE79B9708>) #16
      %76 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %75, <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>) #16
      %77 = fmul <4 x double> %61, %70
      %78 = fdiv <4 x double> %77, %76
      %79 = fsub <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %78
      %80 = bitcast <4 x double> %79 to <4 x i64>
      %81 = fcmp olt <4 x double> %0, zeroinitializer
      %82 = xor <4 x i64> %80, <i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808>
      %83 = bitcast <4 x i64> %82 to <4 x double>
      %84 = select <4 x i1> %81, <4 x double> %83, <4 x double> %79
      %85 = select <4 x i1> %6, <4 x double> %25, <4 x double> %84
      %86 = or <4 x i1> %26, %6
      %87 = sext <4 x i1> %86 to <4 x i64>
      %88 = bitcast <4 x i64> %87 to <4 x double>
      %89 = tail call i32 @llvm.x86.avx.vtestc.pd.256(<4 x double> %88, <4 x double> <double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF, double 0xFFFFFFFFFFFFFFFF>) #16
      %90 = icmp eq i32 %89, 0
      br i1 %90, label %91, label %114
    ; 91:                                               ; preds = %64, %24
      %92 = phi <4 x double> [ %85, %64 ], [ %25, %24 ]
      %93 = tail call fast <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> zeroinitializer, <4 x double> <double 0x3F971D0907EA7A92, double 0x3F971D0907EA7A92, double 0x3F971D0907EA7A92, double 0x3F971D0907EA7A92>) #16
      %94 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %93, <4 x double> <double 0x3FC42210F88B9D43, double 0x3FC42210F88B9D43, double 0x3FC42210F88B9D43, double 0x3FC42210F88B9D43>) #16
      %95 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %94, <4 x double> <double 0x3FE29BE1CFF90D94, double 0x3FE29BE1CFF90D94, double 0x3FE29BE1CFF90D94, double 0x3FE29BE1CFF90D94>) #16
      %96 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %95, <4 x double> <double 0x3FF44744306832AE, double 0x3FF44744306832AE, double 0x3FF44744306832AE, double 0x3FF44744306832AE>) #16
      %97 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %96, <4 x double> <double 0x3FF9FA202DEB88E5, double 0x3FF9FA202DEB88E5, double 0x3FF9FA202DEB88E5, double 0x3FF9FA202DEB88E5>) #16
      %98 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %97, <4 x double> <double 0x3FEFFF5A9E697AE2, double 0x3FEFFF5A9E697AE2, double 0x3FEFFF5A9E697AE2, double 0x3FEFFF5A9E697AE2>) #16
      %99 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> <double 0x3FA47BD61BBB3843, double 0x3FA47BD61BBB3843, double 0x3FA47BD61BBB3843, double 0x3FA47BD61BBB3843>, <4 x double> <double 0x3FD1D7AB774BB837, double 0x3FD1D7AB774BB837, double 0x3FD1D7AB774BB837, double 0x3FD1D7AB774BB837>) #16
      %100 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %99, <4 x double> <double 0x3FF0CFD4CB6CDE9F, double 0x3FF0CFD4CB6CDE9F, double 0x3FF0CFD4CB6CDE9F, double 0x3FF0CFD4CB6CDE9F>) #16
      %101 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %100, <4 x double> <double 0x400315FFDFD5CE91, double 0x400315FFDFD5CE91, double 0x400315FFDFD5CE91, double 0x400315FFDFD5CE91>) #16
      %102 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %101, <4 x double> <double 0x400AFD487397568F, double 0x400AFD487397568F, double 0x400AFD487397568F, double 0x400AFD487397568F>) #16
      %103 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %102, <4 x double> <double 0x400602F24BF3FDB6, double 0x400602F24BF3FDB6, double 0x400602F24BF3FDB6, double 0x400602F24BF3FDB6>) #16
      %104 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %3, <4 x double> %103, <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>) #16
      %105 = fmul <4 x double> %61, %98
      %106 = fdiv <4 x double> %105, %104
      %107 = fsub <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %106
      %108 = bitcast <4 x double> %107 to <4 x i64>
      %109 = fcmp olt <4 x double> %0, zeroinitializer
      %110 = xor <4 x i64> %108, <i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808, i64 -9223372036854775808>
      %111 = bitcast <4 x i64> %110 to <4 x double>
      %112 = select <4 x i1> %109, <4 x double> %111, <4 x double> %107
      %113 = select <4 x i1> %26, <4 x double> %92, <4 x double> %112
      br label %114
    ; 114:                                              ; preds = %11, %64, %91
      %115 = phi <4 x double> [ %21, %11 ], [ %113, %91 ], [ %85, %64 ]
      ret <4 x double> %115
    }
    """, "entry"), _Vec{4,Float64}, Tuple{_Vec{4,Float64}}, data(v)))
end
@inline _verf(v::Vec{2,Float64}, ::False) = __verf(v, has_feature(Val(Symbol("x86_64_sse4.1"))))
__verf(v::Vec{2,Float64}, ::False) = throw("`__verf` with `Vec{2,Float64}` requires a CPU that supports SSE 4.1 instructions.")
@inline function __verf(v::Vec{2,Float64}, ::True)
    Vec(Base.llvmcall(("""
        attributes #0 = { alwaysinline nounwind readnone }
        declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #0
        declare <2 x double> @llvm.x86.sse41.round.pd(<2 x double>, i32) #0
        declare <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double>) #0

        define <2 x double> @entry(<2 x double>) #0 {
        top:
          %1 = bitcast <2 x double> %0 to <2 x i64>
          %2 = and <2 x i64> %1, <i64 9223372036854775807, i64 9223372036854775807>
          %3 = bitcast <2 x i64> %2 to <2 x double>
          %4 = fmul <2 x double> %3, %3
          %5 = bitcast <2 x double> %4 to <2 x i64>
          %6 = fcmp olt <2 x double> %3, <double 6.500000e-01, double 6.500000e-01>
          %7 = sext <2 x i1> %6 to <2 x i64>
          %8 = bitcast <2 x i1> %6 to i2
          %9 = icmp eq i2 %8, 0
          br i1 %9, label %22, label %10
        ; 10:                                               ; preds = %top
          %11 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> <double 0x3F110512D5B20332, double 0x3F110512D5B20332>, <2 x double> <double 0x3F53B7664358865A, double 0x3F53B7664358865A>) #16
          %12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %11, <2 x double> <double 0x3FA4A59A4F02579C, double 0x3FA4A59A4F02579C>) #16
          %13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %12, <2 x double> <double 0x3FC16500F106C0A5, double 0x3FC16500F106C0A5>) #16
          %14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %13, <2 x double> <double 0x3FF20DD750429B61, double 0x3FF20DD750429B61>) #16
          %15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> <double 0x3F37EA4332348252, double 0x3F37EA4332348252>, <2 x double> <double 0x3F8166F75999DBD1, double 0x3F8166F75999DBD1>) #16
          %16 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %15, <2 x double> <double 0x3FB64536CA92EA2F, double 0x3FB64536CA92EA2F>) #16
          %17 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %16, <2 x double> <double 0x3FDD0A84EB1CA867, double 0x3FDD0A84EB1CA867>) #16
          %18 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %4, <2 x double> %17, <2 x double> <double 1.000000e+00, double 1.000000e+00>) #16
          %19 = fmul <2 x double> %14, %0
          %20 = fdiv <2 x double> %19, %18
          %21 = icmp eq i2 %8, -1
          br i1 %21, label %117, label %22
        ; 22:                                               ; preds = %10, %top
          %23 = phi <2 x double> [ %20, %10 ], [ zeroinitializer, %top ]
          %24 = fcmp olt <2 x double> %3, <double 2.200000e+00, double 2.200000e+00>
          %25 = bitcast <2 x i64> %7 to <4 x i32>
          %26 = icmp eq <4 x i32> %25, zeroinitializer
          %27 = sext <4 x i1> %26 to <4 x i32>
          %28 = bitcast <4 x i32> %27 to <2 x i64>
          %29 = select <2 x i1> %24, <2 x i64> %28, <2 x i64> zeroinitializer
          %30 = xor <2 x i64> %5, <i64 -9223372036854775808, i64 -9223372036854775808>
          %31 = bitcast <2 x i64> %30 to <2 x double>
          %32 = fmul <2 x double> %31, <double 0x3FF71547652B82FE, double 0x3FF71547652B82FE>
          %33 = tail call <2 x double> @llvm.x86.sse41.round.pd(<2 x double> %32, i32 0)
          %34 = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %33
          %35 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %34, <2 x double> <double 0x3FE62E42FEE00000, double 0x3FE62E42FEE00000>, <2 x double> %31) #16
          %36 = fmul <2 x double> %33, <double 0x3DEA39EF35793C76, double 0x3DEA39EF35793C76>
          %37 = fsub <2 x double> %35, %36
          %38 = fmul <2 x double> %37, %37
          %39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %38, <2 x double> <double 0x3E66376972BEA4D0, double 0x3E66376972BEA4D0>, <2 x double> <double 0xBEBBBD41C5D26BF1, double 0xBEBBBD41C5D26BF1>) #16
          %40 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %38, <2 x double> %39, <2 x double> <double 0x3F11566AAF25DE2C, double 0x3F11566AAF25DE2C>) #16
          %41 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %38, <2 x double> %40, <2 x double> <double 0xBF66C16C16BEBD93, double 0xBF66C16C16BEBD93>) #16
          %42 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %38, <2 x double> %41, <2 x double> <double 0x3FC555555555553E, double 0x3FC555555555553E>) #16
          %43 = fsub <2 x double> <double -0.000000e+00, double -0.000000e+00>, %38
          %44 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %43, <2 x double> %42, <2 x double> %37) #16
          %45 = fmul <2 x double> %44, %37
          %46 = fsub <2 x double> <double 2.000000e+00, double 2.000000e+00>, %44
          %47 = fdiv <2 x double> %45, %46
          %48 = fsub <2 x double> <double 1.000000e+00, double 1.000000e+00>, %36
          %49 = fadd <2 x double> %48, %35
          %50 = fadd <2 x double> %49, %47
          %51 = fcmp ole <2 x double> %31, <double 0xC086232BDD7ABCD2, double 0xC086232BDD7ABCD2>
          %52 = tail call <4 x i32> @llvm.x86.sse2.cvttpd2dq(<2 x double> %33) #16
          %53 = bitcast <4 x i32> %52 to <2 x i64>
          %54 = ashr <2 x i64> %53, <i64 63, i64 63>
          %55 = bitcast <2 x i64> %54 to <4 x i32>
          %56 = shufflevector <4 x i32> %52, <4 x i32> %55, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
          %57 = bitcast <4 x i32> %56 to <2 x i64>
          %58 = shl <2 x i64> %57, <i64 52, i64 52>
          %59 = add <2 x i64> %58, <i64 4607182418800017408, i64 4607182418800017408>
          %60 = bitcast <2 x i64> %59 to <2 x double>
          %61 = fmul <2 x double> %50, %60
          %62 = select <2 x i1> %51, <2 x double> zeroinitializer, <2 x double> %61
          %63 = fcmp oge <2 x double> %31, <double 0x40862E42FEFA39EF, double 0x40862E42FEFA39EF>
          %64 = select <2 x i1> %63, <2 x double> <double 0x7FF0000000000000, double 0x7FF0000000000000>, <2 x double> %62
          %65 = icmp slt <2 x i64> %29, zeroinitializer
          %66 = bitcast <2 x i1> %65 to i2
          %67 = icmp eq i2 %66, 0
          br i1 %67, label %94, label %68
        ; 68:                                               ; preds = %22
          %69 = tail call fast <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> zeroinitializer, <2 x double> <double 0x3F7CF4CFE0AACBB4, double 0x3F7CF4CFE0AACBB4>) #16
          %70 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %69, <2 x double> <double 0x3FB2488A6B5CB5E5, double 0x3FB2488A6B5CB5E5>) #16
          %71 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %70, <2 x double> <double 0x3FD53DD7A67C7E9F, double 0x3FD53DD7A67C7E9F>) #16
          %72 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %71, <2 x double> <double 0x3FEC1986509E687B, double 0x3FEC1986509E687B>) #16
          %73 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %72, <2 x double> <double 0x3FF54DFE9B258A60, double 0x3FF54DFE9B258A60>) #16
          %74 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %73, <2 x double> <double 0x3FEFFFFFFBBB552B, double 0x3FEFFFFFFBBB552B>) #16
          %75 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> <double 0x3F89A996639B0D00, double 0x3F89A996639B0D00>, <2 x double> <double 0x3FC033C113A7DEEE, double 0x3FC033C113A7DEEE>) #16
          %76 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %75, <2 x double> <double 0x3FE307622FCFF772, double 0x3FE307622FCFF772>) #16
          %77 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %76, <2 x double> <double 0x3FF9E677C2777C3C, double 0x3FF9E677C2777C3C>) #16
          %78 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %77, <2 x double> <double 0x40053B1052DCA8BD, double 0x40053B1052DCA8BD>) #16
          %79 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %78, <2 x double> <double 0x4003ADEAE79B9708, double 0x4003ADEAE79B9708>) #16
          %80 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %79, <2 x double> <double 1.000000e+00, double 1.000000e+00>) #16
          %81 = fmul <2 x double> %64, %74
          %82 = fdiv <2 x double> %81, %80
          %83 = fsub <2 x double> <double 1.000000e+00, double 1.000000e+00>, %82
          %84 = bitcast <2 x double> %83 to <2 x i64>
          %85 = fcmp olt <2 x double> %0, zeroinitializer
          %86 = xor <2 x i64> %84, <i64 -9223372036854775808, i64 -9223372036854775808>
          %87 = select <2 x i1> %85, <2 x i64> %86, <2 x i64> %84
          %88 = bitcast <2 x i64> %87 to <2 x double>
          %89 = select <2 x i1> %6, <2 x double> %23, <2 x double> %88
          %90 = or <2 x i64> %29, %7
          %91 = icmp slt <2 x i64> %90, zeroinitializer
          %92 = bitcast <2 x i1> %91 to i2
          %93 = icmp eq i2 %92, -1
          br i1 %93, label %117, label %94
        ; 94:                                               ; preds = %68, %22
          %95 = phi <2 x double> [ %89, %68 ], [ %23, %22 ]
          %96 = tail call fast <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> zeroinitializer, <2 x double> <double 0x3F971D0907EA7A92, double 0x3F971D0907EA7A92>) #16
          %97 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %96, <2 x double> <double 0x3FC42210F88B9D43, double 0x3FC42210F88B9D43>) #16
          %98 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %97, <2 x double> <double 0x3FE29BE1CFF90D94, double 0x3FE29BE1CFF90D94>) #16
          %99 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %98, <2 x double> <double 0x3FF44744306832AE, double 0x3FF44744306832AE>) #16
          %100 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %99, <2 x double> <double 0x3FF9FA202DEB88E5, double 0x3FF9FA202DEB88E5>) #16
          %101 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %100, <2 x double> <double 0x3FEFFF5A9E697AE2, double 0x3FEFFF5A9E697AE2>) #16
          %102 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> <double 0x3FA47BD61BBB3843, double 0x3FA47BD61BBB3843>, <2 x double> <double 0x3FD1D7AB774BB837, double 0x3FD1D7AB774BB837>) #16
          %103 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %102, <2 x double> <double 0x3FF0CFD4CB6CDE9F, double 0x3FF0CFD4CB6CDE9F>) #16
          %104 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %103, <2 x double> <double 0x400315FFDFD5CE91, double 0x400315FFDFD5CE91>) #16
          %105 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %104, <2 x double> <double 0x400AFD487397568F, double 0x400AFD487397568F>) #16
          %106 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %105, <2 x double> <double 0x400602F24BF3FDB6, double 0x400602F24BF3FDB6>) #16
          %107 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %3, <2 x double> %106, <2 x double> <double 1.000000e+00, double 1.000000e+00>) #16
          %108 = fmul <2 x double> %64, %101
          %109 = fdiv <2 x double> %108, %107
          %110 = fsub <2 x double> <double 1.000000e+00, double 1.000000e+00>, %109
          %111 = bitcast <2 x double> %110 to <2 x i64>
          %112 = fcmp olt <2 x double> %0, zeroinitializer
          %113 = xor <2 x i64> %111, <i64 -9223372036854775808, i64 -9223372036854775808>
          %114 = select <2 x i1> %112, <2 x i64> %113, <2 x i64> %111
          %115 = bitcast <2 x i64> %114 to <2 x double>
          %116 = select <2 x i1> %24, <2 x double> %95, <2 x double> %115
          br label %117
        ; 117:                                              ; preds = %10, %68, %94
          %118 = phi <2 x double> [ %20, %10 ], [ %116, %94 ], [ %89, %68 ]
          ret <2 x double> %118
        }
        """, "entry"), _Vec{2,Float64}, Tuple{_Vec{2,Float64}}, data(v)))
end

@inline function _verf(v::Vec{16,Float32}, ::True)
    Vec(Base.llvmcall(("""
        attributes #0 = { alwaysinline }
        declare <16 x float> @llvm.fma.v16f32(<16 x float>, <16 x float>, <16 x float>) #0
        declare <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float>, i32, <16 x float>, i16, i32) #0
        declare <16 x i32> @llvm.x86.avx512.mask.cvttps2dq.512(<16 x float>, <16 x i32>, i16, i32) #0
        define <16 x float> @entry( <16 x float>) #0 {
        top:
          %1 = bitcast <16 x float> %0 to <8 x i64>
          %2 = and <8 x i64> %1, <i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159, i64 9223372034707292159>
          %3 = bitcast <8 x i64> %2 to <16 x float>
          %4 = fcmp olt <16 x float> %3, <float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000>
          %5 = bitcast <16 x i1> %4 to i16
          %6 = icmp eq i16 %5, 0
          br i1 %6, label %16, label %7
        7:                                                ; preds = %top
          %8 = fmul <16 x float> %3, %3
          %9 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %8, <16 x float> <float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000>, <16 x float> <float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000>) #16
          %10 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %8, <16 x float> %9, <16 x float> <float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000>) #16
          %11 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %8, <16 x float> %10, <16 x float> <float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000>) #16
          %12 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %8, <16 x float> %11, <16 x float> <float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000>) #16
          %13 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %8, <16 x float> %12, <16 x float> <float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000>) #16
          %14 = fmul <16 x float> %13, %0
          %15 = icmp eq i16 %5, -1
          br i1 %15, label %67, label %16
        16:                                               ; preds = %7, %top
          %17 = phi <16 x float> [ %14, %7 ], [ zeroinitializer, %top ]
          %18 = fadd <16 x float> %3, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
          %19 = fdiv <16 x float> %3, %18
          %20 = fadd <16 x float> %19, <float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000>
          %21 = fsub <16 x float> zeroinitializer, %3
          %22 = fmul <16 x float> %21, %3
          %23 = fmul <16 x float> %22, <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>
          %24 = tail call <16 x float> @llvm.x86.avx512.mask.rndscale.ps.512(<16 x float> %23, i32 0, <16 x float> zeroinitializer, i16 -1, i32 4)
          %25 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %24, <16 x float> <float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000, float 0xBFE6300000000000>, <16 x float> %22) #16
          %26 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %24, <16 x float> <float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000, float 0x3F2BD01060000000>, <16 x float> %25) #16
          %27 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %26, <16 x float> <float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000>, <16 x float> <float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000>) #16
          %28 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %26, <16 x float> %27, <16 x float> <float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000>) #16
          %29 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %26, <16 x float> %28, <16 x float> <float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000>) #16
          %30 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %26, <16 x float> %29, <16 x float> <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>) #16
          %31 = fmul <16 x float> %26, %26
          %32 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %30, <16 x float> %31, <16 x float> %26) #16
          %33 = fadd <16 x float> %32, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
          %34 = fcmp ole <16 x float> %22, <float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000>
          %35 = tail call <16 x i32> @llvm.x86.avx512.mask.cvttps2dq.512(<16 x float> %24, <16 x i32> zeroinitializer, i16 -1, i32 4) #16
          %36 = shl <16 x i32> %35, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
          %37 = add <16 x i32> %36, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
          %38 = bitcast <16 x i32> %37 to <16 x float>
          %39 = fmul <16 x float> %33, %38
          %40 = select <16 x i1> %34, <16 x float> zeroinitializer, <16 x float> %39
          %41 = fcmp oge <16 x float> %22, <float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000>
          %42 = select <16 x i1> %41, <16 x float> <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>, <16 x float> %40
          %43 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> <float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000>, <16 x float> <float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000>) #16
          %44 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %43, <16 x float> <float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000>) #16
          %45 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %44, <16 x float> <float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000>) #16
          %46 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %45, <16 x float> <float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000>) #16
          %47 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %46, <16 x float> <float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000>) #16
          %48 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %47, <16 x float> <float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000>) #16
          %49 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %48, <16 x float> <float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000>) #16
          %50 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %49, <16 x float> <float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000>) #16
          %51 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %50, <16 x float> <float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000>) #16
          %52 = tail call <16 x float> @llvm.fma.v16f32(<16 x float> %20, <16 x float> %51, <16 x float> <float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000>) #16
          %53 = fmul <16 x float> %52, %42
          %54 = fsub <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %53
          %55 = fcmp olt <16 x float> %0, zeroinitializer
          %56 = fsub <16 x float> zeroinitializer, %54
          %57 = select <16 x i1> %55, <16 x float> %56, <16 x float> %54
          %58 = select <16 x i1> %4, <16 x float> %17, <16 x float> %57
          %59 = fcmp oeq <16 x float> %3, <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>
          %60 = fcmp ogt <16 x float> %0, zeroinitializer
          %61 = select <16 x i1> %60, <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <16 x float> zeroinitializer
          %62 = select <16 x i1> %55, <16 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <16 x float> zeroinitializer
          %63 = fsub <16 x float> %61, %62
          %64 = fcmp uno <16 x float> %0, zeroinitializer
          %65 = select <16 x i1> %64, <16 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, <16 x float> %63
          %66 = select <16 x i1> %59, <16 x float> %65, <16 x float> %58
          br label %67
        67:                                               ; preds = %7, %16
          %68 = phi <16 x float> [ %14, %7 ], [ %66, %16 ]
          ret <16 x float> %68
        }
        """, "entry"), _Vec{16,Float32}, Tuple{_Vec{16,Float32}}, data(v)))
end

@inline function _verf(v::Vec{8,Float32}, ::True)
    Vec(Base.llvmcall(("""
        attributes #0 = { alwaysinline }
        declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
        declare i32 @llvm.x86.avx.vtestz.ps.256(<8 x float>, <8 x float>) #0
        declare i32 @llvm.x86.avx.vtestc.ps.256(<8 x float>, <8 x float>) #0
        declare <8 x float> @llvm.x86.avx.round.ps.256(<8 x float>, i32) #0
        declare <8 x i32> @llvm.x86.avx.cvtt.ps2dq.256(<8 x float>) #0
        define <8 x float> @entry(<8 x float>) #0 {
        top:
          %1 = bitcast <8 x float> %0 to <8 x i32>
          %2 = and <8 x i32> %1, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
          %3 = bitcast <8 x i32> %2 to <8 x float>
          %4 = fcmp olt <8 x float> %3, <float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000, float 0x3FE5555560000000>
          %5 = sext <8 x i1> %4 to <8 x i32>
          %6 = bitcast <8 x i32> %5 to <8 x float>
          %7 = tail call i32 @llvm.x86.avx.vtestz.ps.256(<8 x float> %6, <8 x float> %6) #16
          %8 = icmp eq i32 %7, 0
          br i1 %8, label %9, label %19
        9:                                               ; preds = %top
          %10 = fmul <8 x float> %3, %3
          %11 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %10, <8 x float> <float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000, float 0xBF43F90760000000>, <8 x float> <float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000, float 0x3F7488D1A0000000>) #16
          %12 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %10, <8 x float> %11, <8 x float> <float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000, float 0xBF9B6C3E80000000>) #16
          %13 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %10, <8 x float> %12, <8 x float> <float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000, float 0x3FBCE1E440000000>) #16
          %14 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %10, <8 x float> %13, <8 x float> <float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000, float 0xBFD8126FC0000000>) #16
          %15 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %10, <8 x float> %14, <8 x float> <float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000, float 0x3FF20DD740000000>) #16
          %16 = fmul <8 x float> %15, %0
          %17 = tail call i32 @llvm.x86.avx.vtestc.ps.256(<8 x float> %6, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>) #16
          %18 = icmp eq i32 %17, 0
          br i1 %18, label %19, label %74
        19:                                               ; preds = %9, %top
          %20 = phi <8 x float> [ %16, %9 ], [ zeroinitializer, %top ]
          %21 = fadd <8 x float> %3, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
          %22 = fdiv <8 x float> %3, %21
          %23 = fadd <8 x float> %22, <float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000, float 0xBFD99999A0000000>
          %24 = or <8 x i32> %1, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
          %25 = bitcast <8 x i32> %24 to <8 x float>
          %26 = fmul <8 x float> %25, %3
          %27 = fmul <8 x float> %26, <float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000, float 0x3FF7154760000000>
          %28 = tail call <8 x float> @llvm.x86.avx.round.ps.256(<8 x float> %27, i32 0)
          %29 = fneg <8 x float> %28
          %30 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %29, <8 x float> <float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000, float 0x3FE6300000000000>, <8 x float> %26) #16
          %31 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %29, <8 x float> <float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000, float 0xBF2BD01060000000>, <8 x float> %30) #16
          %32 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %31, <8 x float> <float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000, float 0x3F56EF19E0000000>, <8 x float> <float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000, float 0x3F8131B160000000>) #16
          %33 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %31, <8 x float> %32, <8 x float> <float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000, float 0x3FA5552AE0000000>) #16
          %34 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %31, <8 x float> %33, <8 x float> <float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000, float 0x3FC55534A0000000>) #16
          %35 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %31, <8 x float> %34, <8 x float> <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>) #16
          %36 = fmul <8 x float> %31, %31
          %37 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %35, <8 x float> %36, <8 x float> %31) #16
          %38 = fadd <8 x float> %37, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
          %39 = fcmp ole <8 x float> %26, <float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000, float 0xC0561814A0000000>
          %40 = tail call <8 x i32> @llvm.x86.avx.cvtt.ps2dq.256(<8 x float> %28) #16
          %41 = shl <8 x i32> %40, <i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23, i32 23>
          %42 = add <8 x i32> %41, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
          %43 = bitcast <8 x i32> %42 to <8 x float>
          %44 = fmul <8 x float> %38, %43
          %45 = select <8 x i1> %39, <8 x float> zeroinitializer, <8 x float> %44
          %46 = fcmp oge <8 x float> %26, <float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000, float 0x40561814A0000000>
          %47 = select <8 x i1> %46, <8 x float> <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>, <8 x float> %45
          %48 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> <float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000, float 0xC00506C220000000>, <8 x float> <float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000, float 0x401ACFC760000000>) #16
          %49 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %48, <8 x float> <float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000, float 0xC019A350A0000000>) #16
          %50 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %49, <8 x float> <float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000, float 0x400A160C60000000>) #16
          %51 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %50, <8 x float> <float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000, float 0xBFF5D50CA0000000>) #16
          %52 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %51, <8 x float> <float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000, float 0x3FC400DE00000000>) #16
          %53 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %52, <8 x float> <float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000, float 0x3FC22EB8E0000000>) #16
          %54 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %53, <8 x float> <float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000, float 0x3FD8994DC0000000>) #16
          %55 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %54, <8 x float> <float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000, float 0x3FC4870500000000>) #16
          %56 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %55, <8 x float> <float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000, float 0xBFF2314C40000000>) #16
          %57 = tail call <8 x float> @llvm.fma.v8f32(<8 x float> %23, <8 x float> %56, <8 x float> <float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000, float 0x3FE141D160000000>) #16
          %58 = fmul <8 x float> %57, %47
          %59 = fsub <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %58
          %60 = bitcast <8 x float> %59 to <8 x i32>
          %61 = fcmp olt <8 x float> %0, zeroinitializer
          %62 = xor <8 x i32> %60, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
          %63 = bitcast <8 x i32> %62 to <8 x float>
          %64 = select <8 x i1> %61, <8 x float> %63, <8 x float> %59
          %65 = select <8 x i1> %4, <8 x float> %20, <8 x float> %64
          %66 = fcmp oeq <8 x float> %3, <float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000, float 0x7FF0000000000000>
          %67 = fcmp ogt <8 x float> %0, zeroinitializer
          %68 = select <8 x i1> %67, <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x float> zeroinitializer
          %69 = select <8 x i1> %61, <8 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <8 x float> zeroinitializer
          %70 = fsub <8 x float> %68, %69
          %71 = fcmp uno <8 x float> %0, zeroinitializer
          %72 = select <8 x i1> %71, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, <8 x float> %70
          %73 = select <8 x i1> %66, <8 x float> %72, <8 x float> %65
          br label %74
        74:                                               ; preds = %9, %19
          %75 = phi <8 x float> [ %16, %9 ], [ %73, %19 ]
          ret <8 x float> %75
        }
        """, "entry"), _Vec{8,Float32}, Tuple{_Vec{8,Float32}}, data(v)))
end

@inline verf(v::Vec{W,Float64}) where {W} = _verf(v, has_feature(Val(:x86_64_avx512f)))

_verf(v::Vec{16,Float32}, ::False) = throw("`verf` with `Vec{16,Float32}` requires a CPU that supports AVX512F instructions")
_verf(v::Vec{8,Float32}, ::False) = throw("`verf` with `Vec{8,Float32}` requires a CPU that supports AVX instructions")
@inline verf(v::Vec{16,Float32}) = _verf(v, has_feature(Val(:x86_64_avx512f)))
@inline verf(v::Vec{8,Float32}) = _verf(v, has_feature(Val(:x86_64_avx)))

