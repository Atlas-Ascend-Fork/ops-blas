/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgemm_ex_param.h"
#include "cgemm_ex_golden.h"
#include "cgemm_ex_npu_wrapper.h"
#include "dtype_compat.h"

class CgemmExTest : public BlasTest<CgemmExParam> {};

TEST_F(CgemmExTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCgemmEx_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, ACLBLAS_C_32, 8, nullptr, ACLBLAS_C_32, 8, &beta,
        nullptr, ACLBLAS_C_32, 8);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

static bool IsRealType(aclblasCgemmExType_t t) { return t == ACLBLAS_R_32 || t == ACLBLAS_H_R_32; }

static bool IsHalfType(aclblasCgemmExType_t t) { return t == ACLBLAS_H_R_32 || t == ACLBLAS_H_C_32; }

static void PackComplexFromReal(const std::vector<float>& src, std::vector<std::complex<float>>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = std::complex<float>(src[i], 0.0f);
    }
}

static void QuantizeFp16Real(std::vector<float>& v)
{
    for (float& x : v) {
        x = fp16ToFloat(floatToFp16(x));
    }
}

static void QuantizeFp16Cplx(std::vector<std::complex<float>>& v)
{
    for (auto& z : v) {
        z = std::complex<float>(fp16ToFloat(floatToFp16(z.real())), fp16ToFloat(floatToFp16(z.imag())));
    }
}

static void PackHalfReal(const std::vector<float>& src, std::vector<uint16_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = floatToFp16(src[i]);
    }
}

static void PackHalfCplx(const std::vector<std::complex<float>>& src, std::vector<uint16_t>& dst)
{
    dst.resize(src.size() * 2);
    for (size_t i = 0; i < src.size(); i++) {
        dst[2 * i] = floatToFp16(src[i].real());
        dst[2 * i + 1] = floatToFp16(src[i].imag());
    }
}

static const void* HostPtrForType(
    aclblasCgemmExType_t t, const std::vector<float>& f32, const std::vector<std::complex<float>>& c32,
    const std::vector<uint16_t>& hr, const std::vector<uint16_t>& hc)
{
    if (t == ACLBLAS_R_32) {
        return static_cast<const void*>(f32.data());
    }
    if (t == ACLBLAS_H_R_32) {
        return static_cast<const void*>(hr.data());
    }
    if (t == ACLBLAS_H_C_32) {
        return static_cast<const void*>(hc.data());
    }
    return static_cast<const void*>(c32.data());
}

static void* HostPtrForTypeMut(
    aclblasCgemmExType_t t, std::vector<float>& f32, std::vector<std::complex<float>>& c32, std::vector<uint16_t>& hr,
    std::vector<uint16_t>& hc)
{
    if (t == ACLBLAS_R_32) {
        return static_cast<void*>(f32.data());
    }
    if (t == ACLBLAS_H_R_32) {
        return static_cast<void*>(hr.data());
    }
    if (t == ACLBLAS_H_C_32) {
        return static_cast<void*>(hc.data());
    }
    return static_cast<void*>(c32.data());
}

static void PrepareTypedMatrix(
    aclblasCgemmExType_t t, std::vector<float>& f32, std::vector<std::complex<float>>& c32, std::vector<uint16_t>& hr,
    std::vector<uint16_t>& hc)
{
    if (IsRealType(t)) {
        for (auto& z : c32) {
            z = std::complex<float>(z.real(), 0.0f);
        }
        for (size_t i = 0; i < f32.size() && i < c32.size(); i++) {
            f32[i] = c32[i].real();
        }
    }
    if (IsHalfType(t)) {
        if (IsRealType(t)) {
            QuantizeFp16Real(f32);
            PackComplexFromReal(f32, c32);
            PackHalfReal(f32, hr);
        } else {
            QuantizeFp16Cplx(c32);
            PackHalfCplx(c32, hc);
        }
    }
}

static bool HasSpecialFill(const BlasFillMode& f)
{
    if (f.pattern == BlasFillMode::P_EXTREME) {
        return true;
    }
    return f.method == BlasFillMode::M_VALUE && (std::isnan(f.val1) || std::isinf(f.val1));
}

static VerifyConfig MakeVerifyCfg(bool halfC, const std::complex<float>& alphaVal)
{
    VerifyConfig cfg;
    if (alphaVal == std::complex<float>(0.0f, 0.0f) && !halfC) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, halfC ? ACL_FLOAT16 : ACL_FLOAT, nullptr, 0);
    }
    return cfg;
}

static void CompareRealC(
    const CgemmExParam& p, size_t cCount, const std::vector<float>& cFloat, const std::vector<uint16_t>& cHr,
    const std::vector<std::complex<float>>& cGolden, const VerifyConfig& cfg)
{
    std::vector<float> npuReal(cCount);
    std::vector<float> gReal(cCount);
    for (size_t i = 0; i < cCount; i++) {
        gReal[i] = cGolden[i].real();
        if (p.typeC == ACLBLAS_H_R_32) {
            npuReal[i] = fp16ToFloat(cHr[i]);
            gReal[i] = fp16ToFloat(floatToFp16(gReal[i]));
        } else {
            npuReal[i] = cFloat[i];
        }
    }
    EXPECT_TRUE(Verifier::verifyVector(npuReal.data(), gReal.data(), cCount, 1, cfg, p.caseName));
}

static void CompareComplexC(
    const CgemmExParam& p, size_t cCount, const std::vector<std::complex<float>>& cC, const std::vector<uint16_t>& cHc,
    const std::vector<std::complex<float>>& cGolden, const VerifyConfig& cfg)
{
    std::vector<float> npuFlat(cCount * 2);
    std::vector<float> goldFlat(cCount * 2);
    for (size_t i = 0; i < cCount; i++) {
        goldFlat[2 * i] = cGolden[i].real();
        goldFlat[2 * i + 1] = cGolden[i].imag();
        if (p.typeC == ACLBLAS_H_C_32) {
            npuFlat[2 * i] = fp16ToFloat(cHc[2 * i]);
            npuFlat[2 * i + 1] = fp16ToFloat(cHc[2 * i + 1]);
            goldFlat[2 * i] = fp16ToFloat(floatToFp16(goldFlat[2 * i]));
            goldFlat[2 * i + 1] = fp16ToFloat(floatToFp16(goldFlat[2 * i + 1]));
        } else {
            npuFlat[2 * i] = cC[i].real();
            npuFlat[2 * i + 1] = cC[i].imag();
        }
    }
    EXPECT_TRUE(Verifier::verifyVector(npuFlat.data(), goldFlat.data(), cCount * 2, 1, cfg, p.caseName));
}

static aclblasStatus_t CallCgemmEx(
    aclblasHandle_t handle, const CgemmExParam& p, const aclblasComplex* alphaPtr, const aclblasComplex* betaPtr,
    const void* aPtr, const void* bPtr, void* cPtr)
{
    return aclblasCgemmEx_npu(
        handle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.typeA, p.lda, bPtr, p.typeB, p.ldb, betaPtr, cPtr,
        p.typeC, p.ldc);
}

struct CaseBufs {
    std::vector<float> aFloat;
    std::vector<float> bFloat;
    std::vector<float> cFloat;
    std::vector<std::complex<float>> aC;
    std::vector<std::complex<float>> bC;
    std::vector<std::complex<float>> cC;
    std::vector<std::complex<float>> cOrig;
    std::vector<uint16_t> aHr;
    std::vector<uint16_t> bHr;
    std::vector<uint16_t> cHr;
    std::vector<uint16_t> aHc;
    std::vector<uint16_t> bHc;
    std::vector<uint16_t> cHc;
    size_t cCount = 0;
};

static void FillCaseBufs(const CgemmExParam& p, CaseBufs& b)
{
    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;
    b.aFloat = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    b.bFloat = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    b.cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);
    b.cCount = static_cast<size_t>(p.ldc) * std::max(1, p.n);
    PackComplexFromReal(b.aFloat, b.aC);
    PackComplexFromReal(b.bFloat, b.bC);
    PackComplexFromReal(b.cFloat, b.cC);
    PrepareTypedMatrix(p.typeA, b.aFloat, b.aC, b.aHr, b.aHc);
    PrepareTypedMatrix(p.typeB, b.bFloat, b.bC, b.bHr, b.bHc);
    PrepareTypedMatrix(p.typeC, b.cFloat, b.cC, b.cHr, b.cHc);
    b.cOrig = b.cC;
}

static void VerifyCase(const CgemmExParam& p, CaseBufs& b, aclblasHandle_t handle)
{
    std::vector<std::complex<float>> cGolden(b.cOrig);
    std::complex<float> alphaVal(p.alphaReal, p.alphaImag);
    std::complex<float> betaVal(p.betaReal, p.betaImag);
    aclblasCgemmEx_cpu(
        handle, p.transA, p.transB, p.m, p.n, p.k, alphaVal, b.aC.data(), p.lda, b.bC.data(), p.ldb, betaVal,
        cGolden.data(), p.ldc);
    VerifyConfig cfg = MakeVerifyCfg(IsHalfType(p.typeC), alphaVal);
    if (IsRealType(p.typeC)) {
        CompareRealC(p, b.cCount, b.cFloat, b.cHr, cGolden, cfg);
        return;
    }
    CompareComplexC(p, b.cCount, b.cC, b.cHc, cGolden, cfg);
}

static void RunCase(const CgemmExParam& p, aclblasHandle_t testHandle)
{
    CaseBufs b;
    FillCaseBufs(p, b);
    aclblasComplex alphaC{p.alphaReal, p.alphaImag};
    aclblasComplex betaC{p.betaReal, p.betaImag};
    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &alphaC;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &betaC;
    const void* aPtr = p.aNull ? nullptr : HostPtrForType(p.typeA, b.aFloat, b.aC, b.aHr, b.aHc);
    const void* bPtr = p.bNull ? nullptr : HostPtrForType(p.typeB, b.bFloat, b.bC, b.bHr, b.bHc);
    void* cPtr = p.cNull ? nullptr : HostPtrForTypeMut(p.typeC, b.cFloat, b.cC, b.cHr, b.cHc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = CallCgemmEx(testHandle, p, alphaPtr, betaPtr, aPtr, bPtr, cPtr);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    aclblasStatus_t ret = CallCgemmEx(testHandle, p, alphaPtr, betaPtr, aPtr, bPtr, cPtr);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    if (p.m <= 0 || p.n <= 0 || cPtr == nullptr) {
        return;
    }
    if (HasSpecialFill(p.aFill) || HasSpecialFill(p.bFill) || HasSpecialFill(p.cFill)) {
        return;
    }
    VerifyCase(p, b, testHandle);
}

INSTANTIATE_TEST_SUITE_P(
    CgemmEx, CgemmExTest, ::testing::ValuesIn(GetCasesFromCsv<CgemmExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmExParam>);

TEST_P(CgemmExTest, CsvDriven)
{
    const auto& p = GetParam();
    aclblasHandle_t testHandle = CgemmExTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }
    RunCase(p, testHandle);
}
