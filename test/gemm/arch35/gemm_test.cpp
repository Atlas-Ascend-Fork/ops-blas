/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_param.h"
#include "gemm_golden.h"
#include "gemm_npu_wrapper.h"

class GemmTest : public BlasTest<GemmParam> {};

TEST_F(GemmTest, SgemmNullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemm_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, 8, nullptr, 8, &beta, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(GemmTest, CgemmNullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCgemm_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, 8, nullptr, 8, &beta, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

static void RunComplexTestCase(const GemmParam& p, aclblasHandle_t testHandle)
{
    std::complex<float> alphaVal(p.alphaReal, p.alphaImag);
    std::complex<float> betaVal(p.betaReal, p.betaImag);

    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;

    auto aFloat = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    auto bFloat = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    auto cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);

    size_t aCount = static_cast<size_t>(p.lda) * std::max(1, physColsA);
    size_t bCount = static_cast<size_t>(p.ldb) * std::max(1, physColsB);
    size_t cCount = static_cast<size_t>(p.ldc) * std::max(1, p.n);

    std::vector<std::complex<float>> aComplex(aCount);
    std::vector<std::complex<float>> bComplex(bCount);
    std::vector<std::complex<float>> cComplex(cCount);
    for (size_t i = 0; i < aCount && i < aFloat.size(); i++)
        aComplex[i] = std::complex<float>(aFloat[i], 0.0f);
    for (size_t i = 0; i < bCount && i < bFloat.size(); i++)
        bComplex[i] = std::complex<float>(bFloat[i], 0.0f);
    for (size_t i = 0; i < cCount && i < cFloat.size(); i++)
        cComplex[i] = std::complex<float>(cFloat[i], 0.0f);

    std::vector<std::complex<float>> cOrig(cComplex);

    aclblasComplex alphaC{p.alphaReal, p.alphaImag};
    aclblasComplex betaC{p.betaReal, p.betaImag};
    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &alphaC;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &betaC;
    const aclblasComplex* aPtr = p.aNull ? nullptr : reinterpret_cast<const aclblasComplex*>(aComplex.data());
    const aclblasComplex* bPtr = p.bNull ? nullptr : reinterpret_cast<const aclblasComplex*>(bComplex.data());
    aclblasComplex* cPtr = p.cNull ? nullptr : reinterpret_cast<aclblasComplex*>(cComplex.data());

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasCgemm_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasCgemm_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || cPtr == nullptr)
        return;

    std::vector<std::complex<float>> cGolden(cOrig);
    // issue #606: 参考调用失败时 golden 停留在输入初值，会产生假失败
    ASSERT_EQ(
        aclblasCgemm_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaVal, aComplex.data(), p.lda, bComplex.data(), p.ldb,
            betaVal, cGolden.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);

    std::vector<float> cNpuFlat(cCount * 2);
    std::vector<float> cGoldenFlat(cCount * 2);
    for (size_t i = 0; i < cCount; i++) {
        cNpuFlat[2 * i] = cComplex[i].real();
        cNpuFlat[2 * i + 1] = cComplex[i].imag();
        cGoldenFlat[2 * i] = cGolden[i].real();
        cGoldenFlat[2 * i + 1] = cGolden[i].imag();
    }

    VerifyConfig cfg;
    if (alphaVal == std::complex<float>(0.0f, 0.0f)) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, cGoldenFlat.data(), cCount * 2);
    }
    EXPECT_TRUE(Verifier::verifyVector(cNpuFlat.data(), cGoldenFlat.data(), cCount * 2, 1, cfg, p.caseName));
}

static void RunRealTestCase(const GemmParam& p, aclblasHandle_t testHandle)
{
    float alphaVal = p.alphaReal;
    float betaVal = p.betaReal;

    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;

    auto aFloat = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    auto bFloat = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    auto cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);

    std::vector<float> cOrig(cFloat);

    const float* alphaPtr = p.alphaNull ? nullptr : &alphaVal;
    const float* betaPtr = p.betaNull ? nullptr : &betaVal;
    const float* aPtr = p.aNull ? nullptr : aFloat.data();
    const float* bPtr = p.bNull ? nullptr : bFloat.data();
    float* cPtr = p.cNull ? nullptr : cFloat.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasSgemm_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasSgemm_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || cPtr == nullptr)
        return;

    size_t cCount = static_cast<size_t>(p.ldc) * p.n;
    std::vector<float> cGolden(cOrig);
    // issue #606: 参考调用失败时 golden 停留在输入初值，会产生假失败
    ASSERT_EQ(
        aclblasSgemm_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaVal, aFloat.data(), p.lda, bFloat.data(), p.ldb,
            betaVal, cGolden.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    if (alphaVal == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, cGolden.data(), cCount);
    }
    EXPECT_TRUE(Verifier::verifyVector(cFloat.data(), cGolden.data(), cCount, 1, cfg, p.caseName));
}

INSTANTIATE_TEST_SUITE_P(
    Gemm, GemmTest, ::testing::ValuesIn(GetCasesFromCsv<GemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmParam>);

TEST_P(GemmTest, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = GemmTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    if (p.isComplex()) {
        RunComplexTestCase(p, testHandle);
    } else {
        RunRealTestCase(p, testHandle);
    }
}
