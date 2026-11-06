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
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "dtype_utils.h"
#include "gemm_ex_param.h"
#include "gemm_ex_golden.h"
#include "gemm_ex_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: prepare all test data (float data, quantized bytes, golden, pointers)
// ═══════════════════════════════════════════════════════════════════════════════

struct PreparedData {
    std::vector<float> aFloat;
    std::vector<float> bFloat;
    std::vector<float> cFloat;
    std::vector<uint8_t> aBytes;
    std::vector<uint8_t> bBytes;
    std::vector<uint8_t> cBytes;
    std::vector<float> aGolden;
    std::vector<float> bGolden;
    std::vector<float> cGolden;
    const void* alphaPtr;
    const void* betaPtr;
    const void* aPtr;
    const void* bPtr;
    void* cPtr;
};

inline PreparedData PrepareTestData(const GemmParam& p)
{
    PreparedData data;
    auto physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    auto physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    auto physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    auto physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;
    data.aFloat = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    data.bFloat = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    data.cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);
    data.aBytes = QuantizeMatrix(data.aFloat, p.aFill, p.Atype, p.aNull);
    data.bBytes = QuantizeMatrix(data.bFloat, p.bFill, p.Btype, p.bNull);
    data.cBytes = QuantizeMatrix(data.cFloat, p.cFill, p.Ctype, p.cNull);
    data.aGolden = PrepareGoldenData(data.aFloat, p.Atype, p.aNull);
    data.bGolden = PrepareGoldenData(data.bFloat, p.Btype, p.bNull);
    data.cGolden = PrepareGoldenData(data.cFloat, p.Ctype, p.cNull);
    data.alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    data.betaPtr = p.betaNull ? nullptr : &p.beta;
    data.aPtr = (data.aBytes.empty() || p.aNull) ? nullptr : data.aBytes.data();
    data.bPtr = (data.bBytes.empty() || p.bNull) ? nullptr : data.bBytes.data();
    data.cPtr = (data.cBytes.empty() || p.cNull) ? nullptr : data.cBytes.data();
    return data;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: run NPU, CPU golden, and verify precision
// ═══════════════════════════════════════════════════════════════════════════════

inline void RunAndVerify(const GemmParam& p, aclblasHandle_t testHandle, PreparedData& data)
{
    aclblasStatus_t ret = aclblasGemmEx_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, data.alphaPtr, data.aPtr, p.Atype, p.lda, data.bPtr, p.Btype,
        p.ldb, data.betaPtr, data.cPtr, p.Ctype, p.ldc, p.computeType);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || data.cPtr == nullptr)
        return;

    size_t cCount = static_cast<size_t>(p.ldc) * p.n;
    std::vector<float> cNpuFloat = dequantizeFromBytes(data.cBytes, p.Ctype, cCount);

    const float* aGoldenPtr = data.aGolden.empty() ? nullptr : data.aGolden.data();
    const float* bGoldenPtr = data.bGolden.empty() ? nullptr : data.bGolden.data();
    float* cGoldenPtr = data.cGolden.empty() ? nullptr : data.cGolden.data();

    // issue #820: golden 返回码必须断言检查，golden 失败时不得以输入初值做参考比对
    ASSERT_EQ(
        aclblasGemmEx_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, data.alphaPtr, aGoldenPtr, ACL_FLOAT, p.lda, bGoldenPtr,
            ACL_FLOAT, p.ldb, data.betaPtr, cGoldenPtr, p.Ctype, p.ldc, p.computeType),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    if (p.Ctype == ACL_FLOAT16) {
        cfg.mereThreshold = 0.0012;
    } else {
        cfg.mereThreshold = getMereThreshold(p.Ctype);
    }
    cfg.mareMultiplier = getMareMultiplier(p.Ctype, p.k, p.computeType);

    EXPECT_TRUE(Verifier::verifyVector(cNpuFloat.data(), data.cGolden.data(), cCount, 1, cfg, p.caseName));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class GemmArch35Test : public BlasTest<GemmParam> {};

// ── TEST_F: null handle ──
TEST_F(GemmArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasGemmEx_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, ACL_FLOAT16, 8, nullptr, ACL_FLOAT16, 8, &beta,
        nullptr, ACL_FLOAT16, 8, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Gemm, GemmArch35Test, ::testing::ValuesIn(GetCasesFromCsv<GemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmParam>);

TEST_P(GemmArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = GemmArch35Test::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    auto data = PrepareTestData(p);

    // Early return for error cases
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasGemmEx_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, data.alphaPtr, data.aPtr, p.Atype, p.lda, data.bPtr, p.Btype,
            p.ldb, data.betaPtr, data.cPtr, p.Ctype, p.ldc, p.computeType);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    RunAndVerify(p, testHandle, data);
}
