/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_grouped_batched_ex_param.h"
#include "gemm_grouped_batched_ex_golden.h"
#include "gemm_grouped_batched_ex_npu_wrapper.h"

class GemmGroupedBatchedExArch35Test : public BlasTest<GemmGroupedBatchedExParam> {};

namespace {

std::vector<float> DecodeMatrixToFloat(const std::vector<uint8_t>& raw, aclDataType dtype, size_t count)
{
    std::vector<float> out(count);
    if (dtype == ACL_FLOAT16) {
        for (size_t i = 0; i < count; i++) {
            out[i] = HalfToFloat(reinterpret_cast<const uint16_t*>(raw.data())[i]);
        }
    } else if (dtype == ACL_BF16) {
        for (size_t i = 0; i < count; i++) {
            out[i] = BfloatToFloat(reinterpret_cast<const uint16_t*>(raw.data())[i]);
        }
    } else {
        auto* outBytes = reinterpret_cast<uint8_t*>(out.data());
        std::copy_n(raw.data(), count * sizeof(float), outBytes);
    }
    return out;
}

void ExtractColumnMajorLogical(const std::vector<float>& stored, int m, int n, int ld,
    std::vector<float>& out)
{
    out.clear();
    out.reserve(static_cast<size_t>(m) * static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            out.push_back(stored[static_cast<size_t>(i) + static_cast<size_t>(j) * ld]);
        }
    }
}

struct ApiValidationCase {
    bool nullHandle = false;
    bool nullTransa = false;
    bool nullTransb = false;
    bool nullM = false;
    bool nullN = false;
    bool nullK = false;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool nullLda = false;
    bool nullLdb = false;
    bool nullLdc = false;
    bool nullAarray = false;
    bool nullBarray = false;
    bool nullCarray = false;
    bool nullGroupSize = false;
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32;
    int n = 32;
    int k = 32;
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = 32;
    int ldb = 32;
    int ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    aclDataType aType = ACL_FLOAT16;
    aclDataType bType = ACL_FLOAT16;
    aclDataType cType = ACL_FLOAT16;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
};

aclblasStatus_t RunApiValidationCase(aclblasHandle_t handle, ApiValidationCase& tc)
{
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);
    return aclblasGemmGroupedBatchedEx(
        tc.nullHandle ? nullptr : handle,
        tc.nullTransa ? nullptr : &tc.transa, tc.nullTransb ? nullptr : &tc.transb,
        tc.nullM ? nullptr : &tc.m, tc.nullN ? nullptr : &tc.n, tc.nullK ? nullptr : &tc.k,
        tc.nullAlpha ? nullptr : &tc.alpha, tc.nullAarray ? nullptr : aPtrs.data(), tc.aType,
        tc.nullLda ? nullptr : &tc.lda, tc.nullBarray ? nullptr : bPtrs.data(), tc.bType,
        tc.nullLdb ? nullptr : &tc.ldb, tc.nullBeta ? nullptr : &tc.beta,
        tc.nullCarray ? nullptr : cPtrs.data(), tc.cType, tc.nullLdc ? nullptr : &tc.ldc,
        tc.groupCount, tc.nullGroupSize ? nullptr : &tc.groupSize, tc.computeType);
}

void ExpectApiStatus(aclblasHandle_t handle, ApiValidationCase& tc, aclblasStatus_t expected)
{
    EXPECT_EQ(static_cast<int>(RunApiValidationCase(handle, tc)), static_cast<int>(expected));
}

bool IsFp8Type(aclDataType type)
{
    return type == ACL_FLOAT8_E4M3FN || type == ACL_FLOAT8_E5M2;
}

BlasFillMode AdjustRandomFp8Fill(BlasFillMode fill, aclDataType dtype, int safeK)
{
    if (IsFp8Type(dtype) && fill.method == BlasFillMode::M_RANDOM) {
        float kBound = 1.0f / std::sqrt(static_cast<float>(safeK));
        fill.val1 = kBound;
        fill.val2 = kBound;
    }
    return fill;
}

void EncodeFloatMatrix(std::vector<uint8_t>& raw, const std::vector<float>& values, aclDataType dtype)
{
    size_t count = values.size();
    raw.resize(count * DtypeElementSizeNpu(dtype));
    for (size_t i = 0; i < count; ++i) {
        if (dtype == ACL_FLOAT16) {
            reinterpret_cast<uint16_t*>(raw.data())[i] = FloatToHalf(values[i]);
        } else if (dtype == ACL_BF16) {
            reinterpret_cast<uint16_t*>(raw.data())[i] = FloatToBfloat(values[i]);
        } else {
            auto* dst = raw.data() + i * sizeof(float);
            const auto* src = reinterpret_cast<const uint8_t*>(&values[i]);
            std::copy_n(src, sizeof(float), dst);
        }
    }
}

void FillFp8Raw(std::vector<uint8_t>& raw, size_t count,
    std::mt19937& rng, std::uniform_int_distribution<int>& fp8Dist)
{
    raw.resize(count);
    for (size_t i = 0; i < count; ++i) {
        raw[i] = static_cast<uint8_t>(fp8Dist(rng));
    }
}

void PrepareInputRaw(std::vector<uint8_t>& raw, aclDataType dtype, int rows, int cols, int ld,
    const BlasFillMode& fill, int seed, std::mt19937& rng, std::uniform_int_distribution<int>& fp8Dist)
{
    if (IsFp8Type(dtype)) {
        FillFp8Raw(raw, static_cast<size_t>(ld) * cols, rng, fp8Dist);
        return;
    }
    std::vector<float> values = makeBlasMatrix(rows, cols, ld, fill, seed);
    EncodeFloatMatrix(raw, values, dtype);
}

void PrepareCaseData(const GemmGroupedBatchedExParam& p,
    std::vector<std::vector<uint8_t>>& aRawData,
    std::vector<std::vector<uint8_t>>& bRawData,
    std::vector<std::vector<uint8_t>>& cRawData,
    std::vector<std::vector<float>>& cInitFloat)
{
    std::mt19937 rng(p.randomSeed ? p.randomSeed : 42);
    std::uniform_int_distribution<int> fp8Dist(0x38, 0x4F);
    for (int g = 0; g < p.groupCount; ++g) {
        int safeM = std::max(1, std::abs(p.mArray[g]));
        int safeN = std::max(1, std::abs(p.nArray[g]));
        int safeK = std::max(1, std::abs(p.kArray[g]));
        int rowsA = (p.transaArray[g] == ACLBLAS_OP_N) ? safeM : safeK;
        int colsA = (p.transaArray[g] == ACLBLAS_OP_N) ? safeK : safeM;
        int rowsB = (p.transbArray[g] == ACLBLAS_OP_N) ? safeK : safeN;
        int colsB = (p.transbArray[g] == ACLBLAS_OP_N) ? safeN : safeK;
        BlasFillMode aFillAdj = AdjustRandomFp8Fill(p.aFill, p.Atype, safeK);
        BlasFillMode bFillAdj = AdjustRandomFp8Fill(p.bFill, p.Btype, safeK);
        int start = GroupInstanceStart(g, p.groupSizeArray.data());
        for (int inst = 0; inst < p.groupSizeArray[g]; ++inst) {
            int idx = start + inst;
            PrepareInputRaw(aRawData[idx], p.Atype, rowsA, colsA, p.ldaArray[g],
                aFillAdj, p.randomSeed + idx, rng, fp8Dist);
            PrepareInputRaw(bRawData[idx], p.Btype, rowsB, colsB, p.ldbArray[g],
                bFillAdj, p.randomSeed + idx + 1000, rng, fp8Dist);
            cInitFloat[idx] = makeBlasMatrix(safeM, safeN, p.ldcArray[g], p.cFill, p.randomSeed + idx + 2000);
            EncodeFloatMatrix(cRawData[idx], cInitFloat[idx], p.Ctype);
        }
    }
}

aclblasStatus_t RunNpuCase(aclblasHandle_t handle, const GemmGroupedBatchedExParam& p,
    const std::vector<std::vector<uint8_t>>& aRawData,
    const std::vector<std::vector<uint8_t>>& bRawData,
    std::vector<std::vector<uint8_t>>& cRawData)
{
    return aclblasGemmGroupedBatchedEx_npu(handle, p.transaArray.data(), p.transbArray.data(),
        p.mArray.data(), p.nArray.data(), p.kArray.data(), p.alphaArray.data(),
        aRawData, p.Atype, p.ldaArray.data(), bRawData, p.Btype, p.ldbArray.data(),
        p.betaArray.data(), cRawData, p.Ctype, p.ldcArray.data(),
        p.groupCount, p.groupSizeArray.data(), p.computeType);
}

void BuildGoldenInput(const std::vector<std::vector<uint8_t>>& cRawData, aclDataType cType,
    const std::vector<std::vector<float>>& cInitFloat, std::vector<std::vector<float>>& cGolden)
{
    for (size_t idx = 0; idx < cInitFloat.size(); ++idx) {
        cGolden[idx] = DecodeMatrixToFloat(cRawData[idx], cType, cInitFloat[idx].size());
    }
}

bool VerifyCaseOutput(const GemmGroupedBatchedExParam& p,
    const std::vector<std::vector<uint8_t>>& cNpuRaw,
    const std::vector<std::vector<float>>& cGolden, const VerifyConfig& cfg)
{
    bool allPassed = true;
    for (int g = 0; g < p.groupCount; ++g) {
        if (p.mArray[g] <= 0 || p.nArray[g] <= 0) { continue; }
        int start = GroupInstanceStart(g, p.groupSizeArray.data());
        for (int inst = 0; inst < p.groupSizeArray[g]; ++inst) {
            int idx = start + inst;
            std::vector<float> npuFloat = DecodeMatrixToFloat(cNpuRaw[idx], p.Ctype, cGolden[idx].size());
            std::vector<float> npuLogical;
            std::vector<float> goldLogical;
            ExtractColumnMajorLogical(npuFloat, p.mArray[g], p.nArray[g], p.ldcArray[g], npuLogical);
            ExtractColumnMajorLogical(cGolden[idx], p.mArray[g], p.nArray[g], p.ldcArray[g], goldLogical);
            std::string caseName = p.caseName + "_g" + std::to_string(g) + "_i" + std::to_string(inst);
            if (!Verifier::verifyVector(npuLogical.data(), goldLogical.data(), npuLogical.size(), 1, cfg, caseName)) {
                allPassed = false;
            }
        }
    }
    return allPassed;
}

} // namespace

TEST_F(GemmGroupedBatchedExArch35Test, NullHandle)
{
    ApiValidationCase tc;
    tc.nullHandle = true;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(GemmGroupedBatchedExArch35Test, NullTransaArray)
{
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;

    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        GemmGroupedBatchedExArch35Test::handle_, nullptr, nullptr,
        &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullTransbArray_L1E01)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, nullptr, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullMArray_L1E02)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, nullptr, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullNArray_L1E03)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, nullptr, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullKArray_L1E04)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, nullptr, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullAlphaArray_L1E05)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, nullptr,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullBetaArray_L1E06)
{
    ApiValidationCase tc;
    tc.nullBeta = true;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, NullLdaArray_L1E07)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, nullptr,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullLdbArray_L1E08)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, nullptr,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullLdcArray_L1E09)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, nullptr,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullAarray_L1E10)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        nullptr, ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullBarray_L1E11)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        nullptr, ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NullCarray_L1E12)
{
    ApiValidationCase tc;
    tc.nullCarray = true;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, NullGroupSize_L1E13)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupCount = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        groupCount, nullptr, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NegativeGroupCount_L1E14)
{
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 32, n = 32, k = 32;
    float alpha = 1.0f, beta = 0.0f;
    int lda = 32, ldb = 32, ldc = 32;
    int groupSize = 1;
    std::vector<const void*> aPtrs(1, nullptr);
    std::vector<const void*> bPtrs(1, nullptr);
    std::vector<void*> cPtrs(1, nullptr);

    aclblasStatus_t ret = aclblasGemmGroupedBatchedEx(
        handle_, &transa, &transb, &m, &n, &k, &alpha,
        aPtrs.data(), ACL_FLOAT16, &lda,
        bPtrs.data(), ACL_FLOAT16, &ldb,
        &beta, cPtrs.data(), ACL_FLOAT16, &ldc,
        -1, &groupSize, ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemmGroupedBatchedExArch35Test, NegativeM_L1E15)
{
    ApiValidationCase tc;
    tc.m = -1;
    tc.lda = 1;
    tc.ldc = 1;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, NegativeN_L1E16)
{
    ApiValidationCase tc;
    tc.n = -1;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, NegativeK_L1E17)
{
    ApiValidationCase tc;
    tc.k = -1;
    tc.ldb = 1;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, NegativeGroupSize_L1E18)
{
    ApiValidationCase tc;
    tc.groupSize = -1;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, InvalidTransaEnum_L1E19)
{
    ApiValidationCase tc;
    tc.transa = static_cast<aclblasOperation_t>(999);
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_ENUM);
}

TEST_F(GemmGroupedBatchedExArch35Test, InvalidTransbEnum_L1E20)
{
    ApiValidationCase tc;
    tc.transb = static_cast<aclblasOperation_t>(999);
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_ENUM);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdaLessThanM_transaN_L1E21)
{
    ApiValidationCase tc;
    tc.m = 64;
    tc.ldc = 64;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdaLessThanK_transaT_L1E22)
{
    ApiValidationCase tc;
    tc.transa = ACLBLAS_OP_T;
    tc.m = tc.lda = tc.ldc = 64;
    tc.k = tc.ldb = 128;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdbLessThanK_transbN_L1E23)
{
    ApiValidationCase tc;
    tc.k = 128;
    tc.ldb = 64;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdbLessThanN_transbT_L1E24)
{
    ApiValidationCase tc;
    tc.transb = ACLBLAS_OP_T;
    tc.n = 128;
    tc.ldb = 64;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdcLessThanM_L1E25)
{
    ApiValidationCase tc;
    tc.m = 64;
    tc.lda = 64;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, UnsupportedDtypeFP32_L1E26)
{
    ApiValidationCase tc;
    tc.aType = tc.bType = tc.cType = ACL_FLOAT;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_NOT_SUPPORTED);
}

TEST_F(GemmGroupedBatchedExArch35Test, ComputeTypeMismatch_L1E27)
{
    ApiValidationCase tc;
    tc.computeType = ACLBLAS_COMPUTE_64F;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_NOT_SUPPORTED);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdaLessThanK_transaC_L1E28)
{
    ApiValidationCase tc;
    tc.transa = ACLBLAS_OP_C;
    tc.m = tc.ldc = tc.lda = 64;
    tc.k = tc.ldb = 128;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(GemmGroupedBatchedExArch35Test, LdbLessThanN_transbC_L1E29)
{
    ApiValidationCase tc;
    tc.transb = ACLBLAS_OP_C;
    tc.n = 128;
    tc.ldb = 64;
    ExpectApiStatus(handle_, tc, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    GemmGroupedBatchedEx, GemmGroupedBatchedExArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GemmGroupedBatchedExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmGroupedBatchedExParam>);

TEST_P(GemmGroupedBatchedExArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    int totalInst = p.TotalInstanceCount();
    std::vector<std::vector<uint8_t>> aRawData(totalInst), bRawData(totalInst), cRawData(totalInst);
    std::vector<std::vector<float>> cInitFloat(totalInst);
    PrepareCaseData(p, aRawData, bRawData, cRawData, cInitFloat);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = RunNpuCase(handle_, p, aRawData, bRawData, cRawData);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    std::vector<std::vector<uint8_t>> cNpuRaw = cRawData;
    aclblasStatus_t ret = RunNpuCase(handle_, p, aRawData, bRawData, cNpuRaw);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<std::vector<float>> cGolden(totalInst);
    BuildGoldenInput(cRawData, p.Ctype, cInitFloat, cGolden);
    aclblasStatus_t cpuRet = aclblasGemmGroupedBatchedEx_cpu(
        handle_, p.transaArray.data(), p.transbArray.data(),
        p.mArray.data(), p.nArray.data(), p.kArray.data(),
        p.alphaArray.data(),
        aRawData, p.Atype, p.ldaArray.data(),
        bRawData, p.Btype, p.ldbArray.data(),
        p.betaArray.data(),
        cGolden, p.Ctype, p.ldcArray.data(),
        p.groupCount, p.groupSizeArray.data(), p.computeType);

    EXPECT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = (p.mereThreshold > 0.0) ? p.mereThreshold : std::pow(2.0, -10);
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;

    bool allPassed = true;
    allPassed = VerifyCaseOutput(p, cNpuRaw, cGolden, cfg);
    EXPECT_TRUE(allPassed);
}
