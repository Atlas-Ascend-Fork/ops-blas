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
#include <limits>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "gemm_grouped_batched_param.h"
#include "gemm_grouped_batched_golden.h"
#include "gemm_grouped_batched_npu_wrapper.h"

namespace {

void CopyFloatSegment(
    std::vector<float>& dst, size_t dstOffset, size_t dstCapacity, const std::vector<float>& src, size_t count)
{
    ASSERT_LE(dstCapacity, dst.size());
    ASSERT_LE(dstOffset, dstCapacity);
    ASSERT_LE(count, dstCapacity - dstOffset);
    ASSERT_LE(count, src.size());
    std::copy_n(src.data(), count, dst.data() + dstOffset);
}

struct GemmGroupedFp32HostData {
    std::vector<float> alphaVec;
    std::vector<float> betaVec;
    std::vector<std::vector<float>> aHostVec;
    std::vector<std::vector<float>> bHostVec;
    std::vector<std::vector<float>> cHostVec;
    std::vector<const float*> aPtrs;
    std::vector<const float*> bPtrs;
    std::vector<float*> cPtrs;
};

struct GemmGroupedSafeGroupDims {
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int safeM = 1;
    int safeN = 1;
    int safeK = 1;
    int safeLda = 1;
    int safeLdb = 1;
    int safeLdc = 1;
    int safeACols = 1;
    int safeBCols = 1;
    int groupSize = 1;
};

int ComputeTestTotalBatchCount(const GemmGroupedBatchedParam& p)
{
    int totalBatchCount = 0;
    if (!p.groupSizeArrayNull) {
        for (int g = 0; g < p.groupCount; g++) {
            totalBatchCount += std::max(0, p.groupSizeArray[g]);
        }
    } else if (p.groupCount > 0) {
        totalBatchCount = p.groupCount;
    }
    return std::max(0, totalBatchCount);
}

GemmGroupedSafeGroupDims GetSafeGroupDims(const GemmGroupedBatchedParam& p, int g)
{
    GemmGroupedSafeGroupDims dims{};
    aclblasOperation_t transaVal = p.transaArrayNull ? ACLBLAS_OP_N : p.transaArray[g];
    aclblasOperation_t transbVal = p.transbArrayNull ? ACLBLAS_OP_N : p.transbArray[g];
    int mVal = p.mArrayNull ? 4 : p.mArray[g];
    int nVal = p.nArrayNull ? 4 : p.nArray[g];
    int kVal = p.kArrayNull ? 4 : p.kArray[g];
    int ldaVal = p.ldaArrayNull ? std::max(1, mVal) : p.ldaArray[g];
    int ldbVal = p.ldbArrayNull ? std::max(1, kVal) : p.ldbArray[g];
    int ldcVal = p.ldcArrayNull ? std::max(1, mVal) : p.ldcArray[g];

    dims.transa = transaVal;
    dims.transb = transbVal;
    dims.safeM = std::max(1, std::abs(mVal));
    dims.safeN = std::max(1, std::abs(nVal));
    dims.safeK = std::max(1, std::abs(kVal));
    dims.safeLda = std::max(dims.safeM, ldaVal > 0 ? ldaVal : dims.safeM);
    dims.safeLdb = std::max(dims.safeK, ldbVal > 0 ? ldbVal : dims.safeK);
    dims.safeLdc = std::max(dims.safeM, ldcVal > 0 ? ldcVal : dims.safeM);
    dims.safeACols = getACols(transaVal, dims.safeM, dims.safeK);
    dims.safeBCols = getBCols(transbVal, dims.safeK, dims.safeN);
    dims.groupSize = p.groupSizeArrayNull ? 1 : p.groupSizeArray[g];
    return dims;
}

void FillOneBatchHostData(
    const GemmGroupedBatchedParam& p, int batchIdx, const GemmGroupedSafeGroupDims& dims, GemmGroupedFp32HostData& data)
{
    if (p.A_fill.method != BlasFillMode::M_NULLPTR) {
        data.aHostVec[batchIdx] =
            makeBlasMatrix(dims.safeLda, dims.safeACols, dims.safeLda, p.A_fill, p.randomSeed + batchIdx * 3);
        data.aPtrs[batchIdx] = data.aHostVec[batchIdx].data();
    }
    if (p.B_fill.method != BlasFillMode::M_NULLPTR) {
        data.bHostVec[batchIdx] =
            makeBlasMatrix(dims.safeLdb, dims.safeBCols, dims.safeLdb, p.B_fill, p.randomSeed + batchIdx * 3 + 1);
        data.bPtrs[batchIdx] = data.bHostVec[batchIdx].data();
    }
    if (p.C_fill.method != BlasFillMode::M_NULLPTR) {
        data.cHostVec[batchIdx] =
            makeBlasMatrix(dims.safeLdc, dims.safeN, dims.safeLdc, p.C_fill, p.randomSeed + batchIdx * 3 + 2);
        data.cPtrs[batchIdx] = data.cHostVec[batchIdx].data();
    }
}

GemmGroupedFp32HostData GenerateGemmGroupedFp32HostData(const GemmGroupedBatchedParam& p, int totalBatchCount)
{
    GemmGroupedFp32HostData data;
    if (!p.alphaArrayNull) {
        data.alphaVec = parseFloatArray(p.alphaArrayStr);
    }
    if (!p.betaArrayNull) {
        data.betaVec = parseFloatArray(p.betaArrayStr);
    }

    data.aHostVec.resize(static_cast<size_t>(totalBatchCount));
    data.bHostVec.resize(static_cast<size_t>(totalBatchCount));
    data.cHostVec.resize(static_cast<size_t>(totalBatchCount));
    data.aPtrs.assign(static_cast<size_t>(totalBatchCount), nullptr);
    data.bPtrs.assign(static_cast<size_t>(totalBatchCount), nullptr);
    data.cPtrs.assign(static_cast<size_t>(totalBatchCount), nullptr);

    int batchIdx = 0;
    for (int g = 0; g < p.groupCount; g++) {
        GemmGroupedSafeGroupDims dims = GetSafeGroupDims(p, g);
        for (int i = 0; i < dims.groupSize; i++) {
            FillOneBatchHostData(p, batchIdx, dims, data);
            batchIdx++;
        }
    }
    return data;
}

const float* ResolveAlphaPtr(const GemmGroupedBatchedParam& p, const GemmGroupedFp32HostData& data)
{
    if (p.alphaArrayNull) {
        return nullptr;
    }
    return data.alphaVec.empty() ? nullptr : data.alphaVec.data();
}

const float* ResolveBetaPtr(const GemmGroupedBatchedParam& p, const GemmGroupedFp32HostData& data)
{
    if (p.betaArrayNull) {
        return nullptr;
    }
    return data.betaVec.empty() ? nullptr : data.betaVec.data();
}

aclblasStatus_t CallGemmGroupedFp32Npu(
    aclblasHandle_t handle, const GemmGroupedBatchedParam& p, const GemmGroupedFp32HostData& data)
{
    const float* alphaPtr = ResolveAlphaPtr(p, data);
    const float* betaPtr = ResolveBetaPtr(p, data);
    const float* const* aArr = (p.A_fill.method == BlasFillMode::M_NULLPTR) ? nullptr : data.aPtrs.data();
    const float* const* bArr = (p.B_fill.method == BlasFillMode::M_NULLPTR) ? nullptr : data.bPtrs.data();
    float* const* cArr = (p.C_fill.method == BlasFillMode::M_NULLPTR) ? nullptr : data.cPtrs.data();

    return aclblasSgemmGroupedBatched_npu(
        handle, p.groupCount, p.transaArrayNull ? nullptr : p.transaArray.data(),
        p.transbArrayNull ? nullptr : p.transbArray.data(), p.mArrayNull ? nullptr : p.mArray.data(),
        p.nArrayNull ? nullptr : p.nArray.data(), p.kArrayNull ? nullptr : p.kArray.data(), alphaPtr, aArr,
        p.ldaArrayNull ? nullptr : p.ldaArray.data(), bArr, p.ldbArrayNull ? nullptr : p.ldbArray.data(), betaPtr, cArr,
        p.ldcArrayNull ? nullptr : p.ldcArray.data(), p.groupSizeArrayNull ? nullptr : p.groupSizeArray.data());
}

void ComputeGemmGroupedFp32Golden(
    aclblasHandle_t handle, const GemmGroupedBatchedParam& p, const GemmGroupedFp32HostData& data,
    const std::vector<std::vector<float>>& cInputVec, int totalBatchCount, std::vector<std::vector<float>>& cGoldenVec)
{
    cGoldenVec.resize(static_cast<size_t>(totalBatchCount));
    std::vector<float*> cGoldenPtrs(static_cast<size_t>(totalBatchCount));
    int batchIdx = 0;
    for (int g = 0; g < p.groupCount; g++) {
        for (int i = 0; i < p.groupSizeArray[g]; i++) {
            cGoldenVec[batchIdx] = cInputVec[batchIdx];
            cGoldenPtrs[batchIdx] = cGoldenVec[batchIdx].data();
            batchIdx++;
        }
    }

    aclblasSgemmGroupedBatched_cpu(
        handle, p.groupCount, p.transaArray.data(), p.transbArray.data(), p.mArray.data(), p.nArray.data(),
        p.kArray.data(), ResolveAlphaPtr(p, data), data.aPtrs.data(), p.ldaArray.data(), data.bPtrs.data(),
        p.ldbArray.data(), ResolveBetaPtr(p, data), cGoldenPtrs.data(), p.ldcArray.data(), p.groupSizeArray.data());
}

void VerifyGemmGroupedFp32Results(
    const GemmGroupedBatchedParam& p, const GemmGroupedFp32HostData& data,
    const std::vector<std::vector<float>>& cGoldenVec)
{
    size_t totalCElements = 0;
    for (int g = 0; g < p.groupCount; g++) {
        totalCElements += static_cast<size_t>(p.ldcArray[g]) * p.nArray[g] * p.groupSizeArray[g];
    }

    std::vector<float> npuFlat(totalCElements);
    std::vector<float> goldFlat(totalCElements);
    size_t offset = 0;
    int batchIdx = 0;
    for (int g = 0; g < p.groupCount; g++) {
        size_t cSize = static_cast<size_t>(p.ldcArray[g]) * p.nArray[g];
        for (int i = 0; i < p.groupSizeArray[g]; i++) {
            CopyFloatSegment(npuFlat, offset, totalCElements, data.cHostVec[batchIdx], cSize);
            CopyFloatSegment(goldFlat, offset, totalCElements, cGoldenVec[batchIdx], cSize);
            offset += cSize;
            batchIdx++;
        }
    }

    if (p.mereThreshold <= 0.0) {
        return;
    }
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    EXPECT_TRUE(Verifier::verifyVector(npuFlat.data(), goldFlat.data(), totalCElements, 1, cfg, p.caseName));
}

void RunGemmGroupedFp32CsvCase(aclblasHandle_t handle, const GemmGroupedBatchedParam& p)
{
    const int totalBatchCount = ComputeTestTotalBatchCount(p);
    GemmGroupedFp32HostData hostData = GenerateGemmGroupedFp32HostData(p, totalBatchCount);
    const std::vector<std::vector<float>> cInputVec = hostData.cHostVec;

    aclblasStatus_t ret = CallGemmGroupedFp32Npu(handle, p, hostData);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<std::vector<float>> cGoldenVec;
    ComputeGemmGroupedFp32Golden(handle, p, hostData, cInputVec, totalBatchCount, cGoldenVec);
    VerifyGemmGroupedFp32Results(p, hostData, cGoldenVec);
}

} // namespace

// ── Test fixture ──

class GemmGroupedBatchedArch35Test : public BlasTest<GemmGroupedBatchedParam> {};

TEST_F(GemmGroupedBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSgemmGroupedBatched_npu(
        nullptr, 1, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(GemmGroupedBatchedArch35Test, TotalBatchOverflow)
{
    // issue #870: sum(groupSize) 超过 uint32 可承载上限时校验阶段返回 INVALID_VALUE，
    // 不进入构建/发射路径（此前 int 溢出经 resize 异常导致进程终止）
    constexpr int kGroupCount = 3;
    std::vector<aclblasOperation_t> transa(kGroupCount, ACLBLAS_OP_N);
    std::vector<aclblasOperation_t> transb(kGroupCount, ACLBLAS_OP_N);
    std::vector<int> m(kGroupCount, 1);
    std::vector<int> n(kGroupCount, 1);
    std::vector<int> k(kGroupCount, 1);
    std::vector<int> lda(kGroupCount, 1);
    std::vector<int> ldb(kGroupCount, 1);
    std::vector<int> ldc(kGroupCount, 1);
    std::vector<float> alpha(kGroupCount, 1.0f);
    std::vector<float> beta(kGroupCount, 0.0f);
    std::vector<int> groupSize(kGroupCount, std::numeric_limits<int>::max());

    float dummyA = 1.0f;
    float dummyB = 1.0f;
    float dummyC = 0.0f;
    std::vector<const float*> aArray(kGroupCount, &dummyA);
    std::vector<const float*> bArray(kGroupCount, &dummyB);
    std::vector<float*> cArray(kGroupCount, &dummyC);

    aclblasStatus_t ret = aclblasSgemmGroupedBatched(
        GemmGroupedBatchedArch35Test::handle_, kGroupCount, transa.data(), transb.data(), m.data(), n.data(), k.data(),
        alpha.data(), aArray.data(), lda.data(), bArray.data(), ldb.data(), beta.data(), cArray.data(), ldc.data(),
        groupSize.data());
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

INSTANTIATE_TEST_SUITE_P(
    GemmGroupedBatched, GemmGroupedBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GemmGroupedBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmGroupedBatchedParam>);

TEST_P(GemmGroupedBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    switch (p.dtype) {
        case 0:
            RunGemmGroupedFp32CsvCase(GemmGroupedBatchedArch35Test::handle_, p);
            break;
        default:
            ASSERT_TRUE(false) << "Unknown dtype: " << p.dtype;
    }
}
