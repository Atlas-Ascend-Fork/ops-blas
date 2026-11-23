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
#include <climits>
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "gemv_strided_batched_param.h"
#include "gemv_strided_batched_golden.h"
#include "gemv_strided_batched_npu_wrapper.h"

static uint16_t FloatToHalf(float val)
{
    uint32_t f = 0U;
    if (memcpy_s(&f, sizeof(f), &val, sizeof(val)) != 0) {
        return 0U;
    }
    const uint32_t sign = (f >> 16) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((f >> 23) & 0xFFU) - 127 + 15;
    const uint32_t mant = (f >> 13) & 0x3FFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1U;
    const uint32_t exp = (h >> 10) & 0x1FU;
    const uint32_t mant = h & 0x3FFU;
    uint32_t f = 0U;
    if (exp == 0U) {
        f = (sign << 31) | (mant << 13);
    } else if (exp == 31U) {
        f = (sign << 31) | 0x7F800000U | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127U - 15U) << 23) | (mant << 13);
    }
    float r = 0.0F;
    if (memcpy_s(&r, sizeof(r), &f, sizeof(f)) != 0) {
        return 0.0F;
    }
    return r;
}

static uint16_t FloatToBf16(float val)
{
    uint32_t f = 0U;
    if (memcpy_s(&f, sizeof(f), &val, sizeof(val)) != 0) {
        return 0U;
    }
    return static_cast<uint16_t>(f >> 16);
}

static float Bf16ToFloat(uint16_t b)
{
    const uint32_t f = static_cast<uint32_t>(b) << 16;
    float r = 0.0F;
    if (memcpy_s(&r, sizeof(r), &f, sizeof(f)) != 0) {
        return 0.0F;
    }
    return r;
}

static void QuantizeSbVec(const std::vector<float>& src, std::vector<uint16_t>& dst, bool isBf16)
{
    dst.resize(src.size());
    for (size_t i = 0U; i < src.size(); ++i) {
        dst[i] = isBf16 ? FloatToBf16(src[i]) : FloatToHalf(src[i]);
    }
}

static void DequantizeSbVec(const std::vector<uint16_t>& src, std::vector<float>& dst, bool isBf16)
{
    dst.resize(src.size());
    for (size_t i = 0U; i < src.size(); ++i) {
        dst[i] = isBf16 ? Bf16ToFloat(src[i]) : HalfToFloat(src[i]);
    }
}

static void QuantizeSbRoundTrip(const std::vector<float>& src, std::vector<float>& dst, bool isBf16)
{
    dst.resize(src.size());
    for (size_t i = 0U; i < src.size(); ++i) {
        dst[i] = isBf16 ? Bf16ToFloat(FloatToBf16(src[i])) : HalfToFloat(FloatToHalf(src[i]));
    }
}

static aclblasStatus_t RunGemvSbNpuCase(
    aclblasHandle_t handle, const GemvStridedBatchedParam& p, const std::vector<float>& aFloat,
    const std::vector<float>& xFloat, const std::vector<float>& yFloat, const float* alphaPtr, const float* betaPtr,
    std::vector<float>& aGolden, std::vector<float>& xGolden, std::vector<float>& goldenFloat,
    std::vector<float>& npuFloat)
{
    if (p.dtype == 1) {
        aGolden = aFloat;
        xGolden = xFloat;
        goldenFloat = yFloat;
        npuFloat = yFloat;
        return aclblasSgemvStridedBatched_npu(
            handle, p.trans, p.m, p.n, alphaPtr, aFloat.data(), p.lda, p.strideA, xFloat.data(), p.incx, p.stridex,
            betaPtr, npuFloat.data(), p.incy, p.stridey, p.batchCount);
    }
    const bool isBf16 = (p.dtype == 3 || p.dtype == 4);
    const bool outIsFloat = (p.dtype == 2 || p.dtype == 4);
    std::vector<uint16_t> aQ;
    std::vector<uint16_t> xQ;
    QuantizeSbVec(aFloat, aQ, isBf16);
    QuantizeSbVec(xFloat, xQ, isBf16);
    QuantizeSbRoundTrip(aFloat, aGolden, isBf16);
    QuantizeSbRoundTrip(xFloat, xGolden, isBf16);
    if (outIsFloat) {
        goldenFloat = yFloat;
        npuFloat = yFloat;
        if (isBf16) {
            return aclblasTSSgemvStridedBatched_npu(
                handle, p.trans, p.m, p.n, alphaPtr, aQ.data(), p.lda, p.strideA, xQ.data(), p.incx, p.stridex, betaPtr,
                npuFloat.data(), p.incy, p.stridey, p.batchCount);
        }
        return aclblasHSSgemvStridedBatched_npu(
            handle, p.trans, p.m, p.n, alphaPtr, aQ.data(), p.lda, p.strideA, xQ.data(), p.incx, p.stridex, betaPtr,
            npuFloat.data(), p.incy, p.stridey, p.batchCount);
    }
    std::vector<uint16_t> yNpu;
    QuantizeSbVec(yFloat, yNpu, isBf16);
    QuantizeSbRoundTrip(yFloat, goldenFloat, isBf16);
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    if (isBf16) {
        ret = aclblasTSTgemvStridedBatched_npu(
            handle, p.trans, p.m, p.n, alphaPtr, aQ.data(), p.lda, p.strideA, xQ.data(), p.incx, p.stridex, betaPtr,
            yNpu.data(), p.incy, p.stridey, p.batchCount);
    } else {
        ret = aclblasHSHgemvStridedBatched_npu(
            handle, p.trans, p.m, p.n, alphaPtr, aQ.data(), p.lda, p.strideA, xQ.data(), p.incx, p.stridex, betaPtr,
            yNpu.data(), p.incy, p.stridey, p.batchCount);
    }
    DequantizeSbVec(yNpu, npuFloat, isBf16);
    return ret;
}

static void VerifyGemvSbOutput(
    const GemvStridedBatchedParam& p, int yCount, const std::vector<float>& npuFloat,
    const std::vector<float>& goldenFloat)
{
    if (yCount == 0 || p.batchCount == 0 || p.mereThreshold <= 0.0) {
        return;
    }
    const int absIncy = std::abs(p.incy);
    std::vector<float> npuLogical(static_cast<size_t>(p.batchCount) * static_cast<size_t>(yCount));
    std::vector<float> cpuLogical(static_cast<size_t>(p.batchCount) * static_cast<size_t>(yCount));
    for (int b = 0; b < p.batchCount; ++b) {
        for (int i = 0; i < yCount; ++i) {
            const int yIdx = (p.incy > 0) ? (i * p.incy) : ((yCount - 1 - i) * absIncy);
            npuLogical[static_cast<size_t>(b) * yCount + i] = npuFloat[static_cast<size_t>(b) * p.stridey + yIdx];
            cpuLogical[static_cast<size_t>(b) * yCount + i] = goldenFloat[static_cast<size_t>(b) * p.stridey + yIdx];
        }
    }
    // tolerance tier follows the OUTPUT dtype: HSS/TSS accumulate in FP32,
    // so they must use the FP32 tier (aligned with the arch35 implementation)
    aclDataType outDtype = ACL_FLOAT;
    if (p.dtype == 0) {
        outDtype = ACL_FLOAT16;
    } else if (p.dtype == 3) {
        outDtype = ACL_BF16;
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, outDtype, cpuLogical.data(), cpuLogical.size());
    EXPECT_TRUE(Verifier::verifyVector(
        npuLogical.data(), cpuLogical.data(), static_cast<size_t>(p.batchCount) * yCount, 1, cfg, p.caseName));
}

class GemvStridedBatchedTest : public BlasTest<GemvStridedBatchedParam> {};

TEST_F(GemvStridedBatchedTest, NullHandle)
{
    float alpha = 1.0F;
    float beta = 0.0F;
    std::vector<float> af(64, 1.0F);
    std::vector<float> xf(8, 1.0F);
    std::vector<float> yf(8, 0.0F);
    const aclblasStatus_t ret = aclblasSgemvStridedBatched_npu(
        nullptr, ACLBLAS_OP_N, 8, 8, &alpha, af.data(), 8, 64, xf.data(), 1, 8, &beta, yf.data(), 1, 8, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    GemvStridedBatched, GemvStridedBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<GemvStridedBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemvStridedBatchedParam>);

// Fill the per-batch x/y buffers from the strided generators; copy length is
// clamped to the remaining capacity so degenerate-stride cases cannot overflow.
static void FillStridedXY(
    const GemvStridedBatchedParam& p, int safeBatch, int64_t safeStrideX, int64_t safeStrideY, int xCount, int yCount,
    std::vector<float>& xFloat, std::vector<float>& yFloat)
{
    for (int b = 0; b < safeBatch; ++b) {
        if (p.x.method != BlasFillMode::M_NULLPTR && p.incx != 0 && p.incx != INT_MIN) {
            auto xB = makeBlasStrided(xCount, p.incx, p.x, p.randomSeed + 1 + b);
            const auto xOff = static_cast<std::size_t>(b) * static_cast<std::size_t>(safeStrideX);
            const auto xLen =
                std::min<std::size_t>(xB.size(), xFloat.size() - std::min<std::size_t>(xOff, xFloat.size()));
            std::copy_n(xB.begin(), xLen, xFloat.begin() + static_cast<ptrdiff_t>(xOff));
        }
        if (p.y.method != BlasFillMode::M_NULLPTR && p.incy != 0 && p.incy != INT_MIN) {
            auto yB = makeBlasStrided(yCount, p.incy, p.y, p.randomSeed + 2 + b);
            const auto yOff = static_cast<std::size_t>(b) * static_cast<std::size_t>(safeStrideY);
            const auto yLen =
                std::min<std::size_t>(yB.size(), yFloat.size() - std::min<std::size_t>(yOff, yFloat.size()));
            std::copy_n(yB.begin(), yLen, yFloat.begin() + static_cast<ptrdiff_t>(yOff));
        }
    }
}

TEST_P(GemvStridedBatchedTest, CsvDriven)
{
    const auto& p = GetParam();
    const int safeM = std::max(1, std::abs(p.m));
    const int safeN = std::max(1, std::abs(p.n));
    const int safeBatch = std::max(1, p.batchCount);
    const int64_t safeStrideA = std::max<int64_t>(1, p.strideA);
    const int64_t safeStrideX = std::max<int64_t>(1, p.stridex);
    const int64_t safeStrideY = std::max<int64_t>(1, p.stridey);
    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? safeN : safeM;
    const int yCount = isTransN ? safeM : safeN;

    auto aFloat = makeBlasArray(static_cast<int64_t>(safeBatch) * safeStrideA, p.a, p.randomSeed);
    std::vector<float> xFloat(static_cast<size_t>(safeBatch) * static_cast<size_t>(safeStrideX), 0.0F);
    std::vector<float> yFloat(static_cast<size_t>(safeBatch) * static_cast<size_t>(safeStrideY), 0.0F);
    FillStridedXY(p, safeBatch, safeStrideX, safeStrideY, xCount, yCount, xFloat, yFloat);
    const float* alphaPtr = &p.alpha;
    const float* betaPtr = &p.beta;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        const float* aPtr = (p.a.method == BlasFillMode::M_NULLPTR) ? nullptr : aFloat.data();
        const float* xPtr = (p.x.method == BlasFillMode::M_NULLPTR) ? nullptr : xFloat.data();
        float* yPtr = (p.y.method == BlasFillMode::M_NULLPTR) ? nullptr : yFloat.data();
        const aclblasStatus_t ret = aclblasSgemvStridedBatched_npu(
            GemvStridedBatchedTest::handle_, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda, p.strideA, xPtr, p.incx,
            p.stridex, betaPtr, yPtr, p.incy, p.stridey, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    std::vector<float> aGolden;
    std::vector<float> xGolden;
    std::vector<float> goldenFloat;
    std::vector<float> npuFloat;
    const aclblasStatus_t ret = RunGemvSbNpuCase(
        GemvStridedBatchedTest::handle_, p, aFloat, xFloat, yFloat, alphaPtr, betaPtr, aGolden, xGolden, goldenFloat,
        npuFloat);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    aclblasGemvStridedBatched_cpu(
        p.trans, p.m, p.n, *alphaPtr, aGolden.data(), p.lda, p.strideA, xGolden.data(), p.incx, p.stridex, *betaPtr,
        goldenFloat.data(), p.incy, p.stridey, p.batchCount);
    VerifyGemvSbOutput(p, yCount, npuFloat, goldenFloat);
}
