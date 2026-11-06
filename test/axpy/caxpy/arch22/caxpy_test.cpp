/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <utility>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "caxpy_param.h"
#include "caxpy_golden.h"
#include "caxpy_npu_wrapper.h"
#include "npu_event_timer.h"

namespace {

constexpr aclblasComplex GUARD_VALUE{12345.25F, -54321.5F};
constexpr size_t GUARD_COUNT = 64;

// Deterministic per-index complex value generator: independent of BlasFillMode
// so cross-checking against the reference cblas-style AXPY is exact, matching
// the layout the legacy hardcoded-array caxpy_test used before the CSV
// migration.
aclblasComplex MakeStridedValue(size_t index, int32_t salt)
{
    const float real = 0.125F * static_cast<float>(static_cast<int32_t>(index % 29) - 14 + salt);
    const float imag = 0.0625F * static_cast<float>(static_cast<int32_t>(index % 23) - 11 - salt);
    return {real, imag};
}

// Build a length-n complex vector (logical, stride-independent index order)
// with GUARD_COUNT sentinel padding on both sides so any out-of-bounds
// read/write by the operator under test corrupts a detectable guard region.
std::vector<aclblasComplex> MakeGuardedVector(int n, int increment, int32_t salt)
{
    const size_t span = CaxpySpanElements(n, increment);
    std::vector<aclblasComplex> data(GUARD_COUNT + span + GUARD_COUNT, GUARD_VALUE);
    for (size_t i = 0; i < span; ++i)
        data[GUARD_COUNT + i] = MakeStridedValue(i, salt);
    return data;
}

} // namespace

// ── Test fixture ──────────────────────────────────────────────────────────────
class CaxpyTest : public BlasTest<CaxpyParam> {};

// ── TEST_F: null handle / null alpha (not expressible via CSV) ───────────────
TEST_F(CaxpyTest, NullHandle)
{
    aclblasComplex alpha{1.0F, 0.0F};
    aclblasStatus_t ret = aclblasCaxpy_npu(nullptr, 1, &alpha, nullptr, 1, nullptr, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(CaxpyTest, NullAlpha)
{
    aclblasComplex x{1.0F, 0.0F};
    aclblasComplex y{1.0F, 0.0F};
    aclblasStatus_t ret = aclblasCaxpy_npu(CaxpyTest::handle_, 1, nullptr, &x, 1, &y, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

namespace {

struct QueuedCaxpyBuffers {
    aclrtStream stream;
    void* x = nullptr;
    void* y = nullptr;
    ~QueuedCaxpyBuffers()
    {
        aclrtSynchronizeStream(stream);
        CaxpyFreeAll(x, y);
    }
};

void CheckQueuedCaxpyCalls(
    aclblasHandle_t handle, aclrtStream stream, int n, int maxStride, const std::vector<std::pair<int, int>>& strides)
{
    auto x = MakeGuardedVector(n, maxStride, 3);
    auto actual = MakeGuardedVector(n, maxStride, -5);
    auto expected = actual;
    const size_t bytes = actual.size() * sizeof(aclblasComplex);
    QueuedCaxpyBuffers buffers{stream};
    ASSERT_EQ(CaxpyAllocCopyH2D(buffers.x, x.data(), bytes), ACL_SUCCESS);
    ASSERT_EQ(CaxpyAllocCopyH2D(buffers.y, actual.data(), bytes), ACL_SUCCESS);
    auto* deviceX = static_cast<aclblasComplex*>(buffers.x) + GUARD_COUNT;
    auto* deviceY = static_cast<aclblasComplex*>(buffers.y) + GUARD_COUNT;
    for (int repeat = 0; repeat < 8; ++repeat) {
        for (const auto& stride : strides) {
            // Change alpha as well as stride to expose reused tiling sources.
            const aclblasComplex alpha{0.125F * (repeat + 1), -0.0625F};
            ASSERT_EQ(
                aclblasCaxpy(handle, n, &alpha, deviceX, stride.first, deviceY, stride.second), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCaxpy_cpu(
                    handle, n, &alpha, x.data() + GUARD_COUNT, stride.first, expected.data() + GUARD_COUNT,
                    stride.second),
                ACLBLAS_STATUS_SUCCESS);
        }
    }
    // All calls above are enqueued without intermediate caller synchronization.
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(actual.data(), bytes, buffers.y, bytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_FLOAT_EQ(actual[i].real, expected[i].real) << "real index=" << i;
        ASSERT_FLOAT_EQ(actual[i].imag, expected[i].imag) << "imag index=" << i;
    }
}

void MeasureCaxpyCase(
    const CaxpyParam& p, aclblasHandle_t handle, aclrtStream stream, const std::vector<aclblasComplex>& xHost,
    std::vector<aclblasComplex>& yHost, const std::vector<aclblasComplex>& initialY)
{
    if (p.measure == 0 || !blas_test::MeasureRequested()) {
        return;
    }
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    std::vector<double> samples;
    const int measureRet = blas_test::MeasureNpuEvent(
        stream, p.warmup, p.measurements,
        [&]() {
            yHost = initialY;
            return true;
        },
        [&]() {
            const aclblasComplex* timedX = xHost.empty() ? nullptr : xHost.data() + GUARD_COUNT;
            aclblasComplex* timedY = yHost.empty() ? nullptr : yHost.data() + GUARD_COUNT;
            return aclblasCaxpy_npu(handle, p.n, &alpha, timedX, p.incx, timedY, p.incy) == ACLBLAS_STATUS_SUCCESS;
        },
        samples);
    ASSERT_EQ(measureRet, 0);
    std::cout << std::setprecision(9) << "PERFORMANCE_SUMMARY_JSON={\"case\":\"" << p.caseName << "\",\"n\":" << p.n
              << ",\"p50_us\":" << blas_test::MedianUs(samples) << ",\"avg_us\":" << blas_test::MeanUs(samples)
              << ",\"warmups\":" << p.warmup << ",\"measurements\":" << p.measurements
              << ",\"timing\":\"npu_event\"}\n";
}

} // namespace

TEST_F(CaxpyTest, QueuedStrideAndAlphaChanges)
{
    CheckQueuedCaxpyCalls(handle_, stream_, 8193, 9, {{1, 1}, {2, 3}, {-3, 2}, {5, -7}, {9, 9}, {1, 1}});
}

TEST_F(CaxpyTest, QueuedPackingFallbacks)
{
    // n=1 exercises increments beyond direct-kernel addressing without a huge
    // allocation: packX only, packY only, both, then a normal dense launch.
    const int large = std::numeric_limits<int>::max();
    CheckQueuedCaxpyCalls(handle_, stream_, 1, 1, {{large, 1}, {1, large}, {-large, large}, {1, 1}});
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Caxpy, CaxpyTest, ::testing::ValuesIn(GetCasesFromCsv<CaxpyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CaxpyParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(CaxpyTest, CsvDriven)
{
    const auto& p = GetParam();
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};

    // Step 1: Generate host data (guarded x/y storage, deterministic values)
    std::vector<aclblasComplex> xHost;
    if (p.nullx == 0 && p.n > 0)
        xHost = MakeGuardedVector(p.n, p.incx, 3);
    std::vector<aclblasComplex> yHost;
    if (p.nully == 0 && p.n > 0)
        yHost = MakeGuardedVector(p.n, p.incy, -5);
    const std::vector<aclblasComplex> initialY = yHost;

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data() + GUARD_COUNT;
    aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data() + GUARD_COUNT;

    // Step 2: Execute on NPU (wrapper handles device memory, nullptr passthrough)
    aclblasStatus_t ret = aclblasCaxpy_npu(CaxpyTest::handle_, p.n, &alpha, xPtr, p.incx, yPtr, p.incy);

    // Step 3: Verify expected return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n == 0)
        return;

    // Step 4: Compute golden on CPU over the same guarded storage, so any
    // stray write into the guard region is caught by the full-buffer compare.
    std::vector<aclblasComplex> xGolden =
        p.nullx == 0 ? MakeGuardedVector(p.n, p.incx, 3) : std::vector<aclblasComplex>{};
    std::vector<aclblasComplex> yGolden =
        p.nully == 0 ? MakeGuardedVector(p.n, p.incy, -5) : std::vector<aclblasComplex>{};
    const aclblasComplex* xGoldenPtr = xGolden.empty() ? nullptr : xGolden.data() + GUARD_COUNT;
    aclblasComplex* yGoldenPtr = yGolden.empty() ? nullptr : yGolden.data() + GUARD_COUNT;
    aclblasStatus_t cpuRet = aclblasCaxpy_cpu(CaxpyTest::handle_, p.n, &alpha, xGoldenPtr, p.incx, yGoldenPtr, p.incy);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification, real/imag components split (MERE/MARE).
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;

    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        yHost.data(), yGolden.data(), yHost.size(), cfg.mereThreshold, cfg.mareMultiplier, 0.0, p.caseName));

    MeasureCaxpyCase(p, handle_, stream_, xHost, yHost, initialY);
}
