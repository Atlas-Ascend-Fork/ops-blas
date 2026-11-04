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
#include <cstdint>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cdgmm_param.h"
#include "cdgmm_golden.h"
#include "cdgmm_npu_wrapper.h"

// ── Helpers for complex (interleaved float) data generation ──────────────────

// Generate `count` complex elements (= 2*count floats) with stride `inc`
// (in complex elements).  Real and imag use the same generator with a seed
// offset so both components vary.  Returns interleaved float vector.
static inline std::vector<float> makeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0)
        return {};

    const int absInc = std::abs(inc);
    const size_t storageEl = static_cast<size_t>((count - 1) * absInc + 1);
    std::vector<float> data(storageEl * 2, 0.0f);

    std::mt19937 rngReal(seed ? seed : 42);
    std::mt19937 rngImag((seed ? seed : 42) + 1000);
    auto genReal = createGenerator(fill, rngReal);
    auto genImag = createGenerator(fill, rngImag);

    for (int i = 0; i < count; i++) {
        int idx = (inc > 0) ? (i * inc) : ((count - 1 - i) * absInc);
        data[static_cast<size_t>(idx) * 2] = genReal->at(i);
        data[static_cast<size_t>(idx) * 2 + 1] = genImag->at(i);
    }
    return data;
}

// Generate a row-major complex matrix: m rows, n cols, row stride lda.
// Returns interleaved float vector of size m * lda * 2.
static inline std::vector<float> makeComplexMatrixRM(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0)
        return {};

    const size_t storageSize = static_cast<size_t>(m) * lda * 2;
    std::vector<float> data(storageSize, 0.0f);

    if (fill.method == BlasFillMode::M_VALUE) {
        for (size_t i = 0; i < storageSize; i++)
            data[i] = fill.val1;
        return data;
    }

    std::mt19937 rngReal(seed ? seed : 42);
    std::mt19937 rngImag((seed ? seed : 42) + 2000);
    auto genReal = createGenerator(fill, rngReal);
    auto genImag = createGenerator(fill, rngImag);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            size_t idx = (static_cast<size_t>(i) * lda + j) * 2;
            data[idx] = genReal->at(i * n + j);
            data[idx + 1] = genImag->at(i * n + j);
        }
    }
    return data;
}

// ── Test fixture ─────────────────────────────────────────────────────────────
class CdgmmArch22Test : public BlasTest<CdgmmParam> {};

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(CdgmmArch22Test, NullHandle)
{
    aclblasStatus_t ret = aclblasCdgmm_npu(nullptr, ACLBLAS_SIDE_LEFT, 4, 4, nullptr, 4, nullptr, 1, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Cdgmm, CdgmmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CdgmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CdgmmParam>);

// issue #780 负例特例：incx = INT32_MIN 的 host 校验短路验证。
// 绕过 npu 包装（strided 设备缓冲按 (m-1)*|incx|+1 分配，该步长下不可达），以最小
// 哑设备缓冲直调公开接口，非空缓冲排除空指针拦截；期望值写死 INVALID_VALUE，
// 保证该校验被移除时用例失败。抽独立函数控制 TEST_P 规模（!830 codecheck）
static void RunCdgmmInt32MinCase(aclblasHandle_t handle, const CdgmmParam& p)
{
    const size_t dummyBytes = 8 * sizeof(aclblasComplex);
    void* dA = nullptr;
    void* dX = nullptr;
    void* dC = nullptr;
    ASSERT_EQ(aclrtMalloc(&dA, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dX, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dC, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    aclblasStatus_t ret = aclblasCdgmm(
        handle, p.mode, p.m, p.n, static_cast<const aclblasComplex*>(dA), p.lda, static_cast<const aclblasComplex*>(dX),
        p.incx, static_cast<aclblasComplex*>(dC), p.ldc);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dC);
}

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(CdgmmArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.incx == INT32_MIN) {
        RunCdgmmInt32MinCase(CdgmmArch22Test::handle_, p);
        return;
    }

    // Step 1: Generate host data (interleaved complex floats)
    // Row-major: x length is m (LEFT only).
    const int xLen = p.m;
    std::vector<float> xHost;
    if (p.nullx == 0 && p.m > 0 && p.n > 0) {
        xHost = makeComplexStrided(xLen, p.incx, p.xFill, p.randomSeed);
    }

    std::vector<float> aHost;
    if (p.nullA == 0 && p.m > 0 && p.n > 0) {
        aHost = makeComplexMatrixRM(p.m, p.n, p.lda, p.aFill, p.randomSeed);
    }

    // C is the output buffer; initialise with sentinel so unmodified padding
    // (if any) matches the golden's untouched region.
    std::vector<float> cHost;
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.m) * 2, kBlasSentinel);
    }

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(xHost.data());
    const aclblasComplex* aPtr = aHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(aHost.data());
    aclblasComplex* cPtr = cHost.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(cHost.data());

    // Step 2: Execute on NPU (wrapper handles nullptr passthrough, device memory)
    aclblasStatus_t ret =
        aclblasCdgmm_npu(CdgmmArch22Test::handle_, p.mode, p.m, p.n, aPtr, p.lda, xPtr, p.incx, cPtr, p.ldc);

    // Step 3: Verify expected return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // m==0 or n==0: operator returns SUCCESS without computing; no output to verify
    if (p.m == 0 || p.n == 0)
        return;

    // Step 4: Compute golden on CPU (row-major LEFT)
    std::vector<float> goldenC(cHost.size(), kBlasSentinel);
    aclblasStatus_t cpuRet = aclblasCdgmm_cpu(
        CdgmmArch22Test::handle_, p.mode, p.m, p.n, aPtr, p.lda, xPtr, p.incx,
        reinterpret_cast<aclblasComplex*>(goldenC.data()), p.ldc);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification — compare entire C storage as floats.
    // C storage = ldc * m * 2 floats (interleaved complex).
    // Padding columns (if ldc > n) are sentinel in both cHost and goldenC.
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;

    EXPECT_TRUE(Verifier::verifyVector(cHost.data(), goldenC.data(), cHost.size(), 1, cfg, p.caseName));
}
