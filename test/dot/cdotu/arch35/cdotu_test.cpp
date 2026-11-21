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

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "cdotu_param.h"
#include "cdotu_golden.h"
#include "cdotu_npu_wrapper.h"

class CdotuArch35Test : public BlasTest<CdotuParam> {};

TEST_F(CdotuArch35Test, NullHandle)
{
    aclblasComplex result{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCdotu_npu(nullptr, 5, nullptr, 1, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// The CONTIGUOUS path (incx == incy == 1) issues 128-bit (float4) loads and thus
// requires a 16B-aligned base pointer. A device buffer offset by one complex (8B)
// is 8B-aligned but not 16B-aligned; such an interior pointer must be rejected with
// INVALID_VALUE rather than relying on unvalidated misaligned 128-bit load behavior.
TEST_F(CdotuArch35Test, MisalignedBase)
{
    const int n = 8;
    void* raw = nullptr;
    ASSERT_EQ(aclrtMalloc(&raw, (n + 1) * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    auto* base = static_cast<aclblasComplex*>(raw);
    aclblasComplex* misX = base + 1;  // 8B-aligned, not 16B-aligned
    aclblasComplex* y = base;

    aclblasStatus_t ret = aclblasCdotu(CdotuArch35Test::handle_, n, misX, 1, y, 1, base);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    // 8B-aligned y (not 16B-aligned) must be rejected identically.
    ret = aclblasCdotu(CdotuArch35Test::handle_, n, y, 1, misX, 1, base);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    aclrtFree(raw);
}

INSTANTIATE_TEST_SUITE_P(
    Cdotu, CdotuArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CdotuParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CdotuParam>);

static bool IsSpecialScalar(float v)
{
    return std::isinf(v) || std::isnan(v);
}

static void VerifyCdotuScalar(float outVal, float goldVal, const std::string& tag)
{
    // Inf/NaN special values (task 3.2 note): judge by consistency - both NaN, or
    // both Inf with the same sign. Finite values fall through to mixed tolerance.
    if (IsSpecialScalar(outVal) || IsSpecialScalar(goldVal)) {
        bool ok = (std::isnan(outVal) && std::isnan(goldVal)) ||
                  (std::isinf(outVal) && std::isinf(goldVal) && ((outVal > 0.0f) == (goldVal > 0.0f)));
        EXPECT_TRUE(ok) << tag << " special-value mismatch: out=" << outVal << " gold=" << goldVal;
        return;
    }
    // COMPLEX64 output: real/imag each verified separately per FLOAT32 standard
    // (task 3.2: |actual-golden| <= atol + rtol*|golden|, max_abs_error <= 1e-2 or 32x ULP).
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldVal);
    EXPECT_TRUE(Verifier::verifyScalar(outVal, goldVal, cfg, tag));
}

static void VerifyCdotuResult(const aclblasComplex& result, const aclblasComplex& golden, const std::string& caseName)
{
    VerifyCdotuScalar(result.real, golden.real, caseName + "_real");
    VerifyCdotuScalar(result.imag, golden.imag, caseName + "_imag");
}

TEST_P(CdotuArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    std::vector<aclblasComplex> xHost = makeBlasComplexStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost = makeBlasComplexStrided(p.n, p.incy, p.y, p.randomSeed);

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    const aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data();
    aclblasComplex result{0.0f, 0.0f};
    aclblasComplex* resultPtr = p.resultIsNull ? nullptr : &result;

    // Performance cases (TC_PF_*) take the warmup + multi-sample path, with the wrapper
    // returning the steady-state kernel time; the remaining precision/edge cases stay
    // single-call to avoid 105 repeated executions slowing the precision regression.
    const bool perfCase = (p.caseName.rfind("TC_PF_", 0) == 0);
    float kernelMs = 0.0f;
    aclblasStatus_t ret = aclblasCdotu_npu(CdotuArch35Test::handle_, p.n, xPtr, p.incx, yPtr, p.incy, resultPtr,
                                           perfCase ? &kernelMs : nullptr);
    if (perfCase) {
        printf("[CDOTU_KERNEL_MS] %s %.6f\n", p.caseName.c_str(), kernelMs);
    }
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    aclblasComplex golden{0.0f, 0.0f};
    bool extremeFill = (p.x.pattern == BlasFillMode::P_EXTREME || p.y.pattern == BlasFillMode::P_EXTREME);
    if (extremeFill) {
        // Extreme values (FLT_MAX/FLT_MIN) overflow float32 products to Inf/NaN; use the
        // float32 sequential reference (cblas semantics) so golden overflows consistently.
        aclblasCdotu_cpu_f32(p.n, xPtr, p.incx, yPtr, p.incy, &golden);
    } else {
        aclblasCdotu_cpu(p.n, xPtr, p.incx, yPtr, p.incy, &golden);
    }

    VerifyCdotuResult(result, golden, p.caseName);
}