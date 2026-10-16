/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "ctpmv_param.h"
#include "ctpmv_golden.h"
#include "ctpmv_npu_wrapper.h"

// 复数填充助手：沿用 fill.h 的复数策略——实部用 seed、虚部用 seed+1000 两次独立生成后交错，
// 保证实/虚同分布且去相关（与 makeBlasComplexMatrix 的注释约定一致）。
static std::vector<aclblasComplex> MakeComplexPacked(int n, bool upper, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasTriangular(n, upper, fill, seed);
    std::vector<float> imagPart = makeBlasTriangular(n, upper, fill, seed + 1000U);
    const size_t sz = realPart.size();
    std::vector<aclblasComplex> data(sz);
    for (size_t i = 0; i < sz; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

static std::vector<aclblasComplex> MakeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> imagPart = makeBlasStrided(count, inc, fill, seed + 1000U);
    const size_t sz = realPart.size();
    std::vector<aclblasComplex> data(sz);
    for (size_t i = 0; i < sz; i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

using CtpmvArch35Test = BlasTest<CtpmvParam>;

// BLAS stride rules: incx >= 0 -> element i at i*|incx|; incx < 0 -> at (n-1-i)*|incx|
static std::pair<std::vector<aclblasComplex>, std::vector<aclblasComplex>> ExtractStrided(
    const aclblasComplex* xResult, const std::vector<aclblasComplex>& xGolden, int n, int incx)
{
    const int absIncx = std::abs(incx);
    std::pair<std::vector<aclblasComplex>, std::vector<aclblasComplex>> out{
        std::vector<aclblasComplex>(static_cast<size_t>(n)), std::vector<aclblasComplex>(static_cast<size_t>(n))};
    for (int i = 0; i < n; ++i) {
        int offset = (incx >= 0) ? (i * absIncx) : ((n - 1 - i) * absIncx);
        out.first[i] = xResult[offset];
        out.second[i] = xGolden[offset];
    }
    return out;
}

static bool GoldenHasNonFinite(const std::vector<aclblasComplex>& v)
{
    for (const auto& c : v) {
        if (!std::isfinite(c.real) || !std::isfinite(c.imag)) {
            return true;
        }
    }
    return false;
}

static void PrintMismatchSamples(const std::vector<float>& out, const std::vector<float>& gold, const char* part)
{
    int shown = 0;
    for (int i = 0; i < static_cast<int>(gold.size()) && shown < 8; ++i) {
        double g = gold[i];
        double o = out[i];
        if (!(std::abs(o - g) <= 1.5259e-5 + 9.7656e-4 * std::abs(g))) {
            std::cout << "[IDX] " << part << " i=" << i << " out=" << o << " gold=" << g << std::endl;
            ++shown;
        }
    }
}

// 任务书 §3.2 权威口径(生态算子开源精度标准)：实/虚分别按 FLOAT32 混合容差判定
// rtol 2^-10 / atol 2^-16 / ratio>=0.99 / max_abs<=max(1e-2, 32*ULP)
static void VerifyMixed(
    const CtpmvParam& p, const std::vector<aclblasComplex>& xResult, const std::vector<aclblasComplex>& xGolden)
{
    std::vector<float> outReal(static_cast<size_t>(p.n));
    std::vector<float> outImag(static_cast<size_t>(p.n));
    std::vector<float> goldReal(static_cast<size_t>(p.n));
    std::vector<float> goldImag(static_cast<size_t>(p.n));
    for (int i = 0; i < p.n; ++i) {
        outReal[i] = xResult[i].real;
        outImag[i] = xResult[i].imag;
        goldReal[i] = xGolden[i].real;
        goldImag[i] = xGolden[i].imag;
    }
    PrintMismatchSamples(outReal, goldReal, "REAL");
    PrintMismatchSamples(outImag, goldImag, "IMAG");
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldReal.data(), static_cast<size_t>(p.n));
    EXPECT_TRUE(Verifier::verifyVector(
        outReal.data(), goldReal.data(), static_cast<size_t>(p.n), 1, cfg, p.caseName + "_real"));
    applyMixedTolerance(cfg, ACL_FLOAT, goldImag.data(), static_cast<size_t>(p.n));
    EXPECT_TRUE(Verifier::verifyVector(
        outImag.data(), goldImag.data(), static_cast<size_t>(p.n), 1, cfg, p.caseName + "_imag"));
}

// EXACT 通道：复数无现成 exact 策略，按实/虚 both-nan 兼容的逐元素比对
static void VerifyExact(
    const CtpmvParam& p, const std::vector<aclblasComplex>& xResult, const std::vector<aclblasComplex>& xGolden)
{
    bool allEq = true;
    for (int i = 0; i < p.n; ++i) {
        // 非有限值等价：NPU SIMT 对 NaN 输入的乘加载荷(±inf)与 CPU cblas(±nan)不同，
        // 特殊值用例只验"非有限性传播"，不比对载荷（对齐任务 README 特殊值用例调整条款）
        bool reEq = (xResult[i].real == xGolden[i].real) ||
                    (!std::isfinite(xResult[i].real) && !std::isfinite(xGolden[i].real));
        bool imEq = (xResult[i].imag == xGolden[i].imag) ||
                    (!std::isfinite(xResult[i].imag) && !std::isfinite(xGolden[i].imag));
        if (!reEq || !imEq) {
            allEq = false;
            std::cout << "[" << p.caseName << "] mismatch at " << i << ": result=(" << xResult[i].real << ","
                      << xResult[i].imag << ") golden=(" << xGolden[i].real << "," << xGolden[i].imag << ")"
                      << std::endl;
            break;
        }
    }
    EXPECT_TRUE(allEq);
}

// Separate test for null handle (not in CSV)
TEST_F(CtpmvArch35Test, NullHandle)
{
    aclblasStatus_t ret =
        aclblasCtpmv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, nullptr, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ctpmv, CtpmvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CtpmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtpmvParam>);

TEST_P(CtpmvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Generate input data
    // For error-path tests, skip data allocation (use nullptr) to avoid huge memory
    // allocation. The API validates parameters before accessing data, so nullptr is safe.
    const bool isErrorPath = (p.expectResult != ACLBLAS_STATUS_SUCCESS);
    const int safeN = (!isErrorPath && p.n > 0) ? p.n : 0;
    const bool isUpper = (p.uplo == ACLBLAS_UPPER);
    std::vector<aclblasComplex> apHost = MakeComplexPacked(safeN, isUpper, p.ap, p.randomSeed);
    std::vector<aclblasComplex> xHost = MakeComplexStrided(safeN, p.incx, p.x, p.randomSeed + 1);

    const aclblasComplex* apPtr = apHost.empty() ? nullptr : apHost.data();
    aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();

    // For null handle test, use nullptr instead of the fixture's handle
    aclblasHandle_t testHandle =
        (p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : CtpmvArch35Test::handle_;

    // Step 2: Compute CPU golden BEFORE NPU call (NPU overwrites x in-place)
    std::vector<aclblasComplex> xGolden;
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0) {
        xGolden = xHost; // copy x for golden computation
        aclblasCtpmv_cpu(CtpmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, apHost.data(), xGolden.data(), p.incx);
    }

    // Step 3: Execute on NPU (in-place: xPtr is overwritten with result)
    aclblasStatus_t ret = aclblasCtpmv_npu(testHandle, p.uplo, p.trans, p.diag, p.n, apPtr, xPtr, p.incx);

    // Step 4: Check return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 5: Verify results — 实部/虚部分别按 MERE/MARE 统计（复数专用验证通道）
    // Both NPU and golden write results in-place with incx stride.
    auto strided = ExtractStrided(xPtr, xGolden, p.n, p.incx);
    // golden 含非有限值(Inf/NaN 用例)走精确分支(both-nan/both-inf 等价)，有限值走 §3.2 混合容差
    if (p.mereThreshold > 0.0 && !GoldenHasNonFinite(strided.second)) {
        VerifyMixed(p, strided.first, strided.second);
    } else {
        VerifyExact(p, strided.first, strided.second);
    }
}
