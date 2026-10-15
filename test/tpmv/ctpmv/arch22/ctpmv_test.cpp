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
#include <cstdlib>
#include <iostream>
#include <vector>

#include "blas_test.h"
#include "ctpmv_golden.h"
#include "ctpmv_npu_wrapper.h"
#include "ctpmv_param.h"
#include "fill.h"
#include "verify.h"

class CtpmvArch22Test : public BlasTest<CtpmvParam> {};

static std::vector<aclblasComplex> MakeComplexArray(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};
    const std::vector<float> values = makeBlasArray(static_cast<int64_t>(count * 2U), fill, seed);
    std::vector<aclblasComplex> output(count);
    for (size_t i = 0; i < count; ++i)
        output[i] = {values[2U * i], values[2U * i + 1U]};
    return output;
}

TEST_F(CtpmvArch22Test, NullHandle)
{
    EXPECT_EQ(
        aclblasCtpmv_npu(
            nullptr, CtpmvArch22Test::stream_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, nullptr, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ctpmv, CtpmvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CtpmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtpmvParam>);

static void LogCtpmvInput(
    const CtpmvParam& p, const std::vector<aclblasComplex>& ap, const std::vector<aclblasComplex>& x)
{
    if (std::getenv("CTPMV_DEBUG_INPUT") != nullptr &&
        (p.caseName.rfind("TC_EX_", 0U) == 0U || p.caseName.rfind("TC_FL_", 0U) == 0U)) {
        std::cout << "[INPUT] " << p.caseName << " ap=" << ap.size() << " x=" << x.size() << std::endl;
        const size_t apPrint = (p.caseName == "TC_FL_126") ? ap.size() : std::min<size_t>(ap.size(), 24U);
        for (size_t i = 0U; i < apPrint; ++i) {
            std::cout << "[INPUT] ap[" << i << "]=" << ap[i].real << "," << ap[i].imag << std::endl;
        }
        const size_t xPrint = (p.caseName == "TC_FL_126") ? x.size() : std::min<size_t>(x.size(), 24U);
        for (size_t i = 0U; i < xPrint; ++i) {
            std::cout << "[INPUT] x[" << i << "]=" << x[i].real << "," << x[i].imag << std::endl;
        }
    }
}

static void LogCtpmvMismatch(
    const CtpmvParam& p, const std::vector<float>& actualValues, const std::vector<float>& goldenValues)
{
    if (std::getenv("CTPMV_DEBUG_MISMATCH") != nullptr) {
        for (size_t i = 0; i < actualValues.size(); ++i) {
            const float actual = actualValues[i];
            const float expected = goldenValues[i];
            const bool specialMismatch =
                (std::isnan(actual) != std::isnan(expected)) || (std::isinf(actual) != std::isinf(expected)) ||
                (std::isinf(actual) && std::isinf(expected) && std::signbit(actual) != std::signbit(expected));
            const double diff = std::abs(static_cast<double>(actual) - expected);
            if (specialMismatch || diff > 1.0e-2) {
                std::cout << "[" << p.caseName << "] mismatch float=" << i << " logical=" << (i / 2U)
                          << " component=" << (i % 2U) << " actual=" << actual << " golden=" << expected
                          << " diff=" << diff << std::endl;
            }
        }
    }
}

static aclblasStatus_t RunCtpmvCase(
    aclblasHandle_t handle, aclrtStream stream, const CtpmvParam& p, const aclblasComplex* apPtr, aclblasComplex* xPtr,
    bool successPath)
{
    const char* perfIterations = std::getenv("CTPMV_PERF_ITERATIONS");
    aclblasStatus_t status;
    if (perfIterations != nullptr && p.caseName.rfind("TC_PF_", 0U) == 0U && successPath) {
        const int iterations = std::atoi(perfIterations);
        const char* perfWarmup = std::getenv("CTPMV_PERF_WARMUP");
        const int warmup = perfWarmup == nullptr ? 10 : std::atoi(perfWarmup);
        float averageUs = 0.0f;
        status = aclblasCtpmv_npu_benchmark(
            handle, stream, p.uplo, p.trans, p.diag, p.n, apPtr, xPtr, p.incx, warmup, iterations, &averageUs);

        std::cout << "[PERF] case=" << p.caseName << " n=" << p.n << " average_us=" << averageUs << " warmup=" << warmup
                  << " iterations=" << iterations << std::endl;
    } else {
        status = aclblasCtpmv_npu(handle, stream, p.uplo, p.trans, p.diag, p.n, apPtr, xPtr, p.incx);
    }
    return status;
}

TEST_P(CtpmvArch22Test, CsvDriven)
{
    const CtpmvParam& p = GetParam();
    const bool successPath = p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0;
    const size_t apCount =
        successPath ?
            static_cast<size_t>(p.n) * (p.n + 1U) / 2U :
            ((p.n > 0 && p.ap.method != BlasFillMode::M_NULLPTR) ? static_cast<size_t>(p.n) * (p.n + 1U) / 2U : 0U);
    const size_t xCount = successPath ? static_cast<size_t>(p.n - 1) * std::abs(p.incx) + 1U :
                                        ((p.n > 0 && p.incx != 0 && p.x.method != BlasFillMode::M_NULLPTR) ?
                                             static_cast<size_t>(p.n - 1) * std::abs(p.incx) + 1U :
                                             0U);
    std::vector<aclblasComplex> ap = MakeComplexArray(apCount, p.ap, p.randomSeed);
    std::vector<aclblasComplex> x = MakeComplexArray(xCount, p.x, p.randomSeed + 1U);
    std::vector<aclblasComplex> golden = x;
    const aclblasComplex* apPtr = ap.empty() ? nullptr : ap.data();
    aclblasComplex* xPtr = x.empty() ? nullptr : x.data();

    LogCtpmvInput(p, ap, x);

    const auto status = RunCtpmvCase(CtpmvArch22Test::handle_, CtpmvArch22Test::stream_, p, apPtr, xPtr, successPath);
    EXPECT_EQ(status, p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0)
        return;
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(
        aclblasCtpmv_cpu(CtpmvArch22Test::handle_, p.uplo, p.trans, p.diag, p.n, apPtr, golden.data(), p.incx),
        ACLBLAS_STATUS_SUCCESS);
    std::vector<float> actualValues(static_cast<size_t>(p.n) * 2U);
    std::vector<float> goldenValues(static_cast<size_t>(p.n) * 2U);
    const int absIncx = std::abs(p.incx);
    for (int i = 0; i < p.n; ++i) {
        const size_t pos = p.incx > 0 ? static_cast<size_t>(i * absIncx) : static_cast<size_t>((p.n - 1 - i) * absIncx);
        actualValues[2U * i] = x[pos].real;
        actualValues[2U * i + 1U] = x[pos].imag;
        goldenValues[2U * i] = golden[pos].real;
        goldenValues[2U * i + 1U] = golden[pos].imag;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, goldenValues.data(), goldenValues.size());
    LogCtpmvMismatch(p, actualValues, goldenValues);
    EXPECT_TRUE(
        Verifier::verifyVector(actualValues.data(), goldenValues.data(), actualValues.size(), 1, config, p.caseName));
}
