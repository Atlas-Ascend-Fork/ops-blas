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
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "sgemv_golden.h"
#include "sgemv_npu_wrapper.h"
#include "sgemv_param.h"
#include "verify.h"

namespace {

constexpr int kPerfWarmup = 5;

bool IsPerfCase(const std::string& name) { return name.rfind("TC_PF_", 0U) == 0U; }

bool IsNullFill(const BlasFillMode& fill) { return fill.method == BlasFillMode::M_NULLPTR; }

std::vector<float> MakeInvalidPlaceholder(const BlasFillMode& fill)
{
    return IsNullFill(fill) ? std::vector<float>{} : std::vector<float>(1U, 1.0F);
}

bool SpecialValuesEquivalent(float actual, float golden)
{
    const bool actualNan = std::isnan(actual);
    const bool goldenNan = std::isnan(golden);
    const bool actualInf = std::isinf(actual);
    const bool goldenInf = std::isinf(golden);
    if (actualNan && goldenNan) {
        return true;
    }
    if ((actualNan && goldenInf) || (actualInf && goldenNan)) {
        return true;
    }
    return actualInf && goldenInf && std::signbit(actual) == std::signbit(golden);
}

bool VerifySgemvOutput(const std::vector<float>& actual, const std::vector<float>& golden, const std::string& name)
{
    std::vector<float> finiteActual;
    std::vector<float> finiteGolden;
    size_t specialCount = 0U;
    size_t specialMismatch = 0U;
    for (size_t i = 0U; i < actual.size(); ++i) {
        const bool actualFinite = std::isfinite(actual[i]);
        const bool goldenFinite = std::isfinite(golden[i]);
        if (actualFinite && goldenFinite) {
            finiteActual.push_back(actual[i]);
            finiteGolden.push_back(golden[i]);
            continue;
        }
        ++specialCount;
        if (!SpecialValuesEquivalent(actual[i], golden[i])) {
            ++specialMismatch;
        }
    }

    std::cout << "[" << name << "] special_count=" << specialCount << " special_mismatch=" << specialMismatch
              << " finite_count=" << finiteActual.size() << std::endl;
    bool finitePass = true;
    if (!finiteActual.empty()) {
        VerifyConfig config;
        applyMixedTolerance(config, ACL_FLOAT, finiteGolden.data(), finiteGolden.size());
        finitePass = Verifier::verifyVector(
            finiteActual.data(), finiteGolden.data(), finiteActual.size(), 1, config, name + "_finite");
    }
    return specialMismatch == 0U && finitePass;
}

std::vector<float> LogicalY(const std::vector<float>& physical, int count, int incy)
{
    std::vector<float> logical;
    if (count <= 0) {
        return logical;
    }
    logical.resize(static_cast<size_t>(count));
    const uint64_t absIncy = sgemv_arch22::AbsStride(incy);
    for (int i = 0; i < count; ++i) {
        const uint64_t index =
            incy > 0 ? static_cast<uint64_t>(i) * absIncy : static_cast<uint64_t>(count - 1 - i) * absIncy;
        logical[static_cast<size_t>(i)] = physical[static_cast<size_t>(index)];
    }
    return logical;
}

struct SgemvHostInputs {
    std::vector<float> a;
    std::vector<float> x;
    std::vector<float> y;
    int xCount = 0;
    int yCount = 0;
};

SgemvHostInputs MakeHostInputs(const SgemvParam& p)
{
    SgemvHostInputs inputs;
    inputs.xCount = p.trans == ACLBLAS_OP_N ? p.n : p.m;
    inputs.yCount = p.trans == ACLBLAS_OP_N ? p.m : p.n;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m < 0 || p.n < 0 || p.incx == INT32_MIN || p.incy == INT32_MIN) {
        inputs.a = MakeInvalidPlaceholder(p.a);
        inputs.x = MakeInvalidPlaceholder(p.x);
        inputs.y = MakeInvalidPlaceholder(p.y);
        return inputs;
    }

    const int matrixLda = std::max(p.lda, 1);
    const int matrixRows = std::max(p.m, 0);
    const int matrixCols = std::max(p.n, 0);
    if (!IsNullFill(p.a)) {
        inputs.a = makeBlasMatrix(matrixRows, matrixCols, matrixLda, p.a, p.randomSeed);
    }
    if (inputs.xCount > 0 && !IsNullFill(p.x)) {
        inputs.x = makeBlasStrided(inputs.xCount, p.incx, p.x, p.randomSeed + 1U);
    }
    if (inputs.yCount > 0 && !IsNullFill(p.y)) {
        inputs.y = makeBlasStrided(inputs.yCount, p.incy, p.y, p.randomSeed + 2U);
    }
    return inputs;
}

bool RunPerfCase(const SgemvParam& p, const SgemvHostInputs& inputs, aclblasHandle_t handle, aclrtStream stream)
{
    int warmup = kPerfWarmup;
    int samples = inputs.a.size() > 1000000U ? 20 : 50;
    if (const char* msprofSamples = std::getenv("ACLBLAS_SGEMV_MSOPROF_SAMPLES"); msprofSamples != nullptr) {
        const int parsedSamples = std::atoi(msprofSamples);
        if (parsedSamples >= 10) {
            warmup = 0;
            samples = parsedSamples;
        }
    }
    const float* alpha = p.alphaFill.method == BlasFillMode::M_NULLPTR ? nullptr : &p.alpha;
    const float* beta = p.betaFill.method == BlasFillMode::M_NULLPTR ? nullptr : &p.beta;
    auto result = sgemv_arch22::aclblasSgemv_npu_perf(
        handle, stream, p.trans, p.m, p.n, alpha, inputs.a.data(), p.lda, inputs.x.data(), p.incx, beta,
        const_cast<float*>(inputs.y.data()), p.incy, warmup, samples);
    std::cout << "[PERF] case=" << p.caseName << " warmup=" << warmup << " samples=" << result.samples
              << " avg_us=" << std::fixed << result.avgUs << std::endl;
    return result.status == ACLBLAS_STATUS_SUCCESS && result.avgUs > 0.0;
}

} // namespace

class SgemvArch22Test : public BlasTest<SgemvParam> {};

namespace {
void VerifyLayoutRegression(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int lda)
{
    const bool noTrans = trans == ACLBLAS_OP_N;
    const int xCount = noTrans ? n : m;
    const int yCount = noTrans ? m : n;
    std::vector<float> a(static_cast<size_t>(lda) * n, -777.0F);
    std::vector<float> x(xCount);
    std::vector<float> expected(yCount, 0.0F);
    std::vector<float> y(yCount, std::numeric_limits<float>::quiet_NaN());
    for (int i = 0; i < xCount; ++i) {
        x[i] = static_cast<float>(i % 5 - 2);
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const float value = static_cast<float>((row * 3 + col * 7) % 17 - 8);
            a[static_cast<size_t>(col) * lda + row] = value;
            expected[noTrans ? row : col] += value * x[noTrans ? col : row];
        }
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    ASSERT_EQ(
        sgemv_arch22::aclblasSgemv_npu(handle, trans, m, n, &alpha, a.data(), lda, x.data(), 1, &beta, y.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(y, expected);
}
} // namespace

TEST_F(SgemvArch22Test, PaddedMatrixFastScalars)
{
    VerifyLayoutRegression(handle_, ACLBLAS_OP_N, 512, 512, 513);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_T, 512, 512, 513);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_C, 513, 257, 521);
}

TEST_F(SgemvArch22Test, NonSymmetricTransposeFastScalars)
{
    VerifyLayoutRegression(handle_, ACLBLAS_OP_T, 1536, 1536, 1536);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_C, 4096, 1024, 4096);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_T, 4097, 257, 4105);
}

TEST_F(SgemvArch22Test, CubeOutputTileCapacity)
{
    VerifyLayoutRegression(handle_, ACLBLAS_OP_N, 4095, 4095, 4095);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_N, 4096, 4096, 4096);
    VerifyLayoutRegression(handle_, ACLBLAS_OP_N, 4097, 4097, 4097);
}

TEST_F(SgemvArch22Test, NullHandle)
{
    float alpha = 1.0F;
    float beta = 0.0F;
    std::vector<float> a(64U, 1.0F);
    std::vector<float> x(8U, 1.0F);
    std::vector<float> y(8U, 0.0F);
    const aclblasStatus_t status = sgemv_arch22::aclblasSgemv_npu(
        nullptr, ACLBLAS_OP_N, 8, 8, &alpha, a.data(), 8, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(status, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SgemvArch22Test, NonAlignedSquareRows)
{
    constexpr int size = 17;
    float alpha = 1.0F;
    float beta = 0.0F;
    std::vector<float> a(size * size, 1.0F);
    std::vector<float> x(size, 1.0F);
    std::vector<float> y(size, 0.0F);
    const aclblasStatus_t status = sgemv_arch22::aclblasSgemv_npu(
        SgemvArch22Test::handle_, ACLBLAS_OP_N, size, size, &alpha, a.data(), size, x.data(), 1, &beta, y.data(), 1);
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    for (float value : y) {
        EXPECT_NEAR(value, static_cast<float>(size), 1.0e-4F);
    }
}

TEST_F(SgemvArch22Test, TransposeMultipleBlocks)
{
    constexpr int rows = 4;
    constexpr int cols = 256;
    float alpha = 1.0F;
    float beta = 0.0F;
    std::vector<float> a(rows * cols, 1.0F);
    std::vector<float> x(rows, 1.0F);
    std::vector<float> y(cols, 0.0F);
    const aclblasStatus_t status = sgemv_arch22::aclblasSgemv_npu(
        SgemvArch22Test::handle_, ACLBLAS_OP_T, rows, cols, &alpha, a.data(), rows, x.data(), 1, &beta, y.data(), 1);
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < y.size(); ++i) {
        if (std::abs(y[i] - static_cast<float>(rows)) > 1.0e-4F) {
            ADD_FAILURE() << "index=" << i << " value=" << y[i];
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sgemv22, SgemvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SgemvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgemvParam>);

TEST_P(SgemvArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    SgemvHostInputs inputs = MakeHostInputs(p);
    const float* alpha = p.alphaFill.method == BlasFillMode::M_NULLPTR ? nullptr : &p.alpha;
    const float* beta = p.betaFill.method == BlasFillMode::M_NULLPTR ? nullptr : &p.beta;

    if (IsPerfCase(p.caseName)) {
        EXPECT_TRUE(RunPerfCase(p, inputs, SgemvArch22Test::handle_, SgemvArch22Test::stream_));
        return;
    }

    std::vector<float> yNpu = inputs.y;
    const aclblasStatus_t status = sgemv_arch22::aclblasSgemv_npu(
        SgemvArch22Test::handle_, p.trans, p.m, p.n, alpha, inputs.a.empty() ? nullptr : inputs.a.data(), p.lda,
        inputs.x.empty() ? nullptr : inputs.x.data(), p.incx, beta, inputs.y.empty() ? nullptr : yNpu.data(), p.incy);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(status, p.expectResult) << p.caseName;
        return;
    }
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS) << p.caseName;

    std::vector<float> yGolden = inputs.y;
    ASSERT_EQ(
        aclblasSgemv_cpu(
            SgemvArch22Test::handle_, p.trans, p.m, p.n, alpha, inputs.a.empty() ? nullptr : inputs.a.data(), p.lda,
            inputs.x.empty() ? nullptr : inputs.x.data(), p.incx, beta, yGolden.empty() ? nullptr : yGolden.data(),
            p.incy),
        ACLBLAS_STATUS_SUCCESS)
        << p.caseName;
    if (inputs.yCount <= 0) {
        EXPECT_EQ(yNpu, yGolden) << p.caseName;
        return;
    }

    const std::vector<float> npuLogical = LogicalY(yNpu, inputs.yCount, p.incy);
    const std::vector<float> goldenLogical = LogicalY(yGolden, inputs.yCount, p.incy);
    EXPECT_TRUE(VerifySgemvOutput(npuLogical, goldenLogical, p.caseName));
}
