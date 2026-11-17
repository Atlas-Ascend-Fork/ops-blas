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
#include <cstring>
#include <cstdlib>
#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <limits>
#include <random>
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif
#include <vector>
#include "acl/acl.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "csv_case_contract.h"
#include "../csyr2_param.h"
#include "../csyr2_golden.h"
#include "csyr2_npu_wrapper.h"
#include "csyr2_test_resources.h"
#include "triangle_preservation.h"
#include "triangle_fp64_reference.h"

class Csyr2Test : public BlasTest<Csyr2Param> {};

TEST(Csyr2NullHandleTest, NullHandle)
{
    aclblasComplex value{1.0F, 0.0F};
    EXPECT_EQ(
        Csyr2Npu(nullptr, ACLBLAS_UPPER, 1, &value, &value, 1, &value, 1, &value, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

static std::vector<Csyr2Param> AccuracyCases()
{
    auto cases = GetCasesFromCsv<Csyr2Param>(ReplaceFileExtension2Csv(__FILE__));
    std::set<std::string> names;
    VerifyCsvComposition(cases, 1222U, 200U, "csyr2_test.csv", names);
    std::string regressionPath = ReplaceFileExtension2Csv(__FILE__);
    regressionPath.insert(regressionPath.size() - 4U, "_numerics");
    auto regression = GetCasesFromCsv<Csyr2Param>(regressionPath);
    VerifyCsvComposition(regression, 18U, 0U, regressionPath, names);
    cases.insert(cases.end(), regression.begin(), regression.end());
    cases.erase(
        std::remove_if(cases.begin(), cases.end(), [](const Csyr2Param& p) { return p.caseName.find("TC_PF_") == 0; }),
        cases.end());
    VerifyCsvComposition(cases, 1040U, 0U, "CSYR2 accuracy selection");
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Csyr2, Csyr2Test, ::testing::ValuesIn(AccuracyCases()), PrintCaseInfoString<Csyr2Param>);

TEST(CsvCaseContractTest, RejectsCoverageLoss)
{
    struct Case {
        std::string caseName;
    };
    const std::vector<Case> cases{{"TC_EX_001"}, {"TC_PF_001"}};
    EXPECT_NO_THROW(VerifyCsvComposition(cases, 2U, 1U, "valid"));
    EXPECT_THROW(VerifyCsvComposition(cases, 3U, 1U, "truncated"), std::runtime_error);
    EXPECT_THROW(VerifyCsvComposition(cases, 2U, 0U, "PF loss"), std::runtime_error);
    EXPECT_THROW(VerifyCsvComposition(std::vector<Case>{}, 0U, 0U, "empty"), std::runtime_error);
    EXPECT_THROW(VerifyCsvComposition(std::vector<Case>{{" "}}, 1U, 0U, "blank"), std::runtime_error);
    EXPECT_THROW((VerifyCsvComposition(std::vector<Case>{{"a"}, {"a"}}, 2U, 0U, "duplicate")), std::runtime_error);
    std::set<std::string> names;
    VerifyCsvComposition(cases, 2U, 1U, "main", names);
    EXPECT_THROW(VerifyCsvComposition(cases, 2U, 1U, "companion", names), std::runtime_error);
}

TEST(TriangleFp64ReferenceTest, KeepsLowOrderTerms)
{
    const aclblasComplex x[] = {{100000000.0F, 0.0F}, {1.0F, 0.0F}};
    const aclblasComplex y[] = {{1.0F, 0.0F}, {1.0F, 0.0F}};
    const aclblasComplex original[4] = {};
    const auto golden = Csyr2GoldenFp64(ACLBLAS_UPPER, 2, {1.0F, 0.0F}, x, 1, y, 1, original, 2);
    EXPECT_DOUBLE_EQ(golden[2].real, 100000001.0);
    EXPECT_NE(golden[2].real, static_cast<double>(static_cast<float>(golden[2].real)));
    TrianglePrecisionStats stats;
    stats.Add(100000000.0F, golden[2].real);
    EXPECT_DOUBLE_EQ(stats.maxerr, 1.0);
}

TEST(TriangleFp64ReferenceTest, TaskBookTolerance)
{
    TrianglePrecisionStats relative;
    relative.Add(1.00005F, 1.0, 0x1p-13, 0x1p-13);
    EXPECT_EQ(relative.bad, 0U);
    TrianglePrecisionStats nearZero;
    nearZero.Add(0.0003F, 0.0, 0x1p-13, 0x1p-13);
    EXPECT_EQ(nearZero.bad, 1U);
}

TEST(TriangleFp64ReferenceTest, HardLimitUsesMaximumErrorPoint)
{
    for (const bool reverse : {false, true}) {
        TrianglePrecisionStats stats;
        if (reverse) {
            stats.Add(131072.03125F, 131072.0);
        }
        stats.Add(3943.54248046875F, 3943.5314331054688);
        if (!reverse) {
            stats.Add(131072.03125F, 131072.0);
        }
        EXPECT_DOUBLE_EQ(stats.maxerr, 0.03125);
        EXPECT_DOUBLE_EQ(stats.maxerrLimit, 0.5);
        EXPECT_TRUE(stats.Passed());
    }
    TrianglePrecisionStats invalid;
    invalid.Add(3943.75F, 3943.5);
    EXPECT_FALSE(invalid.Passed());
}

TEST(TriangleFp64ReferenceTest, Fp32OverflowClassification)
{
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    TrianglePrecisionStats valid;
    valid.Add(infinity, 1.0e78);
    valid.Add(-infinity, -1.0e78);
    valid.Add(nan, std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(valid.Passed());
    for (const auto actual : {nan, -infinity, std::numeric_limits<float>::max()}) {
        TrianglePrecisionStats invalid;
        invalid.Add(actual, 1.0e78);
        EXPECT_FALSE(invalid.Passed());
    }
    TrianglePrecisionStats boundary;
    const float maximum = std::numeric_limits<float>::max();
    boundary.Add(maximum, static_cast<double>(maximum) + std::ldexp(1.0, 103));
    EXPECT_TRUE(boundary.Passed());
    TrianglePrecisionStats premature;
    premature.Add(infinity, static_cast<double>(std::numeric_limits<float>::max()));
    EXPECT_FALSE(premature.Passed());
}

TEST(TriangleFp64ReferenceTest, CherkCancellation)
{
    const aclblasComplex a[] = {{100000000.0F, 0.0F}, {1.0F, 0.0F},          {1.0F, 0.0F},
                                {1.0F, 0.0F},         {-100000000.0F, 0.0F}, {1.0F, 0.0F}};
    const aclblasComplex original[4] = {};
    const auto golden = CherkGoldenFp64(ACLBLAS_UPPER, ACLBLAS_OP_N, 2, 3, 1.0F, a, 2, 0.0F, original, 2);
    EXPECT_DOUBLE_EQ(golden[2].real, 1.0);
    EXPECT_DOUBLE_EQ(golden[2].imag, 0.0);
}

static size_t VectorSpan(int n, int increment)
{
    return n <= 0 ? 0 :
                    static_cast<size_t>(n - 1) *
                            static_cast<size_t>(increment < 0 ? -static_cast<int64_t>(increment) : increment) +
                        1;
}

struct Csyr2AccuracyData {
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> y;
    std::vector<aclblasComplex> original;
    std::vector<aclblasDoubleComplex> golden;
    std::vector<aclblasComplex> actual;
};
static float AccuracyValue(
    const std::string& mode, size_t i, std::mt19937& rng, std::uniform_real_distribution<float>& uniform)
{
    if (mode == "VALUE_NORM_0") {
        return 0.0F;
    }
    if (mode == "VALUE_NORM_INF") {
        return std::numeric_limits<float>::infinity();
    }
    if (mode == "VALUE_NORM_NAN") {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (mode == "RANDOM_ALTER") {
        return i % 2 ? -1.0F : 1.0F;
    }
    if (mode == "RANDOM_EXTREME" || mode == "SPARSE_INF_EXTREME") {
        return i % 2 ? -1.0e10F : 1.0e-10F;
    }
    return uniform(rng);
}
static void FillAccuracyVector(std::vector<aclblasComplex>& values, const std::string& mode, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uniform(-5.0F, 5.0F);

    const float meanReal = uniform(rng);
    const float meanImag = uniform(rng);
    std::uniform_real_distribution<float> sigma(0.1F, 2.0F);
    std::normal_distribution<float> normalReal(meanReal, sigma(rng));
    std::normal_distribution<float> normalImag(meanImag, sigma(rng));
    if (mode == "RANDOM_NORMAL") {
        for (auto& value : values)
            value = {normalReal(rng), normalImag(rng)};
        return;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = {AccuracyValue(mode, i, rng, uniform), AccuracyValue(mode, i, rng, uniform)};
    }
    if (mode == "SPARSE_INF" || mode == "SPARSE_INF_EXTREME") {
        values[values.size() / 3].real = std::numeric_limits<float>::infinity();
        values.back().imag = -std::numeric_limits<float>::infinity();
    }
    if (mode == "SPARSE_NAN") {
        values[values.size() / 2].imag = std::numeric_limits<float>::quiet_NaN();
    }
}
static void PrepareAccuracy(const Csyr2Param& p, const aclblasComplex& alpha, Csyr2AccuracyData& d)
{
    d.x.resize(VectorSpan(p.n, p.incx));
    auto& x = d.x;
    d.y.resize(VectorSpan(p.n, p.incy));
    auto& y = d.y;
    d.original.resize(static_cast<size_t>(p.lda) * p.n);
    auto& original = d.original;
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = {(static_cast<int>(i * 17U % 101U) - 50) / 37.0F, (static_cast<int>(i * 29U % 97U) - 48) / 41.0F};
    }
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = {(static_cast<int>(i * 23U % 103U) - 51) / 43.0F, (static_cast<int>(i * 31U % 107U) - 53) / 47.0F};
    }
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = {(static_cast<int>(i * 13U % 89U) - 44) / 53.0F, (static_cast<int>(i * 19U % 83U) - 41) / 59.0F};
    }
    if (!p.xFill.empty()) {
        std::mt19937 rng(p.seed);
        FillAccuracyVector(x, p.xFill, rng);
        FillAccuracyVector(y, p.yFill, rng);
        FillAccuracyVector(original, p.aFill, rng);
    }
    PoisonUntouched(original.data(), p.n, p.lda, p.uplo == ACLBLAS_UPPER);
    d.golden = Csyr2GoldenFp64(p.uplo, p.n, alpha, x.data(), p.incx, y.data(), p.incy, original.data(), p.lda);
}
static void RunAccuracy(
    aclblasHandle_t handle, aclrtStream stream, const Csyr2Param& p, const aclblasComplex& alpha, Csyr2AccuracyData& d)
{
    const auto& x = d.x;
    const auto& y = d.y;
    const auto& original = d.original;
    ASSERT_EQ(aclrtSetDevice(blas_test_detail::RuntimeDeviceId()), ACL_SUCCESS);
    Csyr2ReviewBuffers buffers;
    buffers.stream = stream;
    void*& dx = buffers.x;
    void*& dy = buffers.y;
    void*& dA = buffers.a;
    const size_t xb = x.size() * sizeof(aclblasComplex);
    const size_t yb = y.size() * sizeof(aclblasComplex);
    const size_t ab = original.size() * sizeof(aclblasComplex);
    const size_t dxBytes = std::max<size_t>(32, xb);
    const size_t dyBytes = std::max<size_t>(32, yb);
    const size_t dABytes = std::max<size_t>(32, ab);
    ASSERT_EQ(aclrtMalloc(&dx, dxBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dy, dyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dA, dABytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dx, xb, x.data(), xb, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dy, yb, y.data(), yb, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dA, ab, original.data(), ab, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    EXPECT_EQ(
        Csyr2Npu(
            handle, p.uplo, p.n, &alpha, p.xFill == "NULLPTR" ? nullptr : static_cast<const aclblasComplex*>(dx),
            p.incx, p.yFill == "NULLPTR" ? nullptr : static_cast<const aclblasComplex*>(dy), p.incy,
            p.aFill == "NULLPTR" ? nullptr : static_cast<aclblasComplex*>(dA), p.lda),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    d.actual.resize(original.size());
    auto& actual = d.actual;
    ASSERT_EQ(aclrtMemcpy(actual.data(), ab, dA, ab, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
}
TEST_P(Csyr2Test, CpuGolden)
{
    const auto& p = GetParam();
    aclblasComplex alpha{p.ar, p.ai};
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0) {
        // Invalid inputs must return before touching device operands.
        aclblasComplex sentinel{3.0F, -2.0F};
        EXPECT_EQ(
            Csyr2Npu(
                handle_, p.uplo, p.n, p.nullAlpha ? nullptr : &alpha, p.xFill == "NULLPTR" ? nullptr : &sentinel,
                p.incx, p.yFill == "NULLPTR" ? nullptr : &sentinel, p.incy, p.aFill == "NULLPTR" ? nullptr : &sentinel,
                p.lda),
            p.expectResult);
        EXPECT_EQ(sentinel.real, 3.0F);
        EXPECT_EQ(sentinel.imag, -2.0F);
        return;
    }

    Csyr2AccuracyData d;
    ASSERT_NO_FATAL_FAILURE(PrepareAccuracy(p, alpha, d));
    ASSERT_NO_FATAL_FAILURE(RunAccuracy(handle_, stream_, p, alpha, d));
    VerifyTriangleFp64(
        p.caseName.c_str(), d.actual.data(), d.original.data(), d.golden.data(), p.n, p.lda, p.uplo == ACLBLAS_UPPER,
        alpha.real == 0 && alpha.imag == 0);
}

#include "csyr2_review_test.h"
