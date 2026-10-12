/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "icamax_golden.h"
#include "icamax_npu_wrapper.h"
#include "icamax_param.h"

namespace {

constexpr uint32_t FIRST_NORMAL_DISTRIBUTION_SEED = 20260619U;

std::string GetIcamaxCsvPath()
{
    std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    const char* useFullCsv = std::getenv("ICAMAX_FULL_CSV");
    if (useFullCsv != nullptr && std::string(useFullCsv) == "1") {
        csvPath.insert(csvPath.find_last_of('.'), "_full");
    }
    return csvPath;
}

bool AccuracyEvidenceEnabled()
{
    const char* value = std::getenv("ICAMAX_ACCURACY_EVIDENCE");
    return value != nullptr && std::string(value) == "1";
}

void EmitAccuracyStatusEvidence(const IcamaxParam& param, aclblasStatus_t status)
{
    if (!AccuracyEvidenceEnabled()) {
        return;
    }
    const bool passed = status == param.expectResult;
    std::cout << "ICAMAX_ACCURACY_RESULT," << param.caseName << ',' << param.n << ',' << param.incx << ','
              << static_cast<int>(param.expectResult) << ',' << static_cast<int>(status)
              << ",,,STATUS," << (passed ? "PASS" : "FAIL") << std::endl;
}

void EmitAccuracyIndexEvidence(
    const IcamaxParam& param, aclblasStatus_t status, int actual, aclblasStatus_t goldenStatus, int expected)
{
    if (!AccuracyEvidenceEnabled()) {
        return;
    }
    const bool passed = status == param.expectResult && goldenStatus == ACLBLAS_STATUS_SUCCESS && actual == expected;
    std::cout << "ICAMAX_ACCURACY_RESULT," << param.caseName << ',' << param.n << ',' << param.incx << ','
              << static_cast<int>(param.expectResult) << ',' << static_cast<int>(status) << ',' << actual << ','
              << expected << ",INDEX," << (passed ? "PASS" : "FAIL") << std::endl;
}

std::vector<aclblasComplex> MakeNormalComplexVector(int64_t length, uint32_t seed)
{
    std::mt19937 parameterRng(seed);
    std::uniform_real_distribution<float> meanDistribution(-5.0F, 5.0F);
    std::uniform_real_distribution<float> sigmaDistribution(0.1F, 2.0F);
    const float realMean = meanDistribution(parameterRng);
    const float realSigma = sigmaDistribution(parameterRng);
    const float imagMean = meanDistribution(parameterRng);
    const float imagSigma = sigmaDistribution(parameterRng);

    std::mt19937 realRng(seed);
    std::mt19937 imagRng(seed + 1000U);
    std::normal_distribution<float> realDistribution(realMean, realSigma);
    std::normal_distribution<float> imagDistribution(imagMean, imagSigma);

    std::vector<aclblasComplex> input(static_cast<size_t>(length));
    for (aclblasComplex& value : input) {
        value = aclblasComplex{realDistribution(realRng), imagDistribution(imagRng)};
    }
    return input;
}

std::vector<aclblasComplex> MakeIcamaxInput(int64_t length, const IcamaxParam& param)
{
    if (length <= 0 || length > std::numeric_limits<int>::max()) {
        return {};
    }
    if (param.x.method == BlasFillMode::M_RANDOM && param.x.pattern == BlasFillMode::P_NORM &&
        param.randomSeed >= FIRST_NORMAL_DISTRIBUTION_SEED) {
        return MakeNormalComplexVector(length, param.randomSeed);
    }
    return makeBlasComplexMatrix(static_cast<int>(length), 1, static_cast<int>(length), param.x, param.randomSeed);
}

} // namespace

class IcamaxTest : public BlasTest<IcamaxParam> {};

TEST_F(IcamaxTest, NullHandle)
{
    int result = -1;
    const aclblasStatus_t status = aclblasIcamax_npu(nullptr, 1, nullptr, 1, &result);
    EXPECT_EQ(status, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(IcamaxTest, NullResult)
{
    const aclblasComplex input{1.0F, -2.0F};
    const aclblasStatus_t status = aclblasIcamax_npu(IcamaxTest::handle_, 1, &input, 1, nullptr);
    EXPECT_EQ(status, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Icamax, IcamaxTest, ::testing::ValuesIn(GetCasesFromCsv<IcamaxParam>(GetIcamaxCsvPath())),
    PrintCaseInfoString<IcamaxParam>);

TEST_P(IcamaxTest, CsvDriven)
{
    const auto& param = GetParam();
    const int64_t xLength = param.n > 0 && param.incx > 0 && param.x.method != BlasFillMode::M_NULLPTR ?
                                static_cast<int64_t>(param.n - 1) * param.incx + 1 :
                                0;

    std::vector<aclblasComplex> input;
    if (xLength > 0) {
        input = MakeIcamaxInput(xLength, param);
    }
    const aclblasComplex* inputPtr = input.empty() ? nullptr : input.data();

    int actual = -1;
    int* actualPtr = param.resultIsNull ? nullptr : &actual;
    const aclblasStatus_t status =
        aclblasIcamax_npu(IcamaxTest::handle_, param.n, inputPtr, param.incx, actualPtr);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(param.expectResult));
    if (status != ACLBLAS_STATUS_SUCCESS || actualPtr == nullptr) {
        EmitAccuracyStatusEvidence(param, status);
        return;
    }

    int expected = -1;
    const aclblasStatus_t goldenStatus =
        aclblasIcamax_cpu(IcamaxTest::handle_, param.n, inputPtr, param.incx, &expected);
    EmitAccuracyIndexEvidence(param, status, actual, goldenStatus, expected);
    ASSERT_EQ(goldenStatus, ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(actual, expected) << "[" << param.caseName << "] index mismatch: NPU=" << actual
                                << " golden=" << expected;
}
