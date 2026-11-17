/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "snrm2_test_common.h"

class Snrm2Test : public BlasTest<Snrm2Param> {};

TEST_F(Snrm2Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Snrm2, Snrm2Test, ::testing::ValuesIn(GetCasesFromCsv<Snrm2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Snrm2Param>);

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(Snrm2Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestSnrm2ErrorPath(p, Snrm2Test::handle_);
    } else if (p.n <= 0 || p.incx <= 0) {
        TestSnrm2NoOpPath(p, Snrm2Test::handle_);
    } else {
        TestSnrm2NormalPath(p, Snrm2Test::handle_);
    }
}

// ---------------------------------------------------------------------------
// Coverage migrated from the former standalone main() binary (issue #920):
// the uniform contiguous case and the extreme-value edge cases keep their
// original buffers, goldens and verdicts; only the harness changes.
// ---------------------------------------------------------------------------
TEST_F(Snrm2Test, UniformContiguousTailBoundary)
{
    // 8*2048+1 leaves one element beyond the 8-element tiling boundary for tail coverage.
    constexpr int64_t totalLength = 8 * 2048 + 1;
    constexpr float valueX = 1.2f;
    std::vector<float> xHost(static_cast<size_t>(totalLength), valueX);

    float result = 0.0f;
    ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
    VerifySnrm2NormResult(
        result, std::abs(valueX) * std::sqrt(static_cast<float>(totalLength)), "UniformContiguousTailBoundary");
}

// Shared buffer shape of the migrated edge cases: 8*2048+1 keeps a one-element
// tail beyond the 8-element tiling boundary, and the pre-call buffer is zeroed.
static std::vector<float> MakeEdgeCaseBuffer(size_t totalLength) { return std::vector<float>(totalLength, 0.0f); }

TEST_F(Snrm2Test, EdgeCaseFiniteValues)
{
    constexpr int64_t totalLength = 8 * 2048 + 1;
    float result = 0.0f;

    // ordinary [3,4]
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = 3.0f;
        xHost[1] = 4.0f;
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        VerifySnrm2NormResult(result, 5.0f, "ordinary34");
    }
    // single 1e20 input
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = 1.0e20f;
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        VerifySnrm2NormResult(result, 1.0e20f, "single1e20");
    }
    // single FLT_MAX input
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = std::numeric_limits<float>::max();
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        VerifySnrm2NormResult(result, std::numeric_limits<float>::max(), "singleFltMax");
    }
    // values span multiple cores; the last value occupies the one-element tail
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = 1.0e20f;
        xHost[2048] = 1.0e20f;
        xHost.back() = 1.0e20f;
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        VerifySnrm2NormResult(
            result, std::hypot(std::hypot(1.0e20f, 1.0e20f), 1.0e20f), "multipleLargeAcrossCoresAndTail");
    }
}

TEST_F(Snrm2Test, EdgeCaseNonFinite)
{
    constexpr int64_t totalLength = 8 * 2048 + 1;
    float result = 0.0f;

    // positive infinity
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = std::numeric_limits<float>::infinity();
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(std::isinf(result) && result > 0.0f) << "Expected +Inf, got " << result;
    }
    // NaN
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = std::numeric_limits<float>::quiet_NaN();
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(std::isnan(result)) << "Expected NaN, got " << result;
    }
    // infinity and NaN
    {
        std::vector<float> xHost = MakeEdgeCaseBuffer(static_cast<size_t>(totalLength));
        xHost[0] = std::numeric_limits<float>::infinity();
        xHost.back() = std::numeric_limits<float>::quiet_NaN();
        ASSERT_EQ(aclblasSnrm2_npu(handle_, totalLength, xHost.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(std::isnan(result)) << "Expected NaN, got " << result;
    }
}
