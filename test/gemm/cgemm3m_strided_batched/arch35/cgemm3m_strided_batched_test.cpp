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
#include <cstring>
#include <iostream>
#include <vector>
#include "blas_test.h"
#include "csv_loader.h"
#include "cgemm3m_strided_batched_golden.h"

namespace cgemm3m_strided_batched_test {
class Cgemm3mSbTest : public BlasTest<Case> {};
constexpr size_t GUARD_ELEMENTS = 16;
constexpr int GUARD_PATTERN = -999;

struct Inputs {
    Matrix a, b, c;
    explicit Inputs(const Invocation& q)
        : a(q.ta == ACLBLAS_OP_N ? q.m : q.k, q.ta == ACLBLAS_OP_N ? q.k : q.m, q.lda, q.sa, q.batch, q.p.S("a_fill"),
            q.p.I("random_seed"), q.p.I("a_offset"), q.p.I("random_seed") % 2 != 0, false),
          b(q.tb == ACLBLAS_OP_N ? q.k : q.n, q.tb == ACLBLAS_OP_N ? q.n : q.k, q.ldb, q.sb, q.batch, q.p.S("b_fill"),
            q.p.I("random_seed") + 1, q.p.I("b_offset"), q.p.I("random_seed") % 2 != 0, false),
          c(q.m, q.n, q.ldc, q.sc, q.batch, q.p.S("c_fill"), q.p.I("random_seed") + 2, q.p.I("c_offset"),
            q.p.I("random_seed") % 2 != 0, false)
    {}
};

void Verify(
    const Invocation& q, const Matrix& c, const std::vector<Complex>& actual, const std::vector<Complex>& expected)
{
    ComponentResult real, imag;
    std::vector<bool> written(actual.size(), false);
    for (int batch = 0; batch < q.batch; ++batch) {
        for (int col = 0; col < q.n; ++col) {
            for (int row = 0; row < q.m; ++row) {
                const int64_t i = c.base + batch * q.sc + int64_t(col) * q.ldc + row;
                real.Add(actual[i].real, expected[i].real);
                imag.Add(actual[i].imag, expected[i].imag);
                written[i] = true;
            }
        }
    }
    for (size_t i = 0; i < written.size(); ++i) {
        if (!written[i]) {
            ASSERT_EQ(std::memcmp(&actual[i], &c.host[i], sizeof(Complex)), 0) << "guard/padding index=" << i;
        }
    }
    std::cout << q.p.caseName << " real_ratio=" << real.Ratio() << " imag_ratio=" << imag.Ratio()
              << " real_max=" << real.maxError << " imag_max=" << imag.maxError
              << " nonfinite=" << real.nonfinite + imag.nonfinite << std::endl;
    EXPECT_TRUE(real.Pass()) << "real excessive=" << real.excessive << " nonfinite=" << real.nonfiniteMatched << "/"
                             << real.nonfinite;
    EXPECT_TRUE(imag.Pass()) << "imag excessive=" << imag.excessive << " nonfinite=" << imag.nonfiniteMatched << "/"
                             << imag.nonfinite;
}

TEST_P(Cgemm3mSbTest, Accuracy)
{
    if (GetParam().caseName.find("TC_PF") == 0) {
        GTEST_SKIP() << "performance case";
    }
    const Invocation q(GetParam(), handle_);
    const auto expectedStatus = GetParam().expectResult;
    if (expectedStatus != ACLBLAS_STATUS_SUCCESS || q.m == 0 || q.n == 0 || q.batch == 0) {
        std::vector<Complex> guard(GUARD_ELEMENTS, {GUARD_PATTERN, GUARD_PATTERN}), after(guard.size());
        DeviceBuffer buffer(guard.size() * sizeof(Complex));
        buffer.copyFromHost(guard.data(), buffer.size());
        auto ptr = reinterpret_cast<Complex*>(buffer.ptr());
        ASSERT_EQ(q.Run(ptr, ptr, ptr), expectedStatus);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        buffer.copyToHost(after.data(), buffer.size());
        EXPECT_EQ(std::memcmp(guard.data(), after.data(), buffer.size()), 0);
        return;
    }
    Inputs data(q);
    std::vector<Complex> golden, actual(data.c.host.size()), repeat(actual.size());
    Golden(q, data.a, data.b, data.c, golden);
    ASSERT_EQ(q.Run(data.a.Ptr(), data.b.Ptr(), data.c.Ptr()), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    data.c.device->copyToHost(actual.data(), data.c.device->size());
    Verify(q, data.c, actual, golden);
    data.c.device->copyFromHost(data.c.host.data(), data.c.device->size());
    ASSERT_EQ(q.Run(data.a.Ptr(), data.b.Ptr(), data.c.Ptr()), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    data.c.device->copyToHost(repeat.data(), data.c.device->size());
    EXPECT_EQ(std::memcmp(actual.data(), repeat.data(), data.c.device->size()), 0) << "determinism";
}

TEST_P(Cgemm3mSbTest, Performance)
{
    if (GetParam().caseName.find("TC_PF") != 0) {
        GTEST_SKIP() << "accuracy case";
    }
    const Invocation q(GetParam(), handle_);
    Inputs data(q);
    constexpr int VALID_SAMPLES = 10;
    for (int sample = 0; sample < VALID_SAMPLES; ++sample) {
        ASSERT_EQ(q.Run(data.a.Ptr(), data.b.Ptr(), data.c.Ptr()), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Cgemm3mStridedBatched, Cgemm3mSbTest,
    ::testing::ValuesIn(GetCasesFromCsv<Case>(ReplaceFileExtension2Csv(__FILE__))), PrintCaseInfoString<Case>);
} // namespace cgemm3m_strided_batched_test
