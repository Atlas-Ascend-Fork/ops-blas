/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CHERK has the same public API and test contract on arch22 and arch35. The
// base-file CSV lookup in the shared test selects this directory's cases.
#include "../arch35/cherk_test.cpp"

TEST(CherkVerifierRegression, NonFiniteMismatch)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, nullptr, 1U);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float one = 1.0f;
    EXPECT_FALSE(Verifier::verifyVector(&nan, &one, 1U, 1, cfg, "nan_finite"));
    EXPECT_FALSE(Verifier::verifyVector(&one, &nan, 1U, 1, cfg, "finite_nan"));
    EXPECT_FALSE(Verifier::verifyVector(&inf, &one, 1U, 1, cfg, "inf_finite"));
    const float negativeInf = -inf;
    EXPECT_FALSE(Verifier::verifyVector(&inf, &negativeInf, 1U, 1, cfg, "opposite_inf"));
    EXPECT_TRUE(Verifier::verifyVector(&nan, &nan, 1U, 1, cfg, "matching_nan"));
    EXPECT_TRUE(Verifier::verifyVector(&inf, &inf, 1U, 1, cfg, "matching_inf"));
    EXPECT_FALSE(Verifier::verifyEach([&](auto consume) { consume(nan, one); }, 1U, cfg, "visitor_nan"));
}

static void RunArch22GoldenRegression(aclblasHandle_t handle, const CherkParam& p)
{
    CherkHostData d;
    ASSERT_TRUE(PrepareHostData(p, d, true));

    ASSERT_EQ(
        aclblasCherk_npu(handle, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCherk_cpu(handle, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc),
        ACLBLAS_STATUS_SUCCESS);

    VerifyUploTriangleComplex(p, d.cPtr, d.cGoldenPtr, d.cCount);
}

// Cover panel boundaries and both sides of the outer-product geometry limits.
TEST_F(CherkArch35Test, OuterK2GoldenRegression)
{
    const csv_map row{
        {"case_name", "OuterK2GoldenRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "120"},
        {"k", "2"},
        {"alpha", "1"},
        {"a_fill", "RANDOM_NORM_5_5"},
        {"lda", "120"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "120"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261136"},
    };
    for (int n : {7, 8, 9, 16, 112, 120, 128, 129, 136}) {
        CherkParam p(row);
        p.n = n;
        p.lda = n + 8;
        p.ldc = n + 8;
        p.caseName += "_n" + std::to_string(n);
        RunArch22GoldenRegression(CherkArch35Test::handle_, p);
    }
}

TEST_F(CherkArch35Test, Tiny1AlphaZeroInfRegression)
{
    const csv_map row{
        {"case_name", "Tiny1AlphaZeroInfRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "1"},
        {"k", "1"},
        {"alpha", "0"},
        {"a_fill", "VALUE_NORM_INF"},
        {"lda", "1"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "1"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261201"},
    };
    RunArch22GoldenRegression(CherkArch35Test::handle_, CherkParam(row));
}

TEST_F(CherkArch35Test, OuterK2AlphaZeroInfRegression)
{
    const csv_map row{
        {"case_name", "OuterK2AlphaZeroInfRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "120"},
        {"k", "2"},
        {"alpha", "0"},
        {"a_fill", "VALUE_NORM_INF"},
        {"lda", "120"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "120"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261202"},
    };
    RunArch22GoldenRegression(CherkArch35Test::handle_, CherkParam(row));
}
