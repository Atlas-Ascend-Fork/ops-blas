/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lt_matmul_test_common.h"

namespace LtMatmul22Test {

inline void CheckLegacyCapsules(aclblasLtHandle_t handle_)
{
    // Bound enumeration to the legacy H0..H4 prefix; appended plans retain their own capsule checks.
    int ids[5]{}, count = 0;
    ASSERT_EQ(
        aclblasLtMatmulAlgoGetIds(
            handle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 5, ids, &count),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(count, 5);
    for (int index = 0; index < count; ++index) {
        aclblasLtMatmulAlgo_t algo{};
        ASSERT_EQ(
            aclblasLtMatmulAlgoInit(
                handle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ids[index], &algo),
            ACLBLAS_STATUS_SUCCESS);
        PackedAlgo packed{};
        CopyBytes(&packed, sizeof(packed), algo.data, sizeof(packed));
        EXPECT_EQ(packed.algoId, 0x22000000U + index);
        EXPECT_EQ(packed.splitK, index == 3 ? 4U : 1U);
        EXPECT_EQ(packed.numBuffers, 1U);
        EXPECT_EQ(packed.flags, 0U);
        EXPECT_EQ(packed.l1mDiv16, index == 3 ? 2U : index == 0 || index == 4 ? 4U : 8U);
        EXPECT_EQ(packed.l1nDiv16, index == 0 || index == 3 ? 4U : 8U);
        uint32_t split = 0;
        size_t written = 0;
        ASSERT_EQ(
            aclblasLtMatmulAlgoConfigGetAttribute(
                &algo, ACLBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split), &written),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(split, index == 3 ? 4U : 1U);
        EXPECT_EQ(written, sizeof(split));
    }
}

TEST_F(RuntimeFixture, H3ConfigAndLegacyCapsules)
{
    ASSERT_NO_FATAL_FAILURE(CheckLegacyCapsules(handle_));
    for (uint32_t split : {1U, 4U, 8U}) {
        auto p = Supplemental("H3ConfigSplit_" + std::to_string(split), 24, 69, 763);
        p.transA = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        if (split != 4U)
            p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        try {
            Invocation operation(p, handle_, stream_);
            // Negative rows use clean query descriptors, so initialize H3 explicitly.
            if (split != 4U) {
                ASSERT_EQ(
                    aclblasLtMatmulAlgoInit(
                        handle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0x22000003,
                        &operation.Algorithm()),
                    ACLBLAS_STATUS_SUCCESS);
            }
            ASSERT_EQ(
                aclblasLtMatmulAlgoConfigSetAttribute(
                    &operation.Algorithm(), ACLBLASLT_ALGO_CONFIG_SPLITK_NUM, &split, sizeof(split)),
                ACLBLAS_STATUS_NOT_SUPPORTED);
            PackedAlgo packed{};
            CopyBytes(&packed, sizeof(packed), operation.Algorithm().data, sizeof(packed));
            EXPECT_EQ(packed.splitK, 4U); // Rejected setters must leave the fixed capsule unchanged.
            packed.splitK = split;
            CopyBytes(operation.Algorithm().data, sizeof(operation.Algorithm().data), &packed, sizeof(packed));
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            EXPECT_TRUE(result.metrics.passed);
        } catch (const std::exception& e) {
            ADD_FAILURE() << e.what();
        }
    }
}

TEST_F(RuntimeFixture, H1TransposedShortKRegression)
{
    for (int combo = 0; combo < 4; ++combo) {
        for (int k : {64, 65, 128, 129}) {
            auto p = Supplemental("H1ShortBoundary_" + std::to_string(combo) + "_" + std::to_string(k), 513, 529, k);
            p.transA = combo & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            p.transB = combo & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            p.algoMode = "heuristic";
            p.workspaceBytes = "auto";
            p.alpha = 0.5f;
            p.beta = -0.5f;
            p.inplace = combo == 3;
            if (p.inplace)
                p.fill[2] = "INPLACE";
            SetPaddedPitches(p);
            CheckSimpleCase(p, handle_, stream_, 0U);
        }
    }
}

} // namespace LtMatmul22Test
