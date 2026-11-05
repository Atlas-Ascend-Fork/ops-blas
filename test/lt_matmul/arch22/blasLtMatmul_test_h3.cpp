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

// Additional private-plan tests; the original 1200 CSV and 71 supplemental tests stay intact.
class H3SplitWorkspaceAndRegressionChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    Param MakeScenario2(int combo, int scenario)
    {
        const int ms[] = {16, 24, 33, 64, 24, 24, 24, 24, 24, 24};
        const int ns[] = {32, 69, 65, 128, 69, 69, 69, 69, 69, 69};
        const int ks[] = {512, 763, 513, 2048, 763, 763, 763, 763, 763, 763};
        auto p = Supplemental(
            "H3Regression_" + std::to_string(combo) + "_" + std::to_string(scenario), ms[scenario], ns[scenario],
            ks[scenario]);
        p.transA = combo & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = combo & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        p.inplace = scenario == 5;
        p.alpha = scenario == 4 || scenario == 5 ? -0.5f :
                  scenario == 6                  ? 0.0f :
                  scenario == 7                  ? 100.0f :
                  scenario == 8                  ? std::numeric_limits<float>::infinity() :
                                                   1.0f;
        p.beta = scenario == 5 ? 0.75f : scenario == 6 ? 1.0f : 0.0f;
        if (p.inplace)
            p.fill[2] = "INPLACE";
        if (scenario == 6)
            p.fill[0] = p.fill[1] = "NULL";
        if (scenario == 9)
            p.fill[0] = p.fill[1] = "NORMAL_RAW_0_1";
        SetPaddedPitches(p);
        return p;
    }

    void CheckScenario2(int combo, int scenario)
    {
        auto p = MakeScenario2(combo, scenario);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            EXPECT_EQ(capacity, size_t(p.m) * ((p.n + 15U) / 16U * 16U) * 16U + 16U);
            PackedAlgo packed{};
            CopyBytes(&packed, sizeof(packed), operation.Algorithm().data, sizeof(packed));
            EXPECT_EQ(packed.algoId, 0x22000003U);
            EXPECT_EQ(packed.splitK, 4U);
            EXPECT_EQ(packed.l1mDiv16, 2U);
            EXPECT_EQ(packed.l1nDiv16, 4U);
            EXPECT_EQ(packed.numBuffers, 1U);
            EXPECT_EQ(packed.flags, 0U);
            DeviceBuffer aligned;
            aligned.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(aligned.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000003U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            aligned.CheckGuards();
            CheckInactiveWorkspace(aligned, base, offset, capacity, "H3 inactive alignment bytes");
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }
    Param MakeScenario1(int kind)
    {
        auto p = Supplemental("H3Budget_" + std::to_string(kind), 24, 69, 763);
        p.transA = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = 24U * 80U * 16U + 16U;
        p.preferenceBytes = kind == 0 ? capacity : kind == 1 ? capacity - 1 : kind == 2 ? 0 : capacity;
        if (kind == 3) {
            p.workspaceBytes = std::to_string(capacity - 1);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        return p;
    }

    void CheckScenario1(int kind)
    {
        auto p = MakeScenario1(kind);
        if (CheckSimpleCase(p, handle_, stream_, kind == 1 ? 0x22000001U : kind == 2 ? 0x22000000U : 0x22000003U))
            ++observations;
    }

public:
    H3SplitWorkspaceAndRegressionChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int combo = 0; combo < 4; ++combo) {
            for (int scenario = 0; scenario < 10; ++scenario) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario2(combo, scenario));
            }
        }
        for (int kind = 0; kind < 4; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(kind));
        }
        std::cout << "H3_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 44);
    }
};

TEST_F(RuntimeFixture, H3SplitWorkspaceAndRegression)
{
    H3SplitWorkspaceAndRegressionChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

} // namespace LtMatmul22Test
