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

class H1MediumTransposedShortKRegressionChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;

    Param MakeScenario1(int scenario)
    {
        const int ms[] = {188, 188, 188, 129, 256, 188, 188, 188};
        const int ns[] = {1686, 1686, 1686, 2049, 1025, 1686, 1686, 1686};
        const int ks[] = {360, 360, 360, 129, 512, 128, 513, 360};
        auto p = Supplemental("H1MediumShort_" + std::to_string(scenario), ms[scenario], ns[scenario], ks[scenario]);
        p.transA = ACLBLAS_OP_T;
        p.transB = ACLBLAS_OP_N;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        p.alpha = scenario == 1 || scenario == 2 ? -0.5f : scenario == 7 ? 0.0f : 1.0f;
        p.beta = scenario == 1 || scenario == 2 ? 0.75f : scenario == 7 ? 1.0f : 0.0f;
        p.inplace = scenario == 2;
        if (p.inplace)
            p.fill[2] = "INPLACE";
        if (scenario == 7)
            p.fill[0] = p.fill[1] = "NULL";
        SetPaddedPitches(p);
        return p;
    }

    void CheckScenario1(int scenario)
    {
        auto p = MakeScenario1(scenario);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            DeviceBuffer shifted;
            shifted.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(shifted.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000001U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            shifted.CheckGuards();
            CheckInactiveWorkspace(shifted, base, offset, capacity, "H1 medium inactive workspace bytes");
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }

public:
    H1MediumTransposedShortKRegressionChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int scenario = 0; scenario < 8; ++scenario) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(scenario));
        }
    }
};

TEST_F(RuntimeFixture, H1MediumTransposedShortKRegression)
{
    H1MediumTransposedShortKRegressionChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

} // namespace LtMatmul22Test
