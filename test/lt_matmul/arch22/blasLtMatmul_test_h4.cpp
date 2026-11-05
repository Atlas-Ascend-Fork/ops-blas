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

class H4ANZWorkspaceAndRegressionChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    Param MakeScenario2(int tb, int scenario)
    {
        const int ms[] = {16, 38, 64, 38, 38, 38, 38, 38};
        const int ns[] = {512, 1372, 1025, 1372, 1372, 1372, 1372, 1372};
        const int ks[] = {128, 209, 511, 209, 209, 209, 209, 209};
        auto p = Supplemental(
            "H4Regression_" + std::to_string(tb) + "_" + std::to_string(scenario), ms[scenario], ns[scenario],
            ks[scenario]);
        p.transA = ACLBLAS_OP_T;
        p.transB = tb ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        p.inplace = scenario == 4;
        p.alpha = scenario == 3 || scenario == 4 ? -0.5f :
                  scenario == 5                  ? 0.0f :
                  scenario == 6                  ? std::numeric_limits<float>::infinity() :
                                                   1.0f;
        p.beta = scenario == 4 ? 0.75f : scenario == 5 ? 1.0f : 0.0f;
        if (p.inplace)
            p.fill[2] = "INPLACE";
        if (scenario == 5)
            p.fill[0] = p.fill[1] = "NULL";
        if (scenario == 7)
            p.fill[0] = p.fill[1] = "NORMAL_RAW_0_1";
        SetPaddedPitches(p);
        return p;
    }

    void CheckScenario2(int tb, int scenario)
    {
        auto p = MakeScenario2(tb, scenario);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = ExpectedH4Capacity(p.m, p.n, p.k);
            EXPECT_EQ(operation.Algorithm().max_workspace_bytes, capacity);
            DeviceBuffer aligned;
            aligned.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(aligned.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000004U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            aligned.CheckGuards();
            if (scenario != 5 && scenario != 6) {
                const size_t productBytes = size_t(p.m) * ((p.n + 15U) / 16U * 16U) * 4U;
                const uintptr_t plane =
                    (reinterpret_cast<uintptr_t>(base + offset) + productBytes + 31U) & ~uintptr_t(31U);
                const size_t height = (p.k + 15U) / 16U * 16U, width = (p.m + 7U) / 8U * 8U;
                std::vector<float> packed(height * width);
                AclCheck(
                    aclrtMemcpy(
                        packed.data(), packed.size() * 4, reinterpret_cast<void*>(plane), packed.size() * 4,
                        ACL_MEMCPY_DEVICE_TO_HOST),
                    "H4 packed A bits");
                const auto input = Inputs(p);
                for (size_t row = 0; row < height; ++row) {
                    for (size_t col = 0; col < width; ++col) {
                        const size_t index = col / 8U * height * 8U + row * 8U + col % 8U;
                        const uint32_t expected = row < size_t(p.k) && col < size_t(p.m) ?
                                                      FloatBits(input[0][row * p.StorageLd(0) + col]) :
                                                      0U;
                        ASSERT_EQ(FloatBits(packed[index]), expected) << "NZ index=" << index;
                    }
                }
            }
            CheckInactiveWorkspace(aligned, base, offset, capacity, "H4 inactive alignment bytes");
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }
    Param MakeScenario1(int kind)
    {
        auto p = Supplemental("H4Budget_" + std::to_string(kind), 38, 1372, 209);
        p.transA = p.transB = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = ExpectedH4Capacity(p.m, p.n, p.k);
        p.preferenceBytes = kind == 0 ? capacity : kind == 1 ? capacity - 1U : kind == 2 ? 0U : capacity;
        if (kind == 3) {
            p.workspaceBytes = std::to_string(capacity - 1U);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        return p;
    }

    void CheckScenario1(int kind)
    {
        auto p = MakeScenario1(kind);
        if (CheckSimpleCase(p, handle_, stream_, kind == 1 ? 0x22000001U : kind == 2 ? 0x22000000U : 0x22000004U))
            ++observations;
    }

public:
    H4ANZWorkspaceAndRegressionChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int tb = 0; tb < 2; ++tb) {
            for (int scenario = 0; scenario < 8; ++scenario) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario2(tb, scenario));
            }
        }
        for (int kind = 0; kind < 4; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(kind));
        }
        std::cout << "H4_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 20);
    }
};

TEST_F(RuntimeFixture, H4ANZWorkspaceAndRegression)
{
    H4ANZWorkspaceAndRegressionChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

} // namespace LtMatmul22Test
