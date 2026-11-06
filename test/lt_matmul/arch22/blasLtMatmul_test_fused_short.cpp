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

class FusedShortPublicSemanticsChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    void ConfigureInitialShapes(Param& p, int scenario)
    {
        if (scenario == 1) {
            p.m = 2048;
            p.n = 1440;
            p.k = 32;
        }
        if (scenario == 2) {
            p.m = 17;
            p.n = 65;
            p.k = 1;
        }
        if (scenario == 3) {
            p.m = 2048;
            p.n = 1153;
            p.k = 17;
        }
        if (scenario == 4) {
            p.m = 16;
            p.n = 16;
            p.k = 16;
        }
    }

    void ConfigureScalarInputs(Param& p, int scenario)
    {
        if (scenario == 5 || scenario == 6)
            p.alpha = -0.5f;
        if (scenario == 6) {
            p.beta = 0.75f;
            p.inplace = true;
        }
        if (scenario == 7 || scenario == 8 || scenario == 9) {
            p.alpha = 0.0f;
            p.beta = scenario == 9 ? 0.5f : 1.0f;
            p.inplace = scenario == 8;
            p.fill[0] = p.fill[1] = "NULLPTR";
            p.fill[2] = p.fill[3] = "BITS_EXACT";
        }
        if (scenario == 10)
            p.alpha = std::numeric_limits<float>::infinity();
        if (scenario == 11)
            p.beta = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 12)
            p.fill[0] = "VALUE_INF";
        if (scenario == 13)
            p.fill[1] = "VALUE_NAN";
        if (scenario == 14)
            p.fill[0] = p.fill[1] = "NORMAL_RAW_0_1";
        if (scenario == 15 || scenario == 16) {
            p.alpha = 0.0f;
            p.beta = scenario == 15 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
            p.fill[0] = p.fill[1] = "NULLPTR";
            p.fill[2] = "BITS_EXACT";
        }
    }

    void ConfigureTailShapes(Param& p, int scenario)
    {
        if (scenario == 18) {
            p.m = 17;
            p.n = 65;
            p.k = 1;
        }
        if (scenario == 19) {
            p.m = 385;
            p.n = 257;
            p.k = 32;
        }
        if (scenario == 20) {
            p.m = 2048;
            p.n = 1153;
            p.k = 17;
        }
        if (scenario == 21)
            p.fill[0] = "VALUE_INF";
        if (scenario == 22)
            p.fill[1] = "VALUE_NAN";
    }

    Param MakeScenario1(int scenario)
    {
        auto p = Supplemental("FusedShortSemantic_" + std::to_string(scenario), 257, 129, 30);
        p.alpha = 1.0f;
        p.beta = 0.0f;
        p.inplace = false;
        p.fill[2] = "RANDOM_NORM_5_5";
        p.transA = p.transB = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        ConfigureInitialShapes(p, scenario);
        ConfigureScalarInputs(p, scenario);
        ConfigureTailShapes(p, scenario);
        p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, p.n + 7, p.n + 11};
        if (scenario >= 17)
            p.ld[1] = p.k;
        if (p.inplace) {
            p.ld[2] = p.ld[3];
            p.fill[2] = "INPLACE";
        }
        return p;
    }

    void CheckScenario1(int scenario)
    {
        auto p = MakeScenario1(scenario);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = ExpectedH10Capacity(p.m, p.n, p.k);
            EXPECT_EQ(operation.Algorithm().max_workspace_bytes, capacity);
            DeviceBuffer aligned;
            aligned.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(aligned.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            auto* workspace = base + offset;
            ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace) % 32U, 16U);
            operation.SetWorkspace(workspace);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x2200000aU);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            EXPECT_EQ(result.metrics.exactMismatches, 0U);
            const bool fused = scenario <= 3 || (scenario >= 12 && scenario <= 14) || scenario >= 17;
            if (scenario == 12 || scenario == 13 || scenario == 21 || scenario == 22)
                EXPECT_GT(result.metrics.nonfinite, 0U);
            EXPECT_EQ(result.workspaceUsed, fused ? capacity - 496U : 0U);
            aligned.CheckGuards();
            const size_t begin = result.workspaceUsed == 0U ?
                                     0U :
                                     ((reinterpret_cast<uintptr_t>(workspace) + 511U) & ~uintptr_t(511U)) -
                                         reinterpret_cast<uintptr_t>(base);
            const size_t end = begin + result.workspaceUsed;
            for (auto region : {std::make_pair(size_t(0), begin), std::make_pair(end, aligned.Bytes() - end)}) {
                std::vector<uint8_t> bytes(region.second);
                if (!bytes.empty()) {
                    AclCheck(
                        aclrtMemcpy(
                            bytes.data(), bytes.size(), base + region.first, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                        "H10 inactive workspace");
                    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == GUARD_BYTE; }));
                }
            }
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }

public:
    FusedShortPublicSemanticsChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (int scenario = 0; scenario < 23; ++scenario) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(scenario));
        }
        std::cout << "FUSED_SHORT_SEMANTIC_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 23);
    }
};

TEST_F(RuntimeFixture, FusedShortPublicSemantics)
{
    FusedShortPublicSemanticsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class FusedShortBudgetCapsulesAndBoundsChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    Param MakeScenario1(int kind)
    {
        auto p = Supplemental("FusedShortContract_" + std::to_string(kind), 257, 129, 30);
        p.alpha = 1.0f;
        p.beta = 0.0f;
        p.inplace = false;
        p.fill[2] = "RANDOM_NORM_5_5";
        p.transA = p.transB = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = ExpectedH10Capacity(p.m, p.n, p.k);
        p.preferenceBytes = kind == 0 ? 0U : kind == 1 ? capacity - 1U : capacity;
        if (kind == 3) {
            p.workspaceBytes = std::to_string(capacity - 1U);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (kind >= 4 && kind <= 12)
            p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        if (kind == 8)
            p.k = 33;
        if (kind == 9)
            p.m = 2049;
        if (kind == 10)
            p.n = 1441;
        if (kind == 11)
            p.transB = ACLBLAS_OP_N;
        if (kind == 12)
            p.ld[1] = 65536;
        if (kind == 13) {
            p.misalignWorkspace = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (kind == 14) {
            // issue #837: 显式 H10（heuristic 序列化）+ 非平凡标量实际走 ordered
            // 路径（无 caller workspace 需求），小于 fused 容量的 workspace
            // 不再被误拒 INVALID_VALUE，数值照常通过
            p.alpha = 2.0f;
            p.beta = 0.5f;
            p.workspaceBytes = "16";
        }
        return p;
    }

    void CheckScenario1(int kind)
    {
        auto p = MakeScenario1(kind);
        try {
            Invocation operation(p, handle_, stream_);
            if (kind >= 4 && kind <= 7) {
                ASSERT_EQ(
                    aclblasLtMatmulAlgoInit(
                        handle_, static_cast<aclblasComputeType_t>(p.computeType), p.scaleType, p.dtype[0], p.dtype[1],
                        p.dtype[2], p.dtype[3], 0x2200000a, &operation.Algorithm()),
                    ACLBLAS_STATUS_SUCCESS);
                PackedAlgo capsule{};
                CopyBytes(&capsule, sizeof(capsule), operation.Algorithm().data, sizeof(capsule));
                if (kind == 4)
                    ++capsule.l1mDiv16;
                if (kind == 5)
                    capsule.numBuffers = 1;
                if (kind == 6)
                    ++capsule.l1nDiv16;
                if (kind == 7)
                    capsule.splitK = 4;
                CopyBytes(operation.Algorithm().data, sizeof(operation.Algorithm().data), &capsule, sizeof(capsule));
            }
            if (kind >= 8 && kind <= 12) {
                ASSERT_EQ(
                    aclblasLtMatmulAlgoInit(
                        handle_, static_cast<aclblasComputeType_t>(p.computeType), p.scaleType, p.dtype[0], p.dtype[1],
                        p.dtype[2], p.dtype[3], 0x2200000a, &operation.Algorithm()),
                    ACLBLAS_STATUS_SUCCESS);
            }
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            EXPECT_TRUE(result.metrics.passed);
            if (kind <= 2)
                EXPECT_EQ(result.algorithm, kind == 0 ? 0x22000000U : kind == 1 ? 0x22000001U : 0x2200000aU);
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }

public:
    FusedShortBudgetCapsulesAndBoundsChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int kind = 0; kind < 15; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(kind));
        }
        std::cout << "FUSED_SHORT_CONTRACT_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 15);
    }
};

TEST_F(RuntimeFixture, FusedShortBudgetCapsulesAndBounds)
{
    FusedShortBudgetCapsulesAndBoundsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

} // namespace LtMatmul22Test
