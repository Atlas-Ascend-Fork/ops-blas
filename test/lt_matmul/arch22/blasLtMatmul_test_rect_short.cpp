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

class RectShortPublicSemanticsChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    void ConfigureScalars(Param& p, int scenario)
    {
        p.alpha = scenario == 2 || scenario == 3 ? -0.5f :
                  scenario == 4                  ? 0.0f :
                  scenario == 5                  ? std::numeric_limits<float>::infinity() :
                                                   1.0f;
        p.beta = scenario == 3 ? 0.75f : scenario == 4 ? 1.0f : 0.0f;
        p.inplace = scenario == 3;
        if (p.inplace)
            p.fill[2] = "INPLACE";
        if (scenario == 4)
            p.fill[0] = p.fill[1] = "NULL";
        if (scenario == 6)
            p.fill[0] = p.fill[1] = "NORMAL_RAW_0_1";
    }

    Param MakeScenario1(int family, int scenario)
    {
        auto p = Supplemental(
            "RectShort_" + std::to_string(family) + "_" + std::to_string(scenario), family ? 641 : 1025,
            family ? 1027 : 513, family ? 97 : 65);
        if (scenario == 1) {
            p.m = family ? 1024 : 2048;
            p.n = family ? 1536 : 1024;
            p.k = family ? 160 : 96;
        }
        p.transA = p.transB = family ? ACLBLAS_OP_N : ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        ConfigureScalars(p, scenario);
        if (scenario == 7)
            p.m = family ? 670 : 1596;
        SetPaddedPitches(p);
        if (scenario == 8)
            p.beta = std::numeric_limits<float>::quiet_NaN();
        return p;
    }

    void CheckScenario1(int family, int scenario)
    {
        auto p = MakeScenario1(family, scenario);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = size_t(p.m) * ((p.n + 15U) / 16U * 16U) * 4U;
            EXPECT_EQ(operation.Algorithm().max_workspace_bytes, capacity);
            DeviceBuffer aligned;
            aligned.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(aligned.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            ASSERT_EQ(reinterpret_cast<uintptr_t>(base + offset) % 32U, 16U);
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, family ? 0x2200000cU : 0x2200000bU);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            EXPECT_EQ(result.workspaceUsed, scenario == 2 || scenario == 3 ? capacity : 0U);
            aligned.CheckGuards();
            for (auto region :
                 {std::make_pair(size_t(0), offset),
                  std::make_pair(offset + capacity, aligned.Bytes() - offset - capacity)}) {
                std::vector<uint8_t> bytes(region.second);
                if (!bytes.empty()) {
                    AclCheck(
                        aclrtMemcpy(
                            bytes.data(), bytes.size(), base + region.first, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                        "RectShort inactive workspace");
                    EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == GUARD_BYTE; }));
                }
            }
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }

public:
    RectShortPublicSemanticsChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (int family = 0; family < 2; ++family) {
            for (int scenario = 0; scenario < 9; ++scenario) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario1(family, scenario));
            }
        }
        std::cout << "RECT_SHORT_SEMANTIC_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 18);
    }
};

TEST_F(RuntimeFixture, RectShortPublicSemantics)
{
    RectShortPublicSemanticsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class RectShortBudgetCapsulesAndBoundsChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    int observations = 0;

    Param MakeScenario1(int family, int kind)
    {
        auto p = Supplemental(
            "RectShortContract_" + std::to_string(family) + "_" + std::to_string(kind), family ? 641 : 1025,
            family ? 1027 : 513, family ? 134 : 75);
        p.transA = p.transB = family ? ACLBLAS_OP_N : ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = size_t(p.m) * ((p.n + 15U) / 16U * 16U) * 4U;
        p.preferenceBytes = kind == 0 ? 0U : kind == 1 ? capacity - 1U : capacity;
        if (kind == 3) {
            p.workspaceBytes = std::to_string(capacity - 1U);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (kind == 4 || kind == 5)
            p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        if (kind == 6) {
            p.k = family ? 161 : 97;
            p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        return p;
    }

    void CheckScenario1(int family, int kind)
    {
        auto p = MakeScenario1(family, kind);
        try {
            Invocation operation(p, handle_, stream_);
            if (kind == 4 || kind == 5) {
                PackedAlgo capsule{};
                CopyBytes(&capsule, sizeof(capsule), operation.Algorithm().data, sizeof(capsule));
                if (kind == 4)
                    ++capsule.l1mDiv16;
                else
                    capsule.numBuffers = 2;
                CopyBytes(operation.Algorithm().data, sizeof(operation.Algorithm().data), &capsule, sizeof(capsule));
            }
            if (kind == 6) {
                ASSERT_EQ(
                    aclblasLtMatmulAlgoInit(
                        handle_, static_cast<aclblasComputeType_t>(p.computeType), p.scaleType, p.dtype[0], p.dtype[1],
                        p.dtype[2], p.dtype[3], family ? 0x2200000c : 0x2200000b, &operation.Algorithm()),
                    ACLBLAS_STATUS_SUCCESS);
            }
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            EXPECT_TRUE(result.metrics.passed);
            if (kind <= 2)
                EXPECT_EQ(result.algorithm, kind < 2 ? 0x22000000U : family ? 0x2200000cU : 0x2200000bU);
            ++observations;
        } catch (const std::exception& e) {
            ADD_FAILURE() << p.caseName << ": " << e.what();
        }
    }

public:
    RectShortBudgetCapsulesAndBoundsChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int family = 0; family < 2; ++family) {
            for (int kind = 0; kind < 7; ++kind) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario1(family, kind));
            }
        }
        std::cout << "RECT_SHORT_CONTRACT_OBSERVATIONS=" << observations << std::endl;
        EXPECT_EQ(observations, 14);
    }
};

TEST_F(RuntimeFixture, RectShortBudgetCapsulesAndBounds)
{
    RectShortBudgetCapsulesAndBoundsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

} // namespace LtMatmul22Test
