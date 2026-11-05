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

class CsvFixture : public RuntimeFixture, public ::testing::WithParamInterface<Param> {};

TEST_P(CsvFixture, Validate)
{
    const auto& p = GetParam();
    SCOPED_TRACE(p.caseName + ": " + p.description);
    try {
        Invocation operation(p, handle_, stream_);
        const char* requested = std::getenv("LT_MATMUL_BENCHMARK");
        const bool benchmark = requested != nullptr && std::string(requested) == "1" && p.caseName.find("TC_PF_") == 0;
        const auto result = operation.Validate(benchmark);
        Record(p, result);
        RecordProperty("case_id", p.caseName);
        RecordProperty("algorithm_id", std::to_string(result.algorithm));
        RecordProperty("workspace_allocated_bytes", std::to_string(result.workspaceAllocated));
        RecordProperty("workspace_used_bytes", std::to_string(result.workspaceUsed));
        RecordProperty("matched_ratio", std::to_string(result.metrics.matchedRatio));
        ASSERT_EQ(result.status, p.expectResult);
        ASSERT_TRUE(result.metrics.passed)
            << "matched_ratio=" << result.metrics.matchedRatio << " max_abs=" << result.metrics.maxAbs
            << " limit=" << result.metrics.maxAbsLimit << " worst_storage_index=" << result.metrics.worstIndex
            << " nonfinite=" << result.metrics.nonfinite << " nonfinite_matched=" << result.metrics.nonfiniteMatched
            << " exact_mismatches=" << result.metrics.exactMismatches;
        if (benchmark)
            ASSERT_EQ(result.sampledCalls, 60U);
    } catch (const std::exception& error) {
        FAIL() << error.what();
    }
}

INSTANTIATE_TEST_SUITE_P(TaskCases, CsvFixture, ::testing::ValuesIn(TaskCases(__FILE__)), PrintCaseInfoString<Param>);
INSTANTIATE_TEST_SUITE_P(
    SupplementalCases, CsvFixture, ::testing::ValuesIn(SupplementalCases()), PrintCaseInfoString<Param>);
INSTANTIATE_TEST_SUITE_P(GapCases, CsvFixture, ::testing::ValuesIn(GapCases(__FILE__)), PrintCaseInfoString<Param>);

TEST_F(RuntimeFixture, ConcurrentStreams)
{
    aclrtStream second = nullptr;
    ASSERT_EQ(aclrtCreateStream(&second), ACL_SUCCESS);
    try {
        auto p = Supplemental("ConcurrentStreamA", 65, 33, 513);
        auto q = Supplemental("ConcurrentStreamB", 129, 17, 65);
        q.transA = ACLBLAS_OP_T;
        q.transB = ACLBLAS_OP_T;
        q.randomSeed += 100;
        Invocation first(p, handle_, stream_), other(q, handle_, second);
        const auto aStatus = first.Launch(), bStatus = other.Launch();
        ASSERT_EQ(aStatus, ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(bStatus, ACLBLAS_STATUS_SUCCESS);
        const auto a = first.Validate(false, true), b = other.Validate(false, true);
        Record(p, a);
        Record(q, b);
        EXPECT_TRUE(a.metrics.passed);
        EXPECT_TRUE(b.metrics.passed);

        auto fusedFirstCase = Supplemental("ConcurrentFusedShortA", 128, 1440, 32);
        fusedFirstCase.transA = fusedFirstCase.transB = ACLBLAS_OP_T;
        fusedFirstCase.alpha = 1.0f;
        fusedFirstCase.beta = 0.0f;
        fusedFirstCase.inplace = false;
        fusedFirstCase.fill[2] = "RANDOM_NORM_5_5";
        fusedFirstCase.algoMode = "heuristic";
        fusedFirstCase.workspaceBytes = "auto";
        SetPaddedPitches(fusedFirstCase);
        auto fusedSecondCase = fusedFirstCase;
        fusedSecondCase.caseName = "ConcurrentFusedShortB";
        fusedSecondCase.randomSeed += 100;
        Invocation fusedA(fusedFirstCase, handle_, stream_), fusedB(fusedSecondCase, handle_, second);
        ASSERT_EQ(fusedA.Launch(), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(fusedB.Launch(), ACLBLAS_STATUS_SUCCESS);
        const auto fusedResultA = fusedA.Validate(false, true), fusedResultB = fusedB.Validate(false, true);
        Record(fusedFirstCase, fusedResultA);
        Record(fusedSecondCase, fusedResultB);
        EXPECT_EQ(fusedResultA.algorithm, 0x2200000aU);
        EXPECT_EQ(fusedResultB.algorithm, 0x2200000aU);
        EXPECT_TRUE(fusedResultA.metrics.passed);
        EXPECT_TRUE(fusedResultB.metrics.passed);
    } catch (const std::exception& error) {
        ADD_FAILURE() << error.what();
    }
    EXPECT_EQ(aclrtDestroyStream(second), ACL_SUCCESS);
}

TEST_F(RuntimeFixture, ActiveAndIgnoredAliases)
{
    for (int kind = 0; kind < 6; ++kind) {
        auto p = Supplemental("Alias_" + std::to_string(kind), 16, 16, 16);
        p.inplace = false;
        p.fill[2] = "RANDOM_NORM_5_5";
        if (kind < 3)
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        if (kind == 2) {
            p.algoMode = "heuristic";
            p.workspaceBytes = "auto";
        }
        if (kind == 3 || kind == 5)
            p.alpha = 0.0f;
        if (kind == 4 || kind == 5)
            p.beta = 0.0f;
        try {
            Invocation operation(p, handle_, stream_);
            if (kind == 0)
                operation.SetOperand(3, operation.Operand(0));
            if (kind == 1 || kind == 4 || kind == 5)
                operation.SetOperand(2, static_cast<uint8_t*>(operation.Operand(3)) + sizeof(float));
            if (kind == 2)
                operation.SetWorkspace(operation.Operand(3));
            if (kind == 3 || kind == 5) {
                operation.SetOperand(0, operation.Operand(3));
                operation.SetOperand(1, operation.Operand(3));
            }
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            EXPECT_TRUE(result.metrics.passed);
        } catch (const std::exception& error) {
            ADD_FAILURE() << "alias " << kind << ": " << error.what();
        }
    }
}

TEST_F(RuntimeFixture, MalformedAndUnsupportedAlgorithmCapsules)
{
    for (bool badMagic : {true, false}) {
        auto p = Supplemental(badMagic ? "AlgorithmBadMagic" : "AlgorithmUnsupportedPlan", 16, 16, 16);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        p.expectResult = badMagic ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_NOT_SUPPORTED;
        try {
            Invocation operation(p, handle_, stream_);
            PackedAlgo packed{};
            CopyBytes(&packed, sizeof(packed), operation.Algorithm().data, sizeof(packed));
            if (badMagic)
                packed.magic = 0;
            else
                packed.splitK = 2;
            CopyBytes(operation.Algorithm().data, sizeof(operation.Algorithm().data), &packed, sizeof(packed));
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
}

TEST_F(RuntimeFixture, UnsupportedAndInvalidEpilogues)
{
    for (int32_t epilogue : {int32_t(ACLBLASLT_EPILOGUE_RELU), int32_t(999)}) {
        auto p = Supplemental("Epilogue_" + std::to_string(epilogue), 16, 16, 16);
        p.expectResult = epilogue == 999 ? ACLBLAS_STATUS_INVALID_ENUM : ACLBLAS_STATUS_NOT_SUPPORTED;
        try {
            Invocation operation(p, handle_, stream_);
            operation.SetEpilogue(epilogue);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
}

class HeuristicPaddedPitchLimitsChecks {
    static constexpr int maxPitch = std::numeric_limits<int32_t>::max();
    aclblasLtHandle_t handle_;
    aclrtStream stream_;

    void CheckScenario1(int n)
    {
        auto p = Supplemental("HeuristicPitch_n" + std::to_string(n), 1, n, 1);
        p.inplace = false;
        p.fill[2] = "RANDOM_NORM_5_5";
        p.ld = {1, n, n, n};
        // The public preference setter accepts at most INT64_MAX. This legal
        // budget exceeds both large candidates' ~8 GiB product capacity, so an
        // H0-only result cannot be explained by insufficient workspace budget.
        p.preferenceBytes = static_cast<size_t>(std::numeric_limits<int64_t>::max());
        try {
            RecordParameterMetadata(p, "descriptor_only_heuristic");
            // Descriptor-only test: no matrix, workspace or golden allocation.
            Descriptors descriptors;
            descriptors.Initialize(p);
            size_t budgetReadback = 0, bytesWritten = 0;
            ASSERT_EQ(
                aclblasLtMatmulPreferenceGetAttribute(
                    descriptors.preference, ACLBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budgetReadback,
                    sizeof(budgetReadback), &bytesWritten),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(bytesWritten, sizeof(budgetReadback));
            ASSERT_EQ(budgetReadback, p.preferenceBytes);
            aclblasLtMatmulHeuristicResult_t algorithms[2]{};
            int count = 0;
            ASSERT_EQ(
                aclblasLtMatmulAlgoGetHeuristic(
                    handle_, descriptors.compute, descriptors.layout[0], descriptors.layout[1], descriptors.layout[2],
                    descriptors.layout[3], descriptors.preference, 2, algorithms, &count),
                ACLBLAS_STATUS_SUCCESS);
            const bool representableH1 = n == maxPitch - 15;
            ASSERT_EQ(count, representableH1 ? 2 : 1);
            PackedAlgo packed{};
            CopyBytes(&packed, sizeof(packed), algorithms[0].algo.data, sizeof(packed));
            const size_t required = representableH1 ? static_cast<size_t>(n) * sizeof(float) : 0;
            EXPECT_EQ(packed.algoId, representableH1 ? 0x22000001U : 0x22000000U);
            EXPECT_EQ(algorithms[0].workspaceSize, required);
            EXPECT_EQ(algorithms[0].algo.max_workspace_bytes, required);
            if (representableH1) {
                CopyBytes(&packed, sizeof(packed), algorithms[1].algo.data, sizeof(packed));
                EXPECT_EQ(packed.algoId, 0x22000000U);
                EXPECT_EQ(algorithms[1].workspaceSize, 0U);
            }
            std::cout << "LT_MATMUL_HEURISTIC_RESULT case=" << p.caseName
                      << " first_algorithm_id=" << (representableH1 ? 0x22000001U : 0x22000000U)
                      << " workspace_bytes=" << required << " budget_bytes=" << budgetReadback
                      << " matrix_allocations=0" << std::endl;
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }

public:
    HeuristicPaddedPitchLimitsChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (int n : {15, maxPitch - 15, maxPitch}) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(n));
        }
    }
};

TEST_F(RuntimeFixture, HeuristicPaddedPitchLimits)
{
    HeuristicPaddedPitchLimitsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class H2WorkspaceAndFallbacksChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    static constexpr int nnShapes[][3] = {{969, 55, 899}, {513, 63, 513},  {1024, 64, 1024}, {511, 55, 899},
                                          {969, 65, 899}, {969, 55, 1025}, {969, 55, 512}};
    static constexpr int smallNNShapes[][3] = {{18, 868, 1341},  {16, 512, 1025}, {32, 1024, 2048},
                                               {15, 868, 1341},  {33, 868, 1341}, {18, 511, 1341},
                                               {18, 1025, 1341}, {18, 868, 1024}, {18, 868, 2049}};

    Param MakeScenario10(int combo, bool staged)
    {
        auto p = Supplemental("H2Layout_" + std::to_string(combo) + "_" + std::to_string(staged), 65, 33, 65);
        p.algoMode = "explicit_h2";
        p.workspaceBytes = "auto";
        p.transA = combo & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = combo & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, 40, 40};
        if (!staged) {
            SetDirectOutput(p);
        }
        return p;
    }

    void CheckScenario10(int combo, bool staged)
    {
        auto p = MakeScenario10(combo, staged);
        SCOPED_TRACE(p.caseName);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            DeviceBuffer aligned;
            aligned.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(aligned.Data());
            const size_t offset = (48U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            ASSERT_EQ((reinterpret_cast<uintptr_t>(base + offset) & 31U), 16U);
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000002U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            aligned.CheckGuards();
            CheckInactiveWorkspace(aligned, base, offset, capacity, "H2 inactive alignment bytes");
        } catch (const std::exception& e) {
            ADD_FAILURE() << e.what();
        }
    }
    Param MakeScenario9(int kind)
    {
        auto p = Supplemental("H2Contract_" + std::to_string(kind), 65, 33, 65);
        p.algoMode = "explicit_h2";
        p.workspaceBytes = "auto";
        if (kind == 0) {
            p.workspaceBytes = std::to_string(ExpectedH2Capacity(p.m, p.n, p.k, false, false) - 1U);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        } else if (kind == 1) {
            p.misalignWorkspace = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        } else if (kind == 2) {
            p.alpha = 0.0f; // Exact scaling must not consume NZ inputs.
            p.fill[0] = p.fill[1] = "NULL";
        } else {
            p.alpha = std::numeric_limits<float>::infinity(); // Ordered numerical fallback.
        }
        return p;
    }

    void CheckScenario9(int kind)
    {
        auto p = MakeScenario9(kind);
        CheckSimpleCase(p, handle_, stream_, 0U);
    }
    Param MakeScenario8(int guardK, int combo, size_t mod512, bool staged)
    {
        auto p = Supplemental(
            "H2Plane512_" + std::to_string(guardK) + "_" + std::to_string(combo) + "_" + std::to_string(mod512) + "_" +
                std::to_string(staged),
            113, 233, guardK);
        p.algoMode = "explicit_h2";
        p.workspaceBytes = "auto";
        p.transA = combo & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = combo & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, p.n + 4, p.n + 4};
        if (!staged) {
            SetDirectOutput(p);
        }
        return p;
    }

    void CheckScenario8(int guardK, int combo, size_t mod512, bool staged)
    {
        auto p = MakeScenario8(guardK, combo, mod512, staged);
        SCOPED_TRACE(p.caseName);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            DeviceBuffer guarded;
            guarded.Initialize(capacity + 512U);
            auto* base = static_cast<uint8_t*>(guarded.Data());
            const size_t offset = (mod512 + 512U - reinterpret_cast<uintptr_t>(base) % 512U) % 512U;
            auto* workspace = base + offset;
            ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace) % 512U, mod512);
            operation.SetWorkspace(workspace);
            std::cout << "LT_MATMUL_ALIGNMENT case=" << p.caseName << " workspace_mod512=" << mod512
                      << " passed_capacity=" << capacity << std::endl;
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000002U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            guarded.CheckGuards();
            CheckInactiveWorkspace(guarded, base, offset, capacity, "H2 exact aligned capacity guard");
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
    Param MakeScenario7(int shape)
    {
        auto p = Supplemental(
            "H2NNHeuristic_" + std::to_string(shape), nnShapes[shape][0], nnShapes[shape][1], nnShapes[shape][2]);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        SetDirectOutput(p);
        return p;
    }

    void CheckScenario7(int shape)
    {
        const int combos = shape == 0 ? 4 : 1;
        for (int combo = 0; combo < combos; ++combo) {
            for (size_t mod512 : {size_t(0), size_t(16)}) {
                for (bool staged : {false, true}) {
                    ASSERT_NO_FATAL_FAILURE(CheckScenario6(shape, combo, mod512, staged));
                }
            }
        }
        auto p = MakeScenario7(shape);
        CheckSimpleCase(p, handle_, stream_, shape < 3 ? 0x22000002U : 0x22000001U);
    }
    Param MakeScenario6(int shape, int combo, size_t mod512, bool staged)
    {
        auto p = Supplemental(
            "H2NNGuard_" + std::to_string(shape) + "_" + std::to_string(combo) + "_" + std::to_string(mod512) + "_" +
                std::to_string(staged),
            nnShapes[shape][0], nnShapes[shape][1], nnShapes[shape][2]);
        ConfigurePackedGuard(p, combo, staged, "explicit_h2");
        return p;
    }

    void CheckScenario6(int shape, int combo, size_t mod512, bool staged)
    {
        auto p = MakeScenario6(shape, combo, mod512, staged);
        CheckNZShiftedGuard(p, handle_, stream_, combo, mod512, "H2NN exact capacity inactive bytes");
    }
    Param MakeScenario5(int budget)
    {
        auto p = Supplemental("H2NNBudget_" + std::to_string(budget), 969, 55, 899);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = ExpectedH2Capacity(p.m, p.n, p.k, false, false);
        p.preferenceBytes = budget == 0 ? capacity : budget == 1 ? capacity - 1U : 0U;
        return p;
    }

    void CheckScenario5(int budget)
    {
        auto p = MakeScenario5(budget);
        CheckSimpleCase(p, handle_, stream_, budget == 0 ? 0x22000002U : budget == 1 ? 0x22000001U : 0x22000000U);
    }
    Param MakeScenario4(int shape)
    {
        auto p = Supplemental(
            "H2SmallNNHeuristic_" + std::to_string(shape), smallNNShapes[shape][0], smallNNShapes[shape][1],
            smallNNShapes[shape][2]);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        SetDirectOutput(p);
        return p;
    }

    void CheckScenario4(int shape)
    {
        const int combos = shape == 0 ? 4 : 1;
        for (int combo = 0; combo < combos; ++combo) {
            for (size_t mod512 : {size_t(0), size_t(16)}) {
                for (bool staged : {false, true}) {
                    ASSERT_NO_FATAL_FAILURE(CheckScenario3(shape, combo, mod512, staged));
                }
            }
        }
        auto p = MakeScenario4(shape);
        CheckSimpleCase(p, handle_, stream_, shape < 3 ? 0x22000002U : 0x22000001U);
    }
    Param MakeScenario3(int shape, int combo, size_t mod512, bool staged)
    {
        auto p = Supplemental(
            "H2SmallNNGuard_" + std::to_string(shape) + "_" + std::to_string(combo) + "_" + std::to_string(mod512) +
                "_" + std::to_string(staged),
            smallNNShapes[shape][0], smallNNShapes[shape][1], smallNNShapes[shape][2]);
        ConfigurePackedGuard(p, combo, staged, "explicit_h2");
        return p;
    }

    void CheckScenario3(int shape, int combo, size_t mod512, bool staged)
    {
        auto p = MakeScenario3(shape, combo, mod512, staged);
        CheckNZShiftedGuard(p, handle_, stream_, combo, mod512, "H2SmallNN exact capacity inactive bytes");
    }
    Param MakeScenario2(int budget)
    {
        auto p = Supplemental("H2SmallNNBudget_" + std::to_string(budget), 18, 868, 1341);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t capacity = ExpectedH2Capacity(p.m, p.n, p.k, false, false);
        p.preferenceBytes = budget == 0 ? capacity : budget == 1 ? capacity - 1U : 0U;
        return p;
    }

    void CheckScenario2(int budget)
    {
        auto p = MakeScenario2(budget);
        CheckSimpleCase(p, handle_, stream_, budget == 0 ? 0x22000002U : budget == 1 ? 0x22000001U : 0x22000000U);
    }
    Param MakeScenario1(int budget)
    {
        auto p = Supplemental("H2Budget_" + std::to_string(budget), 109, 128, 513);
        p.transA = ACLBLAS_OP_T;
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t required = ExpectedH2Capacity(p.m, p.n, p.k, true, false);
        p.preferenceBytes = budget == 0 ? required : budget == 1 ? required - 1U : 0U;
        return p;
    }

    void CheckScenario1(int budget)
    {
        auto p = MakeScenario1(budget);
        CheckSimpleCase(p, handle_, stream_, budget == 0 ? 0x22000002U : budget == 1 ? 0x22000001U : 0x22000000U);
    }

public:
    H2WorkspaceAndFallbacksChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (int combo = 0; combo < 4; ++combo) {
            for (bool staged : {false, true}) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario10(combo, staged));
            }
        }
        for (int kind = 0; kind < 4; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario9(kind));
        }
        // Exercise the aligned input planes and the original staged layout at
        // both ends of the public 16B workspace contract, with exact capacity.
        for (int guardK : {511, 512, 513, 1605, 2047, 2048, 2049}) {
            for (int combo = 0; combo < 4; ++combo) {
                for (size_t mod512 : {size_t(0U), size_t(16U)}) {
                    for (bool staged : {false, true}) {
                        ASSERT_NO_FATAL_FAILURE(CheckScenario8(guardK, combo, mod512, staged));
                    }
                }
            }
        }

        for (int shape = 0; shape < 7; ++shape) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario7(shape));
        }
        for (int budget = 0; budget < 3; ++budget) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario5(budget));
        }

        for (int shape = 0; shape < 9; ++shape) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario4(shape));
        }
        for (int budget = 0; budget < 3; ++budget) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario2(budget));
        }
        for (int budget = 0; budget < 3; ++budget) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(budget));
        }
    }
};

TEST_F(RuntimeFixture, H2WorkspaceAndFallbacks)
{
    H2WorkspaceAndFallbacksChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class H2AlignedDirectWorkspaceChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    static constexpr int shapes[][3] = {{65, 129, 513}, {83, 161, 1106}, {128, 256, 2048}};

    template <typename TShape>
    void CheckScenario1(const TShape& shape, size_t phase)
    {
        auto p = Supplemental(
            "H2AlignedDirect_" + std::to_string(shape[0]) + "_" + std::to_string(phase), shape[0], shape[1], shape[2]);
        p.description = "H2_direct_NZ_512B_planes_with_original_workspace_capacity";
        p.algoMode = "explicit_h2";
        p.workspaceBytes = "auto";
        p.transA = ACLBLAS_OP_T;
        SetDirectOutput(p);
        p.ld = {p.m + 3, p.n + 5, p.n + 7, p.n + 7};
        SCOPED_TRACE(p.caseName);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            DeviceBuffer offsetWorkspace;
            offsetWorkspace.Initialize(capacity + 512U);
            auto* base = static_cast<uint8_t*>(offsetWorkspace.Data());
            const size_t offset = (phase + 512U - reinterpret_cast<uintptr_t>(base) % 512U) % 512U;
            ASSERT_EQ(reinterpret_cast<uintptr_t>(base + offset) % 512U, phase);
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_EQ(result.algorithm, 0x22000002U);
            EXPECT_TRUE(result.metrics.passed);
            offsetWorkspace.CheckGuards();
            CheckInactiveWorkspace(offsetWorkspace, base, offset, capacity, "H2 direct inactive workspace");
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }

public:
    H2AlignedDirectWorkspaceChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (const auto& shape : shapes) {
            for (size_t phase : {size_t(16), size_t(128), size_t(448)}) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario1(shape, phase));
            }
        }
    }
};

TEST_F(RuntimeFixture, H2AlignedDirectWorkspace)
{
    H2AlignedDirectWorkspaceChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class H1Workspace16ByteAlignmentAndFusedLayoutsChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    static constexpr int rows[] = {3,   1,   3,   7,   17,  9,   257, 320, 321, 256, 268,
                                   268, 268, 268, 268, 268, 268, 268, 268, 268, 268, 268};
    static constexpr int columns[] = {17,  17,  31,  65,  129, 31,  681, 681, 681, 681, 576,
                                      577, 768, 769, 681, 681, 681, 681, 681, 681, 681, 681};
    static constexpr int reduction[] = {65,   17,   65,   31,  17,  65,   1323, 1323, 1323, 1323, 1323,
                                        1323, 1323, 1323, 512, 513, 4096, 4097, 1323, 1323, 1323, 1323};

    Param MakeScenario6(int scenario)
    {
        auto p = Supplemental(
            scenario >= 6 ? "H1MoreLongTNOwners_" + std::to_string(scenario - 6) :
            scenario == 0 ? "H1Workspace16Not32Aligned" :
                            "H1FusedOddPitch_" + std::to_string(scenario - 1),
            rows[scenario], columns[scenario], reduction[scenario]);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const int transpose = scenario == 20                 ? 0 :
                              scenario == 21                 ? 3 :
                              scenario >= 6                  ? 1 :
                              scenario == 0 || scenario == 5 ? 0 :
                                                               scenario - 1;
        // Exercise H1's alignment contract even when a larger-workspace
        // algorithm is preferred by the merged default heuristic.
        const size_t h1Capacity = size_t(p.m) * ((size_t(p.n) + 15U) / 16U * 16U) * sizeof(float);
        p.preferenceBytes = h1Capacity;
        if (scenario == 5) {
            p.caseName = "H1CopyWorkspace16Not32Aligned";
            SetDirectOutput(p);
        }
        p.transA = transpose & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = transpose & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.ld = {p.Cols()[0] + 3, p.Cols()[1] + 5, p.n + 4, p.n + 4};
        if (scenario >= 6) {
            SetDirectOutput(p);
            SetPaddedPitches(p);
            if (scenario == 18 || scenario == 19) {
                p.alpha = 0.5f;
                p.beta = -0.5f;
                p.inplace = scenario == 19;
                p.fill[2] = p.inplace ? "INPLACE" : "RANDOM_NORM_5_5";
                p.fill[3] = "RANDOM_NORM_5_5";
                if (p.inplace)
                    p.ld[2] = p.ld[3];
            }
        }
        return p;
    }

    void CheckScenario6(int scenario)
    {
        auto p = MakeScenario6(scenario);
        const size_t h1Capacity = size_t(p.m) * ((size_t(p.n) + 15U) / 16U * 16U) * sizeof(float);
        SCOPED_TRACE(p.caseName);
        try {
            DeviceBuffer offsetWorkspace;
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            ASSERT_EQ(capacity, h1Capacity);
            ASSERT_EQ(reinterpret_cast<uintptr_t>(operation.Operand(3)) % 32U, 0U);
            // Extra allocation belongs to this test harness. Pass exactly the
            // heuristic capacity and force the intended fusion/fallback alignment.
            offsetWorkspace.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(offsetWorkspace.Data());
            ASSERT_EQ(reinterpret_cast<uintptr_t>(base) % 16U, 0U);
            const size_t expectedMod32 = scenario == 0 || scenario == 5 || scenario >= 6 ? 16U : 0U;
            const size_t offset = (expectedMod32 + 32U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            void* workspace = base + offset;
            ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace) % 32U, expectedMod32);
            operation.SetWorkspace(workspace);
            std::cout << "LT_MATMUL_ALIGNMENT case=" << p.caseName << " workspace_mod32=" << expectedMod32
                      << " passed_capacity=" << capacity << " test_extra_allocation_bytes=" << offsetWorkspace.Bytes()
                      << std::endl;
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_EQ(result.algorithm, 0x22000001U);
            EXPECT_TRUE(result.metrics.passed);
            offsetWorkspace.CheckGuards();
            CheckInactiveWorkspace(offsetWorkspace, base, offset, capacity, "D2H inactive workspace alignment bytes");
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
    Param MakeScenario5(int reduction, int combo, size_t mod32, bool staged)
    {
        auto p = Supplemental(
            "H5Guard_" + std::to_string(reduction) + "_" + std::to_string(combo) + "_" + std::to_string(mod32) + "_" +
                std::to_string(staged),
            1683, 139, reduction);
        ConfigurePackedGuard(p, combo, staged, "explicit_h5");
        return p;
    }

    void CheckScenario5(int reduction, int combo, size_t mod32, bool staged)
    {
        auto p = MakeScenario5(reduction, combo, mod32, staged);
        SCOPED_TRACE(p.caseName);
        try {
            Invocation operation(p, handle_, stream_);
            const size_t capacity = operation.Algorithm().max_workspace_bytes;
            ASSERT_EQ(capacity, size_t(p.m) * ((p.n + 15U) / 16U * 16U) * sizeof(float));
            DeviceBuffer extra;
            extra.Initialize(capacity + 32U);
            auto* base = static_cast<uint8_t*>(extra.Data());
            const size_t offset = (mod32 + 32U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
            operation.SetWorkspace(base + offset);
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.algorithm, 0x22000005U);
            EXPECT_EQ(result.status, ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(result.metrics.passed);
            extra.CheckGuards();
            CheckInactiveWorkspace(extra, base, offset, capacity, "H5 exact capacity inactive bytes");
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
    Param MakeScenario4(int kind)
    {
        auto p = Supplemental("H5Contract_" + std::to_string(kind), 33, 139, 68);
        p.algoMode = "explicit_h5";
        p.workspaceBytes = "auto";
        p.expectResult = kind < 2 ? ACLBLAS_STATUS_INVALID_VALUE : ACLBLAS_STATUS_NOT_SUPPORTED;
        if (kind == 0) {
            p.workspaceBytes = std::to_string(size_t(p.m) * 144U * sizeof(float) - 1U);
            p.forgeWorkspaceCapacity = true;
        }
        if (kind == 1)
            p.misalignWorkspace = true;
        return p;
    }

    void CheckScenario4(int kind)
    {
        auto p = MakeScenario4(kind);
        try {
            Invocation operation(p, handle_, stream_);
            if (kind >= 2) {
                PackedAlgo bad{};
                CopyBytes(&bad, sizeof(bad), operation.Algorithm().data, sizeof(bad));
                if (kind == 2)
                    bad.l1nDiv16 = 8U;
                if (kind == 3)
                    bad.splitK = 2U;
                if (kind == 4)
                    bad.flags = 1U;
                CopyBytes(operation.Algorithm().data, sizeof(operation.Algorithm().data), &bad, sizeof(bad));
            }
            const auto result = operation.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            EXPECT_TRUE(result.metrics.passed);
        } catch (const std::exception& error) {
            ADD_FAILURE() << error.what();
        }
    }
    Param MakeScenario3(int kind)
    {
        auto p = Supplemental(
            "H5ShapeBoundary_" + std::to_string(kind), kind == 0 ? 1921 : 33, kind == 1 ? 161 : 139,
            kind == 2 ? 129 : 68);
        p.algoMode = "explicit_h5";
        p.workspaceBytes = "auto";
        p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        return p;
    }

    void CheckScenario3(int kind)
    {
        auto p = MakeScenario3(kind);
        CheckSimpleCase(p, handle_, stream_, 0U);
    }
    Param MakeScenario2(int budget)
    {
        auto p = Supplemental("H5Budget_" + std::to_string(budget), 1684, 138, 68);
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        const size_t required = size_t(p.m) * 144U * sizeof(float);
        p.preferenceBytes = budget == 0 ? required : budget == 1 ? required - 1U : 0U;
        return p;
    }

    void CheckScenario2(int budget)
    {
        auto p = MakeScenario2(budget);
        CheckSimpleCase(p, handle_, stream_, budget == 0 ? 0x22000005U : 0x22000000U);
    }
    Param MakeScenario1(int kind)
    {
        auto p = Supplemental(
            "H5ScalarAndMaximum_" + std::to_string(kind), kind == 4 ? 1920 : 33, kind == 4 ? 160 : 139,
            kind == 1 ? 0 :
            kind == 4 ? 128 :
                        68);
        p.algoMode = "explicit_h5";
        p.workspaceBytes = "auto";
        if (kind == 0) {
            p.alpha = 0.0f;
            p.fill[0] = p.fill[1] = "NULL";
        }
        if (kind == 2)
            p.alpha = std::numeric_limits<float>::infinity();
        if (kind == 3)
            p.beta = std::numeric_limits<float>::infinity();
        return p;
    }

    void CheckScenario1(int kind)
    {
        auto p = MakeScenario1(kind);
        CheckSimpleCase(p, handle_, stream_, 0U);
    }

    void CheckH5Capsule()
    {
        // H5 has its own 96x160 capsule and R capacity; H1 remains 128x128.
        aclblasLtMatmulAlgo_t h5{};
        ASSERT_EQ(
            aclblasLtMatmulAlgoInit(
                handle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0x22000005, &h5),
            ACLBLAS_STATUS_SUCCESS);
        PackedAlgo h5Packed{};
        CopyBytes(&h5Packed, sizeof(h5Packed), h5.data, sizeof(h5Packed));
        EXPECT_EQ(h5Packed.l1mDiv16, 6U);
        EXPECT_EQ(h5Packed.l1nDiv16, 10U);
        EXPECT_EQ(h5Packed.numBuffers, 1U);
        EXPECT_EQ(h5Packed.splitK, 1U);
        EXPECT_EQ(h5Packed.flags, 0U);
        for (const auto field :
             {std::make_pair(ACLBLASLT_ALGO_CONFIG_ID, uint32_t(0x22000005U)),
              std::make_pair(ACLBLASLT_ALGO_CONFIG_TILE_ID, uint32_t(0U)),
              std::make_pair(ACLBLASLT_ALGO_CONFIG_STAGES_ID, uint32_t(1U)),
              std::make_pair(ACLBLASLT_ALGO_CONFIG_SPLITK_NUM, uint32_t(1U))}) {
            uint32_t value = 0;
            size_t written = 0;
            ASSERT_EQ(
                aclblasLtMatmulAlgoConfigGetAttribute(&h5, field.first, &value, sizeof(value), &written),
                ACLBLAS_STATUS_SUCCESS);
            EXPECT_EQ(value, field.second);
            EXPECT_EQ(written, sizeof(value));
            EXPECT_EQ(
                aclblasLtMatmulAlgoConfigSetAttribute(&h5, field.first, &value, sizeof(value)),
                ACLBLAS_STATUS_NOT_SUPPORTED);
        }
        int enumerated[16]{};
        int enumeratedCount = 0;
        ASSERT_EQ(
            aclblasLtMatmulAlgoGetIds(
                handle_, ACLBLAS_COMPUTE_32F, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, enumerated, 16,
                &enumeratedCount),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_GE(enumeratedCount, 4);
        EXPECT_NE(std::find(enumerated, enumerated + enumeratedCount, int(0x22000005)), enumerated + enumeratedCount);
    }

public:
    H1Workspace16ByteAlignmentAndFusedLayoutsChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        constexpr int scenarioCount = sizeof(rows) / sizeof(rows[0]);
        for (int scenario = 0; scenario < scenarioCount; ++scenario) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario6(scenario));
        }

        ASSERT_NO_FATAL_FAILURE(CheckH5Capsule());
        for (int reduction : {68, 128}) {
            for (int combo = 0; combo < 4; ++combo) {
                for (size_t mod32 : {size_t(0), size_t(16)}) {
                    for (bool staged : {false, true}) {
                        ASSERT_NO_FATAL_FAILURE(CheckScenario5(reduction, combo, mod32, staged));
                    }
                }
            }
        }
        for (int kind = 0; kind < 5; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario4(kind));
        }
        for (int kind = 0; kind < 3; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario3(kind));
        }
        for (int budget = 0; budget < 3; ++budget) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario2(budget));
        }
        for (int kind = 0; kind < 5; ++kind) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(kind));
        }
    }
};

TEST_F(RuntimeFixture, H1Workspace16ByteAlignmentAndFusedLayouts)
{
    H1Workspace16ByteAlignmentAndFusedLayoutsChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class H9BoundariesAndFallbacksChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    std::vector<Param> cases;

    void CheckScenario2(int trans, int k)
    {
        auto p = Supplemental("H9LayoutK_" + std::to_string(trans) + "_" + std::to_string(k), 257, 289, k);
        p.transA = trans & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = trans & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        cases.push_back(p);
    }
    void CheckScenario1(int mode)
    {
        auto p = Supplemental("H9Scalar_" + std::to_string(mode), 81, 289, 29);
        p.inplace = false;
        if (mode == 0) {
            p.alpha = 2.0f;
            p.beta = -3.0f;
        }
        ConfigureUnusedProductScalars(p, mode, false);
        ConfigureNonfiniteAndExactScalars(p, mode);
        if (mode == 6) {
            p.alpha = 1.0f;
            p.beta = 0.0f;
            p.k = 0;
        }
        cases.push_back(p);
    }

public:
    H9BoundariesAndFallbacksChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (int trans = 0; trans < 4; ++trans)
            for (int k : {1, 16, 17, 29, 32, 33}) {
                ASSERT_NO_FATAL_FAILURE(CheckScenario2(trans, k));
            }
        for (int n : {16, 17, 31, 32, 33, 287, 288, 289, 543, 544, 545, 551, 559, 560, 561, 576})
            cases.push_back(Supplemental("H9NTail_" + std::to_string(n), 81, n, 29));
        for (auto& p : cases) {
            SetDirectOutput(p);
        }
        for (int mode = 0; mode < 7; ++mode) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(mode));
        }
        for (auto& p : cases) {
            p.algoMode = "explicit_h9";
            p.workspaceBytes = "auto";
            p.ld = {p.Cols()[0] + 1, p.Cols()[1] + 2, p.n + 3, p.n + 3};
            SCOPED_TRACE(p.caseName);
            CheckSimpleCase(p, handle_, stream_, 0x22000009U);
        }
    }
};

TEST_F(RuntimeFixture, H9BoundariesAndFallbacks)
{
    H9BoundariesAndFallbacksChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

class H9WorkspaceCapsuleAndHeuristicChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;

    Param ContractParameter(int mode)
    {
        auto p = Supplemental("H9Contract_" + std::to_string(mode), 729, 551, 29);
        SetDirectOutput(p);
        p.algoMode = "explicit_h9";
        p.workspaceBytes = "auto";
        if (mode == 1) {
            p.workspaceBytes = std::to_string(4U * 729U * 560U - 1U);
            p.forgeWorkspaceCapacity = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (mode == 2) {
            p.misalignWorkspace = true;
            p.expectResult = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (mode == 3) {
            p.expectResult = ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        if (mode == 4) {
            p.preferenceBytes = 0;
            p.algoMode = "heuristic";
            p.workspaceBytes = "0";
        }
        return p;
    }

    void CheckScenario1(int mode)
    {
        auto p = ContractParameter(mode);
        SCOPED_TRACE(p.caseName);
        try {
            Invocation op(p, handle_, stream_);
            if (mode == 3) {
                PackedAlgo x{};
                CopyBytes(&x, sizeof(x), op.Algorithm().data, sizeof(x));
                ++x.l1nDiv16;
                CopyBytes(op.Algorithm().data, sizeof(op.Algorithm().data), &x, sizeof(x));
            }
            DeviceBuffer shifted;
            if (mode == 0 || mode == 5 || mode == 6) {
                const size_t cap = op.Algorithm().max_workspace_bytes;
                shifted.Initialize(cap + 32U);
                auto* base = static_cast<uint8_t*>(shifted.Data());
                const size_t offset = ((mode == 5 ? 0U : 16U) + 32U - reinterpret_cast<uintptr_t>(base) % 32U) % 32U;
                op.SetWorkspace(base + offset);
                if (mode == 6)
                    op.SetOperand(2, op.Operand(3));
            }
            const auto result = op.Validate(false);
            Record(p, result);
            EXPECT_EQ(result.status, p.expectResult);
            if (p.expectResult == ACLBLAS_STATUS_SUCCESS)
                EXPECT_TRUE(result.metrics.passed);
            EXPECT_EQ(result.algorithm, mode == 4 ? 0x22000000U : 0x22000009U);
            shifted.CheckGuards();
            if (shifted.Bytes()) {
                std::vector<uint8_t> bytes(shifted.Bytes());
                AclCheck(
                    aclrtMemcpy(bytes.data(), bytes.size(), shifted.Data(), bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST),
                    "H9 untouched workspace");
                EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [](uint8_t x) { return x == GUARD_BYTE; }));
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << e.what();
        }
    }

public:
    H9WorkspaceCapsuleAndHeuristicChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (int mode = 0; mode < 7; ++mode) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(mode));
        }
    }
};

TEST_F(RuntimeFixture, H9WorkspaceCapsuleAndHeuristic)
{
    H9WorkspaceCapsuleAndHeuristicChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

TEST_F(RuntimeFixture, H2PackBankStrideBoundaries)
{
    // Exercise both UB stride parities, the 256-column tile boundary,
    // odd physical pitches, partial 16-row jobs, and every transpose pair.
    static constexpr int shapes[][3] = {{83, 161, 1106}, {84, 168, 1107},  {81, 159, 1105}, {65, 129, 513},
                                        {127, 255, 33},  {128, 256, 2049}, {129, 257, 17}};
    for (int trans = 0; trans < 4; ++trans) {
        for (const auto& shape : shapes) {
            auto p = Supplemental(
                "H2Bank_" + std::to_string(trans) + "_" + std::to_string(shape[0]), shape[0], shape[1], shape[2]);
            p.transA = trans & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            p.transB = trans & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
            SetDirectOutput(p);
            p.algoMode = "explicit_h2";
            p.workspaceBytes = "auto";
            p.ld = {p.Cols()[0] + 1, p.Cols()[1] + 3, p.n + 5, p.n + 5};
            SCOPED_TRACE(p.caseName);
            CheckSimpleCase(p, handle_, stream_, 0x22000002U);
        }
    }
}

class H1LongTN112BoundariesAndFallbacksChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    std::vector<Param> cases;
    static constexpr int shapes[][3] = {{1125, 344, 1387}, {1124, 343, 1386}, {1126, 345, 1388}, {1025, 337, 1023},
                                        {1024, 336, 1024}, {1280, 384, 2049}, {1125, 335, 513},  {1125, 336, 512},
                                        {768, 344, 1387},  {769, 344, 1387},  {1280, 344, 1387}, {1281, 344, 1387},
                                        {1125, 256, 1387}, {1125, 257, 1387}, {1125, 385, 1387}};

    template <typename TShape>
    void CheckScenario3(const TShape& shape)
    {
        auto p = Supplemental(
            "H1N112Shape_" + std::to_string(shape[0]) + "_" + std::to_string(shape[1]) + "_" + std::to_string(shape[2]),
            shape[0], shape[1], shape[2]);
        p.transA = ACLBLAS_OP_T;
        p.transB = ACLBLAS_OP_N;
        SetDirectOutput(p);
        cases.push_back(p);
    }
    void CheckScenario2(int trans)
    {
        auto p = Supplemental("H1N112Layout_" + std::to_string(trans), 1125, 344, 513);
        p.transA = trans & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = trans & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        SetDirectOutput(p);
        cases.push_back(p);
    }
    void CheckScenario1(int mode)
    {
        auto p = Supplemental("H1N112Scalar_" + std::to_string(mode), 1125, 344, 513);
        p.transA = ACLBLAS_OP_T;
        p.transB = ACLBLAS_OP_N;
        p.inplace = false;
        if (mode == 0) {
            p.alpha = 2.0f;
            p.beta = -3.0f;
        }
        ConfigureUnusedProductScalars(p, mode, true);
        ConfigureNonfiniteAndExactScalars(p, mode);
        if (mode == 6) {
            p.alpha = 32.0f;
            p.beta = 0.0f;
        }
        cases.push_back(p);
    }

public:
    H1LongTN112BoundariesAndFallbacksChecks(aclblasLtHandle_t handle, aclrtStream stream)
        : handle_(handle), stream_(stream)
    {}
    void Run()
    {
        for (const auto& shape : shapes) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario3(shape));
        }
        for (int trans = 0; trans < 4; ++trans) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario2(trans));
        }
        for (int mode = 0; mode < 7; ++mode) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(mode));
        }
        for (auto& p : cases) {
            p.algoMode = "heuristic";
            p.workspaceBytes = "auto";
            // Pin the original H1 budget; H2/H10 have independent default-route
            // tests and must not replace the H1 boundary/fallback coverage here.
            p.preferenceBytes = size_t(p.m) * ((size_t(p.n) + 15U) / 16U * 16U) * sizeof(float);
            p.ld = {p.Cols()[0] + 1, p.Cols()[1] + 3, p.n + 5, p.n + 5};
            SCOPED_TRACE(p.caseName);
            CheckSimpleCase(p, handle_, stream_, 0x22000001U);
        }
    }
};

TEST_F(RuntimeFixture, H1LongTN112BoundariesAndFallbacks)
{
    H1LongTN112BoundariesAndFallbacksChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

inline std::vector<Param> TallNTCases()
{
    static constexpr int shapes[][3] = {{1280, 91, 513},  {1281, 91, 513}, {2004, 91, 1694}, {2559, 91, 513},
                                        {2560, 91, 513},  {2561, 91, 513}, {2004, 64, 513},  {2004, 65, 513},
                                        {2004, 128, 513}, {2004, 129, 513}};
    std::vector<Param> cases;
    for (const auto& shape : shapes) {
        auto p = Supplemental(
            "H1TallNT_" + std::to_string(shape[0]) + "_" + std::to_string(shape[1]) + "_" + std::to_string(shape[2]),
            shape[0], shape[1], shape[2]);
        p.transA = ACLBLAS_OP_N;
        p.transB = ACLBLAS_OP_T;
        SetDirectOutput(p);
        cases.push_back(p);
    }
    auto conjugate = Supplemental("H1TallNC_Padded", 2003, 91, 513);
    conjugate.transA = ACLBLAS_OP_N;
    conjugate.transB = ACLBLAS_OP_C;
    conjugate.alpha = 1.0f;
    conjugate.beta = 0.0f;
    conjugate.inplace = false;
    conjugate.fill[2] = "NULL";
    cases.push_back(conjugate);
    auto staged = Supplemental("H1TallNT_StagedScalars", 2004, 91, 513);
    staged.transA = ACLBLAS_OP_N;
    staged.transB = ACLBLAS_OP_T;
    staged.alpha = 2.0f;
    staged.beta = -0.5f;
    staged.inplace = false;
    cases.push_back(staged);
    return cases;
}

TEST_F(RuntimeFixture, H1TallNarrowNTBoundaries)
{
    auto cases = TallNTCases();
    for (auto& p : cases) {
        p.algoMode = "heuristic";
        p.workspaceBytes = "auto";
        p.preferenceBytes = size_t(p.m) * ((size_t(p.n) + 15U) / 16U * 16U) * sizeof(float);
        p.ld = {p.Cols()[0] + 1, p.Cols()[1] + 3, p.n + 5, p.n + 5};
        SCOPED_TRACE(p.caseName);
        CheckSimpleCase(p, handle_, stream_, 0x22000001U);
    }
}

class H2SmallTNN96BoundariesChecks {
    aclblasLtHandle_t handle_;
    aclrtStream stream_;
    std::vector<Param> cases;
    static constexpr int shapes[][3] = {{65, 129, 513},   {83, 161, 1106}, {127, 191, 2047},
                                        {128, 192, 2048}, {64, 161, 1106}, {129, 161, 1106},
                                        {83, 193, 1106},  {83, 161, 512},  {83, 161, 2049}};

    template <typename TShape>
    void CheckScenario3(const TShape& shape)
    {
        auto p = Supplemental(
            "H2N96Shape_" + std::to_string(shape[0]) + "_" + std::to_string(shape[1]) + "_" + std::to_string(shape[2]),
            shape[0], shape[1], shape[2]);
        p.transA = ACLBLAS_OP_T;
        p.transB = ACLBLAS_OP_N;
        SetDirectOutput(p);
        cases.push_back(p);
    }
    void CheckScenario2(int trans)
    {
        auto p = Supplemental("H2N96Layout_" + std::to_string(trans), 83, 161, 513);
        p.transA = trans & 1 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        p.transB = trans & 2 ? ACLBLAS_OP_T : ACLBLAS_OP_N;
        SetDirectOutput(p);
        cases.push_back(p);
    }
    void CheckScenario1(int mode)
    {
        auto p = Supplemental("H2N96Scalar_" + std::to_string(mode), 83, 161, 513);
        p.transA = ACLBLAS_OP_T;
        p.transB = ACLBLAS_OP_N;
        p.inplace = false;
        if (mode == 0) {
            p.alpha = 0.5f;
            p.beta = -0.5f;
        }
        ConfigureUnusedProductScalars(p, mode, true);
        if (mode == 4) {
            p.alpha = 1.0f;
            p.beta = 0.0f;
            p.fill[0] = p.fill[1] = "BITS_EXACT";
        }
        cases.push_back(p);
    }

public:
    H2SmallTNN96BoundariesChecks(aclblasLtHandle_t handle, aclrtStream stream) : handle_(handle), stream_(stream) {}
    void Run()
    {
        for (const auto& shape : shapes) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario3(shape));
        }
        for (int trans = 0; trans < 4; ++trans) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario2(trans));
        }
        for (int mode = 0; mode < 5; ++mode) {
            ASSERT_NO_FATAL_FAILURE(CheckScenario1(mode));
        }
        for (auto& p : cases) {
            p.algoMode = "explicit_h2";
            p.workspaceBytes = "auto";
            p.ld = {p.Cols()[0] + 1, p.Cols()[1] + 3, p.n + 5, p.n + 5};
            SCOPED_TRACE(p.caseName);
            CheckSimpleCase(p, handle_, stream_, 0x22000002U);
        }
    }
};

TEST_F(RuntimeFixture, H2SmallTNN96Boundaries)
{
    H2SmallTNN96BoundariesChecks checks(handle_, stream_);
    ASSERT_NO_FATAL_FAILURE(checks.Run());
}

// Run this alone in a bounded subprocess (run_validation.py --include-invalid-stream).
// A driver or runtime abort remains a failed supplementary test, never a passing CSV row.
TEST_F(RuntimeFixture, DISABLED_InvalidStreamRejected)
{
    auto p = Supplemental("InvalidStreamRejected", 16, 16, 16);
    p.inplace = false;
    p.fill[2] = "RANDOM_NORM_5_5";
    p.alpha = 1.0f;
    p.beta = 0.0f;
    p.invalidStream = true;
    p.expectResult = ACLBLAS_STATUS_EXECUTION_FAILED;
    try {
        Invocation operation(p, handle_, stream_);
        const auto result = operation.Validate(false);
        Record(p, result);
        EXPECT_EQ(result.status, ACLBLAS_STATUS_EXECUTION_FAILED);
    } catch (const std::exception& error) {
        FAIL() << error.what();
    }
}

} // namespace LtMatmul22Test
