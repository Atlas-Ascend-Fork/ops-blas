/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "triangular_fixture.h"

namespace ctpttr_arch35_test {

using namespace complex_triangular_test;

class CtpttrTest : public TriangularFixture<false> {};

// Preserve the suite name used by profiling and existing GoogleTest filters.
using CtpttrArch35Test = CtpttrTest;

inline std::vector<TriangularCase> Cases()
{
    const std::string source = __FILE__;
    return LoadCases(source.substr(0, source.rfind('.')) + ".csv");
}

INSTANTIATE_TEST_SUITE_P(
    CtpttrArch35, CtpttrArch35Test, ::testing::ValuesIn(Cases()),
    [](const ::testing::TestParamInfo<TriangularCase>& info) { return info.param.caseName; });

TEST_P(CtpttrArch35Test, CsvDriven) { this->Run(GetParam()); }

TEST_F(CtpttrArch35Test, NullHandlePriority)
{
    EXPECT_EQ(
        Invoke<false>(nullptr, static_cast<aclblasFillMode_t>(255), -1, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtpttrArch35Test, EmptyValidationOrder)
{
    EXPECT_EQ(Invoke<false>(this->handle_, ACLBLAS_UPPER, 0, nullptr, nullptr, 0), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(Invoke<false>(this->handle_, ACLBLAS_LOWER, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        Invoke<false>(this->handle_, static_cast<aclblasFillMode_t>(255), 0, nullptr, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CtpttrArch35Test, PayloadsAndPackedOffsets) { this->RunPayloads(); }

TEST_F(CtpttrArch35Test, OutputInitialValues)
{
    for (const char* output : {"VALUE_NORM_0", "INDEX", "RANDOM_EXTREME"}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            auto testCase = ManualCase(5, 11, uplo);
            testCase.aFill = output;
            this->Run(testCase);
        }
    }
}

TEST_F(CtpttrArch35Test, MissingCsvColumnsRejected)
{
    const CaseRow emptyRow;
    EXPECT_THROW(TriangularCase{emptyRow}, std::runtime_error);
}

TEST_F(CtpttrArch35Test, EmptyPreservesBuffers) { this->RunNoOpProtection(); }

TEST_F(CtpttrArch35Test, OverlappingPointersAreCallerError) { this->RunOverlap(); }

TEST_F(CtpttrArch35Test, ComparatorRejectsCorruption) { this->RunComparatorChecks(); }

} // namespace ctpttr_arch35_test
