/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include "blas_test.h"
#include "device.h"
#include "icamin_param.h"
#include "icamin_golden.h"

namespace {
struct IcaminArch22Param : IcaminParam {
    int offset;
    explicit IcaminArch22Param(const csv_map& map) : IcaminParam(map), offset(parseInt(ReadMap(map, "offset", "0"))) {}
};

std::vector<IcaminArch22Param> IcaminCases()
{
    std::string path = ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<IcaminArch22Param>(path);
    path.replace(path.size() - 4, 4, "_contract.csv");
    auto extra = GetCasesFromCsv<IcaminArch22Param>(path);
    cases.insert(cases.end(), extra.begin(), extra.end());
    return cases;
}

std::vector<aclblasComplex> MakeInput(const IcaminArch22Param& p)
{
    // Error and quick-return cases use one dummy element, including INT_MIN strides.
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || p.incx < 1 || p.x.method == BlasFillMode::M_NULLPTR) {
        return {{0.0f, 0.0f}};
    }
    int64_t elements = static_cast<int64_t>(p.n - 1) * p.incx + 1;
    auto flat = makeBlasArray(elements * 2, p.x, p.randomSeed);
    std::vector<aclblasComplex> input(static_cast<size_t>(elements) + p.offset, {0.0f, 0.0f});
    for (int64_t i = 0; i < elements; ++i) {
        input[static_cast<size_t>(i) + p.offset] = {flat[2 * i], flat[2 * i + 1]};
    }
    auto* base = input.data() + p.offset;
    if (p.description.rfind("mix_nan_first_", 0) == 0) {
        base[0] = {NAN, 0.0f};
    } else if (p.description.rfind("mix_nan_mid_", 0) == 0 && p.n > 1) {
        base[p.incx] = {NAN, 0.0f};
    } else if (p.description == "nan_slice_then_minimum") {
        base[static_cast<int64_t>(4096) * p.incx] = {NAN, 0.0f};
        base[static_cast<int64_t>(4097) * p.incx] = {0.0f, 0.0f};
    } else if (p.description == "tie_across_tiles_and_cores") {
        for (int index : {7679, 7680, 245759, 245760, 1048576}) {
            base[static_cast<int64_t>(index) * p.incx] = {0.0f, -0.0f};
        }
    } else if (p.description == "inf_nan_finite") {
        base[0] = {INFINITY, 0.0f};
        base[p.incx] = {NAN, 0.0f};
        base[2 * p.incx] = {0.5f, 0.0f};
    } else if (p.description == "fp32_rounding_tie") {
        base[0] = {1.0f, std::ldexp(1.0f, -24)};
        base[p.incx] = {1.0f, 0.0f};
    } else if (p.description == "subnormal_minimum") {
        base[0] = {2 * std::numeric_limits<float>::denorm_min(), 0.0f};
        base[p.incx] = {std::numeric_limits<float>::denorm_min(), 0.0f};
    } else if (p.description == "overflow_to_inf") {
        base[0] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        base[p.incx] = {std::numeric_limits<float>::max(), 0.0f};
    } else if (p.description == "tail_minimum") {
        base[static_cast<int64_t>(p.n - 1) * p.incx] = {0.0f, 0.0f};
    }
    return input;
}
} // namespace

class IcaminTest : public BlasTest<IcaminArch22Param> {};

INSTANTIATE_TEST_SUITE_P(
    Icamin, IcaminTest, ::testing::ValuesIn(IcaminCases()), PrintCaseInfoString<IcaminArch22Param>);

TEST_P(IcaminTest, CsvDriven)
{
    const auto& p = GetParam();
    auto input = MakeInput(p);
    DeviceBuffer deviceInput(input.size() * sizeof(aclblasComplex));
    deviceInput.copyFromHost(input.data(), input.size() * sizeof(aclblasComplex));
    const auto* x = p.x.method == BlasFillMode::M_NULLPTR ?
                        nullptr :
                        static_cast<const aclblasComplex*>(deviceInput.ptr()) +
                            (p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 && p.incx > 0 ? p.offset : 0);
    constexpr int SENTINEL = -73;
    std::array<int, 3> output{SENTINEL, SENTINEL, SENTINEL};
    DeviceBuffer deviceOutput(sizeof(output));
    deviceOutput.copyFromHost(output.data(), sizeof(output));
    int* result = p.resultIsNull ? nullptr : static_cast<int*>(deviceOutput.ptr()) + 1;

    // Exactly one interface call: msprof op owns warm-up and measured replay counts.
    auto status = aclblasIcamin(handle_, p.n, x, p.incx, result);
    ASSERT_EQ(status, p.expectResult) << p.caseName;
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    deviceOutput.copyToHost(output.data(), sizeof(output));
    EXPECT_EQ(output.front(), SENTINEL) << p.caseName;
    EXPECT_EQ(output.back(), SENTINEL) << p.caseName;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(output[1], SENTINEL) << p.caseName;
        return;
    }
    int golden = -1;
    ASSERT_EQ(
        aclblasIcamin_cpu(handle_, p.n, input.data() + (p.n > 0 && p.incx > 0 ? p.offset : 0), p.incx, &golden),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(output[1], golden) << p.caseName << " n=" << p.n << " incx=" << p.incx;
}

TEST_F(IcaminTest, NullHandle)
{
    EXPECT_EQ(aclblasIcamin(nullptr, 1, nullptr, 1, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(IcaminTest, BoundStreamQuickReturnOrdering)
{
    DeviceBuffer result(sizeof(int));
    for (int incx : {0, -1, INT_MIN}) {
        ASSERT_EQ(aclrtMemsetAsync(result.ptr(), sizeof(int), 0x7f, sizeof(int), stream_), ACL_SUCCESS);
        ASSERT_EQ(
            aclblasIcamin(handle_, INT_MAX, nullptr, incx, static_cast<int*>(result.ptr())), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        int actual = -1;
        result.copyToHost(&actual, sizeof(actual));
        EXPECT_EQ(actual, 0);
    }
}
