/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gb_gtest_context.h"
#include <gtest/gtest.h>
#include <cstdlib>

namespace {
std::vector<gb_test::Row> PerformanceCases()
{
    std::vector<gb_test::Row> selected;
    for (auto& row : gb_test::ReadCSV(gb_test::TaskCasesCSV(__FILE__)))
        if (gb_test::Cell(row, "case_name").find("TC_PF_") == 0)
            selected.push_back(std::move(row));
    // The default smoke and supplemental CSVs contain accuracy cases only.
    // Selecting the full task CSV also registers all 200 performance cases.
    return selected;
}

class SgemmGroupedBatchedArch22Perf : public ::testing::TestWithParam<gb_test::Row> {};
#ifdef GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SgemmGroupedBatchedArch22Perf);
#endif
TEST_P(SgemmGroupedBatchedArch22Perf, CSVKernelTiming)
{
    auto& context = gb_test::GTestContext();
    gb_test::Options options;
    options.mode = "perf";
    options.warmup = 5;
    options.samples = 20;
    if (const char* bytes = std::getenv("GB_MAX_BYTES"))
        options.maxBytes = std::stoull(bytes);
    const auto result = gb_test::Run(GetParam(), context, options);
    if (const char* path = std::getenv("GB_JSONL")) {
        std::ofstream log(path, std::ios::app);
        ASSERT_TRUE(log.good()) << "Cannot open performance evidence file";
        log << gb_test::JSON(result) << '\n';
        log.flush();
        ASSERT_TRUE(log.good()) << "Cannot save performance evidence";
    }
    // This asserts that the measurement ran. Acceptance still requires
    // separate precision evidence and strict raw msprof kernel aggregation.
    EXPECT_EQ(result.status, "MEASURED_NOT_ACCEPTED") << gb_test::JSON(result);
    EXPECT_EQ(result.eventUs.size(), 20u);
}

INSTANTIATE_TEST_SUITE_P(
    TaskCSV, SgemmGroupedBatchedArch22Perf, ::testing::ValuesIn(PerformanceCases()),
    [](const ::testing::TestParamInfo<gb_test::Row>& item) { return gb_test::Cell(item.param, "case_name"); });
} // namespace
