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
#include <mutex>
#include "../../../../blas/gemm_grouped_batched/arch22/sgemm_gb_plan.h"
namespace {
std::vector<gb_test::Row> Cases()
{
    // The maintainer-requested default CSV is a 50-case CI smoke subset.
    // The full task and supplemental suites have independent scripted gates.
    auto rows = gb_test::ReadCSV(gb_test::TaskCasesCSV(__FILE__));
    std::vector<gb_test::Row> selected;
    const char* pf = std::getenv("GB_INCLUDE_PERF_ACCURACY");
    for (auto& r : rows) {
        const auto name = gb_test::Cell(r, "case_name");
        if (pf || name.find("TC_PF_") != 0)
            selected.push_back(std::move(r));
    }
    if (selected.empty())
        throw std::runtime_error("missing SGEMM accuracy cases in task CSV");
    return selected;
}
gb_test::Context& SharedContext() { return gb_test::GTestContext(); }
class SgemmGroupedBatchedArch22 : public ::testing::TestWithParam<gb_test::Row> {};
TEST_P(SgemmGroupedBatchedArch22, CSVAccuracy)
{
    // One context for all tests; synchronization happens before each fixture is released.
    auto& context = SharedContext();
    gb_test::Options options;
    if (const char* b = std::getenv("GB_MAX_BYTES"))
        options.maxBytes = std::stoull(b);
    if (std::getenv("GB_STRICT_ABS_LIMIT"))
        options.ulpLimit = false;
    auto r = gb_test::Run(GetParam(), context, options);
    if (const char* log = std::getenv("GB_JSONL")) {
        std::ofstream f(log, std::ios::app);
        f << gb_test::JSON(r) << '\n';
    }
    if (r.status == "SKIP_RESOURCE")
        GTEST_SKIP() << r.message;
#ifdef GB_CPU_MOCK
    EXPECT_EQ(r.status, "CPU_MODEL_PASS") << gb_test::JSON(r);
#else
    EXPECT_EQ(r.status, "PASS") << gb_test::JSON(r);
#endif
}
INSTANTIATE_TEST_SUITE_P(
    TaskCSV, SgemmGroupedBatchedArch22, ::testing::ValuesIn(Cases()),
    [](const ::testing::TestParamInfo<gb_test::Row>& x) { return gb_test::Cell(x.param, "case_name"); });
// Mandatory golden comparisons for the hand-written TN96 path, independent
// of the selected task CSV (including the default 50-case CI smoke subset).
INSTANTIATE_TEST_SUITE_P(
    Review, SgemmGroupedBatchedArch22,
    ::testing::ValuesIn(gb_test::ReadCSV(gb_test::SiblingCSV(__FILE__, "review_regression.csv"))),
    [](const ::testing::TestParamInfo<gb_test::Row>& x) { return gb_test::Cell(x.param, "case_name"); });

void ExpectCSVCounts(const char* name, size_t accuracy, size_t performance)
{
    const auto rows = gb_test::ReadCSV(gb_test::SiblingCSV(__FILE__, name));
    size_t pf = 0;
    for (const auto& row : rows)
        pf += gb_test::Cell(row, "case_name").find("TC_PF_") == 0;
    EXPECT_EQ(rows.size(), accuracy + performance) << name;
    EXPECT_EQ(pf, performance) << name;
}
TEST(SgemmGroupedBatchedArch22Integrity, CsvCaseCount)
{
    ExpectCSVCounts("sgemm_grouped_batched_test.csv", 50, 0);
    ExpectCSVCounts("sgemm_grouped_batched_test_full.csv", 1000, 200);
    ExpectCSVCounts("supplemental.csv", 306, 0);
    ExpectCSVCounts("review_regression.csv", 3, 0);
}

TEST(SgemmGroupedBatchedArch22Planner, Small8RejectsOverlappingRanges)
{
    int sizes[] = {8, 8}, batches[] = {1, 1};
    aclblasOperation_t ops[] = {ACLBLAS_OP_N, ACLBLAS_OP_N};
    float alpha[] = {1, 1}, beta[] = {0, 0};
    float storage[6][128]{};
    const float* a[] = {storage[0], storage[1]};
    const float* b[] = {storage[2], storage[3]};
    float* c[] = {storage[4], storage[5]};
    sgemm_gb::Arguments args{2, ops, ops, sizes, sizes, sizes, alpha, a, sizes, b, sizes, beta, c, sizes, batches};
    std::vector<sgemm_gb::Small8Packet> packets;
    EXPECT_TRUE(sgemm_gb::PlanSmall8(args, packets));
    for (int delta : {-63, 0, 63}) {
        // Test output-to-input overlap on either side, not only equal pointers.
        a[1] = c[0] + (delta < 0 ? 0 : delta);
        if (delta < 0) {
            c[0] = storage[4] + 63;
            a[1] = storage[4];
        }
        packets.clear();
        EXPECT_FALSE(sgemm_gb::PlanSmall8(args, packets));
        c[0] = storage[4];
    }
    a[1] = storage[1];
    b[1] = c[0] + 1;
    packets.clear();
    EXPECT_FALSE(sgemm_gb::PlanSmall8(args, packets));
    b[1] = storage[3];
    c[1] = c[0] + 1;
    packets.clear();
    EXPECT_FALSE(sgemm_gb::PlanSmall8(args, packets));
    c[1] = c[0] + 64; // Adjacent ranges do not overlap.
    packets.clear();
    EXPECT_TRUE(sgemm_gb::PlanSmall8(args, packets));
}
} // namespace
#include "gb_async_support.h"
TEST(SgemmGroupedBatchedArch22Async, SnapshotAndStreamOrdering)
{
    auto& context = SharedContext();
    EXPECT_NO_THROW(gb_test::AsyncSnapshot(context));
    EXPECT_NO_THROW(gb_test::AsyncSerialC(context));
    EXPECT_NO_THROW(gb_test::AsyncTwoStreams(context));
    if (std::getenv("GB_ASYNC_LARGE")) {
        EXPECT_NO_THROW(gb_test::AsyncSnapshot(context, true));
        EXPECT_NO_THROW(gb_test::AsyncTwoStreams(context, true));
    }
}

TEST(SgemmGroupedBatchedArch22Async, Small8CrossGroupInputDependency)
{
    auto& context = SharedContext();
    auto row = gb_test::AsyncCase();
    for (const char* key : {"mArray", "nArray", "kArray", "ldaArray", "ldbArray", "ldcArray"})
        row[key] = "8;8";
    row["transaArray"] = row["transbArray"] = "N;N";
    row["alphaArray"] = "1;1";
    row["betaArray"] = "0;0";
    row["groupSizeArray"] = "1;1";
    gb_test::Parameters params(row);
    gb_test::Buffer a(64, 0), identity(64, 0), firstC(64, 0), secondC(64, 0);
    std::vector<float> values(64), unit(64, 0.0f), zero(64, 0.0f);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<float>(i + 1);
    for (size_t i = 0; i < 8; ++i)
        unit[i * 8 + i] = 1.0f;
    a.Upload(values);
    identity.Upload(unit);
    for (bool aliasA : {true, false}) {
        firstC.Upload(zero);
        secondC.Upload(zero);
        params.a = {a.Data(), aliasA ? firstC.Data() : identity.Data()};
        params.b = {identity.Data(), aliasA ? identity.Data() : firstC.Data()};
        params.c = {firstC.Data(), secondC.Data()};
        gb_test::CheckBLAS(params.Call(context.handle), "cross-group alias enqueue");
        gb_test::CheckACL(aclrtSynchronizeStream(context.stream), "cross-group alias sync");
        EXPECT_EQ(firstC.ReadChecked(), values);
        EXPECT_EQ(secondC.ReadChecked(), values);
    }
}
