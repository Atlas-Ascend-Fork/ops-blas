/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "blas_test.h"
#include "hgemmStridedBatched_param.h"
#include "hgemmStridedBatched_golden.h"
#include "hgemmStridedBatched_npu_wrapper.h"

namespace {
using namespace hgemm_test;
class HgemmBatchTest : public BlasTest<HgemmStridedBatchedParam> {};

std::vector<HgemmStridedBatchedParam> Cases()
{
    const char* csv = std::getenv("HGEMM_CSV");
    return GetCasesFromCsv<HgemmStridedBatchedParam>(csv ? csv : ReplaceFileExtension2Csv(__FILE__));
}

INSTANTIATE_TEST_SUITE_P(
    HgemmStridedBatched, HgemmBatchTest, ::testing::ValuesIn(Cases()), PrintCaseInfoString<HgemmStridedBatchedParam>);

TEST_P(HgemmBatchTest, CsvDriven)
{
    const auto& p = GetParam();
    CaseData data(p);
    ASSERT_NO_FATAL_FAILURE(data.Upload(handle_));
    ASSERT_EQ(static_cast<int>(data.Call(handle_)), p.expected);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    if (p.workspaceBytes)
        ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
    if (data.negative || data.empty || !data.dc.Ptr())
        return;
    data.dc.Download(data.actual);
    ASSERT_TRUE(data.dc.GuardsIntact()) << "C sub-buffer guard overwritten";
    data.Dump();
    // Profiling has a separate correctness pass; retain the exact launch count.
    if (const char* count = std::getenv("HGEMM_PROFILE_REPEATS")) {
        const int repeats = std::stoi(count);
        ASSERT_GT(repeats, 0);
        for (int j = 0; j < repeats; ++j)
            ASSERT_EQ(data.Call(handle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        return;
    }
    const HgemmStridedBatchedGolden golden{p, data.a, data.b, data.initial, data.actual};
    ASSERT_NO_FATAL_FAILURE(golden.Compare());
    ASSERT_NO_FATAL_FAILURE(data.CheckPadding());
}
} // namespace
