/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../gemm_ex_contract_test.h"
namespace {
using gemm_ex_test::Param;
class CgemmExContract : public BlasTest<Param> {};
INSTANTIATE_TEST_SUITE_P(
    Official, CgemmExContract, ::testing::ValuesIn(GetCasesFromCsv<Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Param>);
TEST_P(CgemmExContract, CsvDriven) { gemm_ex_test::Run<2>(GetParam(), handle_, stream_); }
} // namespace
