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
#include <cmath>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "stpsv_param.h"
#include "stpsv_golden.h"
#include "stpsv_npu_wrapper.h"

class TpsvArch35Test : public BlasTest<TpsvParam> {};

INSTANTIATE_TEST_SUITE_P(
    Tpsv, TpsvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<TpsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<TpsvParam>);

TEST_F(TpsvArch35Test, XSpanReachesUint32Max)
{
    // White-box coverage for the equality boundary of the x-index span guard. The device computes
    // the GM view length as (n-1)*|incx| + 1 in uint32, so a span that lands exactly on UINT32_MAX
    // (4294967295) would wrap xCount to 0 in the kernel. n=4 with incx=1431655765 gives
    // (4-1)*1431655765 = 4294967295 = UINT32_MAX, which is reachable and must be refused up front.
    // The guard rejects at span >= UINT32_MAX (not >), so this exact boundary is refused. Calls the
    // API directly with scratch operands so no device buffer of the full span is needed; validation
    // rejects before any device access, mirroring the arch22 XSpanReachesUint32Max case.
    float scratch = 0.0f;
    const aclblasStatus_t ret = aclblasStpsv(
        TpsvArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, &scratch, &scratch, 1431655765);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_SUPPORTED));
}

TEST_P(TpsvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    bool isUpper = (p.uplo == ACLBLAS_UPPER);
    auto ap = makeBlasTriangular(p.n, isUpper, "RANDOM", p.randomSeed);
    if (p.diag == ACLBLAS_UNIT) {
        float scale = std::min(1.0f, 5.0f / static_cast<float>(p.n));
        for (size_t i = 0; i < ap.size(); ++i)
            ap[i] *= scale;
    } else {
        float boost = std::max(5.0f, static_cast<float>(p.n));
        for (int i = 0; i < p.n; ++i) {
            if (isUpper) {
                ap[TpsvPackedUpperIdxCpu(i, i)] += boost;
            } else {
                ap[TpsvPackedLowerIdxCpu(i, i, p.n)] += boost;
            }
        }
    }
    auto x = makeBlasStrided(p.n, p.incx, "RANDOM", p.randomSeed + 1);
    std::vector<float> golden = x;

    aclblasStatus_t ret =
        aclblasStpsv_npu(TpsvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), x.data(), p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStpsv_cpu(TpsvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), golden.data(), p.incx);

    const int absIncx = std::abs(p.incx);
    const float* outPtr = (p.incx < 0 && p.n > 0) ? x.data() + (p.n - 1) * absIncx : x.data();
    const float* goldPtr = (p.incx < 0 && p.n > 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));
    EXPECT_TRUE(Verifier::verifyVector(
        outPtr, goldPtr, static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
}
