/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <algorithm>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// Parameter struct for aclblasCherk (complex<float> Hermitian rank-k update).
// Mirrors aclblasSsyrkParam with complex-specific additions:
//   - alpha/beta remain float (real scalars, per BLAS CHERK spec)
//   - nullAlpha flag added (§8 遗留问题 3): alpha is a Device pointer; the
//     null_alpha case passes nullptr to the API and expects INVALID_VALUE.
struct CherkParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_1");
    int lda = 0;
    float beta = 0.0f;
    bool nullBeta = false;
    BlasFillMode cFill = BlasFillMode("VALUE_NORM_0");
    int ldc = 0;
    bool nullA = false;
    bool nullC = false;
    bool nullAlpha = false;

    explicit CherkParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_1"));

        // Default lda (column-major):
        //   trans=N -> A is (n x k), lda >= max(1, n)
        //   trans=T/C -> A is (k x n), lda >= max(1, k)
        int defaultLda = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLda)));

        const std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr");
        beta = nullBeta ? 0.0f : parseFloat(betaStr);
        cFill = BlasFillMode(ReadMap(map, "c_fill", "VALUE_NORM_0"));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");
        nullAlpha = (ReadMap(map, "nullAlpha", "0") == "1");
    }
};
