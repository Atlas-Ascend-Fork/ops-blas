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
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct Ssyr2kParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode fillA = BlasFillMode("RANDOM_P1_P1");
    BlasFillMode fillB = BlasFillMode("RANDOM_P1_P1");
    BlasFillMode fillC = BlasFillMode("RANDOM_P1_P1");

    explicit Ssyr2kParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = ParseIntColumn(map, "n", "0");
        k = ParseIntColumn(map, "k", "0");

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr");
        alpha = nullAlpha ? 0.0f : parseFloat(alphaStr, 1.0f);

        std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr");
        beta = nullBeta ? 0.0f : parseFloat(betaStr, 0.0f);

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");

        lda = ParseIntColumn(map, "lda", "0");
        ldb = ParseIntColumn(map, "ldb", "0");
        ldc = ParseIntColumn(map, "ldc", "0");

        fillA = BlasFillMode(ReadMap(map, "fillA", "RANDOM_P1_P1"));
        fillB = BlasFillMode(ReadMap(map, "fillB", "RANDOM_P1_P1"));
        fillC = BlasFillMode(ReadMap(map, "fillC", "RANDOM_P1_P1"));
    }
};
