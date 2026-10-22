/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_PARAM_H
#define STRMM_PARAM_H

#include <algorithm>
#include <string>
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

struct StrmmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int m = 0;
    int n = 0;
    float   alpha = 1.0f;
    bool    nullAlpha = false;
    bool    nullA = false;
    bool    nullB = false;
    bool    nullC = false;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode bFill = parseFill("RANDOM_NORM_5_5");

    explicit StrmmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side   = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo   = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag   = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        m      = std::stoi(ReadMap(map, "m", "0"));
        n      = std::stoi(ReadMap(map, "n", "0"));

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr");
        alpha = nullAlpha ? 0.0f : parseFloat(alphaStr, 1.0f);

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");

        // The A2/A3 task CSV leaves leading dimensions blank for compact matrices.
        const std::string ldaText = ReadMap(map, "lda", "");
        const std::string ldbText = ReadMap(map, "ldb", "");
        const std::string ldcText = ReadMap(map, "ldc", "");
        lda = ldaText.empty() ? std::max(1, side == ACLBLAS_SIDE_LEFT ? m : n) : std::stoi(ldaText);
        ldb = ldbText.empty() ? std::max(1, m) : std::stoi(ldbText);
        ldc = ldcText.empty() ? std::max(1, m) : std::stoi(ldcText);
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));
    }
};

#endif // STRMM_PARAM_H
