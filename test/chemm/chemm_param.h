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

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ChemmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int64_t m = 0;
    int64_t n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    bool nullAlpha = false;
    BlasFillMode aFill;
    std::string aFillSpec;
    bool aNormal = false;
    int64_t lda = 0;
    BlasFillMode bFill;
    std::string bFillSpec;
    bool bNormal = false;
    int64_t ldb = 0;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    bool nullBeta = false;
    BlasFillMode cFill;
    std::string cFillSpec;
    bool cNormal = false;
    int64_t ldc = 0;

    ChemmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        m = static_cast<int64_t>(parseInt64(ReadMap(map, "m", "0")));
        n = static_cast<int64_t>(parseInt64(ReadMap(map, "n", "0")));

        std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
        nullAlpha = (alphaRealStr == "null" || alphaRealStr == "nullptr");
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImagStr, 0.0f);

        aFillSpec = ReadMap(map, "a_fill", "RANDOM_NORM_1E3_1E3");
        aNormal = (aFillSpec == "NORMAL_NORM");
        aFill = parseFill(aFillSpec);
        const std::string ldaStr = ReadMap(map, "lda", "");
        lda = ldaStr.empty() ? (side == ACLBLAS_SIDE_LEFT ? m : n) : static_cast<int64_t>(parseInt64(ldaStr));

        bFillSpec = ReadMap(map, "b_fill", "RANDOM_NORM_1E3_1E3");
        bNormal = (bFillSpec == "NORMAL_NORM");
        bFill = parseFill(bFillSpec);
        const std::string ldbStr = ReadMap(map, "ldb", "");
        ldb = ldbStr.empty() ? m : static_cast<int64_t>(parseInt64(ldbStr));

        std::string betaRealStr = ReadMap(map, "beta_real", "0.5");
        std::string betaImagStr = ReadMap(map, "beta_imag", "0.0");
        nullBeta = (betaRealStr == "null" || betaRealStr == "nullptr");
        betaReal = nullBeta ? 0.0f : parseFloat(betaRealStr, 0.5f);
        betaImag = nullBeta ? 0.0f : parseFloat(betaImagStr, 0.0f);

        cFillSpec = ReadMap(map, "c_fill", "RANDOM_NORM_1E3_1E3");
        cNormal = (cFillSpec == "NORMAL_NORM");
        cFill = parseFill(cFillSpec);
        const std::string ldcStr = ReadMap(map, "ldc", "");
        ldc = ldcStr.empty() ? m : static_cast<int64_t>(parseInt64(ldcStr));
    }
};
