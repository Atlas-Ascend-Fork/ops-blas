/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct CtrsmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int m = 0;
    int n = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    bool alphaNull = false;
    int lda = 1;
    int ldb = 1;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_5_5");
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_5_5");
    bool performance = false;

    explicit CtrsmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "ACLBLAS_SIDE_LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "ACLBLAS_UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "ACLBLAS_OP_N"));
        diag = parseDiagType(ReadMap(map, "diag", "ACLBLAS_NON_UNIT"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));

        const std::string alphaReal = ReadMap(map, "alpha_real", "1");
        alphaNull = alphaReal == "null" || alphaReal == "NULL" || alphaReal == "nullptr";
        alpha.real = parseFloat(alphaReal, 1.0f);
        alpha.imag = parseFloat(ReadMap(map, "alpha_imag", "0"), 0.0f);

        const int order = side == ACLBLAS_SIDE_RIGHT ? n : m;
        const std::string ldaText = ReadMap(map, "lda");
        const std::string ldbText = ReadMap(map, "ldb");
        lda = ldaText.empty() ? std::max(1, order) : parseInt(ldaText);
        ldb = ldbText.empty() ? std::max(1, m) : parseInt(ldbText);
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));
        performance = caseName.rfind("TC_PF_", 0) == 0;
    }
};
