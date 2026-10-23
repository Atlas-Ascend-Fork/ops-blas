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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

struct StrsmbatchedParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int m = 0;
    int n = 0;
    float alpha = 1.0f;
    bool alphaIsNull = false;
    int lda = 0;
    int ldb = 0;
    int batchCount = 1;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_P0.1_P1");
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_P0.1_P1");

    StrsmbatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "ACLBLAS_SIDE_LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "ACLBLAS_UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "ACLBLAS_OP_N"));
        diag = parseDiagType(ReadMap(map, "diag", "ACLBLAS_NON_UNIT"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        const std::string alphaStr = ReadMap(map, "alpha", "1.0");
        alphaIsNull = (alphaStr == "null" || alphaStr == "NULLPTR" || alphaStr == "nullptr");
        alpha = alphaIsNull ? 1.0f : parseFloat(alphaStr, 1.0f);
        const int defaultLda = (side == ACLBLAS_SIDE_LEFT) ? m : n;
        lda = (map.find("lda") != map.end()) ? parseInt(ReadMap(map, "lda", "0")) : defaultLda;
        ldb = (map.find("ldb") != map.end()) ? parseInt(ReadMap(map, "ldb", "0")) : m;
        batchCount = parseInt(ReadMap(map, "batchCount", "1"));
        aFill = parseFill(ReadMap(map, "a", "RANDOM_NORM_P0.1_P1"));
        bFill = parseFill(ReadMap(map, "b", "RANDOM_NORM_P0.1_P1"));
    }
};
