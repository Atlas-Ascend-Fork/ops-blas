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

struct StrsmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int m = 0;
    int n = 0;
    float alpha = 1.0f;
    int lda = 0;
    int ldb = 0;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM");
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM");

    StrsmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"), 1.0f);
        lda = parseInt(ReadMap(map, "lda", "0"));
        ldb = parseInt(ReadMap(map, "ldb", "0"));
        aFill = parseFill(ReadMap(map, "a", "RANDOM_NORM"));
        bFill = parseFill(ReadMap(map, "b", "RANDOM_NORM"));
    }
};
