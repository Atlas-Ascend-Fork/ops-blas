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
#include "fill.h"

struct CtbmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int n = 0;
    int k = 0;
    int lda = 1;
    int incx = 1;
    BlasFillMode aFill;
    BlasFillMode xFill;

    explicit CtbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(k + 1)));
        incx = parseInt(ReadMap(map, "incx", "1"));
        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        xFill = BlasFillMode(ReadMap(map, "x_fill", "RANDOM_NORM_5_5"));
    }
};
