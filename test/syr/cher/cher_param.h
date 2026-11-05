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

#include <algorithm>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct CherParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alpha = 1.0f;
    bool alphaNull = false;
    BlasFillMode xFill = parseFill("RANDOM_NORM_1");
    int incx = 1;
    BlasFillMode aFill = parseFill("RANDOM_NORM_1");
    int lda = 0;

    explicit CherParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        alphaNull = (alphaStr == "null" || alphaStr == "NULLPTR" || alphaStr == "nullptr");
        if (!alphaNull) {
            alpha = parseFloat(alphaStr, 1.0f);
        }

        xFill = parseFill(ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM_NORM_1")));
        incx = parseInt(ReadMap(map, "incx", "1"));
        aFill = parseFill(ReadMap(map, "a_fill", ReadMap(map, "a", "RANDOM_NORM_1")));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
    }
};
