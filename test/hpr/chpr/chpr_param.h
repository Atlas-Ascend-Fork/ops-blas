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

struct ChprParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alpha = 1.0f;
    bool alphaNull = false;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    BlasFillMode ap = parseFill("RANDOM");

    ChprParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        alphaNull = (alphaStr == "NULLPTR" || alphaStr == "null");
        if (!alphaNull) {
            alpha = parseFloat(alphaStr, 1.0f);
        }
        x = parseFill(ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM")));
        incx = parseInt(ReadMap(map, "incx", "1"));
        ap = parseFill(ReadMap(map, "ap_fill", ReadMap(map, "ap", "RANDOM")));
    }
};
