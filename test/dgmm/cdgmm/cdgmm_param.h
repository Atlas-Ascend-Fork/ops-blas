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

struct CdgmmParam : public BlasTestParamBase {
    aclblasSideMode_t mode = ACLBLAS_SIDE_LEFT;
    int m    = 0;
    int n    = 0;
    int incx = 1;
    int lda  = 0;
    int ldc  = 0;
    BlasFillMode xFill = BlasFillMode("RANDOM_1_1");
    BlasFillMode aFill = BlasFillMode("RANDOM_1_1");
    std::string xFillRaw;
    int nullx = 0;
    int nullA = 0;
    int nullC = 0;

    CdgmmParam(const csv_map& csv) : BlasTestParamBase(csv)
    {
        mode  = parseSideMode(ReadMap(csv, "mode", "LEFT"));
        m     = parseInt(ReadMap(csv, "m", "0"));
        n     = parseInt(ReadMap(csv, "n", "0"));
        incx  = parseInt(ReadMap(csv, "incx", "1"));
        // Row-major: lda/ldc are row strides, constrained by n (not m).
        lda   = parseInt(ReadMap(csv, "lda", std::to_string(std::max(1, n))));
        ldc   = parseInt(ReadMap(csv, "ldc", std::to_string(std::max(1, n))));
        xFill = BlasFillMode(ReadMap(csv, "x_fill", "RANDOM_1_1"));
        xFillRaw = ReadMap(csv, "x_fill", "RANDOM_1_1");
        aFill = BlasFillMode(ReadMap(csv, "a_fill", "RANDOM_1_1"));
        nullx = parseInt(ReadMap(csv, "nullx", "0"));
        nullA = parseInt(ReadMap(csv, "nullA", "0"));
        nullC = parseInt(ReadMap(csv, "nullC", "0"));
    }
};
