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

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct CsyrParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    bool alphaNull = false;
    bool alphaOnDevice = false;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    BlasFillMode a = parseFill("RANDOM_NORM_5_5");
    int lda = 0;

    explicit CsyrParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "ACLBLAS_UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        const std::string alphaReal = ReadMap(map, "alpha_real", "1.0");
        alphaNull = (alphaReal == "null" || alphaReal == "NULL" || alphaReal == "NULLPTR");
        alpha.real = parseFloat(alphaReal, 0.0f);
        alpha.imag = parseFloat(ReadMap(map, "alpha_imag", "0.0"), 0.0f);
        alphaOnDevice = (ReadMap(map, "alpha_on_device", "false") == "true");
        x = parseFill(ReadMap(map, "x", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        a = parseFill(ReadMap(map, "a", "RANDOM_NORM_5_5"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
    }
};
