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

#include "csv_loader.h"

struct ChbmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo;
    int n, k, lda, incx, incy;
    aclblasComplex alpha, beta;
    bool nullAlpha, nullBeta;
    std::string aFill, xFill, yFill;

    explicit ChbmvParam(const csv_map& m) : BlasTestParamBase(m)
    {
        uplo = parseFillMode(ReadMap(m, "uplo", "ACLBLAS_UPPER"));
        n = parseInt(ReadMap(m, "n", "0"));
        k = parseInt(ReadMap(m, "k", "0"));
        lda = parseInt(ReadMap(m, "lda", "1"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        const std::string ar = ReadMap(m, "alpha_real", "1");
        const std::string br = ReadMap(m, "beta_real", "0");
        nullAlpha = ar == "null" || ar == "NULLPTR" || ar == "nullptr";
        nullBeta = br == "null" || br == "NULLPTR" || br == "nullptr";
        alpha = {nullAlpha ? 0.0f : parseFloat(ar), parseFloat(ReadMap(m, "alpha_imag", "0"))};
        beta = {nullBeta ? 0.0f : parseFloat(br), parseFloat(ReadMap(m, "beta_imag", "0"))};
        aFill = ReadMap(m, "a_fill", "RANDOM_NORM_5_5");
        xFill = ReadMap(m, "x_fill", "RANDOM_NORM_5_5");
        yFill = ReadMap(m, "y_fill", "RANDOM_NORM_5_5");
    }
};
