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

struct Cher2Param : BlasTestParamBase {
    aclblasFillMode_t uplo;
    int n, incx, incy, lda;
    aclblasComplex alpha;
    bool nullAlpha;
    bool normal = false;
    std::string xFill, yFill, aFill;

    explicit Cher2Param(const csv_map& m) : BlasTestParamBase(m)
    {
        uplo = parseFillMode(ReadMap(m, "uplo"));
        n = parseInt(ReadMap(m, "n"));
        incx = parseInt(ReadMap(m, "incx"));
        incy = parseInt(ReadMap(m, "incy"));
        lda = parseInt(ReadMap(m, "lda"));
        const auto ar = ReadMap(m, "alpha_real");
        nullAlpha = ar == "null" || ar == "NULLPTR";
        alpha = {nullAlpha ? 0.0f : parseFloat(ar), parseFloat(ReadMap(m, "alpha_imag"))};
        xFill = ReadMap(m, "x_fill");
        yFill = ReadMap(m, "y_fill");
        aFill = ReadMap(m, "a_fill");
    }
};
