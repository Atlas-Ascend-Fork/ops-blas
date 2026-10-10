/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAXPY_PARAM_H
#define CAXPY_PARAM_H

#include "csv_loader.h"
#include "fill.h"

struct CaxpyParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    int incy = 1;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    BlasFillMode x = parseFill("RANDOM_NORM_N1_1");
    BlasFillMode y = parseFill("RANDOM_NORM_N1_1");

    explicit CaxpyParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        alphaReal = parseFloat(ReadMap(map, "alpha_real", "1"));
        alphaImag = parseFloat(ReadMap(map, "alpha_imag", "0"));
        x = parseFill(ReadMap(map, "x_fill", "RANDOM_NORM_N1_1"));
        y = parseFill(ReadMap(map, "y_fill", "RANDOM_NORM_N1_1"));
    }
};

#endif // CAXPY_PARAM_H
