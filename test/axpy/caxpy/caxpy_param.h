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

struct CaxpyParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    int incy = 1;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    int nullx = 0;
    int nully = 0;
    int measure = 0;
    int warmup = 30;
    int measurements = 50;

    CaxpyParam(const csv_map& csv) : BlasTestParamBase(csv)
    {
        n = parseInt(ReadMap(csv, "n", "0"));
        incx = parseInt(ReadMap(csv, "incx", "1"));
        incy = parseInt(ReadMap(csv, "incy", "1"));
        alphaReal = parseFloat(ReadMap(csv, "alpha_real", "1.0"));
        alphaImag = parseFloat(ReadMap(csv, "alpha_imag", "0.0"));
        nullx = parseInt(ReadMap(csv, "nullx", "0"));
        nully = parseInt(ReadMap(csv, "nully", "0"));
        measure = parseInt(ReadMap(csv, "measure", "0"));
        warmup = parseInt(ReadMap(csv, "warmup", "30"));
        measurements = parseInt(ReadMap(csv, "measurements", "50"));
    }
};
