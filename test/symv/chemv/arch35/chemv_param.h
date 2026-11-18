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
#include "cann_ops_blas.h"

struct ChemvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo;
    int n, lda, incx, incy;
    aclblasComplex alpha, beta;
    bool nullAlpha, nullBeta;
    BlasFillMode a, x, y;
    explicit ChemvParam(const csv_map& m) : BlasTestParamBase(m)
    {
        const auto u = ReadMap(m, "uplo");
        uplo = u == "ACLBLAS_UPPER" ? ACLBLAS_UPPER :
               u == "ACLBLAS_LOWER" ? ACLBLAS_LOWER :
                                      static_cast<aclblasFillMode_t>(std::stoi(u));
        n = parseInt(ReadMap(m, "n"));
        lda = parseInt(ReadMap(m, "lda"));
        incx = parseInt(ReadMap(m, "incx"));
        incy = parseInt(ReadMap(m, "incy"));
        const auto ar = ReadMap(m, "alpha_real");
        const auto br = ReadMap(m, "beta_real");
        nullAlpha = ar == "null" || ar == "NULLPTR";
        nullBeta = br == "null" || br == "NULLPTR";
        alpha = {nullAlpha ? 0.0f : parseFloat(ar), parseFloat(ReadMap(m, "alpha_imag"))};
        beta = {nullBeta ? 0.0f : parseFloat(br), parseFloat(ReadMap(m, "beta_imag"))};
        a = parseFill(ReadMap(m, "a_fill"));
        x = parseFill(ReadMap(m, "x_fill"));
        y = parseFill(ReadMap(m, "y_fill"));
    }
    bool IsPerformanceCase() const { return caseName.rfind("TC_PF", 0) == 0; }
};
