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

struct ChbmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    int k = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    bool nullAlpha = false;
    BlasFillMode a = parseFill("RANDOM_NORM_5_5");
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    aclblasComplex beta{0.0f, 0.0f};
    bool nullBeta = false;
    BlasFillMode y = parseFill("RANDOM_NORM_5_5");
    int incy = 1;

    std::string aFill;
    std::string xFill;
    std::string yFill;

    explicit ChbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "ACLBLAS_UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        const std::string alphaRealStr = ReadMap(map, "alpha_real", "1");
        const std::string betaRealStr = ReadMap(map, "beta_real", "0");
        nullAlpha = (alphaRealStr == "null" || alphaRealStr == "NULLPTR" || alphaRealStr == "nullptr");
        nullBeta = (betaRealStr == "null" || betaRealStr == "NULLPTR" || betaRealStr == "nullptr");
        alpha = {nullAlpha ? 0.0f : parseFloat(alphaRealStr), parseFloat(ReadMap(map, "alpha_imag", "0"))};
        aFill = ReadMap(map, "a_fill", ReadMap(map, "a", "RANDOM_NORM_5_5"));
        a = parseFill(aFill);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max<int64_t>(1, static_cast<int64_t>(k) + 1))));
        xFill = ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM_NORM_5_5"));
        x = parseFill(xFill);
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = {nullBeta ? 0.0f : parseFloat(betaRealStr), parseFloat(ReadMap(map, "beta_imag", "0"))};
        yFill = ReadMap(map, "y_fill", ReadMap(map, "y", "RANDOM_NORM_5_5"));
        y = parseFill(yFill);
        incy = parseInt(ReadMap(map, "incy", "1"));
    }

    bool IsPerformanceCase() const { return caseName.rfind("TC_PF", 0) == 0; }
};
