/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYR2K_PARAM_H
#define CSYR2K_PARAM_H

#include <algorithm>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct Csyr2kParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    int lda = 1;
    int ldb = 1;
    int ldc = 1;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    BlasFillMode fillA = BlasFillMode("RANDOM_NORM_5_5");
    BlasFillMode fillB = BlasFillMode("RANDOM_NORM_5_5");
    BlasFillMode fillC = BlasFillMode("RANDOM_NORM_5_5");

    explicit Csyr2kParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        std::string alphaReal = ReadMap(map, "alpha_real", "1");
        std::string alphaImag = ReadMap(map, "alpha_imag", "0");
        std::string betaReal = ReadMap(map, "beta_real", "0");
        std::string betaImag = ReadMap(map, "beta_imag", "0");
        nullAlpha = alphaReal == "null" || alphaReal == "nullptr" || alphaImag == "null" || alphaImag == "nullptr";
        nullBeta = betaReal == "null" || betaReal == "nullptr" || betaImag == "null" || betaImag == "nullptr";
        alpha = {nullAlpha ? 0.0f : parseFloat(alphaReal, 1.0f), nullAlpha ? 0.0f : parseFloat(alphaImag, 0.0f)};
        beta = {nullBeta ? 0.0f : parseFloat(betaReal, 0.0f), nullBeta ? 0.0f : parseFloat(betaImag, 0.0f)};

        int minLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        lda = parseInt(ReadMap(map, "lda", std::to_string(minLd)));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(minLd)));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));

        nullA = ReadMap(map, "nullA", "0") == "1";
        nullB = ReadMap(map, "nullB", "0") == "1";
        nullC = ReadMap(map, "nullC", "0") == "1";
        fillA = BlasFillMode(ReadMap(map, "fillA", "RANDOM_NORM_5_5"));
        fillB = BlasFillMode(ReadMap(map, "fillB", "RANDOM_NORM_5_5"));
        fillC = BlasFillMode(ReadMap(map, "fillC", "RANDOM_NORM_5_5"));
    }
};

#endif // CSYR2K_PARAM_H
