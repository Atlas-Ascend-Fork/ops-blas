/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <algorithm>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// Parameter struct for aclblasCher2k (complex<float> Hermitian rank-2k update).
// Column format follows test_cases/cher2k_test.csv (this directory is the format
// definition owner, see /workspace/test_cases/README.md):
//   case_name, description, uplo, trans, n, k,
//   alpha_real, alpha_imag, a_fill, lda, b_fill, ldb, beta, c_fill, ldc,
//   expect_result, nullA, nullB, nullC, mere_threshold, mare_multiplier, random_seed
// Differences vs CherkParam (design §6.4):
//   - alpha is complex  -> split CSV columns alpha_real / alpha_imag
//   - B matrix added    -> b_fill / ldb / nullB (rank-2k takes two operand matrices)
//   - nullBeta added    -> beta == "null" encodes a null device scalar pointer
//   - uplo/trans accept raw numeric strings ("999") which parseEnum turns into
//     an out-of-range enum value, exercising the INVALID_ENUM path (TC_ED_212/213)
struct Cher2kParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_1");
    int lda = 0;
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_1");
    int ldb = 0;
    float beta = 0.0f;
    BlasFillMode cFill = BlasFillMode("RANDOM_NORM_1");
    int ldc = 0;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    bool nullAlpha = false;
    bool nullBeta = false;

    explicit Cher2kParam(const csv_map& map) : BlasTestParamBase(map)
    {
        // parseEnum falls back to std::stoi for unknown tokens, so "999" becomes
        // fill mode 999 / operation 999 — out-of-range values for the API.
        uplo = parseFillMode(ReadMap(map, "uplo", "ACLBLAS_UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "ACLBLAS_OP_N"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        // Complex alpha: "null" in alpha_real encodes a null device pointer
        // (chemm_param.h convention). alpha_imag is ignored in that case.
        std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
        nullAlpha = (alphaRealStr == "null" || alphaRealStr == "nullptr");
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImagStr, 0.0f);

        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_1"));
        // Default lda/ldb (column-major), empty string -> compact default
        // (PF cases leave the columns empty):
        //   trans=N -> A/B are (n x k), ld >= max(1, n)
        //   trans=C -> A/B are (k x n), ld >= max(1, k)
        int defaultLd = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLd)), defaultLd);

        bFill = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_NORM_1"));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(defaultLd)), defaultLd);

        // Real beta: "null" encodes a null device pointer (ssyr2k_param.h convention).
        std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr");
        beta = nullBeta ? 0.0f : parseFloat(betaStr, 0.0f);

        cFill = BlasFillMode(ReadMap(map, "c_fill", "RANDOM_NORM_1"));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))), std::max(1, n));

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");
    }
};
