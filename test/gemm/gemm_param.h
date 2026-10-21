/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_PARAM_H
#define GEMM_PARAM_H

#include <complex>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct GemmParam : public BlasTestParamBase {
    std::string func = "sgemm";
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM");
    BlasFillMode bFill = parseFill("RANDOM");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;
    bool measure = false;
    int warmup = 30;
    int measurements = 50;
    std::string classification;
    double flops = 0.0;
    bool profile = false;

    GemmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        func = ReadMap(map, "func", "sgemm");
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alphaReal = parseFloat(ReadMap(map, "alpha_real", "1.0"));
        alphaImag = parseFloat(ReadMap(map, "alpha_imag", "0.0"));
        betaReal = parseFloat(ReadMap(map, "beta_real", "0.0"));
        betaImag = parseFloat(ReadMap(map, "beta_imag", "0.0"));

        int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM"));
        cFill = parseFill(ReadMap(map, "c_fill", "VALUE_NORM_0"));

        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull = (ReadMap(map, "beta_null", "false") == "true");
        aNull = (ReadMap(map, "a_null", "false") == "true");
        bNull = (ReadMap(map, "b_null", "false") == "true");
        cNull = (ReadMap(map, "c_null", "false") == "true");
        measure = parseInt(ReadMap(map, "measure", "0")) != 0;
        warmup = parseInt(ReadMap(map, "warmup", "30"));
        measurements = parseInt(ReadMap(map, "measurements", "50"));
        classification = ReadMap(map, "classification", "");
        flops = parseDouble(ReadMap(map, "flops", "0"));
        profile = parseInt(ReadMap(map, "profile", "0")) != 0;
    }

    bool isComplex() const { return func == "cgemm"; }
};

#endif // GEMM_PARAM_H
