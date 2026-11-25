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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// Parameter loader for the hgemm FLOAT16 CSV cases:
//   case_name,description,transA,transB,m,n,k,alpha,a_fill,lda,b_fill,ldb,beta,c_fill,ldc,
//   expect_result,mere_threshold,mare_multiplier,random_seed
// Scalar columns accept "null" as a null pointer probe and tensor fill
// columns accept "NULLPTR" as a null device pointer probe. Leading
// dimensions left empty default to the minimum legal value max(1, physical
// rows). transA/transB accept ACLBLAS_OP_N/T/C names; unknown numeric
// values (e.g. 999) pass through so the negative cases hit the
// INVALID_ENUM validation branch.
struct HgemmParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alphaVal = 1.0f;
    float betaVal = 0.0f;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode bFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode cFill = parseFill("RANDOM_NORM_5_5");
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;
    bool isPerf = false;
    int warmup = 5;
    int measurements = 50;
    std::string classification;

    HgemmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        std::string alphaStr = ReadMap(map, "alpha", "1");
        alphaNull = (alphaStr == "null");
        alphaVal = alphaNull ? 0.0f : parseFloat(alphaStr, 1.0f);
        std::string betaStr = ReadMap(map, "beta", "0");
        betaNull = (betaStr == "null");
        betaVal = betaNull ? 0.0f : parseFloat(betaStr, 0.0f);

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));
        cFill = parseFill(ReadMap(map, "c_fill", "RANDOM_NORM_5_5"));
        aNull = (aFill.method == BlasFillMode::M_NULLPTR);
        bNull = (bFill.method == BlasFillMode::M_NULLPTR);
        cNull = (cFill.method == BlasFillMode::M_NULLPTR);

        int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        warmup = parseInt(ReadMap(map, "warmup", "5"));
        measurements = parseInt(ReadMap(map, "measurements", "50"));
        isPerf = (caseName.rfind("TC_PF_", 0) == 0);
        classification = ParseClassification(caseName);
    }

private:
    static std::string ParseClassification(const std::string& caseName)
    {
        // TC_<class>_<seq> - class token between the first and second underscore
        size_t first = caseName.find('_');
        if (first == std::string::npos) {
            return "";
        }
        size_t second = caseName.find('_', first + 1);
        if (second == std::string::npos) {
            return "";
        }
        return caseName.substr(first + 1, second - first - 1);
    }
};
