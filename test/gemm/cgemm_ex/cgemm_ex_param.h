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
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cgemm_ex_type.h"
#include "csv_loader.h"

inline aclblasCgemmExType_t parseAclblasType(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasCgemmExType_t> t = {{"C_32", ACLBLAS_C_32},
                                                                     {"ACLBLAS_C_32", ACLBLAS_C_32},
                                                                     {"R_32", ACLBLAS_R_32},
                                                                     {"ACLBLAS_R_32", ACLBLAS_R_32},
                                                                     {"H_R_32", ACLBLAS_H_R_32},
                                                                     {"ACLBLAS_H_R_32", ACLBLAS_H_R_32},
                                                                     {"H_C_32", ACLBLAS_H_C_32},
                                                                     {"ACLBLAS_H_C_32", ACLBLAS_H_C_32},
                                                                     {"INVALID", static_cast<aclblasCgemmExType_t>(0xFF)}};
    auto it = t.find(s);
    if (it != t.end()) {
        return it->second;
    }
    return static_cast<aclblasCgemmExType_t>(0xFF);
}

struct CgemmExParam : public BlasTestParamBase {
    std::string func = "cgemmex";
    aclblasCgemmExType_t typeA = ACLBLAS_C_32;
    aclblasCgemmExType_t typeB = ACLBLAS_C_32;
    aclblasCgemmExType_t typeC = ACLBLAS_C_32;
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

    CgemmExParam(const csv_map& map) : BlasTestParamBase(map)
    {
        func = ReadMap(map, "func", "cgemmex");
        typeA = parseAclblasType(ReadMap(map, "typeA", "C_32"));
        typeB = parseAclblasType(ReadMap(map, "typeB", "C_32"));
        typeC = parseAclblasType(ReadMap(map, "typeC", "C_32"));
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
        std::string ldaStr = ReadMap(map, "lda", "");
        lda = ldaStr.empty() ? std::max(1, physRowsA) : parseInt(ldaStr);
        int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
        std::string ldbStr = ReadMap(map, "ldb", "");
        ldb = ldbStr.empty() ? std::max(1, physRowsB) : parseInt(ldbStr);
        std::string ldcStr = ReadMap(map, "ldc", "");
        ldc = ldcStr.empty() ? std::max(1, m) : parseInt(ldcStr);
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM"));
        cFill = parseFill(ReadMap(map, "c_fill", "VALUE_NORM_0"));
        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull = (ReadMap(map, "beta_null", "false") == "true");
        aNull = (ReadMap(map, "a_null", "false") == "true");
        bNull = (ReadMap(map, "b_null", "false") == "true");
        cNull = (ReadMap(map, "c_null", "false") == "true");
    }

    bool isComplexC() const { return typeC == ACLBLAS_C_32 || typeC == ACLBLAS_H_C_32; }
};
