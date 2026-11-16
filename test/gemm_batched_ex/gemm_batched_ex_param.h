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

#include <iostream>
#include <string>
#include <algorithm>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "gemm_ex_param.h" // reuse parseAclDataType, physRows, physCols

// Parse aclblasGemmAlgo_t (batched-specific)
inline aclblasGemmAlgo_t parseGemmAlgo(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasGemmAlgo_t> m = {
        {"ACLBLAS_GEMM_DEFAULT", ACLBLAS_GEMM_DEFAULT},  {"DEFAULT", ACLBLAS_GEMM_DEFAULT}, {"0", ACLBLAS_GEMM_DEFAULT},
        {"ACLBLAS_GEMM_ALGO0", ACLBLAS_GEMM_ALGO0},      {"ALGO0", ACLBLAS_GEMM_ALGO0},     {"1", ACLBLAS_GEMM_ALGO0},
        {"ACLBLAS_GEMM_ALGO1", ACLBLAS_GEMM_ALGO1},      {"ALGO1", ACLBLAS_GEMM_ALGO1},     {"2", ACLBLAS_GEMM_ALGO1},
        {"INVALID", static_cast<aclblasGemmAlgo_t>(99)},
    };
    auto it = m.find(s);
    if (it != m.end())
        return it->second;
    try {
        return static_cast<aclblasGemmAlgo_t>(std::stoi(s));
    } catch (...) {
        // issue #918: 未知 algo 字符串静默回退 DEFAULT 会掩盖用例覆盖意图，
        // 对齐 csv_loader.h parseEnum 的"解析失败必须可见"口径输出告警
        std::cout << "[gemm_batched_ex] unknown algo string '" << s << "', using ACLBLAS_GEMM_DEFAULT" << std::endl;
        return ACLBLAS_GEMM_DEFAULT;
    }
}

struct GemmBatchedExParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    aclDataType Atype = ACL_FLOAT16;
    aclDataType Btype = ACL_FLOAT16;
    aclDataType Ctype = ACL_FLOAT16;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int batchCount = 1;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
    aclblasGemmAlgo_t algo = ACLBLAS_GEMM_DEFAULT;
    BlasFillMode aFill = parseFill("RANDOM");
    BlasFillMode bFill = parseFill("RANDOM");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aarrayNull = false;
    bool barrayNull = false;
    bool carrayNull = false;

    GemmBatchedExParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));

        Atype = parseAclDataType(ReadMap(map, "Atype", "FP16"));
        Btype = parseAclDataType(ReadMap(map, "Btype", "FP16"));
        Ctype = parseAclDataType(ReadMap(map, "Ctype", "FP16"));

        computeType = parseComputeType(ReadMap(map, "computeType", "COMPUTE_32F"));
        algo = parseGemmAlgo(ReadMap(map, "algo", "DEFAULT"));

        // Default lda: for column-major, ld >= physical rows of A
        int physRowsA = physRows(m, k, transA);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = physRows(k, n, transB);
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        batchCount = parseInt(ReadMap(map, "batchCount", "1"));

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM"));
        cFill = parseFill(ReadMap(map, "c_fill", "VALUE_NORM_0"));

        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull = (ReadMap(map, "beta_null", "false") == "true");
        aarrayNull = (ReadMap(map, "aarray_null", "false") == "true");
        barrayNull = (ReadMap(map, "barray_null", "false") == "true");
        carrayNull = (ReadMap(map, "carray_null", "false") == "true");
    }
};
