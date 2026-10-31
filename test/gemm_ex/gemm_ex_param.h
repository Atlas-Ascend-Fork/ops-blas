/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_EX_PARAM_H
#define GEMM_EX_PARAM_H

#include <string>
#include <algorithm>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// ── aclDataType parser ──
// Symbolic tokens only (issue #699): the former private numeric aliases
// {"0":ACL_FLOAT16, "2":ACL_FLOAT, ...} contradicted the real aclDataType
// values, and the unbounded std::stoi fallback let out-of-range integers
// reach the operator. Numeric input is now rejected: unknown tokens fail
// loudly and fall back to the historical default (existing CSVs use
// symbolic names only, so their parsing is unchanged).
inline aclDataType parseAclDataType(const std::string& s)
{
    static const std::unordered_map<std::string, aclDataType> m = {
        {"FP16", ACL_FLOAT16},
        {"ACL_FLOAT16", ACL_FLOAT16},
        {"BF16", ACL_BF16},
        {"ACL_BF16", ACL_BF16},
        {"FP32", ACL_FLOAT},
        {"ACL_FLOAT", ACL_FLOAT},
        {"INT8", ACL_INT8},
        {"ACL_INT8", ACL_INT8},
        {"INT32", ACL_INT32},
        {"ACL_INT32", ACL_INT32},
        {"FP8_E4M3", ACL_FLOAT8_E4M3FN},
        {"FP8_E4M3_FN", ACL_FLOAT8_E4M3FN},
        {"FP8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"ACL_FLOAT8_E4M3FN", ACL_FLOAT8_E4M3FN},
        {"FP8_E5M2", ACL_FLOAT8_E5M2},
        {"FP8_E5M2_FN", ACL_FLOAT8_E5M2},
        {"ACL_FLOAT8_E5M2", ACL_FLOAT8_E5M2},
    };
    auto it = m.find(s);
    if (it != m.end())
        return it->second;
    std::cout << "[gemm_ex_param] unknown dtype token '" << s
              << "', numeric input disabled (issue #699); defaulting to FP16" << std::endl;
    return ACL_FLOAT16;
}

// ── Physical dimensions based on transposition (column-major) ──
inline int physRows(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalRows : logicalCols;
}
inline int physCols(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalCols : logicalRows;
}

struct GemmParam : public BlasTestParamBase {
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
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
    BlasFillMode aFill = parseFill("RANDOM");
    BlasFillMode bFill = parseFill("RANDOM");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;

    GemmParam(const csv_map& map) : BlasTestParamBase(map)
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

        // Default lda: for column-major, ld >= physical rows of A
        int physRowsA = physRows(m, k, transA);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = physRows(k, n, transB);
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
    }
};

#endif // GEMM_EX_PARAM_H