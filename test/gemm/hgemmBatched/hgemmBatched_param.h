/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HGEMM_BATCHED_PARAM_H
#define HGEMM_BATCHED_PARAM_H

#include <algorithm>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// Physical (storage) leading dimension for a logical matrix stored column-major.
// op(A) is m x k; A itself is m x k (N) or k x m (T/C).
inline int hgemmBatchedPhysRows(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalRows : logicalCols;
}

inline int hgemmBatchedPhysCols(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalCols : logicalRows;
}

#define HGEMM_BATCHED_SIGNATURE(name) \
    aclblasStatus_t name( \
        aclblasHandle_t handle, \
        aclblasOperation_t transA, aclblasOperation_t transB, \
        int m, int n, int k, \
        const aclblasHalf* alpha, \
        const aclblasHalf* const Aarray[], int lda, \
        const aclblasHalf* const Barray[], int ldb, \
        const aclblasHalf* beta, \
        aclblasHalf* const Carray[], int ldc, \
        int batchCount)

// Parse a transa/transb token.  Accepts both the full enum names and the
// single-letter short forms; any other numeric token (e.g. "999") maps to an
// invalid enum value so the op can be exercised for ACLBLAS_STATUS_INVALID_ENUM.
inline aclblasOperation_t hgemmParseOpTrans(const std::string& s)
{
    if (s == "ACLBLAS_OP_N" || s == "N") {
        return ACLBLAS_OP_N;
    }
    if (s == "ACLBLAS_OP_T" || s == "T") {
        return ACLBLAS_OP_T;
    }
    if (s == "ACLBLAS_OP_C" || s == "C") {
        return ACLBLAS_OP_C;
    }
    // Unknown / numeric-out-of-range token: keep it as a non-enum value.
    return static_cast<aclblasOperation_t>(0xFF);
}

// FLOAT16 scalar column: "null" means a null pointer, otherwise a numeric value
// that is rounded to FLOAT16.
struct HgemmBatchedScalar {
    bool isNull = false;
    float value = 0.0f;
};

inline HgemmBatchedScalar hgemmParseScalar(const std::string& s, float def)
{
    if (s == "null" || s == "NULL" || s == "nullptr") {
        return {true, 0.0f};
    }
    if (s.empty()) {
        return {false, def};
    }
    return {false, parseFloat(s, def)};
}

struct HgemmBatchedParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    HgemmBatchedScalar alpha;
    HgemmBatchedScalar beta;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int batchCount = 1;
    BlasFillMode aFill = parseFill("RANDOM");
    BlasFillMode bFill = parseFill("RANDOM");
    BlasFillMode cFill = parseFill("RANDOM");
    bool alphaNull = false;
    bool betaNull = false;
    bool aarrayNull = false;
    bool barrayNull = false;
    bool carrayNull = false;

    HgemmBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = hgemmParseOpTrans(ReadMap(map, "transA", "N"));
        transB = hgemmParseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));

        alpha = hgemmParseScalar(ReadMap(map, "alpha", "1"), 1.0f);
        beta = hgemmParseScalar(ReadMap(map, "beta", "0"), 0.0f);
        alphaNull = alpha.isNull;
        betaNull = beta.isNull;

        int physRowsA = hgemmBatchedPhysRows(m, k, transA);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = hgemmBatchedPhysRows(k, n, transB);
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        batchCount = parseInt(ReadMap(map, "batchCount", "1"));

        aFill = parseFill(ReadMap(map, "aarray_fill", "RANDOM"));
        bFill = parseFill(ReadMap(map, "barray_fill", "RANDOM"));
        cFill = parseFill(ReadMap(map, "carray_fill", "RANDOM"));

        aarrayNull = (aFill.method == BlasFillMode::M_NULLPTR);
        barrayNull = (bFill.method == BlasFillMode::M_NULLPTR);
        carrayNull = (cFill.method == BlasFillMode::M_NULLPTR);
    }
};

#endif // HGEMM_BATCHED_PARAM_H
