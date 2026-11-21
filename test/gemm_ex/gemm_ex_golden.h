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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "dtype_cast.h"
#include "gemm_ex_param.h"
#include "cblas_compat.h"
#include "dtype_utils.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden helpers
// ═══════════════════════════════════════════════════════════════════════════════

// Validate GEMM parameters (handle, pointers, dimensions, leading dimensions)
inline aclblasStatus_t ValidateGoldenParams(
    aclblasHandle_t handle, const void* alpha, const void* beta, int m, int n, int k, aclblasOperation_t transA,
    aclblasOperation_t transB, int lda, int ldb, int ldc, const void* A, const void* B, const void* C, float betaVal)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
    int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
    if (lda < std::max(1, physRowsA))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, physRowsB))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;

    // issue #1091: alpha=0 时 A/B 可为空指针（与 NPU 侧校验口径一致）
    const float alphaVal = *static_cast<const float*>(alpha);
    if (k > 0 && alphaVal != 0.0f && (A == nullptr || B == nullptr))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (C == nullptr && betaVal != 0.0f)
        return ACLBLAS_STATUS_INVALID_VALUE;

    return ACLBLAS_STATUS_SUCCESS;
}

// Compute one element of C: alpha * op(A) * op(B) + beta * C[i,j]
inline double ComputeGemmElement(
    const float* aData, const float* bData, float cOrig, int i, int j, int k, int lda, int ldb, int ldc,
    aclblasOperation_t transA, aclblasOperation_t transB, float alphaVal, float betaVal)
{
    double sum = 0.0;
    if (k > 0) {
        for (int p = 0; p < k; p++) {
            float aVal = (transA == ACLBLAS_OP_N) ? aData[static_cast<size_t>(p) * lda + i] :
                                                    aData[static_cast<size_t>(i) * lda + p];
            float bVal = (transB == ACLBLAS_OP_N) ? bData[static_cast<size_t>(j) * ldb + p] :
                                                    bData[static_cast<size_t>(p) * ldb + j];
            sum += static_cast<double>(aVal) * static_cast<double>(bVal);
        }
    }
    if (betaVal == 0.0) {
        return static_cast<double>(alphaVal) * sum;
    }
    return static_cast<double>(alphaVal) * sum + static_cast<double>(betaVal) * cOrig;
}

// Quantize output through Ctype for fair comparison with NPU
inline float QuantizeGoldenOutput(float resFloat, aclDataType Atype, aclDataType Btype, aclDataType Ctype)
{
    // FP8 special value handling: NPU FP8 Cube unit produces NaN for
    // non-finite results (NaN/Inf) when inputs are FP8 types. The golden
    // computation in double precision may produce Inf where NPU produces NaN.
    // Align golden with NPU behavior for FP8 special value cases.
    if ((Atype == ACL_FLOAT8_E4M3FN || Atype == ACL_FLOAT8_E5M2 || Btype == ACL_FLOAT8_E4M3FN ||
         Btype == ACL_FLOAT8_E5M2) &&
        !std::isfinite(resFloat)) {
        resFloat = NAN;
    }

    switch (Ctype) {
        case ACL_FLOAT16:
            resFloat = blas_common::HalfToFloat(blas_common::FloatToHalf(resFloat));
            break;
        case ACL_BF16:
            resFloat = blas_common::Bf16ToFloat(blas_common::FloatToBf16(resFloat));
            break;
        case ACL_FLOAT8_E4M3FN:
            resFloat = fp8E4m3ToFloat(floatToFp8E4m3(resFloat));
            break;
        case ACL_FLOAT8_E5M2:
            resFloat = fp8E5m2ToFloat(floatToFp8E5m2(resFloat));
            break;
        default:
            break;
    }
    return resFloat;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden: C = alpha * op(A) * op(B) + beta * C
//
// Signature matches the BLAS API (column-major storage).
// A, B, C are passed as void* — the golden reads them as float arrays
// (pre-quantized through the target dtype for correct numerical values).
// Computation uses double precision. Output is quantized through Ctype.
// ═══════════════════════════════════════════════════════════════════════════════

inline aclblasStatus_t aclblasGemmEx_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb,
    const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType)
{
    (void)computeType;

    const float alphaVal = *static_cast<const float*>(alpha);
    const float betaVal = *static_cast<const float*>(beta);

    aclblasStatus_t status =
        ValidateGoldenParams(handle, alpha, beta, m, n, k, transA, transB, lda, ldb, ldc, A, B, C, betaVal);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const float* aData = static_cast<const float*>(A);
    const float* bData = static_cast<const float*>(B);
    float* cData = static_cast<float*>(C);

    if (cData == nullptr)
        return ACLBLAS_STATUS_SUCCESS;

    // issue #1091: alpha=0 不读取 A/B（README 契约），C=beta*C（A/B 可为空指针或含 NaN）
    if (alphaVal == 0.0f) {
        const size_t cCount = static_cast<size_t>(ldc) * n;
        for (size_t idx = 0; idx < cCount; idx++) {
            cData[idx] = (betaVal == 0.0f) ? 0.0f : QuantizeGoldenOutput(cData[idx] * betaVal, Atype, Btype, Ctype);
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_sgemm(
        CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alphaVal, aData, lda, bData, ldb, betaVal, cData,
        ldc);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            cData[static_cast<size_t>(j) * ldc + i] =
                QuantizeGoldenOutput(cData[static_cast<size_t>(j) * ldc + i], Atype, Btype, Ctype);
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}
