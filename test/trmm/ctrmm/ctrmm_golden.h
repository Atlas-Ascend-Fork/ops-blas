/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <securec.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline CBLAS_SIDE ToCblasSide(aclblasSideMode_t side) { return (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight; }

// CPU golden for aclblasCtrmm. Mirrors the NPU host's decision order so the
// negative cases agree one-for-one, then delegates to Netlib cblas_ctrmm.
// cblas_ctrmm is the BLAS in-place form (B := alpha*op(A)*B on the last
// argument). The out-of-place C of this task is served by first copying the
// active m rows of each B column into C (padding rows keep C's own contents,
// matching the NPU kernel which never writes rows m..ldc-1) and then calling
// cblas_ctrmm with C in the B role.
/**
 * @brief alpha == (0,0) fast path: copy B, then zero the m active rows.
 */
/**
 * @brief Per-column B to C copy helper (byte-identical to the inline blocks).
 */
static aclblasStatus_t CtrmmCpuCopyB2C(aclblasComplex* C, const aclblasComplex* B, int m, int n, int ldb, int ldc)
{
    const size_t colBytes = static_cast<size_t>(m) * sizeof(aclblasComplex);
    if (ldc == ldb) {
        size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
        if (memcpy_s(C, cBytes, B, cBytes) != EOK) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
        for (int j = 0; j < n; ++j) {
            size_t dstOffset = static_cast<size_t>(j) * static_cast<size_t>(ldc) * sizeof(aclblasComplex);
            size_t dstRemain = cBytes - dstOffset;
            const void* src = reinterpret_cast<const uint8_t*>(B) +
                              static_cast<size_t>(j) * static_cast<size_t>(ldb) * sizeof(aclblasComplex);
            if (memcpy_s(reinterpret_cast<uint8_t*>(C) + dstOffset, dstRemain, src, colBytes) != EOK) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// alpha == (0,0) fast path: B is copied into C first (so the padding rows
// carry B's padding, matching the NPU wrapper's B->C pre-copy), then the m
// active rows per column are zeroed. A/B values are not referenced.
static aclblasStatus_t CtrmmCpuAlphaZero(aclblasComplex* C, const aclblasComplex* B, int m, int n, int ldb, int ldc)
{
    if (B != nullptr) {
        aclblasStatus_t st = CtrmmCpuCopyB2C(C, B, m, n, ldb, ldc);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    const size_t colBytes = static_cast<size_t>(m) * sizeof(aclblasComplex);
    const size_t colStride = static_cast<size_t>(ldc) * sizeof(aclblasComplex);
    for (int j = 0; j < n; ++j) {
        void* colPtr = reinterpret_cast<uint8_t*>(C) + static_cast<size_t>(j) * colStride;
        if (memset_s(colPtr, colBytes, 0, colBytes) != EOK) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCtrmm_cpu(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // Enum validation BEFORE the m==0/n==0 quick return, matching the host's
    // order (handle -> enums -> bounds -> no-op): an m==0 call with an illegal
    // enum must be INVALID_VALUE, not SUCCESS.
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (alpha == nullptr || C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const float alphaRe = alpha->real;
    const float alphaIm = alpha->imag;

    // alpha == (0,0): C active region zeroed; A/B not referenced as values,
    // but B is still copied into C first (when available) so the padding rows
    // carry B's padding exactly like the NPU wrapper's B->C pre-copy — the
    // zeroing itself only touches the m active rows per column on both sides.
    if (alphaRe == 0.0f && alphaIm == 0.0f) {
        return CtrmmCpuAlphaZero(C, B, m, n, ldb, ldc);
    }

    if (A == nullptr || B == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Column-major B -> C: per-column copy of the m active rows.
    aclblasStatus_t cpSt = CtrmmCpuCopyB2C(C, B, m, n, ldb, ldc);
    if (cpSt != ACLBLAS_STATUS_SUCCESS) {
        return cpSt;
    }

    // aclblasComplex {float real; float imag} is binary-compatible with the
    // OpenBLAS complex layout; the void* casts are pure reinterprets.
    const float alphaCblas[2] = {alphaRe, alphaIm};
    cblas_ctrmm(
        CblasColMajor, ToCblasSide(side), ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), m, n,
        static_cast<const void*>(alphaCblas), static_cast<const void*>(A), lda, static_cast<void*>(C), ldc);
    return ACLBLAS_STATUS_SUCCESS;
}
