/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2k_host.cpp
 * \brief arch22 (Atlas A2/A3) host implementation of aclblasCsyr2k.
 *
 *        C = alpha*A*B^T + alpha*B*A^T + beta*C   (trans = 'N', A and B are n x k)
 *        C = alpha*A^T*B + alpha*B^T*A + beta*C   (trans = 'T', A and B are k x n)
 *        alpha and beta are complex; C is symmetric and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Three-stage pipeline (see csyr2k_tiling_data.h for the layout contract):
 *          Phase 0 (AIV) split   : A -> (Ar, Ai), B -> (Br, Bi), each packed
 *                                  column-major with the stored row count as its
 *                                  row stride, so the shared Gram-product GEMM
 *                                  can address both operands through one packedLd
 *          Phase 1 (AIC) gemm    : the three real products T1, T2, T3, eight
 *                                  launches in total, K-split at the kernel's
 *                                  compile-time singleK
 *          Phase 2 (AIV) combine : C = alpha*((T1-T2) + i*T3) + beta*C
 *
 *        CSYRK can fold its sign pattern into K-concatenated operands and so
 *        needs only two GEMMs; with two independent operands that trick does not
 *        carry over, which is why this operator keeps three temp planes and
 *        subtracts in Phase 2 instead.
 */

#include <algorithm>
#include <cstdint>

#include "securec.h"
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "common/helper/host_utils.h"
#include "csyr2k_kernel.h"
#include "csyr2k_tiling_data.h"

namespace {
constexpr const char* OP_TAG = "aclblasCsyr2k";
constexpr uint32_t CSYR2K_TEMP_COUNT = 3;   // T1, T2, T3
constexpr uint32_t CSYR2K_PACKED_COUNT = 4; // Ar, Ai, Br, Bi

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }

inline bool IsComplexOne(const aclblasComplex& v) { return v.real == 1.0F && v.imag == 0.0F; }

// The eight real products, as (left plane, right plane, temp plane) index triples
// into the packed and temp buffer sets. trans only decides which GEMM
// specialisation runs, never this list: for trans='T' every product reads the
// transposed operands instead, and A^T*B + B^T*A expands to the same eight terms.
struct Csyr2kProduct {
    uint32_t left;  // index into the packed planes (Ar, Ai, Br, Bi)
    uint32_t right; // index into the packed planes
    uint32_t temp;  // index into the temp planes (T1, T2, T3)
};

static_assert(CSYR2K_TEMP_COUNT == 3U, "Phase 2 reads exactly three temp planes");

// Review #1169 (wangzitao_leo): the third field is the *destination temp plane*,
// not another packed-plane index. Getting it wrong for T2 made T1 and T2 share
// temp[0] (so temp[0] held T1+T2 and temp[1] was never written), which Phase 2
// then read as an uninitialised buffer -- trans='N' and trans='T'/'C' were both
// wrong. The three temp indices must therefore be distinct, one per product.
constexpr Csyr2kProduct CSYR2K_PRODUCTS[8] = {
    {0U, 2U, 0U}, {2U, 0U, 0U}, // T1 -> temp[0] = Ar*Br^T + Br*Ar^T
    {1U, 3U, 1U}, {3U, 1U, 1U}, // T2 -> temp[1] = Ai*Bi^T + Bi*Ai^T
    {0U, 3U, 2U}, {1U, 2U, 2U}, // T3 -> temp[2] = Ar*Bi^T + Ai*Br^T
    {2U, 1U, 2U}, {3U, 0U, 2U}, //              + Br*Ai^T + Bi*Ar^T
};

// ============================================================================
//  Parameter validation
// ============================================================================
// Enum legality and leading dimensions. Must run *before* the n == 0 quick
// return: an illegal uplo/trans is an error regardless of the problem size, which
// is how reference BLAS and the shared CPU golden both behave.
static aclblasStatus_t ValidateCsyr2kEnumsAndDims(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(OP_TAG, "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    // CSYR2K takes the plain transpose; OP_C is accepted and treated as OP_T,
    // because C is symmetric rather than Hermitian so the conjugate is a no-op.
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE(OP_TAG, "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0) {
        OP_LOGE(OP_TAG, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k < 0) {
        OP_LOGE(OP_TAG, "k must be >= 0, got %d", k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int ldMin = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < ldMin) {
        OP_LOGE(OP_TAG, "lda=%d must be >= %d for trans=%d", lda, ldMin, static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < ldMin) {
        OP_LOGE(OP_TAG, "ldb=%d must be >= %d for trans=%d", ldb, ldMin, static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE(OP_TAG, "ldc=%d must be >= max(1,n)=%d", ldc, std::max(1, n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Pointer checks: alpha, beta, and C must never be nullptr regardless of n.
// When n > 0 and k > 0, A and B must also be non-null.
static aclblasStatus_t ValidateCsyr2kPointers(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B,
    const aclblasComplex* beta, const aclblasComplex* C)
{
    if (alpha == nullptr) {
        OP_LOGE(OP_TAG, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(OP_TAG, "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (C == nullptr) {
        OP_LOGE(OP_TAG, "C must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr && n > 0 && k > 0) {
        OP_LOGE(OP_TAG, "A must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (B == nullptr && n > 0 && k > 0) {
        OP_LOGE(OP_TAG, "B must not be nullptr when n>0 and k>0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Phase 0: complex -> packed real planes
// ============================================================================
// One split per operand. The packed row stride is the stored row count, not a
// padded value: the shared Gram-product GEMM derives the transposed operand's row
// stride from the output's M extent, so padding it here would silently
// mis-compute every transposed access.
static void RunCsyr2kSplits(
    void* stream, const CBlas3RankKShape& shape, uint32_t lda, uint32_t ldb, const aclblasComplex* A,
    const aclblasComplex* B, const CBlas3Workspace& ws, uint32_t aivCoreNum)
{
    const uint32_t splitBlocks = std::min(aivCoreNum, std::max(1U, shape.aCols));
    CBlas3SplitTilingData split{};
    split.rows = shape.aRows;
    split.cols = shape.aCols;
    split.packedLd = shape.aRows;

    split.lda = lda;
    csyr2k_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.packed[0], ws.packed[1],
        split);
    split.lda = ldb;
    csyr2k_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)), ws.packed[2], ws.packed[3],
        split);
}

// ============================================================================
//  Phase 1: the eight real products
// ============================================================================
// The first write into a temp plane overwrites it; every later write into the
// same plane accumulates atomically. Exceeding singleK is not diagnosed at
// runtime -- it silently mis-computes -- so the chunk size comes from the kernel
// itself.
static void RunCsyr2kGemms(
    void* stream, aclblasFillMode_t uplo, const CBlas3RankKShape& shape, const CBlas3Workspace& ws, uint32_t aicCoreNum)
{
    const uint32_t kChunk = csyr2k_gemm_single_k();
    CBlas3GemmTilingData gemm = CBlas3MakeRankKGemmTiling(shape, ws.tempLdc, static_cast<uint32_t>(uplo));
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    bool firstWrite[CSYR2K_TEMP_COUNT] = {true, true, true};

    for (uint32_t kBase = 0U; kBase < shape.k; kBase += kChunk) {
        gemm.kBase = kBase;
        gemm.kCount = std::min(kChunk, shape.k - kBase);
        for (const Csyr2kProduct& product : CSYR2K_PRODUCTS) {
            gemm.enAtomic = firstWrite[product.temp] ? 0U : 1U;
            firstWrite[product.temp] = false;
            csyr2k_gemm_kernel_do(
                gemmBlocks, stream, ws.packed[product.left], ws.packed[product.right], ws.temp[product.temp], gemm);
        }
    }
    OP_LOGI(
        OP_TAG, "phase1 gemm: n=%u k=%u packedLd=%u aicBlocks=%u kChunk=%u", shape.n, shape.k, shape.aRows, gemmBlocks,
        kChunk);
}

// ============================================================================
//  Phase 2: complex alpha/beta scaling of the three products
// ============================================================================
static void RunCsyr2kCombine(
    void* stream, aclblasFillMode_t uplo, uint32_t n, uint32_t ldc, aclblasComplex* C, const CBlas3Workspace& ws,
    bool hasTemp, const aclblasComplex& alphaVal, const aclblasComplex& betaVal, uint32_t aivCoreNum)
{
    Csyr2kCombineTilingData combine{};
    combine.n = n;
    combine.ldc = ldc;
    combine.tempLd = ws.tempLdc;
    combine.uploMode = static_cast<uint32_t>(uplo);
    combine.hasTemp = hasTemp ? 1U : 0U;
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaRe = betaVal.real;
    combine.betaIm = betaVal.imag;
    const uint32_t combineBlocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase2 combine: n=%u aivBlocks=%u hasTemp=%d", n, combineBlocks, static_cast<int>(hasTemp));
    csyr2k_combine_kernel_do(
        combineBlocks, stream, ws.temp[0], ws.temp[1], ws.temp[2], reinterpret_cast<uint8_t*>(C), combine);
}

static aclblasStatus_t ReadCsyr2kScalars(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclblasComplex& alphaVal, aclblasComplex& betaVal,
    aclrtStream stream)
{
    bool alphaDev = false;
    aclblasStatus_t status = CheckPtrLocation(alpha, &alphaDev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "failed to locate alpha, status=%d", static_cast<int>(status));
        return status;
    }
    bool betaDev = false;
    status = CheckPtrLocation(beta, &betaDev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "failed to locate beta, status=%d", static_cast<int>(status));
        return status;
    }

    if (!alphaDev) {
        if (memcpy_s(&alphaVal, sizeof(alphaVal), alpha, sizeof(*alpha)) != EOK) {
            OP_LOGE(OP_TAG, "copy host alpha failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        const aclError ret = aclrtMemcpyAsync(
            &alphaVal, sizeof(alphaVal), alpha, sizeof(*alpha), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_TAG, "aclrtMemcpyAsync alpha D2H failed, ret=%d", static_cast<int>(ret));
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (!betaDev) {
        if (memcpy_s(&betaVal, sizeof(betaVal), beta, sizeof(*beta)) != EOK) {
            OP_LOGE(OP_TAG, "copy host beta failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        const aclError ret = aclrtMemcpyAsync(
            &betaVal, sizeof(betaVal), beta, sizeof(*beta), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_TAG, "aclrtMemcpyAsync beta D2H failed, ret=%d", static_cast<int>(ret));
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (alphaDev || betaDev) {
        const aclError ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(OP_TAG, "aclrtSynchronizeStream for D2H failed, ret=%d", static_cast<int>(ret));
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchCsyr2kKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, uint32_t ldc)
{
    void* stream = h->stream;

    aclblasComplex alphaVal{0.0F, 0.0F};
    aclblasComplex betaVal{0.0F, 0.0F};
    const aclblasStatus_t readRet = ReadCsyr2kScalars(alpha, beta, alphaVal, betaVal, stream);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    const bool hasTemp = !IsComplexZero(alphaVal) && (k != 0U);
    OP_LOGD(
        OP_TAG, "alpha=(%f,%f) beta=(%f,%f) hasTemp=%d", static_cast<double>(alphaVal.real),
        static_cast<double>(alphaVal.imag), static_cast<double>(betaVal.real), static_cast<double>(betaVal.imag),
        static_cast<int>(hasTemp));

    // Nothing contributes from A and B and C keeps its value: no kernel at all.
    if (!hasTemp && IsComplexOne(betaVal)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const CBlas3RankKShape shape = CBlas3ResolveRankKShape(trans != ACLBLAS_OP_N, n, k);
    CBlas3Workspace ws{};
    const aclblasStatus_t wsRet =
        CBlas3PrepareWorkspace(h, shape, CSYR2K_TEMP_COUNT, CSYR2K_PACKED_COUNT, !hasTemp, ws, OP_TAG);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    const uint32_t aicCoreNum = GetAicCoreCount();
    if (aivCoreNum == 0U || aicCoreNum == 0U) {
        OP_LOGE(OP_TAG, "core count query failed: aic=%u aiv=%u", aicCoreNum, aivCoreNum);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (hasTemp) {
        RunCsyr2kSplits(stream, shape, lda, ldb, A, B, ws, aivCoreNum);
        RunCsyr2kGemms(stream, uplo, shape, ws, aicCoreNum);
    }

    RunCsyr2kCombine(stream, uplo, n, ldc, C, ws, hasTemp, alphaVal, betaVal, aivCoreNum);
    const aclError err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    if (err != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "kernel launch failed, err=%d", static_cast<int>(err));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCsyr2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t enumSt = ValidateCsyr2kEnumsAndDims(uplo, trans, n, k, lda, ldb, ldc);
    if (enumSt != ACLBLAS_STATUS_SUCCESS) {
        return enumSt;
    }
    const aclblasStatus_t ptrSt = ValidateCsyr2kPointers(n, k, alpha, A, B, beta, C);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCsyr2kKernel(
        static_cast<_aclblas_handle*>(handle), uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha,
        A, static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), beta, C, static_cast<uint32_t>(ldc));
}
