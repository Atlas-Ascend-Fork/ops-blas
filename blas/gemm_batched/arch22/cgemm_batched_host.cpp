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
 * \file cgemm_batched_host.cpp
 * \brief arch22 (Atlas A2/A3) host implementation of aclblasCgemmBatched.
 *
 *        C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i] over complex64 batch
 *        members addressed through device pointer arrays.
 *
 *        Pipeline (see cgemm_batched_tiling_data.h for the layout contract):
 *          Phase 0 (AIV) split  : A[i] -> (Ar, Ai), B[i] -> (Br, Bi)
 *          Phase 1 (AIC) gemm   : T1 = Ar*Br, T2 = Ai*Bi, T3 = Ar*Bi, T4 = Ai*Br
 *          Phase 2 (AIV) combine: C[i] = alpha * ((T1-T2) + j*(T3+T4)) + beta*C[i]
 *
 *        op() never costs a data pass: the packed planes are read plain or
 *        transposed by the Matmul depending on transa / transb, and the
 *        conjugation of op=C happens inside the split. Batches whose planes do
 *        not fit the workspace cap run in multiple stream-ordered rounds.
 */

#include <algorithm>
#include <cstdint>
#include <cmath>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "common/helper/host_utils.h"
#include "cgemm_batched_tiling_data.h"
#include "cgemm_batched_kernel.h"

namespace {
// Compile call tracing explicitly when diagnosing host dispatch. Error and
// warning logs remain available independently of this verbose trace.
#ifdef CGEMM_BATCHED_ENABLE_TRACE
#define CGBT_TRACE(...) OP_LOGD(__VA_ARGS__)
#else
#define CGBT_TRACE(...) ((void)0)
#endif

uint32_t CgbtAicCoreCount(_aclblas_handle* h)
{
    if (h->cgemm_aic_cores == 0U)
        h->cgemm_aic_cores = GetAicCoreCount();
    return h->cgemm_aic_cores;
}

uint32_t CgbtAivCoreCount(_aclblas_handle* h)
{
    if (h->cgemm_aiv_cores == 0U)
        h->cgemm_aiv_cores = GetAivCoreCount();
    return h->cgemm_aiv_cores;
}

constexpr const char* OP_TAG = "aclblasCgemmBatched";

// Batches whose planes exceed this cap run in multiple stream-ordered rounds
// over the same workspace; the cap keeps a single call far away from the
// library's 2GiB workspace limit while keeping large batch counts single-round.
constexpr uint64_t CGBT_WS_CAP_BYTES = 1536ULL * 1024ULL * 1024ULL;

// Validated dispatch cutoffs, in matrix elements. These select kernel
// specializations and cache-friendly workspace layouts, not API limits.
constexpr uint32_t CGBT_SMALL_PANEL_MAX = 64U;
constexpr uint32_t CGBT_WIDE_ROW_MAX = 144U;
constexpr uint32_t CGBT_MID_TILE = 176U;
constexpr uint32_t CGBT_REUSE_TILE = 256U;
constexpr uint32_t CGBT_LARGE_REUSE_TILE = 320U;
constexpr uint32_t CGBT_DIRECT_OUTPUT_MAX = 4096U;
constexpr uint32_t CGBT_WORKSPACE_ROW_ALIGN = 16U;
constexpr uint32_t CGBT_WORKSPACE_WIDE_ROW_ALIGN = 32U;

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }

// Physical (stored) shape of an operand. op(X) is logicalRows x logicalCols;
// trans = N stores it as logicalRows x logicalCols, trans = T/C as the
// transpose.
struct CgbtPhysShape {
    uint32_t rows;
    uint32_t cols;
};

inline CgbtPhysShape CgbtResolvePhys(bool transposed, uint32_t logicalRows, uint32_t logicalCols)
{
    return transposed ? CgbtPhysShape{logicalCols, logicalRows} : CgbtPhysShape{logicalRows, logicalCols};
}

// Per-round workspace geometry, all in floats unless noted.
struct CgbtWsPlan {
    uint32_t ldD;             // D row stride (padded m of C), floats
    uint64_t xStride;         // packed B plane stride, 512B aligned
    uint64_t yStride;         // packed A plane stride, 512B aligned
    uint64_t tStride;         // temp plane stride, 512B aligned
    uint64_t perBatchFloats;  // all eight planes of one batch
    uint32_t batchChunk = 1U; // batches per round
};

// Plane-major storage: each plane contains batchChunk matrices. Offsets below
// describe one batch and must be multiplied by the current round batch count.
inline uint64_t CgbtOffT1(const CgbtWsPlan& plan) { return 2U * plan.xStride + 2U * plan.yStride; }

// ============================================================================
//  Validation (same checks and order as the arch35 implementation)
// ============================================================================
aclblasStatus_t ValidateCgemmBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb,
    int ldc, int batchCount, const aclblasComplex* alpha, const aclblasComplex* beta,
    const aclblasComplex* const Aarray[], const aclblasComplex* const Barray[], aclblasComplex* const Carray[])
{
    CHECK_RET(handle != nullptr, OP_LOGE(OP_TAG, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE(OP_TAG, "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(batchCount >= 0, OP_LOGE(OP_TAG, "batchCount must be >= 0, got %d", batchCount);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "invalid transa=%d", static_cast<int>(transa));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "invalid transb=%d", static_cast<int>(transb));
              return ACLBLAS_STATUS_INVALID_VALUE);

    const bool isTransA = (transa != ACLBLAS_OP_N);
    const bool isTransB = (transb != ACLBLAS_OP_N);
    const int physRowsA = isTransA ? k : m;
    const int physRowsB = isTransB ? n : k;
    CHECK_RET(lda >= std::max(1, physRowsA),
              OP_LOGE(OP_TAG, "invalid lda=%d, expected>=%d", lda, std::max(1, physRowsA));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, physRowsB),
              OP_LOGE(OP_TAG, "invalid ldb=%d, expected>=%d", ldb, std::max(1, physRowsB));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m), OP_LOGE(OP_TAG, "invalid ldc=%d, expected>=%d", ldc, std::max(1, m));
              return ACLBLAS_STATUS_INVALID_VALUE);

    if (batchCount > 0) {
        CHECK_RET(Aarray != nullptr, OP_LOGE(OP_TAG, "Aarray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Barray != nullptr, OP_LOGE(OP_TAG, "Barray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(Carray != nullptr, OP_LOGE(OP_TAG, "Carray must not be nullptr");
                  return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Workspace planning
// ============================================================================
// A 128-byte row boundary avoids alternating partial cache-line transfers
// for medium FP32 workspace planes, while retaining the original BLAS strides.
inline uint32_t CgbtWorkspaceLd(uint32_t rows)
{
    return CeilAlign<uint32_t>(
        rows, rows > CGBT_SMALL_PANEL_MAX && rows <= CGBT_WIDE_ROW_MAX ? CGBT_WORKSPACE_WIDE_ROW_ALIGN :
                                                                         CGBT_WORKSPACE_ROW_ALIGN);
}

// Reserve zero-filled trailing columns for medium transposed operands.
inline uint32_t CgbtWorkspaceCols(uint32_t rows, uint32_t cols)
{
    return rows > CGBT_SMALL_PANEL_MAX && rows <= CGBT_MID_TILE && rows != CgbtWorkspaceLd(rows) &&
                   cols > CGBT_SMALL_PANEL_MAX && cols <= CGBT_MID_TILE ?
               CgbtWorkspaceLd(cols) :
               cols;
}

aclblasStatus_t CgbtAllocateWorkspace(_aclblas_handle* h, uint64_t perBatchBytes, int batchCount, CgbtWsPlan& plan)
{
    const uint64_t capacity = h->workspace_owner == AclblasWorkspaceOwner::User ?
                                  std::min<uint64_t>(CGBT_WS_CAP_BYTES, h->workspace_size) :
                                  CGBT_WS_CAP_BYTES;
    if (h->workspace_owner == AclblasWorkspaceOwner::User && capacity < perBatchBytes) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    plan.batchChunk = static_cast<uint32_t>(batchCount);
    // Chunk when the whole batch does not fit the cap -- and also when the
    // un-chunked total itself would wrap (perBatchBytes is already checked to
    // fit uint64, but perBatchBytes * batchCount is a second multiply that
    // must not be done unchecked).
    uint64_t totalBytes = 0U;
    const bool totalOk = CBlas3TryMulU64(perBatchBytes, static_cast<uint64_t>(batchCount), &totalBytes);
    if (!totalOk || totalBytes > capacity) {
        plan.batchChunk =
            std::max<uint32_t>(1U, static_cast<uint32_t>(capacity / std::max<uint64_t>(perBatchBytes, 1U)));
        CGBT_TRACE(
            OP_TAG, "batch chunked: batchCount=%d chunk=%u perBatchBytes=%llu", batchCount, plan.batchChunk,
            static_cast<unsigned long long>(perBatchBytes));
    }
    uint64_t need = 0U;
    if (!CBlas3TryMulU64(perBatchBytes, plan.batchChunk, &need)) {
        // Unreachable by construction (chunk <= cap / perBatchBytes), kept so
        // the size handed to the allocator is always a checked product.
        OP_LOGE(OP_TAG, "workspace allocation size overflows 64 bits");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t ret = EnsureDefaultWorkspace(h, static_cast<size_t>(need));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%llu bytes", static_cast<unsigned long long>(need));
        return ret;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool CgbtPlaneBytes(uint64_t rows, uint64_t cols, uint64_t& bytes)
{
    return CBlas3TryMulU64(rows, cols, &bytes) && CBlas3TryMulU64(bytes, sizeof(float), &bytes) &&
           CBlas3TryAlignGm(bytes, &bytes);
}

// Sizes one batch's eight planes and picks the round granularity. Every plane
// slot is 512-byte aligned so kernel-side plane bases stay burst friendly.
aclblasStatus_t CgbtPlanWorkspace(
    _aclblas_handle* h, uint32_t m, uint32_t n, uint32_t k, bool isTransA, bool isTransB, int batchCount,
    CgbtWsPlan& plan)
{
    // op(A) is m x k and op(B) is k x n; the physical (stored) shape follows the
    // trans flag.
    const CgbtPhysShape shapeA = CgbtResolvePhys(isTransA, m, k);
    const CgbtPhysShape shapeB = CgbtResolvePhys(isTransB, k, n);
    plan.ldD = CgbtWorkspaceLd(m);

    uint64_t xBytes = 0U;
    uint64_t yBytes = 0U;
    uint64_t tBytes = 0U;
    const bool ok = CgbtPlaneBytes(CgbtWorkspaceLd(shapeB.rows), CgbtWorkspaceCols(shapeB.rows, shapeB.cols), xBytes) &&
                    CgbtPlaneBytes(CgbtWorkspaceLd(shapeA.rows), CgbtWorkspaceCols(shapeA.rows, shapeA.cols), yBytes) &&
                    CgbtPlaneBytes(plan.ldD, n, tBytes);
    if (!ok) {
        OP_LOGE(OP_TAG, "workspace size overflows 64 bits: m=%u n=%u k=%u", m, n, k);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    plan.xStride = xBytes / sizeof(float);
    plan.yStride = yBytes / sizeof(float);
    plan.tStride = tBytes / sizeof(float);

    // per-batch bytes = 2*x + 2*y + 4*t, every step checked (a pre-multiplied
    // 2*xBytes could itself wrap for extreme int dimensions).
    uint64_t twoX = 0U;
    uint64_t twoY = 0U;
    uint64_t fourT = 0U;
    uint64_t perBatchBytes = 0U;
    const bool sumOk = CBlas3TryMulU64(xBytes, 2U, &twoX) && CBlas3TryMulU64(yBytes, 2U, &twoY) &&
                       CBlas3TryMulU64(tBytes, 4U, &fourT) && CBlas3TryAddU64(twoX, twoY, &perBatchBytes) &&
                       CBlas3TryAddU64(perBatchBytes, fourT, &perBatchBytes);
    if (!sumOk) {
        OP_LOGE(OP_TAG, "workspace size overflows 64 bits: m=%u n=%u k=%u", m, n, k);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // Small panels collect their range while resident in UB. Two maxima
    // avoid rereading all four input planes during the numerical epilogue.
    if (m <= 64U && n <= 64U && k <= 64U) {
        const uint32_t largest = std::max(m, std::max(n, k));
        const uint32_t extent = largest <= 16U ? 16U : (largest <= 32U ? 32U : 64U);
        perBatchBytes = std::max<uint64_t>(
            perBatchBytes + 8U, (CGBT_SLOT_PLANES * extent * extent + CGBT_SLOT_FLAG_FLOATS) * sizeof(float));
    } else {
        uint64_t flagBytes = 0U;
        if (!CBlas3TryAlignGm(static_cast<uint64_t>(CgbtAivCoreCount(h)) * sizeof(float), &flagBytes) ||
            !CBlas3TryAddU64(perBatchBytes, flagBytes, &perBatchBytes))
            return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    plan.perBatchFloats = perBatchBytes / sizeof(float);

    return CgbtAllocateWorkspace(h, perBatchBytes, batchCount, plan);
}

// ============================================================================
//  Launch
// ============================================================================
inline uint32_t CgbtClampBlocks(uint64_t jobs, uint32_t cores)
{
    return static_cast<uint32_t>(std::max<uint64_t>(1U, std::min<uint64_t>(jobs, cores)));
}

inline CgbtSplitTilingData CgbtMakeSplitTiling(
    uint32_t batchChunk, const CgbtPhysShape& shape, uint32_t ld, bool conjugate, uint64_t planeFloats)
{
    CgbtSplitTilingData split{};
    split.batchCount = batchChunk;
    split.rows = shape.rows;
    split.cols = shape.cols;
    split.ld = ld;
    split.packedLd = CgbtWorkspaceLd(shape.rows);
    split.negateImag = conjugate ? 1U : 0U;
    split.planeFloats = planeFloats;
    return split;
}

// Phase 0 for both operands in one AIV launch: the kernel walks the
// concatenated A / B column-job spaces, so a small second operand no longer
// pays a whole extra launch.
// With both input components bounded by 1e18 / (K * max(1, |alpha|)),
// reference complex products and partial sums retain ample FP32 range headroom.
// Probe overflow conservatively selects reference arithmetic; inputs are unchanged.
inline float CgbtInputRiskScale(uint32_t k, const aclblasComplex& alpha)
{
    const float magnitude = std::max(1.0F, std::max(std::fabs(alpha.real), std::fabs(alpha.imag)));
    return (CGBT_INPUT_RISK_MULTIPLIER * static_cast<float>(k)) * magnitude;
}

inline void CgbtLaunchSplitFused(
    void* stream, uint32_t aivCoreNum, GM_ADDR ptrA, GM_ADDR ptrB, uint8_t* ws, uint64_t xStride, uint64_t yStride,
    uint32_t batchChunk, const CgbtPhysShape& shapeA, uint32_t lda, bool conjA, const CgbtPhysShape& shapeB,
    uint32_t ldb, bool conjB, GM_ADDR inputFlags, float riskScale)
{
    // Plane offsets are float counts; the workspace base is a byte pointer, so
    // every offset handed to the kernel must be scaled by sizeof(float) first.
    const uint64_t offAr = 2U * xStride * batchChunk * sizeof(float);
    const uint64_t offAi = (2U * xStride + yStride) * batchChunk * sizeof(float);
    const uint64_t offBi = xStride * batchChunk * sizeof(float);
    const CgbtSplitTilingData ta = CgbtMakeSplitTiling(batchChunk, shapeA, lda, conjA, yStride);
    const CgbtSplitTilingData tb = CgbtMakeSplitTiling(batchChunk, shapeB, ldb, conjB, xStride);
    const uint32_t blocks = aivCoreNum;
    CGBT_TRACE(
        OP_TAG, "phase0 split: batch=%u A(rows=%u cols=%u ld=%u conj=%d) B(rows=%u cols=%u ld=%u conj=%d)", batchChunk,
        ta.rows, ta.cols, ta.ld, static_cast<int>(conjA), tb.rows, tb.cols, tb.ld, static_cast<int>(conjB));
    cgemm_batched_split_kernel_do(
        blocks, stream, ptrA, ptrB, ws + offAr, ws + offAi, ws, ws + offBi, ta, tb, inputFlags, riskScale);
}

// Phase 1: all four products in one AIC launch per K-chunk, splitting K so no
// launch exceeds the kernel's compile-time singleK (first chunk overwrites the
// temp planes, later chunks accumulate atomically).
inline void CgbtLaunchGemmProducts(
    void* stream, uint32_t aicCoreNum, GM_ADDR workspace, const CgbtWsPlan& plan, uint32_t batchChunk, uint32_t m,
    uint32_t n, uint32_t k, uint32_t transMode, uint32_t ldX, uint32_t ldY, CgbtGemmTilingData* output = nullptr)
{
    CgbtGemmTilingData gemm{};
    gemm.m = n; // rows of D = C^T
    gemm.n = m; // cols of D = C^T
    gemm.k = k;
    const uint32_t tile = m > 128U && n > 128U && m <= CGBT_MID_TILE && n <= CGBT_MID_TILE && k <= CGBT_REUSE_TILE ?
                              CGBT_MID_TILE :
                              (m > 128U && n > 128U && m <= CGBT_REUSE_TILE && n <= CGBT_REUSE_TILE &&
                                       static_cast<uint64_t>(batchChunk) * CGBT_PRODUCTS >= aicCoreNum ?
                                   CGBT_REUSE_TILE :
                                   (m > CGBT_REUSE_TILE && n > CGBT_REUSE_TILE && m <= CGBT_LARGE_REUSE_TILE &&
                                            n <= CGBT_LARGE_REUSE_TILE && batchChunk >= 8U ?
                                        CGBT_LARGE_REUSE_TILE :
                                        CBLAS3_TILE_M));
    gemm.tileSize = tile;
    gemm.mBlocks = CeilDiv<uint32_t>(n, tile);
    gemm.nBlocks = CeilDiv<uint32_t>(m, tile);
    gemm.batchCount = batchChunk;
    gemm.transMode = transMode;
    gemm.ldX = ldX;
    gemm.ldY = ldY;
    gemm.ldD = plan.ldD;
    gemm.offXr = 0U;                                              // Br
    gemm.offXi = plan.xStride * batchChunk;                       // Bi
    gemm.offYr = 2U * plan.xStride * batchChunk;                  // Ar
    gemm.offYi = (2U * plan.xStride + plan.yStride) * batchChunk; // Ai
    gemm.offT1 = CgbtOffT1(plan) * batchChunk;
    gemm.xPlaneFloats = plan.xStride;
    gemm.yPlaneFloats = plan.yStride;
    gemm.tPlaneFloats = plan.tStride;

    if (output != nullptr) {
        gemm.kCount = k;
        *output = gemm;
        return;
    }

    const uint32_t kChunk = cgemm_batched_gemm_single_k();
    const uint64_t totalTasks = static_cast<uint64_t>(batchChunk) * gemm.mBlocks * gemm.nBlocks * CGBT_PRODUCTS;
    const uint32_t blocks = CgbtClampBlocks(totalTasks, aicCoreNum);
    CGBT_TRACE(
        OP_TAG, "phase1 gemm: M=%u N=%u K=%u transMode=%u products=1-launch aicBlocks=%u kChunk=%u", gemm.m, gemm.n,
        gemm.k, transMode, blocks, kChunk);
    for (uint32_t kBase = 0; kBase < gemm.k; kBase += kChunk) {
        gemm.kBase = kBase;
        gemm.kCount = std::min(kChunk, gemm.k - kBase);
        gemm.enAtomic = (kBase == 0U) ? 0U : 1U;
        cgemm_batched_gemm_kernel_do(blocks, stream, workspace, gemm);
    }
}

inline void CgbtLaunchCombine(
    void* stream, uint32_t aivCoreNum, GM_ADDR workspace, GM_ADDR carray, const CgbtWsPlan& plan, uint32_t batchChunk,
    uint32_t m, uint32_t n, uint32_t ldc, aclblasOperation_t transa, const aclblasComplex& alphaVal,
    const aclblasComplex& betaVal, CgbtCombineTilingData* output = nullptr, GM_ADDR aarray = nullptr,
    GM_ADDR barray = nullptr, CgbtDirectTilingData direct = {})
{
    CgbtCombineTilingData combine{};
    combine.m = m;
    combine.n = n;
    combine.ldc = ldc;
    combine.batchCount = batchChunk;
    combine.ldD = plan.ldD;
    combine.inputFlagCount = aivCoreNum;
    combine.isBetaZero = IsComplexZero(betaVal) ? 1U : 0U;
    combine.preserveBetaOne = transa == ACLBLAS_OP_N && betaVal.real == 1.0F && betaVal.imag == 0.0F;
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaRe = betaVal.real;
    combine.betaIm = betaVal.imag;
    combine.offT1 = CgbtOffT1(plan) * batchChunk;
    combine.offT2 = combine.offT1 + plan.tStride * batchChunk;
    combine.offT3 = combine.offT2 + plan.tStride * batchChunk;
    combine.offT4 = combine.offT3 + plan.tStride * batchChunk;
    combine.tPlaneFloats = plan.tStride;
    combine.wsFloats = plan.perBatchFloats * batchChunk;

    if (output != nullptr) {
        *output = combine;
        return;
    }

    const uint32_t blocks = CgbtClampBlocks(static_cast<uint64_t>(batchChunk) * n, aivCoreNum);
    CGBT_TRACE(
        OP_TAG, "phase2 combine: m=%u n=%u betaZero=%d aivBlocks=%u", m, n, static_cast<int>(combine.isBetaZero),
        blocks);
    cgemm_batched_combine_kernel_do(blocks, stream, workspace, aarray, barray, carray, combine, direct);
}

// Phases 0-2 for one round of batches [batchBase, batchBase + batchChunk).
// The caller-side pointer arrays are advanced by batchBase entries, the
// workspace planes restart at their round-local base.
struct CgbtCall {
    using ConstMatrixPointer = const aclblasComplex*;
    using MatrixPointer = aclblasComplex*;
    _aclblas_handle* h;
    aclblasOperation_t transa, transb;
    int m, n, k, lda, ldb, ldc, batchCount;
    aclblasComplex alphaVal, betaVal;
    const ConstMatrixPointer* Aarray;
    const ConstMatrixPointer* Barray;
    const MatrixPointer* Carray;
};

inline CgbtDirectTilingData CgbtMakeDirectTiling(const CgbtCall& call)
{
    return {
        static_cast<uint32_t>(call.m),
        static_cast<uint32_t>(call.n),
        static_cast<uint32_t>(call.k),
        static_cast<uint32_t>(call.lda),
        static_cast<uint32_t>(call.ldb),
        static_cast<uint32_t>(call.ldc),
        static_cast<uint32_t>(call.batchCount),
        static_cast<uint32_t>(call.transa - ACLBLAS_OP_N),
        static_cast<uint32_t>(call.transb - ACLBLAS_OP_N),
        call.alphaVal.real,
        call.alphaVal.imag,
        call.betaVal.real,
        call.betaVal.imag};
}

inline void CgbtLaunchFusedRound(
    void* stream, uint32_t aicCoreNum, uint32_t aivCoreNum, uint8_t* ws, GM_ADDR aPtrBatch, GM_ADDR bPtrBatch,
    GM_ADDR cPtrBatch, const CgbtWsPlan& plan, const CgbtCall& call, uint32_t batchChunk, const CgbtPhysShape& shapeA,
    const CgbtPhysShape& shapeB, uint32_t transMode)
{
    CgbtFusedTilingData fused{};
    fused.offRangeFlags = (CgbtOffT1(plan) + 4U * plan.tStride) * batchChunk;
    fused.a = CgbtMakeSplitTiling(batchChunk, shapeA, call.lda, call.transa == ACLBLAS_OP_C, plan.yStride);
    fused.b = CgbtMakeSplitTiling(batchChunk, shapeB, call.ldb, call.transb == ACLBLAS_OP_C, plan.xStride);
    CgbtLaunchGemmProducts(
        stream, aicCoreNum, ws, plan, batchChunk, call.m, call.n, call.k, transMode, fused.b.packedLd, fused.a.packedLd,
        &fused.gemm);
    CgbtLaunchCombine(
        stream, aivCoreNum, ws, cPtrBatch, plan, batchChunk, call.m, call.n, call.ldc, call.transa, call.alphaVal,
        call.betaVal, &fused.combine);
    fused.combine.inputRiskScale = CgbtInputRiskScale(call.k, call.alphaVal);
    const uint64_t tasks = static_cast<uint64_t>(batchChunk) * fused.gemm.mBlocks * fused.gemm.nBlocks * CGBT_PRODUCTS;
    cgemm_batched_fused_kernel_do(
        CgbtClampBlocks(tasks, aicCoreNum), stream, aPtrBatch, bPtrBatch, cPtrBatch, ws, fused);
}

aclblasStatus_t CgbtRunRound(const CgbtCall& call, uint32_t batchBase, const CgbtWsPlan& plan, uint32_t batchChunk)
{
    void* stream = call.h->stream;
    uint8_t* ws = static_cast<uint8_t*>(GetEffectiveWorkspace(call.h));
    if (ws == nullptr) {
        OP_LOGE(OP_TAG, "GetEffectiveWorkspace returned nullptr");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const uint32_t aicCoreNum = CgbtAicCoreCount(call.h);
    const uint32_t aivCoreNum = CgbtAivCoreCount(call.h);
    if (aicCoreNum == 0U || aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "core count query failed: aic=%u aiv=%u", aicCoreNum, aivCoreNum);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const bool isTransA = (call.transa != ACLBLAS_OP_N);
    const bool isTransB = (call.transb != ACLBLAS_OP_N);
    // op(A) is m x k and op(B) is k x n; the physical (stored) shape follows the
    // trans flag.
    const CgbtPhysShape shapeA = CgbtResolvePhys(isTransA, call.m, call.k);
    const CgbtPhysShape shapeB = CgbtResolvePhys(isTransB, call.k, call.n);

    // Kernel-side byte offsets of the pointer-array entries for this round.
    constexpr size_t PTR_BYTES = sizeof(void*);
    GM_ADDR aPtrBatch = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<uintptr_t>(call.Aarray) + static_cast<size_t>(batchBase) * PTR_BYTES);
    GM_ADDR bPtrBatch = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<uintptr_t>(call.Barray) + static_cast<size_t>(batchBase) * PTR_BYTES);
    GM_ADDR cPtrBatch = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<uintptr_t>(call.Carray) + static_cast<size_t>(batchBase) * PTR_BYTES);

    const uint32_t transMode = (isTransB ? CGBT_TRANS_X : 0U) | (isTransA ? CGBT_TRANS_Y : 0U);

    if (call.k <= cgemm_batched_gemm_single_k()) {
        CgbtLaunchFusedRound(
            stream, aicCoreNum, aivCoreNum, ws, aPtrBatch, bPtrBatch, cPtrBatch, plan, call, batchChunk, shapeA, shapeB,
            transMode);
        return ACLBLAS_STATUS_SUCCESS;
    }

    CgbtLaunchSplitFused(
        stream, aivCoreNum, aPtrBatch, bPtrBatch, ws, plan.xStride, plan.yStride, batchChunk, shapeA, call.lda,
        call.transa == ACLBLAS_OP_C, shapeB, call.ldb, call.transb == ACLBLAS_OP_C,
        ws + (CgbtOffT1(plan) + 4U * plan.tStride) * batchChunk * sizeof(float),
        CgbtInputRiskScale(call.k, call.alphaVal));
    CgbtLaunchGemmProducts(
        stream, aicCoreNum, ws, plan, batchChunk, call.m, call.n, call.k, transMode, CgbtWorkspaceLd(shapeB.rows),
        CgbtWorkspaceLd(shapeA.rows));
    auto direct = CgbtMakeDirectTiling(call);
    direct.batchCount = batchChunk;
    CgbtLaunchCombine(
        stream, aivCoreNum, ws, cPtrBatch, plan, batchChunk, call.m, call.n, call.ldc, call.transa, call.alphaVal,
        call.betaVal, nullptr, aPtrBatch, bPtrBatch, direct);
    return ACLBLAS_STATUS_SUCCESS;
}

// Early exit (k = 0 or alpha = (0,0)): C[i] = beta * C[i] per batch, bit-exact.
aclblasStatus_t CgbtRunBetaScale(
    _aclblas_handle* h, uint32_t m, uint32_t n, uint32_t ldc, int batchCount, const aclblasComplex& betaVal,
    aclblasComplex* const Carray[])
{
    const uint32_t aivCoreNum = CgbtAivCoreCount(h);
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    CgbtBetaScaleTilingData scale{};
    scale.m = m;
    scale.n = n;
    scale.ldc = ldc;
    scale.batchCount = static_cast<uint32_t>(batchCount);
    scale.isBetaZero = IsComplexZero(betaVal) ? 1U : 0U;
    scale.betaRe = betaVal.real;
    scale.betaIm = betaVal.imag;
    const uint32_t blocks = CgbtClampBlocks(static_cast<uint64_t>(batchCount) * n, aivCoreNum);
    CGBT_TRACE(
        OP_TAG, "early exit: C = beta*C, m=%u n=%u batch=%d betaZero=%d aivBlocks=%u", m, n, batchCount,
        static_cast<int>(scale.isBetaZero), blocks);
    cgemm_batched_beta_scale_kernel_do(
        blocks, h->stream, reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(Carray)), scale);
    return ACLBLAS_STATUS_SUCCESS;
}

inline uint64_t CgbtDirectJobs(const CgbtCall& call)
{
    uint64_t jobs = call.m <= 32 && call.n <= 32 && call.k <= 32 ?
                        call.batchCount :
                        static_cast<uint64_t>(call.m) * call.n * call.batchCount;
    if (call.m == 8 && call.n == call.m && call.k == call.m && call.lda == call.m && call.ldb == call.m &&
        call.ldc == call.m && call.transa == ACLBLAS_OP_N && call.transb == ACLBLAS_OP_N &&
        call.alphaVal.real == 1.0F && call.alphaVal.imag == 0.0F && call.betaVal.real == 0.0F &&
        call.betaVal.imag == 0.0F) {
        const uint32_t group = call.batchCount <= 64 ? 2U : (call.m == 8 && call.batchCount > 256 ? 32U : 8U);
        jobs = (static_cast<uint32_t>(call.batchCount) + group - 1U) / group;
    }
    return jobs;
}

inline aclblasStatus_t CgbtRunDirect(const CgbtCall& call)
{
    CgbtDirectTilingData direct = CgbtMakeDirectTiling(call);
    const uint32_t cores = CgbtAivCoreCount(call.h);
    if (cores == 0U)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    const uint64_t jobs = CgbtDirectJobs(call);
    cgemm_batched_direct_kernel_do(
        CgbtClampBlocks(jobs, cores), call.h->stream,
        reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Aarray)),
        reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Barray)),
        reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Carray)), direct);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CgbtTryVector(const CgbtCall& call, bool& handled)
{
    handled = false;
    const uint32_t vectors = CgbtAivCoreCount(call.h);
    const bool packedSquare = call.m == 16 && call.n == 16 && call.k == 16 && call.transa == ACLBLAS_OP_N &&
                              call.transb == ACLBLAS_OP_N && call.lda == 16 && call.ldb == 16 && call.ldc == 16;
    if (vectors != 0U && (static_cast<uint32_t>(call.batchCount) <= vectors * 2U ||
                          (packedSquare && static_cast<uint32_t>(call.batchCount) <= vectors * 8U))) {
        handled = true;
        const CgbtDirectTilingData direct = CgbtMakeDirectTiling(call);
        const int launched = cgemm_batched_vector16_kernel_do(
            CgbtClampBlocks(call.batchCount, vectors), call.h->stream,
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Aarray)),
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Barray)),
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Carray)), direct, call.h->cgemm_vector_function,
            call.h->cgemm_vector_grouped_function);
        if (launched != 0) {
            OP_LOGE(OP_TAG, "tiny kernel launch failed: %d", launched);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CgbtRunSmall(const CgbtCall& call)
{
    const uint32_t largest = std::max(call.m, std::max(call.n, call.k));
    const uint32_t extent = largest <= 16U ? 16U : (largest <= 32U ? 32U : 64U);
    const uint32_t perBatch = (CGBT_SLOT_PLANES * extent * extent + CGBT_SLOT_FLAG_FLOATS) * sizeof(float);
    const uint64_t capacity = call.h->workspace_owner == AclblasWorkspaceOwner::User ?
                                  std::min<uint64_t>(CGBT_WS_CAP_BYTES, call.h->workspace_size) :
                                  CGBT_WS_CAP_BYTES;
    if (capacity < perBatch)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    const uint32_t chunk = std::min<uint64_t>(call.batchCount, capacity / perBatch);
    const aclblasStatus_t allocation = EnsureDefaultWorkspace(call.h, static_cast<size_t>(chunk) * perBatch);
    if (allocation != ACLBLAS_STATUS_SUCCESS)
        return allocation;
    const uint32_t cores = CgbtAicCoreCount(call.h);
    if (cores == 0U)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    CgbtDirectTilingData direct = CgbtMakeDirectTiling(call);
    for (uint32_t batch = 0U; batch < static_cast<uint32_t>(call.batchCount); batch += chunk) {
        direct.batchCount = std::min(chunk, static_cast<uint32_t>(call.batchCount) - batch);
        const uint64_t offset = static_cast<uint64_t>(batch) * sizeof(void*);
        const int launched = cgemm_batched_small_kernel_do(
            CgbtClampBlocks(direct.batchCount, cores), call.h->stream,
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Aarray) + offset),
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Barray) + offset),
            reinterpret_cast<GM_ADDR>(reinterpret_cast<uintptr_t>(call.Carray) + offset),
            static_cast<GM_ADDR>(GetEffectiveWorkspace(call.h)), direct, call.h->cgemm_panel_function);
        if (launched != 0) {
            OP_LOGE(OP_TAG, "small kernel launch failed: %d", launched);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CgbtRunGeneral(const CgbtCall& call)
{
    const bool isTransA = (call.transa != ACLBLAS_OP_N);
    const bool isTransB = (call.transb != ACLBLAS_OP_N);
    CgbtWsPlan plan;
    const aclblasStatus_t planRet = CgbtPlanWorkspace(
        call.h, static_cast<uint32_t>(call.m), static_cast<uint32_t>(call.n), static_cast<uint32_t>(call.k), isTransA,
        isTransB, call.batchCount, plan);
    if (planRet != ACLBLAS_STATUS_SUCCESS) {
        return planRet;
    }

    for (uint32_t batchBase = 0; batchBase < static_cast<uint32_t>(call.batchCount); batchBase += plan.batchChunk) {
        const uint32_t batchChunk = std::min(plan.batchChunk, static_cast<uint32_t>(call.batchCount) - batchBase);
        const aclblasStatus_t roundRet = CgbtRunRound(call, batchBase, plan, batchChunk);
        if (roundRet != ACLBLAS_STATUS_SUCCESS) {
            return roundRet;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool CgbtUseDirect(const CgbtCall& call)
{
    const bool batchCube = call.transa == ACLBLAS_OP_N && call.transb == ACLBLAS_OP_N && call.m == call.n &&
                           call.m == call.k && (call.m == 16 || call.m == 32 || call.m == 64);
    const bool shortReductionCube = static_cast<uint64_t>(call.m) * call.n > 256U && call.k <= 64;
    return (call.m <= 8 && call.n <= 8 && call.k <= 8) ||
           (static_cast<uint64_t>(call.m) * call.n <= CGBT_DIRECT_OUTPUT_MAX && !batchCube && !shortReductionCube);
}

inline aclblasStatus_t CgbtDispatch(const CgbtCall& call)
{
    // Avoid all vector/panel DMA paths when a complex leading dimension
    // exceeds the uint32 byte-stride representation.
    constexpr uint32_t MAX_DMA_COMPLEX_STRIDE = 0xffffffffU / sizeof(aclblasComplex);
    if (static_cast<uint32_t>(call.lda) > MAX_DMA_COMPLEX_STRIDE ||
        static_cast<uint32_t>(call.ldb) > MAX_DMA_COMPLEX_STRIDE ||
        static_cast<uint32_t>(call.ldc) > MAX_DMA_COMPLEX_STRIDE || CgbtUseDirect(call)) {
        return CgbtRunDirect(call);
    }

    // This AIV implementation keeps all operands and results in UB. Select
    // it before planning or acquiring an unused GM workspace.
    if (call.m <= 16 && call.n <= 16 && call.k <= 16 && call.alphaVal.real == 1.0F && call.alphaVal.imag == 0.0F &&
        IsComplexZero(call.betaVal)) {
        bool handled = false;
        const auto status = CgbtTryVector(call, handled);
        if (handled)
            return status;
    }

    // The direct-output mixed kernel needs only its bounded producer slots,
    // without the four product planes of the general GEMM implementation.
    if (call.m <= 64 && call.n <= 64 && call.k <= 64 && call.alphaVal.real == 1.0F && call.alphaVal.imag == 0.0F &&
        IsComplexZero(call.betaVal)) {
        return CgbtRunSmall(call);
    }

    return CgbtRunGeneral(call);
}

} // namespace

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCgemmBatched(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda, const aclblasComplex* const Barray[],
    int ldb, const aclblasComplex* beta, aclblasComplex* const Carray[], int ldc, int batchCount)
{
    CGBT_TRACE(
        OP_TAG, "entry: transa=%d transb=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d batch=%d", static_cast<int>(transa),
        static_cast<int>(transb), m, n, k, lda, ldb, ldc, batchCount);

    const aclblasStatus_t st = ValidateCgemmBatchedParams(
        handle, transa, transb, m, n, k, lda, ldb, ldc, batchCount, alpha, beta, Aarray, Barray, Carray);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS; // quick return, no side effects
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    const aclblasComplex alphaVal = *alpha; // host-memory scalars
    const aclblasComplex betaVal = *beta;

    if (k == 0 || IsComplexZero(alphaVal)) {
        if (betaVal.real == 1.0F && betaVal.imag == 0.0F) {
            return ACLBLAS_STATUS_SUCCESS; // C unchanged
        }
        return CgbtRunBetaScale(
            h, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(ldc), batchCount, betaVal,
            Carray);
    }

    const CgbtCall call{h,   transa,     transb,   m,       n,      k,      lda,   ldb,
                        ldc, batchCount, alphaVal, betaVal, Aarray, Barray, Carray};
    return CgbtDispatch(call);
}
