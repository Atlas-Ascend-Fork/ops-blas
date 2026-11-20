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
 * \file strmm_host.cpp
 * \brief STRMM Host implementation for ascend950 (DAV_3510)
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/aclblas_handle_internal.h"
#include "strmm_tiling_data.h"
#include "common/helper/syrk_host_utils.h"

struct StrmmMirrorTilingData;
void strmm_mirror_kernel_do(const uint8_t* gmA, uint8_t* gmWorkspaceA, const StrmmMirrorTilingData &tiling,
                              uint32_t numBlocks, void *stream);
struct StrmmGemmTilingData;
void strmm_gemm_kernel_do(uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp,
                            const StrmmGemmTilingData &tiling, uint32_t numBlocks, void *stream);
struct StrmmScaleTilingData;
void strmm_scale_kernel_do(const uint8_t* gmTemp, uint8_t* gmC,
                            const StrmmScaleTilingData &tiling, uint32_t numBlocks, void *stream);

static aclblasStatus_t ValidateStrmmParams(
    aclblasSideMode_t side, int m, int n,
    int lda, int ldb, int ldc,
    const float* alpha)
{
    int dimA = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    int minLda = std::max(1, dimA);
    int minLdb = std::max(1, m);
    int minLdc = std::max(1, m);
    CHECK_RET(
        lda >= minLda,
        OP_LOGE("aclblasStrmm", "lda must be >= max(1, dimA), got lda=%d, dimA=%d", lda, dimA);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= minLdb,
        OP_LOGE("aclblasStrmm", "ldb must be >= max(1, m), got ldb=%d, m=%d", ldb, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= minLdc,
        OP_LOGE("aclblasStrmm", "ldc must be >= max(1, m), got ldc=%d, m=%d", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasStrmm", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static StrmmMirrorTilingData CalMirrorTilingData(
    uint32_t usedAivCoreNum, uint32_t m, uint32_t n, uint32_t lda,
    uint32_t sideMode, uint32_t uploMode, uint32_t transMode, uint32_t diagMode)
{
    StrmmMirrorTilingData tiling{};
    tiling.sideMode = sideMode;
    tiling.uploMode = uploMode;
    tiling.transMode = transMode;
    tiling.diagMode = diagMode;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.dimA = (sideMode == ACLBLAS_SIDE_LEFT) ? m : n;
    tiling.mirrorRowsPerCore = CeilDiv<uint32_t>(tiling.dimA, usedAivCoreNum);
    tiling.lda = lda;
    tiling.nthreads = std::min(CeilAlign<uint32_t>(tiling.mirrorRowsPerCore, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    return tiling;
}

static StrmmGemmTilingData CalGemmTilingData(
    uint32_t usedAicCoreNum, uint32_t m, uint32_t n, uint32_t sideMode,
    uint32_t lda, uint32_t ldb)
{
    StrmmGemmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.sideMode = sideMode;

    uint32_t coreM = CeilDiv<uint32_t>(m, usedAicCoreNum);
    uint32_t coreN = CeilDiv<uint32_t>(n, usedAicCoreNum);
    tiling.usedAicCoreNum = usedAicCoreNum;
    tiling.singleCoreM = std::max<uint32_t>(coreM, STRMM_ARCH35_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreN, STRMM_ARCH35_BASE_N);

    uint32_t tileM = std::min<uint32_t>(STRMM_ARCH35_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(STRMM_ARCH35_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = STRMM_ARCH35_DEFAULT_TILE_K_CHUNK;

    tileM = std::max<uint32_t>(tileM, STRMM_ARCH35_BASE_M);
    tileN = std::max<uint32_t>(tileN, STRMM_ARCH35_BASE_N);
    tileKChunk = std::max<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K);

    if (m < STRMM_ARCH35_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(std::max<uint32_t>(m, STRMM_ARCH35_BASE_M), STRMM_ARCH35_BASE_M));
    }
    if (n < STRMM_ARCH35_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(std::max<uint32_t>(n, STRMM_ARCH35_BASE_N), STRMM_ARCH35_BASE_N));
    }

    uint32_t aSideL1 = CeilAlign<uint32_t>(tileM, STRMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * STRMM_ARCH35_FP32_SIZE;
    uint32_t bSideL1 = CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * CeilAlign<uint32_t>(tileN, STRMM_ARCH35_BASE_N) * STRMM_ARCH35_FP32_SIZE;

    while (STRMM_ARCH35_L1_BUF_NUM * (aSideL1 + bSideL1) > STRMM_ARCH35_L1_SIZE_BYTES && tileKChunk > STRMM_ARCH35_BASE_K) {
        tileKChunk /= 2;
        tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K), STRMM_ARCH35_BASE_K);
        aSideL1 = CeilAlign<uint32_t>(tileM, STRMM_ARCH35_BASE_M) * CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * STRMM_ARCH35_FP32_SIZE;
        bSideL1 = CeilAlign<uint32_t>(tileKChunk, STRMM_ARCH35_BASE_K) * CeilAlign<uint32_t>(tileN, STRMM_ARCH35_BASE_N) * STRMM_ARCH35_FP32_SIZE;
    }

    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.tempRowStride = CeilAlign<uint32_t>(n, STRMM_ARCH35_FIXPIPE_N_ALIGN);

    return tiling;
}

static StrmmScaleTilingData CalScaleTilingData(
    uint32_t usedAivCoreNum, uint32_t m, uint32_t n, uint32_t ldc,
    uint32_t tempRowStride, float alphaVal)
{
    StrmmScaleTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldc = ldc;
    tiling.tempRowStride = tempRowStride;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.scaleRowsPerCore = CeilDiv<uint32_t>(m, usedAivCoreNum);
    tiling.alphaVal = alphaVal;
    tiling.nthreads = std::min(CeilAlign<uint32_t>(tiling.scaleRowsPerCore, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    return tiling;
}

static aclblasStatus_t ResolveStrmmScalars(const float* alpha,
    float* alphaVal, bool* alphaIsDevice, aclrtStream stream)
{
    aclblasStatus_t alphaLocSt = CheckPtrLocation(alpha, alphaIsDevice);
    if (alphaLocSt != ACLBLAS_STATUS_SUCCESS) {
        return alphaLocSt;
    }
    if (*alphaIsDevice) {
        // D2H read 4 bytes (alpha only, strmm has no beta).
        *alphaVal = 0.0f;
        aclError aclRet = aclrtMemcpyAsync(alphaVal, sizeof(float), alpha, sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasStrmm", "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        aclRet = aclrtSynchronizeStream(stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasStrmm", "aclrtSynchronizeStream for alpha D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        *alphaVal = *alpha;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// HandleStrmmAlphaZero — C = 0 when alpha == 0 (fast path).
//   Host mode: alpha value directly available.
//   Device mode: value read via D2H memcpy + sync before entering this path.
//   BLAS: if alpha==0 then A and B need not be set.
// Two cases are handled to stay consistent with the normal scale kernel, which
// only writes the m active rows of each column (C[j*ldc+i], i<m, j<n) and
// leaves the padding rows (m..ldc-1) untouched:
//   - ldc == m (no padding): a single contiguous memset of the m×n block.
//   - ldc >  m (padding): per-column memset of m elements, preserving padding.
// All operations are asynchronous on h->stream; no host-side sync is introduced.
// ==========================================================================
static aclblasStatus_t HandleStrmmAlphaZero(
    aclblasHandle_t handle, uint32_t m, uint32_t n, uint32_t ldc, float* C)
{
    auto* h = static_cast<_aclblas_handle*>(handle);
    size_t colBytes = static_cast<size_t>(m) * STRMM_ARCH35_FP32_SIZE;
    size_t colStride = static_cast<size_t>(ldc) * STRMM_ARCH35_FP32_SIZE;
    if (ldc == m) {
        // No padding — single contiguous memset.
        size_t cBytes = colStride * static_cast<size_t>(n);
        aclError ret = aclrtMemsetAsync(C, cBytes, 0, cBytes, h->stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasStrmm", "aclrtMemsetAsync failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    } else {
        // Padding rows exist — zero each column individually.
        for (uint32_t j = 0; j < n; ++j) {
            float* colPtr = C + static_cast<size_t>(j) * ldc;
            aclError ret = aclrtMemsetAsync(colPtr, colBytes, 0, colBytes, h->stream);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasStrmm", "aclrtMemsetAsync failed, col=%u, ret=%d", j, ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchStrmmPipeline(
    _aclblas_handle* h, const float* A, const float* B, float* C,
    uint32_t dimA, uint32_t uM, uint32_t uN,
    const StrmmMirrorTilingData& mirrorTiling, uint32_t usedAivCoreNum,
    const StrmmGemmTilingData& gemmTiling, uint32_t usedAicCoreNum,
    const StrmmScaleTilingData& scaleTiling, uint32_t usedAivCoreNumScale,
    uint32_t uLda)
{
    size_t workspaceASize = static_cast<size_t>(uLda) * static_cast<size_t>(dimA) * STRMM_ARCH35_FP32_SIZE;
    constexpr size_t GM_ALIGN = 32;
    size_t workspaceASizeAligned = (workspaceASize + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN;
    size_t tempSize = static_cast<size_t>(uN) * static_cast<size_t>(gemmTiling.tempRowStride) * STRMM_ARCH35_FP32_SIZE;
    if (workspaceASizeAligned > SIZE_MAX - tempSize) {
        OP_LOGE("aclblasStrmm", "workspace size overflow: wsAligned=%zu, temp=%zu", workspaceASizeAligned, tempSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t requiredBytes = workspaceASizeAligned + tempSize;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, requiredBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasStrmm", "workspace ensure failed, required=%zu, ret=%d", requiredBytes, wsRet);
        return wsRet;
    }

    uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* workspaceADevice = wsBase;
    uint8_t* tempDevice = wsBase + workspaceASizeAligned;

    strmm_mirror_kernel_do(
        reinterpret_cast<const uint8_t*>(A), workspaceADevice,
        mirrorTiling, usedAivCoreNum, h->stream);
    strmm_gemm_kernel_do(
        workspaceADevice, reinterpret_cast<const uint8_t*>(B), tempDevice,
        gemmTiling, usedAicCoreNum, h->stream);
    strmm_scale_kernel_do(
        tempDevice, reinterpret_cast<uint8_t*>(C),
        scaleTiling, usedAivCoreNumScale, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

struct StrmmTilings {
StrmmMirrorTilingData mirror{};
StrmmGemmTilingData gemm{};
StrmmScaleTilingData scale{};
    uint32_t usedAivCoreNum;
    uint32_t usedAicCoreNum;
    uint32_t usedAivCoreNumScale;
};

static StrmmTilings CalStrmmTilings(
    uint32_t aivCoreNum, uint32_t aicCoreNum,
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag,
    uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    float alphaVal)
{
    StrmmTilings t{};
    uint32_t dimA = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;
    t.usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(dimA, aivCoreNum), 1);
    uint64_t tileCount = static_cast<uint64_t>(CeilDiv<uint32_t>(uM, STRMM_ARCH35_BASE_M))
                       * static_cast<uint64_t>(CeilDiv<uint32_t>(uN, STRMM_ARCH35_BASE_N));
    t.usedAicCoreNum = std::max<uint32_t>(std::min<uint64_t>(tileCount, aicCoreNum), 1);
    t.usedAivCoreNumScale = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);

    t.mirror = CalMirrorTilingData(
        t.usedAivCoreNum, uM, uN, uLda,
        static_cast<uint32_t>(side), static_cast<uint32_t>(uplo),
        static_cast<uint32_t>(trans), static_cast<uint32_t>(diag));
    // Column-major adaptation: GEMM kernel internally works in row-major. To compute the
    // column-major result C(m×n), the GEMM kernel computes C^T (row-major n×m) by swapping
    // m↔n and flipping side. This leverages two properties:
    //   1. Column-major A in memory == row-major A^T: the mirror-filled workspaceA is
    //      stored column-major, so the row-major GEMM kernel reads it as A^T (trans=N) or
    //      (A^T)^T = A (trans=T).
    //   2. Column-major B(m×n) in memory == row-major B^T(n×m): the row-major GEMM kernel
    //      reads B as B^T, which is exactly what C^T = ... * B^T needs.
    // After the swap, side=LEFT (A on the left) becomes side=RIGHT (A on the right)
    // in the kernel's view, and vice versa. lda/ldb are NOT swapped: the GEMM kernel always
    // accesses aGlobal (workspaceA) with tiling.lda and bGlobal (B) with tiling.ldb,
    // regardless of which side each matrix is on. Since aGlobal always holds workspaceA and
    // bGlobal always holds B, passing the original uLda/uLdb directly gives each matrix its
    // correct leading dimension.
    uint32_t gemmSide = (side == ACLBLAS_SIDE_LEFT) ? ACLBLAS_SIDE_RIGHT : ACLBLAS_SIDE_LEFT;
    t.gemm = CalGemmTilingData(
        t.usedAicCoreNum, uN, uM, gemmSide, uLda, uLdb);
    t.scale = CalScaleTilingData(
        t.usedAivCoreNumScale, uM, uN, uLdc, t.gemm.tempRowStride, alphaVal);

    OP_LOGD("aclblasStrmm",
        "mirror tiling: side=%u uplo=%u trans=%u diag=%u aivCores=%u rowsPerCore=%u lda=%u dimA=%u",
        t.mirror.sideMode, t.mirror.uploMode,
        t.mirror.transMode, t.mirror.diagMode,
        t.mirror.usedAivCoreNum, t.mirror.mirrorRowsPerCore, t.mirror.lda, t.mirror.dimA);
    OP_LOGD("aclblasStrmm",
        "gemm tiling: m=%u n=%u side=%u aicCores=%u singleCoreM=%u singleCoreN=%u "
        "tileM=%u tileN=%u tileKChunk=%u lda=%u ldb=%u tempRowStride=%u",
        t.gemm.m, t.gemm.n, t.gemm.sideMode, t.gemm.usedAicCoreNum,
        t.gemm.singleCoreM, t.gemm.singleCoreN,
        t.gemm.tileM, t.gemm.tileN, t.gemm.tileKChunk,
        t.gemm.lda, t.gemm.ldb, t.gemm.tempRowStride);
    OP_LOGD("aclblasStrmm",
        "scale tiling: m=%u n=%u ldc=%u tempRowStride=%u aivCores=%u rowsPerCore=%u",
        t.scale.m, t.scale.n, t.scale.ldc, t.scale.tempRowStride,
        t.scale.usedAivCoreNum, t.scale.scaleRowsPerCore);
    return t;
}

static aclblasStatus_t ExecuteStrmmKernels(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag,
    uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const float* A, const float* B, float alphaVal, float* C)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasStrmm", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasStrmm", "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    StrmmTilings t = CalStrmmTilings(aivCoreNum, aicCoreNum, side, uplo, trans, diag,
        uM, uN, uLda, uLdb, uLdc, alphaVal);

    OP_LOGI("aclblasStrmm", "launching mirror kernel: aivCores=%u", t.usedAivCoreNum);
    OP_LOGI("aclblasStrmm", "launching gemm kernel: aicCores=%u", t.usedAicCoreNum);
    OP_LOGI("aclblasStrmm", "launching scale kernel: aivCores=%u", t.usedAivCoreNumScale);

    uint32_t dimA = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;
    return LaunchStrmmPipeline(h, A, B, C, dimA, uM, uN,
        t.mirror, t.usedAivCoreNum, t.gemm, t.usedAicCoreNum,
        t.scale, t.usedAivCoreNumScale, uLda);
}

aclblasStatus_t aclblasStrmm(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc)
{
    // Correct validation order — handle, enums, m/n before quick return.
    // This ensures invalid handle/enums are caught even when m==0 or n==0.
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasStrmm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE("aclblasStrmm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasStrmm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasStrmm", "trans must be N, T or C, got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(
        diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
        OP_LOGE("aclblasStrmm", "diag must be UNIT or NON_UNIT, got %d", static_cast<int>(diag));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(m >= 0, OP_LOGE("aclblasStrmm", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasStrmm", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);

    // Quick return: m==0 or n==0 — no pointer/ld validation needed (BLAS standard).
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Validate ld and alpha (side/uplo/trans/diag already checked above).
    // A/B/C pointers are validated after alpha resolution below.
    aclblasStatus_t st = ValidateStrmmParams(side, m, n, lda, ldb, ldc, alpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Resolve alpha scalar before alpha==0 check: if alpha is a device pointer
    // the host cannot dereference it. Device alpha is read back via D2H memcpy.
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    float alphaVal;
    bool alphaIsDevice;
    aclblasStatus_t scalarSt = ResolveStrmmScalars(alpha, &alphaVal, &alphaIsDevice, h->stream);
    if (scalarSt != ACLBLAS_STATUS_SUCCESS) {
        return scalarSt;
    }

    CHECK_RET(C != nullptr, OP_LOGE("aclblasStrmm", "C must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    // alpha==0 fast path: C = 0, skip mirror+gemm.
    // BLAS: if alpha==0 then A and B need not be set.
    // Host mode: alpha value directly available.
    // Device mode: value read via D2H memcpy + sync.
    if (alphaVal == 0.0f) {
        return HandleStrmmAlphaZero(handle, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
            static_cast<uint32_t>(ldc), C);
    }

    CHECK_RET(A != nullptr, OP_LOGE("aclblasStrmm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(B != nullptr, OP_LOGE("aclblasStrmm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    return ExecuteStrmmKernels(handle, side, uplo, trans, diag,
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda), static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc),
        A, B, alphaVal, C);
}
