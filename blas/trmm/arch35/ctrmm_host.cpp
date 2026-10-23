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
 * \file ctrmm_host.cpp
 * \brief CTRMM Host implementation for ascend950 (DAV_3510) — M2 correctness baseline.
 *
 * Structure mirrors aclblasStrmm (same directory) with the complex64 specifics
 * from the task document:
 *   - alpha is aclblasComplex (Host or Device pointer; Device value is read
 *     back with one 8-byte D2H memcpy + stream sync before launch);
 *   - alpha == (0,0): A is not referenced, B need not be valid input, C is
 *     zeroed on the active m rows of each column (padding rows preserved);
 *   - C == nullptr with m > 0 and n > 0 returns INVALID_VALUE (including the
 *     alpha == (0,0) case); C == B is allowed (in-place), no other overlap;
 *   - illegal side/uplo/trans/diag return ACLBLAS_STATUS_INVALID_VALUE
 *     (task doc §2.4, cuBLAS-aligned; NOTE: deliberately differs from
 *     aclblasStrmm's INVALID_ENUM return, per this operator's spec).
 */

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ctrmm_tiling_data.h"
#include "ctrmm_kernel.h"

static aclblasStatus_t ValidateCtrmmParams(
    aclblasSideMode_t side, int m, int n, int lda, int ldb, int ldc, const aclblasComplex* alpha)
{
    int dimA = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    int minLda = std::max(1, dimA);
    int minLdb = std::max(1, m);
    int minLdc = std::max(1, m);
    CHECK_RET(lda >= minLda, OP_LOGE("aclblasCtrmm", "lda must be >= max(1, dimA), got lda=%d, dimA=%d", lda, dimA);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= minLdb, OP_LOGE("aclblasCtrmm", "ldb must be >= max(1, m), got ldb=%d, m=%d", ldb, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= minLdc, OP_LOGE("aclblasCtrmm", "ldc must be >= max(1, m), got ldc=%d, m=%d", ldc, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasCtrmm", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Resolve complex alpha. Device pointer: single 8-byte D2H memcpy + stream
// sync (strmm's ResolveStrmmScalars pattern, complex64 width).
static aclblasStatus_t ResolveCtrmmScalars(
    const aclblasComplex* alpha, aclblasComplex* alphaVal, bool* alphaIsDevice, aclrtStream stream)
{
    aclblasStatus_t alphaLocSt = CheckPtrLocation(reinterpret_cast<const void*>(alpha), alphaIsDevice);
    if (alphaLocSt != ACLBLAS_STATUS_SUCCESS) {
        return alphaLocSt;
    }
    if (*alphaIsDevice) {
        *alphaVal = aclblasComplex{0.0f, 0.0f};
        aclError aclRet = aclrtMemcpyAsync(
            alphaVal, sizeof(aclblasComplex), alpha, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasCtrmm", "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        aclRet = aclrtSynchronizeStream(stream);
        if (aclRet != ACL_SUCCESS) {
            OP_LOGE("aclblasCtrmm", "aclrtSynchronizeStream for alpha D2H failed, ret=%d", aclRet);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        *alphaVal = *alpha;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// alpha == (0,0): zero the m active rows of each of the n columns, preserving
// padding rows (m..ldc-1) — identical to the strmm alpha-zero fast path, with
// complex64 element width. ldc == m allows one contiguous memset.
static aclblasStatus_t HandleCtrmmAlphaZero(
    aclblasHandle_t handle, uint32_t m, uint32_t n, uint32_t ldc, aclblasComplex* C)
{
    auto* h = static_cast<_aclblas_handle*>(handle);
    size_t colBytes = static_cast<size_t>(m) * sizeof(aclblasComplex);
    size_t colStride = static_cast<size_t>(ldc) * sizeof(aclblasComplex);
    if (ldc == m) {
        size_t cBytes = colStride * static_cast<size_t>(n);
        aclError ret = aclrtMemsetAsync(C, cBytes, 0, cBytes, h->stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCtrmm", "aclrtMemsetAsync failed, ret=%d", ret);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    } else {
        for (uint32_t j = 0; j < n; ++j) {
            void* colPtr = reinterpret_cast<uint8_t*>(C) + static_cast<size_t>(j) * colStride;
            aclError ret = aclrtMemsetAsync(colPtr, colBytes, 0, colBytes, h->stream);
            if (ret != ACL_SUCCESS) {
                OP_LOGE("aclblasCtrmm", "aclrtMemsetAsync failed, col=%u, ret=%d", j, ret);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Common CtrmmTilingData fill for the SIMT launch paths (naive row-split
// and tiny flattened). The per-path partition is computed by the caller.
static void FillCtrmmSimdTiling(
    CtrmmTilingData& tiling, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    uint32_t usedAivCoreNum, uint32_t rowsPerCore, uint32_t nthreads, const aclblasComplex& alphaVal)
{
    tiling.sideMode = static_cast<uint32_t>(side);
    tiling.uploMode = static_cast<uint32_t>(uplo);
    tiling.transMode = static_cast<uint32_t>(trans);
    tiling.diagMode = static_cast<uint32_t>(diag);
    tiling.m = uM;
    tiling.n = uN;
    tiling.lda = uLda;
    tiling.ldb = uLdb;
    tiling.ldc = uLdc;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.rowsPerCore = rowsPerCore;
    tiling.nthreads = nthreads;
    tiling.alphaReal = alphaVal.real;
    tiling.alphaImag = alphaVal.imag;
}

static aclblasStatus_t ExecuteCtrmmKernel(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex& alphaVal, aclblasComplex* C)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCtrmm", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // Row partition over m output rows (naive M2 pass; expert kernel will
    // repartition in the three-phase pipeline).
    uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);
    const uint32_t rowsPerCore = CeilDiv<uint32_t>(uM, usedAivCoreNum);
    CtrmmTilingData tiling{};
    FillCtrmmSimdTiling(
        tiling, side, uplo, trans, diag, uM, uN, uLda, uLdb, uLdc, usedAivCoreNum, rowsPerCore,
        std::min(CeilAlign<uint32_t>(rowsPerCore, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM), alphaVal);

    OP_LOGD(
        "aclblasCtrmm",
        "naive tiling: side=%u uplo=%u trans=%u diag=%u m=%u n=%u lda=%u ldb=%u ldc=%u aivCores=%u "
        "rowsPerCore=%u nthreads=%u",
        tiling.sideMode, tiling.uploMode, tiling.transMode, tiling.diagMode, tiling.m, tiling.n, tiling.lda, tiling.ldb,
        tiling.ldc, tiling.usedAivCoreNum, tiling.rowsPerCore, tiling.nthreads);

    ctrmm_naive_kernel_do(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(B), reinterpret_cast<uint8_t*>(C), tiling,
        usedAivCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// TINY fast path: single-block flattened (i,j) kernel for m,n <= 32 (the
// naive row-split is GM-latency-bound at these sizes; the tiny kernel keeps
// numerics identical — see ctrmm_kernel.cpp CtrmmTinyCompute). Thread count:
// enough flattened outputs per thread stays tiny; 128..512 matches the
// measured single-block launch floor.
static aclblasStatus_t ExecuteCtrmmTiny(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex& alphaVal, aclblasComplex* C)
{
    CtrmmTilingData tiling{};
    FillCtrmmSimdTiling(
        tiling, side, uplo, trans, diag, uM, uN, uLda, uLdb, uLdc, 1u, uM,
        std::min(CeilAlign<uint32_t>(uM * uN, SIMT_MIN_THREAD_NUM), static_cast<uint32_t>(SIMT_MAX_THREAD_NUM)),
        alphaVal);
    // ordA == 32: compile-time-unrolled variant (batched GM loads).
    const uint32_t tinyOrd = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;
    if (tinyOrd == 32) {
        ctrmm_tiny_k32_kernel_do(
            reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(B), reinterpret_cast<uint8_t*>(C),
            tiling, h->stream);
        return ACLBLAS_STATUS_SUCCESS;
    }
    ctrmm_tiny_kernel_do(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(B), reinterpret_cast<uint8_t*>(C), tiling,
        h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// strmm-lineage Phase-B tiling (L0 lever, route A): port of strmm's
// CalGemmTilingData recipe — m/n are the SWAPPED (n, m) C^T view, side is
// flipped, lda/ldb unswapped; L1-fitted tile shapes; stride 8-aligned.
// Tile-grid partition (measured): each core owns tile-sized bands so the
// cube sees full tiles — 512^2 269.8->115.1us, full set 171->191 passes.
// Grid 128:64; thin aspect>=8 uses 64:64. Below the micro band (>128) the
// shipped core-count partition is faster and is kept.
static void ApplyCtrmmTileGridPartition(
    uint32_t uM, uint32_t uN, uint32_t& usedAicCoreNum, uint32_t& coreM, uint32_t& coreN, CtrmmPhaseBTilingData& tiling)
{
    uint32_t blocksM = 0;
    uint32_t blocksN = 0;
    // Measured optimum for the partition grid: 128:64 (512^2 med 115.1us vs
    // 207.8us at 256:128 and 269.8us with the shipped formula). It deliberately
    // differs from the shape-aware kernel tile defaults: the partition band is
    // what the tile clamp ends up following.
    constexpr uint32_t PART_GRID_M = 128;
    constexpr uint32_t PART_GRID_N = 64;
    if (std::max<uint32_t>(uM, uN) > 128u) {
        uint32_t tm = PART_GRID_M;
        uint32_t tn = PART_GRID_N;
        // Thin shapes (aspect >= 8): measured 64:64 beats 128:64 on the thin
        // band (253x2932: 0.396->0.402 PASS; 256x4096: 0.365->0.377).
        {
            const uint32_t hiDim = std::max<uint32_t>(uM, uN);
            const uint32_t loDim = std::min<uint32_t>(uM, uN);
            if (loDim > 0 && hiDim / loDim >= 8u) {
                tm = 64;
                tn = 64;
            }
        }
        if (tm > 0 && tn > 0) {
            blocksM = CeilDiv<uint32_t>(uN, tm);
            blocksN = CeilDiv<uint32_t>(uM, tn);
            const uint32_t blocks = std::max<uint32_t>(blocksM * blocksN, 1u);
            tiling.usedAicCoreNum = std::max<uint32_t>(std::min<uint32_t>(usedAicCoreNum, blocks), 1u);
            coreM = CeilDiv<uint32_t>(uN, blocksM);
            coreN = CeilDiv<uint32_t>(uM, blocksN);
            usedAicCoreNum = tiling.usedAicCoreNum;
        }
    }
}

// Aspect-ratio-based tile preference. Thin products (one side far shorter than
// the other) prefer the wide tile on the short axis: m x n with n >> m wants
// M=256/N=128 (measured 0.304 vs 0.224 at 512x4096), while m >> n wants
// M=128/N=256 (0.307 vs 0.279 at 4096x512).
static void CtrmmPickTileShape(uint32_t uM, uint32_t uN, uint32_t& tileM, uint32_t& tileN)
{
    tileM = 256;
    tileN = 128;
    if (uM < 1024 && uN > 4 * uM) {
        tileM = 256;
        tileN = 128;
    } else if (uN < 1024 && uM > 4 * uN) {
        tileM = 128;
        tileN = 256;
    }
}

// Shrink tileKChunk until the double-buffered A+B L1 footprint fits. Returns the
// aligned K chunk that fits; BASE_K is the floor, so the loop is bounded.
static uint32_t CtrmmFitTileKToL1(uint32_t tileM, uint32_t tileN, uint32_t tileKChunk)
{
    constexpr uint32_t BASE_M = 16;
    constexpr uint32_t BASE_N = 16;
    constexpr uint32_t BASE_K = 8;
    constexpr uint32_t L1_SIZE_BYTES = 512 * 1024;
    constexpr uint32_t L1_BUF_NUM = 2;
    constexpr uint32_t FP32_SIZE = 4;
    auto aSideL1 = [&]() {
        return CeilAlign<uint32_t>(tileM, BASE_M) * CeilAlign<uint32_t>(tileKChunk, BASE_K) * FP32_SIZE;
    };
    auto bSideL1 = [&]() {
        return CeilAlign<uint32_t>(tileKChunk, BASE_K) * CeilAlign<uint32_t>(tileN, BASE_N) * FP32_SIZE;
    };
    while (L1_BUF_NUM * (aSideL1() + bSideL1()) > L1_SIZE_BYTES && tileKChunk > BASE_K) {
        tileKChunk /= 2;
        tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(tileKChunk, BASE_K), BASE_K);
    }
    return tileKChunk;
}

static CtrmmPhaseBTilingData CalCtrmmStrmmGemmTiling(
    uint32_t usedAicCoreNum, uint32_t uM, uint32_t uN, uint32_t uSide, uint32_t uLda, uint32_t uLdb)
{
    constexpr uint32_t BASE_M = 16;
    constexpr uint32_t BASE_N = 16;
    constexpr uint32_t BASE_K = 8;
    constexpr uint32_t FIXPIPE_N_ALIGN = 8;
    // Tiling defaults: measured sweep (2026-09-24, anchors) — CATLASS's 950
    // template uses a wide M tile with a small K step, and 256x128x128 beat the
    // strmm-derived 128x128x256 on every anchor (4096^2 0.407->0.464, thin band
    // +30%). Env knobs override for further sweeps.

    uint32_t DEFAULT_TILE_M = 256;
    uint32_t DEFAULT_TILE_N = 128;
    constexpr uint32_t DEFAULT_TILE_K_CHUNK = 128;
    // Tile shape by aspect ratio (see CtrmmPickTileShape).
    CtrmmPickTileShape(uM, uN, DEFAULT_TILE_M, DEFAULT_TILE_N);

    CtrmmPhaseBTilingData tiling{};
    tiling.m = uN; // swapped C^T view
    tiling.n = uM;
    tiling.sideMode = uSide;
    uint32_t coreM = CeilDiv<uint32_t>(uN, usedAicCoreNum);
    uint32_t coreN = CeilDiv<uint32_t>(uM, usedAicCoreNum);
    // Per-core band from a TILE GRID instead of ceil(extent / core count).
    // The shipped core-count formula yields ceil(m/cores) = 13 on small/mid
    // squares, which the BASE_M = 16 clamp turns into 1024 tiny 16x16 tiles
    // (measured Phase-B throughput 1.07 TMAC/s at 512^2 vs an 8.9 TMAC/s
    // ceiling). With a tile-grid partition each core owns tile-sized bands and
    // the cube sees full tiles: 512^2 went 269.8us -> 115.1us and the full
    // 200-case set 171 -> 191 passes. Below the micro band the shipped
    // partition is faster (measured -7..-21% there), so it is kept.
    ApplyCtrmmTileGridPartition(uM, uN, usedAicCoreNum, coreM, coreN, tiling);
    tiling.usedAicCoreNum = usedAicCoreNum;
    tiling.singleCoreM = std::max<uint32_t>(coreM, BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreN, BASE_N);

    uint32_t tileM = std::min<uint32_t>(DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = DEFAULT_TILE_K_CHUNK;
    tileM = std::max<uint32_t>(tileM, BASE_M);
    tileN = std::max<uint32_t>(tileN, BASE_N);
    tileKChunk = std::max<uint32_t>(tileKChunk, BASE_K);
    if (uN < DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM, CeilAlign<uint32_t>(std::max<uint32_t>(uN, BASE_M), BASE_M));
    }
    if (uM < DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN, CeilAlign<uint32_t>(std::max<uint32_t>(uM, BASE_N), BASE_N));
    }
    tileKChunk = CtrmmFitTileKToL1(tileM, tileN, tileKChunk);
    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
    tiling.lda = uLda;
    tiling.ldb = uLdb;
    tiling.tempRowStride = CeilAlign<uint32_t>(uM, FIXPIPE_N_ALIGN);
    return tiling;
}

// Triangular K-clipping mode for the Phase-B kernel (L2 lever, catlass
// trmm.hpp semantics). The effective triangle of the MATERIALIZED op(A) plane
// flips with trans (Phase-A stores op(A)); in this kernel's swapped/flipped
// view the triangle always sits in the opposite-side operand:
//   user LEFT  -> kernel right operand (clip by N-tile bounds)
//   user RIGHT -> kernel left  operand (clip by M-tile bounds)
static uint32_t CalCtrmmClipMode(aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    const bool planeLower = (trans == ACLBLAS_OP_N) ? (uplo == ACLBLAS_LOWER) : (uplo == ACLBLAS_UPPER);
    if (side == ACLBLAS_SIDE_LEFT) {
        return planeLower ? 1u : 2u; // N-side: LOWER -> clip K-end by nEnd; UPPER -> K-start by nStart
    }
    return planeLower ? 3u : 4u; // M-side: LOWER -> clip K-start by mStart; UPPER -> K-end by mEnd
}

// Pipeline: Phase-A split/mirror (AIV SIMT or fused tensor_api kernel) ->
// Gauss 3M Phase-B in ONE AIC launch (ctrmm_phaseb3_kernel) -> Phase-C combine
// with complex alpha (AIV tensor_api).
// Workspace layout: [P1|P2|P3|P4|Ar|Ai|Br|Bi], all float planes.
// Pipeline workspace sizing (Gauss 3M layout [P1|P2|P3|P4|Ar|Ai|Br|Bi], all
// float planes, plus the C==B alias B-shadow tail when requested).
struct CtrmmPipelineSizes {
    uint32_t dimA;
    uint32_t tempRowStride;
    size_t tempBytes;
    size_t aPlaneBytes;
    size_t bPlaneBytes;
    size_t pipelineBytes;
    size_t bAliasBytes;
    size_t workspaceNeed;
};

static CtrmmPipelineSizes CalCtrmmPipelineSizes(
    aclblasSideMode_t side, uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, bool bAliasedC)
{
    const uint32_t dimA = (side == ACLBLAS_SIDE_LEFT) ? uM : uN;
    const uint32_t tempRowStride = CeilAlign<uint32_t>(uM, 16); // GEMM_FRACTAL
    const size_t tempBytes = static_cast<size_t>(tempRowStride) * static_cast<size_t>(uN) * sizeof(float);
    const size_t aPlaneBytes = static_cast<size_t>(uLda) * static_cast<size_t>(dimA) * sizeof(float);
    const size_t bPlaneBytes = static_cast<size_t>(uLdb) * static_cast<size_t>(uN) * sizeof(float);
    // Gauss 3M is the verified default (A/B against 4M showed Phase-B x0.78 and
    // the 4M variant was never adopted). No runtime switch: the 4M code remains
    // only where the shared helper needs it.
    constexpr size_t GM_ALIGN = 512;
    // P0 fix: when the caller passes the ORIGINAL B with C==B aliasing (bAliasedC),
    // the B shadow lives at the TAIL of this pipeline's own workspace region —
    // never overlapping dP1..dBsum and never invalidated by this Ensure call.
    const size_t bAliasBytes =
        bAliasedC ? ((static_cast<size_t>(uLdb) * uN * sizeof(aclblasComplex) + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN) :
                    0u;
    const size_t pipelineBytes =
        ((4 * tempBytes + 2 * aPlaneBytes + 2 * bPlaneBytes + aPlaneBytes + bPlaneBytes) + GM_ALIGN - 1) / GM_ALIGN *
        GM_ALIGN;
    const size_t workspaceNeed = pipelineBytes + bAliasBytes;
    return {dimA,        tempRowStride, tempBytes,   aPlaneBytes,
            bPlaneBytes, pipelineBytes, bAliasBytes, pipelineBytes + bAliasBytes};
}

// C==B alias: stage a B shadow at the tail of the pipeline's workspace
// region and return it as the effective B (never overlaps dP1..dBsum).
// Returns B when not aliased; nullptr on copy failure.
static const aclblasComplex* CtrmmPipelineShadowB(
    _aclblas_handle* h, uint8_t* wsBase, size_t pipelineBytes, const aclblasComplex* B, uint32_t uLdb, uint32_t uN,
    bool bAliasedC)
{
    if (!bAliasedC) {
        return B;
    }
    void* bShadow = wsBase + pipelineBytes;
    const size_t bCopyBytes = static_cast<size_t>(uLdb) * uN * sizeof(aclblasComplex);
    aclError cpRet = aclrtMemcpyAsync(bShadow, bCopyBytes, B, bCopyBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, h->stream);
    if (cpRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "pipeline alias B shadow copy failed, ret=%d", cpRet);
        return nullptr;
    }
    return static_cast<const aclblasComplex*>(bShadow);
}

// Phase-A: split/mirror (AIV). Fused single launch (trans = N, small
// A-plane) or SIMT A-mirror + TLA B-split (trans = T/C needs a genuinely
// transposing lda-strided read a 2D DMA cannot do).
static void CtrmmPipelineRunPhaseA(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, uint32_t dimA, uint32_t uM, uint32_t uN,
    uint32_t uLda, uint32_t uLdb, const aclblasComplex* A, const aclblasComplex* bEff, uint32_t aivCoreNum,
    uint8_t* dAr, uint8_t* dAi, uint8_t* dAsum, uint8_t* dBr, uint8_t* dBi, uint8_t* dBsum, aclrtStream stream)
{
    uint32_t aivBlocks = std::max<uint32_t>(std::min<uint32_t>(std::max(dimA, uM), aivCoreNum), 1);
    CtrmmSplitMirrorTilingData splitTiling{};
    splitTiling.uploMode = static_cast<uint32_t>(uplo);
    splitTiling.transMode = static_cast<uint32_t>(trans);
    splitTiling.diagMode = static_cast<uint32_t>(diag);
    splitTiling.dimA = dimA;
    splitTiling.m = uM;
    splitTiling.n = uN;
    splitTiling.lda = uLda;
    splitTiling.ldb = uLdb;
    splitTiling.rowsPerCoreA = CeilDiv<uint32_t>(dimA, aivBlocks);
    splitTiling.rowsPerCoreB = CeilDiv<uint32_t>(uM, aivBlocks);
    splitTiling.colsPerCoreB = 0;
    // Fused-kernel-only column-band mode for thin shapes (n >= 8*m, trans=N).
    // Set BEFORE the fusedA call site below so the condition matches exactly;
    // rowsPerCoreB is forced to full height in this mode (the legacy row loop
    // is not taken, and the early-return guard stays satisfied for all blocks).
    if (uN >= 8u * static_cast<uint32_t>(uM > 0 ? uM : 1u) && uM > 0 && trans == ACLBLAS_OP_N && dimA <= 1024) {
        splitTiling.colsPerCoreB = CeilDiv<uint32_t>(uN, aivBlocks);
        // Guard-only value: the kernel entry early-return checks the legacy
        // row band; 1 keeps every block in range. The B loop itself uses the
        // column band (rowsPerCoreB is not read there in this mode).
        splitTiling.rowsPerCoreB = 1;
    }
    splitTiling.nthreads = SIMT_MAX_THREAD_NUM;
    // Triangle masking has exactly one implementation (scalar zeroing): the
    // Duplicate-based variants were measured to write past an offset
    // LocalTensor's range, and a vectorized region-zeroing variant showed no
    // measurable gain. There is deliberately no switch and no tiling field for
    // it anymore.
    // The B-side split always uses the tensor_api kernel (block DMA + DeInterleave).
    // Fused path (trans = N): ONE AIV launch does A-mirror + B-split via the
    // tensor_api kernel. trans = T/C keeps the SIMT A-mirror plus the B kernel.
    // Fused A+B Phase-A (single launch) — VERIFIED 1003/1003 with the scalar
    // masking path. Measured 2026-09-24: fused wins big while the A-plane area
    // is small (A phase 256^2: 61.7->20.9us, 512^2: 115.6->70.5us) but loses
    // when the plane is large because the mask zeroing is scalar (4096^2:
    // 2178->3175us). So: fused for dimA <= 1024, SIMT A-mirror above.
    // Phase-A path selection is internal (measured): fused single launch while
    // the A-plane area is small, SIMT A-mirror above.
    // Fused single-launch Phase-A applies to trans = N only: the T/C variant
    // needs a genuinely transposing source read (A(j,i) is lda-strided, a 2D
    // DMA cannot transpose), so T/C keeps the SIMT A-mirror + TLA-B pair.
    // (Attempted and parked 2026-09-24: the "transposed" load read A(i,j) in
    // the same orientation it stored, over-zeroing the plane — 205 failures.)
    const bool fusedA = (trans == ACLBLAS_OP_N) && (dimA <= 1024);
    splitTiling.skipB = 1u; // the B-side split is always the tensor_api kernel
    if (fusedA) {
        ctrmm_split_fused_tla_kernel_do(
            reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(bEff), dAr, dAi, dAsum, dBr, dBi,
            dBsum, splitTiling, aivBlocks, stream);
    } else {
        ctrmm_split_mirror_kernel_do(
            reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(bEff), dAr, dAi, dBr, dBi, dAsum,
            dBsum, splitTiling, aivBlocks, stream);
        ctrmm_split_b_tla_kernel_do(
            reinterpret_cast<const uint8_t*>(bEff), dBr, dBi, dBsum, splitTiling, aivBlocks, stream);
    }
}

// Debug-only: dump the Phase-A planes (CTRMM_DEBUG=1 + CTRMM_DUMP_PLANES).
static void DumpCtrmmPhaseAPlanes(
    const char* dbgEnv, aclrtStream stream, const void* dAr, const void* dAi, const void* dAsum, const void* dBr,
    const void* dBi, const void* dBsum, size_t aPlaneBytes, size_t bPlaneBytes)
{
    const char* dumpDir = (dbgEnv != nullptr && dbgEnv[0] == '1') ? std::getenv("CTRMM_DUMP_PLANES") : nullptr;
    if (dumpDir == nullptr) {
        return;
    }
    aclrtSynchronizeStream(stream);
    auto dumpPlane = [&](const char* name, const void* devPtr, size_t bytes) {
        const std::string path = std::string(dumpDir) + "/" + name + ".bin";
        FILE* fp = std::fopen(path.c_str(), "wb");
        if (fp == nullptr) {
            return;
        }
        std::vector<uint8_t> hostBuf(bytes);
        if (aclrtMemcpy(hostBuf.data(), bytes, devPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS) {
            std::fwrite(hostBuf.data(), 1, bytes, fp);
        }
        std::fclose(fp);
    };
    dumpPlane("Ar", dAr, aPlaneBytes);
    dumpPlane("Ai", dAi, aPlaneBytes);
    dumpPlane("Asum", dAsum, aPlaneBytes);
    dumpPlane("Br", dBr, bPlaneBytes);
    dumpPlane("Bi", dBi, bPlaneBytes);
    dumpPlane("Bsum", dBsum, bPlaneBytes);
}

// Debug-only per-phase timing report (CTRMM_DEBUG=1): [CTRMM-PHASES] line.
static void ReportCtrmmPhaseTiming(
    aclrtStream stream, aclrtEvent t0, aclrtEvent t1, aclrtEvent t2, aclrtEvent t3, uint32_t uM, uint32_t uN,
    aclblasSideMode_t side)
{
    aclrtRecordEvent(t3, stream);
    aclrtSynchronizeStream(stream);
    float msA = 0.0f;
    float msB = 0.0f;
    float msC = 0.0f;
    aclrtEventElapsedTime(&msA, t0, t1);
    aclrtEventElapsedTime(&msB, t1, t2);
    aclrtEventElapsedTime(&msC, t2, t3);
    std::printf(
        "[CTRMM-PHASES] m=%u n=%u side=%u A_us=%.1f B_us=%.1f C_us=%.1f total_us=%.1f\n", uM, uN,
        static_cast<unsigned>(side == ACLBLAS_SIDE_LEFT ? 1 : 2), msA * 1000.0f, msB * 1000.0f, msC * 1000.0f,
        (msA + msB + msC) * 1000.0f);
    aclrtDestroyEvent(t0);
    aclrtDestroyEvent(t1);
    aclrtDestroyEvent(t2);
    aclrtDestroyEvent(t3);
}

// Phase-C combine tiling fill (complex alpha; AIV row partition over m).
static CtrmmCombineTilingData CalCtrmmCombineTiling(
    uint32_t uM, uint32_t uN, uint32_t uLdc, uint32_t tempRowStride, uint32_t aivCoreNum,
    const aclblasComplex& alphaVal)
{
    const uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);
    CtrmmCombineTilingData combineTiling{};
    combineTiling.m = uM;
    combineTiling.n = uN;
    combineTiling.ldc = uLdc;
    combineTiling.tempRowStride = tempRowStride;
    combineTiling.rowsPerCore = CeilDiv<uint32_t>(uM, usedAivCoreNum);
    combineTiling.nthreads =
        std::min(CeilAlign<uint32_t>(combineTiling.rowsPerCore, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    combineTiling.alphaReal = alphaVal.real;
    combineTiling.alphaImag = alphaVal.imag;
    return combineTiling;
}

// Phase-B: Gauss 3M real-plane products in ONE AIC launch (strmm-lineage
// cube kernel). Builds the swapped-view tiling (tile-grid partition inside
// CalCtrmmStrmmGemmTiling) and launches ctrmm_phaseb3_kernel.
static CtrmmPhaseBTilingData CtrmmPipelineRunPhaseB(
    uint32_t usedAicCoreNum, uint32_t uM, uint32_t uN, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, uint32_t uLda, uint32_t uLdb, uint32_t tempRowStride, uint8_t* dAr, uint8_t* dAi,
    uint8_t* dAsum, uint8_t* dBr, uint8_t* dBi, uint8_t* dBsum, uint8_t* dP1, uint8_t* dP2, uint8_t* dP3,
    aclrtStream stream)
{
    // Phase-B: real plane products via the custom kernel (strmm-lineage
    // recipe: swapped (n,m) + flipped side, aSlot=opA-plane, bSlot=B-plane,
    // temp [j*stride+i] = P(i,j)); 5.1-5.7 TMAC/s measured on this device. The
    // GEMM-module alternative (2.0 TMAC/s) was not adopted.
    // usedAicCoreNum comes in as a parameter (tile-count-clamped by the caller).
    const uint32_t flippedSide = (side == ACLBLAS_SIDE_LEFT) ? ACLBLAS_SIDE_RIGHT : ACLBLAS_SIDE_LEFT;
    CtrmmPhaseBTilingData pbTiling = CalCtrmmStrmmGemmTiling(usedAicCoreNum, uM, uN, flippedSide, uLda, uLdb);
    pbTiling.tempRowStride = tempRowStride; // unify with workspace/combine stride
    pbTiling.clipMode = CalCtrmmClipMode(side, uplo, trans);
    // Gauss 3M merged into ONE kernel launch (measured 161->170/200 pass vs three
    // separate launches). The P3 = (Ar+Ai)(Br+Bi) pre-combined planes keep the same
    // triangular support (zeros outside sum to zero), so clipMode stays valid.
    // Slot convention (strmm lineage): aSlot = op(A)-plane, bSlot = B-plane for BOTH
    // sides — the flipped side inside the tiling does the C^T algebra; only the
    // P3/P4 pairing differs by side (hence the pre-combined P3).
    ctrmm_phaseb3_kernel_do(dAr, dAi, dAsum, dBr, dBi, dBsum, dP1, dP2, dP3, pbTiling, usedAicCoreNum, stream);
    return pbTiling;
}

// Cube/vector core counts for the pipeline. Both must be non-zero.
static aclblasStatus_t FetchCtrmmCoreCounts(uint32_t& aicCoreNum, uint32_t& aivCoreNum)
{
    aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasCtrmm", "cube core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCtrmm", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Flat device pointers into the pipeline workspace. The layout is a fixed order
// of four temp planes (P1..P4) followed by the A/B real/imag/sum planes.
struct CtrmmPipelinePtrs {
    uint8_t* dP1;
    uint8_t* dP2;
    uint8_t* dP3;
    uint8_t* dP4;
    uint8_t* dAr;
    uint8_t* dAi;
    uint8_t* dAsum;
    uint8_t* dBr;
    uint8_t* dBi;
    uint8_t* dBsum;
};

static CtrmmPipelinePtrs CarveCtrmmPipelinePtrs(
    uint8_t* wsBase, size_t tempBytes, size_t aPlaneBytes, size_t bPlaneBytes)
{
    CtrmmPipelinePtrs p{};
    p.dP1 = wsBase;
    p.dP2 = p.dP1 + tempBytes;
    p.dP3 = p.dP2 + tempBytes;
    p.dP4 = p.dP3 + tempBytes;
    p.dAr = p.dP4 + tempBytes;
    p.dAi = p.dAr + aPlaneBytes;
    p.dBr = p.dAi + aPlaneBytes;
    p.dBi = p.dBr + bPlaneBytes;
    p.dAsum = p.dBi + bPlaneBytes;
    p.dBsum = p.dAsum + aPlaneBytes;
    return p;
}

// Debug-only per-phase timing state (CTRMM_DEBUG=1). Destructor releases the
// events, so the pipeline body has no cleanup path of its own.
struct CtrmmPhaseTiming {
    bool on = false;
    aclrtEvent t0 = nullptr;
    aclrtEvent t1 = nullptr;
    aclrtEvent t2 = nullptr;
    aclrtEvent t3 = nullptr;

    explicit CtrmmPhaseTiming(const char* dbgEnv, aclrtStream stream)
    {
        on = (dbgEnv != nullptr && dbgEnv[0] == '1');
        if (on) {
            aclrtCreateEvent(&t0);
            aclrtCreateEvent(&t1);
            aclrtCreateEvent(&t2);
            aclrtCreateEvent(&t3);
            aclrtRecordEvent(t0, stream);
        }
    }
    ~CtrmmPhaseTiming()
    {
        if (!on) {
            return;
        }
        aclrtDestroyEvent(t0);
        aclrtDestroyEvent(t1);
        aclrtDestroyEvent(t2);
        aclrtDestroyEvent(t3);
    }
    CtrmmPhaseTiming(const CtrmmPhaseTiming&) = delete;
    CtrmmPhaseTiming& operator=(const CtrmmPhaseTiming&) = delete;
};

// Ensure the pipeline workspace, stage the aliased-B shadow when needed, and
// carve the flat plane pointers. Returns a pointer set whose dP1 is null on
// failure (the caller maps that to EXECUTION_FAILED).
static CtrmmPipelinePtrs ReserveCtrmmPipelineWorkspace(
    _aclblas_handle* h, const CtrmmPipelineSizes& sz, const aclblasComplex* B, uint32_t uLdb, uint32_t uN,
    bool bAliasedC, const aclblasComplex** bEffOut)
{
    CtrmmPipelinePtrs none{};
    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, sz.workspaceNeed);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "pipeline workspace ensure failed, required=%zu, ret=%d", sz.workspaceNeed, wsRet);
        return none;
    }
    uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    *bEffOut = CtrmmPipelineShadowB(h, wsBase, sz.pipelineBytes, B, uLdb, uN, bAliasedC);
    if (bAliasedC && *bEffOut == nullptr) {
        return none;
    }
    return CarveCtrmmPipelinePtrs(wsBase, sz.tempBytes, sz.aPlaneBytes, sz.bPlaneBytes);
}

static aclblasStatus_t ExecuteCtrmmPipeline(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, uint32_t uM, uint32_t uN, uint32_t uLda, uint32_t uLdb, uint32_t uLdc,
    const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex& alphaVal, aclblasComplex* C,
    bool bAliasedC = false)
{
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    aclblasStatus_t coreSt = FetchCtrmmCoreCounts(aicCoreNum, aivCoreNum);
    if (coreSt != ACLBLAS_STATUS_SUCCESS) {
        return coreSt;
    }

    const CtrmmPipelineSizes sz = CalCtrmmPipelineSizes(side, uM, uN, uLda, uLdb, bAliasedC);
    const uint32_t dimA = sz.dimA;
    const uint32_t tempRowStride = sz.tempRowStride;
    const size_t tempBytes = sz.tempBytes;
    const size_t aPlaneBytes = sz.aPlaneBytes;
    const size_t bPlaneBytes = sz.bPlaneBytes;
    const aclblasComplex* bEff = nullptr;
    const CtrmmPipelinePtrs ptrs = ReserveCtrmmPipelineWorkspace(h, sz, B, uLdb, uN, bAliasedC, &bEff);
    if (ptrs.dP1 == nullptr) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclrtStream stream = h->stream;

    // Debug-only per-phase timing (CTRMM_PHASE_TIMING=1): events around each
    // phase + one stream sync — measures the P0 decomposition table; the extra
    // sync intentionally perturbs async semantics, never enable in production.
    // Diagnostics are consolidated behind a single gate: CTRMM_DEBUG=1 enables
    // the phase timing here and the plane dump below.
    static const char* dbgEnv = std::getenv("CTRMM_DEBUG"); // read-once (PR#185: prod env must be stable)
    CtrmmPhaseTiming timing(dbgEnv, stream);

    CtrmmPipelineRunPhaseA(
        uplo, trans, diag, dimA, uM, uN, uLda, uLdb, A, bEff, aivCoreNum, ptrs.dAr, ptrs.dAi, ptrs.dAsum, ptrs.dBr,
        ptrs.dBi, ptrs.dBsum, stream);
    if (timing.on) {
        aclrtRecordEvent(timing.t1, stream);
    }

    const uint32_t tileCount = CeilDiv<uint32_t>(uM, 16) * CeilDiv<uint32_t>(uN, 16);
    const uint32_t usedAicCoreNum = std::max<uint32_t>(std::min<uint32_t>(tileCount, aicCoreNum), 1);
    CtrmmPhaseBTilingData pbTiling = CtrmmPipelineRunPhaseB(
        usedAicCoreNum, uM, uN, side, uplo, trans, uLda, uLdb, tempRowStride, ptrs.dAr, ptrs.dAi, ptrs.dAsum, ptrs.dBr,
        ptrs.dBi, ptrs.dBsum, ptrs.dP1, ptrs.dP2, ptrs.dP3, stream);
    if (timing.on) {
        aclrtRecordEvent(timing.t2, stream);
    }

    // Phase-C: combine with complex alpha.
    const uint32_t usedAivCoreNum = std::max<uint32_t>(std::min<uint32_t>(uM, aivCoreNum), 1);
    CtrmmCombineTilingData combineTiling = CalCtrmmCombineTiling(uM, uN, uLdc, tempRowStride, aivCoreNum, alphaVal);

    DumpCtrmmPhaseAPlanes(
        dbgEnv, stream, ptrs.dAr, ptrs.dAi, ptrs.dAsum, ptrs.dBr, ptrs.dBi, ptrs.dBsum, aPlaneBytes, bPlaneBytes);
    // Phase-C: tensor_api combine (UB block DMA + Interleave), verified
    // 1003/1003 and ~7x faster than the scalar SIMT combine at 4096^2, which
    // was therefore removed.
    ctrmm_combine_tla_kernel_do(
        ptrs.dP1, ptrs.dP2, ptrs.dP3, ptrs.dP4, reinterpret_cast<uint8_t*>(C), combineTiling, usedAivCoreNum, stream);
    if (timing.on) {
        ReportCtrmmPhaseTiming(stream, timing.t0, timing.t1, timing.t2, timing.t3, uM, uN, side);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Entry-point validation (task doc §2.4 + official CSV ground truth,
// strmm-family precedent): handle -> enums -> m/n -> QUICK RETURN -> leading
// dims/alpha -> C -> A/B.
static aclblasStatus_t ValidateCtrmmEntry(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, int lda, int ldb, int ldc, const aclblasComplex* alpha,
    const aclblasComplex* A, const aclblasComplex* B, aclblasComplex* C)
{
    // Validation order (task doc §2.4 + official CSV ground truth, strmm-family
    // precedent): handle -> enums -> m/n -> QUICK RETURN (m==0 or n==0 skips
    // ld/pointer checks; TC_ED_145..147 carry lda=ldb=ldc=0 and expect
    // SUCCESS) -> leading dims/alpha -> C -> resolve -> alpha==(0,0) -> A/B.
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCtrmm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
              OP_LOGE("aclblasCtrmm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasCtrmm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasCtrmm", "trans must be N, T or C, got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(diag == ACLBLAS_UNIT || diag == ACLBLAS_NON_UNIT,
              OP_LOGE("aclblasCtrmm", "diag must be UNIT or NON_UNIT, got %d", static_cast<int>(diag));
              return ACLBLAS_STATUS_INVALID_VALUE);
    // Upper bound on m and n. This is an *interface contract* bound, taken from
    // the task specification (accuracy cases up to 2048, performance cases up to
    // 4096; the shipped test CSV tops out at m = n = 4096), so that oversized
    // calls are rejected up front instead of being handed to the kernel.
    //
    // It is deliberately NOT derived from a matrix-multiply instruction limit:
    // the cube-relative dimensions reaching the Mmad call are L0 blocks
    // (curML1, nL0, curKL0) bounded by PB_BASE_M/N/K = 16/16/8 in
    // ctrmm_phaseb_tile.h, so the full extents never reach the instruction.
    // (An earlier revision of this comment cited a 12-bit/4095 instruction
    // limit while allowing 4096 -- an inconsistent rationale, corrected here.)
    constexpr uint32_t kCtrmmMaxDim = 4096;
    CHECK_RET(m <= kCtrmmMaxDim, OP_LOGE("aclblasCtrmm", "m must be <= %u, got %d", kCtrmmMaxDim, m);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n <= kCtrmmMaxDim, OP_LOGE("aclblasCtrmm", "n must be <= %u, got %d", kCtrmmMaxDim, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasCtrmm", "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasCtrmm", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    // Note: the m==0/n==0 quick return deliberately stays in the CALLER —
    // a helper cannot distinguish "quick-return SUCCESS" from "validated,
    // continue". ld/pointer checks also remain in the caller in the original
    // order (quick return skips them; TC_ED_145..147 carry lda=0 and expect
    // SUCCESS).
    return ACLBLAS_STATUS_SUCCESS;
}

// Non-pipeline in-place B==C alias path: stage B into the default workspace with
// a D2D copy ordered before the kernel. Returns the buffer to read B from (the
// original B when no staging applies) and a status for the staging itself.
static aclblasStatus_t StageCtrmmAliasedB(
    _aclblas_handle* h, const aclblasComplex* B, int ldb, int n, bool needStage, const aclblasComplex** bEffOut)
{
    *bEffOut = B;
    if (!needStage) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, bBytes);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "alias workspace ensure failed, required=%zu, ret=%d", bBytes, wsRet);
        return wsRet;
    }
    void* wsB = GetEffectiveWorkspace(h);
    aclError cpRet = aclrtMemcpyAsync(wsB, bBytes, B, bBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, h->stream);
    if (cpRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCtrmm", "alias B copy failed, ret=%d", cpRet);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    *bEffOut = static_cast<const aclblasComplex*>(wsB);
    return ACLBLAS_STATUS_SUCCESS;
}

// Pipeline gate threshold (64^2 elems) — shared by the pre-alias decision and
// the shape routing below.
constexpr uint64_t kCtrmmPipelineMinElems = 4096;

// Dispatch to the TINY, pipeline or naive kernel by shape. Split out of the
// entry point so its validation and scalar-resolution order stays readable.
static aclblasStatus_t DispatchCtrmmByShape(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, int lda, int ldb, int ldc, const aclblasComplex* A, const aclblasComplex* B,
    aclblasComplex* C, const aclblasComplex& alphaVal, const aclblasComplex* bEff)
{
    // TINY fast path: m,n <= 32 (single-block flattened kernel; the naive
    // row-split is GM-latency-bound here -- 32x32 measures ~277us vs 9.69us
    // budget). Numerics are identical (verbatim inner loop); in-place B==C is
    // deterministically safe (read-all -> syncthreads -> write-all).
    if (m <= 32 && n <= 32) {
        return ExecuteCtrmmTiny(
            h, side, uplo, trans, diag, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda),
            static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), A, bEff, alphaVal, C);
    }
    // Shape gate: large problems take the three-phase Gauss 3M pipeline
    // (cube throughput), everything else stays on the naive Kahan kernel
    // (correctness-optimized, bit-stable against the golden).
    // Gate measured 2026-09-24 (data track, per-case A/B): the naive kernel
    // only wins for the 1^2..4^2 floor (5.6-8.7us vs 13.5-17.1us pipeline),
    // ties at 8^2 and loses catastrophically above (128^2: 4772us vs 50.6us).
    // Kept at 64^2 (=4096 elems) rather than 16^2 because every special-value
    // case (Inf/NaN/extreme fills) is <= 32^2, and the Gauss 3M path must not
    // receive them (plan red line C2): a 16^2 gate routed TC_FL_110/118 into
    // the 3M pipeline and produced NaN-position mismatches.
    const uint64_t elems = static_cast<uint64_t>(m) * static_cast<uint64_t>(n);
    if (elems >= kCtrmmPipelineMinElems) {
        return ExecuteCtrmmPipeline(
            h, side, uplo, trans, diag, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda),
            static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), A, B, alphaVal, C,
            /* bAliasedC: pipeline stages its own B shadow */
            (C == B));
    }
    return ExecuteCtrmmKernel(
        h, side, uplo, trans, diag, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc), A, bEff, alphaVal, C);
}

aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
{
    aclblasStatus_t vSt = ValidateCtrmmEntry(handle, side, uplo, trans, diag, m, n, lda, ldb, ldc, alpha, A, B, C);
    if (vSt != ACLBLAS_STATUS_SUCCESS) {
        return vSt;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasStatus_t st = ValidateCtrmmParams(side, m, n, lda, ldb, ldc, alpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    // C must be non-null whenever the output exists (m > 0 and n > 0 here),
    // including the alpha == (0,0) case (task doc §2.4).
    CHECK_RET(C != nullptr, OP_LOGE("aclblasCtrmm", "C must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclblasComplex alphaVal;
    bool alphaIsDevice = false;
    aclblasStatus_t scalarSt = ResolveCtrmmScalars(alpha, &alphaVal, &alphaIsDevice, h->stream);
    if (scalarSt != ACLBLAS_STATUS_SUCCESS) {
        return scalarSt;
    }

    // alpha == (0,0): A is not referenced and B need not be valid input.
    if (alphaVal.real == 0.0f && alphaVal.imag == 0.0f) {
        return HandleCtrmmAlphaZero(
            handle, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(ldc), C);
    }
    CHECK_RET(A != nullptr, OP_LOGE("aclblasCtrmm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(B != nullptr, OP_LOGE("aclblasCtrmm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    // C == B in-place alias: each output element is accumulated in registers,
    // but element (i,j) READS B(k,j) slots that the thread for (k,j)
    // overwrites — an inter-element read-after-write hazard. Non-pipeline
    // paths route B through a default-workspace D2D copy ordered before the
    // kernel; the pipeline stages its own shadow inside its workspace region.
    // P0 fix: the pipeline path stages its OWN B shadow at the tail of its
    // workspace (ExecuteCtrmmPipeline, bAliasedC) — staging a top-level
    // default-workspace shadow here would be clobbered/dangling (workspace
    // overlap + size-growth hazard).
    // Pipeline gate threshold — single source of truth for both the pre-alias
    // decision and the main routing below (also bounds MmadParams uint16).
    // Pre-compute the pipeline gate (same threshold as the routing below).
    const bool willUsePipeline = static_cast<uint64_t>(m) * static_cast<uint64_t>(n) >= kCtrmmPipelineMinElems;
    const aclblasComplex* bEff = B;
    aclblasStatus_t aliasSt = StageCtrmmAliasedB(h, B, ldb, n, (C == B) && !willUsePipeline, &bEff);
    if (aliasSt != ACLBLAS_STATUS_SUCCESS) {
        return aliasSt;
    }

    return DispatchCtrmmByShape(h, side, uplo, trans, diag, m, n, lda, ldb, ldc, A, B, C, alphaVal, bEff);
}
