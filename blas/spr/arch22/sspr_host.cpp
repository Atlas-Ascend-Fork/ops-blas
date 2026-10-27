/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sspr_tiling_data.h"
#include "sspr_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// arch22 (DAV_2201) host for aclblasSspr. arch22 contract notes:
// 1. uplo illegal enum returns ACLBLAS_STATUS_INVALID_ENUM (arch22 contract, not
//    ACLBLAS_STATUS_INVALID_VALUE);
// 2. core count comes from GetAivCoreCount() dynamically (never hardcoded), with column-level
//    multi-core split and the per-core >= 4KB shrink rule (n <= 44 degenerates to single core).

// Validation order (mandatory): uplo enum -> incx == 0 / INT_MIN -> alpha / x / ap nullptr
// (n > 0). (handle nullptr / n < 0 / n == 0 quick return are handled in the entry, before this
// function.)
static aclblasStatus_t ValidateSsprParams(
    aclblasFillMode_t uplo, int incx, const float* alpha, const float* x, const float* ap)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasSspr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSspr", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incx != INT_MIN, OP_LOGE("aclblasSspr", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSspr", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspr", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspr", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Naive/exact band boundary of the largeMode UB budget mirror (SsprUbBytesLarge).
constexpr uint32_t SSPR_DESC_MAX_N = 2048;  // keep in sync with kernel SSPR_NAIVE_MAX_N

// ------------------------------------------------------------------------------------------
// Contract: keep the !IS_INCX1 resident path under the 192KB UB ceiling or the kernel silently
// no-ops. On SsprUbBytes overflow shrink the window ubFormer in 64-element steps until the budget
// closes; when even the 64-element minimum window cannot fit xPhys, degrade to the single-core
// scalar path via SsprTilingData::scalarMode.
// ------------------------------------------------------------------------------------------
constexpr uint64_t SSPR_UB_CEIL_BYTES = 192ULL * 1024ULL;  // UB hardware ceiling
// Total UB bytes of the !IS_INCX1 resident path at window size ubf (see below).
// SSPR_UB_FIXED_BYTES includes +32B for the xgBuf pad-corner slack
// (the kernel allocates xgBuf with +8 elems; see sspr_kernel.cpp InitBuffer)
// and +1280B = 5 slack units x SSPR_GUARD_SLACK_ELEMS x 4B for the Phase D guard
// tail slack (apOut queue depth 2 x 64 + c 64 + sc 64 + tl 64 elements; sspr_tiling_data.h
// SSPR_GUARD_SLACK_ELEMS).
constexpr uint64_t SSPR_UB_FIXED_BYTES = 3648ULL + 96ULL + 32ULL + 5ULL * 64ULL * 4ULL;
        // run/fix descriptors + win table + xg slack + Phase D guard tail slack

// Total UB bytes of the !IS_INCX1 resident path at window size ubf: apIn(2) + apOut(2) + xg +
// th + tl + e + s + c = 10 window-sized regions plus the idx buffer max(ubf, n), the fixed
// buffers, xRes (n) and the raw x window (xPhysLen; pass 0 for the IS_INCX1 case - the formula
// then over-estimates, which is safe: the guard is only consulted for incx != 1).
inline uint64_t SsprUbBytes(uint64_t n, uint64_t xPhysLen, uint64_t ubf)
{
    const uint64_t winRegions = 10ULL;
    const uint64_t xi = (ubf > n) ? ubf : n;
    return (winRegions * ubf + xi) * 4ULL + SSPR_UB_FIXED_BYTES + n * 4ULL + xPhysLen * 4ULL;
}

inline uint64_t SsprUbBytesLarge(uint64_t n)
{
    // largeMode (column-ownership tile path). UB budget MIRROR of the kernel Init
    // allocation of sspr_kernel_large.cpp - the two must not drift, a UB over-allocation
    // silently no-ops the kernel on this silicon. Layout (float32 elements), by band
    // (th/tl hoisted to per-tile buffers T each; head scratch 72; guard mask has its
    // own (seg + 64) buffer):
    //   naive (n <= SSPR_DESC_MAX_N, seg = SSPR_LARGE_SEG_FLOATS_NAIVE, no guard):
    //     xTile(T) + xSeg(seg+8) + headIdx(seg+8) + xi(seg) + apIn 2x(seg+8) + apOut 2x(seg)
    //       = T + 7*seg + 32
    //   exact (n > SSPR_DESC_MAX_N, seg = SSPR_LARGE_SEG_FLOATS, + guard):
    //     xTile(T) + xSeg(seg+8) + headIdx(seg+8) + xi(seg) + th(seg) + tl(seg+64)
    //     + e(seg+64) + c(seg+64) + apIn 2x(seg+8) + apOut 2x(seg+64)
    //       = T + 11*seg + 352
    // (xSeg/headIdx are the head Gather dst/idx at the 64-align granularity; tl doubles
    // as the guard compare mask.)
    // (the +8 slots are the house-style padded copy-in slack; the +64 slots are the
    // Phase D guard tail slack, same pattern as SSPR_UB_FIXED_BYTES).
    if (n <= SSPR_DESC_MAX_N) {
        // xTile(T) + xSeg + headIdx (segNaive + 8 each, at the 64-align head-copy granularity)
        // + xi + apIn 2x(segNaive + 8) + apOut 2x(segNaive) = T + 7*segNaive + 32
        return 4ULL * (SSPR_LARGE_TILE_FLOATS + 7ULL * SSPR_LARGE_SEG_FLOATS_NAIVE + 4ULL * 8ULL);
    }
    // xSeg + headIdx (seg + 8 each) + guard tail slack (tl + e + c + the 2 apOut queue
    // slots, 64 each) + the 2 apIn copy-in pads
    constexpr uint64_t kExactSlack = 5ULL * 64ULL + 4ULL * 8ULL;  // = 352
    return 4ULL * (SSPR_LARGE_TILE_FLOATS + 11ULL * SSPR_LARGE_SEG_FLOATS + kExactSlack);
}

// largeMode gate predicate on the SHAPE alone. BOTH fill modes take the column-ownership tile
// path above the threshold; below it the walk path still wins. The mirrored UB budget is a
// hard check: on overflow the shape falls back to the legacy route (walk
// n <= SSPR_X_RESIDENT_MAX_FLOATS / scalar above) instead of a silent no-op.
static bool SsprLargeGateEligible(uint64_t n, int incx)
{
    return incx == 1 && n > SSPR_LARGE_MIN_N_UPPER && n <= SSPR_LARGE_MAX_N &&
        SsprUbBytesLarge(n) <= SSPR_UB_CEIL_BYTES;
}

// Core-count determination of CalSsprTilingData.
static uint32_t SsprChooseCoreNum(uint32_t aivCoreNum, uint64_t nU64, uint32_t uplo, int incx)
{
    // Multi-core vector kernel: use all AIV cores (bounded by the per-core 4KB share rule).
    // The vector path has no scalar GM access and 32B-row-aligned MTE3 writes only (lossless
    // across cores). n > resident cap degrades to the single-core scalar legacy path - EXCEPT
    // largeMode-eligible shapes: the column-ownership tile path has no resident x and
    // runs the full core count (the WeightedColumnBoundary column split needs every part;
    // shrinking parts here would starve the split).
    uint32_t useCoreNum = aivCoreNum;
    if (nU64 > SSPR_X_RESIDENT_MAX_FLOATS && !SsprLargeGateEligible(nU64, incx)) {
        useCoreNum = 1U;
    }
    useCoreNum = std::min(useCoreNum, SSPR_MAX_CORES);  // bounds array capacity guard
    // Per-core minimum data volume 4KB (MIN_TILING_BITS = 32768 bits): AP total bytes =
    // n(n+1)/2 * 4B = 2n(n+1); shrink core count while each core's share < 4KB, floor = 1 core.
    // n <= 44 (2n(n+1) < 4096) converges to the floor: single-core fallback, never rejected.
    uint64_t apTotalBytes = 2ULL * nU64 * (nU64 + 1U);
    uint32_t maxCoresByBytes = static_cast<uint32_t>(apTotalBytes / SSPR_MIN_CORE_BYTES);
    if (maxCoresByBytes == 0) {
        maxCoresByBytes = 1;
    }
    useCoreNum = std::min(useCoreNum, maxCoresByBytes);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    return useCoreNum;
}

// Cost-balanced split weight selection of CalSsprTilingData.
static void SsprChooseRunCostW(SsprTilingData& tilingData, uint64_t nU64)
{
    // Cost-balanced split weight (equivalent elements per crossed column).
    if (nU64 < 768ULL) {
        tilingData.runCostW = 400U;
    } else if (nU64 < 1536ULL) {
        tilingData.runCostW = 700U;
    } else if (nU64 < 3072ULL) {
        tilingData.runCostW = 750U;
    } else {
        tilingData.runCostW = 850U;
    }
}

// Cost-balanced coreBegin split of CalSsprTilingData.
static void SsprComputeCoreBegin(SsprTilingData& tilingData)
{
    // Cost-balanced split, computed host-side (the device-side solve miscompiles):
    // cost prefix C(x) = x + W * (#columns with ColStart(col) < x); boundary_i = smallest
    // 8-aligned x with C(x) >= i*C_total/K. Columns walked incrementally (targets rise).
    const uint64_t w = tilingData.runCostW;
    const uint64_t k = tilingData.useCoreNum;
    const uint64_t n = tilingData.n;
    const bool upper = (tilingData.uplo == ACLBLAS_UPPER);
    auto colStart = [upper, n](uint64_t c) -> uint64_t {
        return upper ? (c * (c + 1U) / 2U) : (c * (2U * n - c + 1U) / 2U);
    };
    tilingData.coreBegin[0] = 0U;
    tilingData.coreBegin[k] = static_cast<uint32_t>(tilingData.apLen);
    uint64_t prev = 0;
    uint64_t col = 0;
    for (uint64_t i = 1; i < k; ++i) {
        const uint64_t target = (tilingData.apLen + w * n) * i / k;
        while (col + 1U < n && colStart(col + 1U) + w * (col + 1U) <= target) {
            ++col;
        }
        uint64_t x = target - w * col;  // in [ColStart(col), ColStart(col+1) + w)
        x = (x + 7U) & ~7ULL;
        if (x <= prev) {
            x = prev + 8U;
        }
        if (x > tilingData.apLen) {
            x = tilingData.apLen;
        }
        tilingData.coreBegin[i] = static_cast<uint32_t>(x);
        prev = x;
    }
}

// UB-budget clamp of CalSsprTilingData (useCoreNum crosses by
// reference - the scalarMode degrade forces the single-core split).
static void SsprClampUbWindow(SsprTilingData& tilingData, uint64_t nU64, int incx, uint32_t nU32,
    uint32_t& useCoreNum)
{
    if (nU64 <= SSPR_X_RESIDENT_MAX_FLOATS) {
        // UB budget guard for the resident path (see the SsprUbBytes
        // note above). Shrink the window until the 192KB budget closes; if even the
        // 64-element minimum window cannot fit the raw x window, degrade to the single-core
        // scalar legacy path (correct for any xPhysLen).
        // The guard covers BOTH incx paths - the incx == 1 case
        // passes xPhysLen = 0 (no raw x window rides in UB there) and the same clamp plus
        // scalarMode degrade apply.
        uint64_t xPhys = (incx == 1) ? 0ULL : tilingData.xPhysLen;
        uint64_t ubf = tilingData.ubFormer;
        while (ubf > 64ULL && SsprUbBytes(nU64, xPhys, ubf) > SSPR_UB_CEIL_BYTES) {
            ubf -= 64ULL;
        }
        if (SsprUbBytes(nU64, xPhys, ubf) > SSPR_UB_CEIL_BYTES) {
            tilingData.scalarMode = 1U;
            useCoreNum = 1U;
            tilingData.useCoreNum = 1U;
            tilingData.columnsPerCore = nU32;
        } else {
            tilingData.ubFormer = static_cast<uint32_t>(ubf);
        }
    }
}

static SsprTilingData CalSsprTilingData(uint32_t aivCoreNum, int n, aclblasFillMode_t uplo, float alpha, int incx)
{
    SsprTilingData tilingData{};
    uint64_t nU64 = static_cast<uint64_t>(n);
    uint32_t nU32 = static_cast<uint32_t>(n);
    const uint32_t uploU32 = static_cast<uint32_t>(uplo);

    uint32_t useCoreNum = SsprChooseCoreNum(aivCoreNum, nU64, uploU32, incx);

    tilingData.n = nU64;
    tilingData.incx = static_cast<int64_t>(incx);
    tilingData.uplo = static_cast<uint32_t>(uplo);
    tilingData.alpha = alpha;
    uint64_t absIncx = (incx > 0) ? static_cast<uint64_t>(incx) : static_cast<uint64_t>(-static_cast<int64_t>(incx));
    tilingData.xPhysLen = 1ULL + (nU64 - 1U) * absIncx;
    tilingData.apLen = nU64 * (nU64 + 1U) / 2U;
    tilingData.useCoreNum = useCoreNum;
    tilingData.columnsPerCore = CeilDiv<uint32_t>(nU32, useCoreNum);
    // Batched-phase window with ubFormer=3584 for the incx==1 branch (fewer
    // per-window MTE dispatches + flag round trips; UB ~174KB via buffer aliasing). The
    // incx!=1 branch keeps 2048 (xPhys buffer + oversized idx in the UB budget).
    tilingData.ubFormer = (incx == 1) ? SSPR_UB_FORMER_INCX1 : SSPR_UB_FORMER;
    SsprClampUbWindow(tilingData, nU64, incx, nU32, useCoreNum);
    tilingData.ubLoopMax = CeilDiv<uint32_t>(nU32, SSPR_UB_FORMER);
    tilingData.ubTailMax =
        std::min(SSPR_UB_FORMER, nU32 - (tilingData.ubLoopMax - 1U) * SSPR_UB_FORMER);
    tilingData.debug = 0U;  // ABI-stable field: pinned to 0 (no debug knob)
    SsprChooseRunCostW(tilingData, nU64);
    if (tilingData.runCostW > 0U && tilingData.useCoreNum >= 1U) {
        SsprComputeCoreBegin(tilingData);
    }
    tilingData.padMinN = SSPR_PAD_MIN_N;
    return tilingData;
}

aclblasStatus_t aclblasSspr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx, float* ap)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSspr", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(n >= 0, OP_LOGE("aclblasSspr", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSsprParams(uplo, incx, alpha, x, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // alpha is a Host-memory float32 scalar pointer: dereference directly into tiling,
    // no Device read-back. Pointer checks above precede this branch, so
    // alpha == 0 quick return keeps ap bitwise unchanged and never reads x (even NaN/Inf).
    float alphaVal = *alpha;
    if (alphaVal == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSspr", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    SsprTilingData tiling = CalSsprTilingData(aivCoreNum, n, uplo, alphaVal, incx);
    // largeMode gate: column-ownership tile path for large-n incx==1, BOTH fill modes.
    // scalarMode == 0 is guaranteed for incx == 1 (the walk UB clamp never degrades that
    // branch) but re-checked as a second line of defence: scalarMode with a multi-core launch
    // would multiply-apply the scalar RMW. A gate refusal leaves the shape on its legacy
    // route: walk (n <= resident cap) / scalar above.
    if (tiling.scalarMode == 0U && SsprLargeGateEligible(tiling.n, incx)) {
        tiling.largeMode = 1U;
    }

    OP_LOGI(
        "aclblasSspr",
        "launching kernel: useCoreNum=%u, columnsPerCore=%u, n=%d, incx=%d, ubFormer=%u, "
        "large=%u",
        tiling.useCoreNum, tiling.columnsPerCore, n, incx, tiling.ubFormer,
        static_cast<unsigned>(tiling.largeMode));

    aclrtStream stream = h->stream;
    // x crosses the boundary as const - the kernel is strictly read-only on x
    // (see sspr_kernel.cpp), so no const_cast exists on this path.
    sspr_kernel_do(reinterpret_cast<const uint8_t*>(x), reinterpret_cast<uint8_t*>(ap),
        nullptr, tiling, tiling.useCoreNum, stream);

    return ACLBLAS_STATUS_SUCCESS;
}
