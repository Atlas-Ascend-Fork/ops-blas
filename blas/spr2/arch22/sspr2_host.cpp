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
 * \file sspr2_host.cpp
 * \brief aclblasSspr2 Host implementation for Ascend910B3 (DAV_2201 / arch22).
 *
 *        Kernel-direct handle-style BLAS (no aclnn graph framework): parameter
 *        validation + no-op short-circuit + tiling computation + single AIV
 *        kernel launch on the handle stream. alpha is a HOST-memory pointer
 *        and is dereferenced directly (zero Device round trip); n = 0 and
 *        alpha = 0 are quick returns per Netlib sspr2.f and never launch.
 *
 *        Validation order (order
 *        sensitive: n = 0 quick-returns BEFORE any pointer check; alpha is
 *        dereferenced only after its nullptr check):
 *          1. handle nullptr                        -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
 *          2. uplo not U/L                          -> ACLBLAS_STATUS_INVALID_ENUM
 *          3. n < 0 / n > max / inc == 0 / INT_MIN /
 *             1+(n-1)*|inc| > 12288   -> ACLBLAS_STATUS_INVALID_VALUE
 *             (shape bound LX/LY range [0, 12288]; the length
 *             formula evaluates <= 12288 for n == 0 with any |inc|, so the
 *             n = 0 no-op below is unaffected)
 *          4. n == 0                                -> ACLBLAS_STATUS_SUCCESS
 *             (legal no-op, no launch, x/y/AP/alpha not referenced)
 *          5. alpha nullptr                         -> ACLBLAS_STATUS_INVALID_VALUE
 *          6. alpha == 0.0f                         -> ACLBLAS_STATUS_SUCCESS
 *             (quick return, x/y/AP not referenced, AP unchanged, no launch)
 *          7. x/y/AP nullptr (n > 0, alpha != 0)    -> ACLBLAS_STATUS_INVALID_VALUE
 *          8. compute tiling, single launch         -> ACLBLAS_STATUS_SUCCESS
 *             (asynchronous; the caller synchronizes the stream before readback)
 *        Note: the handle-style API receives raw device pointers without shape
 *        metadata, so a packed-length-vs-n check cannot be performed here;
 *        the length contract is enforced by the test layer.
 */

#include <algorithm>
#include <climits>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "sspr2_tiling_data.h"
#include "sspr2_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static const char* OP_TAG = "aclblasSspr2";

// AIV core count query hook. The count is a per-process platform property (the
// CANN PlatformAscendCManager is itself a process-wide singleton), so the value
// is cached after the first call: the handle-style BLAS API launches once per
// invocation and an uncached query costs ~400ns of host latency per call.
// Declared weak so a UT mock can override it with a strong symbol and keep its
// per-test core-count semantics (including the 0 -> fallback warning below)
// without caching across test cases.
__attribute__((weak)) uint32_t Sspr2AivCoreCountCached()
{
    static const uint32_t cached = GetAivCoreCount();
    return cached;
}

// x/y packing path selector: 0 = scalar/direct packing (safe default);
// 1 = GatherMask stride-2 fast path probe (disabled in the production
// configuration; falls back to 0).
static const uint8_t SSPR2_PACK_MODE_DEFAULT = 0;

// Shape-bound length check: 1+(n-1)*|inc| <= SSPR2_ARCH22_MAX_VEC_LEN
// (= 12288, LX/LY range [0, 12288]). Pure int64 arithmetic, safe for every
// n in [0, MAX_N] and |inc| <= INT_MAX (INT_MIN is rejected by the caller).
// n == 0 evaluates the formula to 1-|inc| <= 12288 for every |inc|, so the
// legal no-op quick return is unaffected. For n >= 1 the bound pins
// (n-1)*|inc| <= 12287, which keeps the kernel-side uint32 physical-window
// products (axU*(n-1u), pos) free of wraparound. This is a length gate, NOT
// an |inc|-only gate: the shape contract allows |inc| >
// SSPR2_ARCH22_MAX_ABS_INC for smaller n (e.g. n == 2 admits |inc| <= 12287).
static bool Sspr2VecLenInBounds(int n, int inc)
{
    const int64_t absInc = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    return 1LL + (static_cast<int64_t>(n) - 1LL) * absInc <=
        static_cast<int64_t>(SSPR2_ARCH22_MAX_VEC_LEN);
}

// Steps 2-3 of the validation order: enum and value-range checks only (no
// pointer dereference; safe for any n including n == 0 and n < 0).
static aclblasStatus_t ValidateSspr2ValueParams(aclblasFillMode_t uplo, int n, int incx, int incy)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE(OP_TAG, "uplo must be ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        n <= static_cast<int>(SSPR2_ARCH22_MAX_N),
        OP_LOGE(OP_TAG, "n must be <= %u (spec shape_constraints), got %d", SSPR2_ARCH22_MAX_N, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE(OP_TAG, "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE(OP_TAG, "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE(OP_TAG, "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != INT_MIN, OP_LOGE(OP_TAG, "incy must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    // Shape-bound upper check (before the n == 0 no-op): 1+(n-1)*|inc| <=
    // 12288 per vector. n == 0 passes for any |inc| (formula <= 12288),
    // keeping the no-op semantics below intact.
    CHECK_RET(Sspr2VecLenInBounds(n, incx),
        OP_LOGE(OP_TAG, "1+(n-1)*|incx| must be <= %u (spec shape_constraints), n=%d incx=%d",
            SSPR2_ARCH22_MAX_VEC_LEN, n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(Sspr2VecLenInBounds(n, incy),
        OP_LOGE(OP_TAG, "1+(n-1)*|incy| must be <= %u (spec shape_constraints), n=%d incy=%d",
            SSPR2_ARCH22_MAX_VEC_LEN, n, incy);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Step 5 of the validation order: alpha nullptr check (only reached with
// n > 0; alpha is dereferenced right after this check).
static aclblasStatus_t ValidateSspr2AlphaPtr(const float* alpha)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Step 7 of the validation order: x/y/AP pointer checks (only reached with
// n > 0 AND alpha != 0: alpha == 0 is a no-op that never
// references x/y/AP, so a null x/y/AP is legal there and must not reject).
static aclblasStatus_t ValidateSspr2PtrParams(const float* x, const float* y, const float* ap)
{
    CHECK_RET(x != nullptr, OP_LOGE(OP_TAG, "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE(OP_TAG, "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE(OP_TAG, "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static Sspr2TilingData CalSspr2TilingData(int n, aclblasFillMode_t uplo, float alphaVal, int incx, int incy)
{
    Sspr2TilingData tilingData{};
    tilingData.n = static_cast<int64_t>(n);
    tilingData.uploMode = static_cast<int32_t>(uplo);
    tilingData.ax = (incx >= 0) ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    tilingData.ay = (incy >= 0) ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    tilingData.xIsNeg = (incx < 0) ? 1U : 0U;
    tilingData.yIsNeg = (incy < 0) ? 1U : 0U;
    tilingData.packMode = SSPR2_PACK_MODE_DEFAULT;
    tilingData.segTile = SSPR2_ARCH22_SEG_TILE;
    tilingData.alphaVal = alphaVal;
    // never hard-coded. n >= 1 here (n == 0 quick-returned), so at least 1 core.
    uint32_t aivCoreNum = Sspr2AivCoreCountCached();
    if (aivCoreNum == 0) {
        OP_LOGW(OP_TAG, "GetAivCoreCount returned 0, fallback to 1 core");
        aivCoreNum = 1U;
    }
    // Small-n core thrift: block dispatch + per-core fixed cost scale with
    // the core count while the per-core vector work of a mid-size n stays
    // small, so mid-size n caps the count at ceil(n / CORE_COLS_TARGET)
    // columns per core. n >= 768 still resolves to GetAivCoreCount() (the
    // full core count); tiny n keeps usedCoreNum = n.
    uint32_t nCores = std::min(aivCoreNum, static_cast<uint32_t>(n));
    const uint32_t colCap =
        (static_cast<uint32_t>(n) + SSPR2_ARCH22_CORE_COLS_TARGET - 1u) / SSPR2_ARCH22_CORE_COLS_TARGET;
    nCores = std::min(nCores, colCap);
    // Class-major route: LOWER + direct-inc (incx == incy == 1) + n < 1536
    // flips tiling.classMode to 1 and routes the kernel into
    // Sspr2ClassMajorRun. The class path's core rule is DECOUPLED from the
    // colCap formula: 8/16 two-band cap min(aiv, n, (n < 448) ? 8 : 16) --
    // above the band point the uniform 2-way class split favors 16 cores
    // (intermediate counts are load-skewed and slower), below it the
    // 8-core point stays within noise. Kernel side: C <= 8 runs strided
    // classes {b, b+C, ...}, C > 8 splits class b % 8 into Nc contiguous
    // column sub-segments (uniform 2-way at C = 16). UPPER, n >= 1536 and
    // non-direct-inc keep the column-cap rule.
    const bool classMode = (uplo == ACLBLAS_LOWER && incx == 1 && incy == 1 && n < 1536);
    if (classMode) {
        const uint32_t band = (static_cast<uint32_t>(n) < 448u) ? 8U : 16U;
        nCores = std::min(std::min(aivCoreNum, static_cast<uint32_t>(n)), band);
    }
    if (nCores == 0) {
        OP_LOGW(OP_TAG, "core count resolved to 0, fallback to 1 core");
        nCores = 1U;
    }
    tilingData.usedCoreNum = nCores;
    tilingData.classMode = classMode ? 1U : 0U;
    return tilingData;
}

extern "C" aclblasStatus_t aclblasSspr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* x, int incx,
    const float* y, int incy, float* AP)
{
    // 1. handle (first check)
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = ValidateSspr2ValueParams(uplo, n, incx, incy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 4. n == 0: legal no-op, quick return BEFORE any pointer check; x/y/AP /
    // alpha are not referenced and no kernel is launched.
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 5. alpha nullptr (n > 0; alpha is host memory, dereferenced after this
    // check).
    st = ValidateSspr2AlphaPtr(alpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 6. alpha == 0 quick return BEFORE any x/y/AP pointer check: x/y/AP not
    // referenced, AP unchanged, no launch (alpha == 0 is a no-op that never
    // references x/y/AP).
    if (*alpha == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 7. x/y/AP nullptr checks (only reached with n > 0 and alpha != 0).
    st = ValidateSspr2PtrParams(x, y, AP);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    Sspr2TilingData tiling = CalSspr2TilingData(n, uplo, *alpha, incx, incy);

    // 8. single kernel launch (asynchronous; the caller synchronizes stream).
    sspr2_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float*>(x)),
        reinterpret_cast<GM_ADDR>(const_cast<float*>(y)),
        reinterpret_cast<GM_ADDR>(AP), tiling, tiling.usedCoreNum, stream);

    return ACLBLAS_STATUS_SUCCESS;
}
