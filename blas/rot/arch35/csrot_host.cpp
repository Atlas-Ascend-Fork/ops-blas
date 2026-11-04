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
 * \file csrot_host.cpp
 * \brief Host-side API for BLAS Level-1 complex Givens plane rotation: aclblasCsrot.
 *        Arch35 (ascend950) implementation.
 *
 *        Applies the Givens rotation to the complex vectors x and y in place:
 *            x[i] = c*x[i] + s*y[i]
 *            y[i] = c*y[i] - s*x[i]        (y uses the ORIGINAL x[i])
 *
 *        c and s are REAL scalars, so the rotation applies to the real and the
 *        imag lane of each complex element independently (no cross terms); the
 *        kernels operate on the float-lane view (2 lanes per complex element).
 *
 *        Two compute paths selected by contiguity of the strides:
 *          - incx==1 && incy==1 : contiguous path, SIMD membase kernel
 *          - otherwise          : strided path, SIMT kernel (grid-stride loop)
 *
 *        netlib csrot.f boundary alignment: n<=0 short-circuits to SUCCESS;
 *        zero / negative strides and the identity rotation (c==1 && s==0) are
 *        intentionally not intercepted, matching the reference behavior.
 */

#include <cstdint>
#include <climits>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "csrot_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

static aclblasStatus_t ValidateCsrotParams(
    const aclblasComplex* x, const aclblasComplex* y, const float* c, const float* s)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCsrot", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr || s == nullptr) {
        OP_LOGE("aclblasCsrot", "c/s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// c/s pointer location query: determine whether each scalar pointer lives on
// the host or on the device (NPU HBM). Uses the shared CheckPtrLocation helper
// (aclrtPointerGetAttributes), same as rotg / scalex. On query failure the call
// is rejected (consistent with scalex).
// Tiling computation for the contiguous path (SIMD membase).
// The contiguous kernel uses a TQue three-stage pipeline with five UB tiles:
//   inQueueX / inQueueY (VECIN, MTE2 input)
//   outNewX / outNewY   (VECOUT, Vector output)
//   workBuf             (VECCALC, pure-Vector scratch for the rounded products)
// Each tile is tileSize complex elements, so UB must hold
// bufferCount * tileSize * sizeof(complex64) bytes.
// ==========================================================================
static void CalCsrotTilingContiguous(uint32_t n, uint32_t numBlocks, CsrotTilingData& tiling)
{
    // Even split (no alignment rounding): perCoreN = n/numBlocks base elements per
    // core, remainder = n%numBlocks front cores each get one extra element. Unlike
    // the old align-down scheme, no core starves when n/numBlocks < ELEMENTS_PER_BLOCK:
    // the load stays balanced down to n < numBlocks. The contiguous kernel copies with
    // pure DataCopyPad, whose GM side requires only 1-byte alignment, so the
    // non-block-aligned per-core start offset and element count are both safe.
    // numBlocks >= 1 is an invariant here: n <= 0 already returned at the API entry
    // (so totalN >= 1) and aivCoreNum == 0 was rejected before the tiling split
    // (so min(totalN, aivCoreNum) >= 1). Neither the division nor the modulo can
    // therefore divide by zero.
    // 防御性兜底：调用方已保证 numBlocks >= 1（totalN>=1 且 aivCoreNum>=1），
    // 此处显式阻断除零，满足静态检查并提升极端健壮性。
    if (numBlocks == 0) {
        numBlocks = 1;
    }
    tiling.perCoreN = n / numBlocks;
    tiling.remainder = n % numBlocks;

    // Five UB tiles: inQueueX + inQueueY + outNewX + outNewY + workBuf.
    // tileSize is the per-copy UB cap (in complex elements); kept block-aligned
    // for UB buffer allocation.
    constexpr uint32_t alignUnit = CSROT_ELEMENTS_PER_BLOCK;
    constexpr uint32_t bufferCount = 5;
    uint32_t maxElements = UB_SIZE / (bufferCount * CSROT_ELEMENT_SIZE);
    tiling.tileSize = (maxElements / alignUnit) * alignUnit;
}

// ==========================================================================
// Tiling computation for the strided path (SIMT).
// The kernel handles negative strides internally (negX/negY flags + (n-1-i)*absInc),
// so it consumes only incx/incy from the tiling. The (n-1)*stride product is still
// computed here in int64_t and range-checked before narrowing: the kernel addresses
// memory in the float-lane view (1 complex = CSROT_FLOATS_PER_COMPLEX = 2 lanes), so
// the maximum lane index is |(n-1)*stride| * 2 and must stay inside the kernel's
// uint32_t index domain (risk R5). |(n-1)*stride| == 2^31 (reachable with a negative
// stride, e.g. lastX == INT32_MIN) would make the lane index 2^32 wrap to 0 and
// silently corrupt element 0, so the accepted bound is tightened to ±INT32_MAX; the
// call is rejected with an explicit error instead of launching a broken kernel.
// The narrowed last offsets are returned via outLastX/outLastY for host-side
// diagnostics (OP_LOGD) only; they are NOT consumed by the kernel.
// nthreads: SIMT threads per block. The grid-stride loop only needs as many
// threads as the per-core element count justifies, so this is sized from
// ceilDiv(n,numBlocks) — rounded UP to SIMT_MIN_THREAD_NUM (the hardware's thread
// scheduling granularity) and capped at SIMT_MAX_THREAD_NUM — instead of always
// launching SIMT_MAX_THREAD_NUM threads. A small n (e.g. n=10, 1 block) thus
// launches 128 threads rather than 2048, eliminating the wasted-thread startup
// cost while large n still saturates the full 2048 per block.
// ==========================================================================
static aclblasStatus_t CalCsrotTilingStrided(
    uint32_t n, int incx, int incy, uint32_t numBlocks, CsrotTilingData& tiling, int32_t* outLastX, int32_t* outLastY)
{
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.nthreads =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    int64_t lastX = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    int64_t lastY = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incy);
    // Lane-view bound: kernel computes uint32_t laneIdx = |lastX| * 2, so |lastX| must
    // not exceed INT32_MAX (|lastX| == 2^31 would wrap laneIdx to 0). The contiguous
    // path needs no such check: n is an int (<= INT32_MAX), so totalN * 2 <= 2^32 - 2.
    if (lastX > INT32_MAX || lastX < -static_cast<int64_t>(INT32_MAX) || lastY > INT32_MAX ||
        lastY < -static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(
            "aclblasCsrot", "strided offset overflow: n=%u incx=%d incy=%d lastX=%lld lastY=%lld",
            static_cast<unsigned>(n), incx, incy, static_cast<long long>(lastX), static_cast<long long>(lastY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *outLastX = static_cast<int32_t>(lastX);
    *outLastY = static_cast<int32_t>(lastY);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Resolve the c/s source: query each pointer's location, fill the corresponding
// TilingData fields, and produce the float* pointers to forward to the kernel.
//   host pointer  -> dereference once, store the scalar, forward nullptr
//   device pointer-> store 0.0f placeholder, forward the pointer verbatim
// The two pointers are independent (c on host + s on device is legal). Device
// pointers are forwarded as-is with no dereference, no D2H copy, no stream sync.
// ==========================================================================
static aclblasStatus_t PrepareCsrotCsSource(
    const float* c, const float* s, CsrotTilingData& tiling, float** cPtr, float** sPtr)
{
    bool cIsDevice = false;
    bool sIsDevice = false;
    aclblasStatus_t cLocSt = CheckPtrLocation(c, &cIsDevice);
    if (cLocSt != ACLBLAS_STATUS_SUCCESS) {
        return cLocSt;
    }
    aclblasStatus_t sLocSt = CheckPtrLocation(s, &sIsDevice);
    if (sLocSt != ACLBLAS_STATUS_SUCCESS) {
        return sLocSt;
    }

    tiling.cIsDevice = cIsDevice ? 1u : 0u;
    tiling.sIsDevice = sIsDevice ? 1u : 0u;
    tiling.cosValue = cIsDevice ? 0.0f : (*c);
    tiling.sinValue = sIsDevice ? 0.0f : (*s);
    // Device path forwards the pointer verbatim; host path forwards nullptr (kernel
    // reads the tiling scalar instead).
    *cPtr = cIsDevice ? const_cast<float*>(c) : nullptr;
    *sPtr = sIsDevice ? const_cast<float*>(s) : nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Launch — tiling computation + asynchronous kernel launch
// ==========================================================================
static aclblasStatus_t LaunchCsrotKernel(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const float* s)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsrot", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    bool contiguous = (incx == 1 && incy == 1);

    CsrotTilingData tiling{};
    tiling.totalN = totalN;

    // Resolve c/s source (host scalar vs device GM pointer); fills tiling.cosValue/
    // sinValue/cIsDevice/sIsDevice and yields the pointers to forward.
    float* cPtr = nullptr;
    float* sPtr = nullptr;
    aclblasStatus_t csSt = PrepareCsrotCsSource(c, s, tiling, &cPtr, &sPtr);
    if (csSt != ACLBLAS_STATUS_SUCCESS) {
        return csSt;
    }

    // lastX/lastY: strided-path tail offset ((n-1)*stride), host-side diagnostic only.
    // Contiguous path leaves them at 0 (no stride offset); the kernel never reads them.
    int32_t lastX = 0;
    int32_t lastY = 0;
    uint32_t numBlocks;
    if (contiguous) {
        tiling.tilingKey = 0;
        numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;
        CalCsrotTilingContiguous(totalN, numBlocks, tiling);
    } else {
        tiling.tilingKey = 1;
        uint32_t blocksByThreads = CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM);
        numBlocks = (blocksByThreads < aivCoreNum) ? blocksByThreads : aivCoreNum;
        if (numBlocks == 0) {
            numBlocks = 1;
        }
        aclblasStatus_t stridedSt = CalCsrotTilingStrided(totalN, incx, incy, numBlocks, tiling, &lastX, &lastY);
        if (stridedSt != ACLBLAS_STATUS_SUCCESS) {
            return stridedSt;
        }
    }

    OP_LOGD(
        "aclblasCsrot",
        "tiling: key=%u totalN=%u incx=%d incy=%d lastX=%d lastY=%d c=%f s=%f cIsDevice=%u sIsDevice=%u "
        "perCoreN=%u remainder=%u tileSize=%u numBlocks=%u nthreads=%u",
        tiling.tilingKey, tiling.totalN, incx, incy, lastX, lastY, tiling.cosValue, tiling.sinValue, tiling.cIsDevice,
        tiling.sIsDevice, tiling.perCoreN, tiling.remainder, tiling.tileSize, numBlocks, tiling.nthreads);
    OP_LOGI(
        "aclblasCsrot", "launching kernel: key=%u blocks=%u cIsDevice=%u sIsDevice=%u", tiling.tilingKey, numBlocks,
        tiling.cIsDevice, tiling.sIsDevice);

    // In-place operation: no workspace needed. Pass nullptr; no aclrtMalloc here.
    // x/y are complex64 buffers; the kernels consume the float-lane view (2 lanes
    // per complex element).
    csrot_kernel_do(reinterpret_cast<float*>(x), reinterpret_cast<float*>(y), cPtr, sPtr, numBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API entry — dispatch only (validate then launch)
// ==========================================================================
extern "C" aclblasStatus_t aclblasCsrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const float* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCsrot", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateCsrotParams(x, y, c, s);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return LaunchCsrotKernel(handle, n, x, incx, y, incy, c, s);
}
