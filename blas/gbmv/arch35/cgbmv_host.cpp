/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgbmv_host.cpp
 * \brief single-precision complex gbmv host-side implementation
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cgbmv_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

namespace {

// Warp-uniform shuffle loops in the kernel need whole warps per block.
static_assert(SIMT_MIN_THREAD_NUM % CGBMV_WARP_SIZE == 0, "block size must be a multiple of the warp size");
static_assert(SIMT_MAX_THREAD_NUM % CGBMV_WARP_SIZE == 0, "block size must be a multiple of the warp size");
static_assert(CGBMV_N_MAX_THREADS <= SIMT_MAX_THREAD_NUM, "trans-N block exceeds the SIMT block size limit");
static_assert(CGBMV_N_MAX_THREADS % CGBMV_WARP_SIZE == 0, "trans-N block must hold whole warps");

bool IsComplexZero(const aclblasComplex& c)
{
    return c.real == 0.0f && c.imag == 0.0f;
}

bool IsComplexOne(const aclblasComplex& c)
{
    return c.real == 1.0f && c.imag == 0.0f;
}

// Where a scalar argument lives. Host scalars are read once here; device scalars are forwarded to the kernel
// untouched (no dereference, no D2H copy, no stream sync), which keeps the call asynchronous.
struct CgbmvScalar {
    aclblasComplex value{0.0f, 0.0f};
    bool isDevice = false;
    const aclblasComplex* devPtr = nullptr;
};

aclblasStatus_t ResolveCgbmvScalar(const aclblasComplex* ptr, CgbmvScalar& out)
{
    bool isDevice = false;
    aclblasStatus_t st = CheckPtrLocation(ptr, &isDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgbmv", "failed to query the memory location of alpha/beta");
        return st;
    }
    out.isDevice = isDevice;
    out.devPtr = isDevice ? ptr : nullptr;
    if (!isDevice) {
        out.value = *ptr;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t ToTransCode(aclblasOperation_t trans)
{
    if (trans == ACLBLAS_OP_N) {
        return CGBMV_TRANS_N;
    }
    return (trans == ACLBLAS_OP_T) ? CGBMV_TRANS_T : CGBMV_TRANS_C;
}

// Checks that depend on scalars only; they are enforced even for the m == 0 / n == 0 no-op.
aclblasStatus_t ValidateCgbmvScalars(
    aclblasOperation_t trans, int m, int n, int kl, int ku, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta)
{
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE("aclblasCgbmv", "invalid trans=%d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0 && n >= 0, OP_LOGE("aclblasCgbmv", "invalid m=%d or n=%d", m, n);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(kl >= 0 && ku >= 0, OP_LOGE("aclblasCgbmv", "invalid kl=%d or ku=%d", kl, ku);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(static_cast<int64_t>(lda) >= static_cast<int64_t>(kl) + ku + 1,
              OP_LOGE("aclblasCgbmv", "invalid lda=%d, kl=%d, ku=%d", lda, kl, ku);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0 && incy != 0, OP_LOGE("aclblasCgbmv", "incx=%d / incy=%d must not be zero", incx, incy);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr && beta != nullptr, OP_LOGE("aclblasCgbmv", "alpha/beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCgbmvParams(
    aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, const aclblasComplex* y, int incy)
{
    aclblasStatus_t st = ValidateCgbmvScalars(trans, m, n, kl, ku, lda, incx, incy, alpha, beta);
    if (st != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return st;
    }
    CHECK_RET(kl <= m - 1 && ku <= n - 1, OP_LOGE("aclblasCgbmv", "kl=%d must be <= m-1, ku=%d must be <= n-1", kl, ku);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(A != nullptr && x != nullptr && y != nullptr, OP_LOGE("aclblasCgbmv", "A/x/y must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t CalcGroupSize(uint32_t bandRows)
{
    uint32_t group = 1;
    while (group < bandRows && group < CGBMV_WARP_SIZE) {
        group <<= 1U;
    }
    return group;
}

CgbmvTilingData CalCgbmvTilingData(
    aclblasOperation_t trans, int m, int n, int kl, int ku, const CgbmvScalar& alpha, int lda, int incx,
    const CgbmvScalar& beta, int incy)
{
    CgbmvTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.kl = static_cast<uint32_t>(kl);
    tiling.ku = static_cast<uint32_t>(ku);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = ToTransCode(trans);
    // With a device alpha the launch is shaped for the general path; the kernel switches to SCALE if alpha == 0.
    tiling.mode = (!alpha.isDevice && IsComplexZero(alpha.value)) ? CGBMV_MODE_SCALE : CGBMV_MODE_GBMV;
    if (beta.isDevice) {
        tiling.betaMode = CGBMV_BETA_GENERAL; // re-derived by the kernel from the device value
    } else if (IsComplexZero(beta.value)) {
        tiling.betaMode = CGBMV_BETA_ZERO;
    } else {
        tiling.betaMode = IsComplexOne(beta.value) ? CGBMV_BETA_ONE : CGBMV_BETA_GENERAL;
    }
    bool lanesPerOutput = (tiling.mode == CGBMV_MODE_GBMV && tiling.trans != CGBMV_TRANS_N);
    tiling.groupSize = lanesPerOutput ? CalcGroupSize(tiling.kl + tiling.ku + 1U) : 1U;
    tiling.alphaIsDevice = alpha.isDevice ? 1U : 0U;
    tiling.betaIsDevice = beta.isDevice ? 1U : 0U;
    tiling.alphaR = alpha.value.real;
    tiling.alphaI = alpha.value.imag;
    tiling.betaR = beta.value.real;
    tiling.betaI = beta.value.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

// trans N: rows at or beyond n + kl hold no band element; the kernel only applies y = beta * y to them.
uint32_t CalcNBandRows(const CgbmvTilingData& tiling)
{
    uint64_t bandEnd = static_cast<uint64_t>(tiling.n) + tiling.kl;
    return static_cast<uint32_t>(std::min<uint64_t>(tiling.m, bandEnd));
}

// trans N: the column window of a 32-row tile (at most kl + ku + 32 columns, clipped to n) is split over `slices`
// warps: SIMT_MIN_THREAD_NUM / 32 at least, doubled (up to 32) while each keeps CGBMV_MIN_COLS_PER_SLICE columns.
uint32_t CalcNSlices(uint64_t window)
{
    uint32_t slices = SIMT_MIN_THREAD_NUM / CGBMV_WARP_SIZE;
    uint32_t maxSlices = CGBMV_N_MAX_THREADS / CGBMV_WARP_SIZE;
    while (slices < maxSlices && static_cast<uint64_t>(slices << 1U) * CGBMV_MIN_COLS_PER_SLICE <= window) {
        slices <<= 1U;
    }
    return slices;
}

// trans N: blocks cover the band tiles; extra blocks (up to the core count) help with the rows below the band.
uint32_t CalcNLaunchShape(CgbmvTilingData& tiling, uint32_t aivCoreNum)
{
    uint32_t bandRows = CalcNBandRows(tiling);
    uint64_t window = std::min<uint64_t>(static_cast<uint64_t>(tiling.kl) + tiling.ku + CGBMV_WARP_SIZE, tiling.n);
    tiling.slices = CalcNSlices(window);
    tiling.numThreads = tiling.slices * CGBMV_WARP_SIZE;
    // Long slices amortise the stepped loop's set-up; short ones are cheaper with per-column addressing.
    tiling.nStepped = (CeilDiv<uint64_t>(window, tiling.slices) >= CGBMV_N_STEPPED_MIN_COLS) ? 1U : 0U;
    uint64_t tiles = CeilDiv<uint64_t>(bandRows, CGBMV_WARP_SIZE);
    uint64_t restRows = tiling.m - bandRows; // rows from n + kl on: y = beta * y only
    uint64_t scaleBlocks = (tiling.betaMode == CGBMV_BETA_ONE) ? 0U : CeilDiv<uint64_t>(restRows, tiling.numThreads);
    uint64_t blocks = std::min<uint64_t>(std::max(tiles, scaleBlocks), aivCoreNum);
    return static_cast<uint32_t>(std::max<uint64_t>(blocks, 1U));
}

// Returns the block count and fills tiling.numThreads / tiling.slices / tiling.nStepped.
uint32_t CalcLaunchShape(CgbmvTilingData& tiling, uint32_t aivCoreNum)
{
    if (tiling.mode == CGBMV_MODE_GBMV && tiling.trans == CGBMV_TRANS_N) {
        return CalcNLaunchShape(tiling, aivCoreNum);
    }
    tiling.slices = 1U;
    uint32_t outDim = (tiling.trans == CGBMV_TRANS_N) ? tiling.m : tiling.n;
    uint64_t totalThreads = static_cast<uint64_t>(outDim) * tiling.groupSize;
    uint32_t numBlocks =
        static_cast<uint32_t>(std::min<uint64_t>(CeilDiv<uint64_t>(totalThreads, SIMT_MIN_THREAD_NUM), aivCoreNum));
    numBlocks = std::max(numBlocks, 1U);
    tiling.numThreads = static_cast<uint32_t>(std::min<uint64_t>(
        CeilAlign<uint64_t>(CeilDiv<uint64_t>(totalThreads, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM));
    return numBlocks;
}

// Resolves where alpha/beta live. quickReturn is set when both are host scalars with alpha == 0 and beta == 1;
// with a device scalar the same rule is applied inside the kernel.
aclblasStatus_t PrepareCgbmvScalars(
    const aclblasComplex* alphaPtr, const aclblasComplex* betaPtr, CgbmvScalar& alpha, CgbmvScalar& beta,
    bool& quickReturn)
{
    aclblasStatus_t st = ResolveCgbmvScalar(alphaPtr, alpha);
    if (st == ACLBLAS_STATUS_SUCCESS) {
        st = ResolveCgbmvScalar(betaPtr, beta);
    }
    quickReturn = st == ACLBLAS_STATUS_SUCCESS && !alpha.isDevice && !beta.isDevice && IsComplexZero(alpha.value) &&
                  IsComplexOne(beta.value);
    return st;
}

aclblasStatus_t LaunchCgbmvKernel(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alphaPtr,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* betaPtr,
    aclblasComplex* y, int incy)
{
    CgbmvScalar alpha;
    CgbmvScalar beta;
    bool quickReturn = false;
    aclblasStatus_t st = PrepareCgbmvScalars(alphaPtr, betaPtr, alpha, beta, quickReturn);
    if (st != ACLBLAS_STATUS_SUCCESS || quickReturn) {
        return st;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum != 0, OP_LOGE("aclblasCgbmv", "GetAivCoreCount failed");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    CgbmvTilingData tiling = CalCgbmvTilingData(trans, m, n, kl, ku, alpha, lda, incx, beta, incy);
    uint32_t numBlocks = CalcLaunchShape(tiling, aivCoreNum);

    OP_LOGD(
        "aclblasCgbmv",
        "tiling: m=%u n=%u kl=%u ku=%u lda=%u trans=%u mode=%u betaMode=%u group=%u slices=%u stepped=%u threads=%u "
        "alphaIsDevice=%u betaIsDevice=%u incx=%ld incy=%ld",
        tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, tiling.trans, tiling.mode, tiling.betaMode,
        tiling.groupSize, tiling.slices, tiling.nStepped, tiling.numThreads, tiling.alphaIsDevice, tiling.betaIsDevice,
        static_cast<long>(tiling.incx), static_cast<long>(tiling.incy));
    OP_LOGI("aclblasCgbmv", "launching kernel: numBlocks=%u aivCoreNum=%u", numBlocks, aivCoreNum);

    cgbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(alpha.devPtr)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(beta.devPtr)), numBlocks, tiling, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCgbmv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasCgbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    aclblasStatus_t st = ValidateCgbmvParams(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return LaunchCgbmvKernel(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
}
