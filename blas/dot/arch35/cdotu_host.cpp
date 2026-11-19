/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cdotu_host.cpp
 * \brief cdotu Host-side dispatch for ascend950 (complex64 unconjugated dot product)
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "cdotu_tiling_data.h"

void cdotu_kernel_do(
    uint8_t* inX, uint8_t* inY, uint8_t* result, uint8_t* workSpace,
    const CdotuTilingData& tiling, uint32_t numBlocks, void* stream);

namespace {

constexpr uint32_t CDOTU_SIMT_MIN_THREAD_NUM = 128;
// Dual-accumulator threshold: for n >= 3M the two halves of each float4 feed
// independent FMA chains (halving the serial dependency). Measured to keep float32
// accumulation error within 1e-2 and run faster than a single accumulator; the
// single accumulator is optimal for n < 3M.
constexpr int32_t CDOTU_DUAL_ACC_THRESHOLD = 3000000;
// Per-core thread count for n >= 3M (4M dual-accumulator measured optimal);
// above 1024 slows down due to register pressure.
constexpr uint32_t CDOTU_LARGE_N_THREADS = 1024;
// Per-core thread count for n < 3M (1M/2M measured optimal at 512).
constexpr uint32_t CDOTU_SMALL_N_THREADS = 512;

// Tiling data computation. Only scalar fields are set: the per-core [start, count)
// partition is derived inside the kernel (see cdotu_kernel.cpp) from n / useCoreNum /
// blockIdx, matching sdot/dotex which pass useCoreNum plus strides and let each core
// compute its own slice. This keeps the tiling payload a fixed handful of scalars
// instead of a compile-time-sized per-core array.
//
// useCoreNum is identical to the launched numBlocks and is capped at n/2: this guarantees
// at least 2 elements per core (one float4 pair), eliminating calNum==0 cores at the root
// (their zero-fill in the __global__ context is unreliable, and combine would read stale
// workspace residue). Combine reads exactly useCoreNum slots, all of which are valid.
static CdotuTilingData CalCdotuTilingData(int n, uint32_t vecCoreNum, int incx, int incy, uint32_t nthreads)
{
    CdotuTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.nthreads = nthreads;
    tiling.useDualAcc = (n >= CDOTU_DUAL_ACC_THRESHOLD) ? 1u : 0u;

    uint32_t useCoreNum = std::min(vecCoreNum, static_cast<uint32_t>(n) / 2u);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    tiling.useCoreNum = useCoreNum;

    return tiling;
}

static aclblasStatus_t ValidateCdotuParams(
    aclblasHandle_t handle, int n, int incx, int incy, const aclblasComplex* x, const aclblasComplex* y,
    const aclblasComplex* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCdotu", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCdotu", "n must be >= 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Stride validation: incx/incy == 0 is illegal. cdotu aligns with Netlib semantics and
    // supports negative strides (a negative stride is a valid input, not a no-op), so the
    // generic ops-blas "incx<=0 is a no-op" convention (scalex/asum) does NOT apply here.
    if (incx == 0) {
        OP_LOGE("aclblasCdotu", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCdotu", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Align with the existing arch22 cdot: result must never be nullptr, even when n == 0
    // (the no-op path still writes (0,0) through result). Checked unconditionally, before the
    // n > 0 block that guards x/y.
    if (result == nullptr) {
        OP_LOGE("aclblasCdotu", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0) {
        if (x == nullptr) {
            OP_LOGE("aclblasCdotu", "x must not be nullptr when n > 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (y == nullptr) {
            OP_LOGE("aclblasCdotu", "y must not be nullptr when n > 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        // The CONTIGUOUS path (incx == incy == 1) reinterprets the complex stream as
        // float4 and issues 128-bit loads, which require a 16B-aligned base. An 8B-aligned
        // interior pointer (e.g. base + 1 complex) would make those loads misaligned, whose
        // behavior on dav SIMT is not validated, so reject it up front. The STRIDED path
        // only loads float2 (8B), which any aclblasComplex* satisfies.
        if (incx == 1 && incy == 1) {
            if ((reinterpret_cast<uintptr_t>(x) & 0xFu) != 0u) {
                OP_LOGE("aclblasCdotu", "x must be 16B-aligned when incx == incy == 1");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            if ((reinterpret_cast<uintptr_t>(y) & 0xFu) != 0u) {
                OP_LOGE("aclblasCdotu", "y must be 16B-aligned when incx == incy == 1");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t CdotuExecuteKernel(
    _aclblas_handle* h, const aclblasComplex* x, const aclblasComplex* y, aclblasComplex* result,
    uint32_t numBlocks, const CdotuTilingData& tiling)
{
    size_t workspaceBytes = static_cast<size_t>(tiling.useCoreNum) * 2 * sizeof(float);
    CHECK_RET(workspaceBytes <= GetEffectiveWorkspaceSize(h),
              OP_LOGE("aclblasCdotu", "workspace %zu > handle %zu", workspaceBytes, GetEffectiveWorkspaceSize(h));
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    OP_LOGI("aclblasCdotu", "launching kernel: blocks=%u useCoreNum=%u nthreads=%u", numBlocks, tiling.useCoreNum,
            tiling.nthreads);
    cdotu_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)),
        reinterpret_cast<uint8_t*>(result), workspaceDevice, tiling, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCdotu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result)
{
    aclblasStatus_t status = ValidateCdotuParams(handle, n, incx, incy, x, y, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // cuBLAS/Netlib semantics: n == 0 is a legal no-op, result = (0, 0).
    if (n == 0) {
        aclblasComplex zero = {0.0f, 0.0f};
        aclError memRet = aclrtMemcpy(
            result, sizeof(aclblasComplex), &zero, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            OP_LOGE("aclblasCdotu", "aclrtMemcpy for n==0 zero failed: %d", memRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // numBlocks (launch grid) = number of cores participating in the reduction = min(n, AIV core count).
    // The even-aligned partition requires at least one float4 pair (2 elements) per core:
    // cap at n/2 to avoid calNum==0 cores at the root (their __global__-context zero-fill
    // GM write is unreliable, and Combine would read stale workspace residue).
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCdotu", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);
    uint32_t maxUsableBlocks = static_cast<uint32_t>(n) / 2u;
    if (numBlocks > maxUsableBlocks) {
        numBlocks = maxUsableBlocks;
    }
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    // Thread count: 512 for n < 3M (1M/2M measured best at 512), 1024 for
    // n >= 3M (4M dual-acc best at 1024; 2048 slower due to register pressure
    // from the 4-FMA complex multiply).
    uint32_t perCore = CeilDiv<uint32_t>(static_cast<uint32_t>(n), numBlocks);
    uint32_t maxThreads = (static_cast<uint32_t>(n) >= static_cast<uint32_t>(CDOTU_DUAL_ACC_THRESHOLD))
                              ? CDOTU_LARGE_N_THREADS
                              : CDOTU_SMALL_N_THREADS;
    uint32_t nthreads = std::min(
        std::max(CDOTU_SIMT_MIN_THREAD_NUM, CeilAlign<uint32_t>(perCore, 32)), maxThreads);

    CdotuTilingData tiling = CalCdotuTilingData(n, numBlocks, incx, incy, nthreads);

    OP_LOGD(
        "aclblasCdotu", "tiling: n=%d incx=%d incy=%d useCoreNum=%u numBlocks=%u nthreads=%u", tiling.n, tiling.incx,
        tiling.incy, tiling.useCoreNum, numBlocks, tiling.nthreads);

    return CdotuExecuteKernel(handle, x, y, result, numBlocks, tiling);
}