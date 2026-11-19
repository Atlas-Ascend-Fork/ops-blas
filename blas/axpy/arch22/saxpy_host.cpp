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
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "saxpy_kernel.h"
#include "saxpy_tiling_data.h"

namespace {

int64_t SaxpyAbsIncrement(int64_t increment) { return increment > 0 ? increment : -increment; }

// Span overflow guard, computed in int64.
bool SaxpySpanOverflow(int n, int increment)
{
    const int64_t absIncrement = SaxpyAbsIncrement(static_cast<int64_t>(increment));
    return static_cast<int64_t>(n - 1) * absIncrement > static_cast<int64_t>(INT32_MAX);
}

aclblasStatus_t ValidateSaxpyParams(int n, const float* alpha, const float* x, int incx, const float* y, int incy)
{
    if (alpha == nullptr) {
        OP_LOGE("aclblasSaxpy", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasSaxpy", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasSaxpy", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incx == INT_MIN) {
        OP_LOGE("aclblasSaxpy", "incx must be non-zero and not INT32_MIN, got %d", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0 || incy == INT_MIN) {
        OP_LOGE("aclblasSaxpy", "incy must be non-zero and not INT32_MIN, got %d", incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (SaxpySpanOverflow(n, incx) || SaxpySpanOverflow(n, incy)) {
        OP_LOGE(
            "aclblasSaxpy", "vector span (n-1)*|inc| exceeds int32 range for n=%d, incx=%d, incy=%d", n, incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

SaxpyTilingData CalSaxpyTilingData(int n, int incx, int incy, float alpha, uint32_t coreNum)
{
    SaxpyTilingData tiling = {};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.tileSize = SAXPY_TILE_ELEMS;
    tiling.alpha = alpha;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);

    const uint32_t totalN = static_cast<uint32_t>(n);
    const bool dense = (incx == 1 && incy == 1);
    tiling.mode = dense ? 0U : 1U;
    uint32_t blocks = 0U;
    if (dense) {
        uint32_t maxBlocks = totalN / SAXPY_ELEMS_PER_BLOCK;
        if (maxBlocks == 0U) {
            maxBlocks = 1U;
        }
        blocks = std::min(coreNum, maxBlocks);
        if (blocks == 0U) {
            blocks = 1U;
        }
        tiling.perBlockN = (totalN / blocks / SAXPY_ELEMS_PER_BLOCK) * SAXPY_ELEMS_PER_BLOCK;
        tiling.remainder = totalN - tiling.perBlockN * blocks;
    } else {
        blocks = std::min(coreNum, totalN);
        if (blocks == 0U) {
            blocks = 1U;
        }
        tiling.perBlockN = totalN / blocks;
        tiling.remainder = totalN - tiling.perBlockN * blocks;
    }
    tiling.activeBlocks = blocks;
    return tiling;
}
} // namespace

aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSaxpy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // n <= 0: valid no-op, no memory access.
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t validRet = ValidateSaxpyParams(n, alpha, x, incx, y, incy);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    auto* h = handle;
    if (h->stream == nullptr) {
        OP_LOGE("aclblasSaxpy", "stream is not initialized");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    const float alphaHost = *alpha;
    // alpha == 0: no read of x, no write of y.
    if (alphaHost == 0.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0U) {
        OP_LOGE("aclblasSaxpy", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const SaxpyTilingData tiling = CalSaxpyTilingData(n, incx, incy, alphaHost, coreNum);
    OP_LOGD(
        "aclblasSaxpy", "tiling: n=%d incx=%d incy=%d blocks=%u perBlock=%u remainder=%u tile=%u", n, incx, incy,
        tiling.activeBlocks, tiling.perBlockN, tiling.remainder, tiling.tileSize);
    OP_LOGI(
        "aclblasSaxpy", "launching kernel: blocks=%u, cores=%u, alpha=%.6f", tiling.activeBlocks, coreNum,
        static_cast<double>(alphaHost));

    saxpy_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), tiling, tiling.activeBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
