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
#include <cstddef>
#include <limits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "chpr2_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

aclblasStatus_t aclblasChpr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x,
    const int incx, const aclblasComplex* y, const int incy, aclblasComplex* ap)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasChpr2", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasChpr2", "n must be nonnegative"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasChpr2", "alpha is nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER, OP_LOGE("aclblasChpr2", "uplo must be UPPER or LOWER");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0 && incx != INT_MIN, OP_LOGE("aclblasChpr2", "incx must be nonzero and not INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0 && incy != INT_MIN, OP_LOGE("aclblasChpr2", "incy must be nonzero and not INT_MIN");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr && y != nullptr && ap != nullptr, OP_LOGE("aclblasChpr2", "x, y and ap must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);

    // The task uses Host pointer mode. Zero alpha is checked after all pointer checks.
    const aclblasComplex alphaValue = *alpha;
    if (alphaValue.real == 0.0f && alphaValue.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Element offsets fit in int64_t for int inputs, but their byte spans may
    // not fit size_t. Reject an impossible allocation before a GM address wraps.
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    const uint64_t absIncx = incx < 0 ? -static_cast<int64_t>(incx) : incx;
    const uint64_t absIncy = incy < 0 ? -static_cast<int64_t>(incy) : incy;
    const uint64_t vectorXLength = static_cast<uint64_t>(n - 1) * absIncx + 1;
    const uint64_t vectorYLength = static_cast<uint64_t>(n - 1) * absIncy + 1;
    const uint64_t packedLength = static_cast<uint64_t>(n) * (static_cast<uint64_t>(n) + 1) / 2;
    CHECK_RET(vectorXLength <= maxElements && vectorYLength <= maxElements && packedLength <= maxElements,
              OP_LOGE("aclblasChpr2", "buffer byte length overflows size_t");
              return ACLBLAS_STATUS_INVALID_VALUE);
    const uint32_t coreCount = GetAivCoreCount();
    CHECK_RET(coreCount != 0, OP_LOGE("aclblasChpr2", "GetAivCoreCount failed");
              return ACLBLAS_STATUS_EXECUTION_FAILED);

    Chpr2TilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaReal = alphaValue.real;
    tiling.alphaImag = alphaValue.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.xStart = incx < 0 ? (1 - static_cast<int64_t>(n)) * tiling.incx : 0;
    tiling.yStart = incy < 0 ? (1 - static_cast<int64_t>(n)) * tiling.incy : 0;
    tiling.packedLength = packedLength;
    const uint64_t tiles = (tiling.packedLength + CHPR2_THREADS - 1) / CHPR2_THREADS;
    const uint32_t numBlocks = static_cast<uint32_t>(std::min<uint64_t>(tiles, coreCount));
    tiling.elementsPerBlock = ((tiles + numBlocks - 1) / numBlocks) * CHPR2_THREADS;
    chpr2_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(ap), tiling, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
