/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Validation/Netlib ordering reference: ops-blas PR !800,
// commit 4265ffb4c099de2e74cab9464eb25d418abeeda2 (same license).
#include <algorithm>
#include <limits>
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "log/log.h"
#include "chbmv_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
bool InvalidArguments(
    aclblasFillMode_t uplo, int k, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* a,
    const aclblasComplex* x, const aclblasComplex* beta, const aclblasComplex* y)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasChbmv", "invalid uplo=%d", static_cast<int>(uplo));
              return true);
    CHECK_RET(k >= 0, OP_LOGE("aclblasChbmv", "invalid k=%d", k); return true);
    CHECK_RET(static_cast<int64_t>(lda) >= static_cast<int64_t>(k) + 1,
              OP_LOGE("aclblasChbmv", "invalid lda=%d for k=%d", lda, k);
              return true);
    CHECK_RET(incx != 0 && incx != INT32_MIN, OP_LOGE("aclblasChbmv", "unsupported incx=%d", incx); return true);
    CHECK_RET(incy != 0 && incy != INT32_MIN, OP_LOGE("aclblasChbmv", "unsupported incy=%d", incy); return true);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasChbmv", "alpha is nullptr"); return true);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasChbmv", "beta is nullptr"); return true);
    CHECK_RET(a != nullptr, OP_LOGE("aclblasChbmv", "A is nullptr"); return true);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasChbmv", "x is nullptr"); return true);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasChbmv", "y is nullptr"); return true);
    return false;
}

bool StorageFits(int n, int lda, int64_t sx, int64_t sy)
{
    const uint64_t maxElements = std::numeric_limits<int64_t>::max() / sizeof(aclblasComplex);
    const uint64_t nx = 1 + static_cast<uint64_t>(n - 1) * (sx < 0 ? -sx : sx);
    const uint64_t ny = 1 + static_cast<uint64_t>(n - 1) * (sy < 0 ? -sy : sy);
    CHECK_RET(static_cast<uint64_t>(lda) * n <= maxElements,
              OP_LOGE("aclblasChbmv", "matrix storage exceeds address range: n=%d, lda=%d", n, lda);
              return false);
    CHECK_RET(nx - 1 <= INT32_MAX,
              OP_LOGE("aclblasChbmv", "x span exceeds index range: n=%d, incx=%lld", n, static_cast<long long>(sx));
              return false);
    CHECK_RET(ny - 1 <= INT32_MAX,
              OP_LOGE("aclblasChbmv", "y span exceeds index range: n=%d, incy=%lld", n, static_cast<long long>(sy));
              return false);
    return true;
}

struct Capabilities {
    int32_t device = -1;
    int64_t cores = 0;
    uint64_t ub = 0;
};

const Capabilities* GetCapabilities()
{
    int32_t device = 0;
    const aclError deviceStatus = aclrtGetDevice(&device);
    CHECK_RET(deviceStatus == ACL_SUCCESS, OP_LOGE("aclblasChbmv", "aclrtGetDevice failed: %d", deviceStatus);
              return nullptr);
    // These hardware capacities are immutable for a visible device. Refresh on
    // device switches; never cache context-owned stream or kernel handles.
    static thread_local Capabilities cached;
    if (cached.device != device) {
        int64_t cores = 0;
        const aclError coreStatus = aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores);
        CHECK_RET(coreStatus == ACL_SUCCESS,
                  OP_LOGE("aclblasChbmv", "aclrtGetDeviceInfo failed: device=%d, error=%d", device, coreStatus);
                  return nullptr);
        CHECK_RET(cores > 0, OP_LOGE("aclblasChbmv", "invalid vector core count=%lld", static_cast<long long>(cores));
                  return nullptr);
        auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
        CHECK_RET(platform != nullptr, OP_LOGE("aclblasChbmv", "PlatformAscendC instance is nullptr"); return nullptr);
        uint64_t ub = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        CHECK_RET(ub >= CHBMV_OUTPUT_BYTES,
                  OP_LOGE("aclblasChbmv", "insufficient UB capacity=%llu", static_cast<unsigned long long>(ub));
                  return nullptr);
        cached = {device, cores, ub};
    }
    return &cached;
}

void ConfigureVectorBlocks(ChbmvTilingData& t, int64_t cores)
{
    // Use the available cores for medium sizes as well. Each core owns
    // a 32-byte aligned span and iterates internally in chunks of <=64.
    const uint32_t chunks = (t.n + cores * CHBMV_VECTOR_ROWS - 1) / (cores * CHBMV_VECTOR_ROWS);
    const uint32_t tileRows = ((t.n + cores * chunks - 1) / (cores * chunks) + 7) / 8 * 8;
    if (t.vectorMode == 2) {
        t.blockRows = tileRows * chunks;
        t.vectorRows = tileRows;
    } else {
        const uint32_t groups = (t.n + CHBMV_VECTOR_ROWS - 1) / CHBMV_VECTOR_ROWS;
        t.numBlocks = std::min<uint64_t>(groups, cores);
        t.blockRows = ((groups + t.numBlocks - 1) / t.numBlocks) * CHBMV_VECTOR_ROWS;
        t.vectorRows = CHBMV_VECTOR_ROWS;
    }
    t.numBlocks = (t.n + t.blockRows - 1) / t.blockRows;
}

void ConfigureTiling(ChbmvTilingData& t, int64_t cores, uint64_t ub)
{
    const uint64_t window = CHBMV_VECTOR_ROWS + 2ULL * t.band;
    const uint64_t aCache = t.band <= 32 ? ((window * t.lda * 8 + 31) / 32) * 32 + 32 : 0;
    const uint64_t xCache = ((window * 8 + 31) / 32) * 32 + 32;
    const bool alphaZero = t.alphaReal == 0 && t.alphaImag == 0;
    const uint64_t wideCache = 2ULL * CHBMV_VECTOR_ROWS * CHBMV_MIRROR_TILE * 8;
    const bool narrow = t.band <= 32 && aCache + xCache + CHBMV_VECTOR_SCRATCH_BYTES <= ub;
    const bool wide =
        static_cast<uint64_t>(t.lda) * 8 <= UINT32_MAX && wideCache + xCache + CHBMV_VECTOR_SCRATCH_BYTES <= ub;
    if ((t.n >= 512 && (alphaZero || narrow)) || (t.n >= 128 && !narrow && wide)) {
        t.vectorMode = alphaZero || narrow ? 1 : 2;
        t.aCacheBytes = alphaZero ? 32 : (narrow ? aCache : wideCache);
        t.xCacheBytes = alphaZero ? 32 : xCache;
        ConfigureVectorBlocks(t, cores);
    } else {
        // Own blocks of >=4 complex values, aligned to 32 bytes for unit strides.
        const uint32_t groups = (t.n + 3) / 4;
        t.numBlocks = std::min<uint64_t>(groups, cores);
        t.blockRows = ((groups + t.numBlocks - 1) / t.numBlocks) * 4;
        t.numBlocks = (t.n + t.blockRows - 1) / t.blockRows;
    }
}
} // namespace

aclblasStatus_t aclblasChbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha, const aclblasComplex* a,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    // A2/A3 contract: handle validation precedes n and the empty quick return.
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasChbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasChbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (InvalidArguments(uplo, k, lda, incx, incy, alpha, a, x, beta, y))
        return ACLBLAS_STATUS_INVALID_VALUE;

    const int64_t sx = incx;
    const int64_t sy = incy;
    if (!StorageFits(n, lda, sx, sy))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha->real == 0 && alpha->imag == 0 && beta->real == 1 && beta->imag == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const auto* capabilities = GetCapabilities();
    if (!capabilities)
        return ACLBLAS_STATUS_EXECUTION_FAILED;

    ChbmvTilingData t{};
    t.n = n;
    t.k = k;
    t.band = std::min(k, n - 1);
    t.lda = lda;
    t.upper = uplo == ACLBLAS_UPPER;
    t.incx = sx;
    t.incy = sy;
    t.xStart = sx < 0 ? -static_cast<int64_t>(n - 1) * sx : 0;
    t.yStart = sy < 0 ? -static_cast<int64_t>(n - 1) * sy : 0;
    t.alphaReal = alpha->real;
    t.alphaImag = alpha->imag;
    t.betaReal = beta->real;
    t.betaImag = beta->imag;
    ConfigureTiling(t, capabilities->cores, capabilities->ub);
    // Partial stores from different cores must not share a 32-byte GM transaction.
    // Row-count alignment alone does not ensure this for strides or offset views.
    if (sy != 1 || reinterpret_cast<uintptr_t>(y) % CHBMV_OUTPUT_BYTES != 0) {
        t.numBlocks = 1;
        t.blockRows = t.n;
    }
    const aclError launch = chbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), t, handle->stream);
    CHECK_RET(launch == ACL_SUCCESS, OP_LOGE("aclblasChbmv", "chbmv_kernel_do failed: %d", launch);
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    return ACLBLAS_STATUS_SUCCESS;
}
