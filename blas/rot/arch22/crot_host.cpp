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
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "crot_tiling_data.h"

aclError crot_kernel_do(
    uint8_t* x, uint8_t* y, const uint8_t* c, const uint8_t* s, CrotTilingData tiling, uint32_t blocks, void* stream);

static bool CrotValidStride(int n, int inc, const char* name)
{
    if (inc == 0) {
        OP_LOGE("aclblasCrot", "%s=%d must be non-zero when n > 0", name, inc);
        return false;
    }
    if (inc == INT32_MIN) {
        OP_LOGE("aclblasCrot", "%s=%d must not be INT32_MIN when n > 0", name, inc);
        return false;
    }
    const int64_t magnitude = inc < 0 ? -static_cast<int64_t>(inc) : inc;
    const int64_t span = static_cast<int64_t>(n - 1) * magnitude;
    if (span > INT32_MAX) {
        OP_LOGE(
            "aclblasCrot", "n=%d, %s=%d: (n-1)*abs(%s)=%lld exceeds INT32_MAX=%d", n, name, inc, name,
            static_cast<long long>(span), INT32_MAX);
        return false;
    }
    return true;
}

static bool CrotValidPointer(const void* pointer, const char* name)
{
    if (pointer == nullptr) {
        OP_LOGE("aclblasCrot", "%s is nullptr; must be non-null when n > 0", name);
        return false;
    }
    return true;
}

static CrotTilingData CrotMakeTiling(
    int n, int incx, int incy, const float* c, const aclblasComplex* s, bool cDevice, bool sDevice)
{
    CrotTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.c = cDevice ? 0.0f : *c;
    tiling.sr = sDevice ? 0.0f : s->real;
    tiling.si = sDevice ? 0.0f : s->imag;
    return tiling;
}

static aclblasStatus_t CrotGetBlockCount(
    const aclblasComplex* x, const aclblasComplex* y, const CrotTilingData& tiling, uint32_t& blocks)
{
    blocks = 1;
    // Whole tiles make all inter-core boundaries 32B aligned. Unaligned views
    // and strided writes use one core to avoid overlapping DMA write units.
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(y)) & 31U) == 0;
    if (tiling.incx == 1 && tiling.incy == 1 && aligned) {
        const uint32_t cores = GetAivCoreCount();
        if (cores == 0) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        blocks = std::min(cores, (tiling.n + CROT_TILE_COMPLEX - 1) / CROT_TILE_COMPLEX);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasCrot(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c,
    const aclblasComplex* s)
{
    // arch22 task contract: handle -> negative n -> quick return -> remaining arguments.
    if (handle == nullptr) {
        OP_LOGE("aclblasCrot", "handle is nullptr; a valid handle is required");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCrot", "n=%d must be non-negative", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!CrotValidStride(n, incx, "incx") || !CrotValidStride(n, incy, "incy") || !CrotValidPointer(x, "x") ||
        !CrotValidPointer(y, "y") || !CrotValidPointer(c, "c") || !CrotValidPointer(s, "s")) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    bool cDevice = false;
    bool sDevice = false;
    auto status = CheckPtrLocation(c, &cDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = CheckPtrLocation(s, &sDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const CrotTilingData tiling = CrotMakeTiling(n, incx, incy, c, s, cDevice, sDevice);

    uint32_t blocks = 1;
    status = CrotGetBlockCount(x, y, tiling, blocks);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Host values travel by value in the launch arguments. Device values are
    // read by the kernel after earlier work in handle->stream, with no D2H sync.
    const aclError launchErr = crot_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y),
        cDevice ? reinterpret_cast<const uint8_t*>(c) : nullptr,
        sDevice ? reinterpret_cast<const uint8_t*>(s) : nullptr, tiling, blocks, handle->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasCrot", "crot_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
