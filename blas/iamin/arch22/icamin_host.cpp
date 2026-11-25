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
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "icamin_kernel.h"

namespace {
uint32_t Align64(uint32_t count) { return static_cast<uint32_t>((static_cast<uint64_t>(count) + 63) / 64 * 64); }

IcaminArch22Tiling MakeTiling(uint32_t n, uint32_t incx, uint32_t availableCores)
{
    IcaminArch22Tiling t{};
    t.n = n;
    t.incx = incx;
    t.layout = incx == 1 ? IcaminLayout::CONTIGUOUS :
               incx <= 4 ? IcaminLayout::DENSE_STRIDED :
                           IcaminLayout::SPARSE_STRIDED;
    // One core owns at least one useful vector tile; small inputs avoid a barrier entirely.
    constexpr uint32_t MIN_CORE_ELEMENTS = 4096;
    // The API rejects zero AIV cores and returns before tiling when n is zero.
    uint32_t requested = std::min(availableCores, (n + MIN_CORE_ELEMENTS - 1) / MIN_CORE_ELEMENTS);
    if (requested == 0) {
        requested = 1;
    }
    t.perCore = Align64(static_cast<uint32_t>((static_cast<uint64_t>(n) + requested - 1) / requested));
    t.cores = static_cast<uint32_t>((static_cast<uint64_t>(n) + t.perCore - 1) / t.perCore);
    // A short contiguous slice fits in at most two large single-buffer transfers.
    bool largeTile = incx == 1 && t.perCore <= 2 * ICAMIN_LARGE_TILE;
    t.buffers = largeTile ? 1 : 2;
    uint32_t maxTile = largeTile ? ICAMIN_LARGE_TILE : incx == 1 ? ICAMIN_CONTIGUOUS_TILE : ICAMIN_STRIDED_TILE;
    t.tile = std::min(maxTile, t.perCore);
    uint32_t inputFloats = t.layout == IcaminLayout::CONTIGUOUS    ? 2 * t.tile :
                           t.layout == IcaminLayout::DENSE_STRIDED ? 2 * ((t.tile - 1) * incx + 1) :
                                                                     ICAMIN_SLOT_FLOATS * t.tile;
    t.inputFloats = (inputFloats + 7) / 8 * 8;
    t.slots = Align64(std::max(ICAMIN_GROUP_TILES, t.cores));
    // Short contiguous slices use stream-ordered stages instead of an in-kernel barrier.
    uint32_t tiles = (t.perCore + t.tile - 1) / t.tile;
    t.reduction = t.cores == 1                                                        ? IcaminReduction::SINGLE_CORE :
                  t.layout == IcaminLayout::CONTIGUOUS && tiles <= ICAMIN_GROUP_TILES ? IcaminReduction::STAGED :
                                                                                        IcaminReduction::FUSED;
    return t;
}

uint64_t RequiredUbBytes(const IcaminArch22Tiling& t)
{
    uint64_t bytes = static_cast<uint64_t>(t.buffers) * t.inputFloats * sizeof(float);
    bytes += static_cast<uint64_t>(t.tile) * sizeof(float); // magnitudes
    if (t.layout != IcaminLayout::CONTIGUOUS) {
        bytes += static_cast<uint64_t>(2) * t.tile * sizeof(float); // imaginary part and gather offsets
    }
    bytes += static_cast<uint64_t>(t.slots) * (ICAMIN_SLOT_FLOATS + 1) * sizeof(float);
    // A2/A3 Compare uses one bit per result, with the same 32-byte rounding as InitArena.
    bytes += (static_cast<uint64_t>(t.tile) + 255) / 256 * 32;
    bytes += 2048 + ICAMIN_METADATA_FLOATS * sizeof(float); // ReduceMin scratch and vector metadata
    bytes += 8192;                                          // Select's internal A2/A3 temporary storage
    return bytes;
}
aclblasStatus_t ValidateArguments(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasIcamin", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0 || result == nullptr) {
        OP_LOGE("aclblasIcamin", "invalid n=%d or null result", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || static_cast<int64_t>(n - 1) * incx > INT32_MAX) {
        OP_LOGE("aclblasIcamin", "null x or strided offset overflow: n=%d incx=%d", n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    aclblasStatus_t status = ValidateArguments(handle, n, x, incx, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0 || incx < 1) {
        aclError error = aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
        if (error != ACL_SUCCESS) {
            OP_LOGE("aclblasIcamin", "quick-return write failed, ret=%d", error);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || platform->GetCoreNumAiv() == 0) {
        OP_LOGE("aclblasIcamin", "AIV platform information is unavailable");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    IcaminArch22Tiling tiling = MakeTiling(n, incx, platform->GetCoreNumAiv());
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    if (RequiredUbBytes(tiling) > ubBytes) {
        OP_LOGE("aclblasIcamin", "UB capacity is insufficient");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    void* workspace = nullptr;
    if (tiling.cores > 1) {
        size_t required = static_cast<size_t>(tiling.cores) * ICAMIN_SLOT_FLOATS * sizeof(float);
        if (tiling.reduction == IcaminReduction::STAGED) {
            required += ICAMIN_FIRST_NORM_FLOATS * sizeof(float);
        }
        if (!CheckEffectiveWorkspaceSize(handle, required)) {
            OP_LOGE("aclblasIcamin", "workspace allocation failed");
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        workspace = GetEffectiveWorkspace(handle);
        if (workspace == nullptr) {
            OP_LOGE("aclblasIcamin", "handle workspace is nullptr");
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }
    const aclError launchErr = icamin_arch22_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(result),
        reinterpret_cast<uint8_t*>(workspace), tiling, reinterpret_cast<void*>(handle->stream));
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasIcamin", "icamin_arch22_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
