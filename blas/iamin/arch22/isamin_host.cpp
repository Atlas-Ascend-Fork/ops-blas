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
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "isamin_kernel.h"
#include "isamin_tiling_data.h"

namespace {

constexpr uint32_t ISAMIN_MIN_ELEMENTS_PER_CORE = 4096;

aclblasStatus_t ValidateIsaminParams(
    aclblasHandle_t handle, int n, int incx, const float* x, int* result, bool& quickReturn)
{
    quickReturn = false;
    if (handle == nullptr) {
        OP_LOGE("aclblasIsamin", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasIsamin", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        quickReturn = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasIsamin", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const int64_t lastElement = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    if (lastElement > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(
            "aclblasIsamin", "strided offset overflow: n=%d incx=%d lastElement=%lld", n, incx,
            static_cast<long long>(lastElement));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

IsaminTilingData CalcIsaminTiling(int n, int incx, uint32_t aivCoreNum)
{
    IsaminTilingData tiling{};
    const uint32_t totalN = static_cast<uint32_t>(n);
    const uint32_t blocksByWork = CeilDiv<uint32_t>(totalN, ISAMIN_MIN_ELEMENTS_PER_CORE);
    const uint32_t useCoreNum = std::max(1U, std::min(aivCoreNum, blocksByWork));

    tiling.totalN = totalN;
    tiling.incx = incx;
    tiling.useCoreNum = useCoreNum;
    tiling.perCoreN = CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalN, useCoreNum), ISAMIN_ARCH22_ALIGN_ELEMENTS);
    tiling.tileSize = ISAMIN_ARCH22_TILE_SIZE;
    return tiling;
}

aclblasStatus_t LaunchIsaminKernel(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    bool quickReturn = false;
    const aclblasStatus_t status = ValidateIsaminParams(handle, n, incx, x, result, quickReturn);
    if (status != ACLBLAS_STATUS_SUCCESS || quickReturn) {
        if (quickReturn) {
            const aclError memRet = aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
            if (memRet != ACL_SUCCESS) {
                OP_LOGE("aclblasIsamin", "failed to write quick-return result, ret=%d", static_cast<int>(memRet));
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        return status;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasIsamin", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const IsaminTilingData tiling = CalcIsaminTiling(n, incx, aivCoreNum);
    const size_t requiredWorkspace = static_cast<size_t>(tiling.useCoreNum) * 2U * sizeof(float);
    if (GetEffectiveWorkspace(handle) == nullptr || GetEffectiveWorkspaceSize(handle) < requiredWorkspace) {
        OP_LOGE(
            "aclblasIsamin", "workspace too small: need=%zu, available=%zu", requiredWorkspace,
            GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    isamin_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(result),
        reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle)), tiling, tiling.useCoreNum,
        reinterpret_cast<void*>(handle->stream));
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return LaunchIsaminKernel(handle, n, x, incx, result);
}
