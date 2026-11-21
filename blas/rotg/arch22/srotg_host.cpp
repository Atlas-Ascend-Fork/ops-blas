/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "srotg_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"

aclError srotg_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s, const SrotgTilingData& tiling, uint32_t numBlocks, void* stream);

namespace {

constexpr float SROTG_ZERO = 0.0f;
constexpr float SROTG_ONE = 1.0f;
constexpr float SROTG_SAFMIN = 1.1754943508222875e-38f;
constexpr float SROTG_SAFMAX = 1.7014118346046923e+38f;

// ==========================================================================
// Helper: check pointer location with aclrtPointerGetAttributes
// ==========================================================================
static aclblasStatus_t SrotgCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSrotg", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Core srotg algorithm — reference BLAS CPU implementation
// Reference: http://www.netlib.org/lapack/explore-html/d7/dc5/group__rotg_ga55d839ab27a662d848390a2bbe3ee9d3.html
// ==========================================================================
static float Abs(float value) { return value < SROTG_ZERO ? -value : value; }

static float Max(float lhs, float rhs) { return lhs > rhs ? lhs : rhs; }

static float Min(float lhs, float rhs) { return lhs < rhs ? lhs : rhs; }

static float Sign(float value) { return value < SROTG_ZERO ? -SROTG_ONE : SROTG_ONE; }

static void SrotgCpuCompute(float* a, float* b, float* c, float* s)
{
    const float aValue = *a;
    const float bValue = *b;
    const float absA = Abs(aValue);
    const float absB = Abs(bValue);

    if (absA == SROTG_ZERO && absB == SROTG_ZERO) {
        *a = SROTG_ZERO;
        *b = SROTG_ZERO;
        *c = SROTG_ONE;
        *s = SROTG_ZERO;
        return;
    }

    const float scale = Min(SROTG_SAFMAX, Max(SROTG_SAFMIN, Max(absA, absB)));
    const float scaledA = aValue / scale;
    const float scaledB = bValue / scale;
    const float sigma = absA > absB ? Sign(aValue) : Sign(bValue);
    const float r = sigma * (scale * std::sqrt(scaledA * scaledA + scaledB * scaledB));

    if (r == SROTG_ZERO) {
        *a = r;
        *b = SROTG_ZERO;
        *c = SROTG_ONE;
        *s = SROTG_ZERO;
        return;
    }

    const float cValue = aValue / r;
    const float sValue = bValue / r;
    const float z = absA > absB ? sValue : (cValue == SROTG_ZERO ? SROTG_ONE : SROTG_ONE / cValue);

    *a = r;
    *b = z;
    *c = cValue;
    *s = sValue;
}

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateSrotgParams(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrotg", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE("aclblasSrotg", "a, b, c and s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtContext currentCtx = nullptr;
    aclError ret = aclrtGetCurrentContext(&currentCtx);
    if (ret != ACL_SUCCESS || currentCtx == nullptr) {
        OP_LOGE("aclblasSrotg", "runtime context is not initialized, ret=%d", ret);
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Kernel launch (all-device path)
// ==========================================================================
static aclblasStatus_t LaunchSrotgKernel(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    SrotgTilingData tiling = {};
    tiling.a = reinterpret_cast<uint64_t>(a);
    tiling.b = reinterpret_cast<uint64_t>(b);
    tiling.c = reinterpret_cast<uint64_t>(c);
    tiling.s = reinterpret_cast<uint64_t>(s);

    OP_LOGD("aclblasSrotg", "tiling: scalar op launchBlocks=1");
    OP_LOGI("aclblasSrotg", "launching kernel");

    const aclError launch = srotg_kernel_do(
        reinterpret_cast<uint8_t*>(a), reinterpret_cast<uint8_t*>(b), reinterpret_cast<uint8_t*>(c),
        reinterpret_cast<uint8_t*>(s), tiling, 1, useStream);
    // issue #1152: a failed launch previously reported SUCCESS with stale scalars;
    // surface it as EXECUTION_FAILED like the chbmv host does.
    if (launch != ACL_SUCCESS) {
        OP_LOGE("aclblasSrotg", "srotg_kernel_do failed: %d", launch);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// ==========================================================================
// Public API: aclblasSrotg
//
// All four pointers (a, b, c, s) must reside on the same side:
//   - All host  → CPU computation
//   - All device → device kernel
//   - Mixed     → ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasSrotg(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    aclblasStatus_t status = ValidateSrotgParams(handle, a, b, c, s);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Check pointer locations
    bool aDev = false, bDev = false, cDev = false, sDev = false;
    status = SrotgCheckPtrLocation(a, &aDev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotgCheckPtrLocation(b, &bDev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotgCheckPtrLocation(c, &cDev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotgCheckPtrLocation(s, &sDev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    bool allHost = !aDev && !bDev && !cDev && !sDev;
    bool allDevice = aDev && bDev && cDev && sDev;

    // ── All host: CPU computation ──
    if (allHost) {
        SrotgCpuCompute(a, b, c, s);
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── All device: kernel computation ──
    if (allDevice) {
        return LaunchSrotgKernel(handle, a, b, c, s);
    }

    // ── Mixed host/device pointers: not supported ──
    OP_LOGE(
        "aclblasSrotg",
        "mixed host/device pointers not supported "
        "(a=%s b=%s c=%s s=%s)",
        aDev ? "dev" : "host", bDev ? "dev" : "host", cDev ? "dev" : "host", sDev ? "dev" : "host");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
