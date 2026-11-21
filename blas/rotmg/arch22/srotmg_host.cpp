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
 * \file srotmg_host.cpp
 * \brief Construct modified Givens rotation (BLAS Level 1 scalar operation).
 *        Arch22 (Atlas A2/A3) host-side implementation.
 *
 *        rotmg is a pure scalar computation: it reads four scalars (d1, d2, x1, y1)
 *        and writes back modified d1, d2, x1 together with a 5-element param array
 *        that encodes a modified Givens rotation matrix H.
 *
 *        The five pointers (d1, d2, x1, y1, param) must be all-host or all-device;
 *        mixed locations are rejected per the BLAS standard.
 *
 *        Strategy:
 *        - All host pointers: direct CPU computation (no kernel, no memcpy).
 *        - All device pointers: launch a 1-block AIV kernel so computation and
 *          results stay on device, ready for downstream ops like rotm.
 */

#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "srotmg_tiling_data.h"
#include "srotmg_kernel.h"
#include "srotmg_compute.h"

namespace {

// ==========================================================================
// Helper: check pointer location with aclrtPointerGetAttributes
// ==========================================================================
static aclblasStatus_t SrotmgCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSrotmg", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Core srotmg computation — single-sourced in srotmg_compute.h, shared with
// the arch22 device kernel so both paths stay bit-for-bit identical.
// ==========================================================================
static void SrotmgCpuCompute(float* d1, float* d2, float* x1, float y1, float* param)
{
    float sd1 = *d1;
    float sd2 = *d2;
    float sx1 = *x1;
    float dflag = srotmg::ZERO;
    float dh11 = srotmg::ZERO;
    float dh21 = srotmg::ZERO;
    float dh12 = srotmg::ZERO;
    float dh22 = srotmg::ZERO;

    srotmg::ComputeScalars(sd1, sd2, sx1, y1, dflag, dh11, dh21, dh12, dh22);

    if (dflag == srotmg::FLAG_M2) {
        param[0] = srotmg::FLAG_M2;
        param[1] = srotmg::ZERO;
        param[2] = srotmg::ZERO;
        param[3] = srotmg::ZERO;
        param[4] = srotmg::ZERO;
        return;
    }

    if (dflag < srotmg::ZERO) {
        param[1] = dh11;
        param[2] = dh21;
        param[3] = dh12;
        param[4] = dh22;
    } else if (dflag == srotmg::ZERO) {
        param[1] = srotmg::ZERO;
        param[2] = dh21;
        param[3] = dh12;
        param[4] = srotmg::ZERO;
    } else {
        param[1] = dh11;
        param[2] = srotmg::ZERO;
        param[3] = srotmg::ZERO;
        param[4] = dh22;
    }
    param[0] = dflag;

    *d1 = sd1;
    *d2 = sd2;
    *x1 = sx1;
}

// ==========================================================================
// Pointer-location resolution: query all five pointers on every call.
// A cached verdict is unsafe even with re-validation — the anchor pointer can
// stay put while the other four are freed and reallocated to the other side,
// and five aclrtPointerGetAttributes calls are cheap for this operator.
// ==========================================================================
static aclblasStatus_t SrotmgLocatePointers(
    float* d1, float* d2, float* x1, const float* y1, float* param, bool* d1Dev, bool* d2Dev, bool* x1Dev, bool* y1Dev,
    bool* paramDev)
{
    aclblasStatus_t status;
    status = SrotmgCheckPtrLocation(d1, d1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = SrotmgCheckPtrLocation(d2, d2Dev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = SrotmgCheckPtrLocation(x1, x1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = SrotmgCheckPtrLocation(y1, y1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = SrotmgCheckPtrLocation(param, paramDev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateSrotmgParams(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrotmg", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (d1 == nullptr || d2 == nullptr || x1 == nullptr || y1 == nullptr || param == nullptr) {
        OP_LOGE("aclblasSrotmg", "input pointers contain a nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// ==========================================================================
// Public API: aclblasSrotmg
// All five pointers (d1, d2, x1, y1, param) must reside on the same side:
//   - All host   → CPU computation
//   - All device → device kernel
//   - Mixed      → ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasSrotmg(aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    aclblasStatus_t status = ValidateSrotmgParams(handle, d1, d2, x1, y1, param);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSrotmg", "parameter validation failed, ret=%d", static_cast<int>(status));
        return status;
    }

    // Check pointer locations
    bool d1Dev = false;
    bool d2Dev = false;
    bool x1Dev = false;
    bool y1Dev = false;
    bool paramDev = false;
    status = SrotmgLocatePointers(d1, d2, x1, y1, param, &d1Dev, &d2Dev, &x1Dev, &y1Dev, &paramDev);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSrotmg", "pointer location check failed, ret=%d", static_cast<int>(status));
        return status;
    }

    bool allHost = !d1Dev && !d2Dev && !x1Dev && !y1Dev && !paramDev;
    bool allDevice = d1Dev && d2Dev && x1Dev && y1Dev && paramDev;

    // ── All host: CPU computation ──
    if (allHost) {
        float y1Val = *y1;
        float paramLocal[5] = {srotmg::ZERO, srotmg::ZERO, srotmg::ZERO, srotmg::ZERO, srotmg::ZERO};
        SrotmgCpuCompute(d1, d2, x1, y1Val, paramLocal);
        for (int i = 0; i < 5; i++) {
            param[i] = paramLocal[i];
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── All device: kernel computation ──
    if (allDevice) {
        SrotmgTilingData tiling = {};
        tiling.d1 = reinterpret_cast<uint64_t>(d1);
        tiling.d2 = reinterpret_cast<uint64_t>(d2);
        tiling.x1 = reinterpret_cast<uint64_t>(x1);
        tiling.y1 = reinterpret_cast<uint64_t>(const_cast<float*>(y1));
        tiling.param = reinterpret_cast<uint64_t>(param);

        // issue #1152: a failed launch previously reported SUCCESS with a stale
        // param array consumed later by rotm users; surface it as
        // EXECUTION_FAILED like the chbmv host does.
        const aclError launch = srotmg_kernel_do(tiling, 1, handle->stream);
        if (launch != ACL_SUCCESS) {
            OP_LOGE("aclblasSrotmg", "srotmg_kernel_do failed: %d", launch);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── Mixed host/device pointers: not supported ──
    OP_LOGE(
        "aclblasSrotmg",
        "mixed host/device pointers not supported "
        "(d1=%s d2=%s x1=%s y1=%s param=%s)",
        d1Dev ? "dev" : "host", d2Dev ? "dev" : "host", x1Dev ? "dev" : "host", y1Dev ? "dev" : "host",
        paramDev ? "dev" : "host");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
