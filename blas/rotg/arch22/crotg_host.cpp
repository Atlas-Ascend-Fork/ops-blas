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
#include "crotg_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"

void crotg_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s,
                     const CrotgTilingData& tiling, uint32_t numBlocks, void* stream);

namespace {

// ==========================================================================
// Helper: check pointer location with aclrtPointerGetAttributes
// ==========================================================================
static aclblasStatus_t CrotgCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCrotg", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Degenerate result convention shared by every "a and b both vanish" path:
// r = 0, c = 1, s = 0
// ==========================================================================
static void SetCrotgZeroResult(aclblasComplex* a, float* c, aclblasComplex* s)
{
    *c = 1.0f;
    s->real = 0.0f;
    s->imag = 0.0f;
    a->real = 0.0f;
    a->imag = 0.0f;
}

// ==========================================================================
// Core crotg algorithm — reference BLAS CPU implementation (design 3.2.2)
// Global scaling ma + three-way branch; host side uses std::fmax/fabs/sqrt
// ==========================================================================
static void CrotgCpuCompute(aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    const float a_r = a->real;
    const float a_i = a->imag;
    const float b_r = b->real;
    const float b_i = b->imag;

    // Global scaling: max of the 4 component absolute values
    const float ma = std::fmax(std::fmax(std::fabs(a_r), std::fabs(a_i)),
                               std::fmax(std::fabs(b_r), std::fabs(b_i)));
    if (ma == 0.0f) {
        // All-zero input
        SetCrotgZeroResult(a, c, s);
        return;
    }

    // Normalize to O(1)
    const float a_sr = a_r / ma;
    const float a_si = a_i / ma;
    const float b_sr = b_r / ma;
    const float b_si = b_i / ma;
    const float abs_a = std::sqrt(a_sr * a_sr + a_si * a_si);
    const float abs_b = std::sqrt(b_sr * b_sr + b_si * b_si);
    if (abs_b == 0.0f) {
        // b vanishes while a does not: a keeps its original value (do NOT reuse the
        // all-zero convention here — a and b all zero is already handled by `ma == 0.0f`)
        *c = 1.0f;
        s->real = 0.0f;
        s->imag = 0.0f;
        a->real = a_r;
        a->imag = a_i;
        return;
    }
    if (abs_a == 0.0f) {
        // a vanishes while b does not: |a| is replaced by |b| * ma
        *c = 0.0f;
        s->real = b_sr / abs_b;
        s->imag = -b_si / abs_b;
        a->real = abs_b * ma;
        a->imag = 0.0f;
        return;
    }

    const float norm_s = std::sqrt(abs_a * abs_a + abs_b * abs_b);
    // Zero-divisor guard. The two branches above already returned, so at this point
    // abs_a > 0 and abs_b > 0, hence norm_s = sqrt(abs_a^2 + abs_b^2) > 0 and the guard is
    // unreachable in exact arithmetic — it exists so that every division below is dominated
    // by an explicit "divisor is non-zero" test that a static checker can verify locally
    // (it cannot follow the earlier control flow). It writes the same all-zero convention
    // (c=1, s=0, r=0) as the `ma == 0.0f` branch, which is exactly the crotg result when
    // both |a| and |b| vanish.
    //
    // `== 0.0f` rather than `<= 0.0f`: the two are equivalent on the whole value range of
    // norm_s (a non-negative finite value, +0.0, +Inf, or NaN — std::sqrt never returns a
    // negative value), and in particular both are false for NaN, so this guard does not
    // swallow the NaN-propagation path. The checker only recognises the `== 0.0f` form.
    if (norm_s == 0.0f) {
        SetCrotgZeroResult(a, c, s);
        return;
    }

    *c = abs_a / norm_s;
    const float sgn_ar = a_sr / abs_a;
    const float sgn_ai = a_si / abs_a;
    s->real = (sgn_ar * b_sr + sgn_ai * b_si) / norm_s;
    s->imag = (sgn_ai * b_sr - sgn_ar * b_si) / norm_s;
    a->real = sgn_ar * norm_s * ma;
    a->imag = sgn_ai * norm_s * ma;
}

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCrotgParams(aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b,
                                           float* c, aclblasComplex* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCrotg", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE("aclblasCrotg", "a, b, c and s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtContext currentCtx = nullptr;
    aclError ret = aclrtGetCurrentContext(&currentCtx);
    if (ret != ACL_SUCCESS || currentCtx == nullptr) {
        OP_LOGE("aclblasCrotg", "runtime context is not initialized, ret=%d", ret);
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Kernel launch (all-device path)
// ==========================================================================
static aclblasStatus_t LaunchCrotgKernel(aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b,
                                         float* c, aclblasComplex* s)
{
    CrotgTilingData tiling = {};
    tiling.a = reinterpret_cast<uint64_t>(a);
    tiling.b = reinterpret_cast<uint64_t>(b);
    tiling.c = reinterpret_cast<uint64_t>(c);
    tiling.s = reinterpret_cast<uint64_t>(s);

    // Scalar op: fixed single core (Block Dim = 1)
    crotg_kernel_do(reinterpret_cast<uint8_t*>(a), reinterpret_cast<uint8_t*>(b),
                    reinterpret_cast<uint8_t*>(c), reinterpret_cast<uint8_t*>(s),
                    tiling, 1, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// ==========================================================================
// Public API: aclblasCrotg
// All four pointers (a, b, c, s) must reside on the same side:
//   - All host  → CPU computation
//   - All device → device kernel
//   - Mixed     → ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasCrotg(aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b,
                             float* c, aclblasComplex* s)
{
    aclblasStatus_t status = ValidateCrotgParams(handle, a, b, c, s);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Check pointer locations
    bool aDev = false;
    bool bDev = false;
    bool cDev = false;
    bool sDev = false;
    status = CrotgCheckPtrLocation(a, &aDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = CrotgCheckPtrLocation(b, &bDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = CrotgCheckPtrLocation(c, &cDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = CrotgCheckPtrLocation(s, &sDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    const bool allHost = !aDev && !bDev && !cDev && !sDev;
    const bool allDevice = aDev && bDev && cDev && sDev;

    // ── All host: CPU computation ──
    if (allHost) {
        CrotgCpuCompute(a, b, c, s);
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── All device: kernel computation ──
    if (allDevice) {
        return LaunchCrotgKernel(handle, a, b, c, s);
    }

    // ── Mixed host/device pointers: not supported ──
    OP_LOGE("aclblasCrotg",
            "mixed host/device pointers not supported "
            "(a=%s b=%s c=%s s=%s)",
            aDev ? "dev" : "host", bDev ? "dev" : "host",
            cDev ? "dev" : "host", sDev ? "dev" : "host");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
