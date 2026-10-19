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
 * \file crotg_host.cpp
 * \brief Construct complex Givens rotation parameters (BLAS Level 1 scalar operation).
 *        Arch35 (ascend950) host-side implementation.
 *
 *        crotg reads two complex64 scalars (a, b) and constructs the Givens plane rotation
 *        so that [c s; -conjg(s) c] * [a; b] = [r; 0]. Output: a is overwritten by r
 *        (complex64), c is real float32, s is complex64; b is read-only.
 *
 *        The four pointers (a, b, c, s) must be all-host or all-device; mixed locations
 *        are rejected per the cuBLAS/BLAS standard.
 *
 *        Strategy:
 *        - All host pointers: direct CPU computation (no kernel, no memcpy).
 *        - All device pointers: launch a 1-block SIMT kernel so computation and results
 *          stay on device, ready for downstream ops like crot.
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "crotg_kernel.h"
#include "crotg_common.h"

// ==========================================================================
// Helper: check pointer location with aclrtPointerGetAttributes
// ==========================================================================
static aclblasStatus_t CrotgCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasCrotg", "invalid parameter");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Core crotg algorithm - reference BLAS CPU implementation (Netlib crotg)
// The algorithm body is defined by the CROTG_ALGO macro in crotg_common.h
// (shared with the device kernel), ensuring bit-exact host/device results.
// Reference: https://www.netlib.org/blas/crotg.f
// rnorm is computed via hypotf(anorm, bnorm), whose internal scaling avoids
// overflow/underflow even when anorm/bnorm are near FLT_MAX/FLT_MIN.
// ==========================================================================
static void CrotgCpuCompute(aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    // Save inputs before macro writes back to a->real/a->imag
    float ar = a->real;
    float ai = a->imag;
    float br = b->real;
    float bi = b->imag;

    float cr;
    float sr;
    float si;
    float rr;
    float ri;
    CROTG_ALGO(ar, ai, br, bi, &cr, &sr, &si, &rr, &ri);

    // Write back: a overwritten with r, c is real, s is complex, b not overwritten
    a->real = rr;
    a->imag = ri;
    *c = cr;
    s->real = sr;
    s->imag = si;
}

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCrotgParams(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCrotg", "invalid parameter");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE("aclblasCrotg", "invalid parameter");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API: aclblasCrotg
// All four pointers (a, b, c, s) must reside on the same side:
//   - All host   -> CPU computation
//   - All device -> device kernel
//   - Mixed      -> ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    aclblasStatus_t status = ValidateCrotgParams(handle, a, b, c, s);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Constraint: all four pointers must be on the same side (all-host or all-device).
    // Use a's location as reference and check b/c/s; fail-fast on the first mismatch.
    bool aDev = false;
    status = CrotgCheckPtrLocation(a, &aDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    const char* aSide = aDev ? "dev" : "host";
    auto expectSameSide = [&](void* ptr, const char* name) -> aclblasStatus_t {
        bool dev = false;
        aclblasStatus_t st = CrotgCheckPtrLocation(ptr, &dev);
        if (st != ACLBLAS_STATUS_SUCCESS) return st;
        if (dev != aDev) {
            OP_LOGE("aclblasCrotg", "invalid parameter");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        return ACLBLAS_STATUS_SUCCESS;
    };
    status = expectSameSide(b, "b");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = expectSameSide(c, "c");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = expectSameSide(s, "s");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    // All pointers on the same side: all-host -> CPU compute, all-device -> kernel
    if (!aDev) {
        CrotgCpuCompute(a, b, c, s);
        return ACLBLAS_STATUS_SUCCESS;
    }

    OP_LOGI("aclblasCrotg", "launching kernel: blocks=1");
    crotg_kernel_do(
        reinterpret_cast<uint8_t*>(a),
        reinterpret_cast<uint8_t*>(b),
        reinterpret_cast<uint8_t*>(c),
        reinterpret_cast<uint8_t*>(s),
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
