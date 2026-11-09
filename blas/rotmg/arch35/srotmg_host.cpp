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
 *        Arch35 (ascend950) host-side implementation.
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
 *        - All device pointers: launch a 1-block SIMT kernel so computation and
 *          results stay on device, ready for downstream ops like rotm.
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "srotmg_tiling_data.h"

// Forward declaration — defined in srotmg_kernel.cpp
void srotmg_kernel_do(
    uint8_t* d1, uint8_t* d2, uint8_t* x1, uint8_t* y1, uint8_t* param, const SrotmgTilingData& tiling,
    uint32_t numBlocks, void* stream);

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
// Core srotmg algorithm — reference BLAS CPU implementation
//
// Reference: http://www.netlib.org/blas/srotmg.f
// ==========================================================================
// netlib: flip the stored flag to -1 and reset the implied h slots to +-1
// before each rescale round, exactly as the FORTRAN does inside its loops.
// Shared by the SD1/SD2 rescale loops below (same helper as the arch22
// implementation, srotmg_compute.h).
static void SrotmgFlipFlagAndResetH(float& sh11, float& sh12, float& sh21, float& sh22, float& sflag)
{
    constexpr float ZERO = 0.0f;
    constexpr float ONE = 1.0f;
    if (sflag == ZERO) {
        sh11 = ONE;
        sh22 = ONE;
        sflag = -ONE;
    } else {
        sh21 = -ONE;
        sh12 = ONE;
        sflag = -ONE;
    }
}

static void SrotmgCpuCompute(float* d1, float* d2, float* x1, float y1, float* param)
{
    constexpr float ZERO = 0.0f;
    constexpr float ONE = 1.0f;
    constexpr float GAM = 4096.0f;
    constexpr float GAMSQ = 1.67772e7f;
    constexpr float RGAMSQ = 5.96046e-8f;
    // issue #871: Non-finite guard: inf keeps the rescale loops spinning forever (the
    // golden does too); finite inputs converge within a handful of passes, far below
    // the cap. Same guard as the arch22 implementation (srotmg_compute.h).
    constexpr uint32_t MAX_RESCALE_LOOPS = 64;

    float sd1 = *d1;
    float sd2 = *d2;
    float sx1 = *x1;
    float sy1 = y1;
    float sflag = ZERO;
    float sh11 = ZERO, sh12 = ZERO, sh21 = ZERO, sh22 = ZERO;

    if (sd1 < ZERO) {
        sflag = -ONE;
        sh11 = ZERO;
        sh12 = ZERO;
        sh21 = ZERO;
        sh22 = ZERO;
        sd1 = ZERO;
        sd2 = ZERO;
        sx1 = ZERO;
    } else {
        float sp2 = sd2 * sy1;
        if (sp2 == ZERO) {
            sflag = -2.0f;
            param[0] = sflag;
            param[1] = ZERO;
            param[2] = ZERO;
            param[3] = ZERO;
            param[4] = ZERO;
            return;
        }

        float sp1 = sd1 * sx1;
        float sq2 = sp2 * sy1;
        float sq1 = sp1 * sx1;

        if (std::abs(sq1) > std::abs(sq2)) {
            sh21 = -sy1 / sx1;
            sh12 = sp2 / sp1;
            float su = ONE - sh12 * sh21;

            if (su > ZERO) {
                sflag = ZERO;
                sd1 = sd1 / su;
                sd2 = sd2 / su;
                sx1 = sx1 * su;
            } else {
                sflag = -ONE;
                sh11 = ZERO;
                sh12 = ZERO;
                sh21 = ZERO;
                sh22 = ZERO;
                sd1 = ZERO;
                sd2 = ZERO;
                sx1 = ZERO;
            }
        } else {
            if (sq2 < ZERO) {
                sflag = -ONE;
                sh11 = ZERO;
                sh12 = ZERO;
                sh21 = ZERO;
                sh22 = ZERO;
                sd1 = ZERO;
                sd2 = ZERO;
                sx1 = ZERO;
            } else {
                sflag = ONE;
                sh11 = sp1 / sp2;
                sh22 = sx1 / sy1;
                float su = ONE + sh11 * sh22;
                float stemp = sd2 / su;
                sd2 = sd1 / su;
                sd1 = stemp;
                sx1 = sy1 * su;
            }
        }

        if (sd1 != ZERO) {
            uint32_t guard = 0;
            while ((sd1 <= RGAMSQ) || (sd1 >= GAMSQ)) {
                if (++guard > MAX_RESCALE_LOOPS) {
                    // issue #871: d1 为 ±Inf 等非有限值时 rescale 不收敛，超上限退出（对齐 arch22）
                    break;
                }
                SrotmgFlipFlagAndResetH(sh11, sh12, sh21, sh22, sflag);
                if (sd1 <= RGAMSQ) {
                    sd1 = sd1 * GAM * GAM;
                    sx1 = sx1 / GAM;
                    sh11 = sh11 / GAM;
                    sh12 = sh12 / GAM;
                } else {
                    sd1 = sd1 / (GAM * GAM);
                    sx1 = sx1 * GAM;
                    sh11 = sh11 * GAM;
                    sh12 = sh12 * GAM;
                }
            }
        }

        if (sd2 != ZERO) {
            uint32_t guard = 0;
            while ((std::abs(sd2) <= RGAMSQ) || (std::abs(sd2) >= GAMSQ)) {
                if (++guard > MAX_RESCALE_LOOPS) {
                    // issue #871: d2 为 ±Inf 等非有限值时 rescale 不收敛，超上限退出（对齐 arch22）
                    break;
                }
                SrotmgFlipFlagAndResetH(sh11, sh12, sh21, sh22, sflag);
                if (std::abs(sd2) <= RGAMSQ) {
                    sd2 = sd2 * GAM * GAM;
                    sh21 = sh21 / GAM;
                    sh22 = sh22 / GAM;
                } else {
                    sd2 = sd2 / (GAM * GAM);
                    sh21 = sh21 * GAM;
                    sh22 = sh22 * GAM;
                }
            }
        }
    }

    // STORE results
    if (sflag < ZERO) {
        param[1] = sh11;
        param[2] = sh21;
        param[3] = sh12;
        param[4] = sh22;
    } else if (sflag == ZERO) {
        param[1] = ZERO;
        param[2] = sh21;
        param[3] = sh12;
        param[4] = ZERO;
    } else {
        param[1] = sh11;
        param[2] = ZERO;
        param[3] = ZERO;
        param[4] = sh22;
    }
    param[0] = sflag;

    *d1 = sd1;
    *d2 = sd2;
    *x1 = sx1;
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

// ==========================================================================
// Public API: aclblasSrotmg
//
// All five pointers (d1, d2, x1, y1, param) must reside on the same side:
//   - All host  → CPU computation
//   - All device → device kernel
//   - Mixed     → ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasSrotmg(aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    aclblasStatus_t status = ValidateSrotmgParams(handle, d1, d2, x1, y1, param);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // Check pointer locations
    bool d1Dev = false, d2Dev = false, x1Dev = false, y1Dev = false, paramDev = false;
    status = SrotmgCheckPtrLocation(d1, &d1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotmgCheckPtrLocation(d2, &d2Dev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotmgCheckPtrLocation(x1, &x1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotmgCheckPtrLocation(y1, &y1Dev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = SrotmgCheckPtrLocation(param, &paramDev);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    bool allHost = !d1Dev && !d2Dev && !x1Dev && !y1Dev && !paramDev;
    bool allDevice = d1Dev && d2Dev && x1Dev && y1Dev && paramDev;

    // ── All host: CPU computation ──
    if (allHost) {
        float y1Val = *y1;
        float paramLocal[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        SrotmgCpuCompute(d1, d2, x1, y1Val, paramLocal);
        for (int i = 0; i < 5; i++)
            param[i] = paramLocal[i];
        return ACLBLAS_STATUS_SUCCESS;
    }

    // ── All device: kernel computation ──
    if (allDevice) {
        auto* h = handle;
        SrotmgTilingData tiling{};
        srotmg_kernel_do(
            reinterpret_cast<uint8_t*>(d1), reinterpret_cast<uint8_t*>(d2), reinterpret_cast<uint8_t*>(x1),
            reinterpret_cast<uint8_t*>(const_cast<float*>(y1)), reinterpret_cast<uint8_t*>(param), tiling, 1,
            h->stream);
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
