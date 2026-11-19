/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

static aclblasStatus_t ChprAllocCopyToDevice(void** dst, const void* src, size_t bytes)
{
    if (src == nullptr || bytes == 0) {
        *dst = nullptr;
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclError aclRet = aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dst);
        *dst = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// issue #965: alphaOnDevice 时按设备指针传入，覆盖 Device 标量指针路径
static aclblasStatus_t ChprPrepareAlphaDevice(const float* alpha, void** dAlpha, const float** alphaArg)
{
    aclblasStatus_t st = ChprAllocCopyToDevice(dAlpha, alpha, sizeof(float));
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    *alphaArg = static_cast<const float*>(*dAlpha);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasChpr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap, bool alphaOnDevice = false)
{
    if (handle == nullptr || n <= 0) {
        return aclblasChpr(handle, uplo, n, alpha, x, incx, ap);
    }

    const int allocN = std::max(1, n);
    const int absIncx = std::abs(incx);
    const size_t xBytes = static_cast<size_t>((allocN - 1) * absIncx + 1) * sizeof(aclblasComplex);
    const size_t apBytes = static_cast<size_t>(allocN) * (allocN + 1) / 2 * sizeof(aclblasComplex);

    void* dX = nullptr;
    void* dAP = nullptr;
    void* dAlpha = nullptr;
    const float* alphaArg = alpha;

    aclblasStatus_t st = ChprAllocCopyToDevice(&dX, x, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    st = ChprAllocCopyToDevice(&dAP, ap, apBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        if (dX) {
            aclrtFree(dX);
        }
        return st;
    }

    if (alphaOnDevice && alpha != nullptr) {
        st = ChprPrepareAlphaDevice(alpha, &dAlpha, &alphaArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            aclrtFree(dX);
            aclrtFree(dAP);
            return st;
        }
    }

    aclblasStatus_t ret = aclblasChpr(
        handle, uplo, n, alphaArg, static_cast<const aclblasComplex*>(dX), incx, static_cast<aclblasComplex*>(dAP));

    aclrtSynchronizeDevice();
    if (ret == ACLBLAS_STATUS_SUCCESS && ap != nullptr) {
        aclrtMemcpy(ap, apBytes, dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (dAlpha) {
        aclrtFree(dAlpha);
    }
    if (dX) {
        aclrtFree(dX);
    }
    if (dAP) {
        aclrtFree(dAP);
    }
    return ret;
}
