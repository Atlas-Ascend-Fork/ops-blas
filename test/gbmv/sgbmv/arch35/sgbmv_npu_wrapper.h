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

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasSgbmv_npu(
    aclblasHandle_t handle,
    aclblasOperation_t trans,
    int m, int n, int kl, int ku,
    const float* alpha,
    const float* a, int lda,
    const float* x, int incx,
    const float* beta,
    float* y, int incy)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasSgbmv(handle, trans, m, n, kl, ku,
            alpha, a, lda, x, incx, beta, y, incy);
    }

    const int allocM = std::max(1, m);
    const int allocN = std::max(1, n);
    const int allocLda = std::max(lda, kl + ku + 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);

    const size_t aBytes = static_cast<size_t>(allocLda) * allocN * sizeof(float);
    auto vecBytes = [](int cnt, int step) {
        return (cnt > 0) ? static_cast<size_t>((cnt - 1) * step + 1) * sizeof(float) : sizeof(float);
    };
    const size_t xBytes = vecBytes(xCount, absIncx);
    const size_t yBytes = vecBytes(yCount, absIncy);

    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    if (a != nullptr) {
        aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dA, aBytes, a, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dA); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { if (dA) aclrtFree(dA); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { if (dA) aclrtFree(dA); aclrtFree(dX); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (y != nullptr) {
        aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (dA) aclrtFree(dA); if (dX) aclrtFree(dX); return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (dA) aclrtFree(dA); if (dX) aclrtFree(dX); aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclblasStatus_t ret = aclblasSgbmv(handle, trans, m, n, kl, ku,
        alpha, static_cast<const float*>(dA), lda,
        static_cast<const float*>(dX), incx,
        beta, static_cast<float*>(dY), incy);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (ret == ACLBLAS_STATUS_SUCCESS && y != nullptr) {
        if (aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dA) aclrtFree(dA);
    if (dX) aclrtFree(dX);
    if (dY) aclrtFree(dY);
    return ret;
}

