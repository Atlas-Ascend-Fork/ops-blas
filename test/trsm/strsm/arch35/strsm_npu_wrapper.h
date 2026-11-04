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
#include "acl/acl.h"
#include "cann_ops_blas.h"

static inline aclError AllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr)
        return ACL_SUCCESS;
    aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    ret = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dPtr);
        dPtr = nullptr;
    }
    return ret;
}

static inline void FreeDev(void* dPtr)
{
    if (dPtr)
        aclrtFree(dPtr);
}

inline aclblasStatus_t aclblasStrsm_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
{
    if (handle == nullptr || m <= 0 || n <= 0 || alpha == nullptr || B == nullptr) {
        return aclblasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    }

    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    // issue #773: 非法 lda/ldb 不做前置 H2D 分配，透传 API 由其校验返回 INVALID_VALUE，
    // 避免分配失败返回 ALLOC_FAILED 掩盖接口真实返回码
    if (lda < std::max(1, aDim) || ldb < std::max(1, m)) {
        return aclblasStrsm(handle, side, uplo, trans, diag, m, n, alpha, nullptr, lda, nullptr, ldb);
    }

    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;

    if (A != nullptr && *alpha != 0.0f) {
        aclError aclRet = AllocCopyH2D(dA, A, aBytes);
        if (aclRet != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclError aclRet = AllocCopyH2D(dB, B, bBytes);
    if (aclRet != ACL_SUCCESS) {
        FreeDev(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasStrsm(
        handle, side, uplo, trans, diag, m, n, alpha, static_cast<const float*>(dA), lda, static_cast<float*>(dB), ldb);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasStatus_t getStreamRet = aclblasGetStream(handle, &stream);
        if (getStreamRet != ACLBLAS_STATUS_SUCCESS) {
            FreeDev(dA);
            FreeDev(dB);
            return getStreamRet;
        }
        if (stream != nullptr) {
            aclError syncRet = aclrtSynchronizeStream(stream);
            if (syncRet != ACL_SUCCESS) {
                FreeDev(dA);
                FreeDev(dB);
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        aclError d2hRet = aclrtMemcpy(B, bBytes, dB, bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (d2hRet != ACL_SUCCESS)
            ret = ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    FreeDev(dA);
    FreeDev(dB);
    return ret;
}
