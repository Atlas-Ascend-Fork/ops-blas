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
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"
#include "fill.h"

inline size_t SgeamBufferBytes(aclblasOperation_t trans, int m, int n, int ld)
{
    size_t cols = (trans == ACLBLAS_OP_N) ? static_cast<size_t>(n) : static_cast<size_t>(m);
    return static_cast<size_t>(ld) * cols * sizeof(float);
}

inline std::unique_ptr<DeviceBuffer> SgeamAllocAndCopy(const float* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline std::unique_ptr<DeviceBuffer> SgeamAllocOutput(size_t cBytes)
{
    auto buf = std::make_unique<DeviceBuffer>(cBytes);
    std::vector<float> sentinel(cBytes / sizeof(float), kBlasSentinel);
    buf->copyFromHost(sentinel.data(), cBytes);
    return buf;
}

inline float* SgeamGetCPtr(bool inplaceAC, bool inplaceBC, DeviceBuffer* dA, DeviceBuffer* dB, DeviceBuffer* dC)
{
    if (inplaceAC) {
        return static_cast<float*>(dA->ptr());
    }
    if (inplaceBC) {
        return static_cast<float*>(dB->ptr());
    }
    return static_cast<float*>(dC->ptr());
}

inline void SgeamCopyBack(bool inplaceAC, bool inplaceBC, DeviceBuffer* dA, DeviceBuffer* dB, DeviceBuffer* dC,
                          float* C, size_t aBytes, size_t bBytes, size_t cBytes)
{
    if (inplaceAC && dA) {
        dA->copyToHost(C, aBytes);
    } else if (inplaceBC && dB) {
        dB->copyToHost(C, bBytes);
    } else if (dC) {
        dC->copyToHost(C, cBytes);
    }
}

inline aclblasStatus_t aclblasSgeam_npu(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const float* alpha, const float* A, int lda,
    const float* beta, const float* B, int ldb, float* C, int ldc)
{
    if (m <= 0 || n <= 0) {
        float dummyA = 0.0f;
        float dummyB = 0.0f;
        float dummyC = 0.0f;
        float dummyAlpha = alpha ? *alpha : 0.0f;
        float dummyBeta = beta ? *beta : 0.0f;
        return aclblasSgeam(handle, transa, transb, m, n,
            &dummyAlpha, &dummyA, std::max(lda, 1),
            &dummyBeta, &dummyB, std::max(ldb, 1),
            &dummyC, std::max(ldc, 1));
    }
    size_t aBytes = SgeamBufferBytes(transa, m, n, lda);
    size_t bBytes = SgeamBufferBytes(transb, m, n, ldb);
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);

    bool inplaceAC = (A != nullptr && C == A);
    bool inplaceBC = (B != nullptr && C == B);

    auto dA = SgeamAllocAndCopy(A, aBytes);
    auto dB = SgeamAllocAndCopy(B, bBytes);

    std::unique_ptr<DeviceBuffer> dC;
    if (!inplaceAC && !inplaceBC) {
        dC = SgeamAllocOutput(cBytes);
    }

    float* dC_ptr = SgeamGetCPtr(inplaceAC, inplaceBC, dA.get(), dB.get(), dC.get());

    aclblasStatus_t ret = aclblasSgeam(
        handle, transa, transb, m, n, alpha,
        dA ? static_cast<const float*>(dA->ptr()) : nullptr, lda, beta,
        dB ? static_cast<const float*>(dB->ptr()) : nullptr, ldb, dC_ptr, ldc);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        SgeamCopyBack(inplaceAC, inplaceBC, dA.get(), dB.get(), dC.get(), C, aBytes, bBytes, cBytes);
    }
    return ret;
}
