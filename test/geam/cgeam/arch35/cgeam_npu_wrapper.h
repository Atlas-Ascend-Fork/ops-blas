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

inline size_t CgeamBufferBytes(aclblasOperation_t trans, int m, int n, int ld)
{
    size_t cols = (trans == ACLBLAS_OP_N) ? static_cast<size_t>(n) : static_cast<size_t>(m);
    return static_cast<size_t>(ld) * cols * sizeof(aclblasComplex);
}

inline std::unique_ptr<DeviceBuffer> CgeamAllocAndCopy(const aclblasComplex* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline std::unique_ptr<DeviceBuffer> CgeamAllocOutput(size_t cBytes)
{
    auto buf = std::make_unique<DeviceBuffer>(cBytes);
    std::vector<float> sentinel(cBytes / sizeof(float), kBlasSentinel);
    buf->copyFromHost(sentinel.data(), cBytes);
    return buf;
}

inline aclblasComplex* CgeamGetCPtr(bool inplaceAC, bool inplaceBC, DeviceBuffer* dA, DeviceBuffer* dB, DeviceBuffer* dC)
{
    if (inplaceAC) {
        return static_cast<aclblasComplex*>(dA->ptr());
    }
    if (inplaceBC) {
        return static_cast<aclblasComplex*>(dB->ptr());
    }
    return dC ? static_cast<aclblasComplex*>(dC->ptr()) : nullptr;
}

inline void CgeamCopyBack(bool inplaceAC, bool inplaceBC, DeviceBuffer* dA, DeviceBuffer* dB, DeviceBuffer* dC,
                          aclblasComplex* C, size_t aBytes, size_t bBytes, size_t cBytes)
{
    if (inplaceAC && dA) {
        dA->copyToHost(C, aBytes);
    } else if (inplaceBC && dB) {
        dB->copyToHost(C, bBytes);
    } else if (dC) {
        dC->copyToHost(C, cBytes);
    }
}

inline aclblasStatus_t aclblasCgeam_npu(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* beta, const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
{
    if (m <= 0 || n <= 0) {
        aclblasComplex dummyA = {0.0f, 0.0f};
        aclblasComplex dummyB = {0.0f, 0.0f};
        aclblasComplex dummyC = {0.0f, 0.0f};
        aclblasComplex dummyAlpha = alpha ? *alpha : aclblasComplex{0.0f, 0.0f};
        aclblasComplex dummyBeta = beta ? *beta : aclblasComplex{0.0f, 0.0f};
        return aclblasCgeam(handle, transa, transb, m, n,
            &dummyAlpha, &dummyA, std::max(lda, 1),
            &dummyBeta, &dummyB, std::max(ldb, 1),
            &dummyC, std::max(ldc, 1));
    }
    size_t aBytes = CgeamBufferBytes(transa, m, n, lda);
    size_t bBytes = CgeamBufferBytes(transb, m, n, ldb);
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);

    bool inplaceAC = (A != nullptr && C == A);
    bool inplaceBC = (B != nullptr && C == B);

    auto dA = CgeamAllocAndCopy(A, aBytes);
    auto dB = CgeamAllocAndCopy(B, bBytes);

    std::unique_ptr<DeviceBuffer> dC;
    if (!inplaceAC && !inplaceBC && C != nullptr) {
        dC = CgeamAllocOutput(cBytes);
    }

    aclblasComplex* dC_ptr = CgeamGetCPtr(inplaceAC, inplaceBC, dA.get(), dB.get(), dC.get());

    aclblasStatus_t ret = aclblasCgeam(
        handle, transa, transb, m, n, alpha,
        dA ? static_cast<const aclblasComplex*>(dA->ptr()) : nullptr, lda, beta,
        dB ? static_cast<const aclblasComplex*>(dB->ptr()) : nullptr, ldb, dC_ptr, ldc);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        CgeamCopyBack(inplaceAC, inplaceBC, dA.get(), dB.get(), dC.get(), C, aBytes, bBytes, cBytes);
    }
    return ret;
}
