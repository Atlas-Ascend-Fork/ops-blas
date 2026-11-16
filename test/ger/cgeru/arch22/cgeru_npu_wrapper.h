/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_npu_wrapper.h
 * \brief Test harness for aclblasCgeru on arch22: runs one call with device
 *        resident operands (H2D -> call -> sync -> D2H) and returns the raw
 *        aclblasStatus_t.
 *
 * Argument-invalid combinations are passed through with the host pointers
 * unchanged: the operator must reject them without touching any device
 * buffer, which is exactly the behaviour the negative CSV cases pin down.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace cgeru_arch22_test {

// Physical element count of a strided vector, 0 when the call cannot touch it.
inline uint64_t PhysicalComplexCount(int logicalLength, int increment)
{
    if (logicalLength <= 0 || increment == 0) {
        return 0;
    }
    const uint64_t stride =
        (increment < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(increment)) : static_cast<uint64_t>(increment);
    return 1U + static_cast<uint64_t>(logicalLength - 1) * stride;
}

inline bool ComplexVectorBytes(int logicalLength, int increment, size_t& bytes)
{
    const uint64_t count = PhysicalComplexCount(logicalLength, increment);
    if (count > std::numeric_limits<size_t>::max() / sizeof(aclblasComplex)) {
        return false;
    }
    bytes = static_cast<size_t>(count) * sizeof(aclblasComplex);
    return true;
}

// Pass-through combinations never dereference x/y/A on the host side of the
// operator: handle null, empty dimensions, invalid strides or lda, missing
// operands, and the alpha quick return.
inline bool MustPassThrough(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, const aclblasComplex* A, int lda)
{
    if (handle == nullptr || m <= 0 || n <= 0 || alpha == nullptr || incx == 0 || incy == 0 || lda < std::max(1, m)) {
        return true;
    }
    if (alpha->real == 0.0f && alpha->imag == 0.0f) {
        return true;
    }
    return x == nullptr || y == nullptr || A == nullptr;
}

// Device buffers for one call; released by ReleaseOperands on every exit path.
struct DeviceOperands {
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;
};

inline void ReleaseOperands(DeviceOperands& operands)
{
    if (operands.x != nullptr) {
        aclrtFree(operands.x);
        operands.x = nullptr;
    }
    if (operands.y != nullptr) {
        aclrtFree(operands.y);
        operands.y = nullptr;
    }
    if (operands.a != nullptr) {
        aclrtFree(operands.a);
        operands.a = nullptr;
    }
}

// Mirrors one host-side operand into the device buffer `host -> device`.
inline aclError UploadOperand(const void* host, size_t bytes, void*& device)
{
    aclError ret = aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(device);
        device = nullptr;
    }
    return ret;
}

// Uploads x, y and A; leaves the device pointers in `operands` on success.
inline aclError UploadOperands(
    const aclblasComplex* x, size_t xBytes, const aclblasComplex* y, size_t yBytes, const aclblasComplex* A,
    size_t aBytes, DeviceOperands& operands)
{
    aclError ret = UploadOperand(x, xBytes, operands.x);
    if (ret == ACL_SUCCESS) {
        ret = UploadOperand(y, yBytes, operands.y);
    }
    if (ret == ACL_SUCCESS) {
        ret = UploadOperand(A, aBytes, operands.a);
    }
    return ret;
}

inline aclblasStatus_t aclblasCgeru_npu(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (MustPassThrough(handle, m, n, alpha, x, incx, y, incy, A, lda)) {
        return aclblasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda);
    }

    size_t xBytes = 0;
    size_t yBytes = 0;
    size_t aBytes = 0;
    if (!ComplexVectorBytes(m, incx, xBytes) || !ComplexVectorBytes(n, incy, yBytes) ||
        n > 0 && (static_cast<uint64_t>(lda) * static_cast<uint64_t>(n) >
                  std::numeric_limits<size_t>::max() / sizeof(aclblasComplex))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    aBytes = static_cast<size_t>(static_cast<uint64_t>(lda) * static_cast<uint64_t>(n)) * sizeof(aclblasComplex);

    DeviceOperands operands;
    if (UploadOperands(x, xBytes, y, yBytes, A, aBytes, operands) != ACL_SUCCESS) {
        ReleaseOperands(operands);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t status = aclblasCgeru(
        handle, m, n, alpha, static_cast<const aclblasComplex*>(operands.x), incx,
        static_cast<const aclblasComplex*>(operands.y), incy, static_cast<aclblasComplex*>(operands.a), lda);
    aclError runtimeRet = ACL_SUCCESS;
    if (status == ACLBLAS_STATUS_SUCCESS) {
        runtimeRet = aclrtSynchronizeDevice();
        if (runtimeRet == ACL_SUCCESS) {
            runtimeRet = aclrtMemcpy(A, aBytes, operands.a, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    ReleaseOperands(operands);
    if (runtimeRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return status;
}

} // namespace cgeru_arch22_test
