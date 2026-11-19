/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CGERU_ARCH22_NPU_WRAPPER_H
#define CGERU_ARCH22_NPU_WRAPPER_H

#include <algorithm>
#include <climits>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Host-side convenience wrapper for the arch22 (Atlas 800I A3 / ascend910b) test.
// The public API takes device pointers; this wrapper performs the H2D/D2H copies
// so the GTest code can keep a plain host reference implementation.

// Validation is delegated to aclblasCgeru itself: the wrapper never returns an
// expected error code on its own, otherwise the API's own validation cases would
// become tautological. When the arguments are not usable for a real launch (null
// pointer, illegal dimension/stride/lda), the wrapper issues a "validation call"
// with null device pointers; the library rejects or quick-returns those inputs
// before it would dereference any pointer.
static inline aclError CgeruAllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr || bytes == 0)
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

// |inc| == INT_MIN 时物理跨度 2^31 只有 len <= 1 才可寻址（与 host 侧
// CgeruStrideSpanValid 保持同一语义）。
static inline bool CgeruStrideSpanValid(int length, int inc)
{
    if (inc == 0) {
        return false;
    }
    if (inc != INT_MIN) {
        return true;
    }
    return length <= 1;
}

// 物理长度 1 + (len - 1) * |inc|；INT_MIN 且 len <= 1 时只有一个元素，按 1 计算，
// 避免按 2^31 分配缓冲区。
static inline int64_t CgeruStrideSpanLength(int length, int inc)
{
    if (length <= 0) {
        return 0;
    }
    if (inc == INT_MIN) {
        return length <= 1 ? 1 : 0;
    }
    const int64_t stride = (inc < 0) ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    return 1 + static_cast<int64_t>(length - 1) * stride;
}

static inline bool CgeruNeedsOnePassValidation(aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
                                               const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
                                               const aclblasComplex* A, int lda)
{
    return handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || A == nullptr || m <= 0 ||
           n <= 0 || !CgeruStrideSpanValid(m, incx) || !CgeruStrideSpanValid(n, incy) || lda < std::max(1, m);
}

static inline void CgeruFreeDevice(void*& dX, void*& dY, void*& dA)
{
    if (dA)
        aclrtFree(dA);
    if (dY)
        aclrtFree(dY);
    if (dX)
        aclrtFree(dX);
    dA = nullptr;
    dY = nullptr;
    dX = nullptr;
}

// 分配并上传 x/y/A；返回非 SUCCESS 时三块已分配缓冲都已释放。
static inline aclError CgeruUploadInputs(const aclblasComplex* x, int m, int incx, const aclblasComplex* y, int n,
                                         int incy, const aclblasComplex* A, int lda, void*& dX, void*& dY,
                                         void*& dA)
{
    // Physical length of a strided vector is 1 + (len - 1) * |inc|.
    const size_t xBytes =
        static_cast<size_t>(CgeruStrideSpanLength(m, incx)) * sizeof(aclblasComplex);
    const size_t yBytes =
        static_cast<size_t>(CgeruStrideSpanLength(n, incy)) * sizeof(aclblasComplex);
    // Column-major A: lda rows * n columns.
    const size_t aBytes = static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
    aclError aclRet = CgeruAllocCopyH2D(dX, x, xBytes);
    if (aclRet != ACL_SUCCESS)
        return aclRet;
    aclRet = CgeruAllocCopyH2D(dY, y, yBytes);
    if (aclRet != ACL_SUCCESS) {
        CgeruFreeDevice(dX, dY, dA);
        return aclRet;
    }
    aclRet = CgeruAllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) {
        CgeruFreeDevice(dX, dY, dA);
        return aclRet;
    }
    return ACL_SUCCESS;
}

// 下发、同步并回拷 A；失败时返回 INTERNAL_ERROR。
static inline aclblasStatus_t CgeruRunAndCopyBack(aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
                                                  int incx, int incy, int lda, aclblasComplex* A, void* dX,
                                                  void* dY, void* dA)
{
    aclblasStatus_t ret = aclblasCgeru(
        handle, m, n, alpha, static_cast<const aclblasComplex*>(dX), incx, static_cast<const aclblasComplex*>(dY),
        incy, static_cast<aclblasComplex*>(dA), lda);
    aclError syncRet = aclrtSynchronizeDevice();
    if (ret == ACLBLAS_STATUS_SUCCESS && syncRet != ACL_SUCCESS)
        ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    if (ret == ACLBLAS_STATUS_SUCCESS && A != nullptr && dA != nullptr) {
        const size_t aBytes = static_cast<size_t>(n) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
        aclError copyRet = aclrtMemcpy(A, aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS)
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ret;
}

inline aclblasStatus_t aclblasCgeru_npu(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (CgeruNeedsOnePassValidation(handle, m, n, alpha, x, incx, y, incy, A, lda)) {
        return aclblasCgeru(handle, m, n, alpha, nullptr, incx, nullptr, incy, nullptr, lda);
    }

    void* dX = nullptr;
    void* dY = nullptr;
    void* dA = nullptr;
    if (CgeruUploadInputs(x, m, incx, y, n, incy, A, lda, dX, dY, dA) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclblasStatus_t ret = CgeruRunAndCopyBack(handle, m, n, alpha, incx, incy, lda, A, dX, dY, dA);
    CgeruFreeDevice(dX, dY, dA);
    return ret;
}

#endif // CGERU_ARCH22_NPU_WRAPPER_H
