/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPR2_ARCH22_NPU_WRAPPER_H
#define SSPR2_ARCH22_NPU_WRAPPER_H

// sspr2 arch22 NPU wrapper（host 测试数据 → Device → aclblasSspr2 → 读回）：
// alpha 为 Host 内存指针（直传，不下沉 Device）；x/y 只读 Device 张量；AP 原地
// 覆写 Device 张量。handle=nullptr 或 n<=0 直接前传，保留算子自身校验路径
// （n<0 → INVALID_VALUE、n=0 no-op 先于指针校验等）。x/y/AP 为 nullptr（负向）
// 时不做 Device 分配，nullptr 直达算子。

#include <cstddef>
#include <cstdio>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct Sspr2DeviceBuffers {
    void* dX = nullptr;
    void* dY = nullptr;
    void* dAP = nullptr;

    ~Sspr2DeviceBuffers()
    {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        if (dAP) aclrtFree(dAP);
    }
};

static aclblasStatus_t Sspr2AllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// hostPtr 为空或 bytes=0 时不分配，devPtr 保持 null（nullptr 负向用例直达算子）
static aclblasStatus_t Sspr2TryAllocAndCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return Sspr2AllocAndCopy(devPtr, hostPtr, bytes);
}

inline aclblasStatus_t aclblasSspr2_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha,
    const float* x, int incx, const float* y, int incy, float* ap)
{
    if (handle == nullptr || n <= 0) {
        // n<0（INVALID_VALUE 校验）/ n=0（no-op）/ handle 空（HANDLE_IS_NULLPTR）直达算子
        return aclblasSspr2(handle, uplo, n, alpha, x, incx, y, incy, ap);
    }

    const size_t xBytes = (x != nullptr && incx != 0) ? (static_cast<size_t>(n - 1) * static_cast<size_t>(incx < 0 ? -incx : incx) + 1) * sizeof(float) : 0;
    const size_t yBytes = (y != nullptr && incy != 0) ? (static_cast<size_t>(n - 1) * static_cast<size_t>(incy < 0 ? -incy : incy) + 1) * sizeof(float) : 0;
    const size_t apBytes = (ap != nullptr) ? static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2 * sizeof(float) : 0;

    Sspr2DeviceBuffers bufs;
    aclblasStatus_t st;
    st = Sspr2TryAllocAndCopy(x, xBytes, bufs.dX);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Sspr2TryAllocAndCopy(y, yBytes, bufs.dY);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = Sspr2TryAllocAndCopy(ap, apBytes, bufs.dAP);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    // alpha：Host 指针直传（不下沉 Device）；x/y/AP：有 Device 缓冲用缓冲，否则原样（nullptr 负向）
    aclblasStatus_t ret = aclblasSspr2(
        handle, uplo, n, alpha,
        bufs.dX ? static_cast<const float*>(bufs.dX) : x, incx,
        bufs.dY ? static_cast<const float*>(bufs.dY) : y, incy,
        bufs.dAP ? static_cast<float*>(bufs.dAP) : ap);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclError aclRet = aclrtSynchronizeDevice(); // 同步 Device 全部流
    if (aclRet != ACL_SUCCESS) {
        std::fprintf(stderr, "[sspr2_st] aclrtSynchronizeDevice failed, aclRet=%d\n",
            static_cast<int>(aclRet));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (ap != nullptr && bufs.dAP != nullptr) {
        aclRet = aclrtMemcpy(ap, apBytes, bufs.dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ret;
}

#endif // SSPR2_ARCH22_NPU_WRAPPER_H
