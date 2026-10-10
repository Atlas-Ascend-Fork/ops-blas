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

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cdotc_test_utils.h"

class CdotcDeviceBuffers {
public:
    CdotcDeviceBuffers() = default;
    CdotcDeviceBuffers(const CdotcDeviceBuffers&) = delete;
    CdotcDeviceBuffers& operator=(const CdotcDeviceBuffers&) = delete;
    ~CdotcDeviceBuffers() { Release(); }

    aclblasStatus_t Init(
        int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
        const aclblasComplex* initialResult)
    {
        const size_t xBytes = CdotcStorageCount(n, incx) * sizeof(aclblasComplex);
        const size_t yBytes = CdotcStorageCount(n, incy) * sizeof(aclblasComplex);
        aclblasStatus_t status = AllocateAndCopy(x, xBytes, &x_);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        status = AllocateAndCopy(y, yBytes, &y_);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            Release();
            return status;
        }
        status = AllocateAndCopy(initialResult, sizeof(aclblasComplex), &result_);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            Release();
        }
        return status;
    }

    aclblasComplex* X() const { return static_cast<aclblasComplex*>(x_); }
    aclblasComplex* Y() const { return static_cast<aclblasComplex*>(y_); }
    aclblasComplex* Result() const { return static_cast<aclblasComplex*>(result_); }

    aclblasStatus_t CopyResultToHost(aclblasComplex* result) const
    {
        if (result == nullptr || result_ == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        aclError ret =
            aclrtMemcpy(result, sizeof(aclblasComplex), result_, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST);
        return ret == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    }

private:
    void* x_ = nullptr;
    void* y_ = nullptr;
    void* result_ = nullptr;

    static aclblasStatus_t AllocateAndCopy(const void* source, size_t bytes, void** destination)
    {
        if (source == nullptr) {
            return ACLBLAS_STATUS_SUCCESS;
        }
        aclError ret = aclrtMalloc(destination, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        ret = aclrtMemcpy(*destination, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(*destination);
            *destination = nullptr;
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    void Release()
    {
        if (x_ != nullptr) {
            aclrtFree(x_);
            x_ = nullptr;
        }
        if (y_ != nullptr) {
            aclrtFree(y_);
            y_ = nullptr;
        }
        if (result_ != nullptr) {
            aclrtFree(result_);
            result_ = nullptr;
        }
    }
};

inline aclblasStatus_t aclblasCdotc_npu(
    aclblasHandle_t handle, aclrtStream stream, int n, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, aclblasComplex* result)
{
    if (handle == nullptr || n < 0) {
        return aclblasCdotc(handle, n, x, incx, y, incy, result);
    }

    CdotcDeviceBuffers device;
    aclblasStatus_t status = device.Init(n, x, incx, y, incy, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = aclblasCdotc(handle, n, device.X(), incx, device.Y(), incy, device.Result());
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (result != nullptr) {
        return device.CopyResultToHost(result);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
