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

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "dtype_compat.h"

inline std::vector<uint8_t> axpyExPackVector(const float* src, int n, int inc, int32_t dtype)
{
    if (src == nullptr || n <= 0 || inc == 0 || inc == INT_MIN) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1U;
    const size_t elemSz = dtypeCompatSize(dtype);
    std::vector<uint8_t> dst(totalEl * elemSz, 0);
    for (int i = 0; i < n; ++i) {
        const int idx = (inc > 0) ? (i * inc) : ((n - 1 - i) * absInc);
        const float v = src[idx];
        if (dtype == static_cast<int32_t>(ACL_FLOAT16)) {
            *reinterpret_cast<uint16_t*>(&dst[idx * elemSz]) = floatToFp16(v);
        } else if (dtype == static_cast<int32_t>(ACL_BF16)) {
            *reinterpret_cast<uint16_t*>(&dst[idx * elemSz]) = floatToBf16(v);
        } else {
            *reinterpret_cast<float*>(&dst[idx * elemSz]) = v;
        }
    }
    return dst;
}

inline std::vector<float> axpyExUnpackVector(const uint8_t* src, int n, int inc, int32_t dtype)
{
    if (src == nullptr || n <= 0 || inc == 0 || inc == INT_MIN) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1U;
    const size_t elemSz = dtypeCompatSize(dtype);
    std::vector<float> dst(totalEl, 0.0f);
    for (int i = 0; i < n; ++i) {
        const int idx = (inc > 0) ? (i * inc) : ((n - 1 - i) * absInc);
        if (dtype == static_cast<int32_t>(ACL_FLOAT16)) {
            const uint16_t h = *reinterpret_cast<const uint16_t*>(&src[idx * elemSz]);
            dst[idx] = fp16ToFloat(h);
        } else if (dtype == static_cast<int32_t>(ACL_BF16)) {
            const uint16_t h = *reinterpret_cast<const uint16_t*>(&src[idx * elemSz]);
            dst[idx] = bf16ToFloat(h);
        } else {
            dst[idx] = *reinterpret_cast<const float*>(&src[idx * elemSz]);
        }
    }
    return dst;
}

struct AxpyExNpuWrapper {
    void* dX{nullptr};
    void* dY{nullptr};
    void* dAlpha{nullptr};
    ~AxpyExNpuWrapper()
    {
        if (dAlpha != nullptr) {
            (void)aclrtFree(dAlpha);
        }
        if (dX != nullptr) {
            (void)aclrtFree(dX);
        }
        if (dY != nullptr) {
            (void)aclrtFree(dY);
        }
    }
};

inline aclblasStatus_t axpyExCopyVectorToDevice(
    const float* src, int n, int inc, int32_t dtype, void*& dOut, size_t& outBytes)
{
    dOut = nullptr;
    outBytes = 0;
    if (src == nullptr || n <= 0 || inc == 0 || inc == INT_MIN) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const int absInc = std::abs(inc);
    const size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1U;
    outBytes = totalEl * dtypeCompatSize(dtype);
    aclError aclRet = aclrtMalloc(&dOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        outBytes = 0;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const std::vector<uint8_t> packed = axpyExPackVector(src, n, inc, dtype);
    aclRet = aclrtMemcpy(dOut, outBytes, packed.data(), outBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        (void)aclrtFree(dOut);
        dOut = nullptr;
        outBytes = 0;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t axpyExCopyVectorBackFromDevice(
    void* dSrc, size_t srcBytes, float* dst, int n, int inc, int32_t dtype)
{
    if (dst == nullptr || dSrc == nullptr || n <= 0 || srcBytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    std::vector<uint8_t> raw(srcBytes, 0);
    if (aclrtMemcpy(raw.data(), srcBytes, dSrc, srcBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const std::vector<float> unpacked = axpyExUnpackVector(raw.data(), n, inc, dtype);
    for (size_t i = 0; i < unpacked.size(); ++i) {
        dst[i] = unpacked[i];
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t axpyExPrepareAlpha(const void* alpha, bool alphaOnDevice, void*& dAlpha, const void*& alphaArg)
{
    alphaArg = alpha;
    dAlpha = nullptr;
    if (alpha == nullptr || !alphaOnDevice) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dAlpha, sizeof(float), alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(dAlpha);
        dAlpha = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    alphaArg = dAlpha;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasAxpyEx_npu(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType,
    int incx, void* y, aclDataType yType, int incy, aclDataType executionType, bool alphaOnDevice = true)
{
    if (handle == nullptr || n <= 0 || incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        return aclblasAxpyEx(handle, n, alpha, alphaType, x, xType, incx, y, yType, incy, executionType);
    }
    AxpyExNpuWrapper mem;
    const auto xTypeLoc = static_cast<int32_t>(xType);
    const auto yTypeLoc = static_cast<int32_t>(yType);
    size_t xBytes = 0;
    aclblasStatus_t st = axpyExCopyVectorToDevice(static_cast<const float*>(x), n, incx, xTypeLoc, mem.dX, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    size_t yBytes = 0;
    st = axpyExCopyVectorToDevice(static_cast<const float*>(y), n, incy, yTypeLoc, mem.dY, yBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const void* alphaArg = nullptr;
    st = axpyExPrepareAlpha(alpha, alphaOnDevice, mem.dAlpha, alphaArg);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const aclblasStatus_t ret =
        aclblasAxpyEx(handle, n, alphaArg, alphaType, mem.dX, xType, incx, mem.dY, yType, incy, executionType);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return axpyExCopyVectorBackFromDevice(mem.dY, yBytes, static_cast<float*>(y), n, incy, yTypeLoc);
}
