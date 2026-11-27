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
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "scalex_golden.h"

inline size_t scalexDtypeSize(int32_t dtype)
{
    if (dtype == static_cast<int32_t>(ACL_FLOAT16) || dtype == static_cast<int32_t>(ACL_BF16)) return 2;
    return 4;
}

inline std::vector<uint8_t> scalexPackX(const float* src, int n, int incx, int32_t dtype)
{
    if (src == nullptr || n <= 0) return {};

    int absIncx  = std::abs(incx);
    size_t totalEl  = static_cast<size_t>(n - 1) * static_cast<size_t>(absIncx) + 1;
    size_t elemSz = scalexDtypeSize(dtype);
    std::vector<uint8_t> dst(totalEl * elemSz, 0);

    for (int i = 0; i < n; i++) {
        int idx = (incx > 0) ? (i * incx) : ((n - 1 - i) * absIncx);
        float v = src[idx];
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

inline std::vector<float> scalexUnpackX(const uint8_t* src, int n, int incx, int32_t dtype)
{
    if (src == nullptr || n <= 0) return {};

    int absIncx  = std::abs(incx);
    size_t totalEl  = static_cast<size_t>(n - 1) * static_cast<size_t>(absIncx) + 1;
    size_t elemSz = scalexDtypeSize(dtype);
    std::vector<float> dst(totalEl, 0.0f);

    for (int i = 0; i < n; i++) {
        int idx = (incx > 0) ? (i * incx) : ((n - 1 - i) * absIncx);
        if (dtype == static_cast<int32_t>(ACL_FLOAT16)) {
            uint16_t h = *reinterpret_cast<const uint16_t*>(&src[idx * elemSz]);
            dst[idx] = fp16ToFloat(h);
        } else if (dtype == static_cast<int32_t>(ACL_BF16)) {
            uint16_t h = *reinterpret_cast<const uint16_t*>(&src[idx * elemSz]);
            dst[idx] = bf16ToFloat(h);
        } else {
            dst[idx] = *reinterpret_cast<const float*>(&src[idx * elemSz]);
        }
    }
    return dst;
}

struct ScalexNpuMem {
    void* dX = nullptr;
    void* dAlpha = nullptr;
    ~ScalexNpuMem() {
        if (dAlpha) aclrtFree(dAlpha);
        if (dX) aclrtFree(dX);
    }
};

inline aclblasStatus_t scalexNpuCopyXToDevice(
    const float* x, int n, int incx, int32_t xTypeLoc, void*& dX, size_t& xBytes)
{
    int absIncx = std::abs(incx);
    size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absIncx) + 1;
    xBytes = totalEl * scalexDtypeSize(xTypeLoc);

    if (x == nullptr || xBytes == 0) return ACLBLAS_STATUS_SUCCESS;

    aclError aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;

    std::vector<uint8_t> packed = scalexPackX(x, n, incx, xTypeLoc);
    aclRet = aclrtMemcpy(dX, xBytes, packed.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        dX = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t scalexNpuPrepareAlpha(
    const void* alpha, bool alphaOnDevice, void*& dAlpha, const void*& alphaArg)
{
    alphaArg = alpha;
    if (!alphaOnDevice || alpha == nullptr) return ACLBLAS_STATUS_SUCCESS;

    aclError aclRet = aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;

    aclRet = aclrtMemcpy(dAlpha, sizeof(float), alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAlpha);
        dAlpha = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    alphaArg = dAlpha;
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool scalexNpuCopyBackResult(
    void* dX, size_t xBytes, float* x, int n, int incx, int32_t xTypeLoc)
{
    if (x == nullptr || dX == nullptr) return true;

    std::vector<uint8_t> xOut(xBytes, 0);
    if (aclrtMemcpy(xOut.data(), xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    std::vector<float> xFloatOut = scalexUnpackX(xOut.data(), n, incx, xTypeLoc);
    for (size_t i = 0; i < xFloatOut.size(); i++) {
        x[i] = xFloatOut[i];
    }
    return true;
}

inline aclblasStatus_t aclblasScalex_npu(
    aclblasHandle_t handle,
    int n,
    const void* alpha,
    aclDataType alphaType,
    void* x,
    aclDataType xType,
    int incx,
    aclDataType executionType,
    bool alphaOnDevice = true)
{
    if (handle == nullptr || n <= 0) {
        return aclblasScalex(handle, n, alpha, alphaType, x, xType, incx, executionType);
    }

    ScalexNpuMem mem;
    auto xTypeLoc = static_cast<int32_t>(xType);

    size_t xBytes = 0;
    aclblasStatus_t st = scalexNpuCopyXToDevice(
        static_cast<const float*>(x), n, incx, xTypeLoc, mem.dX, xBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    const void* alphaArg = nullptr;
    st = scalexNpuPrepareAlpha(alpha, alphaOnDevice, mem.dAlpha, alphaArg);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasScalex(
        handle, n, alphaArg, alphaType, mem.dX, xType, incx, executionType);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS &&
        !scalexNpuCopyBackResult(mem.dX, xBytes, static_cast<float*>(x), n, incx, xTypeLoc)) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ret;
}

