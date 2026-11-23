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
#include "dotex_golden.h"
#include "dtype_utils.h"

inline std::vector<uint8_t> dotexPackVector(const float* src, int n, int inc, aclDataType dtype)
{
    if (src == nullptr || n <= 0 || inc == 0 || inc == INT_MIN) {
        return {};
    }
    const int absInc = std::abs(inc);
    const size_t totalEl = static_cast<size_t>(static_cast<int64_t>(n - 1) * absInc) + 1U;
    const size_t elemSz = dtypeByteSize(dtype);
    const size_t safeElemSz = (elemSz == 0U) ? 1U : elemSz;
    std::vector<uint8_t> dst(totalEl * safeElemSz, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t idx = (inc > 0) ? (i * inc) : ((n - 1 - i) * absInc);
        const float v = src[idx];
        if (dtype == ACL_FLOAT16) {
            *reinterpret_cast<uint16_t*>(&dst[idx * safeElemSz]) = floatToFp16(v);
        } else if (dtype == ACL_BF16) {
            *reinterpret_cast<uint16_t*>(&dst[idx * safeElemSz]) = floatToBf16(v);
        } else if (elemSz == sizeof(float)) {
            *reinterpret_cast<float*>(&dst[idx * safeElemSz]) = v;
        }
    }
    return dst;
}

struct DotexNpuWrapper {
    void* dX{nullptr};
    void* dY{nullptr};
    void* dResult{nullptr};
    ~DotexNpuWrapper()
    {
        if (dResult != nullptr) {
            (void)aclrtFree(dResult);
        }
        if (dY != nullptr) {
            (void)aclrtFree(dY);
        }
        if (dX != nullptr) {
            (void)aclrtFree(dX);
        }
    }
};

inline aclblasStatus_t dotexCopyToDevice(void*& dPtr, const std::vector<uint8_t>& packed, size_t bytes)
{
    if (packed.empty() || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dPtr, bytes, packed.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        (void)aclrtFree(dPtr);
        dPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasDotEx_npu(
    aclblasHandle_t handle, int n, const float* x, aclDataType xType, int incx, const float* y, aclDataType yType,
    int incy, float* result, aclDataType resultType, aclDataType executionType)
{
    if (handle == nullptr || n <= 0 || incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        return aclblasDotEx(
            handle, n, static_cast<const void*>(x), xType, incx, static_cast<const void*>(y), yType, incy,
            static_cast<void*>(result), resultType, executionType);
    }

    DotexNpuWrapper mem;
    const std::vector<uint8_t> packedX = dotexPackVector(x, n, incx, xType);
    aclblasStatus_t st = dotexCopyToDevice(mem.dX, packedX, packedX.size());
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const std::vector<uint8_t> packedY = dotexPackVector(y, n, incy, yType);
    st = dotexCopyToDevice(mem.dY, packedY, packedY.size());
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const size_t resultBytes = dtypeByteSize(resultType);
    if (aclrtMalloc(&mem.dResult, resultBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasStatus_t ret = aclblasDotEx(
        handle, n, static_cast<const void*>(mem.dX), xType, incx, static_cast<const void*>(mem.dY), yType, incy,
        static_cast<void*>(mem.dResult), resultType, executionType);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (result != nullptr) {
        if (resultType == ACL_FLOAT) {
            (void)aclrtMemcpy(result, sizeof(float), mem.dResult, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        } else if (resultType == ACL_FLOAT16) {
            uint16_t halfResult = 0;
            (void)aclrtMemcpy(&halfResult, sizeof(uint16_t), mem.dResult, sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            *result = fp16ToFloat(halfResult);
        } else if (resultType == ACL_BF16) {
            uint16_t bf16Result = 0;
            (void)aclrtMemcpy(&bf16Result, sizeof(uint16_t), mem.dResult, sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
            *result = bf16ToFloat(bf16Result);
        }
    }
    return ret;
}
