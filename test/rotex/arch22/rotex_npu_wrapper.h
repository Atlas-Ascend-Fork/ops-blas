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
#include <vector>

#include <securec.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "rotex_golden.h"

// ─────────────────────────────────────────────────────────────────────────────
// RAII guard: automatically frees device memory on scope exit
// ─────────────────────────────────────────────────────────────────────────────
struct RotexNpuWrapper {
    void* ptrs[4] = {};
    int count = 0;

    ~RotexNpuWrapper()
    {
        for (int i = 0; i < count; i++) {
            if (ptrs[i] != nullptr) {
                aclrtFree(ptrs[i]);
            }
        }
    }

    void Track(void* p)
    {
        if (p != nullptr && count < 4) {
            ptrs[count++] = p;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pack: float vector -> dtype bytes for device transfer
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<uint8_t> rotExPackVector(const float* src, int n, int inc, int32_t dtype)
{
    if (src == nullptr || n <= 0) return {};

    int absInc = std::abs(inc);
    int elSize = static_cast<int>(rotExTypeSize(dtype));
    size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
    size_t bytes = totalEl * static_cast<size_t>(elSize);
    std::vector<uint8_t> dst(bytes, 0);

    for (int i = 0; i < n; i++) {
        int srcIdx = (inc > 0) ? (i * inc) : ((n - 1 - i) * absInc);
        float v = src[srcIdx];
        uint8_t* p = &dst[static_cast<size_t>(srcIdx) * elSize];
        size_t dstRemain = bytes - static_cast<size_t>(srcIdx) * static_cast<size_t>(elSize);
        if (dtype == static_cast<int32_t>(ACL_FLOAT16)) {
            uint16_t h = rotExFloatToFp16(v);
            if (memcpy_s(p, dstRemain, &h, sizeof(uint16_t)) != EOK) return {};
        } else if (dtype == static_cast<int32_t>(ACL_BF16)) {
            uint16_t h = rotExFloatToBf16(v);
            if (memcpy_s(p, dstRemain, &h, sizeof(uint16_t)) != EOK) return {};
        } else {
            if (memcpy_s(p, dstRemain, &v, sizeof(float)) != EOK) return {};
        }
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unpack: dtype bytes -> float vector
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<float> rotExUnpackVector(const uint8_t* src, int n, int inc, int32_t dtype)
{
    if (src == nullptr || n <= 0) return {};

    int absInc = std::abs(inc);
    int elSize = static_cast<int>(rotExTypeSize(dtype));
    size_t totalEl = static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
    std::vector<float> dst(totalEl, 0.0f);

    for (int i = 0; i < n; i++) {
        int idx = (inc > 0) ? (i * inc) : ((n - 1 - i) * absInc);
        const uint8_t* p = &src[static_cast<size_t>(idx) * elSize];
        if (dtype == static_cast<int32_t>(ACL_FLOAT16)) {
            uint16_t h;
            if (memcpy_s(&h, sizeof(h), p, sizeof(uint16_t)) != EOK) return {};
            dst[idx] = rotExFp16ToFloat(h);
        } else if (dtype == static_cast<int32_t>(ACL_BF16)) {
            uint16_t h;
            if (memcpy_s(&h, sizeof(h), p, sizeof(uint16_t)) != EOK) return {};
            dst[idx] = rotExBf16ToFloat(h);
        } else {
            float v;
            if (memcpy_s(&v, sizeof(v), p, sizeof(float)) != EOK) return {};
            dst[idx] = v;
        }
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute host buffer byte size (used for direct copy of multi-float types)
// ─────────────────────────────────────────────────────────────────────────────
inline size_t rotExHostBufferBytes(int n, int inc, int32_t dtype)
{
    int absInc = std::abs(inc);
    int elFloats = rotExElemFloats(dtype);
    size_t totalFloats = static_cast<size_t>((n - 1) * absInc * elFloats + elFloats);
    return totalFloats * sizeof(float);
}

// ─────────────────────────────────────────────────────────────────────────────
// RotExAllocCopyH2D: allocate device memory and H2D copy (with optional pack)
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t RotExAllocCopyH2D(
    const void* src, size_t bytes, RotexNpuWrapper& guard, void*& dPtr,
    bool needPack, int n, int inc, int32_t dtype)
{
    if (src == nullptr || bytes == 0) {
        dPtr = nullptr;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    guard.Track(dPtr);
    if (needPack) {
        auto packed = rotExPackVector(static_cast<const float*>(src), n, inc, dtype);
        aclRet = aclrtMemcpy(dPtr, bytes, packed.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        aclRet = aclrtMemcpy(dPtr, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return (aclRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// ─────────────────────────────────────────────────────────────────────────────
// RotExCopyBackD2H: D2H copy (with optional unpack), used for output buffers
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t RotExCopyBackD2H(
    void* dst, size_t bytes, void* dSrc,
    bool needPack, int n, int inc, int32_t dtype)
{
    if (dst == nullptr || dSrc == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet;
    if (needPack) {
        std::vector<uint8_t> outBuf(bytes, 0);
        aclRet = aclrtMemcpy(outBuf.data(), bytes, dSrc, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        auto unpacked = rotExUnpackVector(outBuf.data(), n, inc, dtype);
        if (memcpy_s(dst, unpacked.size() * sizeof(float),
                     unpacked.data(), unpacked.size() * sizeof(float)) != EOK) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    } else {
        aclRet = aclrtMemcpy(dst, bytes, dSrc, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// NPU wrapper for aclblasRotEx
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t aclblasRotEx_npu(
    aclblasHandle_t handle,
    int n,
    void *x,
    aclDataType xType,
    int incx,
    void *y,
    aclDataType yType,
    int incy,
    const void *c,
    const void *s,
    aclDataType csType,
    aclDataType executionType)
{
    // Fast path: pass through for null handle or n <= 0
    if (handle == nullptr || n <= 0 || incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN) {
        return aclblasRotEx(handle, n, x, xType, incx, y, yType, incy, c, s, csType, executionType);
    }

    auto xDtype = static_cast<int32_t>(xType);
    auto yDtype = static_cast<int32_t>(yType);
    auto csDtype = static_cast<int32_t>(csType);

    bool packX = (xDtype == static_cast<int32_t>(ACL_FLOAT16) || xDtype == static_cast<int32_t>(ACL_BF16));
    bool packY = (yDtype == static_cast<int32_t>(ACL_FLOAT16) || yDtype == static_cast<int32_t>(ACL_BF16));

    int absIncX = std::abs(incx);
    int absIncY = std::abs(incy);

    size_t xElSize = rotExTypeSize(xDtype);
    size_t yElSize = rotExTypeSize(yDtype);
    size_t xBytes = static_cast<size_t>((n - 1) * absIncX + 1) * xElSize;
    size_t yBytes = static_cast<size_t>((n - 1) * absIncY + 1) * yElSize;
    size_t csBytes = rotExTypeSize(csDtype);

    RotexNpuWrapper guard;
    void* dX = nullptr;
    void* dY = nullptr;
    void* dC = nullptr;
    void* dS = nullptr;

    // ── Allocate + H2D for all inputs ──
    aclblasStatus_t st;
    st = RotExAllocCopyH2D(x, xBytes, guard, dX, packX, n, incx, xDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = RotExAllocCopyH2D(y, yBytes, guard, dY, packY, n, incy, yDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = RotExAllocCopyH2D(c, csBytes, guard, dC, false, 0, 1, csDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    st = RotExAllocCopyH2D(s, csBytes, guard, dS, false, 0, 1, csDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    // ── Call the operator ──
    aclblasStatus_t ret = aclblasRotEx(
        handle, n, dX, xType, incx, dY, yType, incy,
        dC, dS, csType, executionType);
    if (ret != ACLBLAS_STATUS_SUCCESS) return ret;

    // ── Synchronise ──
    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_EXECUTION_FAILED;

    // ── D2H for y (output, modified by the kernel) ──
    st = RotExCopyBackD2H(y, yBytes, dY, packY, n, incy, yDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    // ── D2H for x (output, modified by the kernel) ──
    st = RotExCopyBackD2H(x, xBytes, dX, packX, n, incx, xDtype);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    return ret;
}
