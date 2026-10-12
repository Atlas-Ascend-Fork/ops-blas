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
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Test-side placement options of one aclblasCgbmv call.
struct CgbmvNpuOptions {
    bool alphaOnDevice = false; // put alpha in device memory (the operator accepts Host or Device scalars)
    bool betaOnDevice = false;
    int aAlignOffset = 0;       // complex elements between the device allocation start and A / x / y; the prefix is
    int xAlignOffset = 0;       // filled with 0xFF bytes (NaN), so reading it is detectable
    int yAlignOffset = 0;
    bool yPrefixIntact = true;  // output: the y prefix is still all 0xFF (the operator did not write before y)
};

namespace cgbmv_npu {

constexpr int PREFIX_BYTE = 0xFF;

inline size_t StridedBytes(int count, int inc)
{
    size_t absInc = static_cast<size_t>(inc == 0 ? 1 : std::abs(inc));
    size_t elems = (count > 0) ? (static_cast<size_t>(count) - 1) * absInc + 1 : 1;
    return elems * sizeof(aclblasComplex);
}

inline size_t OffsetBytes(int elems) { return static_cast<size_t>(elems > 0 ? elems : 0) * sizeof(aclblasComplex); }

struct DeviceBuffers {
    void* a = nullptr;
    void* x = nullptr;
    void* y = nullptr;
    void* alpha = nullptr;
    void* beta = nullptr;

    ~DeviceBuffers()
    {
        for (void* p : {a, x, y, alpha, beta}) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
    }
};

// Allocates prefix + bytes, fills the prefix with PREFIX_BYTE and copies the host data behind it.
// A nullptr host buffer stays nullptr on the device.
inline aclblasStatus_t Upload(void*& base, const void* host, size_t bytes, size_t prefix)
{
    if (host == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(&base, prefix + bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (prefix > 0 && aclrtMemset(base, prefix + bytes, PREFIX_BYTE, prefix) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(static_cast<uint8_t*>(base) + prefix, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline T* At(void* base, size_t prefix)
{
    return (base == nullptr) ? nullptr : reinterpret_cast<T*>(static_cast<uint8_t*>(base) + prefix);
}

inline bool PrefixIntact(void* base, size_t prefix)
{
    if (base == nullptr || prefix == 0) {
        return true;
    }
    std::vector<uint8_t> host(prefix, 0);
    if (aclrtMemcpy(host.data(), prefix, base, prefix, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    for (uint8_t v : host) {
        if (v != static_cast<uint8_t>(PREFIX_BYTE)) {
            return false;
        }
    }
    return true;
}

struct Sizes {
    size_t a = 0;
    size_t x = 0;
    size_t y = 0;
};

inline aclblasStatus_t UploadAll(
    DeviceBuffers& dev, const Sizes& sz, const aclblasComplex* a, const aclblasComplex* x, const aclblasComplex* y,
    const aclblasComplex* alpha, const aclblasComplex* beta, const CgbmvNpuOptions& o)
{
    aclblasStatus_t st = Upload(dev.a, a, sz.a, OffsetBytes(o.aAlignOffset));
    if (st == ACLBLAS_STATUS_SUCCESS) {
        st = Upload(dev.x, x, sz.x, OffsetBytes(o.xAlignOffset));
    }
    if (st == ACLBLAS_STATUS_SUCCESS) {
        st = Upload(dev.y, y, sz.y, OffsetBytes(o.yAlignOffset));
    }
    if (st == ACLBLAS_STATUS_SUCCESS && o.alphaOnDevice) {
        st = Upload(dev.alpha, alpha, sizeof(aclblasComplex), 0);
    }
    if (st == ACLBLAS_STATUS_SUCCESS && o.betaOnDevice) {
        st = Upload(dev.beta, beta, sizeof(aclblasComplex), 0);
    }
    return st;
}

// Waits for the handle stream, copies y back and records whether the alignment prefix before y is untouched.
inline aclblasStatus_t DownloadY(
    aclblasHandle_t handle, const DeviceBuffers& dev, const Sizes& sz, aclblasComplex* y, CgbmvNpuOptions& o)
{
    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const size_t yPrefix = OffsetBytes(o.yAlignOffset);
    if (y != nullptr &&
        aclrtMemcpy(y, sz.y, At<void>(dev.y, yPrefix), sz.y, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    o.yPrefixIntact = PrefixIntact(dev.y, yPrefix);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace cgbmv_npu

// Host-buffer wrapper: copies A/x/y (and optionally alpha/beta) to the device, runs aclblasCgbmv on the handle
// stream, then copies y back. A holds lda * n complex elements, x/y hold 1 + (len - 1) * |inc| elements.
inline aclblasStatus_t aclblasCgbmv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy, CgbmvNpuOptions* options = nullptr)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasCgbmv(handle, trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
    }
    CgbmvNpuOptions defaults;
    CgbmvNpuOptions& o = (options != nullptr) ? *options : defaults;
    const bool isTransN = (trans == ACLBLAS_OP_N);
    const size_t aBytes = static_cast<size_t>(lda > 0 ? lda : 1) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const cgbmv_npu::Sizes sz{
        aBytes, cgbmv_npu::StridedBytes(isTransN ? n : m, incx), cgbmv_npu::StridedBytes(isTransN ? m : n, incy)};
    cgbmv_npu::DeviceBuffers dev;
    aclblasStatus_t st = cgbmv_npu::UploadAll(dev, sz, a, x, y, alpha, beta, o);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    const size_t yPrefix = cgbmv_npu::OffsetBytes(o.yAlignOffset);
    aclblasStatus_t ret = aclblasCgbmv(
        handle, trans, m, n, kl, ku, o.alphaOnDevice ? static_cast<const aclblasComplex*>(dev.alpha) : alpha,
        cgbmv_npu::At<const aclblasComplex>(dev.a, cgbmv_npu::OffsetBytes(o.aAlignOffset)), lda,
        cgbmv_npu::At<const aclblasComplex>(dev.x, cgbmv_npu::OffsetBytes(o.xAlignOffset)), incx,
        o.betaOnDevice ? static_cast<const aclblasComplex*>(dev.beta) : beta,
        cgbmv_npu::At<aclblasComplex>(dev.y, yPrefix), incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    return cgbmv_npu::DownloadY(handle, dev, sz, y, o);
}
