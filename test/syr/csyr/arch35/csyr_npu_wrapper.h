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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Five complex elements on each side detect out-of-range writes while keeping
// the data 8-byte aligned. Adding one float below tests 4-byte-only alignment.
constexpr size_t CSYR_GUARD_BYTES = 40;
constexpr size_t CSYR_MISALIGNED_GUARD_BYTES = CSYR_GUARD_BYTES + sizeof(float);
// Nonzero byte sentinel, compared bit-for-bit before and after the NPU call.
constexpr uint8_t CSYR_GUARD_VALUE = 0xA5;

struct CsyrNpuOptions {
    bool alphaOnDevice = false;
    size_t xGuard = CSYR_GUARD_BYTES;
    size_t aGuard = CSYR_GUARD_BYTES;
};

inline aclError CsyrCopyToDevice(void** device, const void* host, size_t bytes)
{
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        (void)aclrtFree(*device);
        *device = nullptr;
    }
    return ret;
}

inline bool CsyrGetBufferBytes(int n, int incx, int lda, size_t& xBytes, size_t& aBytes, const CsyrNpuOptions& options)
{
    const uint64_t absIncx =
        (incx >= 0) ? static_cast<uint64_t>(incx) : static_cast<uint64_t>(-static_cast<int64_t>(incx));
    const uint64_t xElements = static_cast<uint64_t>(n - 1) * absIncx + 1u;
    const uint64_t aElements = static_cast<uint64_t>(std::max(1, lda)) * static_cast<uint64_t>(n);
    constexpr uint64_t elementBytes = sizeof(aclblasComplex);
    constexpr uint64_t maxSize = std::numeric_limits<size_t>::max();
    const uint64_t maxGuard = std::max(options.xGuard, options.aGuard);
    if (maxGuard > maxSize / 2u) {
        return false;
    }
    const uint64_t guardBytes = 2u * maxGuard;
    if (xElements > (maxSize - guardBytes) / elementBytes || aElements > (maxSize - guardBytes) / elementBytes) {
        return false;
    }
    xBytes = static_cast<size_t>(xElements * elementBytes);
    aBytes = static_cast<size_t>(aElements * elementBytes);
    return true;
}

inline aclError CsyrCopyToGuardedDevice(
    void** allocation, void** data, const void* host, size_t bytes, size_t guardBytes)
{
    const size_t allocationBytes = bytes + 2u * guardBytes;
    aclError ret = aclrtMalloc(allocation, allocationBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemset(*allocation, allocationBytes, CSYR_GUARD_VALUE, allocationBytes);
    }
    if (ret == ACL_SUCCESS) {
        *data = static_cast<uint8_t*>(*allocation) + guardBytes;
        ret = aclrtMemcpy(*data, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (ret != ACL_SUCCESS && *allocation != nullptr) {
        (void)aclrtFree(*allocation);
        *allocation = nullptr;
        *data = nullptr;
    }
    return ret;
}

inline bool CsyrGuardsIntact(const void* allocation, size_t dataBytes, size_t guardBytes)
{
    std::vector<uint8_t> prefix(guardBytes);
    std::vector<uint8_t> suffix(guardBytes);
    const auto* bytes = static_cast<const uint8_t*>(allocation);
    if (aclrtMemcpy(prefix.data(), prefix.size(), bytes, prefix.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(
            suffix.data(), suffix.size(), bytes + guardBytes + dataBytes, suffix.size(), ACL_MEMCPY_DEVICE_TO_HOST) !=
            ACL_SUCCESS) {
        return false;
    }
    const auto isGuard = [](uint8_t value) { return value == CSYR_GUARD_VALUE; };
    return std::all_of(prefix.begin(), prefix.end(), isGuard) && std::all_of(suffix.begin(), suffix.end(), isGuard);
}

inline bool CsyrInputUnchanged(const void* deviceData, const void* hostData, size_t bytes)
{
    std::vector<uint8_t> actual(bytes);
    if (aclrtMemcpy(actual.data(), bytes, deviceData, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return false;
    }
    return std::memcmp(actual.data(), hostData, bytes) == 0;
}

struct CsyrDeviceBuffers {
    void* alpha = nullptr;
    void* xAllocation = nullptr;
    void* aAllocation = nullptr;
    void* x = nullptr;
    void* a = nullptr;

    CsyrDeviceBuffers() = default;
    CsyrDeviceBuffers(const CsyrDeviceBuffers&) = delete;
    CsyrDeviceBuffers& operator=(const CsyrDeviceBuffers&) = delete;

    ~CsyrDeviceBuffers()
    {
        if (alpha != nullptr) {
            (void)aclrtFree(alpha);
        }
        if (xAllocation != nullptr) {
            (void)aclrtFree(xAllocation);
        }
        if (aAllocation != nullptr) {
            (void)aclrtFree(aAllocation);
        }
    }
};

inline bool CsyrBuffersIntact(
    const CsyrDeviceBuffers& buffers, const void* xHost, size_t xBytes, size_t aBytes, const CsyrNpuOptions& options)
{
    const bool guardsIntact =
        (buffers.xAllocation == nullptr || CsyrGuardsIntact(buffers.xAllocation, xBytes, options.xGuard)) &&
        (buffers.aAllocation == nullptr || CsyrGuardsIntact(buffers.aAllocation, aBytes, options.aGuard));
    const bool inputUnchanged = buffers.x == nullptr || CsyrInputUnchanged(buffers.x, xHost, xBytes);
    return guardsIntact && inputUnchanged;
}

inline aclblasStatus_t CsyrFinalizeNpuCall(
    aclblasStatus_t status, const CsyrDeviceBuffers& buffers, const void* xHost, size_t xBytes, void* aHost,
    size_t aBytes, const CsyrNpuOptions& options)
{
    if (aclrtSynchronizeDevice() != ACL_SUCCESS || !CsyrBuffersIntact(buffers, xHost, xBytes, aBytes, options)) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        const bool outputUnchanged = buffers.a == nullptr || CsyrInputUnchanged(buffers.a, aHost, aBytes);
        return outputUnchanged ? status : ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aHost != nullptr && buffers.a != nullptr &&
        aclrtMemcpy(aHost, aBytes, buffers.a, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return status;
}

inline aclblasStatus_t aclblasCsyr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, aclblasComplex* A, int lda, const CsyrNpuOptions& options = {})
{
    if (handle == nullptr || n <= 0) {
        return aclblasCsyr(handle, uplo, n, alpha, x, incx, A, lda);
    }

    size_t xBytes = 0;
    size_t aBytes = 0;
    if (!CsyrGetBufferBytes(n, incx, lda, xBytes, aBytes, options)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CsyrDeviceBuffers buffers;

    if (options.alphaOnDevice && alpha != nullptr &&
        CsyrCopyToDevice(&buffers.alpha, alpha, sizeof(aclblasComplex)) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (x != nullptr &&
        CsyrCopyToGuardedDevice(&buffers.xAllocation, &buffers.x, x, xBytes, options.xGuard) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (A != nullptr &&
        CsyrCopyToGuardedDevice(&buffers.aAllocation, &buffers.a, A, aBytes, options.aGuard) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const auto* alphaArg = options.alphaOnDevice ? static_cast<const aclblasComplex*>(buffers.alpha) : alpha;
    aclblasStatus_t status = aclblasCsyr(
        handle, uplo, n, alphaArg, static_cast<const aclblasComplex*>(buffers.x), incx,
        static_cast<aclblasComplex*>(buffers.a), lda);
    return CsyrFinalizeNpuCall(status, buffers, x, xBytes, A, aBytes, options);
}
