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

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Shared scaffolding for the host-scalar alpha/beta regression helpers
// (ssyrk/ssyr2k/ssyrkx/cherk). The npu wrappers upload scalars to device, so
// those tests call the public API directly with host-side alpha/beta
// addresses; these utilities hold the device buffer/upload/download/verify
// plumbing so each operator only supplies its API call and golden check.
namespace host_scalar {

// Fills the deterministic pattern shared by the host-scalar helpers:
// aHost = 0.5 + i, bHost = 1 + (i % 17), cInit = 1 + i.
inline void FillHostScalarPattern(std::vector<float>& aHost, std::vector<float>& cInit)
{
    for (size_t i = 0; i < aHost.size(); i++) {
        aHost[i] = 0.5f + static_cast<float>(i);
    }
    for (size_t i = 0; i < cInit.size(); i++) {
        cInit[i] = 1.0f + static_cast<float>(i);
    }
}

inline void FillHostScalarPattern(std::vector<float>& aHost, std::vector<float>& bHost, std::vector<float>& cInit)
{
    FillHostScalarPattern(aHost, cInit);
    for (size_t i = 0; i < bHost.size(); i++) {
        bHost[i] = 1.0f + static_cast<float>(i % 17);
    }
}

// Runs one host-scalar shape: allocates the device A/B/C buffers and uploads
// aHost/bHost/cInit (bBytes == 0 skips the B buffer), invokes `call` with the
// device pointers (the lambda invokes the public API with host-side alpha/beta
// addresses), synchronizes, downloads C and hands it to `verify`. Returns
// false on any device API failure; EXPECT failures are reported by the
// callbacks. Device buffers are released on every exit path.
template <typename T, typename Call, typename Verify>
inline bool RunOnDevice(
    aclrtStream stream, size_t aBytes, size_t bBytes, size_t cBytes, const T* aHost, const T* bHost, const T* cInit,
    Call call, Verify verify)
{
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    bool ok = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              (bBytes == 0 || aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
    if (ok) {
        ok = aclrtMemcpy(dA, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
             aclrtMemcpy(dC, cBytes, cInit, cBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
             (bBytes == 0 || aclrtMemcpy(dB, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
    }
    if (ok) {
        const aclblasStatus_t ret = call(dA, (bBytes == 0 ? nullptr : dB), dC);
        EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
        ok = ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    }
    if (ok) {
        std::vector<T> cOut(cBytes / sizeof(T));
        ok = aclrtMemcpy(cOut.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        EXPECT_TRUE(ok);
        if (ok) {
            verify(cOut);
        }
    }
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dB != nullptr) {
        aclrtFree(dB);
    }
    if (dC != nullptr) {
        aclrtFree(dC);
    }
    return ok;
}

// NEAR check of the uplo triangle (i <= j for UPPER) against golden.
inline void VerifyUploNear(int n, int ldc, const std::vector<float>& cOut, const std::vector<float>& golden)
{
    constexpr float eps = 1e-4f;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            EXPECT_NEAR(cOut[idx], golden[idx], eps * (1.0f + std::abs(golden[idx])));
        }
    }
}

// NEAR check of the uplo triangle for complex outputs, componentwise.
inline void VerifyUploNearComplex(
    int n, int ldc, const std::vector<aclblasComplex>& cOut, const std::vector<aclblasComplex>& golden)
{
    constexpr float eps = 1e-4f;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            EXPECT_NEAR(cOut[idx].real, golden[idx].real, eps * (1.0f + std::abs(golden[idx].real)));
            EXPECT_NEAR(cOut[idx].imag, golden[idx].imag, eps * (1.0f + std::abs(golden[idx].imag)));
        }
    }
}

} // namespace host_scalar
