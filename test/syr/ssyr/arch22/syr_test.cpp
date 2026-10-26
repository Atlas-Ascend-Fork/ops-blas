/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

constexpr float EPSILON = 1e-3f;

static uint32_t VerifyResult(const std::vector<float>& output, const std::vector<float>& golden)
{
    constexpr size_t maxPrintSize = 10;
    std::cout << "Output: ";
    for (size_t i = 0; i < std::min(output.size(), maxPrintSize); i++) {
        std::cout << output[i] << " ";
    }
    if (output.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    std::cout << "Golden: ";
    for (size_t i = 0; i < std::min(golden.size(), maxPrintSize); i++) {
        std::cout << golden[i] << " ";
    }
    if (golden.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output[i] - golden[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Index " << i << ": output=" << output[i] << " golden=" << golden[i]
                      << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static void ComputeGoldenSsyrUpper(float* A, const float* x, float alpha, int n, int lda)
{
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

static void ComputeGoldenSsyrLower(float* A, const float* x, float alpha, int n, int lda)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            A[i * lda + j] += alpha * x[i] * x[j];
        }
    }
}

// 设备缓冲统一用 DeviceBuffer 托管：任何失败路径都不再泄漏
// aclrtMalloc 内存，也不跳过 handle/stream/上下文清理（issue #357）
static int32_t runSsyrDevice(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, int n, float alphaVal,
    const std::vector<float>& x, int incx, std::vector<float>& a, int lda)
{
    try {
        DeviceBuffer aDev(a.size() * sizeof(float));
        DeviceBuffer xDev(x.size() * sizeof(float));
        DeviceBuffer alphaDev(sizeof(float));
        aDev.copyFromHost(a.data(), aDev.size());
        xDev.copyFromHost(x.data(), xDev.size());
        alphaDev.copyFromHost(&alphaVal, sizeof(float));

        int32_t ret = aclblasSsyr(handle, uplo, n, alphaDev.floatPtr(), xDev.floatPtr(), incx, aDev.floatPtr(), lda);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            std::cout << "aclblasSsyr failed. ERROR: " << ret << std::endl;
            return ret;
        }

        aclError aclRet = aclrtSynchronizeStream(stream);
        if (aclRet != ACL_SUCCESS) {
            std::cout << "aclrtSynchronizeStream failed." << std::endl;
            return static_cast<int32_t>(aclRet);
        }
        aDev.copyToHost(a.data(), aDev.size());
        return ACLBLAS_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        std::cout << "device memory operation failed: " << e.what() << std::endl;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
}

static int TestSsyrCase(const char* label, aclblasFillMode_t uplo)
{
    std::cout << "Testing syr A2 (uplo=" << (uplo == ACLBLAS_UPPER ? "U" : "L") << "):" << std::endl;

    int32_t deviceId = 0;
    constexpr int N = 128;
    constexpr int lda = N;
    constexpr int incx = 1;
    constexpr float alphaVal = 2.0f;

    std::vector<float> a(N * lda, 1.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            bool keep = (uplo == ACLBLAS_UPPER) ? (j >= i) : (j <= i);
            a[i * lda + j] = keep ? a[i * lda + j] : 0.0f;
        }
        a[i * lda + i] = 3.0f;
    }
    std::vector<float> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i + 1);
    }
    std::vector<float> aGolden = a;
    if (uplo == ACLBLAS_UPPER) {
        ComputeGoldenSsyrUpper(aGolden.data(), x.data(), alphaVal, N, lda);
    } else {
        ComputeGoldenSsyrLower(aGolden.data(), x.data(), alphaVal, N, lda);
    }

    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    int32_t ret = 0;

    aclInit(nullptr);
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        std::cout << "aclrtSetDevice failed." << std::endl;
        aclFinalize();
        return 1;
    }
    ret = aclblasCreate(&handle);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasCreate failed. ERROR: " << ret << std::endl;
        aclFinalize();
        return ret;
    }
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "aclblasSetStream failed. ERROR: " << ret << std::endl;
        goto cleanup;
    }

    ret = runSsyrDevice(handle, stream, uplo, N, alphaVal, x, incx, a, lda);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = static_cast<int32_t>(VerifyResult(a, aGolden));
    }

cleanup:
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (handle != nullptr) {
        aclblasDestroy(handle);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}

static int TestSsyrUpper() { return TestSsyrCase("U", ACLBLAS_UPPER); }

static int TestSsyrLower() { return TestSsyrCase("L", ACLBLAS_LOWER); }

int32_t main(int32_t argc, char* argv[])
{
    int ret1 = TestSsyrUpper();
    int ret2 = TestSsyrLower();

    if (ret1 == 0 && ret2 == 0) {
        std::cout << "[PASS] syr_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] syr_test" << std::endl;
        return 1;
    }
}
