/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file snrm2_test.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "error_check.h"
#include "device.h"

uint32_t VerifyResult(float output, float golden)
{
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    float relError = std::abs(output - golden) / std::abs(golden);
    constexpr float epsilon = 1e-5f;

    if (relError < epsilon) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed! Relative error: " << relError << std::endl;
        return 1;
    }
}

// 设备缓冲统一用 DeviceBuffer 托管：任何失败路径都不再泄漏
// aclrtMalloc 内存，也不跳过 handle/stream/上下文清理（issue #356）
int32_t test_scnrm2()
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 8 * 1024;
    constexpr float realVal = 1.5f;
    constexpr float imagVal = 2.0f;
    constexpr int incx = 1;

    std::vector<aclblasComplex> x(n, aclblasComplex{realVal, imagVal});
    float result = 0.0f;
    int32_t ret = 0;

    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;

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

    {
        try {
            DeviceBuffer xDev(n * sizeof(aclblasComplex));
            DeviceBuffer resultDev(sizeof(float));
            xDev.copyFromHost(x.data(), xDev.size());

            ret = aclblasScnrm2(handle, n, static_cast<aclblasComplex*>(xDev.ptr()), incx, resultDev.floatPtr());
            if (ret != ACLBLAS_STATUS_SUCCESS) {
                std::cout << "aclblasScnrm2 failed. ERROR: " << ret << std::endl;
                goto cleanup;
            }
            aclError aclRet = aclrtSynchronizeStream(stream);
            if (aclRet != ACL_SUCCESS) {
                std::cout << "aclrtSynchronizeStream failed. ERROR: " << aclRet << std::endl;
                ret = static_cast<int32_t>(aclRet);
                goto cleanup;
            }
            resultDev.copyToHost(&result, sizeof(float));
        } catch (const std::exception& e) {
            std::cout << "device memory operation failed: " << e.what() << std::endl;
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
            goto cleanup;
        }

        float golden = std::sqrt(n * (realVal * realVal + imagVal * imagVal));
        ret = static_cast<int32_t>(VerifyResult(result, golden));
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

struct Snrm2TestContext {
    std::vector<float>& x;
    aclblasHandle_t handle;
    aclrtStream stream;
    float* xDevice;
    float* resultDevice;

    uint32_t RunCase(const char* name, float golden, bool expectInf, bool expectNan)
    {
        const uint32_t totalLength = static_cast<uint32_t>(x.size());
        const size_t inputByteSize = x.size() * sizeof(float);
        const size_t outputByteSize = sizeof(float);
        const int64_t incx = 1;
        float result = 0.0f;
        std::cout << "Testing " << name << std::endl;
        aclError copyRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (copyRet != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy %s failed. ERROR: %d\n", name, copyRet);
            return 1;
        }
        auto callRet = aclblasSnrm2(handle, totalLength, xDevice, incx, resultDevice);
        if (callRet != ACLBLAS_STATUS_SUCCESS) {
            LOG_PRINT("aclblasSnrm2 %s failed. ERROR: %d\n", name, callRet);
            return 1;
        }
        aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            LOG_PRINT("aclrtSynchronizeStream %s failed. ERROR: %d\n", name, syncRet);
            return 1;
        }
        copyRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            LOG_PRINT("aclrtMemcpy result for %s failed. ERROR: %d\n", name, copyRet);
            return 1;
        }
        if (expectNan) {
            bool passed = std::isnan(result);
            std::cout << (passed ? "[Success]" : "[Failed]") << " Expected NaN, got " << result << std::endl;
            return passed ? 0 : 1;
        }
        if (expectInf) {
            bool passed = std::isinf(result) && result > 0.0f;
            std::cout << (passed ? "[Success]" : "[Failed]") << " Expected +Inf, got " << result << std::endl;
            return passed ? 0 : 1;
        }
        return VerifyResult(result, golden);
    }

    uint32_t RunEdgeCases()
    {
        uint32_t edgeCaseResult = 0;
        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = 3.0f;
        x[1] = 4.0f;
        edgeCaseResult |= RunCase("ordinary [3,4]", 5.0f, false, false);

        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = 1.0e20f;
        edgeCaseResult |= RunCase("single 1e20 input", 1.0e20f, false, false);

        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = std::numeric_limits<float>::max();
        edgeCaseResult |= RunCase("single FLT_MAX input", std::numeric_limits<float>::max(), false, false);

        // Values span multiple cores; the last value occupies the one-element tail.
        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = 1.0e20f;
        x[2048] = 1.0e20f;
        x.back() = 1.0e20f;
        float multipleLargeGolden = std::hypot(std::hypot(1.0e20f, 1.0e20f), 1.0e20f);
        edgeCaseResult |= RunCase("multiple large inputs across cores and tail", multipleLargeGolden, false, false);

        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = std::numeric_limits<float>::infinity();
        edgeCaseResult |= RunCase("positive infinity", 0.0f, true, false);

        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = std::numeric_limits<float>::quiet_NaN();
        edgeCaseResult |= RunCase("NaN", 0.0f, false, true);

        std::fill(x.begin(), x.end(), 0.0f);
        x[0] = std::numeric_limits<float>::infinity();
        x.back() = std::numeric_limits<float>::quiet_NaN();
        edgeCaseResult |= RunCase("infinity and NaN", 0.0f, false, true);
        return edgeCaseResult;
    }
};

static int32_t runSnrm2Cases()
{
    int32_t deviceId = 0;

    // Leave one element beyond the 8-element tiling boundary for tail coverage.
    constexpr uint32_t totalLength = 8 * 2048 + 1;
    constexpr float valueX = 1.2f;
    constexpr int64_t incx = 1;
    std::vector<float> x(totalLength, valueX);
    float result = 0.0f;
    int32_t ret = 0;

    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;

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

    {
        // DeviceBuffer 声明在 try 之外：生命周期需覆盖 try 内的使用与
        // try 后 Snrm2TestContext 边界用例（跨核+尾块大值、FLT_MAX、Inf/NaN）的引用
        DeviceBuffer xDev(totalLength * sizeof(float));
        DeviceBuffer resultDev(sizeof(float));
        try {
            xDev.copyFromHost(x.data(), xDev.size());

            ret = aclblasSnrm2(handle, totalLength, xDev.floatPtr(), incx, resultDev.floatPtr());
            if (ret != ACLBLAS_STATUS_SUCCESS) {
                std::cout << "aclblasSnrm2 failed. ERROR: " << ret << std::endl;
                goto cleanup;
            }
            aclError aclRet = aclrtSynchronizeStream(stream);
            if (aclRet != ACL_SUCCESS) {
                std::cout << "aclrtSynchronizeStream failed. ERROR: " << aclRet << std::endl;
                ret = static_cast<int32_t>(aclRet);
                goto cleanup;
            }
            resultDev.copyToHost(&result, sizeof(float));
        } catch (const std::exception& e) {
            std::cout << "device memory operation failed: " << e.what() << std::endl;
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
            goto cleanup;
        }

        float golden = std::abs(valueX) * std::sqrt(static_cast<float>(totalLength));
        int snrm2Result = VerifyResult(result, golden);
        if (snrm2Result != 0) {
            LOG_PRINT("normal aclblasSnrm2 case failed.\n");
            ret = 1;
        }

        // 缓冲区由 DeviceBuffer 托管，上下文仅持有非拥有指针
        Snrm2TestContext context{x, handle, stream, xDev.floatPtr(), resultDev.floatPtr()};
        uint32_t edgeCaseResult = context.RunEdgeCases();
        if (edgeCaseResult != 0) {
            ret = 1;
        }
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

int32_t main(int32_t argc, char* argv[])
{
    int32_t ret = runSnrm2Cases();
    if (ret != 0) {
        return ret;
    }

    std::cout << "\n========== Testing scnrm2 ==========\n" << std::endl;
    return test_scnrm2();
}
