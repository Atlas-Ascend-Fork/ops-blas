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

int32_t test_scnrm2()
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 8 * 1024;
    constexpr float realVal = 1.5f;
    constexpr float imagVal = 2.0f;

    std::vector<aclblasComplex> x(n, aclblasComplex{realVal, imagVal});
    float result = 0.0f;
    int incx = 1;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    aclblasComplex* xDevice = nullptr;
    float* resultDevice = nullptr;
    size_t inputByteSize = n * sizeof(aclblasComplex);
    size_t outputByteSize = sizeof(float);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasScnrm2(handle, n, xDevice, incx, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasScnrm2 failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    float golden = std::sqrt(n * (realVal * realVal + imagVal * imagVal));

    return VerifyResult(result, golden);
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
    void Release(int32_t deviceId)
    {
        aclrtFree(xDevice);
        aclrtFree(resultDevice);

        aclblasDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
};

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    // Leave one element beyond the 8-element tiling boundary for tail coverage.
    constexpr uint32_t totalLength = 8 * 2048 + 1;
    constexpr float valueX = 1.2f;
    std::vector<float> x(totalLength, valueX);
    float result = 0.0f;
    int64_t incx = 1;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* xDevice = nullptr;
    float* resultDevice = nullptr;
    size_t inputByteSize = totalLength * sizeof(float);
    size_t outputByteSize = sizeof(float);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasSnrm2(handle, totalLength, xDevice, incx, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSnrm2 failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    float golden = std::abs(valueX) * std::sqrt(static_cast<float>(totalLength));

    int snrm2_result = VerifyResult(result, golden);
    if (snrm2_result != 0) {
        LOG_PRINT("normal aclblasSnrm2 case failed.\n");
    }

    Snrm2TestContext context{x, handle, stream, xDevice, resultDevice};
    uint32_t edgeCaseResult = context.RunEdgeCases();

    context.Release(deviceId);

    if (snrm2_result != 0 || edgeCaseResult != 0) {
        return 1;
    }

    std::cout << "\n========== Testing scnrm2 ==========\n" << std::endl;
    return test_scnrm2();
}
