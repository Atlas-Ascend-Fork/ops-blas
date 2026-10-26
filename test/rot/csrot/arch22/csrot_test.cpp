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
 * \file csrot_test.cpp
 * \brief Test for complex vector rotation
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

uint32_t VerifyResult(
    std::vector<aclblasComplex>& outputX, std::vector<aclblasComplex>& outputY, std::vector<aclblasComplex>& goldenX,
    std::vector<aclblasComplex>& goldenY)
{
    auto printTensor = [](const std::vector<aclblasComplex>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(tensor.size(), maxPrintSize); i++) {
            std::cout << "(" << tensor[i].real << ", " << tensor[i].imag << ") ";
        }
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(outputX, "Output X");
    printTensor(goldenX, "Golden X");
    printTensor(outputY, "Output Y");
    printTensor(goldenY, "Golden Y");

    constexpr float epsilon = 1e-5f;
    size_t errorCount = 0;

    for (size_t i = 0; i < outputX.size(); i++) {
        float relErrorReal = std::abs(outputX[i].real - goldenX[i].real) / (std::abs(goldenX[i].real) + 1e-10f);
        float relErrorImag = std::abs(outputX[i].imag - goldenX[i].imag) / (std::abs(goldenX[i].imag) + 1e-10f);
        if (relErrorReal > epsilon || relErrorImag > epsilon) {
            errorCount++;
        }
    }

    for (size_t i = 0; i < outputY.size(); i++) {
        float relErrorReal = std::abs(outputY[i].real - goldenY[i].real) / (std::abs(goldenY[i].real) + 1e-10f);
        float relErrorImag = std::abs(outputY[i].imag - goldenY[i].imag) / (std::abs(goldenY[i].imag) + 1e-10f);
        if (relErrorReal > epsilon || relErrorImag > epsilon) {
            errorCount++;
        }
    }

    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed! Error count: " << errorCount << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 1024;

    constexpr float theta = M_PI / 4;
    constexpr float c = 0.7071067811865476;
    constexpr float s = 0.7071067811865476;

    // csrot 作用于复数向量：初值带非零虚部，覆盖共轭/虚部路径
    std::vector<aclblasComplex> x(n, aclblasComplex{1.0f, 0.5f});
    std::vector<aclblasComplex> y(n, aclblasComplex{2.0f, -1.0f});

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    int32_t ret = 0;
    handle = nullptr;
    stream = nullptr;
    ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); aclFinalize();
              return ret);
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); goto cleanup);

    {
        // golden 必须基于设备调用前的原始输入计算：csrot 会就地改写 x/y，
        // 在 copyToHost 之后取 x[i]/y[i] 求得的是旋转后的值（issue 修复）
        std::vector<aclblasComplex> goldenX(n);
        std::vector<aclblasComplex> goldenY(n);
        for (uint32_t i = 0; i < n; i++) {
            goldenX[i].real = c * x[i].real + s * y[i].real;
            goldenX[i].imag = c * x[i].imag + s * y[i].imag;
            goldenY[i].real = c * y[i].real - s * x[i].real;
            goldenY[i].imag = c * y[i].imag - s * x[i].imag;
        }

        // 缓冲区按复数元素尺寸分配（issue #358：原实现按 float 尺寸只分配一半）
        DeviceBuffer xDev(n * sizeof(aclblasComplex));
        DeviceBuffer yDev(n * sizeof(aclblasComplex));
        try {
            xDev.copyFromHost(x.data(), xDev.size());
            yDev.copyFromHost(y.data(), yDev.size());
        } catch (const std::exception& e) {
            LOG_PRINT("device memory operation failed: %s\n", e.what());
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
            goto cleanup;
        }

        ret = aclblasCsrot(
            handle, n, static_cast<aclblasComplex*>(xDev.ptr()), 1, static_cast<aclblasComplex*>(yDev.ptr()), 1, &c,
            &s);
        CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCsrot failed. ERROR: %d\n", ret); goto cleanup);

        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
                  LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                  goto cleanup);
        try {
            xDev.copyToHost(x.data(), xDev.size());
            yDev.copyToHost(y.data(), yDev.size());
        } catch (const std::exception& e) {
            LOG_PRINT("device memory operation failed: %s\n", e.what());
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
            goto cleanup;
        }

        ret = static_cast<int32_t>(VerifyResult(x, y, goldenX, goldenY));
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
