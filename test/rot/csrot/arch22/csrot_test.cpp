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

static int32_t RunCsrotNegativeChecks(aclblasHandle_t handle)
{
    aclblasComplex sx{1.0f, 0.0f};
    aclblasComplex sy{2.0f, 0.0f};
    const float cs = 0.7071067811865476f;
    const float sn = 0.7071067811865476f;
    if (aclblasCsrot(nullptr, 4, &sx, 1, &sy, 1, &cs, &sn) != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[Failed] csrot null handle not rejected" << std::endl;
        return 1;
    }
    if (aclblasCsrot(handle, -1, &sx, 1, &sy, 1, &cs, &sn) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] csrot negative n not rejected" << std::endl;
        return 1;
    }
    if (aclblasCsrot(handle, 0, &sx, 1, &sy, 1, &cs, &sn) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] csrot zero n not fast-returned" << std::endl;
        return 1;
    }
    if (aclblasCsrot(handle, 4, nullptr, 1, &sy, 1, &cs, &sn) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasCsrot(handle, 4, &sx, 1, nullptr, 1, &cs, &sn) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] csrot null x/y not rejected" << std::endl;
        return 1;
    }
    if (aclblasCsrot(handle, 4, &sx, 0, &sy, 1, &cs, &sn) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasCsrot(handle, 4, &sx, 1, &sy, 0, &cs, &sn) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] csrot zero stride not rejected" << std::endl;
        return 1;
    }
    std::cout << "[Success] csrot negative checks passed." << std::endl;
    return 0;
}

// Stride regression (issue #638): positive/negative/mixed strides against a
// host-side golden, with sentinel slots that must stay untouched. The last
// case uses n > cores*pass-cap so the strided kernel's multi-element and
// multi-pass splitting is actually exercised (!720 review).

// Values stay bounded and the x/y ranges are separated so no rotated
// component cancels towards zero (the shared 1e-5 relative epsilon has no
// absolute floor); periodicity 90 keeps neighbouring slots distinguishable.
static void BuildCsrotStrideGolden(
    int nElems, int incx, int incy, std::vector<aclblasComplex>& xHost, std::vector<aclblasComplex>& yHost,
    std::vector<aclblasComplex>& goldenX, std::vector<aclblasComplex>& goldenY)
{
    constexpr float cs = 0.7071067811865476f;
    constexpr float sn = 0.7071067811865476f;
    auto physIdx = [](int i, int inc, int n) -> int { return inc > 0 ? i * inc : (n - 1 - i) * (-inc); };
    for (size_t j = 0; j < xHost.size(); j++) {
        xHost[j] = aclblasComplex{10.0f + static_cast<float>(j % 90), 10.0f + static_cast<float>(j % 90)};
    }
    for (size_t j = 0; j < yHost.size(); j++) {
        yHost[j] = aclblasComplex{110.0f + static_cast<float>(j % 90), 110.0f + static_cast<float>(j % 90)};
    }
    goldenX = xHost;
    goldenY = yHost;
    for (int i = 0; i < nElems; i++) {
        const int px = physIdx(i, incx, nElems);
        const int py = physIdx(i, incy, nElems);
        goldenX[px].real = cs * xHost[px].real + sn * yHost[py].real;
        goldenX[px].imag = cs * xHost[px].imag + sn * yHost[py].imag;
        goldenY[py].real = cs * yHost[py].real - sn * xHost[px].real;
        goldenY[py].imag = cs * yHost[py].imag - sn * xHost[px].imag;
    }
}

static int32_t RunCsrotStrideCase(
    aclblasHandle_t handle, aclrtStream stream, int nElems, int incx, int incy, std::vector<aclblasComplex>& xHost,
    std::vector<aclblasComplex>& goldenX, std::vector<aclblasComplex>& yHost, std::vector<aclblasComplex>& goldenY)
{
    constexpr float cs = 0.7071067811865476f;
    constexpr float sn = 0.7071067811865476f;
    try {
        DeviceBuffer xDev(xHost.size() * sizeof(aclblasComplex));
        DeviceBuffer yDev(yHost.size() * sizeof(aclblasComplex));
        xDev.copyFromHost(xHost.data(), xDev.size());
        yDev.copyFromHost(yHost.data(), yDev.size());
        const aclblasStatus_t callRet = aclblasCsrot(
            handle, nElems, static_cast<aclblasComplex*>(xDev.ptr()), incx, static_cast<aclblasComplex*>(yDev.ptr()),
            incy, &cs, &sn);
        if (callRet != ACLBLAS_STATUS_SUCCESS) {
            std::cout << "[Failed] csrot stride case incx=" << incx << " incy=" << incy << " n=" << nElems
                      << " call ret=" << callRet << std::endl;
            return 1;
        }
        CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed\n");
                  return 1);
        xDev.copyToHost(xHost.data(), xDev.size());
        yDev.copyToHost(yHost.data(), yDev.size());
        if (VerifyResult(xHost, yHost, goldenX, goldenY) != 0) {
            std::cout << "[Failed] csrot stride case incx=" << incx << " incy=" << incy << " n=" << nElems << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "[Failed] device memory operation failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

static int32_t RunCsrotStrideCases(aclblasHandle_t handle, aclrtStream stream)
{
    const int caseShapes[][3] = {{2, 1, 4}, {2, -2, 4}, {-1, -1, 4}, {1, -2, 4}, {2, -2, 25000}};
    for (const auto& shape : caseShapes) {
        const int incx = shape[0];
        const int incy = shape[1];
        const int nElems = shape[2];
        const size_t xSlots = static_cast<size_t>((nElems - 1) * (incx < 0 ? -incx : incx) + 1);
        const size_t ySlots = static_cast<size_t>((nElems - 1) * (incy < 0 ? -incy : incy) + 1);
        std::vector<aclblasComplex> xHost(xSlots);
        std::vector<aclblasComplex> yHost(ySlots);
        std::vector<aclblasComplex> goldenX;
        std::vector<aclblasComplex> goldenY;
        BuildCsrotStrideGolden(nElems, incx, incy, xHost, yHost, goldenX, goldenY);
        if (RunCsrotStrideCase(handle, stream, nElems, incx, incy, xHost, goldenX, yHost, goldenY) != 0) {
            return 1;
        }
    }
    std::cout << "[Success] csrot stride cases passed." << std::endl;
    return 0;
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

    if (ret == 0) {
        ret = RunCsrotNegativeChecks(handle);
    }
    if (ret == 0) {
        ret = RunCsrotStrideCases(handle, stream);
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
