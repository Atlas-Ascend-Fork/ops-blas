/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <memory>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "complex.h"

using namespace std;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

aclblasComplex complexMul(const aclblasComplex& a, const aclblasComplex& b) { return a * b; }

void computeGolden(
    const vector<aclblasComplex>& A, const vector<aclblasComplex>& x, vector<aclblasComplex>& y, int m, int n,
    int trans, const aclblasComplex& alpha, const aclblasComplex& beta)
{
    vector<aclblasComplex> yTemp(m);
    for (int i = 0; i < m; i++) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int j = 0; j < n; j++) {
            if (trans == 0) {
                sum += A[i * n + j] * x[j];
            } else if (trans == 1) {
                sum += A[j * m + i] * x[j];
            } else {
                aclblasComplex conjA{A[j * m + i].real, -A[j * m + i].imag};
                sum += conjA * x[j];
            }
        }
        yTemp[i] = sum;
    }

    for (int i = 0; i < m; i++) {
        y[i] = alpha * yTemp[i] + beta * y[i];
    }
}

uint32_t VerifyResult(std::vector<aclblasComplex>& output, std::vector<aclblasComplex>& golden)
{
    auto printTensor = [](std::vector<aclblasComplex>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 10;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(tensor.size(), maxPrintSize); i++) {
            std::cout << "(" << tensor[i].real << "," << tensor[i].imag << ") ";
        }
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    constexpr float epsilon = 1e-3f;
    size_t errorCount = 0;
    float maxError = 0.0f;

    for (size_t i = 0; i < output.size(); i++) {
        float error = blasComplexAbs(output[i] - golden[i]);
        if (error > maxError) {
            maxError = error;
        }
        if (error > epsilon) {
            errorCount++;
        }
    }

    std::cout << "Max error: " << maxError << std::endl;

    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy verification failed." << std::endl;
        return 1;
    }
}

// The file is collected into the cgemv_batched_test gtest target, whose main()
// comes from test/frame/test_main.cpp. Keep the standalone entry point behind a
// guard so the file can still be built and run on its own (same pattern as the
// strmm_test_netlib.cpp adapter in test/trmm/strmm/).
#ifdef CGEMMV_BATCHED_STANDALONE

// RAII guards so every early return releases the resources acquired so far. The
// previous CHECK_RET cleanup chains had to repeat the full release list at each
// failure site and silently leaked whatever was added after the last edit.
namespace {

class AclFinalizer {
public:
    ~AclFinalizer() { aclFinalize(); }
};

class DeviceGuard {
public:
    ~DeviceGuard() { aclrtResetDevice(0); }
};

class HandleGuard {
public:
    explicit HandleGuard(aclblasHandle_t handle) : handle_(handle) {}
    ~HandleGuard()
    {
        if (handle_ != nullptr) {
            aclblasDestroy(handle_);
        }
    }

private:
    aclblasHandle_t handle_;
};

class StreamGuard {
public:
    explicit StreamGuard(aclrtStream stream) : stream_(stream) {}
    ~StreamGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
    }

private:
    aclrtStream stream_;
};

class DeviceBufferGuard {
public:
    explicit DeviceBufferGuard(void** devPtr, size_t bytes) : devPtr_(devPtr)
    {
        aclError ret = aclrtMalloc(devPtr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            *devPtr_ = nullptr;
        }
    }
    ~DeviceBufferGuard()
    {
        if (*devPtr_ != nullptr) {
            aclrtFree(*devPtr_);
        }
    }

private:
    void** devPtr_;
};

struct BatchCase {
    aclblasOperation_t trans;
    int transMode;
    const char* name;
};

} // namespace

static void FillHostData(
    vector<aclblasComplex>& AHost, vector<aclblasComplex>& xHost, int64_t batchCount, int64_t m, int64_t n)
{
    AHost.assign(static_cast<size_t>(batchCount * m * n), aclblasComplex{0.0f, 0.0f});
    xHost.assign(static_cast<size_t>(batchCount * n), aclblasComplex{0.0f, 0.0f});
    for (int b = 0; b < batchCount; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = b * m * n + i * n + j;
                AHost[idx] = aclblasComplex{(i + j + b) * 0.1f, (i - j + b) * 0.1f};
            }
        }
        for (int i = 0; i < n; i++) {
            int idx = b * n + i;
            xHost[idx] = aclblasComplex{(i + b) * 0.2f, (i - b) * 0.2f};
        }
    }
}

static void ComputeBatchGolden(
    vector<aclblasComplex>& yGolden, const vector<aclblasComplex>& AHost, const vector<aclblasComplex>& xHost,
    int64_t batchCount, int64_t m, int64_t n, int transMode, const aclblasComplex& alpha, const aclblasComplex& beta)
{
    for (int b = 0; b < batchCount; b++) {
        vector<aclblasComplex> AComplex(static_cast<size_t>(m * n));
        vector<aclblasComplex> xComplex(static_cast<size_t>(n));
        vector<aclblasComplex> yComplex(static_cast<size_t>(m));
        for (int i = 0; i < m * n; i++) {
            AComplex[i] = AHost[b * m * n + i];
        }
        for (int i = 0; i < n; i++) {
            xComplex[i] = xHost[b * n + i];
        }
        computeGolden(AComplex, xComplex, yComplex, m, n, transMode, alpha, beta);
        for (int i = 0; i < m; i++) {
            yGolden[b * m + i] = yComplex[i];
        }
    }
}

static int UploadPtrArrays(
    aclblasComplex* dAPtrArr, aclblasComplex* dXPtrArr, aclblasComplex* dYPtrArr, const aclblasComplex* aDevice,
    const aclblasComplex* xDevice, aclblasComplex* yDevice, int64_t batchCount, int64_t m, int64_t n,
    size_t ptrArrBytes)
{
    vector<aclblasComplex*> hAPtrArr(static_cast<size_t>(batchCount), nullptr);
    vector<aclblasComplex*> hXPtrArr(static_cast<size_t>(batchCount), nullptr);
    vector<aclblasComplex*> hYPtrArr(static_cast<size_t>(batchCount), nullptr);
    for (int b = 0; b < batchCount; b++) {
        hAPtrArr[b] = const_cast<aclblasComplex*>(aDevice) + b * m * n;
        hXPtrArr[b] = const_cast<aclblasComplex*>(xDevice) + b * n;
        hYPtrArr[b] = yDevice + b * m;
    }
    aclError ret = aclrtMemcpy(dAPtrArr, ptrArrBytes, hAPtrArr.data(), ptrArrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy dAPtrArr failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(dXPtrArr, ptrArrBytes, hXPtrArr.data(), ptrArrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy dXPtrArr failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(dYPtrArr, ptrArrBytes, hYPtrArr.data(), ptrArrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy dYPtrArr failed. ERROR: %d\n", ret);
        return ret;
    }
    return ACL_SUCCESS;
}

static int RunOneCase(
    aclblasHandle_t handle, aclrtStream stream, const BatchCase& oneCase, int64_t m, int64_t n, int64_t lda,
    int64_t incx, int64_t incy, int64_t batchCount, const aclblasComplex& alpha, const aclblasComplex& beta,
    const aclblasComplex* dAPtrArr, const aclblasComplex* dXPtrArr, aclblasComplex* dYPtrArr, aclblasComplex* yDevice,
    size_t yByteSize, const vector<aclblasComplex>& yInitial, vector<aclblasComplex>& yHost,
    vector<aclblasComplex>& yGolden, const vector<aclblasComplex>& AHost, const vector<aclblasComplex>& xHost,
    bool& allPassed)
{
    aclError ret = aclrtMemcpy(yDevice, yByteSize, yInitial.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy y reset before OP_%s failed. ERROR: %d\n", oneCase.name, ret);
        return ret;
    }
    aclblasStatus_t blasRet = aclblasCgemvBatched(
        handle, oneCase.trans, m, n, &alpha, (const aclblasComplex* const*)dAPtrArr, lda,
        (const aclblasComplex* const*)dXPtrArr, incx, &beta, (aclblasComplex* const*)dYPtrArr, incy, batchCount);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasCgemvBatched(OP_%s) failed. ERROR: %d\n", oneCase.name, blasRet);
        return blasRet;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream after OP_%s failed. ERROR: %d\n", oneCase.name, ret);
        return ret;
    }
    ret = aclrtMemcpy(yHost.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy OP_%s result failed. ERROR: %d\n", oneCase.name, ret);
        return ret;
    }
    ComputeBatchGolden(yGolden, AHost, xHost, batchCount, m, n, oneCase.transMode, alpha, beta);
    if (VerifyResult(yHost, yGolden) != 0) {
        allPassed = false;
    }
    return ACL_SUCCESS;
}

int main()
{
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    const AclFinalizer aclFinalizer;

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    const DeviceGuard deviceGuard;

    aclblasHandle_t handle = nullptr;
    aclblasStatus_t blasRet = aclblasCreate(&handle);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
        return blasRet;
    }
    const HandleGuard handleGuard(handle);

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    const StreamGuard streamGuard(stream);

    blasRet = aclblasSetStream(handle, stream);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
        return blasRet;
    }

    const int64_t batchCount = 3;
    const int64_t m = 32;
    const int64_t n = 32;
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    const int64_t lda = m;
    const int64_t incx = 1;
    const int64_t incy = 1;

    vector<aclblasComplex> AHost;
    vector<aclblasComplex> xHost;
    FillHostData(AHost, xHost, batchCount, m, n);
    vector<aclblasComplex> yHost(static_cast<size_t>(batchCount * m));
    vector<aclblasComplex> yGolden(static_cast<size_t>(batchCount * m));
    const vector<aclblasComplex> yInitial(yHost.size(), aclblasComplex{0.0f, 0.0f});

    aclblasComplex* aDevice = nullptr;
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    const size_t aByteSize = static_cast<size_t>(batchCount * m * n) * sizeof(aclblasComplex);
    const size_t xByteSize = static_cast<size_t>(batchCount * n) * sizeof(aclblasComplex);
    const size_t yByteSize = static_cast<size_t>(batchCount * m) * sizeof(aclblasComplex);
    const DeviceBufferGuard aGuard(reinterpret_cast<void**>(&aDevice), aByteSize);
    const DeviceBufferGuard xGuard(reinterpret_cast<void**>(&xDevice), xByteSize);
    const DeviceBufferGuard yGuard(reinterpret_cast<void**>(&yDevice), yByteSize);
    if (aDevice == nullptr || xDevice == nullptr || yDevice == nullptr) {
        LOG_PRINT("aclrtMalloc of the A/x/y device buffers failed\n");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = aclrtMemcpy(aDevice, aByteSize, AHost.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(xDevice, xByteSize, xHost.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(yDevice, yByteSize, yHost.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", ret);
        return ret;
    }

    // The public API takes device-side pointer arrays (cuBLAS cublasCgemvBatched
    // form), so build one device pointer per batch and a device array holding
    // those pointers.
    aclblasComplex* dAPtrArr = nullptr;
    aclblasComplex* dXPtrArr = nullptr;
    aclblasComplex* dYPtrArr = nullptr;
    const size_t ptrArrBytes = static_cast<size_t>(batchCount) * sizeof(aclblasComplex*);
    const DeviceBufferGuard aPtrGuard(reinterpret_cast<void**>(&dAPtrArr), ptrArrBytes);
    const DeviceBufferGuard xPtrGuard(reinterpret_cast<void**>(&dXPtrArr), ptrArrBytes);
    const DeviceBufferGuard yPtrGuard(reinterpret_cast<void**>(&dYPtrArr), ptrArrBytes);
    if (dAPtrArr == nullptr || dXPtrArr == nullptr || dYPtrArr == nullptr) {
        LOG_PRINT("aclrtMalloc of the pointer arrays failed\n");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ret = UploadPtrArrays(dAPtrArr, dXPtrArr, dYPtrArr, aDevice, xDevice, yDevice, batchCount, m, n, ptrArrBytes);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const BatchCase cases[] = {
        {ACLBLAS_OP_N, 0, "N"},
        {ACLBLAS_OP_T, 1, "T"},
        {ACLBLAS_OP_C, 2, "C"},
    };
    bool allPassed = true;
    for (const BatchCase& oneCase : cases) {
        ret = RunOneCase(
            handle, stream, oneCase, m, n, lda, incx, incy, batchCount, alpha, beta, dAPtrArr, dXPtrArr, dYPtrArr,
            yDevice, yByteSize, yInitial, yHost, yGolden, AHost, xHost, allPassed);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }

    return allPassed ? ACL_SUCCESS : 1;
}
#endif // CGEMMV_BATCHED_STANDALONE
