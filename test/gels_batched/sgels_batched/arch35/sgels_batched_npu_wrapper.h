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
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace gels_npu {

inline void freeAll(std::vector<void*>& dA, std::vector<void*>& dC, void* dAptrs, void* dCptrs, void* dInfo)
{
    for (auto* p : dA)
        aclrtFree(p);
    for (auto* p : dC)
        aclrtFree(p);
    aclrtFree(dAptrs);
    aclrtFree(dCptrs);
    aclrtFree(dInfo);
}

inline aclblasStatus_t allocAndCopyBatch(
    std::vector<void*>& dBufs, const std::vector<std::vector<float>>& hostData, size_t bytes, int batchSize)
{
    dBufs.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError ret = aclrtMalloc(&dBufs[b], bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        if (hostData[b].data() != nullptr) {
            ret = aclrtMemcpy(dBufs[b], bytes, hostData[b].data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t allocAndCopyC(
    std::vector<void*>& dC, const std::vector<std::vector<float>>& Chost, size_t cBytes, size_t cUserBytes,
    int batchSize)
{
    dC.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError ret = aclrtMalloc(&dC[b], cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        if (cBytes > cUserBytes) {
            std::vector<float> zeros(cBytes / sizeof(float), 0.0f);
            aclError zeroRet = aclrtMemcpy(dC[b], cBytes, zeros.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (zeroRet != ACL_SUCCESS)
                return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        if (Chost[b].data() != nullptr && cUserBytes > 0) {
            ret = aclrtMemcpy(dC[b], cUserBytes, Chost[b].data(), cUserBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t buildPtrArrays(
    const std::vector<void*>& dA, const std::vector<void*>& dC, void*& dAptrsMem, void*& dCptrsMem)
{
    int bs = static_cast<int>(dA.size());
    size_t ptrBytes = static_cast<size_t>(bs) * sizeof(float*);

    aclError ret = aclrtMalloc(&dAptrsMem, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    ret = aclrtMalloc(&dCptrsMem, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;

    std::vector<float*> hAptrs(bs), hCptrs(bs);
    for (int b = 0; b < bs; b++) {
        hAptrs[b] = static_cast<float*>(dA[b]);
        hCptrs[b] = static_cast<float*>(dC[b]);
    }
    aclError cpyRet = aclrtMemcpy(dAptrsMem, ptrBytes, hAptrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (cpyRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    cpyRet = aclrtMemcpy(dCptrsMem, ptrBytes, hCptrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (cpyRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    return ACLBLAS_STATUS_SUCCESS;
}

inline void copyResultsBack(
    const std::vector<void*>& dA, const std::vector<void*>& dC, int batchSize, int lda, int n, int ldc, int solRows,
    int nrhs, std::vector<std::vector<float>>& Aout, std::vector<std::vector<float>>& Cout)
{
    Aout.resize(batchSize);
    Cout.resize(batchSize);
    for (int b = 0; b < batchSize; b++) {
        Aout[b].resize(static_cast<size_t>(lda) * n);
        Cout[b].resize(static_cast<size_t>(ldc) * nrhs);
        aclrtMemcpy(
            Aout[b].data(), Aout[b].size() * sizeof(float), dA[b], Aout[b].size() * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
        size_t cSolBytes = static_cast<size_t>(ldc) * nrhs * sizeof(float);
        aclrtMemcpy(Cout[b].data(), cSolBytes, dC[b], cSolBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
}

} // namespace gels_npu

static inline aclblasStatus_t allocAllResources(
    std::vector<void*>& dA, std::vector<void*>& dC, void*& dAptrs, void*& dCptrs, void*& dInfo,
    const std::vector<std::vector<float>>& Ahost, const std::vector<std::vector<float>>& Chost, size_t aBytes,
    size_t cBytes, size_t cUserBytes, int batchSize)
{
    aclblasStatus_t st = gels_npu::allocAndCopyBatch(dA, Ahost, aBytes, batchSize);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = gels_npu::allocAndCopyC(dC, Chost, cBytes, cUserBytes, batchSize);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = gels_npu::buildPtrArrays(dA, dC, dAptrs, dCptrs);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    aclError aclRet = aclrtMalloc(&dInfo, static_cast<size_t>(batchSize) * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t runKernelAndCollect(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, int lda, int ldc, int batchSize,
    void* dAptrs, void* dCptrs, void* dInfo, int& hostDevInfo)
{
    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);
    aclblasStatus_t ret = aclblasSgelsBatched(
        handle, trans, m, n, nrhs, static_cast<float* const*>(dAptrs), lda, static_cast<float* const*>(dCptrs), ldc,
        static_cast<int*>(dInfo), batchSize);
    aclrtSynchronizeStream(stream);
    int infoVal = 0;
    aclrtMemcpy(&infoVal, sizeof(int), dInfo, sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    hostDevInfo = infoVal;
    return ret;
}

inline aclblasStatus_t aclblasSgelsBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs,
    const std::vector<std::vector<float>>& Ahost, const std::vector<std::vector<float>>& Chost, int lda, int ldc,
    int& hostDevInfo, int batchSize, std::vector<std::vector<float>>& Aout, std::vector<std::vector<float>>& Cout)
{
    if (handle == nullptr || batchSize < 0) {
        int* devInfoPtr = nullptr;
        return aclblasSgelsBatched(handle, trans, m, n, nrhs, nullptr, lda, nullptr, ldc, devInfoPtr, batchSize);
    }
    if (m == 0 || n == 0 || nrhs == 0 || batchSize == 0) {
        hostDevInfo = 0;
        Aout.clear();
        Cout.clear();
        return ACLBLAS_STATUS_SUCCESS;
    }

    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t cUserBytes = static_cast<size_t>(ldc) * nrhs * sizeof(float);
    const int solRows = (trans == ACLBLAS_OP_N) ? n : m;
    const size_t cBytes = std::max(cUserBytes, static_cast<size_t>(ldc) * std::max({m, n, nrhs}) * sizeof(float));

    std::vector<void*> dA, dC;
    void *dAptrs = nullptr, *dCptrs = nullptr, *dInfo = nullptr;

    aclblasStatus_t st =
        allocAllResources(dA, dC, dAptrs, dCptrs, dInfo, Ahost, Chost, aBytes, cBytes, cUserBytes, batchSize);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        gels_npu::freeAll(dA, dC, dAptrs, dCptrs, dInfo);
        return st;
    }

    aclblasStatus_t ret =
        runKernelAndCollect(handle, trans, m, n, nrhs, lda, ldc, batchSize, dAptrs, dCptrs, dInfo, hostDevInfo);

    gels_npu::copyResultsBack(dA, dC, batchSize, lda, n, ldc, solRows, nrhs, Aout, Cout);
    gels_npu::freeAll(dA, dC, dAptrs, dCptrs, dInfo);
    return ret;
}

inline aclblasStatus_t aclblasSgelsBatched_npu_error(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int nrhs, float* const Aarray[], int lda,
    float* const Carray[], int ldc, int* devInfo, int batchSize)
{
    return aclblasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, devInfo, batchSize);
}

