/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_EX_NPU_WRAPPER_H
#define GEMM_GROUPED_BATCHED_EX_NPU_WRAPPER_H

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"
#include "dtype_cast.h"
#include "gemm_grouped_batched_ex_param.h"

inline size_t DtypeElementSizeNpu(aclDataType dt)
{
    if (dt == ACL_FLOAT16) return 2;
    if (dt == ACL_BF16) return 2;
    if (dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2) return 1;
    return 4;
}

inline size_t MatrixBytes(int rows, int cols, int ld, aclDataType dtype)
{
    return static_cast<size_t>(ld) * static_cast<size_t>(cols) * DtypeElementSizeNpu(dtype);
}

struct GroupedScaleStorage {
    std::vector<uint16_t> alphaHalf;
    std::vector<uint16_t> betaHalf;
    const void* alphaData = nullptr;
    const void* betaData = nullptr;
};

inline aclblasStatus_t ValidateNpuArgs(aclblasHandle_t handle, const aclblasOperation_t transaArray[],
    const aclblasOperation_t transbArray[], const int mArray[], const int nArray[], const int kArray[],
    const float* alphaArray, const int ldaArray[], const int ldbArray[], const float* betaArray,
    const int ldcArray[], int groupCount, const int groupSize[])
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (groupCount < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (groupSize == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (transaArray == nullptr || transbArray == nullptr || mArray == nullptr ||
        nArray == nullptr || kArray == nullptr || alphaArray == nullptr ||
        betaArray == nullptr || ldaArray == nullptr || ldbArray == nullptr ||
        ldcArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline GroupedScaleStorage PrepareScaleStorage(
    const float* alphaArray, const float* betaArray, int groupCount, aclblasComputeType_t computeType)
{
    GroupedScaleStorage storage;
    storage.alphaData = alphaArray;
    storage.betaData = betaArray;
    if (computeType != ACLBLAS_COMPUTE_16F) { return storage; }
    storage.alphaHalf.resize(static_cast<size_t>(groupCount));
    storage.betaHalf.resize(static_cast<size_t>(groupCount));
    for (int g = 0; g < groupCount; ++g) {
        storage.alphaHalf[static_cast<size_t>(g)] = blas_common::FloatToHalf(alphaArray[g]);
        storage.betaHalf[static_cast<size_t>(g)] = blas_common::FloatToHalf(betaArray[g]);
    }
    storage.alphaData = storage.alphaHalf.data();
    storage.betaData = storage.betaHalf.data();
    return storage;
}

inline aclblasStatus_t RunEmptyGroup(aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const GroupedScaleStorage& scales,
    aclDataType Atype, const int ldaArray[], aclDataType Btype, const int ldbArray[],
    aclDataType Ctype, const int ldcArray[], int groupCount, const int groupSize[],
    aclblasComputeType_t computeType)
{
    std::vector<const void*> aPtrHost(1, nullptr);
    std::vector<const void*> bPtrHost(1, nullptr);
    std::vector<void*> cPtrHost(1, nullptr);
    DeviceBuffer dAPtrArray(sizeof(void*));
    DeviceBuffer dBPtrArray(sizeof(void*));
    DeviceBuffer dCPtrArray(sizeof(void*));
    dAPtrArray.copyFromHost(aPtrHost.data(), sizeof(void*));
    dBPtrArray.copyFromHost(bPtrHost.data(), sizeof(void*));
    dCPtrArray.copyFromHost(cPtrHost.data(), sizeof(void*));
    return aclblasGemmGroupedBatchedEx(handle, transaArray, transbArray,
        mArray, nArray, kArray, scales.alphaData,
        reinterpret_cast<const void* const*>(dAPtrArray.ptr()), Atype, ldaArray,
        reinterpret_cast<const void* const*>(dBPtrArray.ptr()), Btype, ldbArray,
        scales.betaData,
        reinterpret_cast<void* const*>(dCPtrArray.ptr()), Ctype, ldcArray,
        groupCount, groupSize, computeType);
}

inline int GroupStartNpu(int group, const int groupSize[])
{
    int start = 0;
    for (int i = 0; i < group; ++i) {
        start += groupSize[i];
    }
    return start;
}

inline void AllocateDeviceMatricesForGroup(int group,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const std::vector<std::vector<uint8_t>>& aHostRaw, aclDataType Atype, const int ldaArray[],
    const std::vector<std::vector<uint8_t>>& bHostRaw, aclDataType Btype, const int ldbArray[],
    const std::vector<std::vector<uint8_t>>& cHostRaw, aclDataType Ctype, const int ldcArray[],
    const int groupSize[], std::vector<std::unique_ptr<DeviceBuffer>>& dA,
    std::vector<std::unique_ptr<DeviceBuffer>>& dB, std::vector<std::unique_ptr<DeviceBuffer>>& dC)
{
    int safeM = std::max(1, std::abs(mArray[group]));
    int safeN = std::max(1, std::abs(nArray[group]));
    int safeK = std::max(1, std::abs(kArray[group]));
    int rowsA = (transaArray[group] == ACLBLAS_OP_N) ? safeM : safeK;
    int colsA = (transaArray[group] == ACLBLAS_OP_N) ? safeK : safeM;
    int rowsB = (transbArray[group] == ACLBLAS_OP_N) ? safeK : safeN;
    int colsB = (transbArray[group] == ACLBLAS_OP_N) ? safeN : safeK;
    int start = GroupStartNpu(group, groupSize);
    for (int inst = 0; inst < groupSize[group]; ++inst) {
        int idx = start + inst;
        size_t aBytes = MatrixBytes(rowsA, colsA, ldaArray[group], Atype);
        size_t bBytes = MatrixBytes(rowsB, colsB, ldbArray[group], Btype);
        size_t cBytes = MatrixBytes(safeM, safeN, ldcArray[group], Ctype);
        dA[idx] = std::make_unique<DeviceBuffer>(aBytes);
        dB[idx] = std::make_unique<DeviceBuffer>(bBytes);
        dC[idx] = std::make_unique<DeviceBuffer>(cBytes);
        dA[idx]->copyFromHost(aHostRaw[idx].data(), aBytes);
        dB[idx]->copyFromHost(bHostRaw[idx].data(), bBytes);
        dC[idx]->copyFromHost(cHostRaw[idx].data(), cBytes);
    }
}

inline void CopyDeviceResultsForGroup(int group, const int mArray[], const int nArray[],
    std::vector<std::vector<uint8_t>>& cHostRaw, aclDataType Ctype, const int ldcArray[],
    const int groupSize[], const std::vector<std::unique_ptr<DeviceBuffer>>& dC)
{
    int safeM = std::max(1, std::abs(mArray[group]));
    int safeN = std::max(1, std::abs(nArray[group]));
    int start = GroupStartNpu(group, groupSize);
    for (int inst = 0; inst < groupSize[group]; ++inst) {
        int idx = start + inst;
        size_t cBytes = MatrixBytes(safeM, safeN, ldcArray[group], Ctype);
        dC[idx]->copyToHost(cHostRaw[idx].data(), cBytes);
    }
}

inline aclblasStatus_t InvokeWithDeviceMatrices(aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const GroupedScaleStorage& scales,
    aclDataType Atype, const int ldaArray[], aclDataType Btype, const int ldbArray[],
    aclDataType Ctype, const int ldcArray[], int groupCount, const int groupSize[],
    aclblasComputeType_t computeType, const std::vector<std::unique_ptr<DeviceBuffer>>& dA,
    const std::vector<std::unique_ptr<DeviceBuffer>>& dB,
    const std::vector<std::unique_ptr<DeviceBuffer>>& dC)
{
    size_t totalInst = dA.size();
    std::vector<const void*> aPtrHost(totalInst);
    std::vector<const void*> bPtrHost(totalInst);
    std::vector<void*> cPtrHost(totalInst);
    for (size_t i = 0; i < totalInst; ++i) {
        aPtrHost[i] = dA[i] ? dA[i]->ptr() : nullptr;
        bPtrHost[i] = dB[i] ? dB[i]->ptr() : nullptr;
        cPtrHost[i] = dC[i] ? dC[i]->ptr() : nullptr;
    }
    size_t ptrBytes = totalInst * sizeof(void*);
    DeviceBuffer dAPtrArray(ptrBytes);
    DeviceBuffer dBPtrArray(ptrBytes);
    DeviceBuffer dCPtrArray(ptrBytes);
    dAPtrArray.copyFromHost(aPtrHost.data(), ptrBytes);
    dBPtrArray.copyFromHost(bPtrHost.data(), ptrBytes);
    dCPtrArray.copyFromHost(cPtrHost.data(), ptrBytes);
    return aclblasGemmGroupedBatchedEx(handle, transaArray, transbArray,
        mArray, nArray, kArray, scales.alphaData,
        reinterpret_cast<const void* const*>(dAPtrArray.ptr()), Atype, ldaArray,
        reinterpret_cast<const void* const*>(dBPtrArray.ptr()), Btype, ldbArray,
        scales.betaData, reinterpret_cast<void* const*>(dCPtrArray.ptr()), Ctype, ldcArray,
        groupCount, groupSize, computeType);
}

inline aclblasStatus_t aclblasGemmGroupedBatchedEx_npu(
    aclblasHandle_t handle,
    const aclblasOperation_t transaArray[],
    const aclblasOperation_t transbArray[],
    const int mArray[],
    const int nArray[],
    const int kArray[],
    const float* alphaArray,
    const std::vector<std::vector<uint8_t>>& aHostRaw,
    aclDataType Atype,
    const int ldaArray[],
    const std::vector<std::vector<uint8_t>>& bHostRaw,
    aclDataType Btype,
    const int ldbArray[],
    const float* betaArray,
    std::vector<std::vector<uint8_t>>& cHostRaw,
    aclDataType Ctype,
    const int ldcArray[],
    int groupCount,
    const int groupSize[],
    aclblasComputeType_t computeType)
{
    aclblasStatus_t status = ValidateNpuArgs(handle, transaArray, transbArray, mArray, nArray, kArray,
        alphaArray, ldaArray, ldbArray, betaArray, ldcArray, groupCount, groupSize);
    if (status != ACLBLAS_STATUS_SUCCESS) { return status; }

    int totalInst = 0;
    for (int g = 0; g < groupCount; g++) totalInst += groupSize[g];
    GroupedScaleStorage scales = PrepareScaleStorage(alphaArray, betaArray, groupCount, computeType);

    if (groupCount == 0) {
        return RunEmptyGroup(handle, transaArray, transbArray, mArray, nArray, kArray,
            scales, Atype, ldaArray, Btype, ldbArray, Ctype, ldcArray,
            groupCount, groupSize, computeType);
    }

    std::vector<std::unique_ptr<DeviceBuffer>> dA(totalInst);
    std::vector<std::unique_ptr<DeviceBuffer>> dB(totalInst);
    std::vector<std::unique_ptr<DeviceBuffer>> dC(totalInst);

    for (int g = 0; g < groupCount; g++) {
        AllocateDeviceMatricesForGroup(g, transaArray, transbArray, mArray, nArray, kArray,
            aHostRaw, Atype, ldaArray, bHostRaw, Btype, ldbArray, cHostRaw, Ctype,
            ldcArray, groupSize, dA, dB, dC);
    }
    aclblasStatus_t ret = InvokeWithDeviceMatrices(handle, transaArray, transbArray,
        mArray, nArray, kArray, scales, Atype, ldaArray, Btype, ldbArray,
        Ctype, ldcArray, groupCount, groupSize, computeType, dA, dB, dC);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    for (int g = 0; g < groupCount; g++) {
        CopyDeviceResultsForGroup(g, mArray, nArray, cHostRaw, Ctype, ldcArray, groupSize, dC);
    }

    return ret;
}

#endif // GEMM_GROUPED_BATCHED_EX_NPU_WRAPPER_H
