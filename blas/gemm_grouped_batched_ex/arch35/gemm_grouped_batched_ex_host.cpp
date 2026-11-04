/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_grouped_batched_ex_host.cpp
 * \brief Host validation/tiling for grouped batched GEMM on DAV_3510.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "gemm_grouped_batched_ex_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"

void gemm_grouped_batched_ex_cube_kernel_do(uint32_t numBlocks, void *stream,
    uint8_t *aarray, uint8_t *barray, uint8_t *workspace, uint8_t *tilingGm, int dtypeCase);
void gemm_grouped_batched_ex_epilogue_kernel_do(uint32_t numBlocks, void *stream,
    uint8_t *carray, uint8_t *workspace, uint8_t *tilingGm, int dtypeCase);

namespace {

constexpr uint32_t EPILOGUE_TILE = 256;

struct GroupedProblem {
    aclDataType aType;
    aclDataType bType;
    aclDataType cType;
    aclblasComputeType_t computeType;
};

struct DeviceAllocation {
    void *ptr = nullptr;
    ~DeviceAllocation()
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }
};

bool IsFp8(aclDataType type)
{
    return type == ACL_FLOAT8_E4M3FN || type == ACL_FLOAT8_E5M2;
}

bool IsValidDtypeCombination(const GroupedProblem &p)
{
    if (p.aType == ACL_FLOAT16 && p.bType == ACL_FLOAT16 && p.cType == ACL_FLOAT16) {
        return p.computeType == ACLBLAS_COMPUTE_16F || p.computeType == ACLBLAS_COMPUTE_32F;
    }
    if (p.aType == ACL_BF16 && p.bType == ACL_BF16 && p.cType == ACL_BF16) {
        return p.computeType == ACLBLAS_COMPUTE_32F;
    }
    if (p.cType != ACL_FLOAT16 || p.computeType != ACLBLAS_COMPUTE_32F ||
        !IsFp8(p.aType) || !IsFp8(p.bType)) {
        return false;
    }
    return true;
}

int GetDtypeCase(const GroupedProblem &p)
{
    if (p.aType == ACL_FLOAT16) { return GROUPED_GEMM_FP16; }
    if (p.aType == ACL_BF16) { return GROUPED_GEMM_BF16; }
    if (p.aType == ACL_FLOAT8_E4M3FN && p.bType == ACL_FLOAT8_E4M3FN) {
        return GROUPED_GEMM_FP8_E4M3_E4M3;
    }
    if (p.aType == ACL_FLOAT8_E5M2 && p.bType == ACL_FLOAT8_E5M2) {
        return GROUPED_GEMM_FP8_E5M2_E5M2;
    }
    if (p.aType == ACL_FLOAT8_E4M3FN && p.bType == ACL_FLOAT8_E5M2) {
        return GROUPED_GEMM_FP8_E4M3_E5M2;
    }
    if (p.aType == ACL_FLOAT8_E5M2 && p.bType == ACL_FLOAT8_E4M3FN) {
        return GROUPED_GEMM_FP8_E5M2_E4M3;
    }
    return -1;
}

float ReadScale(const void *scales, int index, aclblasComputeType_t computeType)
{
    if (computeType == ACLBLAS_COMPUTE_16F) {
        return blas_common::HalfToFloat(static_cast<const uint16_t *>(scales)[index]);
    }
    return static_cast<const float *>(scales)[index];
}

bool IsValidOperation(aclblasOperation_t operation)
{
    return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T || operation == ACLBLAS_OP_C;
}

bool HasRequiredGroupArrays(const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void *alphaArray,
    const int ldaArray[], const int ldbArray[], const void *betaArray,
    const int ldcArray[], const int groupSize[])
{
    return transaArray != nullptr && transbArray != nullptr && mArray != nullptr && nArray != nullptr &&
        kArray != nullptr && alphaArray != nullptr && ldaArray != nullptr && ldbArray != nullptr &&
        betaArray != nullptr && ldcArray != nullptr && groupSize != nullptr;
}

aclblasStatus_t ValidateOneGroup(int index,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[],
    const int ldaArray[], const int ldbArray[], const int ldcArray[], const int groupSize[])
{
    if (!IsValidOperation(transaArray[index]) || !IsValidOperation(transbArray[index])) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "invalid operation in group %d: transa=%d transb=%d",
            index, transaArray[index], transbArray[index]);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (mArray[index] < 0 || nArray[index] < 0 || kArray[index] < 0 || groupSize[index] < 0) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "invalid dimensions in group %d", index);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int minLda = transaArray[index] == ACLBLAS_OP_N ?
        std::max(1, mArray[index]) : std::max(1, kArray[index]);
    int minLdb = transbArray[index] == ACLBLAS_OP_N ?
        std::max(1, kArray[index]) : std::max(1, nArray[index]);
    int minLdc = std::max(1, mArray[index]);
    if (ldaArray[index] < minLda || ldbArray[index] < minLdb || ldcArray[index] < minLdc) {
        OP_LOGE("aclblasGemmGroupedBatchedEx",
            "invalid leading dimension in group %d: lda=%d/%d ldb=%d/%d ldc=%d/%d",
            index, ldaArray[index], minLda, ldbArray[index], minLdb, ldcArray[index], minLdc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateGroupArrays(int groupCount,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void *alphaArray,
    const int ldaArray[], const int ldbArray[], const void *betaArray,
    const int ldcArray[], const int groupSize[])
{
    if (groupCount < 0) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "invalid groupCount=%d", groupCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupCount == 0) { return ACLBLAS_STATUS_SUCCESS; }
    if (!HasRequiredGroupArrays(transaArray, transbArray, mArray, nArray, kArray, alphaArray,
        ldaArray, ldbArray, betaArray, ldcArray, groupSize)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "a per-group host array is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    for (int i = 0; i < groupCount; ++i) {
        aclblasStatus_t status = ValidateOneGroup(
            i, transaArray, transbArray, mArray, nArray, kArray, ldaArray, ldbArray, ldcArray, groupSize);
        if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

void CalculatePartition(GroupedGemmGroupData &g, uint32_t cubeCoreCount, bool fp8)
{
    const int32_t baseM = fp8 ? 32 : 128;
    const int32_t baseN = fp8 ? 16 : 128;
    int32_t mTiles = (g.m + baseM - 1) / baseM;
    int32_t nTiles = (g.n + baseN - 1) / baseN;
    int32_t bestM = 1;
    int32_t bestN = 1;
    int32_t best = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= static_cast<int32_t>(cubeCoreCount); ++mb) {
        int32_t nb = std::min(nTiles, static_cast<int32_t>(cubeCoreCount) / mb);
        nb = std::max(1, nb);
        if (mb * nb > best) {
            best = mb * nb;
            bestM = mb;
            bestN = nb;
        }
    }
    g.mBlocks = bestM;
    g.nBlocks = bestN;
    g.singleCoreM = std::min(g.m, ((g.m + bestM - 1) / bestM + baseM - 1) / baseM * baseM);
    g.singleCoreN = std::min(g.n, ((g.n + bestN - 1) / bestN + baseN - 1) / baseN * baseN);
}

bool AddWithinUint32(uint64_t &accumulator, uint64_t value)
{
    accumulator += value;
    return accumulator <= std::numeric_limits<uint32_t>::max();
}

void InitializeGroupData(GroupedGemmGroupData &g, int index,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void *alphaArray,
    const int ldaArray[], const int ldbArray[], const void *betaArray, const int ldcArray[],
    const int groupSize[], aclblasComputeType_t computeType, uint64_t problemStart, uint64_t workspaceElements)
{
    g.originalM = mArray[index];
    g.originalN = nArray[index];
    g.originalLdc = ldcArray[index];
    g.m = nArray[index];
    g.n = mArray[index];
    g.k = kArray[index];
    g.lda = ldbArray[index];
    g.ldb = ldaArray[index];
    g.ldc = mArray[index];
    g.isTransA = transbArray[index] == ACLBLAS_OP_N ? 0 : 1;
    g.isTransB = transaArray[index] == ACLBLAS_OP_N ? 0 : 1;
    g.alpha = ReadScale(alphaArray, index, computeType);
    g.beta = ReadScale(betaArray, index, computeType);
    g.batchStart = static_cast<uint32_t>(problemStart);
    g.batchCount = static_cast<uint32_t>(groupSize[index]);
    g.workspaceOffset = workspaceElements;
    g.hasGemm = (g.batchCount > 0 && g.originalM > 0 && g.originalN > 0 &&
        g.k > 0 && g.alpha != 0.0f) ? 1 : 0;
}

aclblasStatus_t UpdateTaskRanges(GroupedGemmGroupData &g, uint64_t &problemStart,
    uint64_t &cubeTaskStart, uint64_t &epilogueTaskStart, bool fp8, uint32_t cubeCoreCount)
{
    if (!AddWithinUint32(problemStart, static_cast<uint64_t>(g.batchCount))) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "problem count exceeds uint32 range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (g.originalM > 0 && g.originalN > 0) {
        CalculatePartition(g, cubeCoreCount, fp8);
    } else {
        g.mBlocks = g.nBlocks = 1;
        g.singleCoreM = g.singleCoreN = 1;
    }
    uint64_t cubeTasks = g.hasGemm ?
        static_cast<uint64_t>(g.batchCount) * g.mBlocks * g.nBlocks : 0;
    g.cubeTaskStart = static_cast<uint32_t>(cubeTaskStart);
    g.cubeTaskCount = static_cast<uint32_t>(cubeTasks);
    if (!AddWithinUint32(cubeTaskStart, cubeTasks)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "cube task count exceeds uint32 range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool needsEpilogue = g.originalM > 0 && g.originalN > 0 && g.batchCount > 0 &&
        (g.hasGemm != 0 || g.beta != 1.0f);
    uint64_t tilesPerColumn =
        (static_cast<uint64_t>(g.originalM) + EPILOGUE_TILE - 1) / EPILOGUE_TILE;
    uint64_t epilogueTasks = needsEpilogue ?
        static_cast<uint64_t>(g.batchCount) * g.originalN * tilesPerColumn : 0;
    g.epilogueTaskStart = static_cast<uint32_t>(epilogueTaskStart);
    g.epilogueTaskCount = static_cast<uint32_t>(epilogueTasks);
    if (!AddWithinUint32(epilogueTaskStart, epilogueTasks)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "epilogue task count exceeds uint32 range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t UpdateWorkspaceSize(const GroupedGemmGroupData &g, uint64_t &workspaceElements)
{
    if (g.hasGemm == 0) { return ACLBLAS_STATUS_SUCCESS; }
    uint64_t groupWorkspace = static_cast<uint64_t>(g.batchCount) * g.originalM * g.originalN;
    if (workspaceElements > std::numeric_limits<uint64_t>::max() - groupWorkspace) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    workspaceElements += groupWorkspace;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t BuildTiling(std::vector<GroupedGemmGroupData> &groups,
    GroupedGemmTilingHeader &header, uint64_t &workspaceElements,
    int groupCount, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void *alphaArray,
    const int ldaArray[], const int ldbArray[], const void *betaArray, const int ldcArray[],
    const int groupSize[], aclblasComputeType_t computeType, bool fp8, uint32_t cubeCoreCount)
{
    uint64_t problemStart = 0;
    uint64_t cubeTaskStart = 0;
    uint64_t epilogueTaskStart = 0;
    workspaceElements = 0;
    groups.resize(groupCount);
    for (int i = 0; i < groupCount; ++i) {
        GroupedGemmGroupData &g = groups[i];
        InitializeGroupData(g, i, transaArray, transbArray, mArray, nArray, kArray, alphaArray,
            ldaArray, ldbArray, betaArray, ldcArray, groupSize, computeType, problemStart, workspaceElements);
        aclblasStatus_t status =
            UpdateTaskRanges(g, problemStart, cubeTaskStart, epilogueTaskStart, fp8, cubeCoreCount);
        if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
        status = UpdateWorkspaceSize(g, workspaceElements);
        if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    }
    header.groupCount = static_cast<uint32_t>(groupCount);
    header.problemCount = static_cast<uint32_t>(problemStart);
    header.totalCubeTasks = static_cast<uint32_t>(cubeTaskStart);
    header.totalEpilogueTasks = static_cast<uint32_t>(epilogueTaskStart);
    header.epilogueTile = EPILOGUE_TILE;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AllocateAndCopyTiling(const GroupedGemmTilingHeader &header,
    const std::vector<GroupedGemmGroupData> &groups, DeviceAllocation &device)
{
    size_t bytes = sizeof(header) + groups.size() * sizeof(GroupedGemmGroupData);
    std::vector<uint8_t> host(bytes);
    const auto *headerBytes = reinterpret_cast<const uint8_t *>(&header);
    std::copy_n(headerBytes, sizeof(header), host.data());
    if (!groups.empty()) {
        const auto *groupBytes = reinterpret_cast<const uint8_t *>(groups.data());
        std::copy_n(groupBytes, groups.size() * sizeof(GroupedGemmGroupData), host.data() + sizeof(header));
    }
    if (aclrtMalloc(&device.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(device.ptr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AllocateWorkspace(uint64_t workspaceElements, DeviceAllocation &workspaceDevice)
{
    if (workspaceElements == 0) { return ACLBLAS_STATUS_SUCCESS; }
    if (workspaceElements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    size_t workspaceBytes = static_cast<size_t>(workspaceElements) * sizeof(float);
    return aclrtMalloc(&workspaceDevice.ptr, workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS ?
        ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_ALLOC_FAILED;
}

aclblasStatus_t LaunchGroupedKernels(_aclblas_handle *handle, const GroupedProblem &problem,
    const GroupedGemmTilingHeader &header, uint32_t cubeCoreCount,
    const void *const aarray[], const void *const barray[], void *const carray[],
    DeviceAllocation &workspaceDevice, DeviceAllocation &tilingDevice)
{
    int dtypeCase = GetDtypeCase(problem);
    if (header.totalCubeTasks > 0) {
        uint32_t blocks = std::min(cubeCoreCount, header.totalCubeTasks);
        // B/A are intentionally swapped: column-major GEMM is evaluated as B^T * A^T.
        gemm_grouped_batched_ex_cube_kernel_do(blocks, handle->stream,
            const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(barray)),
            const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(aarray)),
            static_cast<uint8_t *>(workspaceDevice.ptr), static_cast<uint8_t *>(tilingDevice.ptr), dtypeCase);
    }
    if (header.totalEpilogueTasks > 0) {
        uint32_t vectorCoreCount = GetAivCoreCount();
        if (vectorCoreCount == 0) { return ACLBLAS_STATUS_EXECUTION_FAILED; }
        uint32_t blocks = std::min(vectorCoreCount, header.totalEpilogueTasks);
        gemm_grouped_batched_ex_epilogue_kernel_do(blocks, handle->stream,
            const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(carray)),
            static_cast<uint8_t *>(workspaceDevice.ptr), static_cast<uint8_t *>(tilingDevice.ptr), dtypeCase);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasGemmGroupedBatchedEx(aclblasHandle_t handle,
    const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void *alphaArray,
    const void *const aarray[], aclDataType aType, const int ldaArray[],
    const void *const barray[], aclDataType bType, const int ldbArray[],
    const void *betaArray, void *const carray[], aclDataType cType, const int ldcArray[],
    int groupCount, const int groupSize[], aclblasComputeType_t computeType)
{
    auto *h = handle;
    if (h == nullptr) { return ACLBLAS_STATUS_HANDLE_IS_NULLPTR; }
    GroupedProblem problem{aType, bType, cType, computeType};
    if (!IsValidDtypeCombination(problem)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "unsupported dtype combination A=%d B=%d C=%d compute=%d",
            static_cast<int>(aType), static_cast<int>(bType), static_cast<int>(cType),
            static_cast<int>(computeType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    aclblasStatus_t status = ValidateGroupArrays(groupCount, transaArray, transbArray,
        mArray, nArray, kArray, alphaArray, ldaArray, ldbArray, betaArray, ldcArray, groupSize);
    if (status != ACLBLAS_STATUS_SUCCESS || groupCount == 0) { return status; }

    uint64_t problemCount = 0;
    for (int i = 0; i < groupCount; ++i) { problemCount += static_cast<uint32_t>(groupSize[i]); }
    if (problemCount > 0 && (aarray == nullptr || barray == nullptr || carray == nullptr)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "matrix pointer arrays must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t cubeCoreCount = GetAicCoreCount();
    if (cubeCoreCount == 0) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "failed to query cube core count");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    GroupedGemmTilingHeader header{};
    std::vector<GroupedGemmGroupData> groups;
    uint64_t workspaceElements = 0;
    status = BuildTiling(groups, header, workspaceElements, groupCount, transaArray, transbArray,
        mArray, nArray, kArray, alphaArray, ldaArray, ldbArray, betaArray, ldcArray,
        groupSize, computeType, IsFp8(aType) || IsFp8(bType), cubeCoreCount);
    if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    if (header.totalCubeTasks == 0 && header.totalEpilogueTasks == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    DeviceAllocation tilingDevice;
    status = AllocateAndCopyTiling(header, groups, tilingDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    DeviceAllocation workspaceDevice;
    status = AllocateWorkspace(workspaceElements, workspaceDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    status = LaunchGroupedKernels(
        h, problem, header, cubeCoreCount, aarray, barray, carray, workspaceDevice, tilingDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) { return status; }
    if (aclrtSynchronizeStream(h->stream) != ACL_SUCCESS) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "kernel execution failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
