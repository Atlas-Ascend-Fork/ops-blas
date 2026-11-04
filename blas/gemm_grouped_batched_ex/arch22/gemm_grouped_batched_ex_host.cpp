/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <limits>
#include <vector>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "gemm_grouped_batched_ex_kernel.h"

namespace {
bool ValidOp(aclblasOperation_t value)
{
    return value == ACLBLAS_OP_N || value == ACLBLAS_OP_T || value == ACLBLAS_OP_C;
}
bool AddChecked(uint64_t& sum, uint64_t value)
{
    if (value > std::numeric_limits<uint64_t>::max() - sum)
        return false;
    sum += value;
    return true;
}
bool MulChecked(uint64_t a, uint64_t b, uint64_t& value)
{
    if (b != 0 && a > std::numeric_limits<uint64_t>::max() / b)
        return false;
    value = a * b;
    return true;
}
template <uint64_t Divisor>
uint64_t DivideUp(uint64_t value)
{
    static_assert(Divisor > 0, "DivideUp requires a positive divisor");
    return value / Divisor + (value % Divisor != 0);
}

bool ValidShape(aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, int size, int lda, int ldb, int ldc)
{
    return m >= 0 && n >= 0 && k >= 0 && size >= 0 && lda >= std::max(1, ta == ACLBLAS_OP_N ? m : k) &&
           ldb >= std::max(1, tb == ACLBLAS_OP_N ? k : n) && ldc >= std::max(1, m);
}

aclblasStatus_t Validate(
    int count, const aclblasOperation_t* ta, const aclblasOperation_t* tb, const int* m, const int* n, const int* k,
    const void* alpha, const int* lda, const int* ldb, const void* beta, const int* ldc, const int* size,
    uint64_t& instances)
{
    if (count < 0) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "invalid groupCount=%d", count);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (count == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (!ta || !tb || !m || !n || !k || !alpha || !lda || !ldb || !beta || !ldc || !size) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "a per-group host array or scalar array is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    for (int g = 0; g < count; ++g) {
        if (!ValidOp(ta[g]) || !ValidOp(tb[g])) {
            OP_LOGE(
                "aclblasGemmGroupedBatchedEx", "invalid operation in group %d: transa=%d transb=%d", g, ta[g], tb[g]);
            return ACLBLAS_STATUS_INVALID_ENUM;
        }
        if (!ValidShape(ta[g], tb[g], m[g], n[g], k[g], size[g], lda[g], ldb[g], ldc[g])) {
            OP_LOGE(
                "aclblasGemmGroupedBatchedEx", "invalid shape in group %d: m=%d n=%d k=%d size=%d lda=%d ldb=%d ldc=%d",
                g, m[g], n[g], k[g], size[g], lda[g], ldb[g], ldc[g]);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (!AddChecked(instances, static_cast<uint64_t>(size[g]))) {
            OP_LOGE("aclblasGemmGroupedBatchedEx", "instance count overflow at group %d", g);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (instances > std::numeric_limits<size_t>::max() / sizeof(void*)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "pointer table size exceeds size_t range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool IsSmallDirect(const GemmGrouped22Group& group)
{
    return group.scalarDevice == 0 && group.alpha == 1.0f && group.beta == 0.0f && group.m >= 8 && group.n >= 8 &&
           group.k >= 8 && group.m % 8 == 0 && group.ldc % 8 == 0 && group.lda == (group.transA ? group.k : group.m) &&
           group.ldb == (group.transB ? group.n : group.k);
}

void ConfigureGroup(GemmGrouped22Group& group)
{
    const uint64_t plane = static_cast<uint64_t>(group.m) * group.n;
    const bool smallDirect = IsSmallDirect(group);
    group.cube = group.k > 0 && ((group.scalarDevice & 1) || group.alpha != 0.0f) &&
                 (plane > 4096 / static_cast<uint32_t>(std::max(1, group.k)) || smallDirect);
    group.direct = group.cube && group.scalarDevice == 0 && group.alpha == 1.0f && group.beta == 0.0f &&
                   group.m % 8 == 0 && group.ldc % 8 == 0;
    group.repairSmall = smallDirect && plane <= 4096 / static_cast<uint32_t>(std::max(1, group.k));
    group.tempLd = group.direct ? group.ldc : static_cast<uint32_t>(DivideUp<8>(group.m) * 8);
    // Low-K direct outputs amortize Matmul setup over two adjacent column tiles.
    group.cubeTileN = group.direct && group.k <= 64 && group.m >= 256 && group.n >= 128 ? 256 : GEMM_GROUPED22_TILE;
}

aclblasStatus_t AppendCubeTasks(GemmGrouped22Batch& batch, GemmGrouped22Group& group, uint64_t& workspace)
{
    if (group.cube) {
        const uint64_t rowTiles =
            group.cubeTileN == 256 ? DivideUp<256>(group.m) : DivideUp<GEMM_GROUPED22_TILE>(group.m);
        uint64_t tiles = rowTiles * DivideUp<GEMM_GROUPED22_TILE>(group.n);
        if (!MulChecked(tiles, group.instances, group.cubeTasks) || !AddChecked(batch.cubeTasks, group.cubeTasks)) {
            OP_LOGE("aclblasGemmGroupedBatchedEx", "cube task count overflow");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (!group.direct) {
            uint64_t elements = static_cast<uint64_t>(group.tempLd) * group.n;
            if (!MulChecked(elements, group.instances, elements)) {
                OP_LOGE("aclblasGemmGroupedBatchedEx", "workspace element count overflow");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            group.workspaceOffset = workspace;
            if (!AddChecked(workspace, elements)) {
                OP_LOGE("aclblasGemmGroupedBatchedEx", "accumulated workspace size overflow");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t AppendVectorTasks(GemmGrouped22Batch& batch, GemmGrouped22Group& group)
{
    if (!group.direct || group.repairSmall) {
        uint64_t jobs = group.repairSmall ? 1 : DivideUp<GEMM_GROUPED22_TILE>(group.m) * group.n;
        if (!MulChecked(jobs, group.instances, group.vectorTasks) ||
            !AddChecked(batch.vectorTasks, group.vectorTasks)) {
            OP_LOGE("aclblasGemmGroupedBatchedEx", "vector task count overflow");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t Append(GemmGrouped22Batch& batch, GemmGrouped22Group group, uint64_t& workspace)
{
    if (group.m == 0 || group.n == 0 || group.instances == 0)
        return ACLBLAS_STATUS_SUCCESS;
    ConfigureGroup(group);
    group.cubeStart = batch.cubeTasks;
    group.vectorStart = batch.vectorTasks;
    auto status = AppendCubeTasks(batch, group, workspace);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = AppendVectorTasks(batch, group);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    batch.groups[batch.count++] = group;
    return ACLBLAS_STATUS_SUCCESS;
}
struct GroupArguments {
    const aclblasOperation_t* ta;
    const aclblasOperation_t* tb;
    const int* m;
    const int* n;
    const int* k;
    const int* lda;
    const int* ldb;
    const int* ldc;
    const int* size;
    const void* alpha;
    const void* beta;
    int count;
};

bool HasWork(const GroupArguments& args)
{
    for (int g = 0; g < args.count; ++g) {
        if (args.m[g] > 0 && args.n[g] > 0 && args.size[g] > 0)
            return true;
    }
    return false;
}

GemmGrouped22Group MakeGroup(const GroupArguments& args, int first, uint64_t start, bool alphaDevice, bool betaDevice)
{
    GemmGrouped22Group group{};
    group.m = args.m[first];
    group.n = args.n[first];
    group.k = args.k[first];
    group.lda = args.lda[first];
    group.ldb = args.ldb[first];
    group.ldc = args.ldc[first];
    group.transA = args.ta[first] != ACLBLAS_OP_N;
    group.transB = args.tb[first] != ACLBLAS_OP_N;
    group.instances = args.size[first];
    group.instanceStart = start;
    group.scalarDevice = static_cast<uint32_t>(alphaDevice) | (static_cast<uint32_t>(betaDevice) << 1);
    if (alphaDevice)
        group.alphaAddress = reinterpret_cast<uint64_t>(args.alpha) + first * sizeof(float);
    else
        group.alpha = static_cast<const float*>(args.alpha)[first];
    if (betaDevice)
        group.betaAddress = reinterpret_cast<uint64_t>(args.beta) + first * sizeof(float);
    else
        group.beta = static_cast<const float*>(args.beta)[first];
    return group;
}

aclblasStatus_t BuildBatches(
    const GroupArguments& args, std::vector<GemmGrouped22Batch>& batches, uint64_t& maximumWorkspace)
{
    bool alphaDevice = false, betaDevice = false;
    auto status = CheckPtrLocation(args.alpha, &alphaDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "failed to query alpha location: status=%d", status);
        return status;
    }
    status = CheckPtrLocation(args.beta, &betaDevice);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "failed to query beta location: status=%d", status);
        return status;
    }
    uint64_t start = 0;
    for (int first = 0; first < args.count;) {
        GemmGrouped22Batch batch{};
        uint64_t workspace = 0;
        for (; first < args.count && batch.count < GEMM_GROUPED22_CAPACITY; ++first) {
            auto group = MakeGroup(args, first, start, alphaDevice, betaDevice);
            start += group.instances;
            status = Append(batch, group, workspace);
            if (status != ACLBLAS_STATUS_SUCCESS)
                return status;
        }
        if (batch.count)
            batches.push_back(batch);
        maximumWorkspace = std::max(maximumWorkspace, workspace);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchBatches(
    aclblasHandle_t handle, const void* const Aarray[], const void* const Barray[], void* const Carray[],
    const std::vector<GemmGrouped22Batch>& batches, uint64_t maximumWorkspace)
{
    if (maximumWorkspace > std::numeric_limits<size_t>::max() / sizeof(float)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "workspace byte size exceeds size_t range");
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (maximumWorkspace) {
        auto status = EnsureDefaultWorkspace(handle, static_cast<size_t>(maximumWorkspace * sizeof(float)));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE("aclblasGemmGroupedBatchedEx", "failed to allocate workspace: status=%d", status);
            return status;
        }
    }
    const uint32_t cubes = GetAicCoreCount(), vectors = GetAivCoreCount();
    auto* a = reinterpret_cast<uint8_t*>(const_cast<void*>(reinterpret_cast<const void*>(Aarray)));
    auto* b = reinterpret_cast<uint8_t*>(const_cast<void*>(reinterpret_cast<const void*>(Barray)));
    auto* c = reinterpret_cast<uint8_t*>(const_cast<void*>(reinterpret_cast<const void*>(Carray)));
    auto* workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    for (const auto& batch : batches) {
        if ((batch.cubeTasks && cubes == 0) || (batch.vectorTasks && vectors == 0)) {
            OP_LOGE("aclblasGemmGroupedBatchedEx", "core query failed: cubes=%u vectors=%u", cubes, vectors);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        if (batch.cubeTasks)
            GemmGrouped22Cube(std::min<uint64_t>(cubes, batch.cubeTasks), handle->stream, a, b, c, workspace, batch);
        if (batch.vectorTasks)
            GemmGrouped22Vector(
                std::min<uint64_t>(vectors, batch.vectorTasks), handle->stream, a, b, c, workspace, batch);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t aclblasGemmGroupedBatchedEx(
    aclblasHandle_t handle, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const void* alphaArray, const void* const Aarray[],
    aclDataType Atype, const int ldaArray[], const void* const Barray[], aclDataType Btype, const int ldbArray[],
    const void* betaArray, void* const Carray[], aclDataType Ctype, const int ldcArray[], int groupCount,
    const int groupSize[], aclblasComputeType_t computeType)
{
    if (!handle) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (Atype != ACL_FLOAT || Btype != ACL_FLOAT || Ctype != ACL_FLOAT || computeType != ACLBLAS_COMPUTE_32F) {
        OP_LOGE(
            "aclblasGemmGroupedBatchedEx", "unsupported dtype combination A=%d B=%d C=%d compute=%d", Atype, Btype,
            Ctype, computeType);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    uint64_t totalInstances = 0;
    auto status = Validate(
        groupCount, transaArray, transbArray, mArray, nArray, kArray, alphaArray, ldaArray, ldbArray, betaArray,
        ldcArray, groupSize, totalInstances);
    if (status != ACLBLAS_STATUS_SUCCESS || groupCount == 0)
        return status;
    if (totalInstances > 0 && (!Aarray || !Barray || !Carray)) {
        OP_LOGE("aclblasGemmGroupedBatchedEx", "matrix pointer arrays must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const GroupArguments args{transaArray, transbArray, mArray,    nArray,     kArray,    ldaArray,
                              ldbArray,    ldcArray,    groupSize, alphaArray, betaArray, groupCount};
    if (!HasWork(args))
        return ACLBLAS_STATUS_SUCCESS;
    std::vector<GemmGrouped22Batch> batches;
    uint64_t maximumWorkspace = 0;
    status = BuildBatches(args, batches, maximumWorkspace);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return LaunchBatches(handle, Aarray, Barray, Carray, batches, maximumWorkspace);
}
