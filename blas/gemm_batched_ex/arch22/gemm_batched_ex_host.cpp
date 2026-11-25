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
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_batched_ex_tiling.h"

namespace {
constexpr const char* OP = "aclblasGemmBatchedEx";
constexpr uint64_t PACK_LIMIT_BYTES = 256ULL * 1024 * 1024;
struct Problem {
    aclblasOperation_t ta, tb;
    int m, n, k, lda, ldb, ldc, batch;
    aclDataType aType, bType, cType;
    aclblasComputeType_t compute;
    aclblasGemmAlgo_t algo;
};

aclblasStatus_t Validate(
    const Problem& p, aclblasHandle_t handle, const void* alpha, const void* beta, const void* a, const void* b,
    const void* c)
{
    CHECK_RET(handle, OP_LOGE(OP, "handle is null"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(alpha && beta, OP_LOGE(OP, "alpha or beta is null"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(p.ta >= ACLBLAS_OP_N && p.ta <= ACLBLAS_OP_C && p.tb >= ACLBLAS_OP_N && p.tb <= ACLBLAS_OP_C,
              OP_LOGE(OP, "invalid transa=%d transb=%d", p.ta, p.tb);
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(p.m >= 0 && p.n >= 0 && p.k >= 0 && p.batch >= 0,
              OP_LOGE(OP, "invalid m=%d n=%d k=%d batchCount=%d", p.m, p.n, p.k, p.batch);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(p.lda >= std::max(1, p.ta == ACLBLAS_OP_N ? p.m : p.k) &&
                  p.ldb >= std::max(1, p.tb == ACLBLAS_OP_N ? p.k : p.n) && p.ldc >= std::max(1, p.m),
              OP_LOGE(OP, "invalid lda=%d ldb=%d ldc=%d", p.lda, p.ldb, p.ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(p.batch == 0 || (a && b && c), OP_LOGE(OP, "null pointer array, batchCount=%d", p.batch);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        p.aType == ACL_FLOAT && p.bType == ACL_FLOAT && p.cType == ACL_FLOAT && p.compute == ACLBLAS_COMPUTE_32F &&
            p.algo == ACLBLAS_GEMM_DEFAULT,
        OP_LOGE(OP, "unsupported types=%d/%d/%d compute=%d algo=%d", p.aType, p.bType, p.cType, p.compute, p.algo);
        return ACLBLAS_STATUS_NOT_SUPPORTED);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t Scalar(const void* ptr, float& value, int32_t& device, uint64_t& address)
{
    bool isDevice = false;
    aclblasStatus_t status = CheckPtrLocation(ptr, &isDevice);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    device = isDevice;
    address = isDevice ? reinterpret_cast<uint64_t>(ptr) : 0;
    if (!isDevice)
        value = *static_cast<const float*>(ptr);
    return ACLBLAS_STATUS_SUCCESS;
}

bool PreferDirectInputs(const GemmBatchedExArch22Tiling& t)
{
    // Short NN reductions avoid a standalone input pack when direct ND tiles are cheaper.
    return (!t.transA && !t.transB && t.k > 128 && t.k <= 256 && t.route == GemmBatchedExRoute::Regular) ||
           (t.k <= 64 && t.n <= 512 && t.ldc % 8 != 0) || t.temporary ||
           (t.k % 128 == 0 && t.route == GemmBatchedExRoute::Regular) || static_cast<uint64_t>(t.m) * t.n < 16384 ||
           t.k <= 32;
}

// Keep the public layouts at the boundary; align the Cube reduction once per batch.
uint64_t PackInputs(GemmBatchedExArch22Tiling& t, uint64_t capacity)
{
    if (t.lda > UINT32_MAX / sizeof(float) || t.ldb > UINT32_MAX / sizeof(float))
        return 0;
    if (PreferDirectInputs(t))
        return 0;
    int64_t alignment = t.transA && !t.transB ? 8 : 32;
    int64_t reduction = (t.k + alignment - 1) / alignment * alignment;
    if (reduction > INT32_MAX)
        return 0;
    int64_t lda = t.transA ? reduction : (t.m + 15LL) / 16 * 16;
    int64_t ldb = t.transB ? (t.n + 15LL) / 16 * 16 : reduction;
    if (lda > INT32_MAX || ldb > INT32_MAX)
        return 0;
    uint64_t a = lda * (t.transA ? (t.m + 15LL) / 16 * 16 : reduction);
    uint64_t b = ldb * (t.transB ? reduction : (t.n + 15LL) / 16 * 16);
    if (a + b > std::min(PACK_LIMIT_BYTES, capacity) / sizeof(float) / t.batchCount)
        return 0;
    t.packed = 1;
    t.originalK = t.k;
    t.originalLda = t.lda;
    t.originalLdb = t.ldb;
    t.k = static_cast<int32_t>(reduction);
    t.lda = static_cast<int32_t>(lda);
    t.ldb = static_cast<int32_t>(ldb);
    t.packAElements = a;
    t.packBElements = b;
    t.productOffset = (a + b) * t.batchCount;
    t.productLd = static_cast<int32_t>((t.m + 15LL) / 16 * 16);
    t.temporary = 0;
    return t.productOffset * sizeof(float);
}

GemmBatchedExArch22Tiling InitialTiling(const Problem& p)
{
    GemmBatchedExArch22Tiling t{};
    const auto [transa, transb, m, n, k, lda, ldb, ldc, batchCount, aType, bType, cType, compute, algo] = p;
    t.m = m;
    t.n = n;
    t.k = k;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.batchCount = batchCount;
    t.transA = transa != ACLBLAS_OP_N;
    t.transB = transb != ACLBLAS_OP_N;
    return t;
}

uint64_t PlanRowStrips(uint32_t cubeCores, GemmBatchedExArch22Tiling& t, uint32_t& stripHeight)
{
    t.productLd = (t.m + 127LL) / 128 * 128;
    uint32_t budget = cubeCores / static_cast<uint32_t>(t.batchCount);
    if (budget == 0)
        budget = 1;
    if (t.transA && !t.transB) {
        t.productLd = (t.m + 255LL) / 256 * 128;
        budget /= 2;
        if (budget == 0)
            budget = 1;
    }
    const uint32_t alignment = t.transA || t.transB ? 16 : 2;
    stripHeight = std::min<uint64_t>(128, ((t.n + budget - 1ULL) / budget + alignment - 1) / alignment * alignment);
    if (static_cast<uint64_t>(stripHeight) * t.productLd > 192 * 512) {
        t.route = GemmBatchedExRoute::MacroRing;
        return 0;
    }
    return static_cast<uint64_t>(cubeCores) * 2 * ((stripHeight + 15) / 16 * 16) * t.productLd * sizeof(float);
}

void SelectPackedRoute(const Problem& p, bool scaleOnly, GemmBatchedExArch22Tiling& t)
{
    const auto [ta, tb, m, n, k, lda, ldb, ldc, batchCount, aType, bType, cType, compute, algo] = p;
    t.route = !scaleOnly && !t.temporary && m >= 128 && n >= 128 && ldc <= UINT32_MAX / sizeof(float) && k <= 128 &&
                      static_cast<uint64_t>(m) * n >= 32768 ?
                  GemmBatchedExRoute::MacroRing :
                  GemmBatchedExRoute::Regular;
    if (k <= 64 && n <= 512 && ldc % 8 != 0)
        t.route = GemmBatchedExRoute::Regular;
    if (t.route == GemmBatchedExRoute::MacroRing &&
        ((k > 64 && n <= 512 && ldc % 8 == 0) || (k <= 64 && t.transA && t.transB)))
        t.route = GemmBatchedExRoute::RowStrips;
}

uint64_t PlanWorkspace(
    const Problem& p, bool scaleOnly, uint32_t cubeCores, GemmBatchedExArch22Tiling& t, uint32_t& stripHeight,
    uint64_t capacity)
{
    const auto [ta, tb, m, n, k, lda, ldb, ldc, batchCount, aType, bType, cType, compute, algo] = p;
    t.temporary = !scaleOnly && (t.alphaDevice || t.betaDevice || t.alpha != 1.0f || t.beta != 0.0f);
    // General scalars require SGEMM's K order: scaling a reordered Cube sum
    // amplifies its rounding error, especially around cancellation.
    if (t.temporary) {
        t.route = GemmBatchedExRoute::ReferenceOrder;
        return 0;
    }
    // Small reductions also use the vector path to preserve non-finite BLAS semantics.
    if (!scaleOnly && ((static_cast<uint64_t>(m) * n <= 8192ULL && k >= 512) || (m <= 32 && n <= 32 && k <= 32))) {
        t.route = GemmBatchedExRoute::ExactSmall;
        t.productLd = m;
        return 0;
    }
    uint64_t workspaceBytes = t.temporary ? static_cast<uint64_t>(m) * n * batchCount * sizeof(float) : 0;
    t.productLd = m;
    SelectPackedRoute(p, scaleOnly, t);
    workspaceBytes = std::max(workspaceBytes, scaleOnly ? 0 : PackInputs(t, capacity));
    t.route = t.packed ? t.route : GemmBatchedExRoute::Regular;

    if (t.route == GemmBatchedExRoute::RowStrips)
        workspaceBytes += PlanRowStrips(cubeCores, t, stripHeight);
    if (t.route == GemmBatchedExRoute::MacroRing)
        workspaceBytes += static_cast<uint64_t>(cubeCores) * 2 * 128 * 512 * sizeof(float);
    return workspaceBytes;
}

// Packing is optional. A caller-owned workspace never grows or changes ownership.
uint64_t PlanAvailableWorkspace(
    const Problem& p, bool scaleOnly, uint32_t cubeCores, uint32_t vectorCores, GemmBatchedExArch22Tiling& t,
    uint32_t& stripHeight, uint64_t capacity)
{
    const auto original = t;
    const bool inspect = !scaleOnly && p.k >= 512 && static_cast<uint64_t>(p.m) * p.n > 8192;
    const uint64_t policyBytes = inspect ? static_cast<uint64_t>(p.batch) * (64 + vectorCores * 32ULL) : 0;
    if (policyBytes > capacity) {
        t.route = GemmBatchedExRoute::ReferenceOrder;
        return 0;
    }
    const uint64_t available = (capacity - policyBytes) / 64 * 64;
    uint64_t required = PlanWorkspace(p, scaleOnly, cubeCores, t, stripHeight, available);
    if (required > available) {
        t = original;
        stripHeight = 0;
        required = PlanWorkspace(p, scaleOnly, cubeCores, t, stripHeight, 0);
    }
    if (required > available) {
        t = original;
        t.route = GemmBatchedExRoute::ReferenceOrder;
        return 0;
    }
    if (t.route == GemmBatchedExRoute::ReferenceOrder)
        return 0;
    if (inspect) {
        required = (required + 63) / 64 * 64;
        // Store the byte offset here; the launch plan replaces it with the allocated address.
        t.policy = required;
        return required + policyBytes;
    }
    return required;
}

void PlanCubeTiles(const Problem& p, uint32_t cubeCores, uint32_t stripHeight, GemmBatchedExArch22Tiling& t)
{
    const auto [ta, tb, m, n, k, lda, ldb, ldc, batchCount, aType, bType, cType, compute, algo] = p;
    t.tileM = std::min<int64_t>(256, ((n + 15LL) / 16) * 16);
    t.tileN = std::min<int64_t>(256, ((m + 15LL) / 16) * 16);
    uint64_t initialTasks = static_cast<uint64_t>(batchCount) * ((n + 255LL) / 256) * ((m + 255LL) / 256);
    if (initialTasks < cubeCores && k >= 64) {
        t.tileM = std::min(t.tileM, m * static_cast<int64_t>(n) < 32768 ? 32 : 64);
        t.tileN = std::min(t.tileN, m * static_cast<int64_t>(n) < 32768 ? 64 : 128);
    }
    if (k <= 128 && static_cast<uint64_t>(m) * n >= 32768) {
        t.tileM = std::min(t.tileM, 128);
        t.tileN = std::min(t.tileN, 128);
        if (m >= 128 && n >= 128) {
            t.tileM = 128;
            t.tileN = 128;
        }
    }
    if (t.route == GemmBatchedExRoute::MacroRing) {
        t.tileN = 512;
        uint64_t columns = (m + 511ULL) / 512;
        uint64_t targetRows = (2 * cubeCores + static_cast<uint64_t>(batchCount) * columns - 1) /
                              (static_cast<uint64_t>(batchCount) * columns);
        t.tileM = std::min<uint64_t>(128, ((n + targetRows - 1ULL) / targetRows + 15) / 16 * 16);
    }
    if (!t.packed) {
        t.tileM = std::min<int64_t>(128, (n + 15LL) / 16 * 16);
        t.tileN = std::min<int64_t>(128, (m + 15LL) / 16 * 16);
        // Re-apply the small-shape split after the direct-tile cap.
        uint64_t directTasks =
            static_cast<uint64_t>(batchCount) * ((n + t.tileM - 1LL) / t.tileM) * ((m + t.tileN - 1LL) / t.tileN);
        if (directTasks < cubeCores && k >= 64) {
            t.tileM = std::min<int64_t>(t.tileM, m * static_cast<int64_t>(n) < 32768 ? 32 : 64);
            t.tileN = std::min<int64_t>(t.tileN, m * static_cast<int64_t>(n) < 32768 ? 64 : 128);
        }
    }
    if (t.route == GemmBatchedExRoute::RowStrips) {
        t.tileM = stripHeight;
        t.tileN = t.productLd;
    }
}

uint32_t PlanTiles(
    const Problem& p, bool scaleOnly, uint32_t cubeCores, uint32_t vectorCores, uint32_t stripHeight,
    GemmBatchedExArch22Tiling& t)
{
    const auto [ta, tb, m, n, k, lda, ldb, ldc, batchCount, aType, bType, cType, compute, algo] = p;
    if (t.route == GemmBatchedExRoute::ExactSmall || t.route == GemmBatchedExRoute::ReferenceOrder) {
        t.tasks = static_cast<uint64_t>(batchCount) * n * ((m + 63LL) / 64);
        t.vectorBlocks = static_cast<uint32_t>(std::min<uint64_t>(vectorCores, t.tasks));
        return t.vectorBlocks;
    }
    t.vectorBlocks = vectorCores;
    uint32_t blocks = t.vectorBlocks;
    if (!scaleOnly) {
        PlanCubeTiles(p, cubeCores, stripHeight, t);
        t.tasks = static_cast<uint64_t>(batchCount) * ((n + t.tileM - 1LL) / t.tileM) * ((m + t.tileN - 1LL) / t.tileN);

        blocks = static_cast<uint32_t>(std::min<uint64_t>(cubeCores, t.tasks));
    }
    if (t.route != GemmBatchedExRoute::Regular) {
        if (t.route == GemmBatchedExRoute::RowStrips && t.transB)
            blocks = cubeCores;
        t.vectorBlocks = 2 * blocks;
    }
    return blocks;
}

} // namespace

extern "C" aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* const Aarray[], aclDataType Atype, int lda, const void* const Barray[],
    aclDataType Btype, int ldb, const void* beta, void* const Carray[], aclDataType Ctype, int ldc, int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    Problem p{transa, transb, m, n, k, lda, ldb, ldc, batchCount, Atype, Btype, Ctype, computeType, algo};
    aclblasStatus_t status = Validate(p, handle, alpha, beta, Aarray, Barray, Carray);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0 || batchCount == 0)
        return status;
    auto t = InitialTiling(p);
    status = Scalar(alpha, t.alpha, t.alphaDevice, t.alphaAddress);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = Scalar(beta, t.beta, t.betaDevice, t.betaAddress);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    bool scaleOnly = k == 0 || (!t.alphaDevice && t.alpha == 0.0f);
    if (scaleOnly && !t.betaDevice && t.beta == 1.0f)
        return ACLBLAS_STATUS_SUCCESS;
    const uint64_t elements = static_cast<uint64_t>(m) * n;
    CHECK_RET(elements <= std::numeric_limits<uint64_t>::max() / sizeof(float) / batchCount,
              OP_LOGE(OP, "matrix extent overflows the address space");
              return ACLBLAS_STATUS_INVALID_VALUE);
    const uint32_t cubeCores = GetAicCoreCount();
    const uint32_t vectorCores = GetAivCoreCount();
    CHECK_RET(vectorCores > 0 && (scaleOnly || cubeCores > 0), OP_LOGE(OP, "no available cores");
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint32_t stripHeight = 0;
    const uint64_t capacity = handle->workspace_owner == AclblasWorkspaceOwner::User ?
                                  GetEffectiveWorkspaceSize(handle) :
                                  ACLBLAS_MAX_WORKSPACE_SIZE;
    uint64_t workspaceBytes = PlanAvailableWorkspace(p, scaleOnly, cubeCores, vectorCores, t, stripHeight, capacity);
    status = EnsureDefaultWorkspace(handle, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (!scaleOnly && k >= 512 && elements > 8192 && t.route != GemmBatchedExRoute::ReferenceOrder)
        t.policy += reinterpret_cast<uint64_t>(handle->workspace);
    uint32_t blocks = PlanTiles(p, scaleOnly, cubeCores, vectorCores, stripHeight, t);
    CHECK_RET(blocks > 0, OP_LOGE(OP, "no available cores"); return ACLBLAS_STATUS_EXECUTION_FAILED);
    const int32_t launchStatus =
        GemmBatchedExArch22Launch(blocks, handle->stream, Aarray, Barray, Carray, handle->workspace, t, scaleOnly);
    CHECK_RET(launchStatus == ACL_SUCCESS, OP_LOGE(OP, "kernel launch failed, ret=%d", launchStatus);
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    return ACLBLAS_STATUS_SUCCESS;
}
