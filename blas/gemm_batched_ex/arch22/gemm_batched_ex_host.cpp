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

void GemmBatchedEx34DecisionOverlap(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmBatchedEx22Tiling);
void GemmBatchedEx46FixedNT(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmBatchedEx22Tiling);
void GemmBatchedEx47FixedNT(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmBatchedEx22Tiling);

void GemmBatchedEx48Tiny(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmBatchedEx22Tiling);

namespace {
constexpr uint32_t MAX_AIV = 40;
constexpr uint32_t MAX_AIC = 20;

bool ValidOperation(aclblasOperation_t operation)
{
    return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T || operation == ACLBLAS_OP_C;
}

aclblasStatus_t ReadScalar(const void* pointer, float& value, aclrtStream stream)
{
    bool device = false;
    const auto status = CheckPtrLocation(pointer, &device);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (device) {
        return ReadHostScalarFromDevice(static_cast<const float*>(pointer), value, stream, "aclblasGemmBatchedEx");
    }
    value = *static_cast<const float*>(pointer);
    return ACLBLAS_STATUS_SUCCESS;
}

bool CanRotateTail(const GemmBatchedEx22Tiling& t, uint32_t cores)
{
    return cores != 0 && cores <= MAX_AIC && t.transA && t.transB && t.alpha == 1.0f && t.beta == 0.0f && t.k > 0 &&
           t.k <= 96 && t.m >= 768 && t.n >= 768 && t.batches == 1;
}

void PrepareTailRotation(GemmBatchedEx22Tiling& t, uint32_t cores)
{
    if (cores == 0 || !CanRotateTail(t, cores))
        return;
    const uint64_t nr = (static_cast<uint64_t>(t.n) + 511U) / 512U;
    const uint64_t nc = (static_cast<uint64_t>(t.m) + 63U) / 64U;
    const uint64_t tasks = nr * nc;
    if (tasks == 0 || tasks > 128)
        return;
    const uint64_t rm = (static_cast<uint64_t>(t.n) + 255U) / 256U;
    const uint64_t cm = (static_cast<uint64_t>(t.m) + 127U) / 128U;
    const uint64_t rn = (static_cast<uint64_t>(t.n) + 127U) / 128U;
    const uint64_t cn = (static_cast<uint64_t>(t.m) + 255U) / 256U;
    const bool useM = rm * cm <= rn * cn;
    const uint64_t oldRows = useM ? rm : rn;
    const uint64_t oldCols = useM ? cm : cn;
    const uint64_t oldWaves = (oldRows * oldCols + cores - 1U) / cores;
    const uint64_t waves = (tasks + cores - 1U) / cores;
    const uint64_t movement = (static_cast<uint64_t>(t.n) * nc + static_cast<uint64_t>(t.m) * nr) * 100U;
    const uint64_t previousMovement =
        (static_cast<uint64_t>(t.n) * oldCols + static_cast<uint64_t>(t.m) * oldRows) * 165U;
    if (waves * 4U > oldWaves * 5U || movement > previousMovement)
        return;
    const uint32_t rotation = ((nr - 1U) * nc + tasks - (waves - 1U) * cores) % tasks;
    if (rotation == 0)
        return;
    uint64_t original[MAX_AIC]{};
    uint64_t rotated[MAX_AIC]{};
    auto weight = [&](uint64_t task) {
        const uint64_t rows = std::min<uint64_t>(512, static_cast<uint64_t>(t.n) - task / nc * 512U);
        const uint64_t cols = std::min<uint64_t>(64, static_cast<uint64_t>(t.m) - task % nc * 64U);
        return ((rows + 15U) / 16U * 16U) * ((cols + 15U) / 16U * 16U);
    };
    for (uint32_t slot = 0; slot < tasks; ++slot) {
        const uint32_t shifted = slot + rotation;
        original[slot % cores] += weight(slot);
        rotated[slot % cores] += weight(shifted < tasks ? shifted : shifted - tasks);
    }
    uint64_t oldMax = 0;
    uint64_t newMax = 0;
    for (uint32_t core = 0; core < cores; ++core) {
        oldMax = std::max(oldMax, original[core]);
        newMax = std::max(newMax, rotated[core]);
    }
    if (newMax * 100U <= oldMax * 98U)
        t.taskRotation = rotation;
}

bool UseFixedNT5(const GemmBatchedEx22Tiling& t, uint32_t cores)
{
    if (cores == 0 || cores > MAX_AIC || t.transA || !t.transB || t.batches != 1 || t.n <= 0 || t.n > 128 ||
        t.m <= 128 || t.k < 512 || t.k >= 2048)
        return false;
    const uint64_t original =
        ((static_cast<uint64_t>(t.n) + 127U) / 128U) * ((static_cast<uint64_t>(t.m) + 127U) / 128U) * t.batches;
    if (original > 2U * cores)
        return false;
    uint64_t bestTasks = original;
    const uint64_t originalWaves = (original + cores - 1U) / cores;
    uint64_t bestWaves = originalWaves;
    uint64_t bestScore = bestWaves * 128U * 128U;
    uint32_t best = 10;
    constexpr uint32_t heights[]{128, 64, 64, 64, 32};
    constexpr uint32_t widths[]{64, 128, 64, 32, 64};
    for (uint32_t i = 0; i < 5; ++i) {
        const uint64_t height = heights[i];
        const uint64_t width = widths[i];
        const uint64_t tasks = ((static_cast<uint64_t>(t.n) + height - 1U) / height) *
                               ((static_cast<uint64_t>(t.m) + width - 1U) / width) * t.batches;
        const uint64_t waves = (tasks + cores - 1U) / cores;
        if (waves > originalWaves || waves * (height + width) > originalWaves * 256U)
            continue;
        const uint64_t score = waves * height * width;
        if (score < bestScore ||
            (score == bestScore && (waves < bestWaves || (waves == bestWaves && tasks < bestTasks)))) {
            best = i + 5U;
            bestTasks = tasks;
            bestWaves = waves;
            bestScore = score;
        }
    }
    return best == 5;
}

uint8_t* AsGm(const void* pointer) { return static_cast<uint8_t*>(const_cast<void*>(pointer)); }

using InputAddress = const void*;
using OutputAddress = void*;

struct GemmInputs {
    aclblasHandle_t handle;
    aclblasOperation_t transa;
    aclblasOperation_t transb;
    int m;
    int n;
    int k;
    const void* alpha;
    const InputAddress* Aarray;
    aclDataType Atype;
    int lda;
    const InputAddress* Barray;
    aclDataType Btype;
    int ldb;
    const void* beta;
    const OutputAddress* Carray;
    aclDataType Ctype;
    int ldc;
    int batchCount;
    aclblasComputeType_t computeType;
    aclblasGemmAlgo_t algo;
};

aclblasStatus_t ValidateGemmShape(const GemmInputs& p)
{
    if (p.handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (!ValidOperation(p.transa) || !ValidOperation(p.transb))
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (p.m < 0 || p.n < 0 || p.k < 0 || p.batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const bool ta = p.transa != ACLBLAS_OP_N;
    const bool tb = p.transb != ACLBLAS_OP_N;
    if (p.lda < std::max(1, ta ? p.k : p.m) || p.ldb < std::max(1, tb ? p.n : p.k) || p.ldc < std::max(1, p.m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateGemmOperands(const GemmInputs& p)
{
    if (p.alpha == nullptr || p.beta == nullptr || (p.batchCount > 0 && (!p.Aarray || !p.Barray || !p.Carray))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (p.Atype != ACL_FLOAT || p.Btype != ACL_FLOAT || p.Ctype != ACL_FLOAT || p.computeType != ACLBLAS_COMPUTE_32F ||
        p.algo != ACLBLAS_GEMM_DEFAULT) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct GemmExecution {
    const GemmInputs& p;
    GemmBatchedEx22Tiling& t;
    uint32_t vectors;
    uint32_t cubes = 0;
    uint8_t* temporary = nullptr;
    size_t scratchBytes = 0;
};

bool IsWholeTiny(const GemmBatchedEx22Tiling& t)
{
    return t.alpha == 1.0f && t.beta == 0.0f && !t.transA && !t.transB && t.m <= 32 && t.n <= 32 && t.k > 4 &&
           t.k <= 32 && static_cast<uint64_t>(t.m) * t.n <= 4096U / static_cast<uint32_t>(t.k) &&
           static_cast<uint64_t>(t.lda) <= UINT32_MAX / 4U && static_cast<uint64_t>(t.ldb) <= UINT32_MAX / 4U &&
           static_cast<uint64_t>(t.ldc) <= UINT32_MAX / 4U;
}

void LaunchWholeTiny(const GemmExecution& x)
{
    const auto& t = x.t;
    const auto blocks = std::min(x.vectors, static_cast<uint32_t>(x.p.batchCount));
    if (t.m == t.n && t.n == t.k && (t.m == 8 || t.m == 16)) {
        GemmBatchedEx48Tiny(blocks, x.p.handle->stream, AsGm(x.p.Aarray), AsGm(x.p.Barray), AsGm(x.p.Carray), t);
        return;
    }
    GemmBatchedEx22Vector(
        blocks, x.p.handle->stream, AsGm(x.p.Aarray), AsGm(x.p.Barray), AsGm(x.p.Carray), nullptr, t, false);
}

bool IsGeneralVector(const GemmBatchedEx22Tiling& t)
{
    const uint64_t elements = static_cast<uint64_t>(t.m) * t.n;
    return t.k == 0 || t.alpha == 0.0f || t.alpha != 1.0f || t.beta != 0.0f ||
           elements <= 4096U / static_cast<uint32_t>(std::max(1, t.k));
}

void LaunchGeneralVector(const GemmExecution& x)
{
    const auto& t = x.t;
    const bool preload = t.alpha == 1.0f && t.beta == 0.0f && !t.transA && t.m > 0 && t.m <= 32 && t.k > 0 &&
                         t.k <= 32 && static_cast<uint64_t>(t.lda) <= UINT32_MAX / 4U &&
                         (!t.transB || static_cast<uint64_t>(t.ldb) <= UINT32_MAX / 4U);
    const uint32_t blocks =
        preload ? static_cast<uint32_t>(std::min<uint64_t>(x.vectors, static_cast<uint64_t>(t.n) * x.p.batchCount)) :
                  x.vectors;
    GemmBatchedEx22Vector(
        blocks, x.p.handle->stream, AsGm(x.p.Aarray), AsGm(x.p.Barray), AsGm(x.p.Carray), nullptr, t, false);
}

size_t PrepareScratch(GemmBatchedEx22Tiling& t, bool split, bool direct, uint64_t paddedM)
{
    size_t scratch = 0;
    if (split) {
        t.splitK = 512;
        t.parts = (static_cast<uint64_t>(t.k) + t.splitK - 1) / t.splitK;
        scratch = static_cast<size_t>(t.parts) * 128U * 128U * sizeof(float);
    } else if (!direct) {
        scratch = paddedM * t.n * sizeof(float);
        t.outputLd = static_cast<int32_t>(paddedM);
        t.temporaryOutput = true;
    }
    return scratch;
}

bool LaunchAdaptivePolicy(GemmExecution& x, bool direct)
{
    const auto& p = x.p;
    if (direct && p.batchCount <= 4 && x.vectors == 2U * x.cubes) {
        if (UseFixedNT5(x.t, x.cubes)) {
            GemmBatchedEx47FixedNT(x.cubes, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), AsGm(p.Carray), x.t);
            return true;
        }
        GemmBatchedEx34DecisionOverlap(x.cubes, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), AsGm(p.Carray), x.t);
        return true;
    }
    if (p.batchCount <= 4) {
        GemmBatchedEx22PolicyFused(x.vectors, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), AsGm(p.Carray), x.t);
    } else {
        GemmBatchedEx22Policy(x.vectors, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), x.t, false);
        GemmBatchedEx22Policy(x.vectors, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), x.t, true);
    }
    return false;
}

aclblasStatus_t PrepareGemmWorkspace(GemmExecution& x, size_t scratch, bool direct, bool& completed)
{
    const bool adaptive = x.t.k >= 512;
    const size_t policyBytes = adaptive ? static_cast<size_t>(x.p.batchCount) * (64U + x.vectors * 32U) : 0;
    const size_t alignmentReserve = adaptive ? 63U : 0;
    if (scratch > std::numeric_limits<size_t>::max() - policyBytes - alignmentReserve)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (!scratch && !adaptive)
        return ACLBLAS_STATUS_SUCCESS;
    const auto status = EnsureDefaultWorkspace(x.p.handle, scratch + policyBytes + alignmentReserve);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    auto* workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(x.p.handle));
    const size_t available = GetEffectiveWorkspaceSize(x.p.handle);
    size_t prefix = 0;
    if (adaptive) {
        const uint64_t base = reinterpret_cast<uint64_t>(workspace);
        const uint64_t aligned = (base + 63U) / 64U * 64U;
        prefix = static_cast<size_t>(aligned - base) + policyBytes;
        if (prefix > available || scratch > available - prefix)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        x.t.policy = aligned;
        x.t.policyCores = x.vectors;
        completed = LaunchAdaptivePolicy(x, direct);
        if (completed)
            return ACLBLAS_STATUS_SUCCESS;
    }
    x.temporary = workspace + prefix;
    x.scratchBytes = available - prefix;
    return ACLBLAS_STATUS_SUCCESS;
}

void LaunchProductPair(GemmExecution& x)
{
    const auto& p = x.p;
    GemmBatchedEx22Cube(x.cubes, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), AsGm(p.Carray), x.temporary, x.t);
    GemmBatchedEx22Vector(
        x.vectors, p.handle->stream, AsGm(p.Aarray), AsGm(p.Barray), AsGm(p.Carray), x.temporary, x.t, true);
}

aclblasStatus_t LaunchSplitProducts(GemmExecution& x, size_t scratch)
{
    if (scratch == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const uint64_t capacity = x.scratchBytes / scratch;
    if (capacity == 0)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    const uint64_t tiles = ((static_cast<uint64_t>(x.t.m) + 127U) / 128U) *
                           ((static_cast<uint64_t>(x.t.n) + 127U) / 128U) * x.p.batchCount;
    for (uint64_t first = 0; first < tiles; first += x.t.taskCount) {
        x.t.firstTask = first;
        x.t.taskCount =
            static_cast<uint32_t>(std::min<uint64_t>(std::min<uint64_t>(capacity, UINT32_MAX), tiles - first));
        LaunchProductPair(x);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchTemporaryProducts(GemmExecution& x, size_t scratch)
{
    if (scratch == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    const uint64_t capacity = std::min<uint64_t>(x.p.batchCount, x.scratchBytes / scratch);
    if (capacity == 0)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    for (uint32_t first = 0; first < static_cast<uint32_t>(x.p.batchCount); first += x.t.batches) {
        x.t.firstBatch = first;
        x.t.batches = std::min<uint32_t>(capacity, static_cast<uint32_t>(x.p.batchCount) - first);
        LaunchProductPair(x);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchLargeGemm(GemmExecution& x)
{
    x.cubes = std::min(MAX_AIC, GetAicCoreCount());
    if (x.cubes == 0)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    PrepareTailRotation(x.t, x.cubes);
    const bool adaptive = x.t.k >= 512;
    const bool split = x.t.k >= 2048;
    const bool direct = !split && x.t.beta == 0.0f;
    const uint64_t paddedM = (static_cast<uint64_t>(x.t.m) + 7U) / 8U * 8U;
    if (paddedM > INT32_MAX || paddedM * x.t.n > std::numeric_limits<size_t>::max() / sizeof(float))
        return ACLBLAS_STATUS_INVALID_VALUE;
    const size_t scratch = PrepareScratch(x.t, split, direct, paddedM);
    bool completed = false;
    auto status = PrepareGemmWorkspace(x, scratch, direct, completed);
    if (status != ACLBLAS_STATUS_SUCCESS || completed)
        return status;
    if (split) {
        status = LaunchSplitProducts(x, scratch);
    } else if (direct) {
        GemmBatchedEx22Cube(
            x.cubes, x.p.handle->stream, AsGm(x.p.Aarray), AsGm(x.p.Barray), AsGm(x.p.Carray), nullptr, x.t);
    } else {
        status = LaunchTemporaryProducts(x, scratch);
    }
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (adaptive && x.p.batchCount > 4) {
        x.t.firstBatch = 0;
        x.t.batches = static_cast<uint32_t>(x.p.batchCount);
        GemmBatchedEx22Reference(
            x.vectors, x.p.handle->stream, AsGm(x.p.Aarray), AsGm(x.p.Barray), AsGm(x.p.Carray), x.t);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchGemm(GemmExecution& x)
{
    if (x.vectors == 0)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    if (IsWholeTiny(x.t)) {
        LaunchWholeTiny(x);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (IsGeneralVector(x.t)) {
        LaunchGeneralVector(x);
        return ACLBLAS_STATUS_SUCCESS;
    }
    return LaunchLargeGemm(x);
}

} // namespace

aclblasStatus_t aclblasGemmBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* const Aarray[], aclDataType Atype, int lda, const void* const Barray[],
    aclDataType Btype, int ldb, const void* beta, void* const Carray[], aclDataType Ctype, int ldc, int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    const GemmInputs p{handle, transa, transb, m,    n,      k,     alpha, Aarray,     Atype,       lda,
                       Barray, Btype,  ldb,    beta, Carray, Ctype, ldc,   batchCount, computeType, algo};
    auto status = ValidateGemmShape(p);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ValidateGemmOperands(p);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    // Parameter validation precedes every no-op, including pointer checks.
    if (m == 0 || n == 0 || batchCount == 0)
        return ACLBLAS_STATUS_SUCCESS;
    float alphaValue = 0;
    float betaValue = 0;
    status = ReadScalar(alpha, alphaValue, handle->stream);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ReadScalar(beta, betaValue, handle->stream);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if ((k == 0 || alphaValue == 0.0f) && betaValue == 1.0f)
        return ACLBLAS_STATUS_SUCCESS;
    const uint64_t elements = static_cast<uint64_t>(m) * n;
    if (elements > std::numeric_limits<uint64_t>::max() / static_cast<uint32_t>(batchCount))
        return ACLBLAS_STATUS_INVALID_VALUE;
    GemmBatchedEx22Tiling t{
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        transa != ACLBLAS_OP_N,
        transb != ACLBLAS_OP_N,
        ldc,
        0,
        static_cast<uint32_t>(batchCount),
        alphaValue,
        betaValue,
        false};
    GemmExecution execution{p, t, std::min(MAX_AIV, GetAivCoreCount())};
    return LaunchGemm(execution);
}
