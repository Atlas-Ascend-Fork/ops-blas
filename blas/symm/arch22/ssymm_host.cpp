/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "common/helper/host_utils.h"
#include "../ssymm_common_types.h"
#include "../ssymm_common_host.h"
#include "../ssymm_common_kernel.h"
#include "ssymm_kernel_fwd.h"

#define GM_ADDR uint8_t*

struct SsymmLaunchContext {
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    int64_t m;
    int64_t n;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    float alpha;
    float beta;
    float* hostC;
    size_t cSize;
    uint32_t numBlocks;
    void* stream;
    SsymmTilingData tiling;
    uint8_t* aDevice;
    uint8_t* bDevice;
    uint8_t* cDevice;
    uint8_t* tilingDevice;
};

SsymmWorkspaceLayout ComputeSsymmWorkspaceLayout(const SsymmExecutionPlan& plan);
aclblasStatus_t RunSsymmFallbackBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx);
aclblasStatus_t RunSsymmLeftBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx);
aclblasStatus_t RunSsymmRightBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx);

thread_local SsymmRuntimeTrace g_ssymmRuntimeTrace{};

namespace {

// 尝试执行 uint64_t 乘法并检查溢出。
// 失败时不写入结果，调用方据此标记 workspace 或 buffer 计算非法。
bool TryMulU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs == 0 || rhs == 0) {
        *out = 0;
        return true;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

// 尝试执行 uint64_t 加法并检查溢出。
// 主要用于累加 scratch、pack 和 config 等 workspace 字节数。
bool TryAddU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

// 尝试执行 size_t 乘法并检查溢出。
// host 侧 aclrtMalloc 和 aclrtMemcpy 的 size 参数统一通过该函数保护。
bool TryMulSize(size_t lhs, size_t rhs, size_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs == 0 || rhs == 0) {
        *out = 0;
        return true;
    }
    if (lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

// 计算 rows * cols 个 float 元素对应的字节数。
// 内部先算元素个数再乘 sizeof(float)，两步都做溢出检查。
bool TryComputeFloatBufferBytes(size_t rows, size_t cols, size_t* out)
{
    size_t elements = 0;
    return TryMulSize(rows, cols, &elements) && TryMulSize(elements, sizeof(float), out);
}

// 只检查一个 float buffer 的字节数是否可安全计算。
// 参数校验阶段用它提前拦截无法表示的超大输入。
bool CanComputeFloatBufferBytes(size_t rows, size_t cols)
{
    size_t bytes = 0;
    return TryComputeFloatBufferBytes(rows, cols, &bytes);
}

// 将 uint64_t workspace 规划转换为 ACL runtime 使用的 size_t。
// 如果规划无效或超过当前平台 size_t 表示范围，则返回失败。
bool TryConvertWorkspaceBytes(
    const SsymmWorkspaceLayout& layout, size_t* scratchBytes, size_t* densePackBytes, size_t* configBytes)
{
    if (!layout.valid) {
        return false;
    }
    const uint64_t maxSize = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (layout.scratchBytes > maxSize || layout.densePackBytes > maxSize || layout.configBytes > maxSize) {
        return false;
    }
    if (scratchBytes != nullptr) {
        *scratchBytes = static_cast<size_t>(layout.scratchBytes);
    }
    if (densePackBytes != nullptr) {
        *densePackBytes = static_cast<size_t>(layout.densePackBytes);
    }
    if (configBytes != nullptr) {
        *configBytes = static_cast<size_t>(layout.configBytes);
    }
    return true;
}

} // namespace

// 校验公开 aclblasSsymm 入参，并识别 m/n 为 0 的 quick return。
// 同时检查 uint32_t 下转边界和 host/device buffer 字节数是否可安全计算。
aclblasStatus_t ValidateSsymmArgs(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const float* alpha,
    const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc,
    bool* quickReturn)
{
    if (quickReturn != nullptr) {
        *quickReturn = false;
    }
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Quick return: m==0 or n==0 的情况下，不需要执行计算，直接返回成功
    // 此时允许 A/B/C 指针为 nullptr 以及 leading dimensions 为 0，因为不会访问这些数据
    if (m == 0 || n == 0) {
        if (quickReturn != nullptr) {
            *quickReturn = true;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 正常情况下（m>0 且 n>0），所有指针必须非空
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const int64_t maxU32 = static_cast<int64_t>(UINT32_MAX);
    if (m > maxU32 || n > maxU32 || aDim > maxU32 || lda > maxU32 || ldb > maxU32 || ldc > maxU32) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < aDim || ldb < n || ldc < n)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (!CanComputeFloatBufferBytes(static_cast<size_t>(aDim), static_cast<size_t>(lda)) ||
        !CanComputeFloatBufferBytes(static_cast<size_t>(m), static_cast<size_t>(ldb)) ||
        !CanComputeFloatBufferBytes(static_cast<size_t>(m), static_cast<size_t>(ldc))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 根据执行计划补齐 workspace 规划。
// 空 plan 指针直接忽略，便于测试或防御式调用复用。
void PrepareSsymmWorkspace(SsymmExecutionPlan* plan)
{
    if (plan == nullptr) {
        return;
    }
    plan->workspace = ComputeSsymmWorkspaceLayout(*plan);
}

// 启动通用 fallback backend 的薄封装。
// 保持外层调度入口稳定，实际实现集中在 RunSsymmFallbackBackendImpl。
aclblasStatus_t RunSsymmFallbackBackend(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    return RunSsymmFallbackBackendImpl(plan, ctx);
}

// 启动 LEFT cube backend 的薄封装。
// 保持外层调度入口稳定，实际实现集中在 RunSsymmLeftBackendImpl。
aclblasStatus_t RunSsymmLeftBackend(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    return RunSsymmLeftBackendImpl(plan, ctx);
}

// 启动 RIGHT cube backend 的薄封装。
// 保持外层调度入口稳定，实际实现集中在 RunSsymmRightBackendImpl。
aclblasStatus_t RunSsymmRightBackend(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    return RunSsymmRightBackendImpl(plan, ctx);
}

// 将 aclblas 的 uplo 枚举转换为 RIGHT cube 内部枚举。
// 调用前已完成入参校验，因此非 LOWER 的合法值按 UPPER 处理。
static RightCubeUplo ToRightCubeUplo(aclblasFillMode_t uplo)
{
    return (uplo == ACLBLAS_LOWER) ? RightCubeUplo::LOWER : RightCubeUplo::UPPER;
}

// 构造 RIGHT cube backend 的基础 runtime config。
// 它会先选择 chunk-local 策略，再填充 shape、ld、mask 和 alpha/beta。
static SsymmRightCubeUnifiedConfig BuildRightCubeUnifiedConfig(
    aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta)
{
    RightCubeChunkLocalPolicy policy = BuildRightCubeChunkLocalPolicy(
        ToRightCubeUplo(uplo), static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldb), static_cast<uint32_t>(ldc));
    SsymmRightCubeUnifiedConfig config{};
    config.uplo = static_cast<uint32_t>(ToRightCubeUplo(uplo));
    config.m = static_cast<uint32_t>(m);
    config.n = static_cast<uint32_t>(n);
    config.lda = static_cast<uint32_t>(lda);
    config.ldb = static_cast<uint32_t>(ldb);
    config.ldc = static_cast<uint32_t>(ldc);
    config.tm = policy.tm;
    config.tn = policy.tn;
    config.tk = policy.tk;
    config.strategy = static_cast<uint32_t>(policy.strategy);
    config.cubeChunkMask = policy.cubeChunkMask;
    config.fallbackChunkMask = policy.fallbackChunkMask;
    config.alpha = alpha;
    config.beta = beta;
    return config;
}

static std::vector<SsymmLeftCubeConfig> BuildLeftCubeRuntimeConfigs(
    aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta);

// 计算 LeftCube 路径的 workspace 布局（scratch 已提前写入 layout）。
// 遍历所有 runtime config，找出最大 pack/partial 字节数并更新 layout。
static bool ComputeLeftCubeWorkspaceLayout(const SsymmExecutionPlan& plan, SsymmWorkspaceLayout& layout)
{
    const SsymmProblemSpec& spec = plan.spec;
    std::vector<SsymmLeftCubeConfig> runtimeConfigs =
        BuildLeftCubeRuntimeConfigs(spec.uplo, spec.m, spec.n, spec.lda, spec.ldb, spec.ldc, spec.alpha, spec.beta);
    uint64_t maxPackedBytes = 0;
    uint64_t maxPartialBytes = 0;
    for (const SsymmLeftCubeConfig& config : runtimeConfigs) {
        uint64_t packedBytes = 0;
        uint64_t partialBytes = 0;
        uint64_t packedElements = 0;
        uint64_t partialElements = 0;
        if (!TryMulU64(static_cast<uint64_t>(config.rowCount), static_cast<uint64_t>(config.kCount), &packedElements) ||
            !TryMulU64(packedElements, sizeof(float), &packedBytes) ||
            !TryMulU64(static_cast<uint64_t>(config.rowCount), static_cast<uint64_t>(config.n), &partialElements) ||
            !TryMulU64(partialElements, sizeof(float), &partialBytes)) {
            return false;
        }
        if (packedBytes > maxPackedBytes)
            maxPackedBytes = packedBytes;
        if (partialBytes > maxPartialBytes)
            maxPartialBytes = partialBytes;
    }
    // packed A 区末尾需 32B 对齐，确保紧跟其后的 partial 区 GM 地址对齐。
    const uint64_t alignedMaxPackedBytes = ((maxPackedBytes + 31) / 32) * 32;
    if (!TryAddU64(alignedMaxPackedBytes, maxPartialBytes, &layout.densePackBytes)) {
        return false;
    }
    if (!TryMulU64(static_cast<uint64_t>(runtimeConfigs.size()), sizeof(SsymmLeftCubeConfig), &layout.configBytes) ||
        !TryAddU64(layout.scratchBytes, layout.densePackBytes, &layout.totalBytes) ||
        !TryAddU64(layout.totalBytes, layout.configBytes, &layout.totalBytes)) {
        return false;
    }
    return true;
}

// 计算 RightCube 路径的 workspace 布局（scratch 已提前写入 layout）。
// 按策略选择 pack 大小并计算 config 槽数。
static bool ComputeRightCubeWorkspaceLayout(const SsymmExecutionPlan& plan, SsymmWorkspaceLayout& layout)
{
    const SsymmProblemSpec& spec = plan.spec;
    SsymmRightCubeUnifiedConfig rightCubeConfig =
        BuildRightCubeUnifiedConfig(spec.uplo, spec.m, spec.n, spec.lda, spec.ldb, spec.ldc, spec.alpha, spec.beta);
    const RightCubeStrategy strategy = static_cast<RightCubeStrategy>(rightCubeConfig.strategy);
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        layout.densePackBytes = CalcRightCubeChunkWorkspaceBytes(rightCubeConfig);
        RightCubeExecuteRuntimePlan runtimePlan =
            (static_cast<RightCubeUplo>(rightCubeConfig.uplo) == RightCubeUplo::LOWER) ?
                BuildRightCubeLowerRuntimePlan(rightCubeConfig) :
                BuildRightCubeUpperRuntimePlan(rightCubeConfig);
        const uint64_t configSlotCount = static_cast<uint64_t>(runtimePlan.chunkItems.size() + 1);
        if (!TryMulU64(configSlotCount, sizeof(SsymmRightCubeUnifiedConfig), &layout.configBytes)) {
            return false;
        }
    } else {
        layout.configBytes = sizeof(SsymmRightCubeUnifiedConfig);
    }
    if (!TryAddU64(layout.scratchBytes, layout.densePackBytes, &layout.totalBytes) ||
        !TryAddU64(layout.totalBytes, layout.configBytes, &layout.totalBytes)) {
        return false;
    }
    return true;
}

// 计算当前执行计划需要的 workspace 布局。
// fallback 不需要额外空间；LEFT/RIGHT cube 分别调用对应的辅助函数计算各区域字节数。
SsymmWorkspaceLayout ComputeSsymmWorkspaceLayout(const SsymmExecutionPlan& plan)
{
    SsymmWorkspaceLayout layout{};
    const SsymmProblemSpec& spec = plan.spec;
    if (plan.backend == SsymmBackendKind::GenericFallback) {
        return layout;
    }

    uint64_t scratchElements = 0;
    if (!TryMulU64(static_cast<uint64_t>(spec.m), static_cast<uint64_t>(spec.n), &scratchElements) ||
        !TryMulU64(scratchElements, sizeof(float), &layout.scratchBytes)) {
        layout.valid = false;
        return layout;
    }

    bool ok = (plan.backend == SsymmBackendKind::LeftCube) ? ComputeLeftCubeWorkspaceLayout(plan, layout) :
                                                             ComputeRightCubeWorkspaceLayout(plan, layout);
    if (!ok) {
        layout.valid = false;
    }
    return layout;
}

// 构造一个覆盖完整 LEFT 问题的基础 cube config。
// 后续 runtime config 生成会在此基础上改写 row/k 分块范围。
static SsymmLeftCubeConfig BuildLeftCubeConfig(
    aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta)
{
    SsymmLeftCubeConfig config{};
    config.uplo = static_cast<uint32_t>((uplo == ACLBLAS_LOWER) ? RightCubeUplo::LOWER : RightCubeUplo::UPPER);
    config.m = static_cast<uint32_t>(m);
    config.n = static_cast<uint32_t>(n);
    config.lda = static_cast<uint32_t>(lda);
    config.ldb = static_cast<uint32_t>(ldb);
    config.ldc = static_cast<uint32_t>(ldc);
    config.rowBase = 0;
    config.rowCount = static_cast<uint32_t>(m);
    config.colBase = 0;
    config.colCount = static_cast<uint32_t>(n);
    config.kBase = 0;
    config.kCount = static_cast<uint32_t>(m);
    config.alpha = alpha;
    config.beta = beta;
    return config;
}

// 按 row 和 k 维把 LEFT cube 运行拆成多个 runtime config。
// 每个 config 负责一个 row block 和一个 k block，后续依次 launch 并累加。
static std::vector<SsymmLeftCubeConfig> BuildLeftCubeRuntimeConfigs(
    aclblasFillMode_t uplo, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta)
{
    // 最大 tile 尺寸：限制单次 Cube 运算的行数和 K 维度，避免 UB 溢出
    // 256 是基于硬件 Cube 单元处理能力和 UB 容量的平衡选择
    constexpr uint32_t kMaxCubeRows = 256;
    constexpr uint32_t kMaxCubeK = 256;
    std::vector<SsymmLeftCubeConfig> configs;
    for (uint32_t rowBase = 0; rowBase < static_cast<uint32_t>(m); rowBase += kMaxCubeRows) {
        const uint32_t rowCount = std::min<uint32_t>(kMaxCubeRows, static_cast<uint32_t>(m) - rowBase);
        for (uint32_t kBase = 0; kBase < static_cast<uint32_t>(m); kBase += kMaxCubeK) {
            SsymmLeftCubeConfig config = BuildLeftCubeConfig(uplo, m, n, lda, ldb, ldc, alpha, beta);
            config.rowBase = rowBase;
            config.rowCount = rowCount;
            config.kBase = kBase;
            config.kCount = std::min<uint32_t>(kMaxCubeK, static_cast<uint32_t>(m) - kBase);
            configs.push_back(config);
        }
    }
    return configs;
}

// host 将 C 的行切分给多个 core，并下发每个 block 的起始行和行数。
// kDim 根据 side 选择 m 或 n，使 device 侧 LEFT/RIGHT 可以复用统一 tiling 结构。
static SsymmTilingData CalSsymmTilingData(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float alpha, float beta,
    uint32_t coreNum)
{
    SsymmTilingData tiling{};
    tiling.side = static_cast<uint32_t>(side);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.kDim = static_cast<uint32_t>(side == ACLBLAS_SIDE_LEFT ? m : n);
    tiling.rightChunkMask = SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::DIRECT_FULL) |
                            SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIRROR_FULL) |
                            SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG) |
                            SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::TAIL) |
                            SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG_FULL_TILE);
    tiling.rightChunkTn = 0;
    tiling.rightChunkTk = 0;
    tiling.alpha = alpha;
    tiling.beta = beta;

    if (coreNum == 0) {
        coreNum = 1;
    }
    if (coreNum > SSYMM_MAX_CORE_NUM) {
        coreNum = SSYMM_MAX_CORE_NUM;
    }

    // 计算分配策略：每核基础分配量（向下取整）+ 余数
    tiling.rowsPerCore = tiling.m / coreNum;
    tiling.rowRemainder = tiling.m % coreNum;

    // 实际使用的核数 = 至少需要的核数
    // 如果 m < coreNum，则只使用 m 个核
    tiling.useCoreNum = (tiling.m < coreNum) ? tiling.m : coreNum;
    if (tiling.useCoreNum == 0) {
        tiling.useCoreNum = 1;
    }
    return tiling;
}

// 执行通用 fallback backend。
// 该路径直接启动标量 fallback kernel，同步后把 C 从 device 拷回 host 并释放基础 buffer。
aclblasStatus_t RunSsymmFallbackBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    (void)plan;
    // 统一释放 Fallback backend 的基础输入输出 buffer
    auto cleanup = [&]() {
        aclrtFree(ctx.aDevice);
        aclrtFree(ctx.bDevice);
        aclrtFree(ctx.cDevice);
        aclrtFree(ctx.tilingDevice);
    };

    ssymm_kernel_do(
        ctx.aDevice, ctx.bDevice, ctx.cDevice, ctx.tilingDevice, ctx.side, ctx.uplo, ctx.numBlocks, ctx.stream);
    if (aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(ctx.stream)) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(ctx.hostC, ctx.cSize, ctx.cDevice, ctx.cSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    cleanup();
    return ACLBLAS_STATUS_SUCCESS;
}

// 构建 PARTIAL_CUBE 路径的 runtime config 列表。
// 末尾追加一个 strategy=FALLBACK_TO_SCRATCH 的 config 以覆盖剩余 chunk。
static std::vector<SsymmRightCubeUnifiedConfig> BuildRightCubeRuntimeConfigs(
    const SsymmRightCubeUnifiedConfig& rightCubeConfig)
{
    std::vector<SsymmRightCubeUnifiedConfig> configs;
    RightCubeExecuteRuntimePlan runtimePlan =
        (static_cast<RightCubeUplo>(rightCubeConfig.uplo) == RightCubeUplo::LOWER) ?
            BuildRightCubeLowerRuntimePlan(rightCubeConfig) :
            BuildRightCubeUpperRuntimePlan(rightCubeConfig);
    for (const RightCubeExecuteChunkPlanItem& item : runtimePlan.chunkItems) {
        configs.push_back(BuildRightCubeChunkRuntimeConfig(
            rightCubeConfig, item.chunk, runtimePlan.cubeChunkMask, runtimePlan.fallbackChunkMask));
    }
    SsymmRightCubeUnifiedConfig fallbackConfig = rightCubeConfig;
    fallbackConfig.strategy = static_cast<uint32_t>(RightCubeStrategy::FALLBACK_TO_SCRATCH);
    fallbackConfig.cubeChunkMask = runtimePlan.cubeChunkMask;
    fallbackConfig.fallbackChunkMask = runtimePlan.fallbackChunkMask;
    fallbackConfig.chunkColBase = 0;
    fallbackConfig.chunkColCount = 0;
    fallbackConfig.chunkKBase = 0;
    fallbackConfig.chunkKCount = 0;
    configs.push_back(fallbackConfig);
    return configs;
}

// 将 host tiling 结构 H2D 拷贝到设备缓冲，失败返回 INTERNAL_ERROR。
static aclblasStatus_t CopySsymmTilingToDevice(void* tilingDevice, const SsymmTilingData& tiling)
{
    aclError ret =
        aclrtMemcpy(tilingDevice, sizeof(SsymmTilingData), &tiling, sizeof(SsymmTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 为 PARTIAL_CUBE 路径清零 scratch 并写入 scratchTiling。
// scratchTiling 的 alpha/beta 固定为 1/1，ldc 改用 n（scratch 行宽）。
static aclblasStatus_t InitPartialCubeScratchTiling(
    SsymmLaunchContext& ctx, const SsymmRightCubeUnifiedConfig& rightCubeConfig,
    const std::vector<SsymmRightCubeUnifiedConfig>& runtimeConfigs, uint8_t* scratchDevice, size_t scratchBytes)
{
    if (aclrtMemset(scratchDevice, scratchBytes, 0, scratchBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    SsymmTilingData scratchTiling = ctx.tiling;
    scratchTiling.ldc = static_cast<uint32_t>(ctx.n);
    scratchTiling.alpha = 1.0f;
    scratchTiling.beta = 1.0f;
    scratchTiling.rightChunkMask = runtimeConfigs.back().fallbackChunkMask;
    scratchTiling.rightChunkTn = rightCubeConfig.tn;
    scratchTiling.rightChunkTk = rightCubeConfig.tk;
    return CopySsymmTilingToDevice(ctx.tilingDevice, scratchTiling);
}

// 为 FALLBACK_TO_SCRATCH 路径清零 scratch 并写入 scratchTiling。
// scratchTiling 的 alpha/beta 固定为 1/0，ldc 改用 n。
static aclblasStatus_t InitFallbackScratchTiling(SsymmLaunchContext& ctx, uint8_t* scratchDevice, size_t scratchBytes)
{
    if (aclrtMemset(scratchDevice, scratchBytes, 0, scratchBytes) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    SsymmTilingData scratchTiling = ctx.tiling;
    scratchTiling.ldc = static_cast<uint32_t>(ctx.n);
    scratchTiling.alpha = 1.0f;
    scratchTiling.beta = 0.0f;
    return CopySsymmTilingToDevice(ctx.tilingDevice, scratchTiling);
}

// 将 runtime config 数组（或单个 config）H2D 拷贝到 configDevice。
// PARTIAL_CUBE 拷贝全部 configs；其他策略只拷贝基础 config。
static aclblasStatus_t CopyRightCubeRuntimeConfigs(
    const SsymmRightCubeUnifiedConfig& rightCubeConfig, const std::vector<SsymmRightCubeUnifiedConfig>& runtimeConfigs,
    RightCubeStrategy strategy, uint8_t* configDevice, size_t configBytes)
{
    aclError ret = ACL_SUCCESS;
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        if (configBytes != runtimeConfigs.size() * sizeof(SsymmRightCubeUnifiedConfig)) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        ret = aclrtMemcpy(configDevice, configBytes, runtimeConfigs.data(), configBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    } else {
        ret = aclrtMemcpy(
            configDevice, sizeof(SsymmRightCubeUnifiedConfig), &rightCubeConfig, sizeof(SsymmRightCubeUnifiedConfig),
            ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// 释放 RIGHT cube 路径已分配的设备 buffer（alloc/初始化失败时的统一清理入口）。
static void ReleaseRightCubeDeviceBuffers(
    SsymmLaunchContext& ctx, uint8_t** aDenseDevice, uint8_t** scratchDevice, uint8_t** configDevice)
{
    if (*aDenseDevice != nullptr)
        aclrtFree(*aDenseDevice);
    if (*scratchDevice != nullptr)
        aclrtFree(*scratchDevice);
    if (*configDevice != nullptr)
        aclrtFree(*configDevice);
    aclrtFree(ctx.aDevice);
    aclrtFree(ctx.bDevice);
    aclrtFree(ctx.cDevice);
    aclrtFree(ctx.tilingDevice);
}

// 为 RIGHT cube 分配 scratch/pack/config buffer。
// 按策略选择性分配，失败时清理已分配的 buffer。
static aclblasStatus_t AllocRightCubeBuffers(
    size_t scratchBytes, size_t densePackBytes, size_t configBytes, RightCubeStrategy strategy, SsymmLaunchContext& ctx,
    uint8_t** aDenseDevice, uint8_t** scratchDevice, uint8_t** configDevice)
{
    auto cleanup = [&]() { ReleaseRightCubeDeviceBuffers(ctx, aDenseDevice, scratchDevice, configDevice); };
    if (aclrtMalloc(reinterpret_cast<void**>(configDevice), configBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (strategy == RightCubeStrategy::FALLBACK_TO_SCRATCH || strategy == RightCubeStrategy::PARTIAL_CUBE) {
        if (aclrtMalloc(reinterpret_cast<void**>(scratchDevice), scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
            ACL_SUCCESS) {
            cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        if (aclrtMalloc(reinterpret_cast<void**>(aDenseDevice), densePackBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
            ACL_SUCCESS) {
            cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 申请并初始化 RIGHT backend 所需的 workspace 内存。
// 按策略申请 scratch/pack/config，并对 scratch 清零、写入 scratchTiling。
static aclblasStatus_t AllocAndInitRightCubeWorkspace(
    const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx, const SsymmRightCubeUnifiedConfig& rightCubeConfig,
    const std::vector<SsymmRightCubeUnifiedConfig>& runtimeConfigs, RightCubeStrategy strategy, uint8_t** aDenseDevice,
    uint8_t** scratchDevice, uint8_t** configDevice)
{
    size_t scratchBytes = 0;
    size_t densePackBytes = 0;
    size_t configBytes = 0;
    if (!TryConvertWorkspaceBytes(plan.workspace, &scratchBytes, &densePackBytes, &configBytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto cleanup = [&]() { ReleaseRightCubeDeviceBuffers(ctx, aDenseDevice, scratchDevice, configDevice); };
    const aclblasStatus_t allocRet = AllocRightCubeBuffers(
        scratchBytes, densePackBytes, configBytes, strategy, ctx, aDenseDevice, scratchDevice, configDevice);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        return allocRet;
    }
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        const aclblasStatus_t initRet =
            InitPartialCubeScratchTiling(ctx, rightCubeConfig, runtimeConfigs, *scratchDevice, scratchBytes);
        if (initRet != ACLBLAS_STATUS_SUCCESS) {
            cleanup();
            return initRet;
        }
    }
    if (strategy == RightCubeStrategy::FALLBACK_TO_SCRATCH) {
        const aclblasStatus_t initRet = InitFallbackScratchTiling(ctx, *scratchDevice, scratchBytes);
        if (initRet != ACLBLAS_STATUS_SUCCESS) {
            cleanup();
            return initRet;
        }
    }
    const aclblasStatus_t copyRet =
        CopyRightCubeRuntimeConfigs(rightCubeConfig, runtimeConfigs, strategy, *configDevice, configBytes);
    if (copyRet != ACLBLAS_STATUS_SUCCESS) {
        cleanup();
        return copyRet;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 按策略执行 RIGHT cube kernel dispatch。
// PARTIAL_CUBE 依次 launch 各 chunk 再执行 postprocess；其余策略单次 launch。
static aclblasStatus_t DispatchRightCubeKernels(
    SsymmLaunchContext& ctx, const SsymmRightCubeUnifiedConfig& rightCubeConfig,
    const std::vector<SsymmRightCubeUnifiedConfig>& runtimeConfigs, RightCubeStrategy strategy, uint8_t* aDenseDevice,
    uint8_t* scratchDevice, uint8_t* configDevice)
{
    auto cleanup = [&]() {
        if (aDenseDevice != nullptr)
            aclrtFree(aDenseDevice);
        if (scratchDevice != nullptr)
            aclrtFree(scratchDevice);
        if (configDevice != nullptr)
            aclrtFree(configDevice);
        aclrtFree(ctx.aDevice);
        aclrtFree(ctx.bDevice);
        aclrtFree(ctx.cDevice);
        aclrtFree(ctx.tilingDevice);
    };
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        const size_t configStride = sizeof(SsymmRightCubeUnifiedConfig);
        const size_t fallbackIndex = runtimeConfigs.size() - 1;
        for (size_t i = 0; i < fallbackIndex; ++i) {
            ssymm_right_cube_unified_do(
                ctx.aDevice, aDenseDevice, ctx.bDevice, ctx.cDevice, scratchDevice, ctx.tilingDevice,
                configDevice + i * configStride, runtimeConfigs[i], ctx.numBlocks, ctx.stream);
        }
        const SsymmRightCubeUnifiedConfig& fallbackConfig = runtimeConfigs[fallbackIndex];
        const aclblasFillMode_t uplo =
            (static_cast<RightCubeUplo>(fallbackConfig.uplo) == RightCubeUplo::LOWER) ? ACLBLAS_LOWER : ACLBLAS_UPPER;
        ssymm_kernel_do(
            ctx.aDevice, ctx.bDevice, scratchDevice, ctx.tilingDevice, ACLBLAS_SIDE_RIGHT, uplo, ctx.numBlocks,
            ctx.stream);
        ssymm_right_cube_unified_postprocess_do(scratchDevice, ctx.cDevice, configDevice, ctx.stream);
    } else {
        ssymm_right_cube_unified_do(
            ctx.aDevice, aDenseDevice, ctx.bDevice, ctx.cDevice, scratchDevice, ctx.tilingDevice, configDevice,
            rightCubeConfig, ctx.numBlocks, ctx.stream);
    }
    if (aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(ctx.stream)) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(ctx.hostC, ctx.cSize, ctx.cDevice, ctx.cSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    cleanup();
    return ACLBLAS_STATUS_SUCCESS;
}

// 执行 RIGHT cube backend。
// 该路径按策略构建 runtime configs，申请 workspace，再按策略 dispatch kernel。
aclblasStatus_t RunSsymmRightBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    (void)plan;
    const int64_t m = ctx.m;
    const int64_t n = ctx.n;
    const int64_t lda = ctx.lda;
    const int64_t ldb = ctx.ldb;
    const int64_t ldc = ctx.ldc;

    SsymmRightCubeUnifiedConfig rightCubeConfig =
        BuildRightCubeUnifiedConfig(ctx.uplo, m, n, lda, ldb, ldc, ctx.alpha, ctx.beta);
    uint8_t* aDenseDevice = nullptr;
    uint8_t* scratchDevice = nullptr;
    uint8_t* configDevice = nullptr;
    RightCubeStrategy strategy = static_cast<RightCubeStrategy>(rightCubeConfig.strategy);

    std::vector<SsymmRightCubeUnifiedConfig> runtimeConfigs;
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        runtimeConfigs = BuildRightCubeRuntimeConfigs(rightCubeConfig);
    }

    const aclblasStatus_t allocRet = AllocAndInitRightCubeWorkspace(
        plan, ctx, rightCubeConfig, runtimeConfigs, strategy, &aDenseDevice, &scratchDevice, &configDevice);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        return allocRet;
    }
    return DispatchRightCubeKernels(
        ctx, rightCubeConfig, runtimeConfigs, strategy, aDenseDevice, scratchDevice, configDevice);
}

// 申请并初始化 LEFT backend 所需的 workspace 内存。
// 分配 workspace/scratch/config 三块，对 workspace 和 scratch 清零，再把 config 数组 H2D。
static aclblasStatus_t AllocAndInitLeftCubeWorkspace(
    const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx, const std::vector<SsymmLeftCubeConfig>& runtimeConfigs,
    uint8_t** workspaceDevice, uint8_t** scratchDevice, uint8_t** configDevice)
{
    size_t scratchBytes = 0;
    size_t workspaceBytes = 0;
    size_t configBytes = 0;
    if (!TryConvertWorkspaceBytes(plan.workspace, &scratchBytes, &workspaceBytes, &configBytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto cleanup = [&]() {
        if (*workspaceDevice != nullptr)
            aclrtFree(*workspaceDevice);
        if (*scratchDevice != nullptr)
            aclrtFree(*scratchDevice);
        if (*configDevice != nullptr)
            aclrtFree(*configDevice);
        aclrtFree(ctx.aDevice);
        aclrtFree(ctx.bDevice);
        aclrtFree(ctx.cDevice);
        aclrtFree(ctx.tilingDevice);
    };
    if (aclrtMalloc(reinterpret_cast<void**>(workspaceDevice), workspaceBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
            ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(scratchDevice), scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(configDevice), configBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemset(*workspaceDevice, workspaceBytes, 0, workspaceBytes) != ACL_SUCCESS ||
        aclrtMemset(*scratchDevice, scratchBytes, 0, scratchBytes) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (configBytes != runtimeConfigs.size() * sizeof(SsymmLeftCubeConfig)) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(*configDevice, configBytes, runtimeConfigs.data(), configBytes, ACL_MEMCPY_HOST_TO_DEVICE) !=
        ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 执行 LEFT cube backend。
// 该路径按 row/k 分块依次 pack、dense、accum，最后在每个 row block 末尾做 postprocess。
aclblasStatus_t RunSsymmLeftBackendImpl(const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx)
{
    (void)plan;
    const int64_t m = ctx.m;
    const int64_t n = ctx.n;
    const int64_t lda = ctx.lda;
    const int64_t ldb = ctx.ldb;
    const int64_t ldc = ctx.ldc;

    uint8_t* workspaceDevice = nullptr;
    uint8_t* scratchDevice = nullptr;
    uint8_t* configDevice = nullptr;
    std::vector<SsymmLeftCubeConfig> runtimeConfigs =
        BuildLeftCubeRuntimeConfigs(ctx.uplo, m, n, lda, ldb, ldc, ctx.alpha, ctx.beta);

    const aclblasStatus_t allocRet =
        AllocAndInitLeftCubeWorkspace(plan, ctx, runtimeConfigs, &workspaceDevice, &scratchDevice, &configDevice);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        return allocRet;
    }

    auto cleanup = [&]() {
        if (workspaceDevice != nullptr)
            aclrtFree(workspaceDevice);
        if (scratchDevice != nullptr)
            aclrtFree(scratchDevice);
        if (configDevice != nullptr)
            aclrtFree(configDevice);
        aclrtFree(ctx.aDevice);
        aclrtFree(ctx.bDevice);
        aclrtFree(ctx.cDevice);
        aclrtFree(ctx.tilingDevice);
    };

    for (size_t i = 0; i < runtimeConfigs.size(); ++i) {
        const size_t configStride = sizeof(SsymmLeftCubeConfig);
        ssymm_left_cube_do(
            ctx.aDevice, workspaceDevice, ctx.bDevice, scratchDevice, configDevice + i * configStride, ctx.stream);
        const SsymmLeftCubeConfig& config = runtimeConfigs[i];
        if (config.kBase + config.kCount == static_cast<uint32_t>(m)) {
            ssymm_left_cube_postprocess_do(scratchDevice, ctx.cDevice, configDevice + i * configStride, ctx.stream);
        }
        if (aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(ctx.stream)) != ACL_SUCCESS) {
            cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (aclrtMemcpy(ctx.hostC, ctx.cSize, ctx.cDevice, ctx.cSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    cleanup();
    return ACLBLAS_STATUS_SUCCESS;
}

// 分配 device buffer 并将 host 数据 H2D。
// 按 A/B/C/tiling 顺序申请内存；任一步失败则释放已分配的 buffer 并返回错误。
static aclblasStatus_t AllocAndCopyInputBuffers(
    SsymmLaunchContext& ctx, size_t aSize, size_t bSize, size_t cSize, const float* A, const float* B, const float* C,
    uint8_t** aDevice, uint8_t** bDevice, uint8_t** cDevice, uint8_t** tilingDevice)
{
    if (aclrtMalloc(reinterpret_cast<void**>(aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(reinterpret_cast<void**>(bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(*aDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(reinterpret_cast<void**>(cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(*aDevice);
        aclrtFree(*bDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(reinterpret_cast<void**>(tilingDevice), sizeof(SsymmTilingData), ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        aclrtFree(*aDevice);
        aclrtFree(*bDevice);
        aclrtFree(*cDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*aDevice, aSize, A, aSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(*bDevice, bSize, B, bSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(*cDevice, cSize, C, cSize, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*aDevice);
        aclrtFree(*bDevice);
        aclrtFree(*cDevice);
        aclrtFree(*tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ctx.aDevice = *aDevice;
    ctx.bDevice = *bDevice;
    ctx.cDevice = *cDevice;
    ctx.tilingDevice = *tilingDevice;
    ctx.cSize = cSize;
    return ACLBLAS_STATUS_SUCCESS;
}

// 计算 A/B/C 三个矩阵 buffer 的字节数。
// 任一维度溢出则返回 false，aSize/bSize/cSize 保持未定义。
static bool ComputeSsymmBufferSizes(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc, size_t* aSize, size_t* bSize,
    size_t* cSize)
{
    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    return TryComputeFloatBufferBytes(static_cast<size_t>(aDim), static_cast<size_t>(lda), aSize) &&
           TryComputeFloatBufferBytes(static_cast<size_t>(m), static_cast<size_t>(ldb), bSize) &&
           TryComputeFloatBufferBytes(static_cast<size_t>(m), static_cast<size_t>(ldc), cSize);
}

// 在 buffer 和 tiling 就绪后，记录 trace 并按 backend 分派执行。
// 负责 tiling H2D、设置 trace、调用对应 backend。
static aclblasStatus_t DispatchSsymmBackend(
    const SsymmExecutionPlan& plan, SsymmLaunchContext& ctx, uint8_t* aDevice, uint8_t* bDevice, uint8_t* cDevice,
    uint8_t* tilingDevice)
{
    if (CopySsymmTilingToDevice(tilingDevice, ctx.tiling) != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(aDevice);
        aclrtFree(bDevice);
        aclrtFree(cDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    g_ssymmRuntimeTrace = {};
    g_ssymmRuntimeTrace.backendKind = plan.backend;
    DumpSsymmExecutionBackend(ctx.side, ctx.uplo, g_ssymmRuntimeTrace.backendKind);
    if (g_ssymmRuntimeTrace.backendKind == SsymmBackendKind::RightCube) {
        return RunSsymmRightBackend(plan, ctx);
    }
    if (g_ssymmRuntimeTrace.backendKind == SsymmBackendKind::LeftCube) {
        return RunSsymmLeftBackend(plan, ctx);
    }
    return RunSsymmFallbackBackend(plan, ctx);
}

// 验证参数并构建执行计划。
// 处理 quick return 路径和 workspace 有效性检查。
// quickReturn 输出参数：true 表示 m=0/n=0 快速退出，调用方应立即返回 SUCCESS，不执行任何计算。
static aclblasStatus_t ValidateAndBuildPlan(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const float* alpha,
    const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc,
    SsymmExecutionPlan* plan, bool* quickReturn)
{
    if (quickReturn != nullptr) {
        *quickReturn = false;
    }
    g_ssymmRuntimeTrace = {};
    bool qr = false;
    const aclblasStatus_t validateRet =
        ValidateSsymmArgs(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, &qr);
    if (validateRet != ACLBLAS_STATUS_SUCCESS)
        return validateRet;
    if (qr) {
        if (quickReturn != nullptr) {
            *quickReturn = true;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    const SsymmProblemSpec spec = NormalizeSsymmSpec(side, uplo, m, n, alpha, lda, ldb, ldc, beta);
    *plan = BuildSsymmExecutionPlan(spec);
    PrepareSsymmWorkspace(plan);
    if (!plan->workspace.valid)
        return ACLBLAS_STATUS_INVALID_VALUE;

    aclrtContext currentCtx = nullptr;
    if (aclrtGetCurrentContext(&currentCtx) != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// 资源分配与 kernel 下发段（核数查询、stream 获取、tiling、缓冲分配拷贝、backend 派发）。
// 从 RunSsymmHostOrchestration 拆出独立函数：该段在入口侧由 try/catch 包裹，
// 拆分同时保证两个函数都不触碰单函数规模门禁。
aclblasStatus_t AllocateAndDispatchSsymm(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const float* alpha,
    const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc, size_t aSize,
    size_t bSize, size_t cSize, const SsymmExecutionPlan& plan)
{
    uint32_t numBlocks = GetAivCoreCount();
    if (numBlocks == 0)
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    if (numBlocks > SSYMM_MAX_CORE_NUM)
        numBlocks = SSYMM_MAX_CORE_NUM;

    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    SsymmTilingData tiling = CalSsymmTilingData(side, m, n, lda, ldb, ldc, *alpha, *beta, numBlocks);
    SsymmLaunchContext ctx{
        side, uplo,  m,         n,      lda,    ldb,     ldc,     *alpha,  *beta,
        C,    cSize, numBlocks, stream, tiling, nullptr, nullptr, nullptr, nullptr,
    };

    uint8_t* aDevice = nullptr;
    uint8_t* bDevice = nullptr;
    uint8_t* cDevice = nullptr;
    uint8_t* tilingDevice = nullptr;
    const aclblasStatus_t bufRet =
        AllocAndCopyInputBuffers(ctx, aSize, bSize, cSize, A, B, C, &aDevice, &bDevice, &cDevice, &tilingDevice);
    if (bufRet != ACLBLAS_STATUS_SUCCESS)
        return bufRet;

    return DispatchSsymmBackend(plan, ctx, aDevice, bDevice, cDevice, tilingDevice);
}

// 当前 host 编排只覆盖 float、row-major、host scalar alpha/beta 的 ssymm 主路径。
// 函数负责校验、建 plan、拷贝输入、下发 tiling，并按 trace 中的 backend 分派执行。
aclblasStatus_t RunSsymmHostOrchestration(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const float* alpha,
    const float* A, int64_t lda, const float* B, int64_t ldb, const float* beta, float* C, int64_t ldc)
{
    // BLAS 标准规定 m=0 或 n=0 时必须立即返回 SUCCESS，不做任何计算。
    // 此检查必须在 handle 的 null 检查之后、所有其他校验（包括 lda/ldb/指针）之前。
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    SsymmExecutionPlan plan{};
    bool quickReturn = false;
    const aclblasStatus_t planRet =
        ValidateAndBuildPlan(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, &plan, &quickReturn);
    if (planRet != ACLBLAS_STATUS_SUCCESS)
        return planRet;
    if (quickReturn)
        return ACLBLAS_STATUS_SUCCESS;

    size_t aSize = 0;
    size_t bSize = 0;
    size_t cSize = 0;
    if (!ComputeSsymmBufferSizes(side, m, n, lda, ldb, ldc, &aSize, &bSize, &cSize)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // Resource allocation and kernel dispatch allocate via std::vector/aclrt;
    // a std::exception here must not cross the extern "C" boundary
    // (sgemm_grouped_batched contract). Parameter validation stays outside.
    try {
        return AllocateAndDispatchSsymm(
            handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, aSize, bSize, cSize, plan);
    } catch (const std::exception&) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
}

// 公开的 aclblasSsymm API 入口。
// 这里只做一层转发，具体 host 编排和 backend 调度由 RunSsymmHostOrchestration 完成。
aclblasStatus_t aclblasSsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const float* alpha,
    const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    return RunSsymmHostOrchestration(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
