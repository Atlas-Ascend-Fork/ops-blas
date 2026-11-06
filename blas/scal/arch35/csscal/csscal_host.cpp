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
 * \file csscal_host.cpp
 * \brief Host-side API implementation for aclblasCsscal:
 *        x[j] = alpha * x[j]（实数标量 alpha 乘以复数向量 x）
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "csscal_tiling.h"

void csscal_kernel_do(uint8_t* x, uint8_t* workSpace, const CsscalTilingData& tiling, uint32_t numBlocks, void* stream);

static uint32_t GetCachedAivCoreCount()
{
    // A single entry per thread avoids the platform singleton's mutex on the
    // hot path. Re-query on device/context changes; never cache a failure.
    static thread_local int32_t cachedDevice = -1;
    static thread_local aclrtContext cachedContext = nullptr;
    static thread_local uint32_t cachedCores = 0;
    int32_t device = -1;
    aclrtContext context = nullptr;
    if (aclrtGetDevice(&device) != ACL_SUCCESS || aclrtGetCurrentContext(&context) != ACL_SUCCESS) {
        return 0;
    }
    if (cachedCores != 0 && cachedDevice == device && cachedContext == context) {
        return cachedCores;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores != 0) {
        cachedDevice = device;
        cachedContext = context;
        cachedCores = cores;
    }
    return cores;
}

// 计算 incx == 1 的 tiling（连续访问，AIV SIMD 模式）
static CsscalTilingData CalCsscalTilingDataContiguous(uint32_t totalComplexNum, uint32_t aivCoreNum, float alpha)
{
    CsscalTilingData tiling{};
    tiling.totalN = totalComplexNum;
    tiling.incx = 1;

    if (aivCoreNum == 0U) {
        OP_LOGE("aclblasCsscal", "aivCoreNum is 0, skip tiling calculation");
        return tiling;
    }
    constexpr uint32_t alignUnit = CSSCAL_ALIGN_UNIT;
    uint32_t rawPerCore = totalComplexNum / aivCoreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = totalComplexNum - tiling.perCoreN * aivCoreNum;

    // One in-place UB region, clamped to the largest actual core workload.
    // A core requiring multiple tiles always uses CSSCAL_TILE_CAPACITY.
    uint32_t maxCoreN = tiling.perCoreN + tiling.remainder;
    tiling.tileSize = std::min(CSSCAL_TILE_CAPACITY, ((maxCoreN + alignUnit - 1) / alignUnit) * alignUnit);

    tiling.alpha = alpha;
    tiling.nthreads = 0;
    tiling.useCoreNum = 0;

    return tiling;
}

// 计算 incx != 1 的 tiling（跳步访问，SIMT 模式）
static CsscalTilingData CalCsscalTilingDataStrided(int64_t n, int64_t incx, uint32_t aivCoreNum, float alpha)
{
    CsscalTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.alpha = alpha;

    uint32_t useCoreNum = std::min(aivCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum > CSSCAL_MAX_CORE_NUM) {
        useCoreNum = CSSCAL_MAX_CORE_NUM;
    }
    if (useCoreNum == 0U) {
        useCoreNum = 1U; // 调用点已保证非零，防御性检查
    }
    tiling.useCoreNum = useCoreNum;

    // 每核分片只传 baseCount/remain 两个标量，具体 startOffset/calCount 由 kernel 按 blockIdx 推导
    tiling.baseCount = static_cast<uint32_t>(n) / useCoreNum;
    tiling.remain = static_cast<uint32_t>(n) % useCoreNum;

    // issue #834: csscal 实测优选的线程排布（每线程处理 1 个复数=2 个 float 分量，
    // 32 线程起步/256 封顶），与全局 kernel_constant.h 的 SIMT_MIN/MAX_THREAD_NUM
    // （128/2048，面向 float 分量粒度的排布）取值不同。此前函数内重定义与全局
    // 同名常量互相遮蔽，重构时极易误删/误改导致线程数静默变化——改用 CSSCAL_
    // 前缀显式命名并注明差异原因，行为不变。
    constexpr uint32_t CSSCAL_SIMT_MIN_THREAD_NUM = 32;
    constexpr uint32_t CSSCAL_SIMT_MAX_THREAD_NUM = 256;
    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), useCoreNum), CSSCAL_SIMT_MIN_THREAD_NUM),
        CSSCAL_SIMT_MAX_THREAD_NUM);

    tiling.perCoreN = 0;
    tiling.remainder = 0;
    tiling.tileSize = 0;

    return tiling;
}

aclblasStatus_t aclblasCsscal(aclblasHandle_t handle, int n, const float* alpha, aclblasComplex* x, int incx)
{
    // 1. 参数合法性检查
    if (handle == nullptr) {
        OP_LOGE("aclblasCsscal", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // n <= 0 或 incx <= 0 为合法 no-op
    if (n <= 0 || incx <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // n > 0 时检查 alpha 和 x
    if (alpha == nullptr || x == nullptr) {
        OP_LOGE("aclblasCsscal", "alpha and x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // alpha = 1.0 时提前返回（数值结果一致）
    if (*alpha == 1.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // alpha = 0.0 是置零操作，不是 no-op，继续执行

    uint32_t aivCoreNum = GetCachedAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsscal", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t totalN = static_cast<uint32_t>(n);

    CsscalTilingData tiling;
    uint32_t numBlocks;

    if (incx == 1) {
        // 连续访问模式
        // At least 256 complex values per core except for a small single-core
        // vector. The former min(n, cores) launched many entirely idle cores.
        numBlocks = std::min(CeilDiv<uint32_t>(totalN, 256), aivCoreNum);
        tiling = CalCsscalTilingDataContiguous(totalN, numBlocks, *alpha);
    } else {
        // 跳步访问模式
        numBlocks = std::min(CeilDiv<uint32_t>(totalN, 32), aivCoreNum);
        tiling = CalCsscalTilingDataStrided(n, incx, numBlocks, *alpha);
    }

    OP_LOGD(
        "aclblasCsscal", "tiling: n=%d incx=%d numBlocks=%u alpha=%.6f", n, incx, numBlocks,
        static_cast<double>(*alpha));

    csscal_kernel_do(reinterpret_cast<uint8_t*>(x), nullptr, tiling, numBlocks, handle->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
