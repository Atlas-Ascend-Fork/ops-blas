/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMT arch35 Host 侧实现
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）；
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）
//
// 与 SIMD membase 的区别：
//   - 需额外计算 nthreads（每 block 线程数）
//   - block 数按 CeilDiv(n, SIMT_MIN_THREAD_NUM) 与 coreNum 取小
//   - tiling 中包含标量参数（alpha/beta/uplo 等）直接传给 kernel

#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "{{op}}_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

// GetAivCoreCount：由 host_utils.h 提供的公共版本（通过 PlatformAscendCManager 获取 AIV 核数）
// **禁止**在算子 host 文件中重复定义此函数

// TEMPLATE: Validate{Op}Params 参数校验
static aclblasStatus_t Validate
{
    {
        Op
    }
}
Params(/* 算子的参数列表 */)
{
    // TEMPLATE: 按算子需求校验各参数
    // CHECK_RET(ptr != nullptr, OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: Cal{Op}Tiling 在 host 侧计算多核切分参数
// SIMT 特有：计算 nthreads
//   nthreads = min(CeilAlign(CeilDiv(totalWork, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM)
static
{
    {
        Op
    }
}
TilingData Cal
{
    {
        Op
    }
}
TilingData(uint32_t useNumBlocks /* , 算子参数 */)
{
    {
        {
            Op
        }
    }
    TilingData tiling{};
    // TEMPLATE: 计算 nthreads
    // tiling.nthreads = std::min(
    //     CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalWork, useNumBlocks), SIMT_MIN_THREAD_NUM),
    //     SIMT_MAX_THREAD_NUM);
    // TEMPLATE: 填充其他字段（维度参数 + 标量参数）
    return tiling;
}

// TEMPLATE: Launch{Op}Kernel — 负责 tiling 计算 + workspace 获取 + kernel launch
// - 异步执行：launch kernel 后直接返回，不调用 aclrtSynchronizeStream
static aclblasStatus_t Launch
{
    {
        Op
    }
}
Kernel(aclblasHandle_t handle, /* 算子参数 */)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // TEMPLATE: SIMT 的 block 数计算（与 SIMD membase 不同）
    // useNumBlocks = min(CeilDiv(totalWork, SIMT_MIN_THREAD_NUM), aivCoreNum)
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(/* totalWork */, SIMT_MIN_THREAD_NUM), aivCoreNum);

    {
        {
            Op
        }
    }
    TilingData tiling = Cal
    {
        {
            Op
        }
    }
    TilingData(useNumBlocks /* , 参数 */);

    OP_LOGD("aclblas{{Op}}", "tiling: nthreads=%u numBlocks=%u", tiling.nthreads, useNumBlocks);
    OP_LOGI("aclblas{{Op}}", "launching kernel: blocks=%u", useNumBlocks);

    // Workspace 从 handle 获取（禁止自行 aclrtMalloc）；若算子不需要 workspace，传 nullptr
    // 校验当前 handle workspace 是否满足算子需求
    // CHECK_RET(workSpaceNeed <= GetEffectiveWorkspaceSize(h),
    //           OP_LOGE("aclblas{{Op}}", "workspace %zu > handle %zu",
    //                   workSpaceNeed, GetEffectiveWorkspaceSize(h));
    //           return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workSpaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    // Tiling 直接以 const 引用传入（无 H2D 拷贝，无 tilingDevice 分配）
    // 异步 launch，不调用 aclrtSynchronizeStream
    {
        {
            op
        }
    }
    _kernel_do(/* GM_ADDR 各数据指针, */ workSpaceDevice, useNumBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: 公共 API 入口（仅做调度，逻辑委托给 Validate + Launch）
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle /* , 算子参数 */)
{
    // TEMPLATE: 快速返回条件（按算子语义调整）
    // CHECK_RET(n >= 0, OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE);
    // if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    if (handle == nullptr) {
        OP_LOGE("aclblas{{Op}}", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = Validate
    {
        {
            Op
        }
    }
    Params(/* 参数 */);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return Launch
    {
        {
            Op
        }
    }
    Kernel(handle /* , 参数 */);
}
