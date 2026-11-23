/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMD membase Host 侧实现
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）；
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）
//
// 多路径分发可选模式（如 saxpy 的连续 + stride 分支，多 CalTilingData + 多 kernel entry 场景）：
//   static {{Op}}TilingData Cal{{Op}}TilingDataContiguous(/* 连续路径参数 */);
//   static {{Op}}TilingData Cal{{Op}}TilingDataStrided(/* stride 路径参数 */);
//   static aclblasStatus_t Launch{{Op}}Kernel(handle, ...) {
//       if (incx == 1 && incy == 1) { numBlocks = n; tiling = CalContiguous(...); }
//       else                        { numBlocks = ceil_div(n, THREAD_MIN); tiling = CalStrided(...); }
//       {{op}}_kernel_do(..., tiling, numBlocks, stream);  // tiling 中编码 dispatch 信息，kernel_do 内分支
//   }
// 该模式合法，用于同一算子的不同计算路径（连续/散列、大/小矩阵等）。

#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "{{op}}_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// GetAivCoreCount：由 host_utils.h 提供的公共版本（通过 PlatformAscendCManager 获取 AIV 核数）
// **禁止**在算子 host 文件中重复定义此函数

// TEMPLATE: Validate{Op}Params 参数校验
// - 按算子 API 的参数列表自定义校验逻辑
// - 校验顺序：nullptr → 零值 → 非法值
// - 使用 OP_LOGE 输出错误信息
static aclblasStatus_t Validate
{
    {
        Op
    }
}
Params(/* 算子的参数列表 */)
{
    // TEMPLATE: 按算子需求校验各参数
    // if (ptr == nullptr) { OP_LOGE(...); return ACLBLAS_STATUS_INVALID_VALUE; }
    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: Cal{Op}Tiling 在 host 侧计算多核切分参数
// - UB tile 大小一般按 UB_SIZE / buffer_count / sizeof(dtype) 计算并对齐
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
TilingData(/* 算子的维度参数, */ uint32_t coreNum)
{
    {
        {
            Op
        }
    }
    TilingData tiling{};
    // TEMPLATE: 按算子的切分策略填充 tiling 各字段
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

    // TEMPLATE: 确定核数（按算子维度与核数的关系）
    uint32_t numBlocks = /* min(totalWork, aivCoreNum) */;

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
    TilingData(/* 维度参数, */ numBlocks);

    OP_LOGD("aclblas{{Op}}", "tiling: ...");
    OP_LOGI("aclblas{{Op}}", "launching kernel: blocks=%u", numBlocks);

    // Workspace 从 handle 获取（禁止自行 aclrtMalloc）；若算子不需要 workspace，传 nullptr
    // 校验当前 handle workspace 是否满足算子需求
    // CHECK_RET(workSpaceNeed <= GetEffectiveWorkspaceSize(h),
    //           OP_LOGE("aclblas{{Op}}", "workspace %zu > handle %zu",
    //                   workSpaceNeed, GetEffectiveWorkspaceSize(h));
    //           return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workSpaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    // Tiling 直接以 const 引用传入（无 H2D 拷贝，无 tilingDevice 分配）
    {
        {
            op
        }
    }
    _kernel_do(/* GM_ADDR 各数据指针, */ workSpaceDevice, numBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: 公共 API 入口（仅做调度，逻辑委托给 Validate + Launch）
// - 签名与 cann_ops_blas.h 中的声明一致
// - 前置快速返回：n<=0 → SUCCESS，handle==nullptr → HANDLE_IS_NULLPTR
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle /* , 算子参数 */)
{
    // TEMPLATE: 快速返回条件（按算子语义调整）
    // if (n <= 0) return ACLBLAS_STATUS_SUCCESS;

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
