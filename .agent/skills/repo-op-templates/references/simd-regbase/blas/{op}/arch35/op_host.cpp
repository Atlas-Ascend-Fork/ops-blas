/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: SIMD regbase Host 侧实现
// 强制结构：Validate{Op}Params + Launch{Op}Kernel 拆分；dlog 集成（log/log.h）；
// Tiling 传递方式：host 侧以 const 引用传入 kernel_do（无 GM 设备内存分配，无同步）

#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "{{op}}_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

// GetAivCoreCount：由 host_utils.h 提供的公共版本（通过 PlatformAscendCManager 获取 AIV 核数）
// **禁止**在算子 host 文件中重复定义此函数

// TEMPLATE: UB 总大小
// 注意：kernel_constant.h 中定义 UB_SIZE = 248 * 1024（理论最大值）
// 但实际可用 UB 通常小于此值（需预留空间给栈、临时变量等）
// 建议根据算子复杂度选择：
//   - 简单算子：190 * 1024（190KB，保守值）
//   - 复杂算子：240 * 1024（240KB，接近最大值）
constexpr uint32_t UB_SIZE = 190 * 1024; // 190KB for typical Ascend chip

// TEMPLATE: 对齐常量
constexpr uint32_t ALIGN_BYTES = 256; // UB buffer 需要 256B 对齐

static uint32_t AlignUp(uint32_t size, uint32_t align) { return (size + align - 1) & ~(align - 1); }

// TEMPLATE: 计算 UB buffer 布局
struct BufferLayout {
    uint32_t bufInput;
    uint32_t bufOutput;
    uint32_t bufTemp;
    uint32_t total;
};

static BufferLayout ComputeBufferLayout(uint32_t m, uint32_t n)
{
    BufferLayout layout = {};

    // TEMPLATE: 根据算子需求计算各 buffer 大小（256B 对齐）
    // layout.bufInput = AlignUp(m * n * sizeof(float), ALIGN_BYTES);
    // layout.bufOutput = AlignUp(m * sizeof(float), ALIGN_BYTES);
    // layout.bufTemp = AlignUp(m * sizeof(float), ALIGN_BYTES);

    layout.total = layout.bufInput + layout.bufOutput + layout.bufTemp;
    return layout;
}

// TEMPLATE: Validate{Op}Params 参数校验
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
TilingData(
    /* 算子维度参数, 标量参数, */
    uint32_t coreNum, uint32_t usedCoreNum, uint32_t batchPerCore)
{
    {
        {
            Op
        }
    }
    TilingData tiling = {};

    // TEMPLATE: 填充基础参数
    // tiling.m = m;
    // tiling.n = n;
    // tiling.alpha = alpha;
    // tiling.beta = beta;

    // TEMPLATE: 填充多核切分参数
    tiling.coreNum = coreNum;
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = /* batchCount - (usedCoreNum - 1) * batchPerCore */;

    // TEMPLATE: 计算 UB buffer 大小
    // auto layout = ComputeBufferLayout(/* 维度参数 */);
    // tiling.bufInput = layout.bufInput;
    // tiling.bufOutput = layout.bufOutput;
    // tiling.bufTemp = layout.bufTemp;

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

    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // TEMPLATE: 根据算子维度计算 batch 分配
    // uint32_t batchPerCore = (batchCount + coreNum - 1) / coreNum;
    // uint32_t usedCoreNum = (batchCount + batchPerCore - 1) / batchPerCore;

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
    TilingData(
        /* 维度参数, */ coreNum /*, usedCoreNum, batchPerCore */);

    OP_LOGD("aclblas{{Op}}", "tiling: ...");
    OP_LOGI("aclblas{{Op}}", "launching kernel: usedCoreNum=%u", usedCoreNum);

    // Workspace 从 handle 获取（禁止自行 aclrtMalloc）；若算子不需要 workspace，传 nullptr
    // 校验当前 handle workspace 是否满足算子需求
    // CHECK_RET(workSpaceNeed <= GetEffectiveWorkspaceSize(h),
    //           OP_LOGE(...); return ACLBLAS_STATUS_EXECUTION_FAILED);
    uint8_t* workSpaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    // Tiling 直接以 const 引用传入（无 H2D 拷贝，无 tilingDevice 分配）
    // 异步 launch，launch 后不调用 aclrtSynchronizeStream
    {
        {
            op
        }
    }
    _kernel_do(/* GM_ADDR 各数据指针, */ workSpaceDevice, usedCoreNum, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// TEMPLATE: 公共 API 入口（仅做调度，逻辑委托给 Validate + Launch）
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle
                              // TEMPLATE: 添加算子特定参数（如维度、标量、指针等）
)
{
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
