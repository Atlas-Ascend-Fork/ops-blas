/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: Ex 变体 Host 侧实现骨架（arch35 / DAV_3510）
// ============================================================================
// 「Ex 变体」判定特征（满足即适用本模板）：
//   ★ 多数据类型：A/B/C（或 alpha/x）各路带独立 aclDataType
//   ★ 计算精度：aclblasComputeType_t computeType 决定累加精度（COMPUTE_16F / COMPUTE_32F）
//   ★ alpha/beta 为 void*：宿主类型由 computeType 决定（COMPUTE_16F→half, COMPUTE_32F→float）
//   ★ dtype 组合白名单：非法 (Atype,Btype,Ctype,computeType) 返回 ACLBLAS_STATUS_NOT_SUPPORTED
//   ★ dtype 派发：GetDtypeCase() 把 aclDataType 组合映射为 DTypeCase 枚举，驱动 kernel 侧模板选择
//
// 已知 ex 变体形态（参数列表 / 计算模型差异大，本模板不写死任何一种）：
//   ┌─────────────────────┬────────────────────────────────┬──────────────────────┐
//   │ 形态                 │ API 参数特征                    │ 计算模型              │
//   ├─────────────────────┼────────────────────────────────┼──────────────────────┤
//   │ 单矩阵 (gemm_ex)     │ A/B/C 单指针 + m,n,k + algo     │ Cube BlockMmad       │
//   │ 批量 (gemm_batched)  │ Aarray/Barray/Carray + batchCount│ Cube BlockMmad      │
//   │ 分组 (gemm_grouped)  │ 逐组数组 + groupCount/groupSize │ Cube + Epilogue      │
//   │ 向量 (scalex)        │ alpha + x + incx + executionType│ Vector SIMD/SIMT    │
//   └─────────────────────┴────────────────────────────────┴──────────────────────┘
//
// 因此本模板只固化「ex 共性骨架」，参数列表与计算逻辑用占位符标注，按算子 API 填充。
// 共性骨架（所有 ex 变体必备）：
//   1. dtype 工具三件套：IsValidDtypeCombination / IsFP8Type / GetDtypeCase
//   2. Validate{{Op}}Params 拆分（含 dtype 组合校验）
//   3. Launch{{Op}}Kernel：tiling 计算 + dtype 派发 + kernel launch（异步，禁止同步）
// 强制规范（与 simd-membase 一致）：dlog 集成；kernel.h 共用头；无 printf。
// ============================================================================

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "{{op}}_tiling_data.h"
#include "{{op}}_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/dtype_cast.h"

// ============================================================================
// TEMPLATE: dtype 工具三件套（ex 变体必备，所有形态共用）
// ============================================================================

// TEMPLATE: IsValidDtypeCombination —— dtype 白名单（按算子支持表实现）
// - 返回 false 时 host 返回 ACLBLAS_STATUS_NOT_SUPPORTED
// - 形态差异：单矩阵/批量/分组校验 Atype/Btype/Ctype 三元组；向量类校验 alphaType/xType/executionType
//   → 参数个数与具体 aclDataType 取值随算子调整，下方为矩阵类示例
static bool IsValidDtypeCombination(
    aclDataType Atype, aclDataType Btype, aclDataType Ctype, aclblasComputeType_t computeType)
{
    switch (computeType) {
        case ACLBLAS_COMPUTE_16F:
            // TEMPLATE: COMPUTE_16F 通常仅支持 FP16（按算子支持表调整）
            return (Atype == ACL_FLOAT16 && Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16);
        case ACLBLAS_COMPUTE_32F:
            // TEMPLATE: COMPUTE_32F 支持 FP16/BF16/FP32/FP8 组合（按算子支持表增删）
            switch (Atype) {
                case ACL_FLOAT16:
                    return (Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16);
                case ACL_BF16:
                    return (Btype == ACL_BF16 && Ctype == ACL_BF16);
                case ACL_FLOAT:
                    return (Btype == ACL_FLOAT && Ctype == ACL_FLOAT);
                case ACL_FLOAT8_E4M3FN:
                    return ((Btype == ACL_FLOAT8_E4M3FN || Btype == ACL_FLOAT8_E5M2) && Ctype == ACL_FLOAT16);
                case ACL_FLOAT8_E5M2:
                    return ((Btype == ACL_FLOAT8_E5M2 || Btype == ACL_FLOAT8_E4M3FN) && Ctype == ACL_FLOAT16);
                default:
                    return false;
            }
        default:
            return false;
    }
    // VARIANT: 向量类（scalex）—— 校验 alphaType/xType/executionType，如：
    //   if (alphaType != ACL_FLOAT || executionType != ACL_FLOAT) return false;
    //   return xType == ACL_FLOAT16 || xType == ACL_BF16 || xType == ACL_FLOAT;
}

static bool IsFP8Type(aclDataType dtype) { return (dtype == ACL_FLOAT8_E4M3FN || dtype == ACL_FLOAT8_E5M2); }

// TEMPLATE: GetDtypeCase —— aclDataType 组合 → DTypeCase 枚举
// - 与 kernel.cpp 的 kernel_do switch 一一对应
// - 形态差异：矩阵类按 (Atype,Btype,Ctype) 三元组；向量类按 xType 单值
//   → 分支数与枚举值随算子支持表调整
static
{
    {
        Op
    }
}
DTypeCase GetDtypeCase(aclDataType Atype, aclDataType Btype, aclDataType Ctype)
{
    switch (Atype) {
        case ACL_FLOAT16:
            return (Btype == ACL_FLOAT16 && Ctype == ACL_FLOAT16) ?
            {
                {
                    OP
                }
            }
        _DTYPE_FP16: {
            {
                OP
            }
        }
            _DTYPE_INVALID;
        case ACL_BF16:
            return (Btype == ACL_BF16 && Ctype == ACL_BF16) ?
            {
                {
                    OP
                }
            }
        _DTYPE_BF16: {
            {
                OP
            }
        }
            _DTYPE_INVALID;
        case ACL_FLOAT:
            return (Btype == ACL_FLOAT && Ctype == ACL_FLOAT) ?
            {
                {
                    OP
                }
            }
        _DTYPE_FP32: {
            {
                OP
            }
        }
            _DTYPE_INVALID;
        case ACL_FLOAT8_E4M3FN:
            if (Btype == ACL_FLOAT8_E4M3FN && Ctype == ACL_FLOAT16) {
                return
                {
                    {
                        OP
                    }
                }
                _DTYPE_FP8_E4M3;
            }
            if (Btype == ACL_FLOAT8_E5M2 && Ctype == ACL_FLOAT16) {
                return
                {
                    {
                        OP
                    }
                }
                _DTYPE_FP8_E4M3_E5M2;
            }
            return
            {
                {
                    OP
                }
            }
            _DTYPE_INVALID;
        case ACL_FLOAT8_E5M2:
            if (Btype == ACL_FLOAT8_E5M2 && Ctype == ACL_FLOAT16) {
                return
                {
                    {
                        OP
                    }
                }
                _DTYPE_FP8_E5M2;
            }
            if (Btype == ACL_FLOAT8_E4M3FN && Ctype == ACL_FLOAT16) {
                return
                {
                    {
                        OP
                    }
                }
                _DTYPE_FP8_E5M2_E4M3;
            }
            return
            {
                {
                    OP
                }
            }
            _DTYPE_INVALID;
        default:
            return
            {
                {
                    OP
                }
            }
            _DTYPE_INVALID;
    }
    // VARIANT: 向量类 —— 按 xType 单值派发：
    //   switch (xType) { case ACL_FLOAT: return X_DTYPE_FP32; case ACL_FLOAT16: return X_DTYPE_FP16; ... }
}

// ============================================================================
// TEMPLATE: Validate{{Op}}Params —— 参数校验（建议按维度拆分子函数）
// 校验顺序：handle → 业务维度 → dtype 组合 & 指针
// 参数列表完全由算子 API 决定，此处用占位符
// ============================================================================
static aclblasStatus_t Validate
{
    {
        Op
    }
}
Params(aclblasHandle_t handle /* , 算子 API 全部参数 */)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblas{{Op}}", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    // TEMPLATE: 业务维度校验（按算子语义，建议拆为 ValidateDimensions / ValidateLeadingDims 等子函数）
    // - 矩阵类：m,n,k >= 0；lda/ldb/ldc >= max(1, 物理行数)；transa/transb ∈ {N,T,C}
    // - 向量类：n >= 0；incx 语义（scalex: incx<=0 为 no-op）
    // - 批量类：batchCount >= 0
    // - 分组类：groupCount >= 0；逐组维度/ld 校验

    // TEMPLATE: dtype 组合 & 指针校验
    // CHECK_RET(IsValidDtypeCombination(...), OP_LOGE(...); return ACLBLAS_STATUS_NOT_SUPPORTED);
    // if (IsFP8Type(...)) { CHECK_RET(computeType == ACLBLAS_COMPUTE_32F, ...); }
    // CHECK_RET(alpha != nullptr, ...); CHECK_RET(beta != nullptr, ...);
    // CHECK_RET(A != nullptr when k>0, ...);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// TEMPLATE: Cal{{Op}}Tiling —— host 侧切分参数计算
// 切分策略随计算模型差异极大，此处仅给共性框架，具体逻辑按算子填充
// ============================================================================
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
Tiling(/* 算子维度参数, */ uint32_t coreNum)
{
    {
        {
            Op
        }
    }
    TilingData tiling{};
    // VARIANT: 矩阵类（Cube BlockMmad）—— M×N 二维切分 + K 串行归约
    //   - baseM/baseN/baseK 随 dtype 变化（FP16/BF16=128/128/16, FP32=32/16/8, FP8=32/16/32）
    //   - 搜最优 (mBlocks, nBlocks) 使 mBlocks*nBlocks <= coreNum 且利用率最高
    //   - singleCoreM/N 向上对齐到 baseM/baseN
    // VARIANT: 向量类（Vector SIMD/SIMT）—— 元素均分 + UB tile
    //   - perCoreN = (totalN / coreNum) 对齐到 alignUnit
    //   - tileSize = (UB_SIZE / bytePerElement) 对齐到 alignUnit
    //   - incx==1 走 SIMD 连续路径，incx!=1 走 SIMT stride 路径（两套 CalTilingData）
    // VARIANT: 批量/分组类 —— 在矩阵切分基础上叠加 batch/group 维度调度
    //   - 批量：totalTasks = batchCount * mBlocks * nBlocks
    //   - 分组：BuildTiling 平铺 Header + 每组 GroupData 到 GM（host 不再 const 引用传 tiling）
    return tiling;
}

// ============================================================================
// TEMPLATE: Launch{{Op}}Kernel —— tiling 计算 + workspace 获取 + kernel launch
// - 异步执行：launch 后直接返回，禁止 aclrtSynchronizeStream（分组类例外，见下）
// - dtype 派发：把 DTypeCase 传入 kernel_do，由 kernel.cpp switch 到对应模板实例
// ============================================================================
static aclblasStatus_t Launch
{
    {
        Op
    }
}
Kernel(aclblasHandle_t handle /* , 算子 API 参数 */)
{
    auto* h = handle;

    // TEMPLATE: 取核数（矩阵类用 GetAicCoreCount/Cube；向量类用 GetAivCoreCount/Vector）
    uint32_t coreNum = /* GetAicCoreCount() or GetAivCoreCount() */ 0;
    if (coreNum == 0) {
        OP_LOGE("aclblas{{Op}}", "failed to get core count");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

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
    Tiling(/* 维度参数, */ coreNum);
    {
        {
            Op
        }
    }
    DTypeCase dtypeCase = GetDtypeCase(/* aclDataType 组合 */);

    // TEMPLATE: alpha/beta 后处理判定（矩阵类 alpha!=1||beta!=0 时启用）
    // - 启用时 Cube 落 FP32 中间态 tempAB，Vector kernel 做 C = alpha*temp + beta*C
    // - useFP32Output = needPostProcess && (dtypeCase ∈ {FP16, BF16})
    // - 向量类一般无此环节（scalex 直接 Muls 原地写回）
    bool needPostProcess = /* (alpha != 1.0f) || (beta != 0.0f) */ false;
    bool useFP32Output = false; // needPostProcess && (dtypeCase == FP16 || BF16)

    // TEMPLATE: workspace 从 handle 获取（禁止自行 aclrtMalloc）
    // uint8_t* workSpaceDevice = reinterpret_cast<uint8_t*>(aclblasGetEffectiveWorkspace(h));
    // CHECK_RET(needBytes <= aclblasGetEffectiveWorkspaceSize(h), OP_LOGE(...); return
    // ACLBLAS_STATUS_EXECUTION_FAILED);

    // VARIANT: 矩阵类 —— 列主序 swap（列主序 GEMM 转行序 B^T*A^T）
    //   std::swap(tiling.m, tiling.n); std::swap(tiling.lda, tiling.ldb);
    //   std::swap(tiling.isTransA, tiling.isTransB); swap(A指针, B指针);
    //   kernelDtypeCase = useFP32Output ? {FP16|BF16}_OUT_F32 : dtypeCase;

    OP_LOGI(
        "aclblas{{Op}}", "launching kernel: blocks=%u, dtypeCase=%d, needPostProcess=%d",
        /* numBlocks */ 0, static_cast<int>(dtypeCase), static_cast<int>(needPostProcess));

    // TEMPLATE: 主计算 kernel launch（tiling 以 const 引用传入）
    // {{op}}_kernel_do(/* GM_ADDR 数据指针（按算子）, */ numBlocks, tiling, dtypeCase, h->stream);

    // TEMPLATE: alpha/beta 后处理 kernel launch（可选，矩阵类）
    // if (needPostProcess) {
    //     {{op}}_alpha_beta_do(abCores, h->stream, tempAB, cOrig, cOut, abTiling, dtypeCase, useFP32Output);
    // }

    // VARIANT: 分组类 —— tiling 平铺到 GM，需 aclrtSynchronizeStream 等待 cube+epilogue 完成
    //   （分组类是唯一允许同步的形态，因 tiling/workspace 由 host malloc，需确保 kernel 完成后释放）
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// TEMPLATE: 公共 API 入口（仅做调度：校验 → 快速返回 → Launch）
// - 签名与 cann_ops_blas.h 一致（参数列表由算子 API 决定，此处用占位符）
// - alpha/beta 为 void*，按 computeType 读取（COMPUTE_16F→half, COMPUTE_32F→float）
// ============================================================================
aclblasStatus_t aclblas{{Op}}(aclblasHandle_t handle /* , 算子 API 全部参数，含 Atype/Btype/Ctype/computeType */)
{
    OP_LOGI("aclblas{{Op}}", "entry: ..."); // TEMPLATE: 打印关键参数

    aclblasStatus_t st = Validate
    {
        {
            Op
        }
    }
    Params(handle /* , 参数 */);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblas{{Op}}", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }

    // TEMPLATE: 快速返回（按算子语义，如 m==0||n==0||batchCount==0||groupCount==0||n==0）
    // if (...) { return ACLBLAS_STATUS_SUCCESS; }

    // TEMPLATE: 按 computeType 读取 alpha/beta（矩阵类）
    // float alphaVal = *static_cast<const float*>(alpha);  // COMPUTE_32F
    // half  alphaVal = *static_cast<const half*>(alpha);   // COMPUTE_16F
    // VARIANT: 向量类 —— alpha 可能是 Host 或 Device 指针，用 aclrtPointerGetAttributes 判定

    // TEMPLATE: 早退路径（矩阵类 k==0 → C=beta*C；alpha==0 → C=beta*C 或 memset）
    // VARIANT: 向量类 —— incx<=0 为 no-op（scalex 语义）

    return Launch
    {
        {
            Op
        }
    }
    Kernel(handle /* , 参数 */);
}
