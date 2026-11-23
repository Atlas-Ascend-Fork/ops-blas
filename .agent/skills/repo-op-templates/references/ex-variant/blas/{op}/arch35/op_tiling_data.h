/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: Ex 变体 Tiling 数据结构（host/kernel 共享）
// ============================================================================
// ex 变体核心增量：携带「dtype 派发信息」供 kernel 侧选择模板实例。
// 派发链路（所有 ex 变体一致）：
//   1. DTypeCase 枚举：每个有效 dtype 组合对应一个枚举值
//   2. host GetDtypeCase()：aclDataType 组合 → DTypeCase
//   3. kernel_do(..., DTypeCase) 内 switch 派发到各模板 kernel 实例
//
// TilingData 字段完全由算子切分策略决定，不同形态差异极大（下方按形态列示），
// 本模板仅给出「DTypeCase 枚举 + 矩阵类字段示例」，按算子增删。
// ============================================================================

#pragma once

#include <cstdint>

// TEMPLATE: DTypeCase 枚举 —— 覆盖算子支持的全部 dtype 组合
// - 枚举值与 host GetDtypeCase() 返回值、kernel_do switch case 三者一一对应
// - 形态差异：矩阵类按 (Atype,Btype,Ctype) 三元组；向量类按 xType 单值
// - 「OUT_F32」中间态：alpha/beta 后处理时 FP16/BF16 输入落 FP32 中间盘（矩阵类专用）
// 按算子支持表增删以下枚举值，命名建议 {{OP}}_DTYPE_<组合描述>
enum {
    {Op}
} DTypeCase : int32_t{
    {{OP}} _DTYPE_FP16 = 0,          {{OP}} _DTYPE_BF16 = 1,          {{OP}} _DTYPE_FP32 = 2,
    {{OP}} _DTYPE_FP8_E4M3 = 3,      {{OP}} _DTYPE_FP8_E5M2 = 4,      {{OP}} _DTYPE_FP8_E5M2_E4M3 = 5,
    {{OP}} _DTYPE_FP8_E4M3_E5M2 = 6, {{OP}} _DTYPE_FP16_OUT_F32 = -1, // 后处理中间态（可选，矩阵类）
    {{OP}} _DTYPE_BF16_OUT_F32 = -2, {{OP}} _DTYPE_INVALID = -3
    // VARIANT: 向量类 —— 枚举按 xType，如 X_DTYPE_FP32 / X_DTYPE_FP16 / X_DTYPE_BF16，无 OUT_F32
};

// TEMPLATE: TilingData —— 按算子切分策略定义字段，下方为各形态示例
// 选其一或组合，切勿照抄；字段类型保持 POD（host/kernel 共享需可平凡拷贝）。
//
// ┌ 形态 A：单矩阵 (gemm_ex) ────────────────────────────────────────────
// │ 矩阵维度 m/n/k + leading dim lda/ldb/ldc + 多核切分 mBlocks/nBlocks/
// │ singleCoreM/N + Cube tile baseM/N/K/c0Size + 转置标志 + alpha/beta + outputFp32
struct {
    {
        Op
    }
} TilingData
{
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t usedCoreNum;
    int32_t mBlocks;
    int32_t nBlocks;
    int32_t singleCoreM;
    int32_t singleCoreN;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t c0Size;
    int32_t isTransA;
    int32_t isTransB;
    float alpha;
    float beta;
    int32_t hasBeta;
    int32_t outputFp32;
};
// │ ┌ 形态 B：批量 (gemm_batched_ex) —— 在 A 基础上追加 ─────────────────
// │ │ int32_t batchCount; int32_t totalTasks; int32_t cElemSize;
// │ └────────────────────────────────────────────────────────────────────
// │ ┌ 形态 C：分组 (gemm_grouped_batched_ex) —— 改用 GM 平铺结构 ─────────
// │ │ struct TilingHeader { uint32_t groupCount; uint32_t problemCount;
// │ │   uint32_t totalCubeTasks; uint32_t totalEpilogueTasks; ... };
// │ │ struct GroupData { int32_t m,n,k,lda,ldb,ldc,isTransA,isTransB,mBlocks,...;
// │ │   uint32_t batchStart, batchCount, cubeTaskStart, cubeTaskCount, ...; };
// │ │ → host 把 [Header | GroupData[]] malloc 到 GM，kernel 从 GM 读 tiling
// │ └────────────────────────────────────────────────────────────────────
// │ ┌ 形态 D：向量 (scalex) —— 完全不同的字段集 ─────────────────────────
// │ │ uint32_t totalN; uint32_t perCoreN; uint32_t remainder; uint32_t tileSize;
// │ │ float alpha; uint32_t alphaIsDevice; int64_t incx; uint32_t nthreads;
// │ │ uint32_t numBlocks; uint32_t xType;
// │ └────────────────────────────────────────────────────────────────────
