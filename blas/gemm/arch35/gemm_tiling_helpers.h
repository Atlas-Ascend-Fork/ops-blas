/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <utility>

// 共享 tiling 初始化辅助（模板化）：csymm 与 gemm 各自维护独立的 GemmTilingData
// 布局（csymm 额外带 kStart/kEnd/kSegmentCount/maxTilesPerCore，且基准 tile 常量取值
// 不同），因此这里只依赖两者共有字段做赋值，不定义任何 struct 或常量，避免双重定义。
// 仅做字段赋值，不改变运行时行为。调用方 TU 须已定义 GEMM_C0_SIZE（两处取值均为 8）。
template <typename T>
static inline void InitGemmTilingBase(T& t, int m, int n, int k,
                                      int lda, int ldb, int ldc)
{
    t.m = m;
    t.n = n;
    t.k = k;
    t.lda = lda;
    t.ldb = ldb;
    t.ldc = ldc;
    t.cLdc = ldc;
}

template <typename T>
static inline void FillGemmTilingScalars(T& t,
                                         float alphaReal, float alphaImag,
                                         float betaReal, float betaImag,
                                         int isTransA, int isTransB)
{
    t.c0Size = GEMM_C0_SIZE;
    t.isTransA = isTransA;
    t.isTransB = isTransB;
    t.alphaReal = alphaReal;
    t.alphaImag = alphaImag;
    t.betaReal = betaReal;
    t.betaImag = betaImag;
    t.hasBeta = (betaReal != 0.0f || betaImag != 0.0f) ? 1 : 0;
}

template <typename T>
static inline void ApplyColMajorSwap(T& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}
