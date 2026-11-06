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
#include "gemm_ex_tiling.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include <algorithm>

// Host-only launch selection. Compute kernels and their translation units stay
// separate for GemmEx and SgemmEx; workspace/shape planning is shared.
struct RealBackendKernels {
    void (*catlassCopy)(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
    void (*catlass)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*catlassPack)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, GemmExTiling);
    void (*columnOwner)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*dmaPack)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, GemmExTiling);
    void (*moderateNz)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*nzBitcopy)(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*nzFourrowCopy)(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*nzLoopCopy)(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*shortNative)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
    void (*shortNzNative)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
    void (*splitNt)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
    void (*splitSum)(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t, uint32_t);
    void (*ttShort)(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
};

aclblasStatus_t RunGemmExRealBackend(
    const RealBackendKernels& kernels, aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta,
    float* c, int ldc);

extern "C" aclblasStatus_t SgemmExDispatchBackend(
    aclblasHandle_t, aclblasOperation_t, aclblasOperation_t, int, int, int, const float*, const float*, int,
    const float*, int, const float*, float*, int);
extern "C" aclblasStatus_t GemmExArch22Backend(
    aclblasHandle_t, aclblasOperation_t, aclblasOperation_t, int, int, int, const float*, const float*, int,
    const float*, int, const float*, float*, int);
extern "C" aclblasStatus_t GemmExPlanarBackend(
    aclblasHandle_t, aclblasOperation_t, aclblasOperation_t, int, int, int, const float*, const float*, int,
    const float*, int, const float*, float*, int);
extern "C" aclblasStatus_t SgemmExArch22Backend(
    aclblasHandle_t, aclblasOperation_t, aclblasOperation_t, int, int, int, const float*, const float*, int,
    const float*, int, const float*, float*, int);

namespace gemm_ex {
constexpr int SHORT_NN_TILE_ROWS = 368;
constexpr int SHORT_NN_TILE_COLUMNS = 64;
struct RealDispatchParams {
    aclblasHandle_t handle;
    aclblasOperation_t ta;
    aclblasOperation_t tb;
    int m;
    int n;
    int k;
    const float* alpha;
    const float* a;
    int lda;
    const float* b;
    int ldb;
    const float* beta;
    float* c;
    int ldc;

    bool IsCompactIdentity() const
    {
        return *alpha == 1.0f && *beta == 0.0f && ldc == m && lda == (ta == ACLBLAS_OP_N ? m : k) &&
               ldb == (tb == ACLBLAS_OP_N ? k : n);
    }
    bool IsShortNn() const
    {
        // Two row tiles form an even/odd pair with opposite K-segment order.
        // Dispatch additionally checks the grid and output groups against the
        // available Cube/AIV cores; no measured-case window is required.
        return ta == ACLBLAS_OP_N && tb == ACLBLAS_OP_N && m > SHORT_NN_TILE_ROWS && m <= 2 * SHORT_NN_TILE_ROWS &&
               n >= 16 && k > 16 && k <= 32 && IsCompactIdentity();
    }
    uint32_t ShortNnGrid() const
    {
        return ((static_cast<uint32_t>(n) + SHORT_NN_TILE_COLUMNS - 1) / SHORT_NN_TILE_COLUMNS) *
               ((m + SHORT_NN_TILE_ROWS - 1) / SHORT_NN_TILE_ROWS);
    }
    bool IsSmallTt() const
    {
        return ta != ACLBLAS_OP_N && tb != ACLBLAS_OP_N && m >= 16 && n >= 16 && n <= 96 && k > 32 && k <= 2048 &&
               m <= 2048 && !(std::max(m, n) <= 256 && k <= 128) && (m % 16 != 0 || n % 16 != 0 || k % 32 != 0) &&
               IsCompactIdentity();
    }
    GemmExTiling MakeTiling(uint32_t cores) const
    {
        GemmExTiling t{};
        t.m = m;
        t.n = n;
        t.k = k;
        t.lda = lda;
        t.ldb = ldb;
        t.ldc = ldc;
        t.transA = ta == ACLBLAS_OP_N ? 0 : 1;
        t.transB = tb == ACLBLAS_OP_N ? 0 : 1;
        t.alpha = *alpha;
        t.beta = *beta;
        t.cubeBlocks = cores;
        return t;
    }
};

// Share host argument/workspace preparation, while each interface supplies its
// own independently named compute and output kernels.
template <typename Multiply, typename Finish>
inline aclblasStatus_t RunShortNn(
    const RealDispatchParams& p, uint32_t cores, uint32_t vectors, uint32_t grid, Multiply multiply, Finish finish)
{
    const auto t = p.MakeTiling(cores);
    const auto status = EnsureDefaultWorkspace(
        p.handle, static_cast<size_t>(grid) * SHORT_NN_TILE_COLUMNS * SHORT_NN_TILE_ROWS * sizeof(float));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* work = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(p.handle));
    multiply(
        grid, p.handle->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(p.a)),
        reinterpret_cast<uint8_t*>(const_cast<float*>(p.b)), work, t, 0);
    finish(vectors, p.handle->stream, work, reinterpret_cast<uint8_t*>(p.c), t);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace gemm_ex
