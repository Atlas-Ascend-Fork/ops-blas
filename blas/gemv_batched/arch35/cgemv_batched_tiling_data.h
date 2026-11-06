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
 * \file cgemv_batched_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for complex
 *        batched GEMV (cgemv_batched) on arch35 (Ascend 950PR).
 *        y[b] = alpha * op(A[b]) * x[b] + beta * y[b],  op = N / T / C
 */

#pragma once

#include <cstdint>

// UB budget for arch35 AIV (190 KiB). Reserved for complex slabs.
constexpr uint32_t CGEMV_BATCHED_UBUF_SIZE = 190 * 1024;
// Complex element = 2 * float32 = 8 bytes.
constexpr uint32_t CGEMV_BATCHED_CPLX_BYTES = 2u * sizeof(float);
// 32-byte alignment granularity in complex elements (32B / 8B = 4).
constexpr uint32_t CGEMV_BATCHED_ALIGN_CPLX = 4u;

// Core dispatch: outputs are split into batchPerCore-sized slices per core;
// the last core's range is derived in-kernel from usedCoreNum/totalOutputs
// (issue #850: the batchTail field was removed — it was computed and logged
// on the host but never consumed by the cgemv_batched kernel, unlike the
// gemv_batched sister implementation which reads its own tiling's batchTail).
#pragma pack(push, 4)
struct CgemvBatchedTilingData {
    uint32_t trans;        // 0 = N, 1 = T, 2 = C (conjugate transpose)
    float alphaR;          // scalar alpha real part
    float alphaI;          // scalar alpha imaginary part
    float betaR;           // scalar beta real part
    float betaI;           // scalar beta imaginary part
    uint32_t m;            // matrix A rows
    uint32_t n;            // matrix A columns
    uint32_t lda;          // leading dimension (in complex elements)
    int32_t incx;          // x stride (may be negative)
    int32_t incy;          // y stride (may be negative)
    uint32_t outSize;      // output vector length (N->m, T/C->n)
    uint32_t dotSize;      // reduction length      (N->n, T/C->m)
    uint32_t batchCount;   // number of batches
    uint32_t coreNum;      // platform AIV core count
    uint32_t usedCoreNum;  // cores actually used
    uint32_t batchPerCore; // batches per core (base)
    uint32_t tileRed;      // reduction tile size (complex elements)
    uint32_t alphaIsZero;  // alpha == (0,0): scale-only path
    uint32_t betaIsZero;   // beta == (0,0): y input not read
    // 2D decomposition for trans=N: every core owns ALL output rows of one
    // batch and a slice of the reduction (column) dimension, so each DMA
    // burst reads long contiguous column runs instead of short strided
    // fragments. The per-slice partial sums are merged with GM atomics.
    uint32_t colSplits;   // column slices per batch (1 = disabled)
    uint32_t colPerSplit; // reduction columns owned by one slice
};
#pragma pack(pop)
