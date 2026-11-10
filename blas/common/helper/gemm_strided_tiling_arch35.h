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
 * \file gemm_strided_tiling_arch35.h
 * \brief Host/kernel contract for the DAV-3510 strided-batched GEMM implementation.
 */

#pragma once

#include <cstdint>

enum GemmStridedBatchedExDtypeCase : int32_t {
    GEMM_STRIDED_DTYPE_FP16_F16 = 0,
    GEMM_STRIDED_DTYPE_FP16_F32 = 1,
    GEMM_STRIDED_DTYPE_BF16_BF16 = 2,
    GEMM_STRIDED_DTYPE_BF16_F32 = 3,
    GEMM_STRIDED_DTYPE_FP32_F32 = 4,
    GEMM_STRIDED_DTYPE_FP8_E4M3_F16 = 5,
    GEMM_STRIDED_DTYPE_FP8_E5M2_F16 = 6,
    GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F16 = 7,
    GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F16 = 8,
    GEMM_STRIDED_DTYPE_FP8_E4M3_F32 = 9,
    GEMM_STRIDED_DTYPE_FP8_E5M2_F32 = 10,
    GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F32 = 11,
    GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F32 = 12,
    GEMM_STRIDED_DTYPE_INT8_I32 = 13,
    GEMM_STRIDED_DTYPE_INT8_F32 = 14,
    GEMM_STRIDED_DTYPE_COMPLEX32_F32 = 15,
    GEMM_STRIDED_DTYPE_INVALID = -1,
};

enum GemmStridedBatchedExScalarKind : int32_t {
    GEMM_STRIDED_SCALAR_F16 = 0,
    GEMM_STRIDED_SCALAR_F32 = 1,
    GEMM_STRIDED_SCALAR_I32 = 2,
};

enum GemmStridedBatchedExKernelKind : int32_t {
    GEMM_STRIDED_KERNEL_BALANCED = 0,
    GEMM_STRIDED_KERNEL_BATCH_FIRST = 1,
    GEMM_STRIDED_KERNEL_M_MAJOR = 2,
    GEMM_STRIDED_KERNEL_N_MAJOR = 3,
    GEMM_STRIDED_KERNEL_SMALL = 4,
    GEMM_STRIDED_KERNEL_DEEP_K = 5,
    GEMM_STRIDED_KERNEL_SPLIT_K = 6,
    GEMM_STRIDED_KERNEL_PERSISTENT = 7,
};

struct GemmStridedBatchedExTilingData {
    // Cube sees the column-major GEMM through the standard A/B and M/N swap.
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t logicalM;
    int32_t logicalN;

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
    int32_t batchCount;
    int32_t selectedAlgo;
    int32_t splitK;
    int32_t hasBeta;
    int32_t scalarKind;
    int32_t dtypeCase;

    uint64_t totalTasks;
    int64_t strideA;
    int64_t strideB;
    int64_t strideC;
    int64_t workspaceBatchStride;
    int64_t workspaceSplitStride;

    float alpha;
    float beta;
    float alphaImag;
    float betaImag;
    int32_t alphaInt;
    int32_t betaInt;
};
