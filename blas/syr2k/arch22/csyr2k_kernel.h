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
 * \file csyr2k_kernel.h
 * \brief Kernel launcher signatures for CSYR2K on arch22, shared by the host and
 *        the kernel translation unit so that the compiler checks both against one
 *        declaration.
 *
 *        GM_ADDR is guarded: on the host side it falls back to a plain pointer,
 *        while the kernel translation unit includes kernel_operator.h first and
 *        therefore keeps AscendC's __gm__ qualified definition.
 */

#pragma once

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"
#include "csyr2k_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): complex A or B -> the two packed real planes. The shared
// implementation in common/helper/complex_blas3_arch22.h.
void csyr2k_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling);

// Phase 1 (AIC): one real GEMM over the packed planes. Dispatches on
// tiling.transMode, which the host derives from the public trans argument.
void csyr2k_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling);

// Phase 2 (AIV): assemble the complex symmetric result from the three temp planes.
void csyr2k_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR c, Csyr2kCombineTilingData tiling);

// The compile-time singleK of the Matmul static tiling, so the host can split K
// without duplicating the constant.
uint32_t csyr2k_gemm_single_k();
