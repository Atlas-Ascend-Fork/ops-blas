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
 * \file cgemm_kernel.h
 * \brief Kernel launcher signatures for CGEMM on arch22, shared by the host and
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
#include "cgemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): one complex operand -> its two packed real planes. The shared
// implementation in common/helper/complex_blas3_arch22.h; negateImag folds the
// conjugation of trans = 'C' into the imaginary plane.
void cgemm_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR ar, GM_ADDR ai, CBlas3SplitTilingData tiling);

// Phase 1b (AIV): accumulate one real product plane into re or im.
void cgemm_accum_kernel_do(uint32_t blockDim, void* stream, GM_ADDR dst, GM_ADDR src, CgemmAccumTilingData tiling);

// Phase 2 (AIV): assemble the complex result from the two real product planes.
void cgemm_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR re, GM_ADDR im, GM_ADDR c, CgemmCombineTilingData tiling);
