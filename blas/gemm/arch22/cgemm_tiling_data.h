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
 * \file cgemm_tiling_data.h
 * \brief CGEMM-specific tiling for arch22. The de-interleave stage reuses
 *        CBlas3SplitTilingData from the shared complex BLAS-3 layer; only the
 *        epilogue differs between operators.
 */

#pragma once

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2 (AIV): C = alpha * (re + i*im) + beta * C_old over the whole m x n
// output, with re and im the two real products held in the workspace.
//   Cr = ar*re - ai*im + br*Cr_old - bi*Ci_old
//   Ci = ar*im + ai*re + br*Ci_old + bi*Cr_old
// Unlike the rank-K operators there is no triangle: every element of the m x n
// output is written, so the epilogue walks the full matrix column by column.
struct CgemmCombineTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;     // leading dimension of C, in complex elements
    uint32_t ldRe;    // row stride of the real product plane, in floats
    uint32_t ldIm;    // row stride of the imaginary product plane, in floats
    uint32_t hasTemp; // alpha == 0 or k == 0: the product planes are not valid
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
};

// Phase 1b (AIV): accumulate one real product plane into re or im.
//   overwrite != 0 : dst = scale * src
//   overwrite == 0 : dst += scale * src
// Keeps every delegated aclblasSgemm call on beta == 0 so arch22 CGEMM never
// enters sgemm_arch22_epilogue's beta != 0 path (ops-blas#1077).
struct CgemmAccumTilingData {
    uint64_t totalFloats; // ldRe * n (always a multiple of 16 floats = 64B)
    float scale;
    uint32_t overwrite;
};
