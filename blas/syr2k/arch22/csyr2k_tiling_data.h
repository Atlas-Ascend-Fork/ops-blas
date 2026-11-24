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
 * \file csyr2k_tiling_data.h
 * \brief CSYR2K-specific tiling. Phases 0 and 1 reuse the shared structures in
 *        common/helper/complex_blas3_tiling_data.h; only the combine stage
 *        differs between operators.
 */

#pragma once

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2: assemble the complex symmetric result from the three temp planes.
// Phase 1 produces the three real products
//   T1 = Ar*Br^T + Br*Ar^T
//   T2 = Ai*Bi^T + Bi*Ai^T
//   T3 = Ar*Bi^T + Ai*Br^T + Br*Ai^T + Bi*Ar^T
// (trans='T' reads A^T/B instead of A/B; the product list is the same). All
// three are symmetric, so Phase 2 only visits the uplo triangle and its
// column-major (transposed) read of the row-major temp planes needs no pointer
// swap. With S = (A*B^T + B*A^T) = (T1 - T2) + i*T3:
//   Cr = alphaRe*(T1-T2) - alphaIm*T3 + betaRe*Cr_old - betaIm*Ci_old
//   Ci = alphaRe*T3 + alphaIm*(T1-T2) + betaRe*Ci_old + betaIm*Cr_old
// Unlike CHER2K the diagonal imaginary part is a genuine result value: C is
// symmetric, not Hermitian, so it must NOT be forced to zero.
struct Csyr2kCombineTilingData {
    uint32_t n;
    uint32_t ldc;      // leading dimension of C, in complex elements
    uint32_t tempLd;   // row stride of each temp plane, in floats
    uint32_t uploMode; // CBLAS3_UPLO_UPPER / CBLAS3_UPLO_LOWER
    uint32_t hasTemp;  // alpha == 0 or k == 0: no temp plane is valid
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
};
