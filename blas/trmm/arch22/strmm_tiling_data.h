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

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

struct StrmmGemmTilingData {
    CBlas3GemmTilingData gemm;
    uint32_t triangleOnRight; // C^T = B^T * op(A)^T for original LEFT.
    uint32_t lowerTriangle;  // Orientation of op(A)^T, after the transpose.
};

struct StrmmPackTilingData {
    uint32_t dim;
    uint32_t lda;
    uint32_t packedLd;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    uint32_t bRows; // Zero disables the optional aligned B copy.
    uint32_t bCols;
    uint32_t ldb;
    uint32_t packedBld;
};

struct StrmmScaleTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t tempLd;
    float alpha;
};

struct StrmmDirectTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t side;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    float alpha;
};

struct StrmmReferenceTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t tempLd;
    uint32_t side;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    float alpha;
};
