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
 * \file ctpsv_tiling_data.h
 * \brief Tiling data for single-precision complex triangular packed solver.
 */

#pragma once

#include <cstdint>

struct CtpsvTilingData {
    uint64_t ap;         // device pointer to packed A (complex64)
    uint64_t x;          // device pointer to x vector (complex64)
    uint32_t n;
    uint32_t uplo;       // ACLBLAS_UPPER or ACLBLAS_LOWER
    uint32_t trans;      // ACLBLAS_OP_N, ACLBLAS_OP_T, or ACLBLAS_OP_C
    uint32_t diag;       // ACLBLAS_NON_UNIT or ACLBLAS_UNIT
    int64_t incx;
    uint32_t numThreads; // SIMT thread count per block (only for SIMT path, 0 for scalar)
    uint32_t numBlocks;  // SIMT block count (grid size)
};
