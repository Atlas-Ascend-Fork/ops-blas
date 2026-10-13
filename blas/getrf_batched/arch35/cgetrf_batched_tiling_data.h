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
 * \file cgetrf_batched_tiling_data.h
 * \brief Tiling data for batched single-precision complex LU factorization.
 */

#pragma once

#include <cstdint>

struct CgetrfBatchedTilingData {
    uint32_t n;            // matrix dimension (square matrix side length)
    uint32_t lda;          // leading dimension
    uint32_t usedCoreNum;  // actually used Core count
    uint32_t batchPerCore; // batches per core (except last core)
    uint32_t batchTail;    // batches for the last core
    uint32_t usePivot;     // 1 = use pivoting, 0 = no pivoting (PivotArray == NULL)
};
