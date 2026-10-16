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

// Host -> kernel 单次下发参数（与 stpmv 同构：SIMT 网格由 numBlocks/numThreads 决定）
struct CtpmvTilingData {
    uint32_t n;          // 矩阵阶数
    uint32_t uplo;       // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint32_t trans;      // ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C
    uint32_t diag;       // ACLBLAS_NON_UNIT / ACLBLAS_UNIT
    int64_t incx;        // x 步长（支持负值，尾部反向）
    uint32_t numThreads; // SIMT 每 block 线程数
};
