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
#include "cann_ops_blas_common.h"

constexpr uint32_t CHPMV_GROUP_SIZE = 32;
constexpr uint32_t CHPMV_THREADS = 256;
constexpr uint32_t CHPMV_STAGE_THREADS = 1024;
constexpr uint32_t CHPMV_TILE_ROWS = 64;
constexpr uint32_t CHPMV_TILE_COLS = 256;
// 每列补 4 个复数，保持 32B 对齐并分散 SIMT 转置写的 bank group。
constexpr uint32_t CHPMV_STAGE_PITCH = CHPMV_TILE_ROWS + 4U;
constexpr uint32_t CHPMV_STAGE_BYTES = CHPMV_STAGE_PITCH * CHPMV_TILE_COLS * 8U;
constexpr uint32_t CHPMV_X_BYTES = CHPMV_TILE_COLS * 16U + 512U;
constexpr uint32_t CHPMV_ACC_BYTES = CHPMV_TILE_ROWS * 16U;

struct ChpmvTilingData {
    uint32_t n;
    uint32_t uplo;
    uint32_t threads;
    uint32_t strategy;
    int64_t incx;
    int64_t incy;
    aclblasComplex alpha;
    aclblasComplex beta;
};
