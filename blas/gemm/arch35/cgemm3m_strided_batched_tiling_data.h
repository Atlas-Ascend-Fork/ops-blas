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
struct Cgemm3mSbTiling {
    uint32_t m, n, k, lda, ldb, ldc, transa, transb, batch;
    int64_t strideA, strideB, strideC;
    float ar, ai, br, bi;
};
void cgemm3m_sb_direct_do(
    uint32_t blocks, void* stream, const void* a, const void* b, void* c, const Cgemm3mSbTiling& tiling);

constexpr uint32_t CGEMM3M_PANEL_ROWS = 4;
constexpr uint32_t CGEMM3M_PANEL_COLS = 64;
void cgemm3m_sb_panel_do(
    uint32_t blocks, void* stream, const void* a, const void* b, void* c, const Cgemm3mSbTiling& tiling);
