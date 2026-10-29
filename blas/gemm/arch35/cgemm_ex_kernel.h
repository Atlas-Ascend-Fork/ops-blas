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
#include "gemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

void cgemm_ex_deinterleave_do(
    uint32_t numBlocks, void* stream, GM_ADDR src, GM_ADDR reOut, GM_ADDR imOut, int32_t physRows, int32_t physCols,
    int32_t lda, int32_t logicalRows, int32_t transFlag, int32_t conjFlag, GM_ADDR sumOut);

void cgemm_ex_deint2_do(
    uint32_t numBlocks, void* stream, GM_ADDR srcA, GM_ADDR reA, GM_ADDR imA, int32_t physRowsA, int32_t physColsA,
    int32_t ldaA, int32_t logicalRowsA, int32_t conjA, GM_ADDR sumA, GM_ADDR srcB, GM_ADDR reB, GM_ADDR imB,
    int32_t physRowsB, int32_t physColsB, int32_t ldaB, int32_t logicalRowsB, int32_t conjB, GM_ADDR sumB);

void cgemm_ex_cube_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmTilingData& tilingData);

void cgemm_ex_cube_small_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmTilingData& tilingData);

void cgemm_ex_cube3_do(
    uint32_t numBlocks, void* stream, GM_ADDR a1, GM_ADDR b1, GM_ADDR c1, GM_ADDR a2, GM_ADDR b2, GM_ADDR c2,
    GM_ADDR a3, GM_ADDR b3, GM_ADDR c3, const GemmTilingData& tilingData);

void cgemm_ex_mix_do(
    uint32_t numBlocks, void* stream, GM_ADDR a1, GM_ADDR b1, GM_ADDR c1, GM_ADDR a2, GM_ADDR b2, GM_ADDR c2,
    GM_ADDR a3, GM_ADDR b3, GM_ADDR c3, GM_ADDR cOut, int32_t m, int32_t n, int32_t tempLdc, int32_t ldc, float ar,
    float ai, float br, float bi, int32_t cIsComplex, const GemmTilingData& tilingData);

void cgemm_ex_neg_do(uint32_t numBlocks, void* stream, GM_ADDR src, GM_ADDR dst, int32_t count);

void cgemm_ex_combine3_do(
    uint32_t numBlocks, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, int32_t tempLdc, GM_ADDR cInOut, int32_t m,
    int32_t n, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex, int32_t kParts);

void cgemm_ex_small_c32_do(
    uint32_t numBlocks, void* stream, GM_ADDR A, int32_t lda, int32_t transA, int32_t conjA, GM_ADDR B, int32_t ldb,
    int32_t transB, int32_t conjB, GM_ADDR C, int32_t ldc, int32_t cIsComplex, int32_t m, int32_t n, int32_t k,
    float ar, float ai, float br, float bi);
