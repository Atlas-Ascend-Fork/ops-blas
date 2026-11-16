/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_tiling_data.h
 * \brief Tiling descriptor for the arch22 cgeru kernel (complex64 rank-1 update).
 */

#pragma once

#include <cstdint>

// Kernel-side split strategy. Both paths partition columns across AIV cores with
// the balanced rule colStart = n * blockIdx / useCoreNum so no core carries more
// than one column more than any other.
//
//  - Fast path (useFast): incx == 1 && lda == m && m % 4 == 0 && m <= 4096 &&
//    ceil(n / useCoreNum) <= 256. Every A column then is a contiguous 32 B
//    aligned run, the core slab is one contiguous GM range, and the kernel
//    streams it in fixed-size chunks with double-buffered MTE2/V/MTE3 stages.
//  - General path: per-column pieces for strided x, padded lda, or shapes the
//    fast path cannot address.
struct CgeruTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    int64_t incx;
    int64_t incy;
    uint64_t xStart; // complex-element offset of logical x[0] (negative incx folded in)
    uint64_t yStart; // complex-element offset of logical y[0] (negative incy folded in)
    float alphaReal;
    float alphaImag;
    uint32_t useFast;
};
