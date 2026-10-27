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
 * \brief Tiling data for cgeru (arch22): complex64 non-conjugate rank-1 update.
 */

#ifndef CGERU_TILING_DATA_ARCH22_H
#define CGERU_TILING_DATA_ARCH22_H

#include <cstdint>

// Row-chunk cap for the arch22 kernel. The TPipe managed UB layout keeps three
// 2*chunk-float TQue buffers (x in / A in / A out), six chunk-float scratch
// planes, the (2*chunk uint32) interleave table and its two (2*chunk int32)
// build temporaries:
//   3*2*(2*chunk)*4 + 6*chunk*4 + 2*chunk*4 + 2*(2*chunk)*4 <= 192 KiB
// => chunk = 1024 uses 96 KiB, leaving 96 KiB of headroom for TPipe alignment.
constexpr uint32_t CGERU_MAX_CHUNK_ROWS = 1024;
// Interleave table covers a full max-size chunk: idx[2t] = 4t, idx[2t+1] = 4*(chunk + t).
constexpr uint32_t CGERU_INTERLEAVE_ENTRIES = 2 * CGERU_MAX_CHUNK_ROWS;

// Tile descriptor handed to the kernel. The host validates arguments, folds
// negative strides into xStart/yStart, and splits columns evenly across cores.
struct CgeruTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
    uint64_t xStart; // complex-element offset of x[0] (accounts for incx < 0)
    uint64_t yStart; // complex-element offset of y[0] (accounts for incy < 0)
    uint32_t colsPerCore;
};

#endif // CGERU_TILING_DATA_ARCH22_H
