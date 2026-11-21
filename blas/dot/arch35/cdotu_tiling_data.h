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

// Cdotu tiling data structure.
//
// The per-core workload partition (startIdx / calNum) is NOT stored here as arrays:
// it is recomputed inside the kernel from n / useCoreNum / blockIdx using the
// even-aligned partition (see cdotu_kernel.cpp), matching sdot/dotex which pass only
// useCoreNum plus strides and let each core derive its own [start, count). This keeps
// the tiling payload a fixed handful of scalars instead of a per-core 512-byte array,
// and removes the compile-time 64-core assumption (the core count is discovered at
// runtime via GetAivCoreCount).
//
// useDualAcc: enable the dual accumulator for n >= 3M (the two halves of each float4
// feed independent FMA chains), halving the serial dependency length - measured to keep
// float32 accumulation error within 1e-2 and run faster than a single accumulator; for
// n < 3M the single accumulator is optimal (no extra overhead).
struct CdotuTilingData {
    int32_t n;                               // logical element count (complex)
    int32_t incx;                            // x stride (0 already rejected by host)
    int32_t incy;                            // y stride (0 already rejected by host)
    uint32_t useCoreNum;                     // number of cores actually participating in the reduction
    uint32_t nthreads;                       // threads per block (128..1024)
    uint32_t useDualAcc;                     // 1 = dual accumulator, 0 = single accumulator
};