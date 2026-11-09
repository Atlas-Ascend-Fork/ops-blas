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

// CrotTilingData carries the runtime parameters for aclblasCrot.
// Complex64 is stored in GM as two interleaved floats {real, imag}, so the
// kernel addresses element `idx` through float offsets 2*idx and 2*idx+1.
// Strides may be positive, negative or zero; int32_t is required to carry the
// sign. Negative strides are resolved inside the kernel with (n-1-i)*|inc|;
// spans whose (n-1)*|inc| exceeds the int32 range are rejected on the host.
// c/s pointer location: c (float) and s (complex64) may independently be host
// or device pointers (queried on the host via aclrtPointerGetAttributes). For a
// host pointer the host dereferences it once and broadcasts the scalar through
// cosValue / sinRe / sinIm (placeholder 0 when the pointer is on device). For a
// device pointer the host forwards the pointer verbatim and the kernel reads the
// value from GM, so no host dereference / D2H copy / stream sync is needed.
// Two code paths, selected by tilingKey on the host.
//   tilingKey == 0: contiguous path (incx == incy == 1) -> SIMD membase. Uses
//                   perCoreN/remainder; the per-core slice is walked in UB tiles
//                   of CROT_TILE_COMPLEX elements.
//   tilingKey == 1: strided path (any other incx/incy, including 0 and negative)
//                   -> SIMT, uses incx/incy/nthreads.
struct CrotTilingData {
    uint32_t tilingKey; // 0 = contiguous (SIMD), 1 = strided (SIMT)
    uint32_t totalN;    // number of complex elements to rotate
    int32_t incx;       // x stride (positive / negative / zero)
    int32_t incy;       // y stride (positive / negative / zero)
    uint32_t nthreads;  // SIMT threads per block
    // 1 = force the single-thread serial path even without a zero stride. Set by the host
    // when x and y share the same base pointer but use different strides, because then two
    // parallel iterations can touch the same element and the result would depend on the
    // thread schedule instead of matching the sequential golden.
    uint32_t serialOnly;
    float cosValue;     // Givens cosine c (host value; 0 placeholder when cIsDevice)
    float sinRe;        // real part of s (host value; 0 placeholder when sIsDevice)
    float sinIm;        // imag part of s (host value; 0 placeholder when sIsDevice)
    uint32_t cIsDevice; // 1 = c is a device pointer (kernel reads cPtr from GM)
    uint32_t sIsDevice; // 1 = s is a device pointer (kernel reads sPtr from GM)
    // ===== contiguous path fields (valid when tilingKey == 0) =====
    uint32_t perCoreN;  // complex elements per core (n / numBlocks)
    uint32_t remainder; // leading cores that carry one extra element
};

// Contiguous-path UB tile: complex elements per GM<->UB transfer. The kernel
// allocates 14 tile-sized float buffers (aosX/aosY hold 2 each - they are also
// reused as the interleaved output staging - and the eight real/imag results
// plus two scratch buffers hold 1 each), so 14 * 4096 * 4B = 224 KiB fits under
// UB_SIZE (248 KiB).
constexpr uint32_t CROT_TILE_COMPLEX = 4096U;
