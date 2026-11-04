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

// CsrotTilingData carries the runtime parameters for both code paths of aclblasCsrot:
//   tilingKey == 0: contiguous path (SIMD membase), uses perCoreN/remainder/tileSize
//   tilingKey == 1: strided path (SIMT), uses incx/incy
// int32_t is required for incx/incy to carry negative strides.
//
// All element counts (totalN / perCoreN / remainder / tileSize) are in COMPLEX
// units: one aclblasComplex = 2 float lanes (real at the even lane, imag at the
// odd lane). The contiguous kernel operates on the float-lane view, so one
// complex element occupies CSROT_FLOATS_PER_COMPLEX lanes.
//
// c/s pointer location: c and s may each independently be a host pointer or a
// device pointer (queried on the host via aclrtPointerGetAttributes). When a
// pointer is on the host, the host dereferences it once and forwards the scalar
// via cosValue/sinValue (scalar broadcast in the kernel). When a pointer is on
// the device, the host forwards the device pointer verbatim as a GM_ADDR to the
// kernel (no dereference, no D2H copy, no stream sync) and fills the scalar
// field with a 0.0f placeholder; the kernel then reads the value from GM.
// cIsDevice/sIsDevice select which source each path uses. The two are independent
// (c on host + s on device is legal), unlike rotg which requires all-or-nothing.
struct CsrotTilingData {
    // ===== common fields (both paths) =====
    uint32_t tilingKey; // 0 = contiguous (SIMD), 1 = strided (SIMT)
    uint32_t totalN;    // number of complex elements to rotate
    float cosValue;     // Givens cosine c (host value; 0.0f placeholder when cIsDevice)
    float sinValue;     // Givens sine s   (host value; 0.0f placeholder when sIsDevice)
    uint32_t cIsDevice; // 1 = c is a device pointer (kernel reads cPtr from GM), 0 = host scalar
    uint32_t sIsDevice; // 1 = s is a device pointer (kernel reads sPtr from GM), 0 = host scalar

    // ===== contiguous path fields (valid when tilingKey == 0) =====
    uint32_t perCoreN;  // complex elements per core (even split, remainder handled via `remainder`)
    uint32_t remainder; // extra elements assigned to the front cores
    uint32_t tileSize;  // UB tile size in complex elements, aligned to CSROT_ELEMENTS_PER_BLOCK

    // ===== strided path fields (valid when tilingKey == 1) =====
    int32_t incx;      // x stride (positive / negative / zero)
    int32_t incy;      // y stride (positive / negative / zero)
    uint32_t nthreads; // SIMT threads/block: ceilDiv(n,numBlocks) rounded up to
                       // SIMT_MIN_THREAD_NUM, capped at SIMT_MAX_THREAD_NUM
};

// complex64 = 2 interleaved floats (real, imag); matches sizeof(aclblasComplex).
constexpr uint32_t CSROT_FLOATS_PER_COMPLEX = 2;
constexpr uint32_t CSROT_ELEMENT_SIZE = CSROT_FLOATS_PER_COMPLEX * sizeof(float);
// complex64 elements per 32-byte block (= 4)
constexpr uint32_t CSROT_ELEMENTS_PER_BLOCK = 32 / CSROT_ELEMENT_SIZE;
