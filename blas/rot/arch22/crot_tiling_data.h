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

// aclblasCrot (arch22 / Atlas A2/A3) runtime tiling. Mirrors the flat GM layout used by
// the sibling arch22 rot operator (csrot) so the kernel can read the whole struct with
// one reinterpret_cast over the tiling GM buffer.
//
// Complex64 is stored in GM as two interleaved floats {real, imag}, so the kernel
// addresses logical element `idx` through float offsets 2*idx and 2*idx+1. elementCount
// therefore carries the FLOAT component count 2*n (the same convention csrot uses; passing
// n instead would only rotate the first half of the elements).
//
// Strides may be positive or negative. Zero stride and INT32_MIN are rejected on the host
// (arch22 contract, issue #827), so the kernel never needs a zero-stride serial path; a
// negative stride is resolved inside the kernel as (totalN - 1 - logical) * |inc|.
//
// c (float) and s (complex64) are resolved on the host via aclrtPointerGetAttributes and
// stored as plain scalars here (host dereference for host pointers, D2H copy for device
// pointers). The kernel therefore never reads a scalar from GM.
struct CrotTilingData {
    int32_t elementCount; // float component count = 2 * n
    int32_t incx;         // x stride (positive / negative; 0 and INT32_MIN rejected on host)
    int32_t incy;         // y stride (same contract)
    float cosValue;       // Givens cosine c (real)
    float sinRe;          // real part of s
    float sinIm;          // imag part of s
};

// Contiguous SIMD tile: complex elements per GM<->UB transfer.
// The contiguous path allocates 2 interleaved tiles (x/y) plus 10 planar real/imag
// buffers (inputs, results and two scratch) plus 2 re-interleave masks; at 2048 the whole
// working set is 144 KiB, under the 910B 192 KiB UB budget, while halving the number of
// per-tile pipeline sync points versus a 1024-element tile.
constexpr uint32_t CROT_TILE_COMPLEX = 2048U;

// Strided scalar path: complex elements processed per GM staging pass.
constexpr uint32_t CROT_STRIDED_ELEMS_PER_PASS = 1024U;

// Per-element 32B-aligned UB staging slot width (floats) for the strided DataCopyPad store.
constexpr uint32_t CROT_STRIDED_SLOT = 8U;
