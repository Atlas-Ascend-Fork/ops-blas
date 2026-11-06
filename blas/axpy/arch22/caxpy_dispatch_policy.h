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

enum class CaxpyStrideClass : uint32_t {
    DENSE = 0,
    X_ONLY_POSITIVE = 1,
    Y_ONLY_POSITIVE = 2,
    DUAL_SYMMETRIC_POSITIVE = 3,
    DUAL_ASYMMETRIC_POSITIVE = 4,
    NEGATIVE_MIXED = 5,
};

enum class CaxpyKernelVariant : uint32_t {
    DENSE_PIPELINED_ATOMIC = 0,
    STRIDED_SCALAR = 1,
    STRIDED_SHARED_PIPELINED_ATOMIC = 2,
    STRIDED_DISJOINT_PIPELINED_ATOMIC = 3,
    STRIDED_SHARED_PIPELINED_SCALAR_TAIL = 4,
    STRIDED_DISJOINT_PIPELINED_SCALAR_TAIL = 5,
};

constexpr uint64_t CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK = 16;
constexpr uint64_t CAXPY_STRIDED_MAX_PER_BLOCK_PER_WAVE = 8192;
// Fixed tile capacity used by the disjoint pipeline and as the shared
// pipeline's UB buffer allocation size (see CAXPY_STRIDED_TILE_COUNT in
// caxpy_kernel.cpp, which must stay equal to this value).
constexpr uint64_t CAXPY_STRIDED_TILE_COUNT = 512;
// The shared pipeline's lowScratch UB scratch buffer used for the raw span
// read is 38 KiB (CAXPY_DENSE_TILE_BYTES in caxpy_kernel.cpp). A span read of
// `tile` complex elements at increment `incx` covers
// ((tile-1)*|incx|+1) complex elements == ((tile-1)*|incx|+1)*8 bytes of
// physical address range, which must fit in that buffer. Solving for the
// largest tile <= CAXPY_STRIDED_TILE_COUNT that satisfies this bound gives
// the runtime tile capacity the shared pipeline should use for a given incx;
// this generalizes the old fixed CAXPY_SHARED_X_SPAN_MAX_INCREMENT=9 cutoff
// (which was really just "does tile=512 fit"), letting larger |incx| still
// use the faster span-read shared pipeline instead of falling back to
// disjoint sparse reads, as long as the shrunk tile stays usable.
constexpr uint64_t CAXPY_SHARED_LOW_SCRATCH_BYTES = 38 * 1024;
constexpr uint64_t CAXPY_COMPLEX_BYTES = 8;

// Largest tile (in complex elements, capped at CAXPY_STRIDED_TILE_COUNT) whose
// span read at the given |incx| fits in the shared pipeline's lowScratch
// buffer. absIncx must be >= 1 (callers only invoke this for direct-strided,
// non-dense cases where incx != 0).
constexpr uint64_t CaxpySharedTileCapacity(uint64_t absIncx)
{
    const uint64_t maxSpanComplexElements = CAXPY_SHARED_LOW_SCRATCH_BYTES / CAXPY_COMPLEX_BYTES;
    // tile such that (tile-1)*absIncx + 1 <= maxSpanComplexElements
    //             => tile <= (maxSpanComplexElements - 1) / absIncx + 1
    const uint64_t maxTileForIncrement = (maxSpanComplexElements - 1) / absIncx + 1;
    return maxTileForIncrement < CAXPY_STRIDED_TILE_COUNT ? maxTileForIncrement : CAXPY_STRIDED_TILE_COUNT;
}

// The shared pipeline's tail-splitting logic (shared_tile_bounds in
// caxpy_kernel.cpp) rebalances the last two tiles when the natural final tile
// would be shorter than CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK+1
// (CAXPY_INTERLEAVE_COUNT+1 in caxpy_kernel.cpp, same constant) elements, by
// splitting their combined length roughly in half. For that split to never
// itself produce a half shorter than the interleave width, the worst case
// (combined length == tileCapacity+1, the smallest total that still triggers
// the split branch) must divide into two halves each >=
// CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK+1: ceil((tileCapacity+1)/2) and
// floor((tileCapacity+1)/2) both >= CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK+1,
// which holds starting at tileCapacity == 2*(CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK+1)-1.
// A smaller usable-capacity threshold (originally CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK+1
// itself) let the split produce a too-short half, caught by the
// dynamic_shared_boundary_usable regression test (n=8000, incx=303, tile=17)
// only in a Release build where the resulting silent wrong compute went
// uncaught -- a shrunk-tile-capacity variant of the same class of bug as the
// earlier fixed-tile computeOffset drift.
constexpr uint64_t CAXPY_SHARED_MIN_USABLE_TILE_CAPACITY = 2 * (CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK + 1) - 1;

// CAXPY_SHARED_MIN_USABLE_TILE_CAPACITY only bounds *correctness*: below it
// the tail-split can produce too-short tiles. It says nothing about
// performance. As |incx| grows, the shrunk tile's span read
// ((tile-1)*|incx|+1 complex elements) carries a shrinking fraction of useful
// data -- at incx=16 the span is ~6% useful, at incx=100 it is ~1% -- so GM
// bandwidth spent reading skipped-over elements eventually exceeds what the
// disjoint pipeline's per-element sparse reads cost instead. Measured on
// Ascend 910B3 / CANN 9.0 A3, n=1048576 (xlarge, enough tiles per core to
// reach steady state and not be dominated by per-tile dispatch overhead):
// shared span-read is faster than disjoint through incx<=60, comparable
// around incx=60-65, and progressively worse beyond that (roughly +7% at
// incx=70, +97% at incx=150 in one measured run). This cutoff is therefore a
// measured performance crossover, not a UB-buffer-fit or correctness limit;
// re-measure if the UB budget, tile capacity formula, or disjoint pipeline's
// per-element cost changes.
constexpr uint64_t CAXPY_SHARED_SPAN_PERFORMANCE_MAX_INCREMENT = 60;

constexpr uint64_t CaxpyFinalWaveCount(uint64_t n, uint32_t numBlocks)
{
    const uint64_t waveCapacity = static_cast<uint64_t>(numBlocks) * CAXPY_STRIDED_MAX_PER_BLOCK_PER_WAVE;
    if (n <= waveCapacity)
        return n;
    const uint64_t remainder = n % waveCapacity;
    return remainder == 0 ? waveCapacity : remainder;
}

// Preconditions are enforced by aclblasCaxpy argument validation and execution
// setup: incx and incy are non-zero and are not INT64_MIN; numBlocks is non-zero.
constexpr CaxpyStrideClass ClassifyCaxpyStride(int64_t incx, int64_t incy)
{
    if (incx < 0 || incy < 0)
        return CaxpyStrideClass::NEGATIVE_MIXED;
    if (incx == 1 && incy == 1)
        return CaxpyStrideClass::DENSE;
    if (incy == 1)
        return CaxpyStrideClass::X_ONLY_POSITIVE;
    if (incx == 1)
        return CaxpyStrideClass::Y_ONLY_POSITIVE;
    if (incx == incy)
        return CaxpyStrideClass::DUAL_SYMMETRIC_POSITIVE;
    return CaxpyStrideClass::DUAL_ASYMMETRIC_POSITIVE;
}

constexpr uint64_t CaxpyAbsIncrement(int64_t increment)
{
    return increment > 0 ? static_cast<uint64_t>(increment) : static_cast<uint64_t>(-(increment + 1)) + 1;
}

constexpr bool UseCaxpySharedXSpanPipeline(int64_t incx)
{
    const uint64_t absIncx = CaxpyAbsIncrement(incx);
    return absIncx <= CAXPY_SHARED_SPAN_PERFORMANCE_MAX_INCREMENT &&
           CaxpySharedTileCapacity(absIncx) >= CAXPY_SHARED_MIN_USABLE_TILE_CAPACITY;
}

constexpr CaxpyKernelVariant SelectCaxpyKernelVariant(
    uint64_t n, int64_t incx, int64_t incy, bool directStrided, uint32_t numBlocks)
{
    const CaxpyStrideClass strideClass = ClassifyCaxpyStride(incx, incy);
    if (strideClass == CaxpyStrideClass::DENSE || !directStrided)
        return CaxpyKernelVariant::DENSE_PIPELINED_ATOMIC;
    const bool sharedXSpan = UseCaxpySharedXSpanPipeline(incx);
    const uint64_t finalWaveCount = CaxpyFinalWaveCount(n, numBlocks);
    const uint64_t finalWaveBlocks = finalWaveCount < numBlocks ? finalWaveCount : numBlocks;
    const uint64_t minBlockCount = finalWaveCount / finalWaveBlocks;
    if (minBlockCount <= CAXPY_STRIDED_SCALAR_MAX_PER_BLOCK)
        return n == finalWaveCount ? CaxpyKernelVariant::STRIDED_SCALAR :
                                     (sharedXSpan ? CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_SCALAR_TAIL :
                                                    CaxpyKernelVariant::STRIDED_DISJOINT_PIPELINED_SCALAR_TAIL);
    return sharedXSpan ? CaxpyKernelVariant::STRIDED_SHARED_PIPELINED_ATOMIC :
                         CaxpyKernelVariant::STRIDED_DISJOINT_PIPELINED_ATOMIC;
}
