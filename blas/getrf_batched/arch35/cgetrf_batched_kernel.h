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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cgetrf_batched_tiling_data.h"

// Execution configuration and storage limits.
namespace CgetrfConfig {
constexpr uint32_t KIB_BYTES = 1024;
constexpr uint32_t FLOAT_BYTES = sizeof(float);
constexpr uint32_t INT_BYTES = sizeof(int32_t);
constexpr uint32_t COMPLEX_COMPONENTS = 2;
constexpr uint32_t COMPLEX_BYTES = COMPLEX_COMPONENTS * FLOAT_BYTES;
static_assert(FLOAT_BYTES == 4 && INT_BYTES == 4, "Single precision storage");
// Compile-time execution and copy layout for this arch35 implementation.
constexpr uint32_t VECTOR_FLOATS = 64;
constexpr uint32_t WARP_THREADS = 32;
constexpr uint32_t SIMT_GROUPS = 8;
constexpr uint32_t SIMT_THREADS = SIMT_GROUPS * WARP_THREADS;
constexpr uint32_t COPY_BLOCK_BYTES = 32;
constexpr uint32_t COPY_BLOCK_INTS = COPY_BLOCK_BYTES / INT_BYTES;
constexpr uint32_t COPY_BLOCK_FLOATS = COPY_BLOCK_BYTES / FLOAT_BYTES;
constexpr uint32_t ROW_PADDING = COPY_BLOCK_BYTES / COMPLEX_BYTES;
constexpr uint32_t VECTOR_COMPLEX_FLOATS = COMPLEX_COMPONENTS * VECTOR_FLOATS;
constexpr uint32_t VECTOR_COMPLEX_BYTES = VECTOR_FLOATS * COMPLEX_BYTES;
constexpr uint32_t MAX_PANEL_COLUMNS = WARP_THREADS;
constexpr uint32_t META_WORDS = 4 * MAX_PANEL_COLUMNS;
constexpr uint32_t META_BYTES = META_WORDS * INT_BYTES;
constexpr uint32_t PANEL_INFO_SLOT = MAX_PANEL_COLUMNS;
constexpr uint32_t PANEL_ZERO_SLOT = 2 * MAX_PANEL_COLUMNS;
// Preserve reserved gaps in the existing metadata layout, independently of n.
constexpr uint32_t META_SWAP_SLOT = 114;
constexpr uint32_t META_INFO_SLOT = (META_SWAP_SLOT + COPY_BLOCK_INTS) / COPY_BLOCK_INTS * COPY_BLOCK_INTS;
constexpr uint32_t SMALL_INFO_SLOT = VECTOR_FLOATS;
constexpr uint32_t SMALL_RESULT_WORDS = SMALL_INFO_SLOT + COPY_BLOCK_INTS;
constexpr uint32_t SMALL_SCRATCH_WORDS = VECTOR_FLOATS + WARP_THREADS;
constexpr uint32_t SMALL_PIVOT_SLOT = COPY_BLOCK_INTS;
constexpr uint32_t PACKED_MIN_ROWS = WARP_THREADS / 2;
constexpr uint32_t PACKED_MAX_ROWS = WARP_THREADS;
constexpr uint32_t PACKED_GROUPS = SIMT_GROUPS;
constexpr uint32_t PACKED_STRIDE = PACKED_GROUPS * PACKED_MAX_ROWS + ROW_PADDING;
constexpr uint32_t PACKED_INFO_SLOT = PACKED_MAX_ROWS;
constexpr uint32_t PACKED_RESULT_WORDS = PACKED_MAX_ROWS + PACKED_MIN_ROWS;
constexpr uint32_t PAIR_GROUPS = 2;
// Reserved pivot slots, NOT a dispatch cutoff; the actual limit is budget-derived.
constexpr uint32_t PAIR_INFO_SLOT = 160;
constexpr uint32_t PAIR_RESULT_ALIGNMENT = 2 * COPY_BLOCK_INTS;
constexpr uint32_t PAIR_RESULT_WORDS =
    (PAIR_INFO_SLOT + PAIR_RESULT_ALIGNMENT) / PAIR_RESULT_ALIGNMENT * PAIR_RESULT_ALIGNMENT;
// Budgets retained from the validated implementation, not queried hardware limits.
// Static reserves are conservative allowances, not claims of exact compiler usage.
// Revalidate resource usage before changing these values or the compiler version.
constexpr uint32_t UB_BUDGET_BYTES = 224 * KIB_BYTES;
constexpr uint32_t STREAM_DYNAMIC_BYTES = 212 * KIB_BYTES;
constexpr uint32_t STREAM_STATIC_RESERVE_BYTES = 8256;
constexpr uint32_t PACKED_STATIC_RESERVE_BYTES = 16512;
constexpr uint32_t SMALL_DYNAMIC_BYTES = 40 * KIB_BYTES;
constexpr uint32_t PACKED_DYNAMIC_BYTES = 80 * KIB_BYTES;
constexpr uint32_t FALLBACK_DYNAMIC_BYTES = 160 * KIB_BYTES;
// IEEE-754 binary32 encodings, shared by all special-value paths.
constexpr uint32_t FP32_ABS_MASK = 0x7fffffffU;
constexpr uint32_t FP32_INFINITY_BITS = 0x7f800000U;
constexpr int32_t NO_PIVOT_INDEX = INT32_MAX;
static_assert(PANEL_ZERO_SLOT + MAX_PANEL_COLUMNS <= META_SWAP_SLOT, "Disjoint metadata");
static_assert(META_SWAP_SLOT < META_INFO_SLOT && META_INFO_SLOT < META_WORDS, "Control slots fit");
static_assert(PAIR_INFO_SLOT < PAIR_RESULT_WORDS, "Pair info fits");
static_assert(SMALL_INFO_SLOT < SMALL_RESULT_WORDS, "Small info fits");
static_assert(
    VECTOR_FLOATS * VECTOR_COMPLEX_BYTES + SMALL_SCRATCH_WORDS * INT_BYTES + SMALL_RESULT_WORDS * INT_BYTES <=
        SMALL_DYNAMIC_BYTES,
    "Small buffers fit");
} // namespace CgetrfConfig

namespace CgetrfFamilyPolicy {
constexpr uint32_t VECTOR_ROWS = CgetrfConfig::VECTOR_FLOATS;
constexpr uint32_t ROW_QUANTUM = 2 * VECTOR_ROWS;
// Largest streamed capacity; larger inputs use the general global-memory path.
constexpr uint32_t MAX_ROWS = 32 * ROW_QUANTUM;
constexpr uint32_t PANEL_COLUMNS = CgetrfConfig::MAX_PANEL_COLUMNS;
// A resident matrix needs no extra global reloads when the serial front is narrowed.
constexpr uint32_t RESIDENT_PANEL_COLUMNS = PANEL_COLUMNS / 2;
constexpr uint32_t DYNAMIC_BYTES = CgetrfConfig::STREAM_DYNAMIC_BYTES;
constexpr uint32_t STATIC_RESERVE = CgetrfConfig::STREAM_STATIC_RESERVE_BYTES;
constexpr uint32_t TOTAL_BYTES = CgetrfConfig::UB_BUDGET_BYTES;
constexpr uint32_t PAIR_OUTPUT_INTS = CgetrfConfig::PAIR_RESULT_WORDS;
constexpr uint32_t PAIR_STRIDE = 2 * VECTOR_ROWS + CgetrfConfig::ROW_PADDING;
constexpr uint32_t PAIR_METADATA_BYTES =
    CgetrfConfig::PAIR_GROUPS * (CgetrfConfig::META_BYTES + PAIR_OUTPUT_INTS * CgetrfConfig::INT_BYTES);
constexpr uint32_t PAIR_MAX_N =
    (DYNAMIC_BYTES - PAIR_METADATA_BYTES) / (CgetrfConfig::PAIR_GROUPS * PAIR_STRIDE * CgetrfConfig::COMPLEX_BYTES);
static_assert(
    PAIR_MAX_N <= CgetrfConfig::PAIR_INFO_SLOT && PAIR_MAX_N <= PAIR_STRIDE - CgetrfConfig::ROW_PADDING,
    "Pair rows and output slots fit");
constexpr uint32_t ResidentRows(uint32_t n) { return (n + VECTOR_ROWS - 1) / VECTOR_ROWS * VECTOR_ROWS; }
constexpr bool ResidentSupported(uint32_t n)
{
    return n > VECTOR_ROWS && n <= MAX_ROWS &&
           static_cast<uint64_t>(n) * (ResidentRows(n) + CgetrfConfig::ROW_PADDING) * CgetrfConfig::COMPLEX_BYTES +
                   CgetrfConfig::META_BYTES <=
               DYNAMIC_BYTES;
}
constexpr uint32_t ResidentLimit()
{
    uint32_t n = VECTOR_ROWS + 1;
    while (ResidentSupported(n))
        ++n;
    return n - 1;
}
constexpr uint32_t RESIDENT_MAX_N = ResidentLimit();
static_assert(RESIDENT_MAX_N <= 3 * VECTOR_ROWS, "Resident launch covers all capacity classes");
constexpr uint32_t PanelColumns(uint32_t rows)
{
    const uint32_t available =
        (DYNAMIC_BYTES - CgetrfConfig::META_BYTES) / ((rows + CgetrfConfig::ROW_PADDING) * CgetrfConfig::COMPLEX_BYTES);
    uint32_t panel = PANEL_COLUMNS;
    // Reserve at least half a panel of space for the trailing block.
    while (panel + panel / 2 > available)
        panel /= 2;
    return panel;
}
constexpr uint32_t SlabColumns(uint32_t rows)
{
    const uint32_t available =
        (DYNAMIC_BYTES - CgetrfConfig::META_BYTES) / ((rows + CgetrfConfig::ROW_PADDING) * CgetrfConfig::COMPLEX_BYTES);
    const uint32_t panel = PanelColumns(rows);
    if (available <= panel)
        return 0;
    uint32_t slab = 1;
    while (2 * slab <= available - panel)
        slab *= 2;
    return slab;
}
constexpr uint32_t StorageBytes(uint32_t rows)
{
    return (PanelColumns(rows) + SlabColumns(rows)) * (rows + CgetrfConfig::ROW_PADDING) * CgetrfConfig::COMPLEX_BYTES +
           CgetrfConfig::META_BYTES;
}
constexpr bool StreamSupported(uint32_t n) { return n > VECTOR_ROWS && n <= MAX_ROWS; }
static_assert(DYNAMIC_BYTES + STATIC_RESERVE <= TOTAL_BYTES, "Static storage is reserved");
} // namespace CgetrfFamilyPolicy

// Pivot selection helpers.
namespace CgetrfPivot {
using namespace CgetrfConfig;
// Compatibility interval observed with the cuBLAS 120803 reference.
// Not a hardware limit or a shape-specific performance threshold.
constexpr uint32_t REFERENCE_PANEL_COLUMNS = 16;
// Nonnegative IEEE-754 values have the same ordering as their unsigned bits.
// Do not let NaN enter a maximum followed by an equality-based index reduction.
__simt_callee__ inline uint32_t MagnitudeKey(float real, float imag)
{
    const uint32_t r = __float_as_uint(real) & FP32_ABS_MASK;
    const uint32_t i = __float_as_uint(imag) & FP32_ABS_MASK;
    const uint32_t sum = __float_as_uint(__uint_as_float(r) + __uint_as_float(i)) & FP32_ABS_MASK;
    // NaN keys exceed infinity, so the existing maximum also detects NaNs.
    return sum;
}
template <class MatrixPtr>
__simt_callee__ inline uint32_t OrderedNanPivot(
    MatrixPtr matrix, uint32_t stride, uint32_t col, uint32_t first, uint32_t n, uint32_t lane)
{
    uint32_t best = 0;
    uint32_t index = first;
    const uint32_t start = first + lane;
    if (start < n) {
        const uint64_t at = COMPLEX_COMPONENTS * (start + static_cast<uint64_t>(col) * stride);
        best = MagnitudeKey(matrix[at], matrix[at + 1]);
        index = start;
    }
    for (uint32_t row = start + WARP_THREADS; row < n; row += WARP_THREADS) {
        const uint64_t at = COMPLEX_COMPONENTS * (row + static_cast<uint64_t>(col) * stride);
        const uint32_t value = MagnitudeKey(matrix[at], matrix[at + 1]);
        if (value <= FP32_INFINITY_BITS && value > best) {
            best = value;
            index = row;
        }
    }
    // cuBLAS observations match this ordered 32-lane comparison tree.
    // A NaN on the left retains that candidate; a NaN on the right cannot win.
    for (uint32_t delta = (WARP_THREADS / 2); delta != 0; delta >>= 1) {
        const uint32_t other = asc_shfl_down(best, delta);
        const uint32_t otherIndex = asc_shfl_down(index, delta);
        if (lane + delta < WARP_THREADS && other <= FP32_INFINITY_BITS &&
            (other > best || (other == best && otherIndex < index))) {
            best = other;
            index = otherIndex;
        }
    }
    return asc_shfl(index, 0);
}
__simt_callee__ inline bool IsZero(float real, float imag)
{
    return ((__float_as_uint(real) | __float_as_uint(imag)) & FP32_ABS_MASK) == 0;
}
} // namespace CgetrfPivot

// Common complex arithmetic and small-matrix helpers.
namespace CgetrfDetail {
using namespace AscendC;
using namespace CgetrfConfig;

// The blocked cuBLAS reference accumulates trailing updates in 16-column
// panels. Its complex beta=(1,0) multiplication is an identity for finite
// inputs, but the zero cross-products must still propagate Inf/NaN.
constexpr uint32_t REFERENCE_PANEL_COLUMNS = CgetrfPivot::REFERENCE_PANEL_COLUMNS;
__simd_callee__ inline void ApplyComplexBetaOne(
    Reg::RegTensor<float>& real, Reg::RegTensor<float>& imag, Reg::MaskReg& active, Reg::MaskReg& valid)
{
    Reg::RegTensor<float> zeroReal;
    Reg::RegTensor<float> zeroImag;
    Reg::Muls(zeroReal, real, 0.0f, active);
    Reg::Muls(zeroImag, imag, 0.0f, active);
    Reg::Sub(real, real, zeroImag, valid);
    Reg::Add(imag, imag, zeroReal, valid);
}

template <bool ComplexBeta>
__simd_callee__ inline void UpdatePanelRow(
    __ubuf__ float* matrix, uint32_t row, uint32_t pivot, uint32_t stride, uint32_t firstCol, uint32_t endCol,
    Reg::RegTensor<float>& lr, Reg::RegTensor<float>& li, Reg::MaskReg& active, Reg::MaskReg& storeMask)
{
    Reg::RegTensor<float> ar, ai, ur, ui, pr, pi, tmp, rr, ii;
    for (uint16_t c = firstCol; c < endCol; ++c) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ur, matrix + 2 * (pivot + c * stride));
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ui, matrix + 2 * (pivot + c * stride) + 1);
        Reg::Mul(pr, lr, ur, active);
        Reg::Mul(tmp, li, ui, active);
        Reg::Sub(pr, pr, tmp, active);
        Reg::Mul(pi, lr, ui, active);
        Reg::Mul(tmp, li, ur, active);
        Reg::Add(pi, pi, tmp, active);
        auto* address = matrix + 2 * (row + c * stride);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, address);
        if constexpr (ComplexBeta)
            ApplyComplexBetaOne(ar, ai, active, storeMask);
        // Inactive products are zero (ZEROING arithmetic), so subtracting
        // them preserves completed rows without a second row selection.
        Reg::Sub(rr, ar, pr, storeMask);
        Reg::Sub(ii, ai, pi, storeMask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(address, rr, ii, storeMask);
    }
}

__simd_callee__ inline void RefinedDivide(
    Reg::RegTensor<float>& result, Reg::RegTensor<float>& numerator, Reg::RegTensor<float>& denominator,
    Reg::MaskReg& mask)
{
    Reg::RegTensor<float> negativeDen;
    Reg::RegTensor<float> residual;
    Reg::RegTensor<float> correction;
    Reg::MaskReg usable;
    residual = numerator;
    Reg::Div(result, numerator, denominator, mask);
    Reg::Muls(negativeDen, denominator, -1.0f, mask);
    Reg::MulAddDst(residual, result, negativeDen, mask);
    Reg::Div(correction, residual, denominator, mask);
    // An undefined residual (for example Inf-Inf) must not destroy the
    // original quotient. Finite inputs keep the same refinement arithmetic.
    Reg::Compare<float, CMPMODE::EQ>(usable, correction, correction, mask);
    Reg::Add<float, Reg::MaskMergeMode::MERGING>(result, result, correction, usable);
}

__simd_callee__ inline void ScaleComplexColumn(
    Reg::RegTensor<float>& lr, Reg::RegTensor<float>& li, Reg::RegTensor<float>& ar, Reg::RegTensor<float>& ai,
    Reg::RegTensor<float>& dr, Reg::RegTensor<float>& di, Reg::RegTensor<float>& den, Reg::MaskReg& active)
{
    Reg::RegTensor<float> tmp;
    Reg::Mul(lr, ar, dr, active);
    Reg::Mul(tmp, ai, di, active);
    Reg::Add(lr, lr, tmp, active);
    RefinedDivide(lr, lr, den, active);
    Reg::Mul(li, ai, dr, active);
    Reg::Mul(tmp, ar, di, active);
    Reg::Sub(li, li, tmp, active);
    RefinedDivide(li, li, den, active);
    Reg::Select(lr, lr, ar, active);
    Reg::Select(li, li, ai, active);
}

__simd_callee__ inline uint32_t SmallNanPivot(__ubuf__ float* matrix, __ubuf__ float* scratch, uint32_t col, uint32_t n)
{
    Reg::RegTensor<float> real, imag, magnitude;
    uint32_t count = VECTOR_FLOATS;
    Reg::MaskReg full = Reg::UpdateMask<float>(count);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(real, imag, matrix + VECTOR_COMPLEX_FLOATS * col);
    Reg::Abs(real, real, full);
    Reg::Abs(imag, imag, full);
    Reg::Add(magnitude, real, imag, full);
    Reg::StoreAlign(scratch, magnitude, full);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::SCALAR_LOAD>();
    auto* keys = reinterpret_cast<__ubuf__ uint32_t*>(scratch);
    auto* indices = keys + VECTOR_FLOATS;
    for (uint32_t lane = 0; lane < WARP_THREADS; ++lane) {
        const uint32_t start = col + lane;
        uint32_t best = start < n ? (keys[start] & FP32_ABS_MASK) : 0;
        uint32_t index = start < n ? start : col;
        for (uint32_t row = start + WARP_THREADS; row < n; row += WARP_THREADS) {
            const uint32_t value = keys[row] & FP32_ABS_MASK;
            if (value <= FP32_INFINITY_BITS && value > best) {
                best = value;
                index = row;
            }
        }
        keys[lane] = best;
        indices[lane] = index;
    }
    for (uint32_t delta = (WARP_THREADS / 2); delta != 0; delta >>= 1) {
        for (uint32_t lane = 0; lane < delta; ++lane) {
            const uint32_t other = keys[lane + delta], index = indices[lane + delta];
            if (other <= FP32_INFINITY_BITS && (other > keys[lane] || (other == keys[lane] && index < indices[lane]))) {
                keys[lane] = other;
                indices[lane] = index;
            }
        }
    }
    const uint32_t pivot = indices[0];
    Reg::LocalMemBar<Reg::MemType::SCALAR_STORE, Reg::MemType::VEC_STORE>();
    return pivot;
}

__simd_callee__ inline uint32_t SmallPivot(__ubuf__ float* matrix, __ubuf__ float* scratch, uint32_t col, uint32_t n)
{
    Reg::RegTensor<float> real, imag, absolute, maximum;
    Reg::RegTensor<int32_t> rows, result;
    uint32_t count = n;
    Reg::MaskReg valid = Reg::UpdateMask<float>(count);
    Reg::MaskReg active, equal, nan;
    Reg::Arange(rows, 0);
    Reg::Compares<int32_t, CMPMODE::GE>(active, rows, static_cast<int32_t>(col), valid);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(real, imag, matrix + VECTOR_COMPLEX_FLOATS * col);
    Reg::Abs(real, real, active);
    Reg::Abs(imag, imag, active);
    Reg::Add(absolute, real, imag, active);
    Reg::RegTensor<float> infinity;
    Reg::RegTensor<int32_t> invalid;
    Reg::Duplicate(infinity, __builtin_inff());
    Reg::Duplicate(invalid, -1);
    Reg::Compare<float, CMPMODE::NE>(nan, absolute, absolute, active);
    // The existing reductions also flag a NaN column, without a separate scan.
    Reg::Select(absolute, infinity, absolute, nan);
    Reg::Select(rows, invalid, rows, nan);
    Reg::Reduce<Reg::ReduceType::MAX>(maximum, absolute, active);
    uint32_t oneCount = 1;
    Reg::MaskReg one = Reg::UpdateMask<float>(oneCount);
    Reg::Duplicate(maximum, maximum, active);
    Reg::Compare<float, CMPMODE::EQ>(equal, absolute, maximum, active);
    Reg::Reduce<Reg::ReduceType::MIN>(result, rows, equal);
    Reg::StoreAlign(reinterpret_cast<__ubuf__ int32_t*>(scratch) + SMALL_PIVOT_SLOT, result, one);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::SCALAR_LOAD>();
    const int32_t pivot = reinterpret_cast<__ubuf__ int32_t*>(scratch)[SMALL_PIVOT_SLOT];
    return pivot < 0 ? SmallNanPivot(matrix, scratch, col, n) : static_cast<uint32_t>(pivot);
}

__simd_callee__ inline void SmallSwap(__ubuf__ float* matrix, uint32_t n, uint32_t row, uint32_t pivot)
{
    if (row != pivot) {
        for (uint16_t col = 0; col < n; ++col) {
            uint32_t a = VECTOR_COMPLEX_FLOATS * col + 2 * row;
            uint32_t b = VECTOR_COMPLEX_FLOATS * col + 2 * pivot;
            float real = matrix[a];
            float imag = matrix[a + 1];
            matrix[a] = matrix[b];
            matrix[a + 1] = matrix[b + 1];
            matrix[b] = real;
            matrix[b + 1] = imag;
        }
        Reg::LocalMemBar<Reg::MemType::SCALAR_STORE, Reg::MemType::VEC_LOAD>();
    }
}
__simd_vf__ inline void SmallFactorSimd(
    __ubuf__ float* matrix, __ubuf__ float* scratch, __ubuf__ int* pivots, uint32_t n, uint32_t usePivot)
{
    Reg::RegTensor<float> real, imag, lr, li, tmp, den;
    Reg::RegTensor<int32_t> rows;
    Reg::Arange(rows, 0);
    uint32_t count = n;
    Reg::MaskReg valid = Reg::UpdateMask<float>(count);
    Reg::MaskReg active;
    int firstZero = 0;
    for (uint16_t col = 0; col < n; ++col) {
        uint32_t pivot = usePivot ? SmallPivot(matrix, scratch, col, n) : col;
        pivots[col] = static_cast<int>(pivot) + 1;
        uint32_t pidx = VECTOR_COMPLEX_FLOATS * col + 2 * pivot;
        float dr = matrix[pidx];
        float di = matrix[pidx + 1];
        auto* bits = reinterpret_cast<volatile __ubuf__ uint32_t*>(matrix);
        if (((bits[pidx] | bits[pidx + 1]) & FP32_ABS_MASK) == 0) {
            if (firstZero == 0)
                firstZero = col + 1;
            continue;
        }
        SmallSwap(matrix, n, col, pivot);
        Reg::Compares<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(col), valid);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(real, imag, matrix + VECTOR_COMPLEX_FLOATS * col);
        Reg::Duplicate(den, dr);
        Reg::Muls(den, den, dr, active);
        Reg::Duplicate(tmp, di);
        Reg::Muls(tmp, tmp, di, active);
        Reg::Add(den, den, tmp, active);
        Reg::Muls(lr, real, dr, active);
        Reg::Muls(tmp, imag, di, active);
        Reg::Add(lr, lr, tmp, active);
        RefinedDivide(lr, lr, den, active);
        Reg::Muls(li, imag, dr, active);
        Reg::Muls(tmp, real, di, active);
        Reg::Sub(li, li, tmp, active);
        RefinedDivide(li, li, den, active);
        Reg::Select(lr, lr, real, active);
        Reg::Select(li, li, imag, active);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(matrix + VECTOR_COMPLEX_FLOATS * col, lr, li, valid);
        if ((col + 1) % REFERENCE_PANEL_COLUMNS == 0)
            UpdatePanelRow<true>(matrix, 0, col, VECTOR_FLOATS, col + 1, n, lr, li, active, valid);
        else
            UpdatePanelRow<false>(matrix, 0, col, VECTOR_FLOATS, col + 1, n, lr, li, active, valid);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::SCALAR_LOAD>();
    }
    pivots[SMALL_INFO_SLOT] = firstZero;
}

__aicore__ inline void StoreSmallResults(
    LocalTensor<int> pivot, __gm__ int* pivots, __gm__ int* infos, uint32_t batch, uint32_t n, bool usePivot)
{
    if (infos != nullptr) {
        GlobalTensor<int> globalInfo;
        globalInfo.SetGlobalBuffer(infos + batch);
        DataCopyPad(globalInfo, pivot[SMALL_INFO_SLOT], DataCopyExtParams{1, sizeof(int), 0, 0, 0});
    }
    if (usePivot) {
        GlobalTensor<int> globalPivot;
        globalPivot.SetGlobalBuffer(pivots + static_cast<uint64_t>(batch) * n);
        DataCopyPad(globalPivot, pivot, DataCopyExtParams{1, n * INT_BYTES, 0, 0, 0});
    }
}

__aicore__ inline void RunSmallSimd(
    GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos, uint32_t startBatch, uint32_t numBatch,
    const CgetrfBatchedTilingData& tiling)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf, scratchBuf, pivotBuf;
    pipe.InitBuffer(matrixBuf, VECTOR_FLOATS * VECTOR_COMPLEX_BYTES);
    pipe.InitBuffer(scratchBuf, SMALL_SCRATCH_WORDS * sizeof(uint32_t));
    pipe.InitBuffer(pivotBuf, SMALL_RESULT_WORDS * sizeof(int));
    auto matrix = matrixBuf.Get<float>();
    auto scratch = scratchBuf.Get<float>();
    auto pivot = pivotBuf.Get<int>();
    uint32_t n = tiling.n;
    for (uint32_t b = 0; b < numBatch; ++b) {
        uint32_t batch = startBatch + b;
        uint64_t address = reinterpret_cast<__gm__ uint64_t*>(aarray)[batch];
        GlobalTensor<float> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(address));
        uint32_t ubStride = (VECTOR_COMPLEX_BYTES -
                             ((n * COMPLEX_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES) * COPY_BLOCK_BYTES) /
                            COPY_BLOCK_BYTES;
        DataCopyPad(
            matrix, global,
            DataCopyExtParams{
                static_cast<uint16_t>(n), n * COMPLEX_BYTES, (tiling.lda - n) * COMPLEX_BYTES, ubStride, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<SmallFactorSimd>(
            reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(scratch.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int*>(pivot.GetPhyAddr()), n, tiling.usePivot);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(
            global, matrix,
            DataCopyExtParams{
                static_cast<uint16_t>(n), n * COMPLEX_BYTES, ubStride, (tiling.lda - n) * COMPLEX_BYTES, 0});
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        StoreSmallResults(pivot, pivots, infos, batch, n, tiling.usePivot);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}
} // namespace CgetrfDetail
