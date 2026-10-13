/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgetrf_batched_kernel.cpp
 * \brief Kernel-side implementation for batched single-precision complex LU factorization (SIMT).
 */

#include <cstdint>
#include "acl/acl_rt.h"
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cgetrf_batched_kernel.h"

// Packed small-matrix factorization.
namespace CgetrfPackedFamily {
using namespace AscendC;
using namespace CgetrfConfig;
constexpr uint32_t STRIDE = PACKED_STRIDE;
constexpr uint32_t RESULT_INTS = PACKED_RESULT_WORDS;
constexpr uint32_t INFO = PACKED_INFO_SLOT;
constexpr uint32_t DYNAMIC_BYTES = PACKED_DYNAMIC_BYTES;
template <uint32_t Rows>
constexpr uint32_t PACKS = PACKED_GROUPS * Rows / VECTOR_FLOATS;
static_assert(
    PACKED_MAX_ROWS * STRIDE * COMPLEX_BYTES + PACKED_GROUPS * RESULT_INTS * INT_BYTES <= DYNAMIC_BYTES,
    "Packed storage fits");
static_assert(DYNAMIC_BYTES + PACKED_STATIC_RESERVE_BYTES <= UB_BUDGET_BYTES, "Reserve static SIMT storage");

template <uint32_t Rows>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREADS) inline void SelectAndSwap(
    __ubuf__ float* packed, __ubuf__ int* result, uint32_t k, uint32_t usePivot, uint32_t n)
{
    const uint32_t group = threadIdx.x / WARP_THREADS, lane = threadIdx.x % WARP_THREADS;
    auto* matrix = packed + group * Rows * 2;
    auto* output = result + group * RESULT_INTS;
    uint32_t best = 0;
    int index = k;
    if (lane >= k && lane < n) {
        const uint32_t at = 2 * (lane + k * STRIDE);
        const float real = matrix[at], imag = matrix[at + 1];
        const uint32_t value = CgetrfPivot::MagnitudeKey(real, imag);
        if (value > best) {
            best = value;
            index = lane;
        }
    }
    const uint32_t largest = asc_reduce_max(best);
    uint32_t reduced = asc_reduce_min(best == largest ? index : NO_PIVOT_INDEX);
    if (largest > FP32_INFINITY_BITS)
        reduced = CgetrfPivot::OrderedNanPivot(matrix, STRIDE, k, k, n, lane);
    const uint32_t pivot = usePivot ? reduced : k;
    const uint32_t diag = 2 * (pivot + k * STRIDE);
    // A positive non-NaN maximum already proves the selected pivot is nonzero.
    // Zero, NaN and no-pivot columns still require the exact component check.
    const bool zero = (!usePivot || largest == 0 || largest > FP32_INFINITY_BITS) &&
                      CgetrfPivot::IsZero(matrix[diag], matrix[diag + 1]);
    if (lane == 0) {
        output[k] = pivot + 1;
        if (k == 0)
            output[INFO] = 0;
        if (zero && output[INFO] == 0)
            output[INFO] = k + 1;
    }
    if (!zero && pivot != k && lane < n) {
        const uint32_t a = 2 * (k + lane * STRIDE), b = 2 * (pivot + lane * STRIDE);
        const float real = matrix[a], imag = matrix[a + 1];
        matrix[a] = matrix[b];
        matrix[a + 1] = matrix[b + 1];
        matrix[b] = real;
        matrix[b + 1] = imag;
    }
    asc_syncthreads();
}

template <uint32_t Rows>
__simd_callee__ inline void Indices(
    Reg::RegTensor<int32_t>& rows, Reg::RegTensor<uint32_t>& diagonal, uint32_t k, Reg::MaskReg& full, uint32_t n)
{
    Reg::RegTensor<int32_t> lanes, shifted, offset, negative;
    Reg::RegTensor<uint32_t> next;
    Reg::MaskReg groupMask, invalid;
    Reg::Arange(lanes, 0);
    rows = lanes;
    Reg::Duplicate(diagonal, k);
    for (uint16_t group = 1; group < VECTOR_FLOATS / Rows; ++group) {
        Reg::Compares<int32_t, CMPMODE::GE>(groupMask, lanes, static_cast<int32_t>(group * Rows), full);
        Reg::Duplicate(offset, static_cast<int32_t>(group * Rows));
        Reg::Sub(shifted, lanes, offset, full);
        Reg::Select(rows, shifted, rows, groupMask);
        Reg::Duplicate(next, static_cast<uint32_t>(k + group * Rows));
        Reg::Select(diagonal, next, diagonal, groupMask);
    }
    Reg::Duplicate(negative, -1);
    Reg::Compares<int32_t, CMPMODE::GE>(invalid, rows, static_cast<int32_t>(n), full);
    Reg::Select(rows, negative, rows, invalid);
}

__simd_callee__ inline void UpdateTrailing(
    __ubuf__ float* matrix, uint32_t k, Reg::RegTensor<float>& lr, Reg::RegTensor<float>& li,
    Reg::RegTensor<uint32_t>& diagonal, Reg::MaskReg& full, Reg::MaskReg& active, uint32_t n)
{
    Reg::RegTensor<float> ar, ai, ur, ui, pr, pi, tmp, rr, ii;
    for (uint16_t c = k + 1; c < n; ++c) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, matrix + 2 * c * STRIDE);
        Reg::Gather(ur, ar, diagonal);
        Reg::Gather(ui, ai, diagonal);
        Reg::Mul(pr, lr, ur, active);
        Reg::Mul(tmp, li, ui, active);
        Reg::Sub(pr, pr, tmp, active);
        Reg::Mul(pi, lr, ui, active);
        Reg::Mul(tmp, li, ur, active);
        Reg::Add(pi, pi, tmp, active);
        // ZEROING arithmetic already makes inactive products zero.
        Reg::Sub(rr, ar, pr, full);
        Reg::Sub(ii, ai, pi, full);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(matrix + 2 * c * STRIDE, rr, ii, full);
    }
}

__simd_callee__ inline void ScaleAndUpdate(
    __ubuf__ float* matrix, uint32_t k, uint32_t n, Reg::RegTensor<int32_t>& baseRows,
    Reg::RegTensor<uint32_t>& diagonal, Reg::MaskReg& full)
{
    Reg::RegTensor<float> ar, ai, dr, di, lr, li, den, tmp, absReal, absImag;
    Reg::RegTensor<int32_t> rows, negative;
    Reg::MaskReg zero, active;
    rows = baseRows;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, matrix + 2 * k * STRIDE);
    Reg::Gather(dr, ar, diagonal);
    Reg::Gather(di, ai, diagonal);
    Reg::Abs(absReal, dr, full);
    Reg::Abs(absImag, di, full);
    Reg::Add(tmp, absReal, absImag, full);
    Reg::Compares<float, CMPMODE::EQ>(zero, tmp, 0.0f, full);
    Reg::Duplicate(negative, -1);
    Reg::Select(rows, negative, rows, zero);
    Reg::Compares<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(k), full);
    Reg::Mul(den, dr, dr, full);
    Reg::Mul(tmp, di, di, full);
    Reg::Add(den, den, tmp, full);
    CgetrfDetail::ScaleComplexColumn(lr, li, ar, ai, dr, di, den, active);
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(matrix + 2 * k * STRIDE, lr, li, full);
    UpdateTrailing(matrix, k, lr, li, diagonal, full, active, n);
}

template <uint32_t Rows>
__simd_vf__ inline void PackedStep(__ubuf__ float* matrix, uint32_t k, uint32_t n)
{
    Reg::RegTensor<int32_t> rows;
    Reg::RegTensor<uint32_t> diagonal;
    uint32_t count = VECTOR_FLOATS;
    Reg::MaskReg full = Reg::UpdateMask<float>(count);
    Indices<Rows>(rows, diagonal, k, full, n);
    for (uint16_t pack = 0; pack < PACKS<Rows>; ++pack) {
        ScaleAndUpdate(matrix + pack * VECTOR_COMPLEX_FLOATS, k, n, rows, diagonal, full);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <uint32_t Rows>
__aicore__ inline void LoadGroup(
    LocalTensor<float> matrix, GM_ADDR aarray, uint32_t batch, uint32_t validGroups, uint32_t lda, uint32_t n)
{
    const uint32_t gap =
        (STRIDE * COMPLEX_BYTES - (n * COMPLEX_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES * COPY_BLOCK_BYTES) /
        COPY_BLOCK_BYTES;
    for (uint32_t group = 0; group < PACKS<Rows> * VECTOR_FLOATS / Rows; ++group) {
        const uint32_t sourceGroup = group < validGroups ? group : validGroups - 1;
        GlobalTensor<float> global;
        global.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(aarray)[batch + sourceGroup]));
        // Unused tail slots repeat a valid input; never read outside the batch array.
        DataCopyPad(
            matrix[group * Rows * 2], global,
            DataCopyExtParams{static_cast<uint16_t>(n), n * COMPLEX_BYTES, (lda - n) * COMPLEX_BYTES, gap, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}

template <uint32_t Rows>
__aicore__ inline void StoreGroup(
    LocalTensor<float> matrix, LocalTensor<int> result, GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos,
    uint32_t batch, uint32_t validGroups, uint32_t lda, uint32_t usePivot, uint32_t n)
{
    const uint32_t gap =
        (STRIDE * COMPLEX_BYTES - (n * COMPLEX_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES * COPY_BLOCK_BYTES) /
        COPY_BLOCK_BYTES;
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    for (uint32_t group = 0; group < validGroups; ++group) {
        GlobalTensor<float> global;
        global.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(aarray)[batch + group]));
        DataCopyPad(
            global, matrix[group * Rows * 2],
            DataCopyExtParams{static_cast<uint16_t>(n), n * COMPLEX_BYTES, gap, (lda - n) * COMPLEX_BYTES, 0});
    }
    // UB strides count aligned copy blocks; GM strides count bytes.
    // Each group occupies RESULT_INTS words in UB, but outputs are dense in GM.
    constexpr uint32_t resultBlocks = RESULT_INTS * INT_BYTES / COPY_BLOCK_BYTES;
    static_assert(RESULT_INTS * INT_BYTES % COPY_BLOCK_BYTES == 0, "Aligned result groups");
    if (usePivot) {
        GlobalTensor<int> gp;
        gp.SetGlobalBuffer(pivots + static_cast<uint64_t>(batch) * n);
        const uint32_t pivotBlocks = (n * INT_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES;
        DataCopyPad(
            gp, result,
            DataCopyExtParams{static_cast<uint16_t>(validGroups), n * INT_BYTES, resultBlocks - pivotBlocks, 0, 0});
    }
    if (infos != nullptr) {
        GlobalTensor<int> gi;
        gi.SetGlobalBuffer(infos + batch);
        DataCopyPad(
            gi, result[INFO], DataCopyExtParams{static_cast<uint16_t>(validGroups), INT_BYTES, resultBlocks - 1, 0, 0});
    }
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

template <uint32_t Rows>
__aicore__ inline void Run(
    GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos, uint32_t start, uint32_t count,
    const CgetrfBatchedTilingData& tiling)
{
    static_assert(Rows == PACKED_MIN_ROWS || Rows == PACKED_MAX_ROWS, "Row capacities divide the vector length");
    const uint32_t n = tiling.n;
    constexpr uint32_t groups = PACKS<Rows> * VECTOR_FLOATS / Rows;
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf, resultBuf;
    pipe.InitBuffer(matrixBuf, n * STRIDE * COMPLEX_BYTES);
    pipe.InitBuffer(resultBuf, groups * RESULT_INTS * INT_BYTES);
    auto matrix = matrixBuf.Get<float>();
    auto result = resultBuf.Get<int>();
    auto* data = reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr());
    auto* output = reinterpret_cast<__ubuf__ int*>(result.GetPhyAddr());
    for (uint32_t b = 0; b < count; b += groups) {
        const uint32_t valid = count - b < groups ? count - b : groups;
        LoadGroup<Rows>(matrix, aarray, start + b, valid, tiling.lda, n);
        for (uint32_t k = 0; k < n; ++k) {
            asc_vf_call<SelectAndSwap<Rows>>(dim3{groups * WARP_THREADS, 1, 1}, data, output, k, tiling.usePivot, n);
            asc_vf_call<PackedStep<Rows>>(data, k, n);
        }
        StoreGroup<Rows>(matrix, result, aarray, pivots, infos, start + b, valid, tiling.lda, tiling.usePivot, n);
    }
}
} // namespace CgetrfPackedFamily

// Shared blocked-update helpers.
namespace CgetrfDetail {
using namespace AscendC;
using namespace CgetrfConfig;
// Disjoint metadata regions permit panel widths up to 32.
constexpr uint32_t PANEL_INFO = PANEL_INFO_SLOT;
constexpr uint32_t PANEL_ZERO = PANEL_ZERO_SLOT;
template <bool ComplexBeta>
__simd_callee__ inline void UpdateSlabColumn(
    __ubuf__ float* panel, __ubuf__ float* slab, uint32_t n, uint32_t stride, uint32_t first, uint32_t k,
    uint32_t slabWidth)
{
    Reg::RegTensor<float> lr, li, ar, ai, pr, pi, temp, ur, ui;
    Reg::RegTensor<int32_t> rows;
    uint32_t vectorCount = VECTOR_FLOATS;
    Reg::MaskReg full = Reg::UpdateMask<float>(vectorCount), active;
    for (uint16_t r = (first + k) / VECTOR_FLOATS * VECTOR_FLOATS; r < n; r += VECTOR_FLOATS) {
        Reg::Arange(rows, static_cast<int32_t>(r));
        uint32_t validCount = n - r;
        Reg::MaskReg valid = Reg::UpdateMask<float>(validCount);
        Reg::Compares<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(first + k), valid);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(lr, li, panel + 2 * (r + k * stride));
        for (uint16_t c = 0; c < slabWidth; ++c) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ur, slab + 2 * (first + k + c * stride));
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(ui, slab + 2 * (first + k + c * stride) + 1);
            Reg::Mul(pr, lr, ur, active);
            Reg::Mul(temp, li, ui, active);
            Reg::Sub(pr, pr, temp, active);
            Reg::Mul(pi, lr, ui, active);
            Reg::Mul(temp, li, ur, active);
            Reg::Add(pi, pi, temp, active);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, slab + 2 * (r + c * stride));
            if constexpr (ComplexBeta)
                ApplyComplexBetaOne(ar, ai, active, full);
            // Mul/Sub/Add use ZEROING: inactive products are zero. Full
            // subtraction preserves completed rows without extra selections.
            Reg::Sub(ar, ar, pr, full);
            Reg::Sub(ai, ai, pi, full);
            Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(slab + 2 * (r + c * stride), ar, ai, full);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::SCALAR_LOAD>();
}

__simd_vf__ inline void UpdateSlabSimd(
    __ubuf__ float* panel, __ubuf__ float* slab, __ubuf__ int* metadata, uint32_t n, uint32_t stride, uint32_t first,
    uint32_t panelWidth, uint32_t slabWidth)
{
    for (uint16_t k = 0; k < panelWidth; ++k) {
        if (metadata[PANEL_ZERO + k])
            continue;
        // Choose once per column; the inner update has no runtime branch and
        // keeps the complex beta operation in registers without a second pass.
        if ((first + k + 1) % REFERENCE_PANEL_COLUMNS == 0)
            UpdateSlabColumn<true>(panel, slab, n, stride, first, k, slabWidth);
        else
            UpdateSlabColumn<false>(panel, slab, n, stride, first, k, slabWidth);
    }
}

__aicore__ inline void CopySlabIn(
    LocalTensor<float> local, GlobalTensor<float> global, uint32_t n, uint32_t lda, uint32_t stride, uint32_t col,
    uint32_t width)
{
    const uint32_t localGap =
        (stride * COMPLEX_BYTES - (n * COMPLEX_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES * COPY_BLOCK_BYTES) /
        COPY_BLOCK_BYTES;
    DataCopyPad(
        local, global[COMPLEX_COMPONENTS * static_cast<uint64_t>(col) * lda],
        DataCopyExtParams{static_cast<uint16_t>(width), n * COMPLEX_BYTES, (lda - n) * COMPLEX_BYTES, localGap, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}

__aicore__ inline void CopySlabOut(
    GlobalTensor<float> global, LocalTensor<float> local, uint32_t n, uint32_t lda, uint32_t stride, uint32_t col,
    uint32_t width)
{
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    const uint32_t localGap =
        (stride * COMPLEX_BYTES - (n * COMPLEX_BYTES + COPY_BLOCK_BYTES - 1) / COPY_BLOCK_BYTES * COPY_BLOCK_BYTES) /
        COPY_BLOCK_BYTES;
    DataCopyPad(
        global[COMPLEX_COMPONENTS * static_cast<uint64_t>(col) * lda], local,
        DataCopyExtParams{static_cast<uint16_t>(width), n * COMPLEX_BYTES, localGap, (lda - n) * COMPLEX_BYTES, 0});
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

} // namespace CgetrfDetail

// Resident and streamed matrix factorization.
namespace CgetrfStreamFamily {
using namespace AscendC;
using namespace CgetrfConfig;
constexpr uint32_t SWAPPED = META_SWAP_SLOT;
constexpr uint32_t INFO_OUT = META_INFO_SLOT;
constexpr uint32_t DYNAMIC_BYTES = CgetrfFamilyPolicy::DYNAMIC_BYTES;

template <uint32_t RowCapacity>
__simt_callee__ inline uint32_t Pivot(__ubuf__ float* panel, uint32_t col, uint32_t k, uint32_t n)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    const uint32_t N = n;
    uint32_t best = 0;
    int row = static_cast<int>(k);
    for (uint32_t r = k + threadIdx.x; r < N; r += WARP_THREADS) {
        const uint32_t at = 2 * (r + col * STRIDE);
        const float real = panel[at], imag = panel[at + 1];
        const uint32_t value = CgetrfPivot::MagnitudeKey(real, imag);
        if (value > best) {
            best = value;
            row = static_cast<int>(r);
        }
    }
    const uint32_t largest = asc_reduce_max(best);
    uint32_t reduced = asc_reduce_min(best == largest ? row : NO_PIVOT_INDEX);
    if (largest > FP32_INFINITY_BITS)
        reduced = CgetrfPivot::OrderedNanPivot(panel, STRIDE, col, k, n, threadIdx.x);
    return reduced;
}

template <uint32_t RowCapacity>
__simt_vf__ __aicore__ LAUNCH_BOUND(WARP_THREADS) inline void SelectAndSwap(
    __ubuf__ float* panel, __ubuf__ int* meta, uint32_t first, uint32_t col, uint32_t width, uint32_t usePivot,
    uint32_t n)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    const uint32_t N = n;
    const uint32_t k = first + col;
    const uint32_t pivot = usePivot ? Pivot<RowCapacity>(panel, col, k, N) : k;
    const uint32_t diag = 2 * (pivot + col * STRIDE);
    const bool zero = CgetrfPivot::IsZero(panel[diag], panel[diag + 1]);
    if (threadIdx.x == 0) {
        if (col == 0) {
            meta[CgetrfDetail::PANEL_INFO] = 0;
            meta[SWAPPED] = 0;
        }
        if (!zero && pivot != k)
            meta[SWAPPED] = 1;
        meta[col] = static_cast<int>(pivot) + 1;
        meta[CgetrfDetail::PANEL_ZERO + col] = zero;
        if (zero && meta[CgetrfDetail::PANEL_INFO] == 0)
            meta[CgetrfDetail::PANEL_INFO] = k + 1;
    }
    if (!zero && pivot != k) {
        for (uint32_t c = threadIdx.x; c < width; c += WARP_THREADS) {
            const uint32_t a = 2 * (k + c * STRIDE), b = 2 * (pivot + c * STRIDE);
            const float real = panel[a], imag = panel[a + 1];
            panel[a] = panel[b];
            panel[a + 1] = panel[b + 1];
            panel[b] = real;
            panel[b + 1] = imag;
        }
    }
    asc_syncthreads();
}

template <uint32_t RowCapacity, uint32_t PanelColumns>
__simd_vf__ inline void ScaleAndUpdate(
    __ubuf__ float* panel, __ubuf__ int* meta, uint32_t first, uint32_t col, uint32_t width, uint32_t n)
{
    static_assert(PanelColumns > 0, "Positive panel width");
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    const uint32_t N = n;
    if (meta[CgetrfDetail::PANEL_ZERO + col])
        return;
    const uint32_t k = first + col;
    Reg::RegTensor<float> ar, ai, lr, li, dr, di, den, tmp;
    Reg::RegTensor<int32_t> rows;
    uint32_t fullCount = VECTOR_FLOATS;
    Reg::MaskReg full = Reg::UpdateMask<float>(fullCount), active;
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dr, panel + 2 * (k + col * STRIDE));
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(di, panel + 2 * (k + col * STRIDE) + 1);
    Reg::Mul(den, dr, dr, full);
    Reg::Mul(tmp, di, di, full);
    Reg::Add(den, den, tmp, full);
    for (uint16_t r = k / VECTOR_FLOATS * VECTOR_FLOATS; r < N; r += VECTOR_FLOATS) {
        uint32_t count = N - r;
        Reg::MaskReg valid = Reg::UpdateMask<float>(count);
        Reg::Arange(rows, static_cast<int32_t>(r));
        Reg::Compares<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(k), valid);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, panel + 2 * (r + col * STRIDE));
        CgetrfDetail::ScaleComplexColumn(lr, li, ar, ai, dr, di, den, active);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(panel + 2 * (r + col * STRIDE), lr, li, full);
        if constexpr (CgetrfDetail::REFERENCE_PANEL_COLUMNS % PanelColumns == 0) {
            CgetrfDetail::UpdatePanelRow<false>(panel, r, k, STRIDE, col + 1, width, lr, li, active, full);
        } else {
            if ((k + 1) % CgetrfDetail::REFERENCE_PANEL_COLUMNS == 0)
                CgetrfDetail::UpdatePanelRow<true>(panel, r, k, STRIDE, col + 1, width, lr, li, active, full);
            else
                CgetrfDetail::UpdatePanelRow<false>(panel, r, k, STRIDE, col + 1, width, lr, li, active, full);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// Replay swaps in original pivot order, each column owned by one thread.
template <uint32_t RowCapacity>
__simt_vf__ __aicore__ LAUNCH_BOUND(WARP_THREADS) inline void SwapSlab(
    __ubuf__ float* slab, __ubuf__ int* meta, uint32_t first, uint32_t panelWidth, uint32_t slabWidth)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    for (uint32_t c = threadIdx.x; c < slabWidth; c += WARP_THREADS) {
        for (uint32_t col = 0; col < panelWidth; ++col) {
            const uint32_t k = first + col, pivot = meta[col] - 1;
            if (meta[CgetrfDetail::PANEL_ZERO + col] || k == pivot)
                continue;
            const uint32_t a = 2 * (k + c * STRIDE), b = 2 * (pivot + c * STRIDE);
            const float real = slab[a], imag = slab[a + 1];
            slab[a] = slab[b];
            slab[a + 1] = slab[b + 1];
            slab[b] = real;
            slab[b + 1] = imag;
        }
    }
    asc_syncthreads();
}

template <uint32_t RowCapacity>
__aicore__ inline void Outside(
    LocalTensor<float> panel, LocalTensor<float> slab, LocalTensor<int> meta, GlobalTensor<float> global, uint32_t lda,
    uint32_t first, uint32_t width, uint32_t n)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    const uint32_t N = n;
    auto* pp = reinterpret_cast<__ubuf__ float*>(panel.GetPhyAddr());
    auto* sp = reinterpret_cast<__ubuf__ float*>(slab.GetPhyAddr());
    auto* mp = reinterpret_cast<__ubuf__ int*>(meta.GetPhyAddr());
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    const bool hasSwaps = meta.GetValue(SWAPPED) != 0;
    for (uint32_t col = 0; col < N;) {
        // An identity row permutation cannot change earlier columns. This is
        // determined from computed pivots for every input, never from its shape.
        if (col < first && !hasSwaps) {
            col = first;
            continue;
        }
        if (col == first) {
            col += width;
            continue;
        }
        const uint32_t end = col < first ? first : N;
        constexpr uint32_t SLAB = CgetrfFamilyPolicy::SlabColumns(RowCapacity);
        const uint32_t slabWidth = end - col < SLAB ? end - col : SLAB;
        CgetrfDetail::CopySlabIn(slab, global, N, lda, STRIDE, col, slabWidth);
        if (hasSwaps) {
            asc_vf_call<SwapSlab<RowCapacity>>(dim3{WARP_THREADS, 1, 1}, sp, mp, first, width, slabWidth);
        }
        if (col > first) {
            asc_vf_call<CgetrfDetail::UpdateSlabSimd>(pp, sp, mp, N, STRIDE, first, width, slabWidth);
        }
        CgetrfDetail::CopySlabOut(global, slab, N, lda, STRIDE, col, slabWidth);
        col += slabWidth;
    }
}

// Keep disjoint column ranges in one allocation; preserve the streamed pivot/update order.
template <uint32_t RowCapacity>
__aicore__ inline void ResidentOutside(
    __ubuf__ float* matrix, __ubuf__ int* mp, LocalTensor<int> meta, uint32_t first, uint32_t width, uint32_t n)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    auto* front = matrix + 2 * first * STRIDE;
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    const bool hasSwaps = meta.GetValue(SWAPPED) != 0;
    if (hasSwaps && first != 0) {
        asc_vf_call<SwapSlab<RowCapacity>>(dim3{WARP_THREADS, 1, 1}, matrix, mp, first, width, first);
    }
    const uint32_t next = first + width;
    if (next < n) {
        auto* trailing = matrix + 2 * next * STRIDE;
        if (hasSwaps) {
            asc_vf_call<SwapSlab<RowCapacity>>(dim3{WARP_THREADS, 1, 1}, trailing, mp, first, width, n - next);
        }
        asc_vf_call<CgetrfDetail::UpdateSlabSimd>(front, trailing, mp, n, STRIDE, first, width, n - next);
    }
}

template <uint32_t RowCapacity>
__aicore__ inline int FactorResident(
    LocalTensor<float> matrix, LocalTensor<int> meta, __gm__ int* pivots, uint32_t batch,
    const CgetrfBatchedTilingData& tiling)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    auto* data = reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr());
    auto* mp = reinterpret_cast<__ubuf__ int*>(meta.GetPhyAddr());
    int firstZero = 0;
    for (uint32_t first = 0; first < tiling.n; first += CgetrfFamilyPolicy::RESIDENT_PANEL_COLUMNS) {
        const uint32_t remaining = tiling.n - first;
        const uint32_t width = remaining < CgetrfFamilyPolicy::RESIDENT_PANEL_COLUMNS ?
                                   remaining :
                                   CgetrfFamilyPolicy::RESIDENT_PANEL_COLUMNS;
        auto* front = data + 2 * first * STRIDE;
        for (uint32_t col = 0; col < width; ++col) {
            asc_vf_call<SelectAndSwap<RowCapacity>>(
                dim3{WARP_THREADS, 1, 1}, front, mp, first, col, width, tiling.usePivot, tiling.n);
            asc_vf_call<ScaleAndUpdate<RowCapacity, CgetrfFamilyPolicy::RESIDENT_PANEL_COLUMNS>>(
                front, mp, first, col, width, tiling.n);
        }
        ResidentOutside<RowCapacity>(data, mp, meta, first, width, tiling.n);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        const int zero = meta.GetValue(CgetrfDetail::PANEL_INFO);
        if (firstZero == 0 && zero != 0)
            firstZero = zero;
        if (tiling.usePivot) {
            GlobalTensor<int> gp;
            gp.SetGlobalBuffer(pivots + static_cast<uint64_t>(batch) * tiling.n + first);
            DataCopyPad(gp, meta, DataCopyExtParams{1, width * INT_BYTES, 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
    return firstZero;
}

template <uint32_t RowCapacity>
__aicore__ inline void RunResident(
    GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos, uint32_t start, uint32_t count,
    const CgetrfBatchedTilingData& tiling)
{
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    static_assert(RowCapacity % CgetrfFamilyPolicy::VECTOR_ROWS == 0, "Full vector storage");
    if (tiling.n > CgetrfFamilyPolicy::RESIDENT_MAX_N || tiling.n > RowCapacity ||
        tiling.n <= RowCapacity - CgetrfFamilyPolicy::VECTOR_ROWS)
        return;
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf, metaBuf;
    pipe.InitBuffer(matrixBuf, tiling.n * STRIDE * COMPLEX_BYTES);
    pipe.InitBuffer(metaBuf, META_BYTES);
    auto matrix = matrixBuf.Get<float>();
    auto meta = metaBuf.Get<int>();
    for (uint32_t batch = start; batch < start + count; ++batch) {
        GlobalTensor<float> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(aarray)[batch]));
        CgetrfDetail::CopySlabIn(matrix, global, tiling.n, tiling.lda, STRIDE, 0, tiling.n);
        const int zero = FactorResident<RowCapacity>(matrix, meta, pivots, batch, tiling);
        CgetrfDetail::CopySlabOut(global, matrix, tiling.n, tiling.lda, STRIDE, 0, tiling.n);
        if (infos != nullptr) {
            meta.SetValue(INFO_OUT, zero);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            GlobalTensor<int> gi;
            gi.SetGlobalBuffer(infos + batch);
            DataCopyPad(gi, meta[INFO_OUT], DataCopyExtParams{1, INT_BYTES, 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}

__aicore__ inline void StoreInfo(LocalTensor<int> meta, __gm__ int* infos, uint32_t batch, int firstZero)
{
    if (infos != nullptr) {
        meta.SetValue(INFO_OUT, firstZero);
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        GlobalTensor<int> gi;
        gi.SetGlobalBuffer(infos + batch);
        DataCopyPad(gi, meta[INFO_OUT], DataCopyExtParams{1, INT_BYTES, 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
}

template <uint32_t RowCapacity>
__aicore__ inline void Run(
    GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos, uint32_t start, uint32_t count,
    const CgetrfBatchedTilingData& tiling)
{
    static_assert(
        RowCapacity % CgetrfFamilyPolicy::ROW_QUANTUM == 0 && RowCapacity <= CgetrfFamilyPolicy::MAX_ROWS,
        "Bounded row capacity");
    constexpr uint32_t PANEL = CgetrfFamilyPolicy::PanelColumns(RowCapacity);
    constexpr uint32_t STRIDE = RowCapacity + ROW_PADDING;
    constexpr uint32_t SLAB = CgetrfFamilyPolicy::SlabColumns(RowCapacity);
    static_assert(CgetrfFamilyPolicy::StorageBytes(RowCapacity) <= DYNAMIC_BYTES, "Storage fits");
    const uint32_t N = tiling.n;
    TPipe pipe;
    TBuf<TPosition::VECCALC> panelBuf, slabBuf, metaBuf;
    pipe.InitBuffer(panelBuf, PANEL * STRIDE * COMPLEX_BYTES);
    pipe.InitBuffer(slabBuf, SLAB * STRIDE * COMPLEX_BYTES);
    pipe.InitBuffer(metaBuf, META_BYTES);
    auto panel = panelBuf.Get<float>(), slab = slabBuf.Get<float>();
    auto meta = metaBuf.Get<int>();
    auto* pp = reinterpret_cast<__ubuf__ float*>(panel.GetPhyAddr());
    auto* mp = reinterpret_cast<__ubuf__ int*>(meta.GetPhyAddr());
    for (uint32_t b = start; b < start + count; ++b) {
        GlobalTensor<float> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(aarray)[b]));
        int firstZero = 0;
        for (uint32_t first = 0; first < N; first += PANEL) {
            const uint32_t width = N - first < PANEL ? N - first : PANEL;
            CgetrfDetail::CopySlabIn(panel, global, N, tiling.lda, STRIDE, first, width);
            for (uint32_t col = 0; col < width; ++col) {
                asc_vf_call<SelectAndSwap<RowCapacity>>(
                    dim3{WARP_THREADS, 1, 1}, pp, mp, first, col, width, tiling.usePivot, N);
                asc_vf_call<ScaleAndUpdate<RowCapacity, PANEL>>(pp, mp, first, col, width, N);
            }
            CgetrfDetail::CopySlabOut(global, panel, N, tiling.lda, STRIDE, first, width);
            Outside<RowCapacity>(panel, slab, meta, global, tiling.lda, first, width, N);
            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);
            const int zero = meta.GetValue(CgetrfDetail::PANEL_INFO);
            if (firstZero == 0 && zero != 0)
                firstZero = zero;
            if (tiling.usePivot) {
                GlobalTensor<int> gp;
                gp.SetGlobalBuffer(pivots + static_cast<uint64_t>(b) * N + first);
                DataCopyPad(gp, meta, DataCopyExtParams{1, width * INT_BYTES, 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        StoreInfo(meta, infos, b, firstZero);
    }
}
} // namespace CgetrfStreamFamily

// Grouped panel factorization helpers.
namespace CgetrfGroupedFront {
using namespace AscendC;
using namespace CgetrfConfig;
template <uint32_t PanelColumns>
__simd_callee__ inline void ScaleOne(
    __ubuf__ float* matrix, __ubuf__ int* metadata, uint32_t n, uint32_t stride, uint32_t first, uint32_t col,
    uint32_t width)
{
    static_assert(PanelColumns > 0, "Positive panel width");
    if (metadata[CgetrfDetail::PANEL_ZERO + col])
        return;
    const uint32_t k = first + col;
    Reg::RegTensor<float> ar, ai, lr, li, dr, di, den, tmp;
    Reg::RegTensor<int32_t> rows;
    uint32_t fullCount = VECTOR_FLOATS;
    Reg::MaskReg full = Reg::UpdateMask<float>(fullCount), active;
    // Diagonal values and their denominator are computed once, with no scalar UB read.
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(dr, matrix + 2 * (k + k * stride));
    Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(di, matrix + 2 * (k + k * stride) + 1);
    Reg::Mul(den, dr, dr, full);
    Reg::Mul(tmp, di, di, full);
    Reg::Add(den, den, tmp, full);
    for (uint16_t r = k / VECTOR_FLOATS * VECTOR_FLOATS; r < n; r += VECTOR_FLOATS) {
        uint32_t count = n - r;
        Reg::MaskReg valid = Reg::UpdateMask<float>(count);
        Reg::Arange(rows, static_cast<int32_t>(r));
        Reg::Compares<int32_t, CMPMODE::GT>(active, rows, static_cast<int32_t>(k), valid);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, matrix + 2 * (r + k * stride));
        CgetrfDetail::ScaleComplexColumn(lr, li, ar, ai, dr, di, den, active);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(matrix + 2 * (r + k * stride), lr, li, full);
        // Reuse these multipliers for every remaining panel column; no UB reload.
        // If panel width divides the reference interval, its update boundary
        // is always the last panel column: no in-panel values need beta.
        if constexpr (CgetrfDetail::REFERENCE_PANEL_COLUMNS % PanelColumns == 0) {
            CgetrfDetail::UpdatePanelRow<false>(matrix, r, k, stride, k + 1, first + width, lr, li, active, full);
        } else {
            if ((k + 1) % CgetrfDetail::REFERENCE_PANEL_COLUMNS == 0)
                CgetrfDetail::UpdatePanelRow<true>(matrix, r, k, stride, k + 1, first + width, lr, li, active, full);
            else
                CgetrfDetail::UpdatePanelRow<false>(matrix, r, k, stride, k + 1, first + width, lr, li, active, full);
        }
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <uint32_t Groups, uint32_t PanelColumns>
__simd_vf__ inline void ScalePairAndUpdate(
    __ubuf__ float* matrix, __ubuf__ int* metadata, uint32_t n, uint32_t stride, uint32_t first, uint32_t col,
    uint32_t width)
{
    static_assert(Groups == 1 || Groups == 2, "Only one or two resident matrices");
    ScaleOne<PanelColumns>(matrix, metadata, n, stride, first, col, width);
    if constexpr (Groups == 2) {
        ScaleOne<PanelColumns>(matrix + 2 * stride * n, metadata + META_WORDS, n, stride, first, col, width);
    }
}
} // namespace CgetrfGroupedFront

// Paired-matrix factorization.
namespace CgetrfPairFamily {
using namespace AscendC;
using namespace CgetrfConfig;
// Two complete matrices, selected solely by the declared storage budget.
constexpr uint32_t MAX_N = CgetrfFamilyPolicy::PAIR_MAX_N;
constexpr uint32_t STRIDE = CgetrfFamilyPolicy::PAIR_STRIDE;
constexpr uint32_t META_INTS = META_WORDS;
constexpr uint32_t COLUMNS = 4;
constexpr uint32_t OUTPUT_INTS = CgetrfFamilyPolicy::PAIR_OUTPUT_INTS;
static_assert(
    2 * (2 * MAX_N * STRIDE * INT_BYTES + META_INTS * INT_BYTES + OUTPUT_INTS * INT_BYTES) <=
        CgetrfFamilyPolicy::DYNAMIC_BYTES,
    "Paired matrices exceed dynamic UB reservation");

__simt_callee__ inline uint32_t WarpPivot(__ubuf__ float* matrix, uint32_t k, uint32_t lane, uint32_t n)
{
    uint32_t best = 0;
    int index = static_cast<int>(k);
    for (uint32_t r = k + lane; r < n; r += WARP_THREADS) {
        const uint32_t at = 2 * (r + k * STRIDE);
        const float real = matrix[at], imag = matrix[at + 1];
        const uint32_t value = CgetrfPivot::MagnitudeKey(real, imag);
        if (value > best) {
            best = value;
            index = static_cast<int>(r);
        }
    }
    const uint32_t largest = asc_reduce_max(best);
    uint32_t reduced = asc_reduce_min(best == largest ? index : NO_PIVOT_INDEX);
    if (largest > FP32_INFINITY_BITS)
        reduced = CgetrfPivot::OrderedNanPivot(matrix, STRIDE, k, k, n, lane);
    return reduced;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(PAIR_GROUPS* WARP_THREADS) inline void SelectPair(
    __ubuf__ float* data, __ubuf__ int* meta, __ubuf__ int* result, uint32_t first, uint32_t col, uint32_t usePivot,
    uint32_t n)
{
    const uint32_t group = threadIdx.x / WARP_THREADS, lane = threadIdx.x % WARP_THREADS, k = first + col;
    const uint32_t matrixFloats = 2 * n * STRIDE;
    auto* matrix = data + group * matrixFloats;
    auto* metadata = meta + group * META_INTS;
    auto* output = result + group * OUTPUT_INTS;
    const uint32_t pivot = usePivot ? WarpPivot(matrix, k, lane, n) : k;
    const uint32_t diag = 2 * (pivot + k * STRIDE);
    const bool zero = CgetrfPivot::IsZero(matrix[diag], matrix[diag + 1]);
    if (lane == 0) {
        metadata[col] = static_cast<int>(pivot) + 1;
        metadata[CgetrfDetail::PANEL_ZERO + col] = zero;
        output[k] = static_cast<int>(pivot) + 1;
        if (k == 0)
            output[PAIR_INFO_SLOT] = 0;
        if (zero && output[PAIR_INFO_SLOT] == 0)
            output[PAIR_INFO_SLOT] = static_cast<int>(k) + 1;
    }
    if (!zero && pivot != k) {
        for (uint32_t c = lane; c < n; c += WARP_THREADS) {
            const uint32_t a = 2 * (k + c * STRIDE), b = 2 * (pivot + c * STRIDE);
            const float real = matrix[a], imag = matrix[a + 1];
            matrix[a] = matrix[b];
            matrix[a + 1] = matrix[b + 1];
            matrix[b] = real;
            matrix[b + 1] = imag;
        }
    }
    asc_syncthreads();
}

__aicore__ inline void FactorPair(
    LocalTensor<float> matrix, LocalTensor<int> metadata, LocalTensor<int> output, uint32_t usePivot, uint32_t n)
{
    auto* data = reinterpret_cast<__ubuf__ float*>(matrix.GetPhyAddr());
    auto* meta = reinterpret_cast<__ubuf__ int*>(metadata.GetPhyAddr());
    auto* result = reinterpret_cast<__ubuf__ int*>(output.GetPhyAddr());
    constexpr uint32_t columns = COLUMNS;
    const uint32_t matrixFloats = 2 * n * STRIDE;
    for (uint32_t first = 0; first < n; first += columns) {
        const uint32_t width = n - first < columns ? n - first : columns;
        for (uint32_t col = 0; col < width; ++col) {
            asc_vf_call<SelectPair>(
                dim3{PAIR_GROUPS * WARP_THREADS, 1, 1}, data, meta, result, first, col, usePivot, n);
            asc_vf_call<CgetrfGroupedFront::ScalePairAndUpdate<2, COLUMNS>>(data, meta, n, STRIDE, first, col, width);
        }
        if (first + width < n) {
            for (uint32_t group = 0; group < 2; ++group) {
                auto* base = data + group * matrixFloats;
                asc_vf_call<CgetrfDetail::UpdateSlabSimd>(
                    base + 2 * first * STRIDE, base + 2 * (first + width) * STRIDE, meta + group * META_INTS, n, STRIDE,
                    first, width, n - first - width);
            }
        }
    }
}

__aicore__ inline void Run(
    GM_ADDR aarray, __gm__ int* pivots, __gm__ int* infos, uint32_t startBatch, uint32_t numBatch,
    const CgetrfBatchedTilingData& tiling)
{
    const uint32_t n = tiling.n;
    const uint32_t matrixFloats = 2 * n * STRIDE;
    TPipe pipe;
    TBuf<TPosition::VECCALC> matrixBuf, metadataBuf, outputBuf;
    pipe.InitBuffer(matrixBuf, 2 * matrixFloats * sizeof(float));
    pipe.InitBuffer(metadataBuf, 2 * META_INTS * sizeof(int));
    pipe.InitBuffer(outputBuf, 2 * OUTPUT_INTS * sizeof(int));
    auto matrix = matrixBuf.Get<float>();
    auto metadata = metadataBuf.Get<int>();
    auto output = outputBuf.Get<int>();
    for (uint32_t b = 0; b < numBatch; b += 2) {
        const uint32_t count = numBatch - b < 2 ? 1 : 2;
        for (uint32_t group = 0; group < 2; ++group) {
            const uint32_t sourceGroup = group < count ? group : count - 1;
            GlobalTensor<float> global;
            global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(
                reinterpret_cast<__gm__ uint64_t*>(aarray)[startBatch + b + sourceGroup]));
            CgetrfDetail::CopySlabIn(matrix[group * matrixFloats], global, n, tiling.lda, STRIDE, 0, n);
        }
        FactorPair(matrix, metadata, output, tiling.usePivot, n);
        for (uint32_t group = 0; group < count; ++group) {
            const uint32_t batch = startBatch + b + group;
            GlobalTensor<float> global;
            global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(aarray)[batch]));
            CgetrfDetail::CopySlabOut(global, matrix[group * matrixFloats], n, tiling.lda, STRIDE, 0, n);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            if (tiling.usePivot) {
                GlobalTensor<int> gp;
                gp.SetGlobalBuffer(pivots + static_cast<uint64_t>(batch) * n);
                DataCopyPad(gp, output[group * OUTPUT_INTS], DataCopyExtParams{1, n * INT_BYTES, 0, 0, 0});
            }
            if (infos != nullptr) {
                GlobalTensor<int> gi;
                gi.SetGlobalBuffer(infos + batch);
                DataCopyPad(gi, output[group * OUTPUT_INTS + PAIR_INFO_SLOT], DataCopyExtParams{1, INT_BYTES, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}
} // namespace CgetrfPairFamily

using namespace AscendC;
using namespace CgetrfConfig;

constexpr uint32_t CGETRF_SIMT_MAX_THREADS = SIMT_THREADS;
static_assert(
    (CGETRF_SIMT_MAX_THREADS & (CGETRF_SIMT_MAX_THREADS - 1)) == 0,
    "CGETRF_SIMT_MAX_THREADS must be a power of 2 for tree reduction correctness");

// Complex matrices are stored as interleaved float pairs: real, imag.
// Reading the storage through float pointers avoids address-space copies of C structs.
// The pointer array is stored as raw bytes (GM_ADDR) to avoid __gm__ pointer-to-pointer issues.
__simt_callee__ __aicore__ inline __gm__ float* ReadComplexStoragePtr(GM_ADDR aarrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(aarrayBase) + batchIdx;
    uint64_t rawAddr = *addrSlot;
    return reinterpret_cast<__gm__ float*>(rawAddr);
}

__simt_callee__ __aicore__ inline bool IsComplexZero(float real, float imag) { return CgetrfPivot::IsZero(real, imag); }

template <typename MatrixPtr>
__simt_callee__ __aicore__ inline void Rank1Update(MatrixPtr A, uint32_t lda, uint32_t col, uint32_t updateSize)
{
    if (updateSize > 0) {
        for (uint32_t localR = threadIdx.x; localR < updateSize; localR += blockDim.x) {
            uint32_t row = col + 1 + localR;
            uint64_t multiplierIdx = COMPLEX_COMPONENTS * (row + static_cast<uint64_t>(col) * lda);
            float multiplierReal = A[multiplierIdx];
            float multiplierImag = A[multiplierIdx + 1];
            for (uint32_t localC = 0; localC < updateSize; localC++) {
                uint32_t cc = col + 1 + localC;
                uint64_t upperIdx = COMPLEX_COMPONENTS * (col + static_cast<uint64_t>(cc) * lda);
                uint64_t valueIdx = COMPLEX_COMPONENTS * (row + static_cast<uint64_t>(cc) * lda);
                float upperReal = A[upperIdx];
                float upperImag = A[upperIdx + 1];
                float productReal = multiplierReal * upperReal - multiplierImag * upperImag;
                float productImag = multiplierReal * upperImag + multiplierImag * upperReal;
                // Keep the reference blocked update's complex beta=(1,0).
                if ((col + 1) % CgetrfPivot::REFERENCE_PANEL_COLUMNS == 0) {
                    float oldReal = A[valueIdx], oldImag = A[valueIdx + 1];
                    A[valueIdx] = oldReal - 0.0f * oldImag;
                    A[valueIdx + 1] = oldImag + 0.0f * oldReal;
                }
                A[valueIdx] -= productReal;
                A[valueIdx + 1] -= productImag;
            }
        }
    }
}

template <typename MatrixPtr>
__simt_callee__ __aicore__ inline int FindPivotRow(
    MatrixPtr A, uint32_t lda, uint32_t col, uint32_t n, __ubuf__ float* partialVals, __ubuf__ int* partialIdx)
{
    uint32_t localMax = 0;
    auto* keys = reinterpret_cast<__ubuf__ uint32_t*>(partialVals);
    int localRow = static_cast<int>(col);
    for (uint32_t row = col + threadIdx.x; row < n; row += blockDim.x) {
        uint64_t valueIdx = COMPLEX_COMPONENTS * (row + static_cast<uint64_t>(col) * lda);
        uint32_t absVal = CgetrfPivot::MagnitudeKey(A[valueIdx], A[valueIdx + 1]);
        if (absVal > localMax) {
            localMax = absVal;
            localRow = static_cast<int>(row);
        }
    }
    keys[threadIdx.x] = localMax;
    partialIdx[threadIdx.x] = localRow;
    asc_syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (keys[threadIdx.x + stride] > keys[threadIdx.x] ||
                (keys[threadIdx.x + stride] == keys[threadIdx.x] &&
                 partialIdx[threadIdx.x + stride] < partialIdx[threadIdx.x])) {
                keys[threadIdx.x] = keys[threadIdx.x + stride];
                partialIdx[threadIdx.x] = partialIdx[threadIdx.x + stride];
            }
        }
        asc_syncthreads();
    }
    if (keys[0] > FP32_INFINITY_BITS) {
        if (threadIdx.x < WARP_THREADS) {
            const uint32_t row = CgetrfPivot::OrderedNanPivot(A, lda, col, col, n, threadIdx.x);
            if (threadIdx.x == 0)
                partialIdx[0] = static_cast<int>(row);
        }
        asc_syncthreads();
    }
    return partialIdx[0];
}

template <typename MatrixPtr>
__simt_callee__ __aicore__ inline void SwapRows(MatrixPtr A, uint32_t lda, uint32_t row1, uint32_t row2, uint32_t n)
{
    if (row1 != row2) {
        for (uint32_t c = threadIdx.x; c < n; c += blockDim.x) {
            uint64_t firstIdx = COMPLEX_COMPONENTS * (row1 + static_cast<uint64_t>(c) * lda);
            uint64_t secondIdx = COMPLEX_COMPONENTS * (row2 + static_cast<uint64_t>(c) * lda);
            float tmpReal = A[firstIdx];
            float tmpImag = A[firstIdx + 1];
            A[firstIdx] = A[secondIdx];
            A[firstIdx + 1] = A[secondIdx + 1];
            A[secondIdx] = tmpReal;
            A[secondIdx + 1] = tmpImag;
        }
    }
    asc_syncthreads();
}

template <typename MatrixPtr>
__simt_callee__ __aicore__ inline void ComputeMultipliers(MatrixPtr A, uint32_t lda, uint32_t col, uint32_t n)
{
    uint64_t diagonalIdx = COMPLEX_COMPONENTS * (col + static_cast<uint64_t>(col) * lda);
    float diagonalReal = A[diagonalIdx];
    float diagonalImag = A[diagonalIdx + 1];
    if (IsComplexZero(diagonalReal, diagonalImag)) {
        asc_syncthreads();
        return;
    }
    float denominator = diagonalReal * diagonalReal + diagonalImag * diagonalImag;
    for (uint32_t row = col + 1 + threadIdx.x; row < n; row += blockDim.x) {
        uint64_t valueIdx = COMPLEX_COMPONENTS * (row + static_cast<uint64_t>(col) * lda);
        float numeratorReal = A[valueIdx];
        float numeratorImag = A[valueIdx + 1];
        A[valueIdx] = (numeratorReal * diagonalReal + numeratorImag * diagonalImag) / denominator;
        A[valueIdx + 1] = (numeratorImag * diagonalReal - numeratorReal * diagonalImag) / denominator;
    }
    asc_syncthreads();
}

template <typename MatrixPtr>
__simt_callee__ __aicore__ inline void ApplyColumnElimination(MatrixPtr A, uint32_t lda, uint32_t col, uint32_t n)
{
    ComputeMultipliers(A, lda, col, n);
    Rank1Update(A, lda, col, n - col - 1);
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void StoreBatchInfo(
    __gm__ int* infoArray, uint32_t batchIdx, __ubuf__ int* sharedInfoBuf)
{
    if (threadIdx.x == 0 && infoArray != nullptr) {
        infoArray[batchIdx] = sharedInfoBuf[0];
    }
    asc_syncthreads();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRF_SIMT_MAX_THREADS) inline void CgetrfSimtPivot(
    uint32_t n, uint32_t lda, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase, __gm__ int* pivotArray,
    __gm__ int* infoArray)
{
    if (pivotArray == nullptr) {
        return;
    }

    __ubuf__ float partialVals[CGETRF_SIMT_MAX_THREADS];
    __ubuf__ int partialIdx[CGETRF_SIMT_MAX_THREADS];
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadComplexStoragePtr(aarrayBase, batchIdx);
        __gm__ int* piv = pivotArray + static_cast<uint64_t>(batchIdx) * n;

        if (threadIdx.x == 0) {
            sharedInfoBuf[0] = 0;
        }
        asc_syncthreads();

        for (uint32_t col = 0; col < n; col++) {
            int pivotRow = FindPivotRow(A, lda, col, n, partialVals, partialIdx);
            uint64_t pivotIdx =
                COMPLEX_COMPONENTS * (static_cast<uint32_t>(pivotRow) + static_cast<uint64_t>(col) * lda);
            float pivotReal = A[pivotIdx];
            float pivotImag = A[pivotIdx + 1];

            if (threadIdx.x == 0) {
                piv[col] = pivotRow + 1;
                if (IsComplexZero(pivotReal, pivotImag) && sharedInfoBuf[0] == 0) {
                    sharedInfoBuf[0] = static_cast<int>(col) + 1;
                }
            }
            asc_syncthreads();
            if (IsComplexZero(pivotReal, pivotImag)) {
                continue;
            }

            SwapRows(A, lda, col, static_cast<uint32_t>(pivotRow), n);
            ApplyColumnElimination(A, lda, col, n);
        }

        StoreBatchInfo(infoArray, batchIdx, sharedInfoBuf);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRF_SIMT_MAX_THREADS) inline void CgetrfSimtNoPivot(
    uint32_t n, uint32_t lda, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase, __gm__ int* infoArray)
{
    __ubuf__ int sharedInfoBuf[1];

    for (uint32_t b = 0; b < numBatch; b++) {
        uint32_t batchIdx = startBatch + b;
        __gm__ float* A = ReadComplexStoragePtr(aarrayBase, batchIdx);

        if (threadIdx.x == 0) {
            sharedInfoBuf[0] = 0;
        }
        asc_syncthreads();

        for (uint32_t col = 0; col < n; col++) {
            uint64_t diagonalIdx = COMPLEX_COMPONENTS * (col + static_cast<uint64_t>(col) * lda);
            float diagonalReal = A[diagonalIdx];
            float diagonalImag = A[diagonalIdx + 1];

            // Check singularity
            if (threadIdx.x == 0) {
                if (IsComplexZero(diagonalReal, diagonalImag) && sharedInfoBuf[0] == 0) {
                    sharedInfoBuf[0] = static_cast<int>(col) + 1;
                }
            }
            asc_syncthreads();
            if (IsComplexZero(diagonalReal, diagonalImag)) {
                continue;
            }

            ApplyColumnElimination(A, lda, col, n);
        }

        StoreBatchInfo(infoArray, batchIdx, sharedInfoBuf);
    }
}

extern "C" __global__ __aicore__ void cgetrf_packed_small_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const CgetrfBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t block = GetBlockIdx(), start = block * tiling.batchPerCore;
    const uint32_t count = block == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    if (count == 0)
        return;
    if (tiling.n <= PACKED_MIN_ROWS) {
        CgetrfPackedFamily::Run<PACKED_MIN_ROWS>(
            aarray, reinterpret_cast<__gm__ int*>(pivotArray), reinterpret_cast<__gm__ int*>(infoArray), start, count,
            tiling);
    } else {
        CgetrfPackedFamily::Run<PACKED_MAX_ROWS>(
            aarray, reinterpret_cast<__gm__ int*>(pivotArray), reinterpret_cast<__gm__ int*>(infoArray), start, count,
            tiling);
    }
}

aclError cgetrf_packed_small_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, const CgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgetrf_packed_small_kernel<<<numBlocks, PACKED_DYNAMIC_BYTES, stream>>>(aarray, pivotArray, infoArray, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" __global__ __aicore__ void cgetrf_small_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const CgetrfBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t block = GetBlockIdx();
    const uint32_t start = block * tiling.batchPerCore;
    const uint32_t count = block == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    if (count != 0)
        CgetrfDetail::RunSmallSimd(
            aarray, reinterpret_cast<__gm__ int*>(pivotArray), reinterpret_cast<__gm__ int*>(infoArray), start, count,
            tiling);
}

aclError cgetrf_small_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, const CgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgetrf_small_kernel<<<numBlocks, SMALL_DYNAMIC_BYTES, stream>>>(aarray, pivotArray, infoArray, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" __global__ __aicore__ void cgetrf_stream_range_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const CgetrfBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t block = GetBlockIdx(), start = block * tiling.batchPerCore;
    const uint32_t count = block == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    if (count == 0 || tiling.n <= CgetrfFamilyPolicy::VECTOR_ROWS || tiling.n > CgetrfFamilyPolicy::MAX_ROWS)
        return;
    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    if (tiling.n <= CgetrfFamilyPolicy::PAIR_MAX_N) {
        CgetrfPairFamily::Run(aarray, pivots, infos, start, count, tiling);
        return;
    }
    if (tiling.n <= CgetrfFamilyPolicy::RESIDENT_MAX_N) {
        if (tiling.n <= 2 * CgetrfFamilyPolicy::VECTOR_ROWS) {
            CgetrfStreamFamily::RunResident<2 * CgetrfFamilyPolicy::VECTOR_ROWS>(
                aarray, pivots, infos, start, count, tiling);
        } else {
            CgetrfStreamFamily::RunResident<3 * CgetrfFamilyPolicy::VECTOR_ROWS>(
                aarray, pivots, infos, start, count, tiling);
        }
        return;
    }
    const uint32_t capacity = (tiling.n + CgetrfFamilyPolicy::ROW_QUANTUM - 1) / CgetrfFamilyPolicy::ROW_QUANTUM *
                              CgetrfFamilyPolicy::ROW_QUANTUM;
    static_assert(
        CgetrfFamilyPolicy::RESIDENT_MAX_N >= CgetrfFamilyPolicy::ROW_QUANTUM,
        "The first row-capacity class is handled by resident storage");
    if (capacity <= 2 * CgetrfFamilyPolicy::ROW_QUANTUM) {
        CgetrfStreamFamily::Run<2 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    } else if (capacity <= 3 * CgetrfFamilyPolicy::ROW_QUANTUM) {
        CgetrfStreamFamily::Run<3 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    } else if (capacity <= 4 * CgetrfFamilyPolicy::ROW_QUANTUM) {
        CgetrfStreamFamily::Run<4 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    } else if (capacity <= 8 * CgetrfFamilyPolicy::ROW_QUANTUM) {
        CgetrfStreamFamily::Run<8 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    } else if (capacity <= 16 * CgetrfFamilyPolicy::ROW_QUANTUM) {
        CgetrfStreamFamily::Run<16 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    } else {
        CgetrfStreamFamily::Run<32 * CgetrfFamilyPolicy::ROW_QUANTUM>(aarray, pivots, infos, start, count, tiling);
    }
}

aclError cgetrf_stream_range_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, const CgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgetrf_stream_range_kernel<<<numBlocks, CgetrfFamilyPolicy::DYNAMIC_BYTES, stream>>>(
        aarray, pivotArray, infoArray, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" __global__ __aicore__ void cgetrf_batched_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const CgetrfBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockId = GetBlockIdx();
    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = (blockId == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);

    if (tiling.usePivot) {
        asc_vf_call<CgetrfSimtPivot>(
            dim3{CGETRF_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, numBatch, startBatch, aarray, pivots, infos);
    } else {
        asc_vf_call<CgetrfSimtNoPivot>(
            dim3{CGETRF_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, numBatch, startBatch, aarray, infos);
    }
}

aclError cgetrf_batched_kernel_do(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, const CgetrfBatchedTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // DMA byte strides are 32-bit; large padding uses the 64-bit SIMT path.
    const bool copyStrideFits = (static_cast<uint64_t>(tiling.lda) - tiling.n) * COMPLEX_BYTES <= UINT32_MAX;
    // Authoritative family policy: no fixed-workload route is consulted.
    if (copyStrideFits && tiling.n <= PACKED_MAX_ROWS) {
        return cgetrf_packed_small_kernel_do(aarray, pivotArray, infoArray, tiling, numBlocks, stream);
    }
    if (copyStrideFits && tiling.n <= CgetrfFamilyPolicy::VECTOR_ROWS) {
        return cgetrf_small_kernel_do(aarray, pivotArray, infoArray, tiling, numBlocks, stream);
    }
    if (copyStrideFits && CgetrfFamilyPolicy::StreamSupported(tiling.n)) {
        return cgetrf_stream_range_kernel_do(aarray, pivotArray, infoArray, tiling, numBlocks, stream);
    }
    // Reserve room for it instead of requesting the full 224 KiB as dynamic UB.
    const uint32_t dynamicUbBytes = FALLBACK_DYNAMIC_BYTES;
    cgetrf_batched_kernel<<<numBlocks, dynamicUbBytes, stream>>>(aarray, pivotArray, infoArray, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
