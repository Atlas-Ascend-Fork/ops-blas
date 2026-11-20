/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/helper/devkit_version_compat.h"
#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"
#include "csyrkx_kernel.h"

// Precision compatibility for N and nonconjugating T/C. The T/C recurrence
// below is complemented by the separately implemented N beta-first path.
// Unlike split-real GEMM, each T/C output follows Netlib's complex FP32 dot order:
//   real = fl(real + fl(fl(ar*br) - fl(ai*bi)))
//   imag = fl(imag + fl(fl(ar*bi) + fl(ai*br)))
// Vector lanes represent independent output columns, NEVER reduction indices.
namespace csyrkx_ordered {
using namespace AscendC;

// Four rows compile with a 256-byte VF stack limit: no vector-register spills.
// Eight rows fit the default stack, but spill 18 vector slots in the hot T VF.
constexpr uint32_t ROWS = 4;
constexpr uint32_t COLS = 64;
constexpr uint32_t K_CHUNK = 128;
constexpr uint32_t BLOCK_FLOATS = 8;
// Four extra complex values rotate consecutive columns through UB banks.
constexpr uint32_t INPUT_STRIDE = 2 * (K_CHUNK + 4);
constexpr uint32_t OUTPUT_STRIDE = 2 * (ROWS + 4);
constexpr uint32_t A_OFFSET = 0;
constexpr uint32_t B_OFFSET = A_OFFSET + ROWS * INPUT_STRIDE;
constexpr uint32_t SUM_OFFSET = B_OFFSET + COLS * INPUT_STRIDE;
constexpr uint32_t C_OFFSET = SUM_OFFSET + 2 * ROWS * COLS;
constexpr uint32_t OUT_OFFSET = C_OFFSET + COLS * OUTPUT_STRIDE;
constexpr uint32_t UB_FLOATS = OUT_OFFSET + COLS * OUTPUT_STRIDE;
static_assert(UB_FLOATS * sizeof(float) == 82048, "Update the documented UB budget");
static_assert(UB_FLOATS * sizeof(float) <= 192 * 1024, "Ordered kernel exceeds UB budget");
static_assert(
    INPUT_STRIDE % BLOCK_FLOATS == 0 && OUTPUT_STRIDE % BLOCK_FLOATS == 0, "DMA column starts must be 32-byte aligned");

// Explicit names avoid unsupported arrays or dynamic indexing of RegTensor.
#define CSYRKX_ORDERED_ROWS(M) M(0) M(1) M(2) M(3)

// Register/mask types are deduced so the host/archless pass never needs
// MicroAPI types in a function signature. Namespace aliases stay in VF bodies.
template <uint32_t Row, typename FloatRegister, typename MaskRegister>
__simd_callee__ __attribute__((always_inline)) inline void AccumulateRow(
    __ubuf__ float* a, uint32_t l, FloatRegister& br, FloatRegister& bi, FloatRegister& real, FloatRegister& imag,
    MaskRegister& mask)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    V::RegTensor<float> ar, ai, rr, ii, ri, ir;
    V::LoadAlign<float, V::LoadDist::DIST_BRC_B32>(ar, a + Row * INPUT_STRIDE + 2 * l);
    V::LoadAlign<float, V::LoadDist::DIST_BRC_B32>(ai, a + Row * INPUT_STRIDE + 2 * l + 1);
    // Do not replace these instructions with MulAddDst, MADD, or four separate
    // dot products: their rounding points are part of this kernel's contract.
    V::Mul(rr, ar, br, mask);
    V::Mul(ii, ai, bi, mask);
    V::Mul(ri, ar, bi, mask);
    V::Mul(ir, ai, br, mask);
    V::Sub(rr, rr, ii, mask);
    V::Add(ri, ri, ir, mask);
    V::Add(real, real, rr, mask);
    V::Add(imag, imag, ri, mask);
}

template <bool FullRows>
__simd_vf__ inline void AccumulateChunk(
    __ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* sums, uint32_t rows, uint32_t cols, uint32_t count,
    bool first)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    uint32_t active = cols;
    V::MaskReg mask = V::UpdateMask<float>(active);
    V::RegTensor<uint32_t> indices;
    V::Arange(reinterpret_cast<V::RegTensor<int32_t>&>(indices), 0);
    V::Muls(indices, indices, INPUT_STRIDE, mask);
    V::RegTensor<float> br, bi;
#define DECLARE_ROW(R) V::RegTensor<float> real##R, imag##R;
    CSYRKX_ORDERED_ROWS(DECLARE_ROW)
#undef DECLARE_ROW
#define LOAD_ROW(R)                                           \
    if (FullRows || rows > R) {                               \
        if (first) {                                          \
            V::Duplicate(real##R, 0.0f);                      \
            V::Duplicate(imag##R, 0.0f);                      \
        } else {                                              \
            V::LoadAlign(real##R, sums + (2 * R) * COLS);     \
            V::LoadAlign(imag##R, sums + (2 * R + 1) * COLS); \
        }                                                     \
    }
    CSYRKX_ORDERED_ROWS(LOAD_ROW)
#undef LOAD_ROW
    // Ascending K, one dependent accumulator pair per output, across ALL chunks.
    // A zero count initializes sums for scaleOnly without touching A or B.
#pragma clang loop unroll(disable)
    for (uint32_t l = 0; l < count; ++l) {
        V::Gather(br, b + 2 * l, indices, mask);
        V::Gather(bi, b + 2 * l + 1, indices, mask);
#define UPDATE_ROW(R)                                           \
    if (FullRows || rows > R) {                                 \
        AccumulateRow<R>(a, l, br, bi, real##R, imag##R, mask); \
    }
        CSYRKX_ORDERED_ROWS(UPDATE_ROW)
#undef UPDATE_ROW
    }
#define STORE_ROW(R)                                             \
    if (FullRows || rows > R) {                                  \
        V::StoreAlign(sums + (2 * R) * COLS, real##R, mask);     \
        V::StoreAlign(sums + (2 * R + 1) * COLS, imag##R, mask); \
    }
    CSYRKX_ORDERED_ROWS(STORE_ROW)
#undef STORE_ROW
    // Storing/reloading FP32 between chunks introduces no new rounding.
    V::LocalMemBar<V::MemType::VEC_STORE, V::MemType::VEC_LOAD>();
}

template <uint32_t Row, bool BetaZero, typename IndexRegister, typename MaskRegister>
__simd_callee__ __attribute__((always_inline)) inline void FinishRow(
    __ubuf__ float* sums, __ubuf__ float* oldC, __ubuf__ float* out, IndexRegister& columns, IndexRegister& cIndices,
    IndexRegister& outIndices, MaskRegister& columnsMask, uint32_t row, bool upper, bool scaleOnly, float ar, float ai,
    float br, float bi)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    V::MaskReg selected;
    V::RegTensor<uint32_t> thisRow;
    V::Duplicate(thisRow, row + Row);
    // Form the mask even for a whole tile: no dependence on MaskReg assignment.
    if (upper) {
        V::Compare<uint32_t, CMPMODE::LE>(selected, thisRow, columns, columnsMask);
    } else {
        V::Compare<uint32_t, CMPMODE::GE>(selected, thisRow, columns, columnsMask);
    }
    V::RegTensor<float> real, imag, rr, ii, ri, ir;
    V::LoadAlign(real, sums + (2 * Row) * COLS);
    V::LoadAlign(imag, sums + (2 * Row + 1) * COLS);
    if (!scaleOnly) {
        V::Muls(rr, real, ar, selected);
        V::Muls(ii, imag, ai, selected);
        V::Muls(ri, imag, ar, selected);
        V::Muls(ir, real, ai, selected);
        V::Sub(real, rr, ii, selected);
        V::Add(imag, ri, ir, selected);
    }
    if constexpr (!BetaZero) {
        V::RegTensor<float> cr, ci;
        V::Gather(cr, oldC + 2 * Row, cIndices, selected);
        V::Gather(ci, oldC + 2 * Row + 1, cIndices, selected);
        V::Muls(rr, cr, br, selected);
        V::Muls(ii, ci, bi, selected);
        V::Muls(ri, ci, br, selected);
        V::Muls(ir, cr, bi, selected);
        V::Sub(rr, rr, ii, selected);
        V::Add(ri, ri, ir, selected);
        V::Add(real, real, rr, selected);
        V::Add(imag, imag, ri, selected);
    }
    // For crossing lower-triangle tiles, outIndices removes each column's
    // skipped prefix. Every MTE3 source then starts at an aligned UB column.
    V::Scatter(out + 2 * Row, real, outIndices, selected);
    V::Scatter(out + 2 * Row + 1, imag, outIndices, selected);
}

template <bool BetaZero>
__simd_vf__ inline void FinishTile(
    __ubuf__ float* sums, __ubuf__ float* oldC, __ubuf__ float* out, uint32_t row, uint32_t col, uint32_t rows,
    uint32_t cols, bool upper, bool whole, bool scaleOnly, float ar, float ai, float br, float bi)
{
    namespace V = AscendC::MicroAPI;
    uint32_t active = cols;
    V::MaskReg mask = V::UpdateMask<float>(active);
    V::RegTensor<uint32_t> lanes, columns, cIndices, outIndices;
    V::Arange(reinterpret_cast<V::RegTensor<int32_t>&>(lanes), 0);
    V::Adds(columns, lanes, col, mask);
    V::Muls(cIndices, lanes, OUTPUT_STRIDE, mask);
    // A zero scalar add is an explicit register copy supported by MicroAPI.
    V::Adds(outIndices, cIndices, 0u, mask);
    if (!upper && !whole) {
        V::RegTensor<uint32_t> first, base;
        V::Duplicate(base, row);
        V::Max(first, columns, base, mask);
        V::Sub(first, first, base, mask);
        V::Muls(first, first, 2u, mask);
        V::Sub(outIndices, outIndices, first, mask);
    }
#define FINISH_ROW(R)                                                                                     \
    if (rows > R) {                                                                                       \
        FinishRow<R, BetaZero>(                                                                           \
            sums, oldC, out, columns, cIndices, outIndices, mask, row, upper, scaleOnly, ar, ai, br, bi); \
    }
    CSYRKX_ORDERED_ROWS(FINISH_ROW)
#undef FINISH_ROW
    V::LocalMemBar<V::MemType::VEC_STORE, V::MemType::VEC_LOAD>();
}
// N transport is K-major and fits the existing T buffer allocations.
constexpr uint32_t N_A_STRIDE = 2 * ROWS;
constexpr uint32_t N_B_STRIDE = 2 * COLS;
static_assert(K_CHUNK * N_A_STRIDE <= ROWS * INPUT_STRIDE, "N A staging exceeds allocation");
static_assert(K_CHUNK * N_B_STRIDE <= COLS * INPUT_STRIDE, "N B staging exceeds allocation");

template <bool BetaZero>
__simd_vf__ inline void InitializeN(
    __ubuf__ float* sums, __ubuf__ float* oldC, uint32_t rows, uint32_t cols, float betaReal, float betaImag)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    uint32_t active = cols;
    auto mask = V::UpdateMask<float>(active);
    V::RegTensor<uint32_t> indices;
    V::Arange(reinterpret_cast<V::RegTensor<int32_t>&>(indices), 0);
    V::Muls(indices, indices, OUTPUT_STRIDE, mask);
    for (uint32_t r = 0; r < rows; ++r) {
        V::RegTensor<float> real, imag, rr, ii, ri, ir;
        if constexpr (BetaZero) {
            V::Duplicate(real, 0.0f);
            V::Duplicate(imag, 0.0f);
        } else {
            V::Gather(real, oldC + 2 * r, indices, mask);
            V::Gather(imag, oldC + 2 * r + 1, indices, mask);
            // Netlib beta==1 leaves C untouched, even for nonfinite C.
            if (betaReal != 1.0f || betaImag != 0.0f) {
                V::Muls(rr, real, betaReal, mask);
                V::Muls(ii, imag, betaImag, mask);
                V::Muls(ri, imag, betaReal, mask);
                V::Muls(ir, real, betaImag, mask);
                V::Sub(real, rr, ii, mask);
                V::Add(imag, ri, ir, mask);
            }
        }
        V::StoreAlign(sums + (2 * r) * COLS, real, mask);
        V::StoreAlign(sums + (2 * r + 1) * COLS, imag, mask);
    }
    V::LocalMemBar<V::MemType::VEC_STORE, V::MemType::VEC_LOAD>();
}

// N's arithmetic is independent of the already-validated T implementation.
template <uint32_t Row, typename FloatRegister, typename MaskRegister>
__simd_callee__ __attribute__((always_inline)) inline void AccumulateRowN(
    __ubuf__ float* a, uint32_t l, FloatRegister& tr, FloatRegister& ti, FloatRegister& real, FloatRegister& imag,
    MaskRegister& nonzero)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    V::RegTensor<float> ar, ai, p0, p1;
    V::LoadAlign<float, V::LoadDist::DIST_BRC_B32>(ar, a + l * N_A_STRIDE + 2 * Row);
    V::LoadAlign<float, V::LoadDist::DIST_BRC_B32>(ai, a + l * N_A_STRIDE + 2 * Row + 1);
    V::Mul(p0, tr, ar, nonzero);
    V::Mul(p1, ti, ai, nonzero);
    V::Sub(p0, p0, p1, nonzero);
    V::Add(p0, real, p0, nonzero);
    // Select preserves skipped B==0 lanes even if masked arithmetic zeros them.
    V::Select(real, p0, real, nonzero);
    V::Mul(p0, tr, ai, nonzero);
    V::Mul(p1, ti, ar, nonzero);
    V::Add(p0, p0, p1, nonzero);
    V::Add(p0, imag, p0, nonzero);
    V::Select(imag, p0, imag, nonzero);
}

template <bool FullRows>
__simd_vf__ inline void AccumulateChunkN(
    __ubuf__ float* a, __ubuf__ float* b, __ubuf__ float* sums, uint32_t rows, uint32_t cols, uint32_t count,
    float alphaReal, float alphaImag)
{
#pragma clang fp contract(off)
    namespace V = AscendC::MicroAPI;
    uint32_t active = cols;
    auto mask = V::UpdateMask<float>(active);
    V::RegTensor<uint32_t> indices;
    V::Arange(reinterpret_cast<V::RegTensor<int32_t>&>(indices), 0);
    V::Muls(indices, indices, 2u, mask);
#define DECLARE_N_ROW(R) V::RegTensor<float> real##R, imag##R;
    CSYRKX_ORDERED_ROWS(DECLARE_N_ROW)
#undef DECLARE_N_ROW
#define LOAD_N_ROW(R)                                     \
    if (FullRows || rows > R) {                           \
        V::LoadAlign(real##R, sums + (2 * R) * COLS);     \
        V::LoadAlign(imag##R, sums + (2 * R + 1) * COLS); \
    }
    CSYRKX_ORDERED_ROWS(LOAD_N_ROW)
#undef LOAD_N_ROW
#pragma clang loop unroll(disable)
    for (uint32_t l = 0; l < count; ++l) {
        V::RegTensor<float> br, bi, tr, ti, p0, p1;
        V::MaskReg nzReal, nzImag, nonzero;
        V::Gather(br, b + l * N_B_STRIDE, indices, mask);
        V::Gather(bi, b + l * N_B_STRIDE + 1, indices, mask);
        V::Compares<float, CMPMODE::NE>(nzReal, br, 0.0f, mask);
        V::Compares<float, CMPMODE::NE>(nzImag, bi, 0.0f, mask);
        V::Or(nonzero, nzReal, nzImag, mask);
        // Original zero B skips the entry; NaN B compares nonzero.
        V::Muls(tr, br, alphaReal, nonzero);
        V::Muls(p0, bi, alphaImag, nonzero);
        V::Sub(tr, tr, p0, nonzero);
        V::Muls(ti, bi, alphaReal, nonzero);
        V::Muls(p1, br, alphaImag, nonzero);
        V::Add(ti, ti, p1, nonzero);
#define UPDATE_N_ROW(R)                                             \
    if (FullRows || rows > R) {                                     \
        AccumulateRowN<R>(a, l, tr, ti, real##R, imag##R, nonzero); \
    }
        CSYRKX_ORDERED_ROWS(UPDATE_N_ROW)
#undef UPDATE_N_ROW
    }
#define STORE_N_ROW(R)                                           \
    if (FullRows || rows > R) {                                  \
        V::StoreAlign(sums + (2 * R) * COLS, real##R, mask);     \
        V::StoreAlign(sums + (2 * R + 1) * COLS, imag##R, mask); \
    }
    CSYRKX_ORDERED_ROWS(STORE_N_ROW)
#undef STORE_N_ROW
    V::LocalMemBar<V::MemType::VEC_STORE, V::MemType::VEC_LOAD>();
}
#undef CSYRKX_ORDERED_ROWS

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

// GM strides are in bytes; UB strides are in 32-byte blocks. Explicit fallback
// retains 64-bit GM addressing even when a legal leading dimension's byte
// stride is too large for the DMA descriptor's uint32_t srcStride.
__aicore__ inline void LoadColumns(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t columns,
    uint32_t complexRows, uint32_t ld, uint32_t ubStride)
{
    DataCopyExtParams p{};
    p.blockCount = columns;
    p.blockLen = 2 * complexRows * sizeof(float);
    const uint64_t sourceGap = 2ULL * (ld - complexRows) * sizeof(float);
    const uint32_t rounded = (2 * complexRows + BLOCK_FLOATS - 1) / BLOCK_FLOATS * BLOCK_FLOATS;
    p.dstStride = (ubStride - rounded) / BLOCK_FLOATS;
    DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    if (sourceGap <= 0xffffffffULL) {
        p.srcStride = static_cast<uint32_t>(sourceGap);
        DataCopyPad(dst, src[offset], p, padding);
    } else {
        p.blockCount = 1;
        p.srcStride = 0;
        p.dstStride = 0;
        for (uint32_t j = 0; j < columns; ++j) {
            DataCopyPad(dst[j * ubStride], src[offset + 2ULL * j * ld], p, padding);
        }
    }
}

// N initializes every active lane before the triangular output mask is applied.
// Clear skipped C lanes instead of reading the opposite triangle to fill them.
__simd_vf__ inline void ClearOldCTile(__ubuf__ float* oldC)
{
    namespace V = AscendC::MicroAPI;
    auto mask = V::CreateMask<float, V::MaskPattern::ALL>();
    V::RegTensor<float> zero;
    V::Duplicate(zero, 0.0f);
    for (uint32_t offset = 0; offset < COLS * OUTPUT_STRIDE; offset += COLS) {
        V::StoreAlign(oldC + offset, zero, mask);
    }
    V::LocalMemBar<V::MemType::VEC_STORE, V::MemType::VEC_LOAD>();
}

__aicore__ inline void LoadOldCTriangle(
    const LocalTensor<float>& dst, const GlobalTensor<float>& c, const CsyrkxTilingData& t, uint32_t row, uint32_t col,
    uint32_t rows, uint32_t cols, bool whole)
{
    static_assert(2 * ROWS == BLOCK_FLOATS, "C fringe padding must fit one DMA block");
    if (whole) {
        LoadColumns(dst, c, 2ULL * (static_cast<uint64_t>(col) * t.ldc + row), cols, rows, t.ldc, OUTPUT_STRIDE);
        return;
    }
    asc_vf_call<ClearOldCTile>(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()));
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    for (uint32_t j = 0; j < cols;) {
        const uint32_t globalCol = col + j;
        uint32_t first = 0;
        uint32_t end = rows;
        if (t.upper) {
            if (globalCol < row) {
                ++j;
                continue;
            }
            end = MinU32(rows, globalCol - row + 1);
        } else {
            first = globalCol > row ? globalCol - row : 0;
            if (first >= rows) {
                ++j;
                continue;
            }
        }
        const uint64_t offset = 2ULL * (static_cast<uint64_t>(globalCol) * t.ldc + row);
        if (first == 0 && end == rows) {
            // Batch wholly selected columns; only the diagonal fringe needs
            // individual DMA descriptors. Keep the large-LD fallback as well.
            const uint32_t count = t.upper ? cols - j : MinU32(cols - j, row - globalCol + 1);
            LoadColumns(dst[j * OUTPUT_STRIDE], c, offset, count, rows, t.ldc, OUTPUT_STRIDE);
            j += count;
            continue;
        }
        const DataCopyExtParams copy{1, 2 * (end - first) * uint32_t(sizeof(float)), 0, 0, 0};
        // ROWS=4: at most six padding floats on either side, within one
        // 32-byte block. Left padding retains oldC[j*OUTPUT_STRIDE + 2*r]
        // indexing while the GM read contains only selected complex values.
        const DataCopyPadExtParams<float> pad{
            true, static_cast<uint8_t>(2 * first), static_cast<uint8_t>(2 * (ROWS - end)), 0.0f};
        DataCopyPad(dst[j * OUTPUT_STRIDE], c[offset + 2 * first], copy, pad);
        ++j;
    }
}

__aicore__ inline void StoreTile(
    const GlobalTensor<float>& c, const LocalTensor<float>& out, const CsyrkxTilingData& t, uint32_t row, uint32_t col,
    uint32_t rows, uint32_t cols, bool whole)
{
    DataCopyExtParams p{};
    const uint64_t destinationGap = 2ULL * (t.ldc - rows) * sizeof(float);
    if (whole && destinationGap <= 0xffffffffULL) {
        p.blockCount = cols;
        p.blockLen = 2 * rows * sizeof(float);
        const uint32_t rounded = (2 * rows + BLOCK_FLOATS - 1) / BLOCK_FLOATS * BLOCK_FLOATS;
        p.srcStride = (OUTPUT_STRIDE - rounded) / BLOCK_FLOATS;
        p.dstStride = static_cast<uint32_t>(destinationGap);
        DataCopyPad(c[2ULL * (static_cast<uint64_t>(col) * t.ldc + row)], out, p);
        return;
    }
    // Exact byte lengths: no read/modify/write of the other triangle or padding.
    // FinishTile has already packed any lower-triangle prefix out of each column.
    p.blockCount = 1;
    for (uint32_t j = 0; j < cols; ++j) {
        const uint32_t globalCol = col + j;
        uint32_t first = 0;
        uint32_t end = rows;
        if (t.upper) {
            if (globalCol < row) {
                continue;
            }
            end = MinU32(rows, globalCol - row + 1);
        } else {
            first = globalCol > row ? globalCol - row : 0;
            if (first >= rows) {
                continue;
            }
        }
        p.blockLen = 2 * (end - first) * sizeof(float);
        DataCopyPad(c[2ULL * (static_cast<uint64_t>(globalCol) * t.ldc + row + first)], out[j * OUTPUT_STRIDE], p);
    }
}

__aicore__ inline void RunTile(
    const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    const LocalTensor<float>& ub, const CsyrkxTilingData& t, uint32_t row, uint32_t col, uint32_t rows, uint32_t cols)
{
    auto* p = reinterpret_cast<__ubuf__ float*>(ub.GetPhyAddr());
    const bool whole = t.upper ? row + rows - 1 <= col : row >= col + cols - 1;
    if (t.scaleOnly || t.k == 0) {
        asc_vf_call<AccumulateChunk<false>>(p + A_OFFSET, p + B_OFFSET, p + SUM_OFFSET, rows, cols, 0, true);
    } else {
        for (uint32_t first = 0; first < t.k;) {
            const uint32_t count = MinU32(K_CHUNK, t.k - first);
            LoadColumns(
                ub[A_OFFSET], a, 2ULL * (static_cast<uint64_t>(row) * t.lda + first), rows, count, t.lda, INPUT_STRIDE);
            LoadColumns(
                ub[B_OFFSET], b, 2ULL * (static_cast<uint64_t>(col) * t.ldb + first), cols, count, t.ldb, INPUT_STRIDE);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (rows == ROWS) {
                asc_vf_call<AccumulateChunk<true>>(
                    p + A_OFFSET, p + B_OFFSET, p + SUM_OFFSET, rows, cols, count, first == 0);
            } else {
                asc_vf_call<AccumulateChunk<false>>(
                    p + A_OFFSET, p + B_OFFSET, p + SUM_OFFSET, rows, cols, count, first == 0);
            }
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            first += count;
        }
    }
    if (!t.betaZero) {
        LoadOldCTriangle(ub[C_OFFSET], c, t, row, col, rows, cols, whole);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<FinishTile<false>>(
            p + SUM_OFFSET, p + C_OFFSET, p + OUT_OFFSET, row, col, rows, cols, t.upper, whole, t.scaleOnly, t.ar, t.ai,
            t.br, t.bi);
    } else {
        // Neither GM nor UB C is read in the beta=0 specialization.
        asc_vf_call<FinishTile<true>>(
            p + SUM_OFFSET, p + C_OFFSET, p + OUT_OFFSET, row, col, rows, cols, t.upper, whole, t.scaleOnly, t.ar, t.ai,
            t.br, t.bi);
    }
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreTile(c, ub[OUT_OFFSET], t, row, col, rows, cols, whole);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

__aicore__ inline void RunTileN(
    const GlobalTensor<float>& a, const GlobalTensor<float>& b, const GlobalTensor<float>& c,
    const LocalTensor<float>& ub, const CsyrkxTilingData& t, uint32_t row, uint32_t col, uint32_t rows, uint32_t cols)
{
    auto* p = reinterpret_cast<__ubuf__ float*>(ub.GetPhyAddr());
    const bool whole = t.upper ? row + rows - 1 <= col : row >= col + cols - 1;
    if (t.betaZero) {
        asc_vf_call<InitializeN<true>>(p + SUM_OFFSET, p + C_OFFSET, rows, cols, t.br, t.bi);
    } else {
        LoadOldCTriangle(ub[C_OFFSET], c, t, row, col, rows, cols, whole);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<InitializeN<false>>(p + SUM_OFFSET, p + C_OFFSET, rows, cols, t.br, t.bi);
    }
    if (!t.scaleOnly) {
        for (uint32_t first = 0; first < t.k;) {
            const uint32_t count = MinU32(K_CHUNK, t.k - first);
            // For N, GM columns are successive K entries, each a contiguous
            // output-row (A) or output-column (B) span. T buffers are reused.
            LoadColumns(
                ub[A_OFFSET], a, 2ULL * (static_cast<uint64_t>(first) * t.lda + row), count, rows, t.lda, N_A_STRIDE);
            LoadColumns(
                ub[B_OFFSET], b, 2ULL * (static_cast<uint64_t>(first) * t.ldb + col), count, cols, t.ldb, N_B_STRIDE);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            if (rows == ROWS) {
                asc_vf_call<AccumulateChunkN<true>>(
                    p + A_OFFSET, p + B_OFFSET, p + SUM_OFFSET, rows, cols, count, t.ar, t.ai);
            } else {
                asc_vf_call<AccumulateChunkN<false>>(
                    p + A_OFFSET, p + B_OFFSET, p + SUM_OFFSET, rows, cols, count, t.ar, t.ai);
            }
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            first += count;
        }
    }
    // Pure packing only: beta was incorporated before K and alpha before A.
    asc_vf_call<FinishTile<true>>(
        p + SUM_OFFSET, p + C_OFFSET, p + OUT_OFFSET, row, col, rows, cols, t.upper, whole, true, t.ar, t.ai, t.br,
        t.bi);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreTile(c, ub[OUT_OFFSET], t, row, col, rows, cols, whole);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

extern "C" __global__ __aicore__ void csyrkx_ordered_kernel(
    GM_ADDR aPtr, GM_ADDR bPtr, GM_ADDR cPtr, CsyrkxTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (t.n == 0) {
        return;
    }
    GlobalTensor<float> a, b, c;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPtr));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bPtr));
    c.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cPtr));
    LocalTensor<float> ub(TPosition::VECCALC, 0, UB_FLOATS);
    const uint64_t groups = (static_cast<uint64_t>(t.n) + COLS - 1) / COLS;
    constexpr uint32_t ROW_SUBTILES = COLS / ROWS;
    const uint64_t tasks = groups * (groups + 1) / 2 * ROW_SUBTILES;
    // Enumerate ONLY triangular 64x64 macrotiles, subdivided into 4x64.
    // A rectangular grid followed by triangle rejection badly imbalances cyclic
    // assignment whenever the number of row tiles is a multiple of core count.
    for (uint64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const uint64_t macro = task / ROW_SUBTILES;
        uint64_t low = 0, high = groups;
        while (low + 1 < high) {
            const uint64_t middle = (low + high) / 2;
            if (middle * (middle + 1) / 2 <= macro) {
                low = middle;
            } else {
                high = middle;
            }
        }
        const uint64_t inner = macro - low * (low + 1) / 2;
        const uint64_t row64 = (t.upper ? inner : low) * COLS + (task % ROW_SUBTILES) * ROWS;
        const uint64_t col64 = (t.upper ? low : inner) * COLS;
        if (row64 >= t.n || col64 >= t.n) {
            continue;
        }
        const uint32_t row = static_cast<uint32_t>(row64);
        const uint32_t col = static_cast<uint32_t>(col64);
        const uint32_t rows = MinU32(ROWS, t.n - row);
        const uint32_t cols = MinU32(COLS, t.n - col);
        if ((t.upper && row > col + cols - 1) || (!t.upper && row + rows - 1 < col)) {
            continue;
        }
        if (t.trans) {
            RunTile(a, b, c, ub, t, row, col, rows, cols);
        } else {
            RunTileN(a, b, c, ub, t, row, col, rows, cols);
        }
    }
}
} // namespace csyrkx_ordered

void csyrkx_ordered_do(
    uint32_t blocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, const CsyrkxTilingData& t)
{
    csyrkx_ordered::csyrkx_ordered_kernel<<<blocks, nullptr, stream>>>(
        const_cast<uint8_t*>(a), const_cast<uint8_t*>(b), c, t);
}
#endif
