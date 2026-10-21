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
 * \file sgbmv_kernel_vec.cpp
 * \brief Vector-fallback (wide-band N/T) and small-band staged paths of the arch22
 * sgbmv kernel; split out of sgbmv_kernel.cpp for source-size limits.
 */


// ============================================================================
// Small-band vectorized fast paths (band <= 32; the scalar loops are
// scalar-pipe bound at 0.88-0.97 utilization). Diagonal
// formulation over the staged full columns: for output lane i (row tileStart+i
// on the N side), y[i] += AB(j, c(i, j)) * x[c(i, j)] with c = r + ku - j, so
// for a fixed band row j the A elements sit at stride lda in the staged
// columns (one strided Gather) and the x elements at stride 1 (direct tensor
// offset). Semantically-out-of-range (lane, j) pairs are neutralized by staged
// zero margins on both A and x, so every lane keeps its exact scalar-loop
// value. Returns false when the per-tile margins do not fit the UB budget; the
// caller then uses the original scalar path for that tile.
// ============================================================================
// Fold the max magnitude of the n8 (8-aligned, fully written) absolute-value
// tensor in ubB down to <= 16 lanes with halving elementwise Max, then scan
// the remaining lanes on the scalar pipe. The fold ping-pongs between ubB and
// ubS so dst never overlaps the src ranges (elementwise ops only permit
// dst == src0 aliasing). Avoids the Level-2 Reduce* family (their internals
// use GetTPipePtr()->FetchEventID, and this kernel deliberately manages fixed
// event IDs without a TPipe).
__aicore__ inline float SmallBandFoldMaxAbs(LocalTensor<float>& ubB, LocalTensor<float>& ubS, uint32_t n8)
{
    uint32_t m = n8;
    bool inB = true;
    while (m > 16U) {
        uint32_t half = (m / 2U) & ~(SIMD_WIDTH - 1U);  // 8-aligned half, <= m/2
        if (half == 0U) {
            break;
        }
        LocalTensor<float>& src = inB ? ubB : ubS;
        LocalTensor<float>& dst = inB ? ubS : ubB;
        // dst[i] = max(src[i], src[half + i]) for i < m - half; every source
        // index is covered exactly (the [half, m-half) band is read twice,
        // which is idempotent for max).
        Max(dst, src, src[half], m - half);
        PIPE_BARRIER(V);
        m -= half;
        inB = !inB;
    }
    SET_FLAG(V, S, EVENT_ID2);
    WAIT_FLAG(V, S, EVENT_ID2);
    auto p = reinterpret_cast<__ubuf__ const float*>((inB ? ubB : ubS).GetPhyAddr());
    float r = 0.0f;
    for (uint32_t i = 0U; i < m; ++i) {
        float v = p[i];
        if (v != v) {
            SET_FLAG(S, V, EVENT_ID2);
            WAIT_FLAG(S, V, EVENT_ID2);
            return v;  // NaN: caller's <= test fails and takes the fallback
        }
        if (v > r) {
            r = v;
        }
    }
    SET_FLAG(S, V, EVENT_ID2);
    WAIT_FLAG(S, V, EVENT_ID2);
    return r;
}

// Budget and geometry check for the small-band staged-diagonal tile: computes
// the staged row/column geometry and rejects tiles whose A/x staging (with
// the zero margins) exceeds the UB budget (caller then keeps the scalar path).
__aicore__ inline bool VecSmallBandCheckBudget(
    uint32_t tileStart, uint32_t tileEnd, uint32_t colStartAligned, uint32_t colEndAligned,
    const SgbmvTilingData& tiling, uint32_t& rows8, uint32_t& totalCols, uint32_t& marginU, uint32_t& xFloats8)
{
    const uint32_t numRows = tileEnd - tileStart;
    const uint32_t rows8Local = (numRows + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    const uint32_t totalColsLocal = colEndAligned - colStartAligned;
    // Highest (spurious included) column touched: c = tileStart + rows8 - 1 + ku.
    uint32_t marginULocal = (tileStart + rows8Local + tiling.ku > colEndAligned)
                           ? (tileStart + rows8Local + tiling.ku - colEndAligned) : 0U;
    const uint32_t aFloats = (SMALLBAND_A_MARGIN_COLS + totalColsLocal + marginULocal + 8U) * tiling.lda;
    if (((aFloats + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U)) * BYTES_PER_FLOAT > MAX_UB_A_BYTES) {
        return false;
    }
    const uint32_t xFloats = SMALLBAND_A_MARGIN_COLS + totalColsLocal + marginULocal + 16U;
    if (xFloats > 2048U) {
        return false;
    }
    rows8 = rows8Local;
    totalCols = totalColsLocal;
    marginU = marginULocal;
    xFloats8 = (xFloats + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    return true;
}

// Stage A with a lower zero margin (spurious low columns), the real aligned
// columns, and an upper zero margin for tiles that run past column n-1.
// The upper pad starts at the 8-float-aligned end of the staged run (the
// DMA's right pad fills the sub-alignment slack with zeros); vector-op
// tensor offsets must stay 8-float aligned.
__aicore__ inline void VecSmallBandStageA(
    LocalTensor<float>& ubA, GlobalTensor<float>& aGlobal, uint64_t aGmOff, uint32_t totalCols, uint32_t marginU,
    const SgbmvTilingData& tiling)
{
    const uint32_t loMargin = SMALLBAND_A_MARGIN_COLS * tiling.lda;
    Duplicate(ubA, 0.0f, loMargin);
    CopyGmToUb(ubA[loMargin], aGlobal[aGmOff], totalCols * tiling.lda);
    {
        uint32_t padFrom = loMargin + ((totalCols * tiling.lda + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U));
        uint32_t padEnd =
            (((SMALLBAND_A_MARGIN_COLS + totalCols + marginU + 9U) * tiling.lda) + SIMD_WIDTH - 1U) &
            ~(SIMD_WIDTH - 1U);
        if (padEnd > padFrom) {
            Duplicate(ubA[padFrom], 0.0f, padEnd - padFrom);
        }
    }
}

// Stage x with the same margins: [0, 32) zeros, [32, 32+totalCols) real,
// tail zeros to xFloats8. The margin Duplicates do not overlap the DMA data
// range (the DMA's own right pad and the tail Duplicate both write zeros,
// so their overlap is benign).
__aicore__ inline void VecSmallBandStageX(
    LocalTensor<float>& ubX, GlobalTensor<float>& xGlobal, uint32_t colStartAligned, uint32_t totalCols,
    uint32_t xFloats8)
{
    CopyGmToUb(ubX[SMALLBAND_A_MARGIN_COLS], xGlobal[colStartAligned], totalCols);
    Duplicate(ubX, 0.0f, SMALLBAND_A_MARGIN_COLS);
    {
        uint32_t xTailFrom = SMALLBAND_A_MARGIN_COLS + ((totalCols + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U));
        if (xFloats8 > xTailFrom) {
            Duplicate(ubX[xTailFrom], 0.0f, xFloats8 - xTailFrom);
        }
    }
}

// Mixed-gate overflow/tolerance guard (band in [3,8]).
// The vector Mul rounds each product before the Add while the scalar path
// compiles to fused fmadd, so overflow-class data (band*maxA*maxX beyond
// FLT_MAX) yields nan/-inf where golden has inf/finite-huge, and
// large-but-finite data amplifies the double
// rounding to ~2*band^2*maxA*maxX*2^-24*|alpha| absolute error. Requiring
// band^2*maxA*maxX <= 10486/|alpha| (== 2.5e-3 * 2^22/(|alpha|*...), a 4x
// margin under the verifier's 1e-2 floor) bounds both. Tiles failing the
// test return false and keep the bit-exact scalar path; band >= 9 tiles
// are unguarded exactly as before. The guard is a pre-loop elementwise fold
// so the hot j-loop below stays byte-identical to the band >= 9 path — a
// guarded loop copy measurably perturbs the unguarded tiles' code layout.
__aicore__ inline bool VecSmallBandGuardOk(
    LocalTensor<float>& ubA, LocalTensor<float>& ubX, LocalTensor<float>& ubB, LocalTensor<float>& ubS,
    uint32_t totalCols, uint32_t marginU, uint32_t xFloats8, const SgbmvTilingData& tiling)
{
    const uint32_t band = tiling.kl + tiling.ku + 1U;
    if (band <= SIMD_WIDTH) {
        const uint32_t aGuardEnd =
            (((SMALLBAND_A_MARGIN_COLS + totalCols + marginU + 9U) * tiling.lda) + SIMD_WIDTH - 1U) &
            ~(SIMD_WIDTH - 1U);
        if (aGuardEnd > 1024U || xFloats8 > 1024U) {
            return false;
        }
        Abs(ubB, ubA, aGuardEnd);
        float magA = SmallBandFoldMaxAbs(ubB, ubS, aGuardEnd);
        Abs(ubB, ubX, xFloats8);
        float magX = SmallBandFoldMaxAbs(ubB, ubS, xFloats8);
        float alphaAbs = tiling.alpha >= 0.0f ? tiling.alpha : -tiling.alpha;
        if (alphaAbs < 1.0f) {
            alphaAbs = 1.0f;
        }
        const float bandF = static_cast<float>(static_cast<int32_t>(band));
        // NaN magnitudes (NaN inputs) and Inf/overflow products all fail the
        // <= test and take the scalar fallback.
        if (!(bandF * bandF * magA * magX <= 10486.0f / alphaAbs)) {
            return false;
        }
    }
    return true;
}

// Diagonal-formulation accumulation: j descending == column index ascending,
// matching the original scalar row loop's accumulation order.
__aicore__ inline void VecSmallBandAccumulateDiagonal(
    LocalTensor<float>& ubY, LocalTensor<float>& ubA, LocalTensor<float>& ubX, LocalTensor<float>& ubB,
    LocalTensor<float>& ubS, LocalTensor<float>& ubT, LocalTensor<uint32_t>& tbl, LocalTensor<uint32_t>& tbl4,
    uint32_t tileStart, uint32_t colStartAligned, uint32_t rows8, const SgbmvTilingData& tiling)
{
    const uint32_t band = tiling.kl + tiling.ku + 1U;
    MakeOffsetSequence(tbl.ReinterpretCast<int32_t>(), rows8, static_cast<int32_t>(tiling.lda * BYTES_PER_FLOAT));
    MakeOffsetSequence(tbl4.ReinterpretCast<int32_t>(), rows8, static_cast<int32_t>(BYTES_PER_FLOAT));
    Duplicate(ubY, 0.0f, rows8);
    PIPE_BARRIER(V);

    for (int32_t j = static_cast<int32_t>(band) - 1; j >= 0; --j) {
        // A element for lane i: ubA[(c_i - colStartAligned + 32) * lda + j], c_i = tileStart + i + ku - j.
        uint32_t ju = static_cast<uint32_t>(j);
        uint32_t aBase = (tileStart + tiling.ku - ju - colStartAligned + SMALLBAND_A_MARGIN_COLS) * tiling.lda + ju;
        Gather(ubB, ubA, tbl, aBase * BYTES_PER_FLOAT, rows8);
        uint32_t xOff = tileStart + tiling.ku - ju - colStartAligned + SMALLBAND_A_MARGIN_COLS;
        Gather(ubS, ubX, tbl4, xOff * BYTES_PER_FLOAT, rows8);
        PIPE_BARRIER(V);
        Mul(ubT, ubB, ubS, rows8);
        PIPE_BARRIER(V);
        Add(ubY, ubY, ubT, rows8);
        PIPE_BARRIER(V);
    }
}

__aicore__ bool ProcessSgbmvNFastTileSmallBand(
    uint32_t tileStart, uint32_t tileEnd, uint32_t colStartAligned, uint32_t colEndAligned,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, GlobalTensor<float>& xGlobal,
    GlobalTensor<float>& yGlobal, uint64_t aGmOff, LocalTensor<float>& ubA, LocalTensor<float>& ubY,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    const uint32_t numRows = tileEnd - tileStart;
    uint32_t rows8 = 0U;
    uint32_t totalCols = 0U;
    uint32_t marginU = 0U;
    uint32_t xFloats8 = 0U;
    if (!VecSmallBandCheckBudget(tileStart, tileEnd, colStartAligned, colEndAligned, tiling, rows8, totalCols,
        marginU, xFloats8)) {
        return false;
    }

    auto ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X);
    auto ubB = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_A_VALUES);
    auto ubS = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_X_VALUES);
    auto ubT = buf.GetBuffer<BufferType::ASCEND_UB, float>(UB_OFFSET_SB_SCRATCH);
    auto tbl = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_A_INDICES);
    auto tbl4 = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(UB_OFFSET_TBL4);

    VecSmallBandStageA(ubA, aGlobal, aGmOff, totalCols, marginU, tiling);
    VecSmallBandStageX(ubX, xGlobal, colStartAligned, totalCols, xFloats8);
    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);

    if (!VecSmallBandGuardOk(ubA, ubX, ubB, ubS, totalCols, marginU, xFloats8, tiling)) {
        return false;
    }

    VecSmallBandAccumulateDiagonal(
        ubY, ubA, ubX, ubB, ubS, ubT, tbl, tbl4, tileStart, colStartAligned, rows8, tiling);
    WriteBackContiguousVec8(tileStart, numRows, rows8, tiling.alpha, tiling.beta, yGlobal, ubY, buf);
    return true;
}

// Eight independent fmadd accumulators for the fallback column dots; the
// struct keeps them as plain scalar members so the unrolled chain stays
// register-allocated.
struct FallbackDotAcc8 {
    float s0;
    float s1;
    float s2;
    float s3;
    float s4;
    float s5;
    float s6;
    float s7;
};

// incx == 1 column dot: contiguous x run, 8-way unrolled fmadd chain.
__aicore__ inline float TFallbackDotContiguous(
    uint32_t rowStart, uint32_t len, uint64_t aOff, __gm__ const float* aGm, __gm__ const float* xGm)
{
    FallbackDotAcc8 s{};
    uint32_t k = 0U;
    for (; k + 7U < len; k += UNROLL_STEP) {
        s.s0 += aGm[aOff + k + 0U] * xGm[rowStart + k + 0U];
        s.s1 += aGm[aOff + k + 1U] * xGm[rowStart + k + 1U];
        s.s2 += aGm[aOff + k + 2U] * xGm[rowStart + k + 2U];
        s.s3 += aGm[aOff + k + 3U] * xGm[rowStart + k + 3U];
        s.s4 += aGm[aOff + k + 4U] * xGm[rowStart + k + 4U];
        s.s5 += aGm[aOff + k + 5U] * xGm[rowStart + k + 5U];
        s.s6 += aGm[aOff + k + 6U] * xGm[rowStart + k + 6U];
        s.s7 += aGm[aOff + k + 7U] * xGm[rowStart + k + 7U];
    }
    float acc = ((s.s0 + s.s1) + (s.s2 + s.s3)) + ((s.s4 + s.s5) + (s.s6 + s.s7));
    for (; k < len; ++k) {
        acc += aGm[aOff + k] * xGm[rowStart + k];
    }
    return acc;
}

// Strided x index for output row r (negative incx mirrors from the top).
__aicore__ inline int64_t TFallbackXIndex(uint32_t row, const SgbmvTilingData& tiling)
{
    return (tiling.incx >= 0) ? (static_cast<int64_t>(row) * tiling.incx) :
                                (static_cast<int64_t>(tiling.m - 1U - row) * (-tiling.incx));
}

// incx != 1 column dot: per-lane strided x index, 8-way unrolled.
__aicore__ inline float TFallbackDotStrided(
    uint32_t rowStart, uint32_t len, uint64_t aOff, const SgbmvTilingData& tiling, __gm__ const float* aGm,
    __gm__ const float* xGm)
{
    FallbackDotAcc8 s{};
    uint32_t k = 0U;
    for (; k + 7U < len; k += UNROLL_STEP) {
        uint32_t row = rowStart + k;
        int64_t xIdx0 = TFallbackXIndex(row + 0U, tiling);
        int64_t xIdx1 = TFallbackXIndex(row + 1U, tiling);
        int64_t xIdx2 = TFallbackXIndex(row + 2U, tiling);
        int64_t xIdx3 = TFallbackXIndex(row + 3U, tiling);
        int64_t xIdx4 = TFallbackXIndex(row + 4U, tiling);
        int64_t xIdx5 = TFallbackXIndex(row + 5U, tiling);
        int64_t xIdx6 = TFallbackXIndex(row + 6U, tiling);
        int64_t xIdx7 = TFallbackXIndex(row + 7U, tiling);
        s.s0 += aGm[aOff + k + 0U] * xGm[xIdx0];
        s.s1 += aGm[aOff + k + 1U] * xGm[xIdx1];
        s.s2 += aGm[aOff + k + 2U] * xGm[xIdx2];
        s.s3 += aGm[aOff + k + 3U] * xGm[xIdx3];
        s.s4 += aGm[aOff + k + 4U] * xGm[xIdx4];
        s.s5 += aGm[aOff + k + 5U] * xGm[xIdx5];
        s.s6 += aGm[aOff + k + 6U] * xGm[xIdx6];
        s.s7 += aGm[aOff + k + 7U] * xGm[xIdx7];
    }
    float acc = ((s.s0 + s.s1) + (s.s2 + s.s3)) + ((s.s4 + s.s5) + (s.s6 + s.s7));
    for (; k < len; ++k) {
        uint32_t row = rowStart + k;
        int64_t xIdx = TFallbackXIndex(row, tiling);
        acc += aGm[aOff + k] * xGm[xIdx];
    }
    return acc;
}

// One output column of the scalar T fallback: clipped row range + dot product.
__aicore__ inline float TFallbackComputeColumn(
    uint32_t col, const SgbmvTilingData& tiling, __gm__ const float* aGm, __gm__ const float* xGm)
{
    uint32_t rowStart = (col >= tiling.ku) ? (col - tiling.ku) : 0U;
    uint32_t rowEnd = MinVal(col + tiling.kl + 1U, tiling.m);
    float acc = 0.0f;
    if (rowStart < rowEnd) {
        uint32_t len = rowEnd - rowStart;
        uint64_t aOff = static_cast<uint64_t>(tiling.lda) * col + (tiling.ku + rowStart - col);
        if (tiling.incx == 1) {
            acc = TFallbackDotContiguous(rowStart, len, aOff, aGm, xGm);
        } else {
            acc = TFallbackDotStrided(rowStart, len, aOff, tiling, aGm, xGm);
        }
    }
    return acc;
}

__aicore__ void ProcessSgbmvTFallback(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    for (uint32_t col = startRow; col < endRow; ++col) {
        float acc = TFallbackComputeColumn(col, tiling, aGm, xGm);
        WriteY(col, tiling.n, tiling.incy, tiling.alpha, tiling.beta, acc, yGm);
    }
}

// ============================================================================
// Vectorized fallback for the wide-band domain (band >= 130).
//
// N path: per-core row subtile [ts, te); y accumulates in UB. Columns are
// processed as contiguous A-segment AXPYs:
//   - full phase (segment covers the whole subtile, the common case when
//     band >= subtile height): 2D blockCount batch DMA, blockLen = Lt*4,
//     srcStride = (lda-1-Lt)*4 -> one MTE2 op moves up to 24KB.
//   - prefix phase (segment starts at subtile top, off == 0): per-column
//     exact DMA + Muls + Add.
//   - suffix phase (segment slides up): per-column DMA + Muls + Gather
//     (variant table) + Add at the 8-aligned floor offset. The variant
//     table maps the <h leading lanes to a dedicated zero slot so the
//     misaligned head adds exact zeros.
// Each A element is read exactly once globally (row split, clipped segments).
//
// T path: per-core column subtile; each output is a contiguous-segment dot
// product. Interior columns (full band) are batch DMA'd (blockCount, rows
// padded to a power-of-two multiple of 8 for the halving reduction); the x
// operand is preloaded once per subtile and shifted per column with a linear
// Gather. Reduction is pairwise halving down to 8 lanes, stored into an
// [L, 8] partial matrix and folded once by ReduceDataBlock.
// ============================================================================
// Unified UB layout for the vector fallback paths (N and T are exclusive).
// Table region: [VB_TBL, +8*(VEC_L_MAX+16)) = 8 N suffix head-shift variant
// tables (N path) or the T linear gather table (T path); the N linear table
// follows at VB_TBL_LIN.
constexpr uint32_t VB_Y = 0U;                                     // (VEC_L_MAX+16) floats
constexpr uint32_t VB_YP = 4096U;                                 // VEC_T_L_MAX*8 floats
constexpr uint32_t VB_YORIG = VB_YP + VEC_T_L_MAX * 8U * BYTES_PER_FLOAT; // (VEC_L_MAX+16) floats
constexpr uint32_t VB_XWIN = VB_YORIG + (VEC_L_MAX + 16U) * BYTES_PER_FLOAT;
constexpr uint32_t VB_ABUF = VB_XWIN + VEC_XWIN_ELEMS * BYTES_PER_FLOAT;
constexpr uint32_t VB_XS = VB_ABUF + 2U * VEC_ABUF_ELEMS * BYTES_PER_FLOAT;
constexpr uint32_t VB_P = VB_XS + VEC_ROWP_MAX * BYTES_PER_FLOAT;
constexpr uint32_t VB_TBL = VB_P + VEC_ROWP_MAX * BYTES_PER_FLOAT;
constexpr uint32_t VB_TBL_ELEMS = 8U * (VEC_L_MAX + 16U); // 8 N variant tables / T linear table
constexpr uint32_t VB_A2 = VB_TBL + VB_TBL_ELEMS * 4U;    // aligned A staging (rowP floats)
constexpr uint32_t VB_TBL_LIN = VB_A2 + VEC_ROWP_MAX * BYTES_PER_FLOAT; // N linear table
static_assert(VB_Y + (VEC_L_MAX + 16U) * BYTES_PER_FLOAT <= VB_YP, "VB_Y must fit before VB_YP");
static_assert(VB_TBL_ELEMS >= VEC_ROWP_MAX, "T linear table must fit in table region");
static_assert(VB_TBL_LIN + (VEC_L_MAX + 16U) * 4U <= 192U * 1024U, "vector fallback buffers must fit in UB");

// Writeback for the vector paths: y = alpha * acc + beta * y on a contiguous
// range. Uses the vector-fallback layout's own y-orig buffer (rows can exceed
// the 192-float region the fast-path writeback relies on).
__aicore__ inline void VecWriteBack(
    uint32_t tileStart, uint32_t numRows, float alpha, float beta, GlobalTensor<float>& yGlobal,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    auto ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_Y);
    PIPE_BARRIER(V);
    uint32_t n8 = (numRows + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    if (alpha != 1.0f) {
        Muls(ubY, ubY, alpha, n8);
        PIPE_BARRIER(V);
    }
    if (beta != 0.0f) {
        auto yOrig = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_YORIG);
        VecCopyIn(yOrig, yGlobal[tileStart], numRows);
        SET_FLAG(MTE2, V, EVENT_ID1);
        WAIT_FLAG(MTE2, V, EVENT_ID1);
        Muls(yOrig, yOrig, beta, n8);
        PIPE_BARRIER(V);
        Add(ubY, ubY, yOrig, n8);
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);
    CopyUbToGm(yGlobal[tileStart], ubY, numRows);
    SET_FLAG(MTE3, V, EVENT_ID0);
    WAIT_FLAG(MTE3, V, EVENT_ID0);
    PIPE_BARRIER(ALL);
}

// ---- N vector path (all GM reads 32B-aligned; alignment restored in UB by a
// base-shifted linear Gather) -----------------------------------------------

// Per-column AXPY: slot holds [aOff-r, aOff+len) zero-padded; ubA2 gets the
// aligned segment via gather base r*4; products scaled by xj; off == 0 adds
// directly, off > 0 (sliding suffix window) shifts with the h-variant table.
__aicore__ inline void VecNColumnCompute(
    LocalTensor<float> ubY, LocalTensor<float> ubT, LocalTensor<float> ubA2,
    LocalTensor<uint32_t> tblLin, LocalTensor<uint32_t> tblVar, LocalTensor<float> slot, float xj,
    uint32_t r, uint32_t off, uint32_t len, uint32_t Lt)
{
    uint32_t lenr8 = (len + r + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    if (Lt + 16U > lenr8) {
        Duplicate(slot[lenr8], 0.0f, Lt + 16U - lenr8);
        PIPE_BARRIER(V);
    }
    if (off == 0U) {
        Gather(ubA2, slot, tblLin, r * BYTES_PER_FLOAT, Lt);
        PIPE_BARRIER(V);
        Muls(ubT, ubA2, xj, Lt);
        PIPE_BARRIER(V);
        Add(ubY, ubY, ubT, Lt);
        PIPE_BARRIER(V);
        return;
    }
    // sliding suffix window: the h-variant table lands the products directly
    // on the 8-aligned floor offset (leading lanes read the zero slot)
    uint32_t h = off & (SIMD_WIDTH - 1U);
    Gather(ubA2, slot, tblVar[h * (VEC_L_MAX + 16U)], r * BYTES_PER_FLOAT, Lt);
    PIPE_BARRIER(V);
    Muls(ubT, ubA2, xj, Lt);
    PIPE_BARRIER(V);
    uint32_t len8 = (len + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    uint32_t offA = off - h;
    uint32_t endA = (off + len8) & ~(SIMD_WIDTH - 1U);
    if (endA > offA) {
        Add(ubY[offA], ubY[offA], ubT, endA - offA);
        PIPE_BARRIER(V);
    }
}

// Column segment geometry for the N path: rows [r0, r1) of column jc clipped
// to the subtile [ts, te).
__aicore__ inline uint32_t VecNColOff(uint32_t jc, uint32_t ts, const SgbmvTilingData& tiling)
{
    return (jc > ts + tiling.ku) ? (jc - tiling.ku - ts) : 0U;
}

__aicore__ inline uint32_t VecNColLen(uint32_t jc, uint32_t ts, uint32_t te, const SgbmvTilingData& tiling)
{
    uint32_t off = VecNColOff(jc, ts, tiling);
    uint32_t r1 = MinVal(jc + tiling.kl + 1U, te);
    return (r1 > ts + off) ? (r1 - (ts + off)) : 0U;
}

// Stage one column segment from an aligned GM base: [aOff-r, aOff+len).
__aicore__ inline void VecNIssueColumn(
    LocalTensor<float> slot, GlobalTensor<float>& aGlobal, uint32_t jc, uint32_t ts, uint32_t te,
    const SgbmvTilingData& tiling)
{
    uint32_t off = VecNColOff(jc, ts, tiling);
    uint32_t len = VecNColLen(jc, ts, te, tiling);
    uint64_t aOff = static_cast<uint64_t>(tiling.lda) * jc + (tiling.ku + ts + off - jc);
    uint32_t r = static_cast<uint32_t>(aOff & 7ULL);
    VecCopyIn(slot, aGlobal[aOff - r], len + r);
}

// Build the 8 head-shift variant tables once per core (suffix columns only).
__aicore__ inline void VecNBuildVariantTables(LocalTensor<uint32_t> tblAll)
{
    auto ptr = reinterpret_cast<__ubuf__ uint32_t*>(tblAll.GetPhyAddr());
    const uint32_t stride = VEC_L_MAX + 16U;
    const uint32_t zeroSlotBytes = VEC_ZERO_SLOT * BYTES_PER_FLOAT;
    for (uint32_t h = 0U; h < SIMD_WIDTH; ++h) {
        __ubuf__ uint32_t* dst = ptr + h * stride;
        for (uint32_t m = 0U; m < h; ++m) {
            dst[m] = zeroSlotBytes;
        }
        for (uint32_t m = h; m < stride; ++m) {
            dst[m] = (m - h) * BYTES_PER_FLOAT;
        }
    }
    SET_FLAG(S, V, EVENT_ID0);
    WAIT_FLAG(S, V, EVENT_ID0);
}

// Column range geometry for one N vector subtile [ts, te): the intersecting
// column span plus the full-phase prefix/suffix boundaries.
__aicore__ inline void VecNTileColumnRange(
    uint32_t ts, uint32_t te, const SgbmvTilingData& tiling, uint32_t& cLo, uint32_t& cHi, uint32_t& suffixStart,
    uint32_t& prefixEnd)
{
    cLo = (ts >= tiling.kl) ? ts - tiling.kl : 0U;
    cHi = MinVal(tiling.n, te + tiling.ku);
    suffixStart = MinVal(cHi, ts + tiling.ku + 1U);
    prefixEnd = cLo;
    if (tiling.kl < te) {
        uint32_t pe = te - 1U - tiling.kl;
        prefixEnd = MaxVal(cLo, MinVal(cHi, pe));
    }
}

// Byte gap between consecutive same-residue columns in the full-phase 2D
// DMA: one row of 8 columns advances 8*(lda-1) elements minus the Lt+8
// block already copied. DataCopyExtParams::srcStride is a uint32 field, so
// callers must route strides wider than 32 bits to the per-column path.
__aicore__ inline uint64_t VecNFullGapBytes(uint32_t Lt, const SgbmvTilingData& tiling)
{
    return (8ULL * (tiling.lda - 1U) - (Lt + 8U)) * BYTES_PER_FLOAT;
}

// Issue one residue-class batch DMA into the ping-pong slot of the given
// parity and signal its MTE2->V event.
__aicore__ inline void VecNIssueBatchDma(
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, uint32_t slotParity, GlobalTensor<float>& aGlobal, uint64_t srcBase,
    uint32_t cnt, uint32_t Lt, uint32_t srcGap)
{
    auto slot = buf.GetBuffer<BufferType::ASCEND_UB, float>(
        VB_ABUF + slotParity * VEC_ABUF_ELEMS * BYTES_PER_FLOAT);
    AscendC::DataCopyPad(
        slot, aGlobal[srcBase],
        AscendC::DataCopyExtParams(static_cast<uint16_t>(cnt), (Lt + 8U) * BYTES_PER_FLOAT, srcGap, 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0.0f));
    SetMte2SlotReady(slotParity);
}

// Consume one staged batch: gather each column to alignment, scale by xj and
// accumulate into ubY; then drain the V pipe before this slot may be
// overwritten by the DMA staged in the next iteration (load-bearing: measured
// corruption without it).
__aicore__ inline void VecNConsumeAndDrainBatch(
    LocalTensor<float>& ubY, LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, uint32_t cp, uint32_t j, uint32_t cnt0, uint32_t r,
    __ubuf__ const float* xPtr, uint32_t xAl, uint32_t Lt)
{
    auto slot = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + cp * VEC_ABUF_ELEMS * BYTES_PER_FLOAT);
    for (uint32_t c = 0U; c < cnt0; ++c) {
        uint32_t jc = j + c * SIMD_WIDTH;
        float xj = xPtr[jc - xAl];
        if (xj == 0.0f) {
            continue;
        }
        Gather(ubA2, slot[c * (Lt + 8U)], tblLin, r * BYTES_PER_FLOAT, Lt);
        PIPE_BARRIER(V);
        Muls(ubT, ubA2, xj, Lt);
        PIPE_BARRIER(V);
        Add(ubY, ubY, ubT, Lt);
        PIPE_BARRIER(V);
    }
    SET_FLAG(V, S, EVENT_ID2);
    WAIT_FLAG(V, S, EVENT_ID2);
}

// Full phase [prefixEnd, suffixStart): residue-class nburst batches.
// aOff(j+1) - aOff(j) = lda-1, so columns with j%8 == rho share one
// alignment residue r; one 2D DMA moves JC whole subtile columns.
// Batch ping-pong: batch i+1 DMA is issued
// before the V pipe waits on batch i (rotating EVENT_ID0/1 per
// slot; slot reuse is guarded by the (V,S) drain of the previous
// iteration, which is ordered before the next issue on the S pipe).
__aicore__ inline void VecNRunFullPhase(
    uint32_t ts, uint32_t Lt, uint32_t prefixEnd, uint32_t fullEnd, const SgbmvTilingData& tiling,
    GlobalTensor<float>& aGlobal, __ubuf__ const float* xPtr, uint32_t xAl, LocalTensor<float>& ubY,
    LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblLin,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf)
{
    // keep batch rows below the in-slot zero region used by the suffix
    // variant tables (entries point VEC_ZERO_SLOT elements into the src)
    const uint32_t JC = MaxVal(1U, MinVal(VEC_ABUF_ELEMS, VEC_ZERO_SLOT) / (Lt + 8U));
    const uint32_t srcGap = static_cast<uint32_t>(VecNFullGapBytes(Lt, tiling)); // fits: caller-side gate
    for (uint32_t rho = 0U; rho < SIMD_WIDTH; ++rho) {
        uint32_t jFirst = prefixEnd + ((rho + SIMD_WIDTH - (prefixEnd & (SIMD_WIDTH - 1U))) & (SIMD_WIDTH - 1U));
        if (jFirst >= fullEnd) {
            continue;
        }
        uint64_t aOff0 = static_cast<uint64_t>(tiling.lda) * jFirst + (tiling.ku + ts - jFirst);
        uint32_t r = static_cast<uint32_t>(aOff0 & 7ULL);
        uint32_t j = jFirst;
        uint32_t cp = 0U;
        uint32_t cnt0 = MinVal(JC, (fullEnd - j + SIMD_WIDTH - 1U) / SIMD_WIDTH);
        VecNIssueBatchDma(buf, cp, aGlobal, aOff0 - r, cnt0, Lt, srcGap);
        uint32_t nextJ = j + cnt0 * SIMD_WIDTH;
        bool hasNext = nextJ < fullEnd;
        while (hasNext) {
            uint32_t cnt1 = MinVal(JC, (fullEnd - nextJ + SIMD_WIDTH - 1U) / SIMD_WIDTH);
            VecNIssueBatchDma(buf, cp ^ 1U, aGlobal,
                aOff0 - r + static_cast<uint64_t>(nextJ - jFirst) * (tiling.lda - 1U), cnt1, Lt, srcGap);
            WaitMte2SlotReady(cp);
            VecNConsumeAndDrainBatch(ubY, ubT, ubA2, tblLin, buf, cp, j, cnt0, r, xPtr, xAl, Lt);
            j = nextJ;
            cnt0 = cnt1;
            cp ^= 1U;
            nextJ = j + cnt0 * SIMD_WIDTH;
            hasNext = nextJ < fullEnd;
        }
        WaitMte2SlotReady(cp);
        VecNConsumeAndDrainBatch(ubY, ubT, ubA2, tblLin, buf, cp, j, cnt0, r, xPtr, xAl, Lt);
    }
}

// Consume one staged prefix/suffix column (exact per-column DMA) and drain
// the V pipe before this slot may be overwritten by the DMA staged in the next
// iteration (load-bearing: measured corruption without it).
__aicore__ inline void VecNConsumeEdgeColumn(
    LocalTensor<float>& ubY, LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblAll,
    LocalTensor<uint32_t>& tblLin, LocalTensor<float>& slot, __ubuf__ const float* xPtr, uint32_t xAl, uint32_t jj,
    uint32_t ts, uint32_t te, const SgbmvTilingData& tiling, uint32_t Lt)
{
    float xj = xPtr[jj - xAl];
    if (xj != 0.0f) {
        uint32_t off = VecNColOff(jj, ts, tiling);
        uint32_t len = VecNColLen(jj, ts, te, tiling);
        uint64_t aOff = static_cast<uint64_t>(tiling.lda) * jj + (tiling.ku + ts + off - jj);
        uint32_t r = static_cast<uint32_t>(aOff & 7ULL);
        VecNColumnCompute(ubY, ubT, ubA2, tblLin, tblAll, slot, xj, r, off, len, Lt);
    }
    SET_FLAG(V, S, EVENT_ID0);
    WAIT_FLAG(V, S, EVENT_ID0);
}

// Prefix [cLo, prefixEnd) and suffix [suffixStart, cHi): per-column
// exact DMA with 1-deep issue-ahead over two staging slots.
__aicore__ inline void VecNRunEdgePhases(
    uint32_t cLo, uint32_t prefixEnd, uint32_t fullEnd, uint32_t suffixStart, uint32_t cHi, uint32_t ts, uint32_t te,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, LocalTensor<float>& abuf0,
    LocalTensor<float>& abuf1, __ubuf__ const float* xPtr, uint32_t xAl, LocalTensor<float>& ubY,
    LocalTensor<float>& ubT, LocalTensor<float>& ubA2, LocalTensor<uint32_t>& tblAll,
    LocalTensor<uint32_t>& tblLin, uint32_t Lt)
{
    for (uint32_t phase = 0U; phase < 3U; ++phase) {
        uint32_t jLo = (phase == 0U) ? cLo : ((phase == 1U) ? fullEnd : suffixStart);
        uint32_t jHi = (phase == 0U) ? prefixEnd : ((phase == 1U) ? suffixStart : cHi);
        if (jLo >= jHi) {
            continue;
        }
        uint32_t jj = jLo;
        uint32_t cp = 0U;
        VecNIssueColumn((cp == 0U) ? abuf0 : abuf1, aGlobal, jj, ts, te, tiling);
        SetMte2SlotReady(cp);
        while (jj < jHi) {
            uint32_t jn = jj + 1U;
            if (jn < jHi) {
                VecNIssueColumn((cp == 0U) ? abuf1 : abuf0, aGlobal, jn, ts, te, tiling);
                SetMte2SlotReady(cp ^ 1U);
            }
            WaitMte2SlotReady(cp);
            auto slot = (cp == 0U) ? abuf0 : abuf1;
            VecNConsumeEdgeColumn(ubY, ubT, ubA2, tblAll, tblLin, slot, xPtr, xAl, jj, ts, te, tiling, Lt);
            jj = jn;
            cp ^= 1U;
        }
    }
}

__aicore__ inline void VecNProcessTile(
    uint32_t ts, uint32_t te, const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal,
    GlobalTensor<float>& xGlobal, GlobalTensor<float>& yGlobal, AsdopsBuffer<ArchType::ASCEND_V220>& buf,
    bool& tablesBuilt, LocalTensor<uint32_t>& tblLin)
{
    const uint32_t Lt = te - ts; // multiple of 8, in [32, min(512, lda-1, band+1)]
    auto ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_Y);
    auto ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XWIN);
    auto ubT = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XS);
    auto ubA2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_A2);
    auto abuf0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF);
    auto abuf1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + VEC_ABUF_ELEMS * BYTES_PER_FLOAT);
    auto tblAll = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(VB_TBL);

    uint32_t cLo = 0U;
    uint32_t cHi = 0U;
    uint32_t suffixStart = 0U;
    uint32_t prefixEnd = 0U;
    VecNTileColumnRange(ts, te, tiling, cLo, cHi, suffixStart, prefixEnd);

    Duplicate(ubY, 0.0f, Lt + 8U);
    Duplicate(ubT[VEC_ZERO_SLOT], 0.0f, SIMD_WIDTH);
    // the suffix variant gather reads its zero slot from the A staging slot
    Duplicate(abuf0[VEC_ZERO_SLOT], 0.0f, SIMD_WIDTH);
    Duplicate(abuf1[VEC_ZERO_SLOT], 0.0f, SIMD_WIDTH);
    if (cLo >= cHi) {
        // No column intersects this subtile (possible when kl >= n): the
        // accumulator stays zero and only alpha/beta scaling applies.
        VecWriteBack(ts, Lt, tiling.alpha, tiling.beta, yGlobal, buf);
        return;
    }
    if (suffixStart < cHi && !tablesBuilt) {
        VecNBuildVariantTables(tblAll);
        tablesBuilt = true;
    }
    const uint32_t xAl = cLo & ~(SIMD_WIDTH - 1U); // aligned x window base
    VecCopyIn(ubX, xGlobal[xAl], cHi - xAl);
    SET_FLAG(MTE2, S, EVENT_ID0);
    WAIT_FLAG(MTE2, S, EVENT_ID0);
    PIPE_BARRIER(V);

    auto xPtr = reinterpret_cast<__ubuf__ const float*>(ubX.GetPhyAddr());

    // ---- full phase [prefixEnd, suffixStart): residue-class nburst batches.
    // aOff(j+1) - aOff(j) = lda-1, so columns with j%8 == rho share one
    // alignment residue r; one 2D DMA moves JC whole subtile columns.
    uint32_t fullEnd = suffixStart; // last full-phase column may over-read GM
    if (fullEnd == tiling.n && fullEnd > prefixEnd) {
        fullEnd -= 1U;
    }
    // Column strides wider than the uint32 srcStride field of the batch DMA
    // (lda beyond ~2^27) would wrap; run the full phase column-exact through
    // the per-column path below instead, whose GM offsets are all uint64.
    if (VecNFullGapBytes(Lt, tiling) > 0xFFFFFFFFULL) {
        fullEnd = prefixEnd;
    }
    if (fullEnd > prefixEnd) {
        VecNRunFullPhase(ts, Lt, prefixEnd, fullEnd, tiling, aGlobal, xPtr, xAl, ubY, ubT, ubA2, tblLin, buf);
    }

    // ---- prefix [cLo, prefixEnd) and suffix [suffixStart, cHi): per-column
    // exact DMA with 1-deep issue-ahead
    VecNRunEdgePhases(cLo, prefixEnd, fullEnd, suffixStart, cHi, ts, te, tiling, aGlobal, abuf0, abuf1, xPtr, xAl,
        ubY, ubT, ubA2, tblAll, tblLin, Lt);

    VecWriteBack(ts, Lt, tiling.alpha, tiling.beta, yGlobal, buf);
}

__aicore__ void ProcessSgbmvNVec(
    uint32_t startRow, uint32_t endRow, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.n);
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.n);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling.m);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    auto tblLin = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(VB_TBL_LIN);
    MakeOffsetSequence(tblLin.ReinterpretCast<int32_t>(), VEC_L_MAX + 16U, static_cast<int32_t>(BYTES_PER_FLOAT));
    const uint32_t band = tiling.kl + tiling.ku + 1U;
    bool tablesBuilt = false;
    uint32_t ts = startRow;
    while ((endRow - ts) >= VEC_ROW_MIN) {
        uint32_t avail = endRow - ts;
        uint32_t Lt = MinVal(MinVal(MinVal(avail, VEC_L_MAX), tiling.lda - 1U), band + 1U) & ~(SIMD_WIDTH - 1U);
        if (Lt < VEC_ROW_MIN) {
            break;
        }
        VecNProcessTile(ts, ts + Lt, tiling, aGlobal, xGlobal, yGlobal, buf, tablesBuilt, tblLin);
        ts += Lt;
    }
    if (ts < endRow) {
        ProcessSgbmvNFallback(ts, endRow, tiling, aGm, xGm, yGm);
    }
}

// ---- T vector path (all GM reads 32B-aligned) -------------------------------

__aicore__ inline uint32_t VecTEdgeRowStart(uint32_t j, const SgbmvTilingData& tiling)
{
    return (j >= tiling.ku) ? j - tiling.ku : 0U;
}

__aicore__ inline uint32_t VecTEdgeLen(uint32_t j, const SgbmvTilingData& tiling)
{
    uint32_t r0 = VecTEdgeRowStart(j, tiling);
    uint32_t r1 = MinVal(j + tiling.kl + 1U, tiling.m);
    return (r1 > r0) ? (r1 - r0) : 0U;
}

__aicore__ inline uint32_t VecTNextValid(uint32_t from, uint32_t endCol, const SgbmvTilingData& tiling)
{
    while (from < endCol && VecTEdgeLen(from, tiling) == 0U) {
        ++from;
    }
    return from;
}

// Stage one column segment from an aligned GM base: [aOff-r, aOff+len).
__aicore__ inline void VecTIssueColumn(
    LocalTensor<float> slot, GlobalTensor<float>& aGlobal, uint32_t j, const SgbmvTilingData& tiling)
{
    uint32_t r0 = VecTEdgeRowStart(j, tiling);
    uint32_t len = VecTEdgeLen(j, tiling);
    uint64_t aOff = static_cast<uint64_t>(tiling.lda) * j + (tiling.ku + r0 - j);
    uint32_t r = static_cast<uint32_t>(aOff & 7ULL);
    VecCopyIn(slot, aGlobal[aOff - r], len + r);
}

// Stage one column segment from its exact (possibly non-32B-aligned) GM base:
// data lands at slot[0..len) with a DMA zero tail to 8-alignment, so the
// per-column compute needs no A-side alignment gather (non-32B GM src bases
// are accepted; values bit-identical to the aligned-base + gather form).
__aicore__ inline void VecTIssueColumnAligned(
    LocalTensor<float> slot, GlobalTensor<float>& aGlobal, uint32_t j, const SgbmvTilingData& tiling)
{
    uint32_t r0 = VecTEdgeRowStart(j, tiling);
    uint32_t len = VecTEdgeLen(j, tiling);
    uint64_t aOff = static_cast<uint64_t>(tiling.lda) * j + (tiling.ku + r0 - j);
    uint32_t lenBurst = len * BYTES_PER_FLOAT;
    uint8_t rightPad = static_cast<uint8_t>((SIMD_WIDTH - (len % SIMD_WIDTH)) % SIMD_WIDTH);
    AscendC::DataCopyPad(slot, aGlobal[aOff],
        AscendC::DataCopyExtParams(1, static_cast<int32_t>(lenBurst), 0, 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, rightPad, 0.0f));
}

// Next staging unit at or after column j. Compact storage:
//   - interior columns (j >= ku && j < m - kl, full band length) form one
//     contiguous GM run -> one multi-block DMA covering up to batchK columns
//     (utype 2);
//   - consecutive edge columns group into one slot of per-column aligned
//     DMAs sharing a single MTE2->V flag (utype 1), sub-slices at band8
//     stride.
// Non-compact storage keeps the original single-column units (utype 0) with
// the original issue/compute path, byte-identical to the per-column behavior.
constexpr uint32_t VEC_T_UNIT_SINGLE = 0U;
constexpr uint32_t VEC_T_UNIT_GROUP = 1U;
constexpr uint32_t VEC_T_UNIT_BATCH = 2U;

__aicore__ inline bool VecTNextUnit(uint32_t j, uint32_t ce, uint32_t iLo, uint32_t iHi, uint32_t batchK,
    uint32_t groupK, const SgbmvTilingData& tiling, uint32_t& uj, uint32_t& uc, uint32_t& ut)
{
    uint32_t v = VecTNextValid(j, ce, tiling);
    if (v >= ce) {
        return false;
    }
    uj = v;
    if (batchK >= 2U && v >= iLo && v < iHi) {
        uc = MinVal(batchK, iHi - v);
        ut = VEC_T_UNIT_BATCH;
        return true;
    }
    if (groupK >= 2U) {
        uint32_t limit = (v < iLo) ? iLo : ce;
        uint32_t cnt = 1U;
        while (cnt < groupK && v + cnt < limit && VecTEdgeLen(v + cnt, tiling) > 0U) {
            ++cnt;
        }
        if (cnt >= 2U) {
            uc = cnt;
            ut = VEC_T_UNIT_GROUP;
            return true;
        }
    }
    uc = 1U;
    ut = VEC_T_UNIT_SINGLE;
    return true;
}

// Issue one unit into its ring slot.
__aicore__ inline void VecTIssueUnit(LocalTensor<float> slot, GlobalTensor<float>& aGlobal, uint32_t uj, uint32_t uc,
    uint32_t ut, const SgbmvTilingData& tiling, uint32_t bandT, uint32_t band8T)
{
    if (ut == VEC_T_UNIT_SINGLE) {
        VecTIssueColumn(slot, aGlobal, uj, tiling);
        return;
    }
    if (ut == VEC_T_UNIT_GROUP) {
        for (uint32_t l = 0U; l < uc; ++l) {
            VecTIssueColumnAligned(slot[l * band8T], aGlobal, uj + l, tiling);
        }
        return;
    }
    uint8_t pad8 = static_cast<uint8_t>(band8T - bandT);
    AscendC::DataCopyPad(slot, aGlobal[static_cast<uint64_t>(bandT) * uj],
        AscendC::DataCopyExtParams(static_cast<int32_t>(uc), static_cast<int32_t>(bandT * BYTES_PER_FLOAT), 0, 0, 0),
        AscendC::DataCopyPadExtParams<float>(false, 0, pad8, 0.0f));
}

// One column dot product: A shifted back into alignment via gather base
// srcOff*4, x shifted by o elements via the same linear table, multiply,
// pairwise-halve down to 8 lanes, store into the [L, 8] partial matrix.
// len8/p2 version: the gathers and Mul run on ceil8(len) lanes
// (not the tile-wide pow2 rowP), ubP is zero-padded up to p2 = pow2(len8), and
// the halving starts at p2 — edge columns no longer pay the full-band cost.
__aicore__ inline void VecTColumnCompute(
    LocalTensor<float> ubX, LocalTensor<float> ubXs, LocalTensor<float> ubA2, LocalTensor<float> ubP,
    LocalTensor<float> ubYP, LocalTensor<uint32_t> tblLin, LocalTensor<float> slot, uint32_t srcOff, uint32_t o,
    uint32_t len, bool padSlot, uint32_t colIdx)
{
    uint32_t len8 = (len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    uint32_t p2 = VecPow2RoundUp(len8);
    if (padSlot) {
        // per-column staging holds len + srcOff floats zero-padded to lenr8;
        // the gather may read up to 8 floats past that.
        uint32_t lenr8 = (len + srcOff + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
        if (srcOff + len8 > lenr8) {
            Duplicate(slot[lenr8], 0.0f, SIMD_WIDTH);
        }
    }
    Gather(ubA2, slot, tblLin, srcOff * BYTES_PER_FLOAT, len8);
    Gather(ubXs, ubX, tblLin, o * BYTES_PER_FLOAT, len8);
    PIPE_BARRIER(V);
    Mul(ubP, ubA2, ubXs, len8);
    if (p2 > len8) {
        Duplicate(ubP[len8], 0.0f, p2 - len8);
    }
    PIPE_BARRIER(V);
    uint32_t ln = p2;
    while (ln > SIMD_WIDTH) {
        Add(ubP, ubP, ubP[ln / 2U], ln / 2U);
        PIPE_BARRIER(V);
        ln /= 2U;
    }
    Adds(ubYP[colIdx * SIMD_WIDTH], ubP, 0.0f, SIMD_WIDTH);
    PIPE_BARRIER(V);
}

// Batched-column variant: the multi-block staging lands each interior
// column at an 8-aligned slot offset with data at [0, band) and a DMA zero
// tail, so the A-side alignment Gather is unnecessary — the slot slice is
// passed to Mul directly. Only the x-shift Gather remains (one gather per
// column instead of two). The last halving level folds straight into the
// partial-matrix row (one Add replaces an Add + copy Adds).
__aicore__ inline void VecTColumnComputeAligned(
    LocalTensor<float> ubX, LocalTensor<float> ubXs, LocalTensor<float> ubP, LocalTensor<float> ubYP,
    LocalTensor<uint32_t> tblLin, LocalTensor<float> slot, uint32_t o, uint32_t len, uint32_t colIdx)
{
    uint32_t len8 = (len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    uint32_t p2 = VecPow2RoundUp(len8);
    Gather(ubXs, ubX, tblLin, o * BYTES_PER_FLOAT, len8);
    PIPE_BARRIER(V);
    Mul(ubP, slot, ubXs, len8);
    if (p2 > len8) {
        Duplicate(ubP[len8], 0.0f, p2 - len8);
    }
    PIPE_BARRIER(V);
    uint32_t ln = p2;
    while (ln > 2U * SIMD_WIDTH) {
        Add(ubP, ubP, ubP[ln / 2U], ln / 2U);
        PIPE_BARRIER(V);
        ln /= 2U;
    }
    if (ln > SIMD_WIDTH) {
        Add(ubYP[colIdx * SIMD_WIDTH], ubP, ubP[SIMD_WIDTH], SIMD_WIDTH);
    } else {
        Adds(ubYP[colIdx * SIMD_WIDTH], ubP, 0.0f, SIMD_WIDTH);
    }
    PIPE_BARRIER(V);
}

// Two columns with shared barriers: the independent per-column vector ops
// (gathers, muls, pads, halving steps) are issued back-to-back with a single
// PIPE_BARRIER between dependency levels, halving the per-column barrier count
// (the remaining TVec cases are barrier-bound after the 8-slot ring removed
// the DMA-latency stalls). Scratch: ubA2/ubXs/ubP regions each split in half;
// requires p2 <= VEC_ROWP_MAX/2 for both columns (band <= 2040).
struct VecTColParams {
    uint32_t srcOff;
    uint32_t o;
    uint32_t len;
    bool padSlot;
    uint32_t colIdx;
};

__aicore__ inline void VecTColumnComputePair(LocalTensor<float> ubX, LocalTensor<uint32_t> tblLin,
    LocalTensor<float> ubYP, LocalTensor<float>& slot0, LocalTensor<float>& slot1, const VecTColParams& p0,
    const VecTColParams& p1, LocalTensor<float> a20, LocalTensor<float> a21, LocalTensor<float> xs0,
    LocalTensor<float> xs1, LocalTensor<float> p0t, LocalTensor<float> p1t)
{
    uint32_t len8a = (p0.len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    uint32_t p2a = VecPow2RoundUp(len8a);
    uint32_t len8b = (p1.len + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    uint32_t p2b = VecPow2RoundUp(len8b);
    if (p0.padSlot) {
        uint32_t lenr8a = (p0.len + p0.srcOff + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
        if (p0.srcOff + len8a > lenr8a) {
            Duplicate(slot0[lenr8a], 0.0f, SIMD_WIDTH);
        }
    }
    if (p1.padSlot) {
        uint32_t lenr8b = (p1.len + p1.srcOff + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
        if (p1.srcOff + len8b > lenr8b) {
            Duplicate(slot1[lenr8b], 0.0f, SIMD_WIDTH);
        }
    }
    Gather(a20, slot0, tblLin, p0.srcOff * BYTES_PER_FLOAT, len8a);
    Gather(a21, slot1, tblLin, p1.srcOff * BYTES_PER_FLOAT, len8b);
    Gather(xs0, ubX, tblLin, p0.o * BYTES_PER_FLOAT, len8a);
    Gather(xs1, ubX, tblLin, p1.o * BYTES_PER_FLOAT, len8b);
    PIPE_BARRIER(V);
    Mul(p0t, a20, xs0, len8a);
    Mul(p1t, a21, xs1, len8b);
    if (p2a > len8a) {
        Duplicate(p0t[len8a], 0.0f, p2a - len8a);
    }
    if (p2b > len8b) {
        Duplicate(p1t[len8b], 0.0f, p2b - len8b);
    }
    PIPE_BARRIER(V);
    uint32_t lnA = p2a;
    uint32_t lnB = p2b;
    while (lnA > SIMD_WIDTH || lnB > SIMD_WIDTH) {
        if (lnA > SIMD_WIDTH) {
            Add(p0t, p0t, p0t[lnA / 2U], lnA / 2U);
            lnA /= 2U;
        }
        if (lnB > SIMD_WIDTH) {
            Add(p1t, p1t, p1t[lnB / 2U], lnB / 2U);
            lnB /= 2U;
        }
        PIPE_BARRIER(V);
    }
    Adds(ubYP[p0.colIdx * SIMD_WIDTH], p0t, 0.0f, SIMD_WIDTH);
    Adds(ubYP[p1.colIdx * SIMD_WIDTH], p1t, 0.0f, SIMD_WIDTH);
    PIPE_BARRIER(V);
}

__aicore__ inline void VecTStageTileWindow(
    LocalTensor<float>& ubX, GlobalTensor<float>& xGlobal, uint32_t cs, uint32_t ce,
    const SgbmvTilingData& tiling, uint32_t& xAl)
{
    const uint32_t xLo = (cs >= tiling.ku) ? cs - tiling.ku : 0U;
    const uint32_t xHi = MinVal(tiling.m, ce + tiling.kl);
    xAl = xLo & ~(SIMD_WIDTH - 1U); // aligned x window base
    const uint32_t width = (xHi > xAl) ? (xHi - xAl) : 0U;
    const uint32_t width8 = (width + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;

    if (width > 0U) {
        VecCopyIn(ubX, xGlobal[xAl], width);
    }
    // zero slack: gathers read up to o + rowP elements; beyond the window
    // (and its 8-alignment pad) the values must be exact zeros.
    Duplicate(ubX[width8], 0.0f, VEC_XWIN_ELEMS - width8);
    PIPE_BARRIER(V);
}

// Ring-slot pipeline configuration for one T vector tile.
struct VecTPipelineConfig {
    uint32_t slots;
    uint32_t slotElems;
    uint32_t batchK;
    uint32_t groupK;
    uint32_t iLo;
    uint32_t iHi;
};

// 8-deep column pipeline: hides ~0.4-0.5us of unhidden MTE2 latency per
// column on the wide-band cases (aiv busy only ~53% of task duration
// without it). The two staging slots become a ring of up
// to 8 slots (event IDs 0..7, one per slot); slot reuse stays guarded by
// the (V,S) drain before the refill DMA.
// Unit staging: see VecTIssueUnit / VecTNextUnit. One MTE2->V wait
// per unit (batch of interior columns / group of edge columns) instead of
// one per column; edge columns also drop the A-side alignment gather.
__aicore__ inline void VecTInitPipelineConfig(
    uint32_t cs, uint32_t ce, const SgbmvTilingData& tiling, uint32_t& bandT, uint32_t& band8T,
    VecTPipelineConfig& cfg)
{
    bandT = tiling.kl + tiling.ku + 1U;
    cfg.slots = (VEC_ABUF_ELEMS * 2U) / (bandT + 32U);
    if (cfg.slots > 8U) {
        cfg.slots = 8U;
    }
    if (cfg.slots < 2U) {
        cfg.slots = 2U;
    }
    cfg.slotElems = ((VEC_ABUF_ELEMS * 2U) / cfg.slots) & ~(SIMD_WIDTH - 1U);
    band8T = (bandT + SIMD_WIDTH - 1U) & ~(SIMD_WIDTH - 1U);
    // bandT = kl+ku+1 >= 1 for any real band; only an unsigned wraparound of
    // kl+ku can zero it, so clamp the aligned stride to one SIMD group.
    if (band8T == 0U) {
        band8T = SIMD_WIDTH;
    }
    cfg.batchK = 0U;
    cfg.groupK = 0U;
    if (tiling.lda == bandT) {
        cfg.batchK = cfg.slotElems / band8T;
        cfg.groupK = cfg.batchK;
        if (cfg.batchK < 2U) {
            cfg.batchK = 0U;
        }
        if (cfg.groupK < 2U) {
            cfg.groupK = 0U;
        }
    }
    cfg.iLo = (cs >= tiling.ku) ? cs : tiling.ku;
    cfg.iHi = (tiling.m > tiling.kl) ? MinVal(ce, tiling.m - tiling.kl) : cs;
}

// Prefill the slot ring: issue the first units and record their descriptors.
__aicore__ inline uint32_t VecTPrefillRing(
    uint32_t cs, uint32_t ce, const VecTPipelineConfig& cfg, uint32_t bandT, uint32_t band8T,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, AsdopsBuffer<ArchType::ASCEND_V220>& buf,
    uint32_t jstart[8], uint32_t jcount[8], uint32_t jtype[8], uint32_t& un, uint32_t& unc, uint32_t& unt,
    bool& haveNext)
{
    uint32_t live = 0U;
    haveNext = VecTNextUnit(cs, ce, cfg.iLo, cfg.iHi, cfg.batchK, cfg.groupK, tiling, un, unc, unt);
    for (uint32_t g = 0U; g < cfg.slots && haveNext; ++g) {
        jstart[g] = un;
        jcount[g] = unc;
        jtype[g] = unt;
        VecTIssueUnit(
            buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + g * cfg.slotElems * BYTES_PER_FLOAT), aGlobal, un,
            unc, unt, tiling, bandT, band8T);
        SET_FLAG(MTE2, V, g);
        ++live;
        haveNext = VecTNextUnit(un + unc, ce, cfg.iLo, cfg.iHi, cfg.batchK, cfg.groupK, tiling, un, unc, unt);
    }
    return live;
}

// Consume one single-column unit slot (optionally paired with the next slot:
// two columns with shared barriers).
__aicore__ inline void VecTConsumeSingleSlot(
    LocalTensor<float>& ubX, LocalTensor<float>& ubXs, LocalTensor<float>& ubA2, LocalTensor<float>& ubP,
    LocalTensor<float>& ubYP, LocalTensor<uint32_t>& tblLin, LocalTensor<float>& a20, LocalTensor<float>& a21,
    LocalTensor<float>& xs0, LocalTensor<float>& xs1, LocalTensor<float>& p0t, LocalTensor<float>& p1t,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, uint32_t head, uint32_t second, bool paired, const uint32_t jstart[8],
    uint32_t xAl, uint32_t cs, uint32_t slotElems, const SgbmvTilingData& tiling)
{
    auto slot0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + head * slotElems * BYTES_PER_FLOAT);
    uint32_t jj0 = jstart[head];
    uint32_t len0 = VecTEdgeLen(jj0, tiling);
    uint32_t r00 = VecTEdgeRowStart(jj0, tiling);
    uint64_t aOff0 = static_cast<uint64_t>(tiling.lda) * jj0 + (tiling.ku + r00 - jj0);
    uint32_t r0 = static_cast<uint32_t>(aOff0 & 7ULL);
    if (paired) {
        auto slot1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + second * slotElems * BYTES_PER_FLOAT);
        uint32_t jj1 = jstart[second];
        uint32_t len1 = VecTEdgeLen(jj1, tiling);
        uint32_t r01 = VecTEdgeRowStart(jj1, tiling);
        uint64_t aOff1 = static_cast<uint64_t>(tiling.lda) * jj1 + (tiling.ku + r01 - jj1);
        uint32_t r1 = static_cast<uint32_t>(aOff1 & 7ULL);
        VecTColParams pa;
        pa.srcOff = r0;
        pa.o = r00 - xAl;
        pa.len = len0;
        pa.padSlot = true;
        pa.colIdx = jj0 - cs;
        VecTColParams pb;
        pb.srcOff = r1;
        pb.o = r01 - xAl;
        pb.len = len1;
        pb.padSlot = true;
        pb.colIdx = jj1 - cs;
        VecTColumnComputePair(ubX, tblLin, ubYP, slot0, slot1, pa, pb, a20, a21, xs0, xs1, p0t, p1t);
    } else {
        VecTColumnCompute(ubX, ubXs, ubA2, ubP, ubYP, tblLin, slot0, r0, r00 - xAl, len0, true, jj0 - cs);
    }
}

// multi-column unit (batch/group): columns share one staged slot;
// each column's segment sits at an 8-aligned (jj - ustart) * band8T
// offset with a DMA zero tail, so the per-column compute is the
// gather-free aligned form with srcOff = 0.
__aicore__ inline void VecTConsumeMultiColumnUnit(
    LocalTensor<float>& ubX, LocalTensor<float>& ubXs, LocalTensor<float>& ubP, LocalTensor<float>& ubYP,
    LocalTensor<uint32_t>& tblLin, AsdopsBuffer<ArchType::ASCEND_V220>& buf, uint32_t head, const uint32_t jstart[8],
    const uint32_t jcount[8], const uint32_t jtype[8], uint32_t xAl, uint32_t cs, uint32_t slotElems, uint32_t bandT,
    uint32_t band8T, const SgbmvTilingData& tiling)
{
    uint32_t u0 = jstart[head];
    for (uint32_t jj = u0; jj < u0 + jcount[head]; ++jj) {
        auto colSlot = buf.GetBuffer<BufferType::ASCEND_UB, float>(
            VB_ABUF + (head * slotElems + (jj - u0) * band8T) * BYTES_PER_FLOAT);
        uint32_t r0j = VecTEdgeRowStart(jj, tiling);
        uint32_t lenj = (jtype[head] == VEC_T_UNIT_BATCH) ? bandT : VecTEdgeLen(jj, tiling);
        VecTColumnComputeAligned(ubX, ubXs, ubP, ubYP, tblLin, colSlot, r0j - xAl, lenj, jj - cs);
    }
}

// Refill the freed slot(s): issue the next unit into each and record its
// descriptor, or retire the slot when the column range is exhausted.
__aicore__ inline void VecTRefillSlots(
    uint32_t head, uint32_t second, uint32_t nSlots, uint32_t ce, const VecTPipelineConfig& cfg, uint32_t bandT,
    uint32_t band8T, const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal,
    AsdopsBuffer<ArchType::ASCEND_V220>& buf, uint32_t jstart[8], uint32_t jcount[8], uint32_t jtype[8],
    uint32_t& un, uint32_t& unc, uint32_t& unt, bool& haveNext, uint32_t& live)
{
    for (uint32_t s = 0U; s < nSlots; ++s) {
        uint32_t idx = (s == 0U) ? head : second;
        auto slot = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_ABUF + idx * cfg.slotElems * BYTES_PER_FLOAT);
        if (haveNext) {
            jstart[idx] = un;
            jcount[idx] = unc;
            jtype[idx] = unt;
            VecTIssueUnit(slot, aGlobal, un, unc, unt, tiling, bandT, band8T);
            SET_FLAG(MTE2, V, idx);
            haveNext = VecTNextUnit(un + unc, ce, cfg.iLo, cfg.iHi, cfg.batchK, cfg.groupK, tiling, un, unc, unt);
        } else {
            --live;
        }
    }
}

// Consume the ring head until every unit is computed: wait the head slot's
// MTE2->V event (two slots at once when pairing), compute, drain the V pipe,
// then refill the freed slot(s).
__aicore__ inline void VecTRunColumnPipeline(
    uint32_t cs, uint32_t ce, uint32_t xAl, const VecTPipelineConfig& cfg, uint32_t bandT, uint32_t band8T,
    const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal, AsdopsBuffer<ArchType::ASCEND_V220>& buf,
    uint32_t jstart[8], uint32_t jcount[8], uint32_t jtype[8], uint32_t& un, uint32_t& unc, uint32_t& unt,
    bool& haveNext, uint32_t& live, LocalTensor<float>& ubX, LocalTensor<float>& ubXs, LocalTensor<float>& ubA2,
    LocalTensor<float>& ubP, LocalTensor<float>& ubYP, LocalTensor<uint32_t>& tblLin)
{
    uint32_t head = 0U;
    const bool canPair = cfg.slots >= 2U && bandT <= 2048U;
    auto a20 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_A2);
    auto a21 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_A2 + (VEC_ROWP_MAX / 2U) * BYTES_PER_FLOAT);
    auto xs0 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XS);
    auto xs1 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XS + (VEC_ROWP_MAX / 2U) * BYTES_PER_FLOAT);
    auto p0t = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_P);
    auto p1t = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_P + (VEC_ROWP_MAX / 2U) * BYTES_PER_FLOAT);
    while (live > 0U) {
        uint32_t second = (head + 1U == cfg.slots) ? 0U : head + 1U;
        bool paired =
            canPair && live >= 2U && jtype[head] == VEC_T_UNIT_SINGLE && jtype[second] == VEC_T_UNIT_SINGLE;
        if (paired) {
            WAIT_FLAG(MTE2, V, head);
            WAIT_FLAG(MTE2, V, second);
        } else {
            WAIT_FLAG(MTE2, V, head);
        }
        if (jtype[head] == VEC_T_UNIT_SINGLE) {
            VecTConsumeSingleSlot(ubX, ubXs, ubA2, ubP, ubYP, tblLin, a20, a21, xs0, xs1, p0t, p1t, buf, head,
                second, paired, jstart, xAl, cs, cfg.slotElems, tiling);
        } else {
            VecTConsumeMultiColumnUnit(
                ubX, ubXs, ubP, ubYP, tblLin, buf, head, jstart, jcount, jtype, xAl, cs, cfg.slotElems, bandT, band8T,
                tiling);
        }
        // drain the V pipe before these slots may be overwritten by refill DMAs
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        uint32_t nSlots = paired ? 2U : 1U;
        VecTRefillSlots(head, second, nSlots, ce, cfg, bandT, band8T, tiling, aGlobal, buf, jstart, jcount, jtype,
            un, unc, unt, haveNext, live);
        head = (nSlots == 2U) ? ((second + 1U == cfg.slots) ? 0U : second + 1U)
                              : ((head + 1U == cfg.slots) ? 0U : head + 1U);
    }
}

// fold the [L, 8] partial matrix into L outputs (shipped reduce pattern)
__aicore__ inline void VecTFoldPartialMatrix(LocalTensor<float>& ubY, LocalTensor<float>& ubYP, uint32_t L)
{
    uint32_t fullRows = L / SIMD_WIDTH * SIMD_WIDTH;
    if (fullRows != 0U) {
        ReduceDataBlock<ReduceType::SUM>(ubY, ubYP, 64U, fullRows / SIMD_WIDTH, 1, 1, 8);
    }
    if (L > fullRows) {
        ReduceDataBlock<ReduceType::SUM>(
            ubY[fullRows], ubYP[fullRows * SIMD_WIDTH], (L - fullRows) * SIMD_WIDTH, 1, 1, 1, 8);
    }
    PIPE_BARRIER(V);
}

__aicore__ inline void VecTProcessTile(
    uint32_t cs, uint32_t ce, const SgbmvTilingData& tiling, GlobalTensor<float>& aGlobal,
    GlobalTensor<float>& xGlobal, GlobalTensor<float>& yGlobal, AsdopsBuffer<ArchType::ASCEND_V220>& buf,
    LocalTensor<uint32_t>& tblLin)
{
    const uint32_t L = ce - cs;
    auto ubY = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_Y);
    auto ubYP = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_YP);
    auto ubX = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XWIN);
    auto ubXs = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_XS);
    auto ubA2 = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_A2);
    auto ubP = buf.GetBuffer<BufferType::ASCEND_UB, float>(VB_P);

    Duplicate(ubYP, 0.0f, L * SIMD_WIDTH);
    uint32_t xAl = 0U;
    VecTStageTileWindow(ubX, xGlobal, cs, ce, tiling, xAl);

    uint32_t bandT = 0U;
    uint32_t band8T = 0U;
    VecTPipelineConfig cfg;
    VecTInitPipelineConfig(cs, ce, tiling, bandT, band8T, cfg);

    uint32_t jstart[8];
    uint32_t jcount[8];
    uint32_t jtype[8];
    uint32_t un = 0U;
    uint32_t unc = 0U;
    uint32_t unt = 0U;
    bool haveNext = false;
    uint32_t live = VecTPrefillRing(
        cs, ce, cfg, bandT, band8T, tiling, aGlobal, buf, jstart, jcount, jtype, un, unc, unt, haveNext);
    VecTRunColumnPipeline(cs, ce, xAl, cfg, bandT, band8T, tiling, aGlobal, buf, jstart, jcount, jtype, un, unc, unt,
        haveNext, live, ubX, ubXs, ubA2, ubP, ubYP, tblLin);

    VecTFoldPartialMatrix(ubY, ubYP, L);

    VecWriteBack(cs, L, tiling.alpha, tiling.beta, yGlobal, buf);
}

__aicore__ void ProcessSgbmvTVec(
    uint32_t startCol, uint32_t endCol, const SgbmvTilingData& tiling, GM_ADDR a, GM_ADDR x, GM_ADDR y)
{
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.n);
    xGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tiling.m);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), tiling.n);

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    const uint32_t band = tiling.kl + tiling.ku + 1U;
    const uint32_t band8 = (band + SIMD_WIDTH - 1U) / SIMD_WIDTH * SIMD_WIDTH;
    const uint32_t rowP = VecPow2RoundUp(band8);
    auto tblLin = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(VB_TBL);
    MakeOffsetSequence(tblLin.ReinterpretCast<int32_t>(), rowP, static_cast<int32_t>(BYTES_PER_FLOAT));
    for (uint32_t cs = startCol; cs < endCol; cs += VEC_T_L_MAX) {
        uint32_t ce = MinVal(cs + VEC_T_L_MAX, endCol);
        VecTProcessTile(cs, ce, tiling, aGlobal, xGlobal, yGlobal, buf, tblLin);
    }
}
