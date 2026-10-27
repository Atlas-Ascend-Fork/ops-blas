/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and
 * conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// R18.1 isolated translation unit of the R18 largeMode column-ownership tile path.
// The R18 fused build kept this path inside sspr_kernel.cpp's shared SsprKernel; the 6 large
// __aicore__ inline methods + the column-boundary binary search got inlined into the shared
// Process()/Init(), and the merged mega-function's register allocation / frame layout /
// code layout changed the sspr_kernel codegen enough to tax the 2us micro shapes (n=1
// +4.3%, n<=64 +2-7.6%). Moving the whole path here restores sspr_kernel.cpp to the pre-R18
// head (004e506) byte-for-byte, so the degradation is removed BY CONSTRUCTION, while the
// largeMode semantics are untouched (statement-by-statement move, no algorithm edits).
//
// R19 changes (all under the bitwise mandate - the per-element arithmetic statement order of
// the exact Dekker chain + Phase D guard is the R18.1 sequence):
//   1. LOWER enabled: the geometry (rowBegin/rowEnd by uplo, ColStart, partial head) was
//      already general from R18 - the host gate (SsprLargeGateEligible) was UPPER-only.
//      R19 opens the gate for both fill modes (threshold 1300 unchanged). UPPER behavior is
//      UNCHANGED: its columns always start at the tile base (offset 0, 64-aligned), so
//      UPPER never takes the head path and its segmenting is identical to R18/R18.1.
//   2. Head alignment granularity 64 (was segLen): a column's first segment inside the tile
//      starts at an ARBITRARY row, whose xTile slice start is not 32B aligned (V ops at
//      unaligned UB offsets are broken on this silicon). R18 re-fetched the head factors
//      from GM (DataCopyPad into xSeg, the !333 partial-head maneuver) and aligned heads up
//      to the next segLen boundary: for LOWER that was avg segLen/2 = 1024 elements of GM
//      re-fetch + 15-op chain per column - half the triangle's elements through the head
//      path. R19 rounds the head up to the next 64-element boundary (avg 32, <= 63
//      elements) and GATHERS just those few factors from the resident xTile (the walk
//      path's unaligned-run idiom: ArithProgression byte-offset idx + element-level
//      Gather, no alignment constraint). A direct GM DataCopyPad re-fetch was ~100ns/head
//      faster but produced TORN, NON-DETERMINISTIC head lanes (the adjacent MTE2_V pair
//      does not reliably order a directly-issued GM copy against an immediately following
//      V consumption at that issue pacing) and was REJECTED on correctness; the Gather
//      variant is run-to-run deterministic (P0 repeat runs bitwise-identical).
//      Aligned segments start at 64-multiple offsets (256B-aligned - the walk path issues
//      V ops on 8-element-granular slice offsets, so 64 covers it with margin); segLen
//      stays a 64-multiple so in-tile segment offsets stay 64-aligned after the head.
//   3. REJECTED (kept as the R20 candidate, see eval_r19/): the th/tl Dekker-split hoist
//      (computing the 4 split ops once per tile instead of once per segment per column,
//      -4 of 15 vector ops per element, -17% on the exact band at 4021U/4096L). It required
//      2 extra tile-sized UB regions, which forced T 16384 -> 8192 and a SECOND tile load
//      for n in (8192, 16384] - and that 2-tile configuration showed a rare intermittent
//      nondeterminism (1-2ulp guard-fallback lanes at 16384U, ~50% of 40-rep P0 runs, and
//      one in-suite TC_LG_008 blip). The single-tile R18.1 skeleton (T = 16384) has never
//      shown it in the whole R18/R18.1 campaign, so R19 ships the proven skeleton and
//      gives up the hoist's -17% pending a root cause of the 2-tile divergence.
//
// Entry contract: sspr_kernel_do (sspr_kernel.cpp) routes tiling.largeMode == 1 here and
// never lets the template kernels see a largeMode tiling (they receive it 恒 0). The
// in-kernel `tiling.largeMode != 1U` guard is a second line of defence.

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "sspr_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t SSPR_IN_BUF_NUM = 2;   // apIn double-buffered queue, depth 2
constexpr uint32_t SSPR_OUT_BUF_NUM = 2;  // apOut double buffering
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t SSPR_ALIGN_ELEMS = 8;  // 32B alignment granularity in float32 elements
// Head/segment alignment granularity in float32 elements. Aligned segments start at
// HEAD_ALIGN-multiple in-tile offsets (256B-aligned UB slice bases - the walk path proves
// 8-element granularity suffices for V ops, 64 keeps margin); a column head is the
// sub-granularity prefix that must Gather its row factors into xSeg.
constexpr uint32_t SSPR_LARGE_HEAD_ALIGN_ELEMS = 64;
constexpr float SPLIT_CONST = 4097.0f;     // Dekker splitter 2^12+1 for f32 (exact path, n > SSPR_NAIVE_MAX_N)
constexpr float FLT_MAX_F = 3.4028234663852886e+38f;
// n above this uses the Dekker exact path; n <= this uses the 2-op naive path.
// Keep in sync with sspr_kernel.cpp (SSPR_NAIVE_MAX_N) and the host SSPR_DESC_MAX_N
// (sspr_host.cpp).
constexpr uint32_t SSPR_NAIVE_MAX_N = 2048;
// R19 head scratch: a head is < SSPR_LARGE_HEAD_ALIGN_ELEMS elements; +8 covers the house
// padded copy-in corner (rightPadding rounds the fetch up one extra 32B block).
constexpr uint32_t SSPR_LARGE_HEAD_FLOATS = SSPR_LARGE_HEAD_ALIGN_ELEMS + SSPR_ALIGN_ELEMS;

// ── R18 largeMode inter-core column split, ported verbatim from the !333 UnifiedContiguousColumns
// cost model (its kernel kChunkFloats/kCostPer* block). A core's time is not proportional to the
// number of packed elements it owns: owning k columns costs roughly
//     sum over the k columns of len  +  k*p  +  q*(extra transfers)
// where p is a flat charge every column pays (AP base address, scalar broadcast of x[col], loop
// and descriptor setup, first DMA round trip) and q is charged again for each further
// kLargeChunkFloats-sized piece a long column has to walk. Equalising element counts ignores
// both (at n = 32768 the slowest core took 2x the mean under the equal-column split).
// Implemented below as kLargeCostPerColumn * columns + kLargeCostPerElement * elements +
// kLargeCostPerTransfer * transfers (every column contributes at least one transfer, so only the
// SUM kLargeCostPerColumn + kLargeCostPerTransfer is identifiable for len <= chunk - see the
// original calibration notes; p sits in the 4096-8192 range). Selected by timing 44 (n, uplo,
// incx) configurations over a candidate grid on 910B3 plus a LOWER-only sweep (25 coefficient
// pairs): LOWER keeps the same coefficients. Each term has an O(1) closed form so the boundary
// search stays O(log n).
// NOTE (R18): these coefficients were calibrated under 2-16 parts on !333; the 40-parts rescan
// is a follow-up item (env-independent constants here).
constexpr uint64_t kLargeChunkFloats = 8192;
constexpr uint64_t kLargeCostPerColumn = 2048;
constexpr uint64_t kLargeCostPerElement = 1;
constexpr uint64_t kLargeCostPerTransfer = 4096;

// sum_{len=1..m} ceil(len / grain)
__aicore__ inline uint64_t LargeCeilRunSum(uint64_t m, uint64_t grain)
{
    if (m == 0ULL) {
        return 0ULL;
    }
    if (grain == 0ULL) {
        return 0ULL;  // unreachable at runtime (grain == kLargeChunkFloats, constexpr);
                      // guard lives in this function's entry scope for the div-by-zero scan.
    }
    const uint64_t q = m / grain;
    const uint64_t r = m - q * grain;
    return grain * q * (q + 1ULL) / 2ULL + r * (q + 1ULL);
}

__aicore__ inline uint64_t LargeColumnWorkPrefix(uint32_t columns, uint32_t n, uint32_t uplo)
{
    // Column c has length c+1 (UPPER) or n-c (LOWER); the sums below run over [0, columns).
    const uint64_t elements = uplo == ACLBLAS_UPPER ?
                                  static_cast<uint64_t>(columns) * (columns + 1ULL) / 2ULL :
                                  static_cast<uint64_t>(columns) * (2ULL * n - columns + 1ULL) / 2ULL;
    const uint64_t transfers = uplo == ACLBLAS_UPPER ? LargeCeilRunSum(columns, kLargeChunkFloats) :
                                                       LargeCeilRunSum(n, kLargeChunkFloats) -
                                                           LargeCeilRunSum(static_cast<uint64_t>(n) - columns,
                                                               kLargeChunkFloats);
    return kLargeCostPerColumn * static_cast<uint64_t>(columns) + kLargeCostPerElement * elements +
        kLargeCostPerTransfer * transfers;
}

__aicore__ inline uint64_t LargePartitionTarget(uint64_t total, uint32_t part, uint32_t parts)
{
    if (parts == 0U) {
        return 0ULL;  // unreachable at runtime (parts == useCoreNum, host fills >= 1);
                      // guard lives in this function's entry scope for the div-by-zero scan.
    }
    return (total / parts) * part + (total % parts) * part / parts;
}

__aicore__ inline uint32_t WeightedColumnBoundary(uint32_t part, uint32_t parts, uint32_t n, uint32_t uplo)
{
    if (part == 0U) {
        return 0U;
    }
    if (part >= parts) {
        return n;
    }
    const uint64_t target = LargePartitionTarget(LargeColumnWorkPrefix(n, n, uplo), part, parts);
    uint32_t lo = 0U;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        if (LargeColumnWorkPrefix(mid, n, uplo) < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// ── R18 largeMode: column-ownership tile path ─────────────────────────────────────────────
// Skeleton ported from the !333 UnifiedContiguousColumns design, adapted to this kernel's
// idioms: immediate-adjacent Set/Wait flag pairs instead of its PipeBarrier<PIPE_ALL>,
// depth-2 TQue double buffering kept, house-style pad-corner copy-in (rightPadding rounds the
// copy up one extra 32B block; every buffer that receives a padded copy-in carries +8 elems),
// and the exact/naive arithmetic chain copied STATEMENT BY STATEMENT from the walk path's
// UpdateRuns / GuardFiniteness (kernel arithmetic core - copy, do not refactor). The
// per-element result is bitwise-identical to the walk path on the same lanes; only the
// element->core ownership changes (columns instead of 32B-row-aligned element ranges).
class SsprKernelLarge {
public:
    __aicore__ inline SsprKernelLarge() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint64_t ColStart(uint64_t col) const
    {
        return isUpper ? (col * (col + 1U) / 2U) : (col * (2U * n - col + 1U) / 2U);
    }
    __aicore__ inline void ProcessLargeColumns(uint32_t colStart, uint32_t colEnd);
    __aicore__ inline void LargeProcessColumn(uint32_t col, uint32_t tileRow, uint32_t tileEnd);
    __aicore__ inline void LargeProcessHead(uint32_t& row, uint32_t tileRow, uint32_t rowBegin,
        uint32_t segmentEnd, uint64_t apBase, float axCol);
    __aicore__ inline void LargeCopyIn(uint64_t apOff, uint32_t len);
    __aicore__ inline void LargeCopyOut(uint64_t apOff, uint32_t len);
    __aicore__ inline void LargeComputeSegment(LocalTensor<float> xg, float axCol,
        TBuf<TPosition::VECCALC>& thSrc, TBuf<TPosition::VECCALC>& tlSrc, uint32_t len);

    TPipe pipe;
    GlobalTensor<float> xGM;
    GlobalTensor<float> apGM;
    TQue<QuePosition::VECIN, SSPR_IN_BUF_NUM> apInQueue;
    TQue<QuePosition::VECOUT, SSPR_OUT_BUF_NUM> apOutQueue;
    TBuf<TPosition::VECCALC> xTileBuf;   // x row-factor tile (SSPR_LARGE_TILE_FLOATS)
    TBuf<TPosition::VECCALC> xSegBuf;    // head Gather dst (< headAlign + 8)
    TBuf<TPosition::VECCALC> headIdxBuf; // head Gather idx (byte offsets of x[row..))
    TBuf<TPosition::VECCALC> headThBuf; // head inline Dekker hi scratch
    TBuf<TPosition::VECCALC> headTlBuf; // head inline Dekker lo scratch
    TBuf<TPosition::VECCALC> xiBuf;
    TBuf<TPosition::VECCALC> thBuf;    // per-segment Dekker hi (exact band)
    TBuf<TPosition::VECCALC> tlBuf;    // per-segment Dekker lo; doubles as the guard mask
    TBuf<TPosition::VECCALC> eBuf;
    TBuf<TPosition::VECCALC> cBuf;

    uint64_t n = 0;
    uint32_t uplo = ACLBLAS_UPPER;
    float alpha = 0.0f;
    uint32_t useCoreNum = 1;
    uint32_t vecIdx = 0;
    bool isUpper = true;
    uint32_t segLen = SSPR_LARGE_SEG_FLOATS;  // segment length (by arithmetic band)
    bool largeExact = false;  // arithmetic band: n > SSPR_NAIVE_MAX_N -> Dekker + guard
};

// Buffer set of Init (largeMode). Segment length by arithmetic band: naive 2-op chain
// (n <= SSPR_NAIVE_MAX_N) needs xi only; the exact Dekker chain + finiteness guard adds
// th/tl/e/c (tl doubles as the guard compare mask - dead after its last Axpy read - the
// R18.1-proven aliasing) plus the head inline split scratch. UB totals (floats),
// exact band, T = 16384 / segLen = 2048:
//   T + 11*seg + 5*64 + 4*8 = 16384 + 22528 + 352 = 39,264 (153.4KB)
//   (11*seg = xi + th + tl + e + c + 2*apIn + 2*apOut + xSeg + headIdx; 5*64 slack =
//    e + c + tl + 2*apOut; 4*8 = xSeg/headIdx + the 2 apIn copy-in pads)
// naive band: T + 7*segNaive + 4*8 = 16384 + 28672 + 32 = 45,088 (176.1KB)
//   (7*segNaive = xi + xSeg + headIdx + 2*apIn + 2*apOut; 4*8 = xSeg/headIdx/apIn pads)
// Both under the 192KB ceiling, MIRRORED by the host SsprUbBytesLarge gate (a UB
// over-allocation silently no-ops the kernel on this silicon, so the gate must refuse
// before launch).
__aicore__ inline void SsprKernelLarge::Init(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling)
{
    vecIdx = static_cast<uint32_t>(GetBlockIdx());
    n = tiling.n;
    uplo = tiling.uplo;
    alpha = tiling.alpha;
    useCoreNum = tiling.useCoreNum;
    isUpper = (uplo == ACLBLAS_UPPER);
    largeExact = (n > SSPR_NAIVE_MAX_N);
    segLen = largeExact ? SSPR_LARGE_SEG_FLOATS : SSPR_LARGE_SEG_FLOATS_NAIVE;

    xGM.SetGlobalBuffer((__gm__ float*)x);
    apGM.SetGlobalBuffer((__gm__ float*)ap);

    pipe.InitBuffer(xTileBuf, SSPR_LARGE_TILE_FLOATS * BYTENUM_PER_FLOAT32);
    // Head scratch. A head is < SSPR_LARGE_HEAD_ALIGN_ELEMS elements, so 72 floats would
    // suffice; the buffers are deliberately sized segLen + 8 (the R18.1 xSeg size) so the
    // downstream buffer offsets match the R18.1 layout - the MTE queue slot addresses are
    // sensitive to the UB layout (a layout shift of ~16KB cost +2.8% on the UPPER exact
    // band in R19 probes; same-layout builds measured R18.1-parity). +8 = the house
    // padded-copy-in slack.
    pipe.InitBuffer(xSegBuf, (segLen + SSPR_ALIGN_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(headIdxBuf, (segLen + SSPR_ALIGN_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(xiBuf, segLen * BYTENUM_PER_FLOAT32);
    if (largeExact) {
        pipe.InitBuffer(headThBuf, SSPR_LARGE_HEAD_FLOATS * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(headTlBuf, SSPR_LARGE_HEAD_FLOATS * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(thBuf, segLen * BYTENUM_PER_FLOAT32);
        // tl/e/c/apOut carry +SSPR_GUARD_SLACK_ELEMS: the Phase D guard touches them at
        // count = ceil64(len) (sub-64/non-64-multiple guard counts are broken on 910B3).
        pipe.InitBuffer(tlBuf, (segLen + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(eBuf, (segLen + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(cBuf, (segLen + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    }
    // +8 per apIn slot: padded copy-in slack. apOut slack only on the exact band (guard).
    pipe.InitBuffer(apInQueue, SSPR_IN_BUF_NUM, (segLen + SSPR_ALIGN_ELEMS) * BYTENUM_PER_FLOAT32);
    const uint32_t apOutSlack = largeExact ? SSPR_GUARD_SLACK_ELEMS : 0U;
    pipe.InitBuffer(apOutQueue, SSPR_OUT_BUF_NUM,
        (segLen + apOutSlack) * BYTENUM_PER_FLOAT32);
}

// Column ownership. The core's range is a weighted COLUMN range solved in-kernel
// (WeightedColumnBoundary, ported cost model) - NOT coreBegin, which stays the element-axis
// walk split of the other TU. Empty ranges (n small vs parts) return.
__aicore__ inline void SsprKernelLarge::Process()
{
    if (vecIdx >= useCoreNum) {
        return;
    }
    if (n > SSPR_LARGE_MAX_N) {
        return;  // second line of defence: the host gate caps n at the tile size; a
                 // hostile tiling would otherwise walk past the 16384-float xTile.
    }
    const uint32_t n32 = static_cast<uint32_t>(n);
    const uint32_t colStart = WeightedColumnBoundary(vecIdx, useCoreNum, n32, uplo);
    const uint32_t colEnd = WeightedColumnBoundary(vecIdx + 1U, useCoreNum, n32, uplo);
    if (colStart < colEnd) {
        ProcessLargeColumns(colStart, colEnd);
    }
}

// Per-core column range of Process: the x row-factor tile is loaded once per tile
// (n <= SSPR_LARGE_TILE_FLOATS means exactly one tile in the supported range), ordered by
// an immediate-adjacent MTE2_V pair, then every column of [colStart, colEnd) that
// intersects the tile is processed. Column c's segment within the tile is
// [max(rowBegin, tileRow), min(rowEnd, tileEnd)); rowBegin is 0 for UPPER (columns always
// start at the tile base -> offset 0, no head) and c for LOWER (arbitrary offset -> the
// head Gather of LargeProcessColumn). Aligned segments start at
// SSPR_LARGE_HEAD_ALIGN_ELEMS-multiple offsets (>= 32B-aligned; the walk path issues V ops
// on 8-element-granular slices, so 64 keeps margin).
__aicore__ inline void SsprKernelLarge::ProcessLargeColumns(uint32_t colStart, uint32_t colEnd)
{
    LocalTensor<float> xTile = xTileBuf.Get<float>();
    const uint32_t n32 = static_cast<uint32_t>(n);
    for (uint32_t tileRow = 0; tileRow < n32; tileRow += SSPR_LARGE_TILE_FLOATS) {
        const uint32_t tileEnd = (n32 - tileRow < SSPR_LARGE_TILE_FLOATS) ? n32
            : tileRow + SSPR_LARGE_TILE_FLOATS;
        const uint32_t tileCount = tileEnd - tileRow;
        // xGM[tileRow] is 32B aligned (tileRow is a tile multiple); xTile is exactly
        // SSPR_LARGE_TILE_FLOATS, so the non-aligned tail pads up to at most the buffer end.
        if (tileCount % SSPR_ALIGN_ELEMS == 0U) {
            DataCopy(xTile, xGM[tileRow], tileCount);
        } else {
            const uint32_t paddingNumU = SSPR_ALIGN_ELEMS - tileCount % SSPR_ALIGN_ELEMS;  // [1,7]
            DataCopyExtParams copyParams{1, tileCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
            DataCopyPad(xTile, xGM[tileRow], copyParams, padParams);
        }
        int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        for (uint32_t col = colStart; col < colEnd; ++col) {
            LargeProcessColumn(col, tileRow, tileEnd);
        }
    }
}

// Partial-head block of LargeProcessColumn (statements moved verbatim, in order): gathers the
// head's row factors from the resident xTile into xSeg (buffer base 0), runs the compute chain
// on xSeg and copies the head result out; advances row to the next HEAD_ALIGN boundary (clamped
// to segmentEnd). th/tl arrive from the head scratch (headThBuf/headTlBuf) here, vs the
// per-segment thBuf/tlBuf on the aligned path.
__aicore__ inline void SsprKernelLarge::LargeProcessHead(uint32_t& row, uint32_t tileRow,
    uint32_t rowBegin, uint32_t segmentEnd, uint64_t apBase, float axCol)
{
    uint32_t alignedRow = (row + SSPR_LARGE_HEAD_ALIGN_ELEMS - 1U) &
        ~(SSPR_LARGE_HEAD_ALIGN_ELEMS - 1U);
    if (alignedRow > segmentEnd) {
        alignedRow = segmentEnd;
    }
    const uint32_t count = alignedRow - row;
    // The head's AP copy-in is issued first so its MTE2 overlaps the Gather V ops.
    LargeCopyIn(apBase + row - rowBegin, count);
    LocalTensor<float> xTile = xTileBuf.Get<float>();
    LocalTensor<float> xSeg = xSegBuf.Get<float>();
    LocalTensor<int32_t> headIdx = headIdxBuf.Get<int32_t>();
    ArithProgression(headIdx, static_cast<int32_t>((row - tileRow) * BYTENUM_PER_FLOAT32),
        static_cast<int32_t>(BYTENUM_PER_FLOAT32), count);
    PipeBarrier<PIPE_V>();
    Gather(xSeg, xTile, headIdx.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
        count);
    PipeBarrier<PIPE_V>();
    LargeComputeSegment(xSeg, axCol, headThBuf, headTlBuf, count);
    LargeCopyOut(apBase + row - rowBegin, count);
    row = alignedRow;
}

// One column of ProcessLargeColumns. Segments are software-pipelined exactly like the walk
// window loop (CopyIn(k+1) overlaps compute(k); depth-2 queues), except segments are
// AP-CONTIGUOUS WITHIN ONE COLUMN (apOff advances by len) instead of windows of a core
// element range. Column factor: one scalar GM read per column, axCol = alpha * x[col] (the
// same single per-column rounding as the golden).
__aicore__ inline void SsprKernelLarge::LargeProcessColumn(uint32_t col, uint32_t tileRow,
    uint32_t tileEnd)
{
    const uint32_t rowBegin = isUpper ? 0U : col;
    const uint32_t rowEnd = isUpper ? col + 1U : static_cast<uint32_t>(n);
    uint32_t row = (rowBegin > tileRow) ? rowBegin : tileRow;
    const uint32_t segmentEnd = (rowEnd < tileEnd) ? rowEnd : tileEnd;
    if (row >= segmentEnd) {
        return;  // column does not intersect this tile
    }
    const uint64_t apBase = ColStart(col);
    const float axCol = alpha * xGM.GetValue(col);

    // Partial head: the column's first row inside the tile sits at a non-64-multiple offset
    // (LOWER diagonals - UPPER columns always start at the tile base). V ops at unaligned UB
    // offsets are broken on this silicon, so the head cannot read the xTile slices directly;
    // its row factors are GATHERED from the resident xTile into xSeg (buffer base 0) and
    // the head runs the full inline Dekker chain on xSeg. The head stops at the next
    // 64-element boundary (was segLen in R18), so it is <= 63 elements per column per tile.
    if (((row - tileRow) & (SSPR_LARGE_HEAD_ALIGN_ELEMS - 1U)) != 0U) {
        LargeProcessHead(row, tileRow, rowBegin, segmentEnd, apBase, axCol);
    }
    if (row >= segmentEnd) {
        return;  // partial head consumed the whole tile-column intersection
    }

    // Aligned segments: xTile slices at HEAD_ALIGN-multiple offsets, pipelined over the
    // queues.
    uint64_t apOff = apBase + row - rowBegin;
    uint32_t len = (segmentEnd - row < segLen) ? segmentEnd - row : segLen;
    LargeCopyIn(apOff, len);
    while (row < segmentEnd) {
        const uint32_t nextRow = row + len;
        uint64_t nextApOff = 0ULL;
        uint32_t nextLen = 0U;
        if (nextRow < segmentEnd) {
            nextApOff = apOff + len;  // contiguous within the column
            nextLen = (segmentEnd - nextRow < segLen) ? segmentEnd - nextRow : segLen;
            LargeCopyIn(nextApOff, nextLen);
        }
        LocalTensor<float> xTile = xTileBuf.Get<float>();
        LargeComputeSegment(xTile[row - tileRow], axCol, thBuf, tlBuf, len);
        LargeCopyOut(apOff, len);
        row = nextRow;
        apOff = nextApOff;
        len = nextLen;
    }
}

// Segment copy-in of LargeProcessColumn. The AP segment may start at ANY GM offset (packed
// column bases are not 8-aligned), so DataCopyPad is unconditional; the rightPadding rounds
// the copy up to a 32B boundary (8 when len%8==0 - the exactly-one-32B-block pad corner),
// which is why each apIn slot carries +8 elems.
__aicore__ inline void SsprKernelLarge::LargeCopyIn(uint64_t apOff, uint32_t len)
{
    LocalTensor<float> apIn = apInQueue.AllocTensor<float>();
    const DataCopyExtParams copyParams{1, len * BYTENUM_PER_FLOAT32, 0, 0, 0};
    const uint32_t paddingNumU = SSPR_ALIGN_ELEMS - len % SSPR_ALIGN_ELEMS;  // [1,8]
    const DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
    DataCopyPad(apIn, apGM[apOff], copyParams, padParams);
    apInQueue.EnQue<float>(apIn);
}

// Segment copy-out of LargeProcessColumn: exact byte length to the possibly unaligned GM
// destination (3-arg DataCopyPad, the unaligned copy-out idiom). NOTE: unlike the walk path
// (32B-row-aligned inter-core element ranges), two cores may RMW the SAME 32B GM block at a
// column boundary here - byte-precise MTE3 writes keep the updates disjoint; the P0
// cross-core partial-row test (repeat-run bitwise determinism) guards this.
__aicore__ inline void SsprKernelLarge::LargeCopyOut(uint64_t apOff, uint32_t len)
{
    LocalTensor<float> apOut = apOutQueue.DeQue<float>();
    const DataCopyExtParams copyParams{1, len * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPad(apGM[apOff], apOut, copyParams);
    apOutQueue.FreeTensor(apOut);
}

// Per-segment arithmetic of LargeProcessColumn (also the head body on xSeg): the exact
// statement sequence of the walk path's UpdateRuns (one run) + GuardFiniteness,
// buffer-RENAMED only - sc aliases apOut (the plain mul+add result is only consumed by the
// guard's Select fallback), the guard's Abs scratch aliases the dead e (e's last read is
// Add(c, sc, e)), the guard's compare mask aliases the dead tl (last read Axpy(e, tl, al)).
// th/tl arrive as their TBuf sources (thBuf/tlBuf on the aligned path, headThBuf/headTlBuf
// on the head) so this one body replaces the LargeComputeSegment / LargeComputeHead twin
// pair. The Get stays INSIDE the exact branch, exactly where the twins had it: thBuf/tlBuf
// are only InitBuffer'ed on the exact band, so the naive branch must never touch them.
// Same sync idiom: the MTE2_V pair orders apIn before the V reads, the V_MTE3 pair orders
// the V writes before the copy-out, both immediately adjacent.
__aicore__ inline void SsprKernelLarge::LargeComputeSegment(LocalTensor<float> xg, float axCol,
    TBuf<TPosition::VECCALC>& thSrc, TBuf<TPosition::VECCALC>& tlSrc, uint32_t len)
{
    LocalTensor<float> apIn = apInQueue.DeQue<float>();
    LocalTensor<float> apOut = apOutQueue.AllocTensor<float>();
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);

    LocalTensor<float> xi = xiBuf.Get<float>();
    if (!largeExact) {
        // naive 2-op chain - bitwise the UpdateRuns naive branch / ComputeWindowVector.
        Muls(xi, xg, axCol, len);
        Add(apOut, apIn, xi, len);
    } else {
        LocalTensor<float> th = thSrc.Get<float>();
        LocalTensor<float> tl = tlSrc.Get<float>();
        LocalTensor<float> e = eBuf.Get<float>();
        LocalTensor<float> c = cBuf.Get<float>();
        // Dekker 12-op chain, copied statement by statement from UpdateRuns (sc == apOut).
        float c1 = SPLIT_CONST * axCol;
        float ah = c1 - (c1 - axCol);
        float al = axCol - ah;
        Muls(xi, xg, axCol, len);
        Add(apOut, apIn, xi, len);
        Muls(th, xg, SPLIT_CONST, len);
        Sub(tl, th, xg, len);
        Sub(th, th, tl, len);
        Sub(tl, xg, th, len);
        Muls(e, th, ah, len);
        Sub(e, e, xi, len);
        Axpy(e, tl, ah, len);
        Axpy(e, th, al, len);
        Axpy(e, tl, al, len);
        Add(c, apOut, e, len);
        // Phase D finiteness guard, copied from GuardFiniteness at count = ceil64(len)
        // (sub-64 / non-64-multiple counts are broken on 910B3), with the alias renames.
        const uint32_t lenPad = (len + 63U) & ~63U;
        Abs(e, c, lenPad);  // |c| into the dead e (guard mask input)
        PipeBarrier<PIPE_V>();
        Compares(tl.ReinterpretCast<uint16_t>()[0], e, FLT_MAX_F, CMPMODE::LT, lenPad);
        PipeBarrier<PIPE_V>();
        Select(apOut, tl.ReinterpretCast<uint16_t>()[0], c, apOut,
            SELMODE::VSEL_TENSOR_TENSOR_MODE, lenPad);
        PipeBarrier<PIPE_V>();
    }
    int32_t evVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evVMte3);
    WaitFlag<HardEvent::V_MTE3>(evVMte3);
    apInQueue.FreeTensor(apIn);
    apOutQueue.EnQue<float>(apOut);
}
}  // namespace

// LargeMode column-ownership tile kernel (R18/R18.1/R19). Launched by sspr_kernel_large_do only
// when tiling.largeMode == 1 (host gate: incx == 1 with n inside (SSPR_LARGE_MIN_N_UPPER,
// SSPR_LARGE_MAX_N] for BOTH fill modes, desc workspace unavailable, UB budget closed by
// SsprUbBytesLarge). The in-kernel largeMode re-check is a defensive second line; x is
// strictly READ-ONLY (same contract as sspr_kernel).
__global__ __aicore__ __vector__ void sspr_kernel_large(GM_ADDR x, GM_ADDR ap, const SsprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.largeMode != 1U) {
        return;
    }
    SsprKernelLarge op;
    op.Init(x, ap, tiling);
    op.Process();
}

// Launch wrapper of sspr_kernel_large (cross-TU entry; sspr_kernel_do calls this so the
// __global__ symbol stays inside this TU). Same launch ABI and stream discipline as
// sspr_kernel_do; numBlocks = tiling.useCoreNum (host fills 40 parts for the eligible band).
// x crosses this boundary as const and the kernel treats it strictly read-only; the launch
// ABI parameter (GM_ADDR) has no const-qualified form, so the same documented const_cast
// re-arms the pointer here; no device code writes through it.
void sspr_kernel_large_do(const uint8_t* x, uint8_t* ap, const SsprTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    sspr_kernel_large<<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(x), ap, tiling);
}
