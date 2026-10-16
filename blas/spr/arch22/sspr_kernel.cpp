/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "cann_ops_blas_common.h"
#include "sspr_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t kChunkFloats = 8192;
constexpr uint32_t kMaxPackedStride = 128;
// Maximum vector length admitted to the tiled strided path; larger n uses ScalarColumns.
constexpr uint32_t kMaxCanonicalN = 32768;
constexpr uint32_t kPackSpanFloats = 4096;
constexpr uint32_t kPackOffsetFloats = 4096;
constexpr uint32_t kLocalRowFloats = 4096;
constexpr uint32_t kLocalColFloats = 8192;

// Cost of the columns [0, columns), used to split columns across AI vector cores.
//
// A core's time is not proportional to the number of packed elements it owns. Owning k
// columns costs roughly
//
//     sum over the k columns of len  +  k * p  +  q * (extra transfers)
//
// where p is a flat charge every column pays regardless of length (AP base address, scalar
// broadcast of x[col], loop and descriptor setup, first DMA round trip) and q is charged
// again for each further kChunkFloats piece a long column has to walk. Equalising element
// counts ignores both: at n = 32768 the slowest core took 2x the mean, because half the
// columns of a triangular matrix are short and a core owning many of them pays p many times
// while moving few elements.
//
// Implemented below as kCostPerColumn * columns + kCostPerElement * elements +
// kCostPerTransfer * transfers, which is the same function since every column contributes at
// least one transfer:
//
//     kCostPerColumn + len + kCostPerTransfer * ceil(len / kChunkFloats)
//   = (kCostPerColumn + kCostPerTransfer) + len + kCostPerTransfer * floor((len - 1) / kChunkFloats)
//        \___________ p ________________/                              \_ extra transfers _/
//
// Note which quantity is actually identifiable. For len <= kChunkFloats the ceiling is 1, so
// only the sum p = kCostPerColumn + kCostPerTransfer affects the split; the two constants are
// separable only where columns span several transfers. Measurements bear that out: candidates
// sharing a sum scored within 0.2 percentage points of each other, far inside the ~2.7% per
// case run-to-run spread, so do not read the individual values as measurements of a per-column
// and a per-transfer cost. What the data supports is p in the 4096-8192 range (p = 0 scores
// 0.71x, i.e. worse than an equal-column split) and a smaller positive extra-transfer charge.
//
// Selected by timing 44 (n, uplo, incx) configurations over a grid of candidates on 910B3,
// then 9 repeats of the top four. A later LOWER-only sweep tested 25 coefficient pairs and
// confirmed four finalists at warmup 300 / measurement 1000 over seven sizes. The alternatives
// gained at most 2.0% geometric mean but regressed individual sizes by up to 10.2%, so LOWER
// retains the same coefficients as UPPER. Each term has an O(1) closed form so the boundary
// search below stays O(log n).
constexpr uint64_t kCostPerColumn = 2048;   // with kCostPerTransfer: p = 6144 per column
constexpr uint64_t kCostPerElement = 1;     // per packed element streamed
constexpr uint64_t kCostPerTransfer = 4096; // per kChunkFloats-sized transfer

// sum_{len=1..m} ceil(len / grain)
__aicore__ inline uint64_t CeilRunSum(uint64_t m, uint64_t grain)
{
    if (m == 0ULL) {
        return 0ULL;
    }
    const uint64_t q = m / grain;
    const uint64_t r = m - q * grain;
    return grain * q * (q + 1ULL) / 2ULL + r * (q + 1ULL);
}

__aicore__ inline uint64_t ColumnWorkPrefix(uint32_t columns, uint32_t n, uint32_t uplo)
{
    // Column c has length c+1 (UPPER) or n-c (LOWER); the sums below run over [0, columns).
    const uint64_t elements = uplo == ACLBLAS_UPPER ?
                                  static_cast<uint64_t>(columns) * (columns + 1ULL) / 2ULL :
                                  static_cast<uint64_t>(columns) * (2ULL * n - columns + 1ULL) / 2ULL;
    const uint64_t transfers = uplo == ACLBLAS_UPPER ? CeilRunSum(columns, kChunkFloats) :
                                                       CeilRunSum(n, kChunkFloats) -
                                                           CeilRunSum(static_cast<uint64_t>(n) - columns, kChunkFloats);
    return kCostPerColumn * static_cast<uint64_t>(columns) + kCostPerElement * elements + kCostPerTransfer * transfers;
}

__aicore__ inline uint64_t PartitionTarget(uint64_t total, uint32_t part, uint32_t parts)
{
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
    const uint64_t target = PartitionTarget(ColumnWorkPrefix(n, n, uplo), part, parts);
    uint32_t lo = 0U;
    uint32_t hi = n;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2U;
        if (ColumnWorkPrefix(mid, n, uplo) < target) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__aicore__ inline uint64_t UpperColumnBase(uint32_t col) { return static_cast<uint64_t>(col) * (col + 1ULL) / 2ULL; }

__aicore__ inline uint64_t LowerColumnBase(uint32_t col, uint32_t n)
{
    return static_cast<uint64_t>(col) * (2ULL * n - col + 1ULL) / 2ULL;
}

__aicore__ inline uint64_t XPhysicalIndex(uint32_t logical, uint32_t n, int64_t incx)
{
    const uint64_t absIncx = static_cast<uint64_t>(incx >= 0 ? incx : -incx);
    return incx >= 0 ? static_cast<uint64_t>(logical) * absIncx : static_cast<uint64_t>(n - 1U - logical) * absIncx;
}

__aicore__ inline void ScalarColumns(
    GlobalTensor<float>& xGm, GlobalTensor<float>& apGm, const SsprTilingData& tiling, uint32_t colStart,
    uint32_t colEnd)
{
    for (uint32_t col = colStart; col < colEnd; ++col) {
        const float axCol = tiling.alpha * xGm.GetValue(XPhysicalIndex(col, tiling.n, tiling.incx));
        if (tiling.uplo == ACLBLAS_UPPER) {
            const uint64_t base = UpperColumnBase(col);
            for (uint32_t row = 0; row <= col; ++row) {
                const uint64_t apIndex = base + row;
                apGm.SetValue(
                    apIndex, apGm.GetValue(apIndex) + axCol * xGm.GetValue(XPhysicalIndex(row, tiling.n, tiling.incx)));
            }
        } else {
            const uint64_t base = LowerColumnBase(col, tiling.n);
            for (uint32_t row = col; row < tiling.n; ++row) {
                const uint64_t apIndex = base + (row - col);
                apGm.SetValue(
                    apIndex, apGm.GetValue(apIndex) + axCol * xGm.GetValue(XPhysicalIndex(row, tiling.n, tiling.incx)));
            }
        }
    }
}

#if __DAV_C220_VEC__
__aicore__ inline void CanonicalizeStridedRange(
    LocalTensor<float> cache, LocalTensor<float> span, LocalTensor<uint32_t> offsets, __gm__ float* source, uint32_t n,
    int64_t incx, uint32_t logicalBase, uint32_t logicalCount)
{
    const uint32_t increment = static_cast<uint32_t>(incx > 0 ? incx : -incx);
    auto ubSpan = reinterpret_cast<__ubuf__ float*>(span.GetPhyAddr());

    // CachedStridedColumns is dispatched only for 1 <= |incx| <= kMaxPackedStride (128).
    // Thus a 4096-float span holds at least 32 logical elements. Round the tile
    // down to a multiple of 8 so every Gather destination stays 32-byte aligned.
    const uint32_t logicalTile = (kPackSpanFloats / increment) & ~7U;

    for (uint32_t localStart = 0; localStart < logicalCount; localStart += logicalTile) {
        const uint32_t count = (logicalCount - localStart < logicalTile) ? logicalCount - localStart : logicalTile;
        const uint32_t globalStart = logicalBase + localStart;
        const uint32_t physicalStart = incx > 0 ? globalStart * increment : (n - globalStart - count) * increment;
        const uint32_t spanCount = (count - 1U) * increment + 1U;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t sourceIndex = incx > 0 ? i * increment : (count - 1U - i) * increment;
            offsets.SetValue(i, sourceIndex * sizeof(float));
        }
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
        copy_gm_to_ubuf_align_b32(ubSpan, source + physicalStart, 0, 1, spanCount * sizeof(float), 0, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Gather(cache[localStart], span, offsets, 0U, count);
        PipeBarrier<PIPE_V>();
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
    }
}
__aicore__ inline void CachedStridedColumns(
    __gm__ float* gmX, GlobalTensor<float>& apGm, const SsprTilingData& tiling, uint32_t colStart, uint32_t colEnd)
{
    constexpr uint32_t kRowTile = kLocalRowFloats;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xColBuf;
    TBuf<TPosition::VECCALC> xRowBuf;
    TBuf<TPosition::VECCALC> spanBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> xWorkBuf;
    TQue<QuePosition::VECIN, 2> apQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(xColBuf, kLocalColFloats * sizeof(float));
    pipe.InitBuffer(xRowBuf, kRowTile * sizeof(float));
    pipe.InitBuffer(spanBuf, kPackSpanFloats * sizeof(float));
    pipe.InitBuffer(offsetBuf, kPackOffsetFloats * sizeof(uint32_t));
    pipe.InitBuffer(xWorkBuf, kRowTile * sizeof(float));
    pipe.InitBuffer(apQueue, 2, kRowTile * sizeof(float));
    pipe.InitBuffer(outQueue, 2, kRowTile * sizeof(float));
    LocalTensor<float> xCol = xColBuf.Get<float>();
    LocalTensor<float> xRow = xRowBuf.Get<float>();
    LocalTensor<float> xWork = xWorkBuf.Get<float>();
    LocalTensor<float> span = spanBuf.Get<float>();
    LocalTensor<uint32_t> offsets = offsetBuf.Get<uint32_t>();
    auto ubXCol = reinterpret_cast<__ubuf__ float*>(xCol.GetPhyAddr());
    auto ubOffsets = reinterpret_cast<__ubuf__ uint32_t*>(offsets.GetPhyAddr());
    const uint32_t colCount = colEnd - colStart;
    if (colCount > kLocalColFloats) {
        return;
    }
    CanonicalizeStridedRange(xCol, span, offsets, gmX, tiling.n, tiling.incx, colStart, colCount);

    for (uint32_t tileRow = 0; tileRow < tiling.n; tileRow += kRowTile) {
        const uint32_t tileEnd = (tiling.n - tileRow < kRowTile) ? tiling.n : tileRow + kRowTile;
        const uint32_t fullCount = tileEnd - tileRow;
        CanonicalizeStridedRange(xRow, span, offsets, gmX, tiling.n, tiling.incx, tileRow, fullCount);
        const DataCopyExtParams fullCopy{1, fullCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        const DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};

        const uint32_t rectBegin = tiling.uplo == ACLBLAS_UPPER ? (colStart > tileEnd ? colStart : tileEnd) : colStart;
        const uint32_t rectEnd = tiling.uplo == ACLBLAS_UPPER ? colEnd : (colEnd < tileRow ? colEnd : tileRow);
        if (rectBegin < rectEnd) {
            const uint32_t firstCol = rectBegin;
            const uint64_t firstOffset = tiling.uplo == ACLBLAS_UPPER ?
                                             UpperColumnBase(firstCol) + tileRow :
                                             LowerColumnBase(firstCol, tiling.n) + (tileRow - firstCol);
            LocalTensor<float> firstAp = apQueue.AllocTensor<float>();
            DataCopyPad(firstAp, apGm[firstOffset], fullCopy, pad);
            apQueue.EnQue(firstAp);
            for (uint32_t col = rectBegin; col < rectEnd; ++col) {
                if (col + 1U < rectEnd) {
                    const uint32_t nextCol = col + 1U;
                    const uint64_t nextOffset = tiling.uplo == ACLBLAS_UPPER ?
                                                    UpperColumnBase(nextCol) + tileRow :
                                                    LowerColumnBase(nextCol, tiling.n) + (tileRow - nextCol);
                    LocalTensor<float> nextAp = apQueue.AllocTensor<float>();
                    DataCopyPad(nextAp, apGm[nextOffset], fullCopy, pad);
                    apQueue.EnQue(nextAp);
                }
                LocalTensor<float> apIn = apQueue.DeQue<float>();
                LocalTensor<float> out = outQueue.AllocTensor<float>();
                const float axCol = tiling.alpha * ubXCol[col - colStart];
                Muls(out, xRow, axCol, fullCount);
                Add(out, out, apIn, fullCount);
                outQueue.EnQue(out);
                apQueue.FreeTensor(apIn);
                out = outQueue.DeQue<float>();
                const uint64_t apOffset = tiling.uplo == ACLBLAS_UPPER ?
                                              UpperColumnBase(col) + tileRow :
                                              LowerColumnBase(col, tiling.n) + (tileRow - col);
                DataCopyPad(apGm[apOffset], out, fullCopy);
                outQueue.FreeTensor(out);
            }
        }

        const uint32_t diagBegin = colStart > tileRow ? colStart : tileRow;
        const uint32_t diagEnd = colEnd < tileEnd ? colEnd : tileEnd;
        if (diagBegin < diagEnd) {
            if (tiling.uplo == ACLBLAS_UPPER) {
                for (uint32_t col = diagBegin; col < diagEnd; ++col) {
                    const uint32_t count = col + 1U - tileRow;
                    const uint64_t apOffset = UpperColumnBase(col) + tileRow;
                    DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                    LocalTensor<float> apIn = apQueue.AllocTensor<float>();
                    DataCopyPad(apIn, apGm[apOffset], copy, pad);
                    apQueue.EnQue(apIn);
                    apIn = apQueue.DeQue<float>();
                    LocalTensor<float> out = outQueue.AllocTensor<float>();
                    const float axCol = tiling.alpha * ubXCol[col - colStart];
                    Muls(out, xRow, axCol, count);
                    Add(out, out, apIn, count);
                    outQueue.EnQue(out);
                    apQueue.FreeTensor(apIn);
                    out = outQueue.DeQue<float>();
                    DataCopyPad(apGm[apOffset], out, copy);
                    outQueue.FreeTensor(out);
                }
            } else {
                for (uint32_t phase = 0; phase < 8U; ++phase) {
                    for (uint32_t i = 0; i < fullCount; ++i) {
                        ubOffsets[i] = (i + phase) * sizeof(float);
                    }
                    set_flag(PIPE_S, PIPE_V, EVENT_ID2);
                    wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
                    uint32_t col = diagBegin + ((phase + 8U - (diagBegin & 7U)) & 7U);
                    for (; col < diagEnd; col += 8U) {
                        const uint32_t localStart = col - tileRow;
                        const uint32_t count = tileEnd - col;
                        Gather(xWork, xRow[localStart - phase], offsets, 0U, count);
                        PipeBarrier<PIPE_V>();
                        const uint64_t apOffset = LowerColumnBase(col, tiling.n);
                        DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                        LocalTensor<float> apIn = apQueue.AllocTensor<float>();
                        DataCopyPad(apIn, apGm[apOffset], copy, pad);
                        apQueue.EnQue(apIn);
                        apIn = apQueue.DeQue<float>();
                        LocalTensor<float> out = outQueue.AllocTensor<float>();
                        const float axCol = tiling.alpha * ubXCol[col - colStart];
                        Muls(out, xWork, axCol, count);
                        Add(out, out, apIn, count);
                        outQueue.EnQue(out);
                        apQueue.FreeTensor(apIn);
                        out = outQueue.DeQue<float>();
                        DataCopyPad(apGm[apOffset], out, copy);
                        outQueue.FreeTensor(out);
                    }
                    set_flag(PIPE_V, PIPE_S, EVENT_ID2);
                    wait_flag(PIPE_V, PIPE_S, EVENT_ID2);
                }
            }
        }
    }
}

__aicore__ inline void UnifiedContiguousColumns(
    GlobalTensor<float>& xGm, GlobalTensor<float>& apGm, const SsprTilingData& tiling, uint32_t colStart,
    uint32_t colEnd)
{
    constexpr uint32_t kRowTile = 16384U;
    constexpr uint32_t kPipelineFloats = 4096U;
    TPipe pipe;
    TBuf<TPosition::VECCALC> xTileBuf;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 2> apQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    pipe.InitBuffer(xTileBuf, kRowTile * sizeof(float));
    pipe.InitBuffer(xQueue, 1, kPipelineFloats * sizeof(float));
    pipe.InitBuffer(apQueue, 2, kPipelineFloats * sizeof(float));
    pipe.InitBuffer(outQueue, 2, kPipelineFloats * sizeof(float));
    LocalTensor<float> xTile = xTileBuf.Get<float>();
    const DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};

    for (uint32_t tileRow = 0; tileRow < tiling.n; tileRow += kRowTile) {
        const uint32_t tileEnd = (tiling.n - tileRow < kRowTile) ? tiling.n : tileRow + kRowTile;
        const uint32_t tileCount = tileEnd - tileRow;
        const DataCopyExtParams tileCopy{1, tileCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(xTile, xGm[tileRow], tileCopy, pad);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t col = colStart; col < colEnd; ++col) {
            const uint32_t rowBegin = tiling.uplo == ACLBLAS_UPPER ? 0U : col;
            const uint32_t rowEnd = tiling.uplo == ACLBLAS_UPPER ? col + 1U : tiling.n;
            uint32_t row = rowBegin > tileRow ? rowBegin : tileRow;
            const uint32_t segmentEnd = rowEnd < tileEnd ? rowEnd : tileEnd;
            if (row >= segmentEnd) {
                continue;
            }
            const uint64_t apBase =
                tiling.uplo == ACLBLAS_UPPER ? UpperColumnBase(col) : LowerColumnBase(col, tiling.n);
            const float axCol = tiling.alpha * xGm.GetValue(col);

            // A LOWER diagonal suffix may start at an arbitrary Unified Buffer offset.
            // Read only the first partial pipeline segment directly; every later
            // 4096-element segment starts at a zero-based aligned offset in xTile.
            const uint32_t localStart = row - tileRow;
            if ((localStart & (kPipelineFloats - 1U)) != 0U) {
                uint32_t alignedRow = (row + kPipelineFloats - 1U) & ~(kPipelineFloats - 1U);
                if (alignedRow > segmentEnd) {
                    alignedRow = segmentEnd;
                }
                const uint32_t count = alignedRow - row;
                const uint64_t apOffset = apBase + row - rowBegin;
                const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                LocalTensor<float> xIn = xQueue.AllocTensor<float>();
                LocalTensor<float> apIn = apQueue.AllocTensor<float>();
                DataCopyPad(xIn, xGm[row], copy, pad);
                DataCopyPad(apIn, apGm[apOffset], copy, pad);
                xQueue.EnQue(xIn);
                apQueue.EnQue(apIn);
                xIn = xQueue.DeQue<float>();
                apIn = apQueue.DeQue<float>();
                LocalTensor<float> out = outQueue.AllocTensor<float>();
                Muls(out, xIn, axCol, count);
                Add(out, out, apIn, count);
                outQueue.EnQue(out);
                xQueue.FreeTensor(xIn);
                apQueue.FreeTensor(apIn);
                out = outQueue.DeQue<float>();
                DataCopyPad(apGm[apOffset], out, copy);
                outQueue.FreeTensor(out);
                row = alignedRow;
            }

            if (row < segmentEnd) {
                uint32_t currentCount = (segmentEnd - row < kPipelineFloats) ? segmentEnd - row : kPipelineFloats;
                uint64_t currentOffset = apBase + row - rowBegin;
                DataCopyExtParams currentCopy{1, currentCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                LocalTensor<float> firstAp = apQueue.AllocTensor<float>();
                DataCopyPad(firstAp, apGm[currentOffset], currentCopy, pad);
                apQueue.EnQue(firstAp);

                while (row < segmentEnd) {
                    const uint32_t nextRow = row + currentCount;
                    if (nextRow < segmentEnd) {
                        const uint32_t nextCount =
                            (segmentEnd - nextRow < kPipelineFloats) ? segmentEnd - nextRow : kPipelineFloats;
                        const uint64_t nextOffset = apBase + nextRow - rowBegin;
                        const DataCopyExtParams nextCopy{1, nextCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                        LocalTensor<float> nextAp = apQueue.AllocTensor<float>();
                        DataCopyPad(nextAp, apGm[nextOffset], nextCopy, pad);
                        apQueue.EnQue(nextAp);
                    }

                    LocalTensor<float> apIn = apQueue.DeQue<float>();
                    LocalTensor<float> out = outQueue.AllocTensor<float>();
                    Muls(out, xTile[row - tileRow], axCol, currentCount);
                    Add(out, out, apIn, currentCount);
                    outQueue.EnQue(out);
                    apQueue.FreeTensor(apIn);
                    out = outQueue.DeQue<float>();
                    DataCopyPad(apGm[currentOffset], out, currentCopy);
                    outQueue.FreeTensor(out);

                    row = nextRow;
                    if (row < segmentEnd) {
                        currentCount = (segmentEnd - row < kPipelineFloats) ? segmentEnd - row : kPipelineFloats;
                        currentOffset = apBase + row - rowBegin;
                        currentCopy =
                            DataCopyExtParams{1, currentCount * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                    }
                }
            }
        }
    }
}

#endif
} // namespace

extern "C" __global__ __aicore__ __vector__ void sspr_kernel(GM_ADDR x, GM_ADDR ap, const SsprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GlobalTensor<float> xGm;
    GlobalTensor<float> apGm;
    const uint64_t absIncx = static_cast<uint64_t>(tiling.incx >= 0 ? tiling.incx : -tiling.incx);
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(tiling.n - 1U) * absIncx + 1ULL);
    apGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(ap), static_cast<uint64_t>(tiling.n) * (tiling.n + 1ULL) / 2ULL);

    const uint32_t block = GetBlockIdx();
    const uint32_t blocks = GetBlockNum();
    uint32_t colStart;
    uint32_t colEnd;
    // The cost model is accurate for every n and uplo, so it replaces the previous
    // equal-column split unconditionally. The old code only enabled a weighted split for
    // strided large-n cases because its element-count model was wrong for short columns.
    colStart = WeightedColumnBoundary(block, blocks, tiling.n, tiling.uplo);
    colEnd = WeightedColumnBoundary(block + 1U, blocks, tiling.n, tiling.uplo);
    if (colStart >= colEnd) {
        return;
    }

#if __DAV_C220_VEC__
    if (tiling.incx == 1) {
        UnifiedContiguousColumns(xGm, apGm, tiling, colStart, colEnd);
        return;
    }
    if (tiling.n <= kMaxCanonicalN &&
        static_cast<uint32_t>(tiling.incx > 0 ? tiling.incx : -tiling.incx) <= kMaxPackedStride) {
        CachedStridedColumns(reinterpret_cast<__gm__ float*>(x), apGm, tiling, colStart, colEnd);
        return;
    }
#endif
    ScalarColumns(xGm, apGm, tiling, colStart, colEnd);
}

void sspr_kernel_do(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sspr_kernel<<<numBlocks, nullptr, stream>>>(x, ap, tiling);
}
