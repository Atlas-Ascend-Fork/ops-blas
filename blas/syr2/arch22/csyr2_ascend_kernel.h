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

#ifndef CSYR2_ASCEND_KERNEL_H
#define CSYR2_ASCEND_KERNEL_H

#include "csyr2_tiling.h"
#include "csyr2_vector_helpers.h"

template <uint32_t TILING_KEY, uint32_t kFullMode = 0, bool kUnitAlpha = false>
class Csyr2AscendKernel {
public:
    /* 0 UPPER 单位正步长；1 LOWER 单位正步长；2 UPPER 跨步；3 LOWER 跨步；4/5 为单位步长窗口路径。 */
    static constexpr bool kUpper =
        (TILING_KEY == CSYR2_TILING_KEY_UPPER_UNIT || TILING_KEY == CSYR2_TILING_KEY_UPPER_STRIDED ||
         TILING_KEY == CSYR2_TILING_KEY_UPPER_WINDOWED);
    static constexpr bool kUnitPos =
        (TILING_KEY == CSYR2_TILING_KEY_UPPER_UNIT || TILING_KEY == CSYR2_TILING_KEY_LOWER_UNIT ||
         TILING_KEY == CSYR2_TILING_KEY_UPPER_WINDOWED || TILING_KEY == CSYR2_TILING_KEY_LOWER_WINDOWED);
    static constexpr bool kWindowed =
        (TILING_KEY == CSYR2_TILING_KEY_UPPER_WINDOWED || TILING_KEY == CSYR2_TILING_KEY_LOWER_WINDOWED);

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy, uint32_t usedCoreNum,
        uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag);

    __aicore__ inline void InitDirectStorage();

    __aicore__ inline void InitDirect(
        GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, uint32_t tileRow, float alphaReal, float alphaImag);

    __aicore__ inline void InitVectorViews(GM_ADDR x, GM_ADDR y, GM_ADDR A, uint32_t blockIdx);

    __aicore__ inline void SelectLowerPacking();

    __aicore__ inline void SelectUpperReuse();

    __aicore__ inline void SelectWindowBatch(uint32_t ilvBytes, uint32_t planeBytes);

    __aicore__ inline void InitWindowStorage(uint32_t ilvBytes, uint32_t planeBytes);

    __aicore__ inline void InitUnitStorage(uint32_t ilvBytes, uint32_t planeBytes);

    __aicore__ inline void InitBuffers(GM_ADDR x, GM_ADDR y, GM_ADDR A);

    /* BuildXyIlv 的 ready 前置条件（与其内部判定逐字一致），供 Init 预发装载用。 */
    __aicore__ inline bool XyIlvFeasibleUnit();

    /* 连发 x/y 两条 GM 装载进 VECIN 队列，让传输延迟与建表的 V 工作重叠。 */
    __aicore__ inline void IssueXyPreload(bool feasible);

    __aicore__ inline void ProcessLowerBalancedGroups();

    __aicore__ inline void ProcessFullLower();

    __aicore__ inline void ReleaseHeldXY();

    __aicore__ inline void Process();

    template <int Batch>
    __aicore__ inline void ProcessFullLowerWindow();

private:
    template <bool kAlphaOne>
    __aicore__ inline void ProcessColumn(int32_t j);

    __aicore__ inline bool TryProcessLowerPacked(int32_t gStep);

    template <bool kAlphaOne>
    __aicore__ inline void ProcessColumns();

    template <bool kAlphaOne>
    __aicore__ inline void ProcessWindowColumns(int32_t row0, int32_t rowEnd);

    template <bool kAlphaOne>
    __aicore__ inline void ProcessWindowedColumns();

    /* 系数预计算：t1 = alpha*y、t2 = alpha*x 整段向量算好落 UB。
     * t1 占 xyPlaneBuf（此时已无平面用途），t2 占 idxBuf（建表/Gather 已结束）。
     * t1[2j]=ar*yr-ai*yi、t1[2j+1]=ar*yi+ai*yr，借 xs/ys=interleave(-im,re) 一乘一加完成。 */
    __aicore__ inline void PrecomputeCoefIlv();

    __aicore__ inline void LoadCoef(int32_t j, float* t1r, float* t1i, float* t2r, float* t2i);

    /* 整窗 LOWER：本核相邻 winGroupB_ 列打包，对角上方前缀掩码置零，一次 CopyOutCols 写出。 */
    template <int Batch>
    __aicore__ inline void ComputeLowerWindowBatch(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, AscendC::LocalTensor<float> phaseCoef1,
        AscendC::LocalTensor<float> phaseCoef2, int32_t pitchFloats, int32_t j0, int32_t b);

    __aicore__ inline void ApplyLowerPrefixMask(AscendC::LocalTensor<float> aOut, uint64_t zmask, int32_t pitchFloats);

    template <int Batch>
    __aicore__ inline void MaskLowerWindowPrefix(
        AscendC::LocalTensor<float> aOut, int32_t j0, int32_t row0, int32_t pitchFloats, int32_t b);

    template <bool Window>
    __aicore__ inline int64_t LowerColumnFloatOffset(int64_t base, int32_t row, int32_t kk, int32_t lda);

    template <int Batch, bool Window, bool Rotated = false>
    __aicore__ inline void StoreLowerBatch(
        AscendC::LocalTensor<float> storage, uint32_t rotation, int64_t base, int32_t row, int32_t lda, int32_t cur,
        int32_t pitchFloats, int32_t b);

    template <int Batch>
    __aicore__ inline void ProcessLowerWindowGroup();

    __aicore__ inline __attribute__((always_inline)) void WriteUpperSingleColumn(
        int32_t j, int32_t lda, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys);

    __aicore__ inline __attribute__((always_inline)) void WriteUpperColumnPair(
        int32_t j, int32_t lda, uint32_t secondOff, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys);

    /* UPPER 单 tile：四列共用一个 VECOUT tensor，op-major 交错四条独立链，写回逐列连续。 */
    template <int Batch>
    __aicore__ inline void LoadUpperBatchCoefficients(
        int32_t j, float (&c1r)[Batch], float (&c1i)[Batch], float (&c2r)[Batch], float (&c2i)[Batch],
        uint32_t (&n2)[Batch]);

    template <bool Add>
    __aicore__ inline void ComputeUpperBatchElement(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, float coef, uint32_t n2);

    template <int Batch, bool Add>
    __aicore__ inline void ComputeUpperBatchLayer(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[Batch],
        const uint32_t (&n2)[Batch], uint32_t segOff);

    template <int Batch>
    __aicore__ inline void WriteUpperBatchColumns(
        int32_t j, int32_t lda, uint32_t segOff, AscendC::LocalTensor<float> aOut, const uint32_t (&n2)[Batch]);

    template <int Batch>
    __aicore__ inline void WriteUpperBatch(
        int32_t j, int32_t lda, uint32_t segOff, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys);

    __aicore__ inline void ProcessUpperSingleTileQuad();

    /* UPPER 单 tile：两个本核列共用一个 VECOUT tensor，写回仍按列发起。 */
    __aicore__ inline void ProcessUpperSingleTilePair();

    __aicore__ inline void ProcessUpperSingleTile();
    /* 合并表：交织位置 2k 取 aRe[k]，2k+1 取 aIm[k]。aIm 起点是 planeFloats_。
     * Gather 的偏移量单位是字节（见 cgerc_host.cpp CreateCgercOffset）。 */
    __aicore__ inline void BuildSwapTable();

    __aicore__ inline __attribute__((always_inline)) void GatherRotatedComplex(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src, AscendC::LocalTensor<uint32_t> table,
        uint32_t n2, uint8_t repeats);

    __aicore__ inline void BuildXYDirectSwap();

    __aicore__ inline void BuildInterleaveTable();

    __aicore__ inline void CopyInFloats(
        AscendC::LocalTensor<float> dst, int64_t gmFloatOff, int32_t floats, AscendC::GlobalTensor<float> src);

    __aicore__ inline void CopyOutFloats(int64_t gmFloatOff, AscendC::LocalTensor<float> src, int32_t floats);

    /* MTE2 不能直写 VECCALC。经 VECIN 队列再 Adds 过去。 */
    __aicore__ inline void CopyInViaQue(
        AscendC::LocalTensor<float> dst, int64_t gmFloatOff, int32_t floats, AscendC::GlobalTensor<float> src);

    __aicore__ inline __attribute__((always_inline)) void PrepareInterleavedInput(
        AscendC::LocalTensor<float>& ilv, AscendC::LocalTensor<float>& held, AscendC::GlobalTensor<float> gm,
        int32_t nIlv);

    /* X/Y 保持 GM 交织；Xs/Ys = interleave(-im, re)，复用合并表。每核一次。 */
    __aicore__ inline void PrepareXyIlv(int32_t ilvRows, int32_t nIlv, int32_t xyPitch);

    __aicore__ inline void BuildXyIlv();

    /* 交织 → 两平面。pattern 1 取偶（实），2 取奇（虚）。 */
    __aicore__ inline void SplitPlanes(
        AscendC::LocalTensor<float> ilv, AscendC::LocalTensor<float> re, AscendC::LocalTensor<float> im, int32_t cur);

    __aicore__ inline void LoadXYWindow(int32_t i0, int32_t cur);

    __aicore__ inline void LoadXYInterleavedWindow(int32_t i0, int32_t cur);

    __aicore__ inline bool IsInfValue(float value) const;

    __aicore__ inline void ComputeGeneralCoefValues(
        float xr, float xi, float yr, float yi, float* t1r, float* t1i, float* t2r, float* t2i);

    template <bool kAlphaOne>
    __aicore__ inline void ComputeCoefValues(
        float xr, float xi, float yr, float yi, float* t1r, float* t1i, float* t2r, float* t2i);

    template <bool kAlphaOne, bool kXyKnownReady = false>
    __aicore__ inline void ComputeCoef(int32_t j, float* t1r, float* t1i, float* t2r, float* t2i);

    /* GM->UB: srcStride 字节，dstStride 32B。UB->GM 相反。dstStride=0 时每块在 UB 自动垫到 32B。 */
    __aicore__ inline void CopyInCols(AscendC::LocalTensor<float> dst, int64_t gmComplexOff, int32_t cur, int32_t b);

    __aicore__ inline void CopyOutCols(int64_t gmComplexOff, AscendC::LocalTensor<float> src, int32_t cur, int32_t b);

    template <bool UseY, int Component>
    __aicore__ inline float UnitAlphaCoefficientBits(int32_t j);

    template <int Batch>
    __aicore__ inline void ComputeLowerPackedCount(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, AscendC::LocalTensor<float> phaseCoef1,
        AscendC::LocalTensor<float> phaseCoef2, int32_t pitchFloats, int32_t j0, int32_t b);

    template <int Batch>
    __aicore__ inline uint64_t BuildLowerShortMask(int32_t j0, int32_t i0, int32_t cur, int32_t b);

    template <int Batch>
    __aicore__ inline void ZeroLongLowerPrefixes(
        AscendC::LocalTensor<float> aOut, int32_t j0, int32_t i0, int32_t cur, int32_t pitchFloats, int32_t b);

    template <int Batch>
    __aicore__ inline void ProcessLowerPackedGeneric(int32_t j0, int32_t b);

    template <int Batch>
    __aicore__ inline void LoadLowerBatchCoefficients(
        int32_t j0, int32_t b, float (&c1r)[Batch], float (&c1i)[Batch], float (&c2r)[Batch], float (&c2i)[Batch]);

    template <int Batch>
    __aicore__ inline void ComputeLowerPackedAxpyLayer(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[Batch],
        int32_t pitchFloats, int32_t b, uint32_t n2);

    template <int Batch>
    __aicore__ inline void ComputeLowerPackedFull(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, const float (&c1r)[Batch],
        const float (&c1i)[Batch], const float (&c2r)[Batch], const float (&c2i)[Batch], int32_t pitchFloats,
        int32_t b);

    template <int Batch>
    __aicore__ inline void MaskLowerFullPrefix(
        AscendC::LocalTensor<float> aOut, int32_t j0, int32_t i0, int32_t cur, int32_t pitchFloats, int32_t b);

    template <int Batch>
    __aicore__ inline void ProcessLowerPackedFull(int32_t j0, int32_t b);

    template <int Batch, bool kCompleteGroup = false>
    __aicore__ inline void ProcessLowerPackedOut(int32_t j0, int32_t b);

    template <bool kAlphaOne>
    __aicore__ inline void ProcessGroup(int32_t j0, int32_t b);

    template <bool kAlphaOne>
    __aicore__ inline void LoadPackedUpperCoefficients(
        int32_t j0, int32_t b, int32_t i0, int32_t cur, float (&c1r)[8], float (&c1i)[8], float (&c2r)[8],
        float (&c2i)[8], uint32_t (&nv)[8]);

    __aicore__ inline void ComputePackedUpperLayer(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> src, const float (&coef)[8],
        const uint32_t (&nv)[8], int32_t b, int32_t pitchFloats);

    template <bool kAlphaOne>
    __aicore__ inline void PackedUpperChunk(
        int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff, int32_t pitchFloats, uint32_t nAll,
        AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv, AscendC::LocalTensor<float> xs,
        AscendC::LocalTensor<float> ys);

    template <bool kAlphaOne>
    __aicore__ inline void PackedLowerChunk(
        int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff, int32_t pitchFloats, uint32_t nAll,
        AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv, AscendC::LocalTensor<float> xs,
        AscendC::LocalTensor<float> ys);

    template <bool kAlphaOne>
    __aicore__ inline void PackedChunk(int32_t j0, int32_t b, int32_t i0, int32_t cur, int64_t gmComplexOff);

    template <AscendC::HardEvent Event>
    __aicore__ inline __attribute__((always_inline)) void SyncEvent();

    /* aRe += xr*t1r - xi*t1i + yr*t2r - yi*t2i
     * aIm += xr*t1i + xi*t1r + yr*t2i + yi*t2r */
    __aicore__ inline void Accumulate(
        AscendC::LocalTensor<float> aRe, AscendC::LocalTensor<float> aIm, int32_t cur, float t1r, float t1i, float t2r,
        float t2i, int32_t xyOff);

    __aicore__ inline void FinishInterleavedChunk(
        AscendC::LocalTensor<float> aOut, AscendC::LocalTensor<float> xIlv, AscendC::LocalTensor<float> yIlv,
        AscendC::LocalTensor<float> xs, AscendC::LocalTensor<float> ys, uint32_t n2, int64_t gmComplexOff, int32_t cur,
        float t1r, float t1i, float t2r, float t2i, int32_t prefix);

    __aicore__ inline void WindowChunk(
        int64_t gmComplexOff, int32_t windowRowOff, int32_t cur, float t1r, float t1i, float t2r, float t2i,
        int32_t prefix);

    __aicore__ inline void FinishPlanarChunk(
        AscendC::LocalTensor<float> aPlanes, AscendC::LocalTensor<float> aRe, AscendC::LocalTensor<float> aIm,
        int32_t i0, int32_t cur, int64_t gmComplexOff, float t1r, float t1i, float t2r, float t2i);

    __aicore__ inline void ChunkStep(
        int64_t gmComplexOff, int32_t i0, int32_t cur, float t1r, float t1i, float t2r, float t2i, int32_t prefix);

    bool skip_{true};
    Csyr2TilingData td_{};
    int32_t jBegin_{0};
    int32_t jEnd_{0};
    int32_t step_{1};
    int32_t tileRow_{1};
    int32_t ilvFloats_{0};
    int32_t planeFloats_{0};
    int32_t packB_{1};
    int32_t bufNum_{2};
    int32_t unionLen_{0};
    bool xyReuse_{false};
    bool xyIlvReady_{false};
    bool upperPair_{false};
    bool upperQuad_{false};
    bool upperOct_{false};
    bool upperPackedHeld_{false};
    bool lowerPackedReady_{false};
    bool coefReady_{false};
    bool xyPreload_{false};
    bool lowerHeld_{false};
    int32_t winGroupB_{1};
    int32_t xyIlvRows_{0};
    int32_t xyIlvPitch_{0};
    static constexpr bool kRotateLower = TILING_KEY == CSYR2_TILING_KEY_LOWER_UNIT && kFullMode == 2 && kUnitAlpha;
    AscendC::TPipe pipe_;
    AscendC::LocalTensor<float> xIlvHeld_;
    AscendC::LocalTensor<float> yIlvHeld_;
    AscendC::TQue<AscendC::QuePosition::VECIN, CSYR2_BUFFER_NUM> aInQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, CSYR2_BUFFER_NUM> aOutQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> layoutPad_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aPlaneBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xyPlaneBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xyStageBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tilingBuf_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    AscendC::GlobalTensor<float> aGm_;
};

#endif
