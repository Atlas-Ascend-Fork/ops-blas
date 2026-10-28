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

#include "csyr2_ascend_kernel.h"

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, int32_t incx, int32_t incy,
    uint32_t usedCoreNum, uint32_t tileRow, uint32_t xyReuse, float alphaReal, float alphaImag)
{
    // Retain the original UB layout while launch arguments own the metadata.
    pipe_.InitBuffer(tilingBuf_, 64);
    td_.n = n;
    td_.lda = lda;
    td_.incx = incx;
    td_.incy = incy;
    td_.uplo = kUpper ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    td_.alphaReal = alphaReal;
    td_.alphaImag = alphaImag;
    td_.usedCoreNum = usedCoreNum;
    td_.tileRow = tileRow;
    td_.bufferNum = CSYR2_BUFFER_NUM;
    td_.xyReuse = xyReuse;
    InitBuffers(x, y, A);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitDirectStorage()
{
    const uint32_t ilvBytes = static_cast<uint32_t>(ilvFloats_) * sizeof(float);
    const uint32_t planeBytes = static_cast<uint32_t>(planeFloats_) * sizeof(float);
    // Rotated LOWER repeats may read up to 127 padded floats beyond a logical X/Y slice.
    pipe_.InitBuffer(aInQue_, 2, ilvBytes + (kUpper ? 0u : (kRotateLower ? 768u : 256u)));
    if constexpr (kUpper)
        pipe_.InitBuffer(layoutPad_, 256u);
    pipe_.InitBuffer(
        aOutQue_, 2,
        ilvBytes * (kUpper ? (upperOct_ ? 8u : 4u) : static_cast<uint32_t>(packB_)) + (kRotateLower ? 512u : 0u));
    if constexpr (kUpper) {
        pipe_.InitBuffer(xyPlaneBuf_, planeBytes * 4u + 768u);
    } else if constexpr (kUnitAlpha) {
        pipe_.InitBuffer(layoutPad_, planeBytes * 6u + ilvBytes + 768u - (kRotateLower ? 1024u : 0u));
    } else {
        pipe_.InitBuffer(aPlaneBuf_, planeBytes * 2u);
        pipe_.InitBuffer(xyPlaneBuf_, planeBytes * 4u + 256u);
        pipe_.InitBuffer(xyStageBuf_, ilvBytes + 512u);
    }
    const int32_t tmpFloats = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
    pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(tmpFloats) * sizeof(float) + 768u);
    pipe_.InitBuffer(idxBuf_, ilvBytes);
    IssueXyPreload(true);
    BuildSwapTable();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitDirect(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, int32_t n, int32_t lda, uint32_t tileRow, float alphaReal, float alphaImag)
{
    // Direct arguments avoid loading tiling metadata from global memory.
    pipe_.InitBuffer(tilingBuf_, 64);
    td_.n = n;
    td_.lda = lda;
    td_.incx = 1;
    td_.incy = 1;
    td_.uplo = kUpper ? ACLBLAS_UPPER : ACLBLAS_LOWER;
    td_.alphaReal = alphaReal;
    td_.alphaImag = alphaImag;
    td_.usedCoreNum = AscendC::GetBlockNum();
    td_.tileRow = tileRow;
    td_.bufferNum = CSYR2_BUFFER_NUM;
    td_.xyReuse = 1;
    skip_ = false;
    jBegin_ = static_cast<int32_t>(AscendC::GetBlockIdx());
    jEnd_ = n;
    step_ = static_cast<int32_t>(td_.usedCoreNum);
    tileRow_ = static_cast<int32_t>(tileRow);
    ilvFloats_ = Csyr2CeilAlign < kRotateLower ? 128 : CSYR2_VEC_FLOATS > (tileRow_ * 2);
    planeFloats_ = Csyr2CeilAlign<CSYR2_VEC_FLOATS>(tileRow_);
    packB_ = kUpper ? 1 : 4;
    bufNum_ = 2;
    lowerPackedReady_ = !kUpper;
    xyReuse_ = true;
    unionLen_ = kUpper ? n : n - jBegin_;
    upperPair_ = kUpper;
    upperQuad_ = kUpper;
    upperOct_ = kUpper && n <= 1024;
    upperPackedHeld_ = false;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(n) * 2);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(n) * 2);
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A), static_cast<uint64_t>(n) * lda * 2);
    InitDirectStorage();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitVectorViews(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, uint32_t blockIdx)
{
    const int64_t absIncx = (td_.incx < 0) ? -static_cast<int64_t>(td_.incx) : td_.incx;
    const int64_t absIncy = (td_.incy < 0) ? -static_cast<int64_t>(td_.incy) : td_.incy;
    const uint64_t xFloats = static_cast<uint64_t>((td_.n - 1) * absIncx + 1) * 2ULL;
    const uint64_t yFloats = static_cast<uint64_t>((td_.n - 1) * absIncy + 1) * 2ULL;
    const uint64_t aFloats = static_cast<uint64_t>(td_.lda) * static_cast<uint64_t>(td_.n) * 2ULL;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), xFloats);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), yFloats);
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A), aFloats);
    jBegin_ = static_cast<int32_t>(blockIdx);
    jEnd_ = td_.n;
    step_ = static_cast<int32_t>(td_.usedCoreNum);
    if (step_ < 1) {
        step_ = 1;
    }

    tileRow_ = (td_.tileRow > 0) ? static_cast<int32_t>(td_.tileRow) : 1;
    if (tileRow_ > td_.n) {
        /* 钳到 4 对齐而非 n 本身：微矩阵 host 会抬 tileRow 到 4 对齐以启用 ready 快路。 */
        const int32_t alignedN = ((td_.n + 3) / 4) * 4;
        if constexpr (kUpper || !kUnitPos || kWindowed) {
            if (tileRow_ > alignedN) {
                tileRow_ = alignedN;
            }
        }
    }
    if (tileRow_ < 1) {
        tileRow_ = 1;
    }
    /* 交织缓冲按 64 float 对齐：GatherMask 整 repeat 读，尾块会多读到对齐边界。 */
    ilvFloats_ = Csyr2CeilAlign<CSYR2_VEC_FLOATS>(tileRow_ * 2);
    planeFloats_ = Csyr2CeilAlign<CSYR2_VEC_FLOATS>(tileRow_);
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::SelectLowerPacking()
{
    packB_ = 1;
    bufNum_ = static_cast<int32_t>(CSYR2_BUFFER_NUM);
    lowerPackedReady_ = false;
    /* UPPER 打包组实测慢于成对路径（依赖链交错 + 逐列连续写），unit 单 tile 恒 packB=1，
     * 由 upperPair_ 的容量判定接管。 */
    if constexpr (!kUpper && kUnitPos && !kWindowed) {
        const int32_t nIlv = td_.n * 2;
        const int32_t xyPitch = Csyr2CeilAlign<8>(nIlv);
        const int32_t tmpF = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
        lowerPackedReady_ = (td_.xyReuse != 0) && td_.n >= 64 && (tileRow_ & 3) == 0 && td_.n <= tileRow_ &&
                            td_.n <= planeFloats_ && nIlv <= ilvFloats_ && nIlv <= planeFloats_ * 2 &&
                            xyPitch + nIlv <= tmpF;
        if (lowerPackedReady_) {
            const uint32_t ilvB = static_cast<uint32_t>(ilvFloats_) * sizeof(float);
            const uint32_t planeB = static_cast<uint32_t>(planeFloats_) * sizeof(float);
            constexpr uint32_t kUbBytes = 192u * 1024u;
            const uint32_t known = CSYR2_UB_FIXED_BYTES + planeB * 2u + planeB * 4u + ilvB +
                                   static_cast<uint32_t>(tmpF) * sizeof(float) + ilvB;
            const uint32_t remain = (kUbBytes > known) ? (kUbBytes - known) : 0u;
            for (int32_t candB = 4; candB >= 2; --candB) {
                const uint32_t needDb = 2u * (1u + static_cast<uint32_t>(candB)) * ilvB;
                if (needDb <= remain) {
                    packB_ = candB;
                    bufNum_ = 2;
                    break;
                }
            }
        }
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::SelectUpperReuse()
{
    if constexpr (kUpper) {
        unionLen_ = jEnd_;
    } else {
        unionLen_ = td_.n - jBegin_;
    }
    if (unionLen_ < 0) {
        unionLen_ = 0;
    }
    xyReuse_ = (td_.xyReuse != 0) && (unionLen_ <= planeFloats_) && (unionLen_ * 2 <= ilvFloats_);
    upperPair_ = false;
    upperPackedHeld_ = false;
    if constexpr (kUpper && kUnitPos) {
        /* 成对路径布局：无 aPlane、xyStage 仅错位，total = FIXED + ilvB*7 + planeB*4 + tmpB。 */
        const uint32_t ilvB = static_cast<uint32_t>(ilvFloats_) * sizeof(float);
        const uint32_t planeB = static_cast<uint32_t>(planeFloats_) * sizeof(float);
        const int32_t tmpF = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
        const uint32_t pairTotal =
            CSYR2_UB_FIXED_BYTES + ilvB * 7u + planeB * 4u + static_cast<uint32_t>(tmpF) * sizeof(float);
        upperPair_ = (pairTotal <= 192u * 1024u) && (packB_ == 1) && (bufNum_ == 2) && xyReuse_ &&
                     (tileRow_ >= jEnd_) && ((tileRow_ & 3) == 0);
        const uint32_t quadTotal =
            CSYR2_UB_FIXED_BYTES + ilvB * 11u + planeB * 4u + static_cast<uint32_t>(tmpF) * sizeof(float);
        upperQuad_ = upperPair_ && (quadTotal <= 192u * 1024u);
        upperOct_ = false; /* 8 列交错实测负收益（512 段劣化），保持 quad 上限 */
        /* 打包 UPPER 路径后续不再使用 VECIN 队列。双槽可让 x/y
         * 各占一个物理槽常驻，省掉 CopyInViaQue 的两次 UB 内 Adds。 */
        upperPackedHeld_ = (packB_ >= 2) && (bufNum_ == 2) && xyReuse_ && (tileRow_ >= jEnd_) && ((tileRow_ & 3) == 0);
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::SelectWindowBatch(
    uint32_t ilvBytes, uint32_t planeBytes)
{
    winGroupB_ = Csyr2SelectWindowBatchCount(td_.n, tileRow_, ilvBytes, planeBytes, kUpper);
    if constexpr (kUpper) {
        const uint32_t quadTotal = CSYR2_WINDOW_UB_FIXED_BYTES + ilvBytes * 13u + planeBytes * 2u;
        upperQuad_ = (tileRow_ >= td_.n) && (quadTotal <= 192u * 1024u);
        const uint32_t octTotal = CSYR2_WINDOW_UB_FIXED_BYTES + ilvBytes * 21u + planeBytes * 2u;
        upperOct_ = (kFullMode == 1) && td_.n >= 640 && tileRow_ >= td_.n && octTotal <= 192u * 1024u;
    }
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitWindowStorage(
    uint32_t ilvBytes, uint32_t planeBytes)
{
    SelectWindowBatch(ilvBytes, planeBytes);
    pipe_.InitBuffer(aInQue_, 2u, ilvBytes);
    uint32_t outBatch = kUpper ? (upperOct_ ? 8u : (upperQuad_ ? 4u : 2u)) : 1u;
    if (winGroupB_ > static_cast<int32_t>(outBatch)) {
        outBatch = static_cast<uint32_t>(winGroupB_);
    }
    if constexpr (kUpper) {
        if (td_.n >= 640)
            pipe_.InitBuffer(layoutPad_, 256u);
    }
    pipe_.InitBuffer(aOutQue_, 2u, ilvBytes * outBatch);
    pipe_.InitBuffer(xyPlaneBuf_, planeBytes * 2u + 256u);
    pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(ilvFloats_ * 2) * sizeof(float) + 768u);
    pipe_.InitBuffer(idxBuf_, ilvBytes);
    IssueXyPreload(tileRow_ >= td_.n);
    if constexpr (kFullMode == 1)
        BuildSwapTable();
    else
        BuildInterleaveTable();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitUnitStorage(
    uint32_t ilvBytes, uint32_t planeBytes)
{
    const uint32_t outBatch = upperOct_ ? 8u : (upperQuad_ ? 4u : (upperPair_ ? 2u : static_cast<uint32_t>(packB_)));
    /* UPPER 打包路径 aInQue 只承载常驻 x/y（各一槽）与 BuildXyIlv 暂存，槽宽 ilvB 即可。 */
    const uint32_t inBatch = (kUpper || lowerPackedReady_) ? 1u : static_cast<uint32_t>(packB_);
    pipe_.InitBuffer(aInQue_, static_cast<uint32_t>(bufNum_), ilvBytes * inBatch);
    if constexpr (kUpper) {
        if (td_.n >= 640)
            pipe_.InitBuffer(layoutPad_, 256u);
    }
    pipe_.InitBuffer(aOutQue_, static_cast<uint32_t>(bufNum_), ilvBytes * outBatch);
    if (!upperPair_) {
        pipe_.InitBuffer(aPlaneBuf_, planeBytes * 2); /* aRe | aIm 连续，合并表按此步长建 */
    }
    pipe_.InitBuffer(xyPlaneBuf_, planeBytes * 4 + 256u); /* xRe | xIm | yRe | yIm */
    /* 成对热路只保留 512B 错开，使 tmpBuf_ 及后续缓冲基址与原布局一致。 */
    pipe_.InitBuffer(xyStageBuf_, (upperPair_ ? 0u : ilvBytes) + 512u); /* x / y 交织落地 */
    /* tmpBuf_ 两种用法：建表要 2*ilvFloats_ 个 int32，计算要 3 个平面。按大的开。 */
    const int32_t tmpFloats = (ilvFloats_ * 2 > planeFloats_ * 3) ? (ilvFloats_ * 2) : (planeFloats_ * 3);
    pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(tmpFloats) * sizeof(float) + 768u);
    pipe_.InitBuffer(idxBuf_, ilvBytes); /* 合并偏移表，常驻 */

    IssueXyPreload(XyIlvFeasibleUnit());
    if constexpr (kFullMode == 1)
        BuildSwapTable();
    else
        BuildInterleaveTable();
}

template <uint32_t TILING_KEY, uint32_t kFullMode, bool kUnitAlpha>
__aicore__ inline void Csyr2AscendKernel<TILING_KEY, kFullMode, kUnitAlpha>::InitBuffers(
    GM_ADDR x, GM_ADDR y, GM_ADDR A)
{
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= td_.usedCoreNum) {
        skip_ = true;
        return;
    }
    skip_ = false;
    InitVectorViews(x, y, A, blockIdx);
    SelectLowerPacking();
    SelectUpperReuse();

    const uint32_t ilvBytes = static_cast<uint32_t>(ilvFloats_) * sizeof(float);
    const uint32_t planeBytes = static_cast<uint32_t>(planeFloats_) * sizeof(float);
    if constexpr (kWindowed) {
        /* 整窗覆盖时按列成组写出：total(b) = WINDOW_FIXED + ilvB*(5+2b) + planeB*2。
         * UPPER 组内零填充逐列递增可到 8；LOWER 对角前缀掩码限制 b<=4。 */
        InitWindowStorage(ilvBytes, planeBytes);
        return;
    }
    InitUnitStorage(ilvBytes, planeBytes);
}
