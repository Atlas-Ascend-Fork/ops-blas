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
 * \file ctrsm_batched_kernel_aiv_canon_b_deinterleave.h
 * \brief B 矩阵解交织器：把用户 B 转成补零对齐的 SoA 工作区。
 *        列分块路径：单行超出 UB，逐行逐列块加载解交织写出
 */
    __aicore__ inline void DeinterleaveBLeftChunked(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gBcR, AscendC::GlobalTensor<float>& gBcI,
        AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ub,
        int32_t rStart, int32_t rEnd, int32_t deintCount,
        int32_t colOff, int32_t nColsAligned)
    {
        int32_t maxChunkCols = 256;
        AscendC::LocalTensor<float> ubI = cfg_->bufPanelB_imag->Get<float>();
        for (int32_t i = rStart; i < rEnd; i++) {
            int64_t rowGmBase = (int64_t)i * cfg_->td->ldb * 2 + colOff * 2;
            for (int32_t cs = 0; cs < deintCount; cs += maxChunkCols) {
                int32_t chunkCols = (maxChunkCols < deintCount - cs) ? maxChunkCols : (deintCount - cs);
                int32_t aosFloats = chunkCols * 2;
                AscendC::DataCopyExtParams rChunk(1, (uint32_t)(aosFloats * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(ubSrc, gB[rowGmBase + cs * 2], rChunk, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                cvt_->DeinterleaveRow(ubSrc, ub, chunkCols, 1.0f);
                AscendC::DataCopyPad(gBcR[(int64_t)i * nColsAligned + cs], ub,
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0});
                AscendC::DataCopyPad(gBcI[(int64_t)i * nColsAligned + cs], ubI,
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // 左侧模式下拆分B矩阵AoS为SoA（批量DMA加载+Gather解交织，按buildRow范围分行）
    // 当单行AoS数据超过UB容量时，按列分块处理
    // split 模式下 nColsOrig = localNCols, gmB 已偏移到列起始位置
    __aicore__ inline void DeinterleaveBLeft(__gm__ float* gmB,
                                              __gm__ float* gmBcR, __gm__ float* gmBcI)
    {
        int32_t kDim = cfg_->kDim, nColsAligned = cfg_->nColsAligned;
        int32_t kDimOrig = cfg_->kDimOrig, nColsOrig = cfg_->nColsOrig;
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::GlobalTensor<float> gB, gBcR, gBcI;
        int32_t mOrig = cfg_->td->m;
        int32_t nRead = nColsOrig;
        int32_t colOff = cfg_->nColsOffset;
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)mOrig * cfg_->td->ldb * 2));
        gBcR.SetGlobalBuffer(gmBcR, (uint32_t)((int64_t)kDim * nColsAligned));
        gBcI.SetGlobalBuffer(gmBcI, (uint32_t)((int64_t)kDim * nColsAligned));
        int32_t totalRows = (mOrig < kDimOrig) ? mOrig : kDimOrig;
        int32_t rStart = (cfg_->buildRowStart < totalRows) ? cfg_->buildRowStart : totalRows;
        int32_t rEnd = (cfg_->buildRowEnd < totalRows) ? cfg_->buildRowEnd : totalRows;
        int32_t srcRowFloats = nRead * 2;
        int32_t srcRowUBFloats = CEIL_ALIGN(srcRowFloats * (int32_t)sizeof(float), 32)
                                 / (int32_t)sizeof(float);
        int32_t deintCount = nRead;

        if (srcRowUBFloats <= cfg_->srcBufFloats) {
            int32_t batchRows = cfg_->srcBufFloats / srcRowUBFloats;
            if (batchRows > (rEnd - rStart)) batchRows = rEnd - rStart;
            if (batchRows < 1) batchRows = 1;
            uint32_t srcStrideBytes = (uint32_t)((cfg_->td->ldb * 2 - nRead * 2) * (int32_t)sizeof(float));
            for (int32_t bStart = rStart; bStart < rEnd; bStart += batchRows) {
                int32_t bCount = (batchRows < rEnd - bStart) ? batchRows : (rEnd - bStart);
                DeinterleaveBLeftBatch(gB, gBcR, gBcI, ubSrc, ub,
                    bStart, bCount, srcRowFloats, srcRowUBFloats, srcStrideBytes,
                    deintCount, nColsAligned);
            }
        } else {
            DeinterleaveBLeftChunked(gB, gBcR, gBcI, ubSrc, ub,
                rStart, rEnd, deintCount, colOff, nColsAligned);
        }
    }

    // DeinterleaveBLeft 内层：搬入一批行到UB，逐行解交织并写出到 gmBc
    __aicore__ inline void DeinterleaveBLeftBatch(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gBcR, AscendC::GlobalTensor<float>& gBcI,
        AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ub,
        int32_t bStart, int32_t bCount,
        int32_t srcRowFloats, int32_t srcRowUBFloats, uint32_t srcStrideBytes,
        int32_t deintCount, int32_t nColsAligned)
    {
        if (bCount > 1) {
            AscendC::DataCopyExtParams rBatch((uint16_t)bCount,
                (uint32_t)(srcRowFloats * sizeof(float)), srcStrideBytes, 0, 0);
            AscendC::DataCopyPad(ubSrc, gB[(int64_t)bStart * cfg_->td->ldb * 2 + cfg_->nColsOffset * 2],
                rBatch, {false, 0, 0, 0});
        } else {
            AscendC::DataCopyExtParams rRow(1,
                (uint32_t)(srcRowFloats * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(ubSrc, gB[(int64_t)bStart * cfg_->td->ldb * 2 + cfg_->nColsOffset * 2],
                rRow, {false, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        for (int32_t j = 0; j < bCount; j++) {
            int32_t i = bStart + j;
            AscendC::LocalTensor<float> rowSrc = ubSrc[j * srcRowUBFloats];
            AscendC::LocalTensor<float> ubI = cfg_->bufPanelB_imag->Get<float>();
            if (j > 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            }
            cvt_->DeinterleaveRow(rowSrc, ub, deintCount, 1.0f);
            AscendC::DataCopyPad(gBcR[(int64_t)i * nColsAligned], ub,
                {1, (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0});
            AscendC::DataCopyPad(gBcI[(int64_t)i * nColsAligned], ubI,
                {1, (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }

    // 右侧模式下拆分B矩阵AoS为SoA：解交织与转置融合为单 pass（省掉 temp GM 中转）
    __aicore__ inline void DeinterleaveBRight(__gm__ float* gmB,
                                               __gm__ float* gmBcR, __gm__ float* gmBcI,
                                               __gm__ float* gmTemp)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig, nColsOrig = cfg_->nColsOrig;
        int32_t mOrig = cfg_->td->m, nOrig = cfg_->td->n;
        int32_t nColsAligned = cfg_->nColsAligned;
        int32_t rEnd = (mOrig < nColsOrig) ? mOrig : nColsOrig;
        int32_t deintCount = (nOrig < kDimOrig) ? nOrig : kDimOrig;
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ubDst = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::GlobalTensor<float> gB, gBcR, gBcI;
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)mOrig * cfg_->td->ldb * 2));
        gBcR.SetGlobalBuffer(gmBcR, (uint32_t)((int64_t)kDim * nColsAligned));
        gBcI.SetGlobalBuffer(gmBcI, (uint32_t)((int64_t)kDim * nColsAligned));
        constexpr int32_t TS = CtrsmAivCfg::TS;
        AscendC::DataCopyPadExtParams<float> padP{false, 0, 0, 0};
        int32_t ldb2 = cfg_->td->ldb * 2;
        for (int32_t cb = 0; cb < deintCount; cb += TS) {
            int32_t tc = (TS < deintCount - cb) ? TS : (deintCount - cb);
            int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
            int32_t aosColFloats = tc * 2;
            int32_t rowUB = CEIL_ALIGN(aosColFloats, FLOAT_ALIGN);
            for (int32_t rb = 0; rb < rEnd; rb += 16) {
                int32_t tr = (16 < rEnd - rb) ? 16 : (rEnd - rb);
                DeinterleaveBRightTile(gB, gBcR, gBcI, ubSrc, ub, ubDst, padP,
                    ldb2, rb, tr, cb, tc, tcA, aosColFloats, rowUB, nColsAligned);
            }
        }
    }

    // 单 tile：加载 tr 行的列子块 AoS → Gather 解交织 → TransposeTileBlock 写 gmBc
    __aicore__ inline void DeinterleaveBRightTile(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gBcR, AscendC::GlobalTensor<float>& gBcI,
        AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ub,
        AscendC::LocalTensor<float>& ubDst,
        AscendC::DataCopyPadExtParams<float>& padP,
        int32_t ldb2, int32_t rb, int32_t tr, int32_t cb,
        int32_t tc, int32_t tcA, int32_t aosColFloats, int32_t rowUB, int32_t nColsAligned)
    {
        if (tr > 1) {
            AscendC::DataCopyExtParams rp((uint16_t)tr, (uint32_t)(aosColFloats * sizeof(float)),
                (uint32_t)((ldb2 - aosColFloats) * sizeof(float)),
                (uint32_t)(((rowUB - aosColFloats) * sizeof(float)) / 32), 0);
            AscendC::DataCopyPad(ubSrc, gB[(int64_t)rb * ldb2 + cb * 2], rp, padP);
        } else {
            AscendC::DataCopyExtParams rp(1, (uint32_t)(aosColFloats * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(ubSrc, gB[(int64_t)rb * ldb2 + cb * 2], rp, padP);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<float> ubI = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<int32_t> offEven = cfg_->bufGatherEven->Get<int32_t>();
        AscendC::LocalTensor<int32_t> offOdd = cfg_->bufGatherOdd->Get<int32_t>();
        int32_t gatherChunk = cfg_->gatherChunk;
        for (int32_t j = 0; j < tr; j++) {
            AscendC::LocalTensor<float> rowAoS = ubSrc[j * rowUB];
            AscendC::LocalTensor<float> rowR = ub[j * tcA];
            for (int32_t c = 0; c < tcA; c += gatherChunk) {
                int32_t len = (gatherChunk < tcA - c) ? gatherChunk : (tcA - c);
                AscendC::Gather(rowR[c], rowAoS[c * 2], offEven.ReinterpretCast<uint32_t>(), 0, len);
            }
            AscendC::LocalTensor<float> rowI = ubI[j * tcA];
            for (int32_t c = 0; c < tcA; c += gatherChunk) {
                int32_t len = (gatherChunk < tcA - c) ? gatherChunk : (tcA - c);
                AscendC::Gather(rowI[c], rowAoS[c * 2], offOdd.ReinterpretCast<uint32_t>(), 0, len);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        TransposeTileBlock(ub, ubDst, gBcR, tcA, tr, tc, nColsAligned, cb, rb);
        TransposeTileBlock(ubI, ubDst, gBcI, tcA, tr, tc, nColsAligned, cb, rb);
    }

    // 阶段1：将gmB的[tBeg,tEnd)行解交织写入SoA临时区temp（dual-AIV 两核各写一段temp行）
    // 当单行AoS数据超过UB容量时，自动按列分块处理
    __aicore__ inline void DeinterleaveBRightToTemp(__gm__ float* gmB, __gm__ float* gmTemp,
                                                     int32_t tBeg, int32_t tEnd)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig, nColsOrig = cfg_->nColsOrig;
        int32_t mOrig = cfg_->td->m, nOrig = cfg_->td->n;
        int32_t rEnd = (tEnd < mOrig) ? tEnd : mOrig;
        if (rEnd > nColsOrig) rEnd = nColsOrig;
        if (tBeg < 0) tBeg = 0;
        if (tBeg >= rEnd) return;
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ubI = cfg_->bufPanelB_imag->Get<float>();
        __gm__ float* tempR = gmTemp;
        __gm__ float* tempI = gmTemp + (int64_t)nColsOrig * kDim;
        AscendC::GlobalTensor<float> gB, gTR, gTI;
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)mOrig * cfg_->td->ldb * 2));
        gTR.SetGlobalBuffer(tempR, (uint32_t)((int64_t)nColsOrig * kDim));
        gTI.SetGlobalBuffer(tempI, (uint32_t)((int64_t)nColsOrig * kDim));
        int32_t srcRowFloats = nOrig * 2;
        int32_t srcRowUBFloats = CEIL_ALIGN(srcRowFloats * (int32_t)sizeof(float), 32)
                                 / (int32_t)sizeof(float);
        int32_t deintCount = (nOrig < kDimOrig) ? nOrig : kDimOrig;

        if (srcRowUBFloats <= cfg_->srcBufFloats) {
            DeinterleaveBRightToTempFast(gB, gTR, gTI, ubSrc, ub, ubI,
                tBeg, rEnd, kDimOrig, kDim, deintCount, srcRowFloats, srcRowUBFloats);
        } else {
            DeinterleaveBRightToTempChunked(gB, gTR, gTI, ubSrc, ub, ubI,
                tBeg, rEnd, kDimOrig, kDim, deintCount);
        }
    }

    // 快速路径：整行装得下 UB，按行批量处理
    __aicore__ inline void DeinterleaveBRightToTempFast(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gTR, AscendC::GlobalTensor<float>& gTI,
        AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ub,
        AscendC::LocalTensor<float>& ubI,
        int32_t tBeg, int32_t rEnd,
        int32_t kDimOrig, int32_t kDim, int32_t deintCount,
        int32_t srcRowFloats, int32_t srcRowUBFloats)
    {
        int32_t nOrig = cfg_->td->n;
        if (srcRowUBFloats < 1) srcRowUBFloats = 1;
        int32_t batchRows = cfg_->srcBufFloats / srcRowUBFloats;
        if (batchRows > (rEnd - tBeg)) batchRows = rEnd - tBeg;
        if (batchRows < 1) batchRows = 1;
        uint32_t srcStrideBytes = (uint32_t)((cfg_->td->ldb * 2 - nOrig * 2) * (int32_t)sizeof(float));
        for (int32_t bStart = tBeg; bStart < rEnd; bStart += batchRows) {
            int32_t bCount = (batchRows < rEnd - bStart) ? batchRows : (rEnd - bStart);
            if (bCount > 1) {
                AscendC::DataCopyExtParams rBatch((uint16_t)bCount,
                    (uint32_t)(srcRowFloats * sizeof(float)), srcStrideBytes, 0, 0);
                AscendC::DataCopyPad(ubSrc, gB[(int64_t)bStart * cfg_->td->ldb * 2],
                    rBatch, {false, 0, 0, 0});
            } else {
                AscendC::DataCopyExtParams rRow(1, (uint32_t)(srcRowFloats * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(ubSrc, gB[(int64_t)bStart * cfg_->td->ldb * 2],
                    rRow, {false, 0, 0, 0});
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t j = 0; j < bCount; j++) {
                int32_t i = bStart + j;
                AscendC::LocalTensor<float> rowSrc = ubSrc[j * srcRowUBFloats];
                cvt_->DeinterleaveRow(rowSrc, ub, deintCount, 1.0f);
                AscendC::DataCopyPad(gTR[(int64_t)i * kDim], ub,
                    {1, (uint32_t)(kDimOrig * sizeof(float)), 0, 0, 0});
                AscendC::DataCopyPad(gTI[(int64_t)i * kDim], ubI,
                    {1, (uint32_t)(kDimOrig * sizeof(float)), 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // 列分块路径：单行超出 UB，按列分段加载、解交织、写出
    __aicore__ inline void DeinterleaveBRightToTempChunked(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gTR, AscendC::GlobalTensor<float>& gTI,
        AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ub,
        AscendC::LocalTensor<float>& ubI,
        int32_t tBeg, int32_t rEnd,
        int32_t kDimOrig, int32_t kDim, int32_t deintCount)
    {
        int32_t maxChunkCols = 256;
        for (int32_t i = tBeg; i < rEnd; i++) {
            int64_t rowGmBase = (int64_t)i * cfg_->td->ldb * 2;
            for (int32_t cs = 0; cs < deintCount; cs += maxChunkCols) {
                int32_t chunkCols = (maxChunkCols < deintCount - cs) ? maxChunkCols : (deintCount - cs);
                int32_t aosFloats = chunkCols * 2;
                AscendC::DataCopyExtParams rChunk(1, (uint32_t)(aosFloats * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(ubSrc, gB[rowGmBase + cs * 2], rChunk, {false, 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
                cvt_->DeinterleaveRow(ubSrc, ub, chunkCols, 1.0f);
                AscendC::DataCopyPad(gTR[(int64_t)i * kDim + cs], ub,
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0});
                AscendC::DataCopyPad(gTI[(int64_t)i * kDim + cs], ubI,
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0});
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }
