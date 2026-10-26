/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

// Cross-block WAR hazard handshake for scale kernels whose cIn/temp* buffers
// are TBufs shared by every block of one core. The previous block's V compute
// may still be reading those TBufs when this block's MTE2 load is issued
// (MTE2 and V are independent pipes with no implicit ordering), so without
// this V-to-MTE2 handshake the load can overwrite the buffers first and the
// previous block then computes on this block's data.
//
// The handshake is split into a release (V side) and an acquire (MTE2 side)
// so that consecutive blocks overlap:
//   - Release is issued on the V pipe right after this block's compute, i.e.
//     once every V read of the shared input TBufs has retired (the WAR edge).
//     It must NOT be queued behind the block-end WaitFlag<MTE3_V>: the V pipe
//     stalls there until the MTE3 store retires, which would delay the next
//     block's release until the store is done and force MTE2(k+1) to
//     serialize behind MTE3(k) with no cross-block overlap.
//   - Acquire is issued on the MTE2 pipe before this block's loads touch the
//     shared input TBufs.
// Wiring per kernel: one release before the block loop primes block 0 (the V
// pipe is empty there, so it fires immediately), each block ends with a
// release for its successor, and one acquire after the loop consumes the
// final block's release so the Set/Wait counts stay balanced (a residual Set
// without a matching Wait is undefined behavior on Ascend 950).
// Note: these raw event-ID-0 flags must stay out of SyrkScaleProcess, which
// is also used by TQue-based kernels (e.g. ssyrkx) whose queue machinery
// shares the per-core event-ID pool.
__aicore__ inline void SyrkCrossBlockWarRelease()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
}

__aicore__ inline void SyrkCrossBlockWarAcquire()
{
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
}

// Reads the C block columns [jBase, jBase+cols), rows [iBase, iBase+rows) --
// column-major, contiguous -- from GM into cInUb. cSrcStride/cGmOffset are
// also written out for reuse by the store of the same block.
__aicore__ inline void SyrkLoadCBlock(
    const AscendC::GlobalTensor<float>& cGM, AscendC::LocalTensor<float>& cInUb,
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t ldc,
    int64_t& cSrcStride, int64_t& cGmOffset)
{
    cSrcStride = static_cast<int64_t>(ldc - rows) * sizeof(float);
    AscendC::DataCopyExtParams cpC{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), cSrcStride, 0, 0};
    AscendC::DataCopyPadExtParams<float> ppC{true, 0, 0, 0.0f};
    cGmOffset = static_cast<int64_t>(jBase) * ldc + iBase;
    AscendC::DataCopyPad(cInUb, cGM[cGmOffset], cpC, ppC);
}

// Block tail of the TBuf-based scale ProcessBlock, symmetric to SyrkLoadCBlock:
// releases the cross-block input-TBuf WAR edge, stores the scaled C block, and
// re-arms the output-TBuf handshake. cGmOffset/cSrcStride are the SyrkLoadCBlock
// outputs for the same block. The release -> V_MTE3 handoff -> store -> MTE3_V
// handshake order must be preserved exactly:
//   - Release: every V read of the shared cIn/temp* TBufs (the scale compute and
//     any in-place Add on temp) has retired at this point, so the next block's
//     MTE2 loads may overwrite them (WAR edge). Released before the V->MTE3
//     store handoff: queued after the block-end WaitFlag<MTE3_V> it could only
//     fire once the store retires, and MTE2(k+1) would then serialize behind
//     MTE3(k) with no cross-block overlap.
//   - V->MTE3 handoff: the store may read cUb only after the V compute retires.
//   - Store: column-major C block, the exact inverse of the SyrkLoadCBlock read.
//   - MTE3->V cross-block handshake: the output TBuf is reused by the next
//     block's V compute while this block's MTE3 store may still be reading it
//     (V and MTE3 are independent pipes with no implicit ordering). Without
//     this handshake the next compute can retire first and clobber lanes the
//     store burst has not read yet, leaking neighbouring-block values into C.
__aicore__ inline void SyrkStoreCBlockAndSync(
    const AscendC::GlobalTensor<float>& cGM, AscendC::LocalTensor<float>& cUb,
    int64_t cGmOffset, int64_t cSrcStride, uint32_t rows, uint32_t cols)
{
    SyrkCrossBlockWarRelease();

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

    AscendC::DataCopyExtParams cpOut{static_cast<uint16_t>(cols),
        static_cast<uint32_t>(rows * sizeof(float)), 0,
        cSrcStride, 0};
    AscendC::DataCopyPad(cGM[cGmOffset], cUb, cpOut);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}

__aicore__ inline void SyrkScaleApply(
    AscendC::LocalTensor<float>& cUb, AscendC::LocalTensor<float>& cInUb,
    AscendC::LocalTensor<float>& tempUb,
    uint32_t offset, int32_t count, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    if (!skipTemp) {
        if (isBetaZero) {
            Muls(cUb[offset], tempUb[offset], alpha, count);
        } else {
            Muls(cUb[offset], cInUb[offset], beta, count);
            Axpy(cUb[offset], tempUb[offset], alpha, count);
        }
    } else {
        if (isBetaZero) {
            // beta == 0: C is not a valid input (cuBLAS contract), so a pure
            // write-zero replaces 0 * C_in and never propagates NaN/Inf from C.
            Duplicate(cUb[offset], 0.0f, count);
        } else {
            Muls(cUb[offset], cInUb[offset], beta, count);
        }
    }
}

__aicore__ inline void SyrkScaleComputeResult(
    uint8_t uploMode, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols,
    uint32_t ubColStride, AscendC::LocalTensor<float>& cUb, AscendC::LocalTensor<float>& cInUb,
    AscendC::LocalTensor<float>& tempUb, bool skipTemp, bool isBetaZero,
    float alpha, float beta)
{
    bool uploUpper = (uploMode == ACLBLAS_UPPER);

    bool isFullInterior = uploUpper
        ? (jBase >= iBase + rows)
        : (jBase + cols <= iBase);

    if (isFullInterior) {
        int32_t totalCount = static_cast<int32_t>(ubColStride * cols);
        SyrkScaleApply(cUb, cInUb, tempUb, 0, totalCount, skipTemp, isBetaZero, alpha, beta);
        return;
    }

    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        uint32_t colOffset = c * ubColStride;

        uint32_t uploCount;
        if (uploUpper) {
            uploCount = (absJ >= iBase) ? Min(absJ - iBase + 1, rows) : 0;
        } else {
            uploCount = (absJ < iBase) ? rows : (rows - (absJ - iBase));
        }
        uint32_t nonUploCount = rows - uploCount;

        if (nonUploCount > 0 && uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(rows));
        }

        uint32_t computeCount = uploUpper ? uploCount : rows;
        if (computeCount > 0) {
            SyrkScaleApply(cUb, cInUb, tempUb, colOffset,
                static_cast<int32_t>(computeCount), skipTemp, isBetaZero, alpha, beta);
        }

        if (nonUploCount > 0 && !uploUpper) {
            Muls(cUb[colOffset], cInUb[colOffset], 1.0f, static_cast<int32_t>(nonUploCount));
        }
    }
}

template <typename Op>
__aicore__ inline void SyrkScaleProcess(
    Op& op, uint32_t rowStart, uint32_t rowEnd,
    uint32_t n, uint8_t uploMode, uint32_t scaleBlock)
{
    if (rowStart >= rowEnd) {
        return;
    }
    bool uploUpper = (uploMode == ACLBLAS_UPPER);

    for (uint32_t iBase = rowStart; iBase < rowEnd; iBase += scaleBlock) {
        uint32_t rows = Min<uint32_t>(scaleBlock, rowEnd - iBase);
        uint32_t iEnd = iBase + rows - 1;

        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? n : Min(iEnd + 1, n);

        for (uint32_t jBase = jStart; jBase < jLimit; jBase += scaleBlock) {
            uint32_t cols = Min<uint32_t>(scaleBlock, jLimit - jBase);
            op.ProcessBlock(iBase, jBase, rows, cols);
        }
    }
}
