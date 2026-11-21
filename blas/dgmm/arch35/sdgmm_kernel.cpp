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
 * \file sdgmm_kernel.cpp
 * \brief Device-side kernel for aclblasSdgmm (arch35).
 *        Column-major storage (BLAS convention).
 *
 *        mode=R: per-column scalar broadcast multiply.
 *        mode=L: per-row-segment vector elementwise multiply,
 *                x segment resident in UB and reused across columns.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "sdgmm_kernel.h"

// ==========================================================================
// mode=R: per-column scalar broadcast multiply.
// For each column j: read scalar x[j*incx] from GM via GetValue,
// then Muls to scale the A column segment.
// ==========================================================================
template <typename T>
__aicore__ inline void SdgmmProcessRight(
    AscendC::GlobalTensor<T>& xGm, AscendC::GlobalTensor<T>& aGm, AscendC::GlobalTensor<T>& cGm,
    const SdgmmTilingData& tiling,
    uint32_t startCol, uint32_t endCol,
    uint32_t startM, uint32_t endM,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& bufA, AscendC::TBuf<AscendC::TPosition::VECCALC>& bufC)
{
    int64_t absIncx = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                         : -static_cast<int64_t>(tiling.incx);

    AscendC::DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
    bool firstIter = true;

    for (uint32_t j = startCol; j < endCol; j++) {
        uint64_t xIdx;
        if (tiling.incx >= 0) {
            xIdx = static_cast<uint64_t>(j) * static_cast<uint64_t>(tiling.incx);
        } else {
            xIdx = static_cast<uint64_t>(tiling.n - 1 - j) * static_cast<uint64_t>(absIncx);
        }
        T xj = xGm.GetValue(xIdx);

        uint32_t rowOffset = startM;
        for (uint32_t remain = endM - startM; remain > 0;) {
            uint32_t curM = (remain > tiling.tileM) ? tiling.tileM : remain;

            if (!firstIter) {
                event_t eVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eVMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eVMte2);
                event_t eMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eMte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eMte3V);
            }

            AscendC::LocalTensor<T> ubA = bufA.Get<T>();
            uint64_t aOffset = static_cast<uint64_t>(j) * static_cast<uint64_t>(tiling.lda) + rowOffset;
            AscendC::DataCopyExtParams cpA{1, curM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(ubA, aGm[aOffset], cpA, noPad);

            event_t eMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eMte2V);

            AscendC::LocalTensor<T> ubC = bufC.Get<T>();
            AscendC::Muls<T>(ubC, ubA, xj, static_cast<int32_t>(curM));
            AscendC::PipeBarrier<PIPE_V>();

            event_t eVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eVMte3);

            uint64_t cOffset = static_cast<uint64_t>(j) * static_cast<uint64_t>(tiling.ldc) + rowOffset;
            AscendC::DataCopyExtParams cpC{1, curM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(cGm[cOffset], ubC, cpC);

            firstIter = false;
            rowOffset += curM;
            remain -= curM;
        }
    }
}

// ==========================================================================
// mode=L: per-row-segment vector elementwise multiply.
// Outer loop over m row-segments: load the x segment once into UB and reuse
// it for all columns owned by this core.
// incx==1: contiguous Copy GM->UB (fast path).
// incx!=1: strided scalar load via GetValue + SetValue loop.
// ==========================================================================
template <typename T>
__aicore__ inline void SdgmmProcessLeft(
    AscendC::GlobalTensor<T>& xGm, AscendC::GlobalTensor<T>& aGm, AscendC::GlobalTensor<T>& cGm,
    const SdgmmTilingData& tiling,
    uint32_t startCol, uint32_t endCol,
    uint32_t startM, uint32_t endM,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& bufA, AscendC::TBuf<AscendC::TPosition::VECCALC>& bufC,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& bufX)
{
    int64_t absIncx = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                         : -static_cast<int64_t>(tiling.incx);

    AscendC::DataCopyPadExtParams<T> noPad{false, 0, 0, 0};

    uint32_t rowOffset = startM;
    for (uint32_t remain = endM - startM; remain > 0;) {
        uint32_t curM = (remain > tiling.tileM) ? tiling.tileM : remain;

        AscendC::LocalTensor<T> ubX = bufX.Get<T>();
        if (tiling.incx == 1) {
            AscendC::DataCopyExtParams cpX{1, curM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(ubX, xGm[rowOffset], cpX, noPad);
            event_t eMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eMte2V);
        } else {
            for (uint32_t i = 0; i < curM; i++) {
                int64_t logicalIdx = static_cast<int64_t>(rowOffset + i);
                int64_t xOffset = (tiling.incx >= 0)
                    ? logicalIdx * tiling.incx
                    : (static_cast<int64_t>(tiling.m - 1) - logicalIdx) * absIncx;
                ubX.SetValue(i, xGm.GetValue(static_cast<uint64_t>(xOffset)));
            }
        }

        bool firstCol = true;
        for (uint32_t j = startCol; j < endCol; j++) {
            if (!firstCol) {
                event_t eVMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eVMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eVMte2);
                event_t eMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eMte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eMte3V);
            }

            AscendC::LocalTensor<T> ubA = bufA.Get<T>();
            uint64_t aOffset = static_cast<uint64_t>(j) * static_cast<uint64_t>(tiling.lda) + rowOffset;
            AscendC::DataCopyExtParams cpA{1, curM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(ubA, aGm[aOffset], cpA, noPad);

            event_t eMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eMte2V);

            AscendC::LocalTensor<T> ubC = bufC.Get<T>();
            AscendC::Mul<T>(ubC, ubA, ubX, static_cast<int32_t>(curM));
            AscendC::PipeBarrier<PIPE_V>();

            event_t eVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eVMte3);

            uint64_t cOffset = static_cast<uint64_t>(j) * static_cast<uint64_t>(tiling.ldc) + rowOffset;
            AscendC::DataCopyExtParams cpC{1, curM * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(cGm[cOffset], ubC, cpC);

            firstCol = false;
        }

        rowOffset += curM;
        remain -= curM;
    }
}

// ==========================================================================
// Kernel entry -- handles all mode/incx combinations.
// 2D block decomposition: blockIdx = colBlock * mBlocks + mBlock.
// ==========================================================================
extern "C" __global__ __aicore__ void sdgmm_aiv_kernel(
    const GM_ADDR x, const GM_ADDR A, GM_ADDR C, const SdgmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t blockIdx = AscendC::GetBlockIdx();
    uint32_t mBlocks = tiling.mBlocks;
    if (mBlocks == 0) {
        mBlocks = 1;
    }

    uint32_t colBlock = blockIdx / mBlocks;
    uint32_t mBlock = blockIdx % mBlocks;
    uint32_t numColBlocks = AscendC::GetBlockNum() / mBlocks;

    uint32_t startCol, endCol;
    if (colBlock < tiling.remainder) {
        startCol = colBlock * (tiling.perCoreN + 1);
        endCol = startCol + tiling.perCoreN + 1;
    } else {
        startCol = colBlock * tiling.perCoreN + tiling.remainder;
        endCol = startCol + tiling.perCoreN;
    }

    uint32_t startMTile, endMTile;
    if (mBlock < tiling.mTileRemainder) {
        startMTile = mBlock * (tiling.perCoreMTile + 1);
        endMTile = startMTile + tiling.perCoreMTile + 1;
    } else {
        startMTile = mBlock * tiling.perCoreMTile + tiling.mTileRemainder;
        endMTile = startMTile + tiling.perCoreMTile;
    }
    uint32_t startM = startMTile * tiling.tileM;
    uint32_t endM = endMTile * tiling.tileM;
    if (endM > tiling.m) {
        endM = tiling.m;
    }

    if (tiling.m == 0 || tiling.n == 0 || tiling.tileM == 0 ||
        startCol >= endCol || startM >= endM) {
        return;
    }

    AscendC::TPipe pipe;

    uint32_t xLen = (tiling.mode == SDGMM_MODE_LEFT) ? tiling.m : tiling.n;
    int64_t absIncx = (tiling.incx >= 0) ? static_cast<int64_t>(tiling.incx)
                                         : -static_cast<int64_t>(tiling.incx);
    uint64_t xTotalEl = static_cast<uint64_t>(xLen - 1) * static_cast<uint64_t>(absIncx) + 1;

    // AscendC::GlobalTensor::SetGlobalBuffer requires non-const __gm__ float*.
    // The const_cast is safe: xGm/aGm are only read (GetValue/DataCopyPad),
    // never written. This is a framework API limitation that cannot be avoided.
    AscendC::GlobalTensor<float> xGm, aGm, cGm;
    xGm.SetGlobalBuffer(const_cast<__gm__ float*>(reinterpret_cast<const __gm__ float*>(x)), xTotalEl);
    aGm.SetGlobalBuffer(const_cast<__gm__ float*>(reinterpret_cast<const __gm__ float*>(A)),
                        static_cast<uint64_t>(tiling.n) * static_cast<uint64_t>(tiling.lda));
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C),
                        static_cast<uint64_t>(tiling.n) * static_cast<uint64_t>(tiling.ldc));

    AscendC::TBuf<AscendC::TPosition::VECCALC> bufA, bufC, bufX;
    pipe.InitBuffer(bufA, tiling.tileM * sizeof(float));
    pipe.InitBuffer(bufC, tiling.tileM * sizeof(float));
    if (tiling.mode == SDGMM_MODE_LEFT) {
        pipe.InitBuffer(bufX, tiling.tileM * sizeof(float));
    }

    if (tiling.mode == SDGMM_MODE_RIGHT) {
        SdgmmProcessRight<float>(xGm, aGm, cGm, tiling, startCol, endCol,
                                  startM, endM, bufA, bufC);
    } else {
        SdgmmProcessLeft<float>(xGm, aGm, cGm, tiling, startCol, endCol,
                                 startM, endM, bufA, bufC, bufX);
    }
}

// Kernel launcher: asynchronously launches the kernel.
// tiling.mode is the normalized value (SDGMM_MODE_LEFT / SDGMM_MODE_RIGHT).
void sdgmm_kernel_do(const GM_ADDR x, const GM_ADDR A, GM_ADDR C,
                     const SdgmmTilingData& tiling,
                     uint32_t numBlocks, void* stream)
{
    sdgmm_aiv_kernel<<<numBlocks, nullptr, stream>>>(x, A, C, tiling);
}
