/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file sgeam_kernel.cpp
 * @brief Sgeam kernel implementation (SIMD/membase, arch35)
 *        Computes: C = alpha * op(A) + beta * op(B)
 *
 *        Architecture: TPipe + TQue<VECIN>/TQue<VECOUT> + TBuf<VECCALC>
 *        EnQue/DeQue provides automatic MTE <-> Vector synchronization.
 *
 *        Buffer planning:
 *          - inQueue_ (VECIN): A tile data, managed by TQue for MTE2<->V sync
 *          - outQueue_ (VECOUT): C tile data, managed by TQue for V<->MTE3 sync
 *          - calcBuf_ (VECCALC): Scratch for B data when beta != 0; unused otherwise
 *
 *        Two dispatch paths:
 *          PATH A: NN multi-column (colsIter > 1), strided DataCopyPad for NC columns
 *          PATH B: Single-column / Trans / ConjTrans
 *
 *        Multi-core decomposition: 2D grid (colBlocks x mBlocks)
 */

#include "kernel_operator.h"
#include "sgeam_kernel.h"
#include "sgeam_tiling_data.h"

namespace {

using namespace AscendC;

// ============================================================================
// SgeamAIV: operator class for Sgeam vector kernel
// ============================================================================
class SgeamAIV {
public:
    __aicore__ inline SgeamAIV() {}
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, const SgeamTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPathA();
    __aicore__ inline void ProcessPathB();
    __aicore__ inline void ComputeCoreRanges(const SgeamTilingData& tiling);

    // PATH A helpers (NN multi-column)
    __aicore__ inline void CopyInA_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC);
    __aicore__ inline void CopyInB_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC);
    __aicore__ inline void Compute_PathA(uint32_t curM, uint32_t actualNC);
    __aicore__ inline void CopyOutC_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC);

    // PATH B helpers (single-column / Trans / ConjTrans)
    __aicore__ inline void CopyInA_PathB(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t srcStrideA);
    __aicore__ inline void CopyInB_PathB(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t srcStrideB);
    __aicore__ inline void Compute_PathB(uint32_t curM);
    __aicore__ inline void CopyOutC_PathB(uint32_t col, uint32_t rowStart, uint32_t curM);

    // GM addresses
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;

    // Queues & buffer
    TQue<TPosition::VECIN, 2> inQueue_;   // MTE2 input for A (EnQue signals MTE2 done)
    TQue<TPosition::VECOUT, 2> outQueue_; // Vector output for C (EnQue signals Vector done)
    TBuf<TPosition::VECCALC> calcBuf_;    // Scratch for B data (no MTE sync needed)

    // Tiling fields cached from SgeamTilingData
    uint32_t m_;
    uint32_t n_;

    uint32_t alphaIsZero_;
    float alpha_;
    aclblasOperation_t opA_;
    uint32_t lda_;

    uint32_t betaIsZero_;
    float beta_;
    aclblasOperation_t opB_;
    uint32_t ldb_;

    uint32_t ldc_;
    uint32_t tileM_;
    uint32_t colsIter_;

    // Per-core tile ranges
    uint32_t colStart_;
    uint32_t colEnd_;
    uint32_t mTileStart_;
    uint32_t mTileEnd_;
};

// ============================================================================
// ComputeCoreRanges: decode 2D block index into per-core column and m-tile ranges.
// On invalid range, sets colEnd_ or mTileEnd_ to 0 to signal early exit in Process().
// ============================================================================
__aicore__ inline void SgeamAIV::ComputeCoreRanges(const SgeamTilingData& tiling)
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t mBlocks = tiling.mBlocks;
    uint32_t colBlock = blockIdx / mBlocks;
    uint32_t mBlock = blockIdx % mBlocks;

    // Calculate column range
    if (colBlock < tiling.remainder) {
        colStart_ = colBlock * (tiling.perCoreN + 1);
        colEnd_ = colStart_ + tiling.perCoreN + 1;
    } else {
        colStart_ = colBlock * tiling.perCoreN + tiling.remainder;
        colEnd_ = colStart_ + tiling.perCoreN;
    }

    // Validate column range
    if (colStart_ >= n_ || colStart_ >= colEnd_) {
        colEnd_ = 0; // Signal Process() to return early
        return;
    }
    if (colEnd_ > n_) {
        colEnd_ = n_;
    }

    // Calculate m-tile range
    if (mBlock < tiling.mTileRemainder) {
        mTileStart_ = mBlock * (tiling.perCoreMTile + 1);
        mTileEnd_ = mTileStart_ + tiling.perCoreMTile + 1;
    } else {
        mTileStart_ = mBlock * tiling.perCoreMTile + tiling.mTileRemainder;
        mTileEnd_ = mTileStart_ + tiling.perCoreMTile;
    }

    // Validate m-tile range
    uint32_t totalMTiles = (m_ + tileM_ - 1) / tileM_;
    if (mTileStart_ >= totalMTiles || mTileStart_ >= mTileEnd_) {
        mTileEnd_ = 0; // Signal Process() to return early
        return;
    }
    if (mTileEnd_ > totalMTiles) {
        mTileEnd_ = totalMTiles;
    }
}

// ============================================================================
// Init: setup global tensors, compute per-core ranges, allocate UB buffers.
// TPipe pointer passed from kernel entry (R3: TPipe must not be a member).
// ============================================================================
__aicore__ inline void SgeamAIV::Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, const SgeamTilingData& tiling, TPipe* pipe)
{
    // Cache tiling fields to member variables
    m_ = tiling.m;
    n_ = tiling.n;

    alphaIsZero_ = tiling.alphaIsZero;
    opA_ = tiling.opA;
    lda_ = tiling.lda;
    alpha_ = tiling.alpha;

    betaIsZero_ = tiling.betaIsZero;
    ldb_ = tiling.ldb;
    opB_ = tiling.opB;
    beta_ = tiling.beta;

    ldc_ = tiling.ldc;
    tileM_ = tiling.tileM;
    colsIter_ = tiling.colsIter;

    // Set GM buffers
    aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
    cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));

    // Compute per-core column and m-tile ranges
    ComputeCoreRanges(tiling);
    if (colEnd_ == 0 || mTileEnd_ == 0) {
        return;
    }

    // Buffer size: PATH A needs colsIter * tileM elements, PATH B needs tileM elements.
    // Allocate the maximum (colsIter >= 1, so colsIter * tileM >= tileM).
    uint32_t bufElems = colsIter_ * tileM_;
    uint32_t bufSize = bufElems * sizeof(float);

    // Double-buffer queues (num=2) enable MTE2<->Vector pipeline overlap
    pipe->InitBuffer(inQueue_, 2, bufSize);
    pipe->InitBuffer(outQueue_, 2, bufSize);
    pipe->InitBuffer(calcBuf_, bufSize);
}

// ============================================================================
// Process: dispatch to PATH A or PATH B based on op and colsIter
// ============================================================================
__aicore__ inline void SgeamAIV::Process()
{
    if (colStart_ >= colEnd_ || mTileStart_ >= mTileEnd_) {
        return;
    }

    if (opA_ == ACLBLAS_OP_N && opB_ == ACLBLAS_OP_N && colsIter_ > 1) {
        ProcessPathA();
    } else {
        ProcessPathB();
    }
}

// ============================================================================
// CopyInA_PathA: strided DataCopyPad loads NC columns of A at once.
// ============================================================================
__aicore__ inline void SgeamAIV::CopyInA_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC)
{
    LocalTensor<float> bufA = inQueue_.AllocTensor<float>();
    if (!alphaIsZero_) {
        const uint64_t baseOff = static_cast<uint64_t>(col) * static_cast<uint64_t>(lda_) + rowStart;
        const int64_t colGap = static_cast<int64_t>(lda_ - curM) * static_cast<int64_t>(sizeof(float));
        DataCopyExtParams cpA{
            static_cast<uint16_t>(actualNC), static_cast<uint32_t>(curM * sizeof(float)), colGap, 0, 0};
        DataCopyPadExtParams<float> npA;
        DataCopyPad<float, PaddingMode::Compact>(bufA, aGm_[baseOff], cpA, npA);
    }
    inQueue_.EnQue(bufA); // Signal MTE2 done for A
}

// ============================================================================
// CopyInB_PathA: strided DataCopyPad loads NC columns of B when beta != 0.
// ============================================================================
__aicore__ inline void SgeamAIV::CopyInB_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC)
{
    if (!betaIsZero_) {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        const uint64_t baseOff = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldb_) + rowStart;
        const int64_t colGap = static_cast<int64_t>(ldb_ - curM) * static_cast<int64_t>(sizeof(float));
        DataCopyExtParams cpB{
            static_cast<uint16_t>(actualNC), static_cast<uint32_t>(curM * sizeof(float)), colGap, 0, 0};
        DataCopyPadExtParams<float> npB;
        DataCopyPad<float, PaddingMode::Compact>(ubB, bGm_[baseOff], cpB, npB);
        event_t eMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eMte2V_);
        WaitFlag<HardEvent::MTE2_V>(eMte2V_);
    }
}

// ============================================================================
// Compute_PathA: C = alpha*A + beta*B for the multi-column PATH A case.
// ============================================================================
__aicore__ inline void SgeamAIV::Compute_PathA(uint32_t curM, uint32_t actualNC)
{
    const uint32_t totalElems = actualNC * curM;
    LocalTensor<float> aIn = inQueue_.DeQue<float>();
    LocalTensor<float> cOut = outQueue_.AllocTensor<float>();

    if (alphaIsZero_ && betaIsZero_) {
        Duplicate<float>(cOut, 0.0f, totalElems);
    } else if (alphaIsZero_) {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        Muls<float>(cOut, ubB, beta_, totalElems);
    } else if (betaIsZero_) {
        Muls<float>(cOut, aIn, alpha_, totalElems);
    } else {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        Muls<float>(cOut, aIn, alpha_, totalElems);
        // Reuse aIn as scratch for beta*B to save a buffer
        Muls<float>(aIn, ubB, beta_, totalElems);
        Add<float>(cOut, cOut, aIn, totalElems);
    }
    outQueue_.EnQue(cOut); // Signal Vector done
    inQueue_.FreeTensor(aIn);
}

// ============================================================================
// CopyOutC_PathA: strided DataCopyPad writes NC columns of C.
// ============================================================================
__aicore__ inline void SgeamAIV::CopyOutC_PathA(uint32_t col, uint32_t rowStart, uint32_t curM, uint32_t actualNC)
{
    LocalTensor<float> cFinal = outQueue_.DeQue<float>();
    const uint64_t baseOff = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldc_) + rowStart;
    const int64_t colGap = static_cast<int64_t>(ldc_ - curM) * static_cast<int64_t>(sizeof(float));
    DataCopyExtParams cpS{static_cast<uint16_t>(actualNC), static_cast<uint32_t>(curM * sizeof(float)), 0, colGap, 0};
    DataCopyPad<float, PaddingMode::Compact>(cGm_[baseOff], cFinal, cpS);
    outQueue_.FreeTensor(cFinal);
}

// ============================================================================
// PATH A: NN multi-column — strided DataCopyPad loads NC columns at once
// Each tile: CopyIn A -> EnQue -> CopyIn B -> Compute -> EnQue C -> CopyOut C
// ============================================================================
__aicore__ inline void SgeamAIV::ProcessPathA()
{
    for (uint32_t col = colStart_; col < colEnd_; col += colsIter_) {
        const uint32_t actualNC = ((colEnd_ - col) < colsIter_) ? (colEnd_ - col) : colsIter_;
        for (uint32_t mTile = mTileStart_; mTile < mTileEnd_; ++mTile) {
            const uint32_t rowStart = mTile * tileM_;
            uint32_t rowEnd = rowStart + tileM_;
            if (rowEnd > m_) {
                rowEnd = m_;
            }
            const uint32_t curM = rowEnd - rowStart;

            CopyInA_PathA(col, rowStart, curM, actualNC);
            CopyInB_PathA(col, rowStart, curM, actualNC);
            Compute_PathA(curM, actualNC);
            CopyOutC_PathA(col, rowStart, curM, actualNC);
        }
    }
}

// ============================================================================
// CopyInA_PathB: single-shot for NN, strided Compact for T/C (one column).
// ============================================================================
__aicore__ inline void SgeamAIV::CopyInA_PathB(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t srcStrideA)
{
    LocalTensor<float> bufA = inQueue_.AllocTensor<float>();
    if (!alphaIsZero_) {
        if (opA_ == ACLBLAS_OP_N) {
            const uint64_t offset = static_cast<uint64_t>(col) * static_cast<uint64_t>(lda_) + rowStart;
            DataCopyExtParams cp{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> np;
            DataCopyPad(bufA, aGm_[offset], cp, np);
        } else {
            const uint64_t baseOff =
                static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(lda_);
            DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), srcStrideA, 0, 0};
            DataCopyPadExtParams<float> np;
            DataCopyPad<float, PaddingMode::Compact>(bufA, aGm_[baseOff], cp, np);
        }
    }
    inQueue_.EnQue(bufA); // Signal MTE2 done for A
}

// ============================================================================
// CopyInB_PathB: single-shot for NN, strided Compact for T/C (one column).
// ============================================================================
__aicore__ inline void SgeamAIV::CopyInB_PathB(uint32_t col, uint32_t rowStart, uint32_t curM, int64_t srcStrideB)
{
    if (!betaIsZero_) {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        if (opB_ == ACLBLAS_OP_N) {
            const uint64_t offset = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldb_) + rowStart;
            DataCopyExtParams cp{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> np;
            DataCopyPad(ubB, bGm_[offset], cp, np);
        } else {
            const uint64_t baseOff =
                static_cast<uint64_t>(col) + static_cast<uint64_t>(rowStart) * static_cast<uint64_t>(ldb_);
            DataCopyExtParams cp{static_cast<uint16_t>(curM), static_cast<uint32_t>(sizeof(float)), srcStrideB, 0, 0};
            DataCopyPadExtParams<float> np;
            DataCopyPad<float, PaddingMode::Compact>(ubB, bGm_[baseOff], cp, np);
        }
        event_t eMte2V_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eMte2V_);
        WaitFlag<HardEvent::MTE2_V>(eMte2V_);
    }
}

// ============================================================================
// Compute_PathB: C = alpha*A + beta*B for the single-column PATH B case.
// ============================================================================
__aicore__ inline void SgeamAIV::Compute_PathB(uint32_t curM)
{
    LocalTensor<float> aIn = inQueue_.DeQue<float>();
    LocalTensor<float> cOut = outQueue_.AllocTensor<float>();

    if (alphaIsZero_ && betaIsZero_) {
        Duplicate<float>(cOut, 0.0f, curM);
    } else if (alphaIsZero_) {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        Muls<float>(cOut, ubB, beta_, curM);
    } else if (betaIsZero_) {
        Muls<float>(cOut, aIn, alpha_, curM);
    } else {
        LocalTensor<float> ubB = calcBuf_.Get<float>();
        Muls<float>(cOut, aIn, alpha_, curM);
        // Reuse aIn as scratch for beta*B to save a buffer
        Muls<float>(aIn, ubB, beta_, curM);
        Add<float>(cOut, cOut, aIn, curM);
    }
    outQueue_.EnQue(cOut); // Signal Vector done
    inQueue_.FreeTensor(aIn);
}

// ============================================================================
// CopyOutC_PathB: single-shot DataCopyPad writes one column of C.
// ============================================================================
__aicore__ inline void SgeamAIV::CopyOutC_PathB(uint32_t col, uint32_t rowStart, uint32_t curM)
{
    LocalTensor<float> cFinal = outQueue_.DeQue<float>();
    const uint64_t offset = static_cast<uint64_t>(col) * static_cast<uint64_t>(ldc_) + rowStart;
    DataCopyExtParams cp{1, static_cast<uint32_t>(curM * sizeof(float)), 0, 0, 0};
    DataCopyPad(cGm_[offset], cFinal, cp);
    outQueue_.FreeTensor(cFinal);
}

// ============================================================================
// PATH B: Single-column / Trans / ConjTrans — one column per iteration
// Each tile: CopyIn A -> EnQue -> CopyIn B -> Compute -> EnQue C -> CopyOut C
// ============================================================================
__aicore__ inline void SgeamAIV::ProcessPathB()
{
    const int64_t srcStrideA = static_cast<int64_t>(lda_ - 1) * static_cast<int64_t>(sizeof(float));
    const int64_t srcStrideB = static_cast<int64_t>(ldb_ - 1) * static_cast<int64_t>(sizeof(float));

    for (uint32_t col = colStart_; col < colEnd_; ++col) {
        for (uint32_t mTile = mTileStart_; mTile < mTileEnd_; ++mTile) {
            const uint32_t rowStart = mTile * tileM_;
            uint32_t rowEnd = rowStart + tileM_;
            if (rowEnd > m_) {
                rowEnd = m_;
            }
            const uint32_t curM = rowEnd - rowStart;

            CopyInA_PathB(col, rowStart, curM, srcStrideA);
            CopyInB_PathB(col, rowStart, curM, srcStrideB);
            Compute_PathB(curM);
            CopyOutC_PathB(col, rowStart, curM);
        }
    }
}

} // namespace

// ============================================================================
// Kernel entry point
// ============================================================================
extern "C" __global__ __aicore__ void sgeam_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, SgeamTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.m == 0 || tiling.n == 0) {
        return;
    }

    TPipe pipe;
    SgeamAIV op;
    op.Init(A, B, C, tiling, &pipe);
    op.Process();
}

void sgeam_kernel_do(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const SgeamTilingData& tiling, uint32_t numBlocks, aclrtStream stream)
{
    sgeam_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tiling);
}
