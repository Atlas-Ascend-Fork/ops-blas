/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED.
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "strsmbatched_kernel.h"

using namespace AscendC;
using namespace matmul;

namespace {

constexpr uint32_t UB_TILE_STRIDE = 64U;
constexpr uint32_t UB_TILE_ELEMS = UB_TILE_STRIDE * UB_TILE_STRIDE;
constexpr uint32_t FLOAT_BLOCK_ELEMS = 8U;
constexpr uint32_t UB_TILE_BLOCKS = UB_TILE_STRIDE / FLOAT_BLOCK_ELEMS;

constexpr MatmulShapeParams MM_SHAPE{
    256U, 256U, STRSMBATCHED_GEMM_SINGLE_K, STRSMBATCHED_GEMM_TILE, STRSMBATCHED_GEMM_TILE, STRSMBATCHED_GEMM_BASE_K};
constexpr MatmulFuncParams MM_FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                   true,  true};
constexpr MatmulBiasParams MM_BIAS{false};
constexpr MatmulConfig MM_CFG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(MM_SHAPE, MM_FUNC, MM_BIAS);

template <bool TRANS>
using MmInput = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using MmOutput = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <bool L_TRANS, bool R_TRANS>
constexpr MatmulApiStaticTiling MM_TILING =
    GetMatmulApiTiling<MmInput<L_TRANS>, MmInput<R_TRANS>, MmOutput, MmOutput>(MM_CFG);

struct StepGeometry {
    uint32_t kPanel;
    uint32_t bs;
    bool hasTrail;
    uint32_t gemmKPanel;
    uint32_t gemmBs;
    uint32_t kTrail;
    uint32_t trailLen;
};

struct AivUbContext {
    LocalTensor<float> bUb;
    LocalTensor<float> bWorkUb;
    LocalTensor<float> tempUb;
    LocalTensor<float> brcbUb;
    LocalTensor<float> diagBrcbUb;
    LocalTensor<float> diagUb;
    LocalTensor<float> onesUb;
    LocalTensor<uint32_t> idxUb;
    LocalTensor<uint32_t> diagIdxUb;
};

__aicore__ inline uint32_t CeilDivU(uint32_t a, uint32_t b) { return (b == 0U) ? 0U : ((a + b - 1U) / b); }

__aicore__ inline __gm__ float* GetBatchPtr(GM_ADDR ptrArray, uint32_t b)
{
    const uint64_t addr = reinterpret_cast<__gm__ const uint64_t*>(ptrArray)[b];
    return reinterpret_cast<__gm__ float*>(addr);
}

__aicore__ inline __gm__ float* GetBatchPtrA(GM_ADDR aPtr, uint32_t b, const StrsmbatchedTilingData& t)
{
    if ((t.reserved & 1U) != 0U) {
        const uint32_t aDim = (t.side == 0U) ? t.m : t.n;
        return reinterpret_cast<__gm__ float*>(aPtr) + static_cast<uint64_t>(b) * t.lda * aDim;
    }
    return GetBatchPtr(aPtr, b);
}

__aicore__ inline __gm__ float* GetBatchPtrB(GM_ADDR bPtr, uint32_t b, const StrsmbatchedTilingData& t)
{
    if ((t.reserved & 2U) != 0U) {
        return reinterpret_cast<__gm__ float*>(bPtr) + static_cast<uint64_t>(b) * t.ldb * t.n;
    }
    return GetBatchPtr(bPtr, b);
}

__aicore__ inline bool IsForwardSweep(const StrsmbatchedTilingData& t)
{
    if (t.side == 0U) {
        return (t.uplo == 1U) ? (t.trans == 0U) : (t.trans != 0U);
    }
    return (t.uplo == 0U) ? (t.trans == 0U) : (t.trans != 0U);
}

__aicore__ inline void InitTransposeIdxUb(
    LocalTensor<uint32_t>& idxUb, LocalTensor<uint32_t>& diagIdxUb, LocalTensor<float>& onesUb)
{
    Duplicate(onesUb, 1.0f, UB_TILE_STRIDE);
    Duplicate(onesUb[UB_TILE_STRIDE], 0.0f, UB_TILE_STRIDE);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0U; i < UB_TILE_STRIDE; ++i) {
        idxUb.SetValue(i, i * UB_TILE_STRIDE * static_cast<uint32_t>(sizeof(float)));
        diagIdxUb.SetValue(i, i * (UB_TILE_STRIDE + 1U) * static_cast<uint32_t>(sizeof(float)));
    }
    for (uint32_t r = 0U; r < FLOAT_BLOCK_ELEMS; ++r) {
        onesUb.SetValue(UB_TILE_STRIDE + r * FLOAT_BLOCK_ELEMS + r, 1.0f);
    }
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> idxInt = idxUb.ReinterpretCast<int32_t>();
    for (uint32_t step = 1U; step < 32U; step <<= 1U) {
        Adds(
            idxInt[step * UB_TILE_STRIDE], idxInt, static_cast<int32_t>(step * sizeof(float)),
            static_cast<int32_t>(step * UB_TILE_STRIDE));
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void LoadMatrixTileUb(
    LocalTensor<float>& dstUb, const GlobalTensor<float>& srcGm, uint64_t baseOffset, uint32_t ld, uint32_t rows,
    uint32_t cols)
{
    if ((rows & 7U) == 0U && (ld & 7U) == 0U && (baseOffset & 7ULL) == 0ULL && ld >= rows) {
        if (rows == UB_TILE_STRIDE && ld == UB_TILE_STRIDE) {
            DataCopy(dstUb, srcGm[baseOffset], cols * UB_TILE_STRIDE);
        } else {
            const uint16_t rowBlocks = static_cast<uint16_t>(rows / FLOAT_BLOCK_ELEMS);
            const DataCopyParams params{
                static_cast<uint16_t>(cols), rowBlocks, static_cast<uint16_t>((ld - rows) / FLOAT_BLOCK_ELEMS),
                static_cast<uint16_t>(UB_TILE_BLOCKS - rowBlocks)};
            DataCopy(dstUb, srcGm[baseOffset], params);
        }
        return;
    }
    const uint32_t rowBlocks = CeilDivU(rows, FLOAT_BLOCK_ELEMS);
    const uint32_t ubGapBlocks = (UB_TILE_BLOCKS >= rowBlocks) ? (UB_TILE_BLOCKS - rowBlocks) : 0U;
    const uint32_t srcGapBytes = (ld >= rows) ? ((ld - rows) * static_cast<uint32_t>(sizeof(float))) : 0U;
    const DataCopyExtParams inParams{
        static_cast<uint16_t>(cols), rows * static_cast<uint32_t>(sizeof(float)), srcGapBytes, ubGapBlocks, 0U};
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
    DataCopyPad(dstUb, srcGm[baseOffset], inParams, padParams);
}

__aicore__ inline void StoreMatrixTileUb(
    const GlobalTensor<float>& dstGm, uint64_t baseOffset, uint32_t ld, const LocalTensor<float>& srcUb, uint32_t rows,
    uint32_t cols)
{
    if ((rows & 7U) == 0U && (ld & 7U) == 0U && (baseOffset & 7ULL) == 0ULL && ld >= rows) {
        if (rows == UB_TILE_STRIDE && ld == UB_TILE_STRIDE) {
            DataCopy(dstGm[baseOffset], srcUb, cols * UB_TILE_STRIDE);
        } else {
            const uint16_t rowBlocks = static_cast<uint16_t>(rows / FLOAT_BLOCK_ELEMS);
            const DataCopyParams params{
                static_cast<uint16_t>(cols), rowBlocks, static_cast<uint16_t>(UB_TILE_BLOCKS - rowBlocks),
                static_cast<uint16_t>((ld - rows) / FLOAT_BLOCK_ELEMS)};
            DataCopy(dstGm[baseOffset], srcUb, params);
        }
        return;
    }
    const uint32_t rowBlocks = CeilDivU(rows, FLOAT_BLOCK_ELEMS);
    const uint32_t ubGapBlocks = (UB_TILE_BLOCKS >= rowBlocks) ? (UB_TILE_BLOCKS - rowBlocks) : 0U;
    const uint32_t dstGapBytes = (ld >= rows) ? ((ld - rows) * static_cast<uint32_t>(sizeof(float))) : 0U;
    const DataCopyExtParams outParams{
        static_cast<uint16_t>(cols), rows * static_cast<uint32_t>(sizeof(float)), ubGapBlocks, dstGapBytes, 0U};
    DataCopyPad(dstGm[baseOffset], srcUb, outParams);
}

__aicore__ inline void LoadColChunkUb(
    LocalTensor<float>& dstUb, const GlobalTensor<float>& srcGm, uint64_t baseOffset, uint32_t ld, uint32_t rows,
    uint32_t cols)
{
    if ((rows & 7U) == 0U && (ld & 7U) == 0U && (baseOffset & 7ULL) == 0ULL) {
        if (ld == rows) {
            DataCopy(dstUb, srcGm[baseOffset], rows * cols);
        } else {
            const DataCopyParams params{
                static_cast<uint16_t>(cols), static_cast<uint16_t>(rows / FLOAT_BLOCK_ELEMS),
                static_cast<uint16_t>((ld - rows) / FLOAT_BLOCK_ELEMS), 0U};
            DataCopy(dstUb, srcGm[baseOffset], params);
        }
        return;
    }
    const uint32_t srcGapBytes = (ld >= rows) ? ((ld - rows) * static_cast<uint32_t>(sizeof(float))) : 0U;
    const DataCopyExtParams inParams{
        static_cast<uint16_t>(cols), rows * static_cast<uint32_t>(sizeof(float)), srcGapBytes, 0U, 0U};
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
    DataCopyPad(dstUb, srcGm[baseOffset], inParams, padParams);
}

__aicore__ inline void StoreColChunkUb(
    const GlobalTensor<float>& dstGm, uint64_t baseOffset, uint32_t ld, const LocalTensor<float>& srcUb, uint32_t rows,
    uint32_t cols)
{
    if ((rows & 7U) == 0U && (ld & 7U) == 0U && (baseOffset & 7ULL) == 0ULL) {
        if (ld == rows) {
            DataCopy(dstGm[baseOffset], srcUb, rows * cols);
        } else {
            const DataCopyParams params{
                static_cast<uint16_t>(cols), static_cast<uint16_t>(rows / FLOAT_BLOCK_ELEMS), 0U,
                static_cast<uint16_t>((ld - rows) / FLOAT_BLOCK_ELEMS)};
            DataCopy(dstGm[baseOffset], srcUb, params);
        }
        return;
    }
    const uint32_t dstGapBytes = (ld >= rows) ? ((ld - rows) * static_cast<uint32_t>(sizeof(float))) : 0U;
    const DataCopyExtParams outParams{
        static_cast<uint16_t>(cols), rows * static_cast<uint32_t>(sizeof(float)), 0U, dstGapBytes, 0U};
    DataCopyPad(dstGm[baseOffset], srcUb, outParams);
}

__aicore__ inline void SyncMte2ToVOnly()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
}

__aicore__ inline void SyncMte2ToV()
{
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void SyncVToMte3()
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
}

__aicore__ inline void SyncMte3ToNext()
{
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

__aicore__ inline bool IsExtremeFloatBits(uint32_t bits)
{
    const uint32_t exp = (bits >> 23U) & 0xFFU;
    const uint32_t mant = bits & 0x7FFFFFU;
    return (exp >= 240U) || (exp <= 2U && mant != 0U);
}

__aicore__ inline bool NeedsScalarSafeSolve(
    const LocalTensor<float>& aUb, const LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, float alpha)
{
    (void)rows;
    (void)cols;
    union {
        float f;
        uint32_t u;
    } conv;
    conv.f = alpha;
    if (IsExtremeFloatBits(conv.u)) {
        return true;
    }
    conv.f = aUb.GetValue(0);
    if (IsExtremeFloatBits(conv.u)) {
        return true;
    }
    conv.f = bUb.GetValue(0);
    return IsExtremeFloatBits(conv.u);
}

__aicore__ inline void ScaleTileScalar(LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, float alpha)
{
    for (uint32_t j = 0U; j < cols; ++j) {
        for (uint32_t i = 0U; i < rows; ++i) {
            const uint32_t idx = j * UB_TILE_STRIDE + i;
            bUb.SetValue(idx, alpha * bUb.GetValue(idx));
        }
    }
}

__aicore__ inline void SolveLeftNoTransScalar(
    const LocalTensor<float>& aUb, LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, uint32_t uplo, uint32_t diag)
{
    for (uint32_t j = 0U; j < cols; ++j) {
        for (uint32_t s = 0U; s < rows; ++s) {
            const uint32_t k = (uplo == 0U) ? (rows - 1U - s) : s;
            float bkj = bUb.GetValue(j * UB_TILE_STRIDE + k);
            if (bkj == 0.0f) {
                continue;
            }
            if (diag == 0U) {
                bkj /= aUb.GetValue(k * UB_TILE_STRIDE + k);
                bUb.SetValue(j * UB_TILE_STRIDE + k, bkj);
            }
            const uint32_t iStart = (uplo == 0U) ? 0U : (k + 1U);
            const uint32_t iEnd = (uplo == 0U) ? k : rows;
            for (uint32_t i = iStart; i < iEnd; ++i) {
                const uint32_t idx = j * UB_TILE_STRIDE + i;
                bUb.SetValue(idx, bUb.GetValue(idx) - bkj * aUb.GetValue(k * UB_TILE_STRIDE + i));
            }
        }
    }
}

__aicore__ inline void SolveLeftTransScalar(
    const LocalTensor<float>& aUb, LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, uint32_t uplo, uint32_t diag)
{
    for (uint32_t j = 0U; j < cols; ++j) {
        for (uint32_t s = 0U; s < rows; ++s) {
            const uint32_t i = (uplo == 0U) ? s : (rows - 1U - s);
            float temp = bUb.GetValue(j * UB_TILE_STRIDE + i);
            const uint32_t kStart = (uplo == 0U) ? 0U : (i + 1U);
            const uint32_t kEnd = (uplo == 0U) ? i : rows;
            for (uint32_t k = kStart; k < kEnd; ++k) {
                temp -= aUb.GetValue(i * UB_TILE_STRIDE + k) * bUb.GetValue(j * UB_TILE_STRIDE + k);
            }
            if (diag == 0U) {
                temp /= aUb.GetValue(i * UB_TILE_STRIDE + i);
            }
            bUb.SetValue(j * UB_TILE_STRIDE + i, temp);
        }
    }
}

__aicore__ inline void SolveRightNoTransScalar(
    const LocalTensor<float>& aUb, LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, uint32_t uplo, uint32_t diag)
{
    for (uint32_t s = 0U; s < cols; ++s) {
        const uint32_t j = (uplo == 0U) ? s : (cols - 1U - s);
        const uint32_t kStart = (uplo == 0U) ? 0U : (j + 1U);
        const uint32_t kEnd = (uplo == 0U) ? j : cols;
        for (uint32_t k = kStart; k < kEnd; ++k) {
            const float akj = aUb.GetValue(j * UB_TILE_STRIDE + k);
            if (akj != 0.0f) {
                for (uint32_t i = 0U; i < rows; ++i) {
                    const uint32_t idx = j * UB_TILE_STRIDE + i;
                    bUb.SetValue(idx, bUb.GetValue(idx) - akj * bUb.GetValue(k * UB_TILE_STRIDE + i));
                }
            }
        }
        if (diag == 0U) {
            const float temp = 1.0f / aUb.GetValue(j * UB_TILE_STRIDE + j);
            for (uint32_t i = 0U; i < rows; ++i) {
                const uint32_t idx = j * UB_TILE_STRIDE + i;
                bUb.SetValue(idx, bUb.GetValue(idx) * temp);
            }
        }
    }
}

__aicore__ inline void SolveRightTransScalar(
    const LocalTensor<float>& aUb, LocalTensor<float>& bUb, uint32_t rows, uint32_t cols, uint32_t uplo, uint32_t diag)
{
    for (uint32_t s = 0U; s < cols; ++s) {
        const uint32_t k = (uplo == 0U) ? (cols - 1U - s) : s;
        if (diag == 0U) {
            const float temp = 1.0f / aUb.GetValue(k * UB_TILE_STRIDE + k);
            for (uint32_t i = 0U; i < rows; ++i) {
                const uint32_t idx = k * UB_TILE_STRIDE + i;
                bUb.SetValue(idx, temp * bUb.GetValue(idx));
            }
        }
        const uint32_t jStart = (uplo == 0U) ? 0U : (k + 1U);
        const uint32_t jEnd = (uplo == 0U) ? k : cols;
        for (uint32_t j = jStart; j < jEnd; ++j) {
            const float ajk = aUb.GetValue(k * UB_TILE_STRIDE + j);
            if (ajk != 0.0f) {
                for (uint32_t i = 0U; i < rows; ++i) {
                    const uint32_t idx = j * UB_TILE_STRIDE + i;
                    bUb.SetValue(idx, bUb.GetValue(idx) - ajk * bUb.GetValue(k * UB_TILE_STRIDE + i));
                }
            }
        }
    }
}

__aicore__ inline void SolveTileScalarNetlib(
    const LocalTensor<float>& aUb, LocalTensor<float>& bUb, uint32_t rows, uint32_t cols,
    const StrsmbatchedTilingData& t, bool doScale)
{
    if (doScale && t.alpha != 1.0f) {
        ScaleTileScalar(bUb, rows, cols, t.alpha);
    }
    if (t.side == 0U) {
        if (t.trans == 0U) {
            SolveLeftNoTransScalar(aUb, bUb, rows, cols, t.uplo, t.diag);
        } else {
            SolveLeftTransScalar(aUb, bUb, rows, cols, t.uplo, t.diag);
        }
    } else {
        if (t.trans == 0U) {
            SolveRightNoTransScalar(aUb, bUb, rows, cols, t.uplo, t.diag);
        } else {
            SolveRightTransScalar(aUb, bUb, rows, cols, t.uplo, t.diag);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void TransposeUbTile(
    LocalTensor<float>& dstUb, const LocalTensor<float>& srcUb, const LocalTensor<uint32_t>& idxUb, uint32_t dstRows,
    uint32_t dstCols)
{
    if (dstRows == 0U || dstCols == 0U) {
        return;
    }
    const uint64_t rowMask = static_cast<uint64_t>(CeilDivU(dstRows, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
    const uint32_t cnt0 = (dstCols > 32U) ? 32U : dstCols;
    Gather(dstUb, srcUb, idxUb, 0U, rowMask, static_cast<uint8_t>(cnt0), static_cast<uint8_t>(UB_TILE_BLOCKS));
    if (dstCols > 32U) {
        const uint32_t cnt1 = dstCols - 32U;
        Gather(
            dstUb[32U * UB_TILE_STRIDE], srcUb, idxUb, 32U * static_cast<uint32_t>(sizeof(float)), rowMask,
            static_cast<uint8_t>(cnt1), static_cast<uint8_t>(UB_TILE_BLOCKS));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void PrepareDiagInvBrcb(AivUbContext& ub, const LocalTensor<float>& aUb, uint32_t bs)
{
    if (bs == 0U) {
        return;
    }
    Gather(ub.diagUb, aUb, ub.diagIdxUb, 0U, static_cast<uint64_t>(bs), 1U, 8U);
    PipeBarrier<PIPE_V>();
    Div(ub.diagUb, ub.onesUb, ub.diagUb, static_cast<uint64_t>(bs), 1U, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    PipeBarrier<PIPE_V>();
    const uint32_t diagBlocks = CeilDivU(bs, FLOAT_BLOCK_ELEMS);
    Brcb(ub.diagBrcbUb, ub.diagUb, static_cast<uint8_t>(diagBlocks), BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ExpandAstepBrcbAll(
    LocalTensor<float>& brcbUb, LocalTensor<float>& aStepUb, const LocalTensor<float>& diagBrcbUb, uint32_t numCols,
    uint32_t diag)
{
    if (numCols == 0U) {
        return;
    }
    if (diag == 0U) {
        const uint64_t colMask = static_cast<uint64_t>(CeilDivU(numCols, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
        Mul(aStepUb, aStepUb, diagBrcbUb, colMask, static_cast<uint8_t>(numCols), BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        PipeBarrier<PIPE_V>();
    }
    for (uint32_t colStart = 0U; colStart < numCols; colStart += 16U) {
        const uint32_t cnt = (colStart + 16U <= numCols) ? 16U : (numCols - colStart);
        Brcb(
            brcbUb[colStart * UB_TILE_STRIDE * FLOAT_BLOCK_ELEMS], aStepUb[colStart * UB_TILE_STRIDE],
            static_cast<uint8_t>(cnt * UB_TILE_BLOCKS), BrcbRepeatParams(1, 8));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void EliminateColumnsVec(
    LocalTensor<float>& workUb, AivUbContext& ub, uint32_t numCols, uint32_t vecLen, bool forward, uint32_t diag)
{
    if (numCols == 0U || vecLen == 0U) {
        return;
    }
    const uint64_t vecMask = static_cast<uint64_t>(CeilDivU(vecLen, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
    for (uint32_t s = 0U; s + 1U < numCols; ++s) {
        const uint32_t k = forward ? s : (numCols - 1U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint32_t jEnd = forward ? numCols : k;
        const uint8_t rem = static_cast<uint8_t>(jEnd - jStart);
        const uint32_t brcbBase = (k * UB_TILE_STRIDE + jStart) * FLOAT_BLOCK_ELEMS;
        Mul(ub.tempUb[jStart * UB_TILE_STRIDE], ub.brcbUb[brcbBase], workUb[k * UB_TILE_STRIDE], vecMask, rem,
            BinaryRepeatParams(1, 0, 1, 8, 1, 0));
        PipeBarrier<PIPE_V>();
        Sub(workUb[jStart * UB_TILE_STRIDE], workUb[jStart * UB_TILE_STRIDE], ub.tempUb[jStart * UB_TILE_STRIDE],
            vecMask, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
    }
    if (diag == 0U) {
        Mul(workUb, workUb, ub.diagBrcbUb, vecMask, static_cast<uint8_t>(numCols),
            BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SolveTileVectorized(
    AivUbContext& ub, uint32_t rows, uint32_t cols, const StrsmbatchedTilingData& t, bool doScale)
{
    const bool forward = IsForwardSweep(t);
    if (t.side == 1U) {
        if (doScale && t.alpha != 1.0f) {
            Muls(ub.bUb, ub.bUb, t.alpha, cols * UB_TILE_STRIDE);
            PipeBarrier<PIPE_V>();
        }
        EliminateColumnsVec(ub.bUb, ub, cols, rows, forward, t.diag);
    } else {
        TransposeUbTile(ub.bWorkUb, ub.bUb, ub.idxUb, cols, rows);
        if (doScale && t.alpha != 1.0f) {
            Muls(ub.bWorkUb, ub.bWorkUb, t.alpha, rows * UB_TILE_STRIDE);
            PipeBarrier<PIPE_V>();
        }
        EliminateColumnsVec(ub.bWorkUb, ub, rows, cols, forward, t.diag);
        TransposeUbTile(ub.bUb, ub.bWorkUb, ub.idxUb, rows, cols);
    }
}

__aicore__ inline bool ComputeLinearTaskRange(
    uint32_t totalTasks, uint32_t aivNum, uint32_t aivIdx, uint32_t& taskStart, uint32_t& taskEnd)
{
    if (aivNum == 0U || totalTasks == 0U) {
        return false;
    }
    const uint32_t baseTasks = totalTasks / aivNum;
    const uint32_t remTasks = totalTasks % aivNum;
    taskStart = aivIdx * baseTasks + ((aivIdx < remTasks) ? aivIdx : remTasks);
    taskEnd = taskStart + baseTasks + ((aivIdx < remTasks) ? 1U : 0U);
    return true;
}

__aicore__ inline void PrepareAstepBrcb(
    AivUbContext& ub, const StrsmbatchedTilingData& t, uint32_t bs, bool aColMajorStep)
{
    if (t.diag == 0U) {
        PrepareDiagInvBrcb(ub, ub.tempUb, bs);
    }
    if (!aColMajorStep) {
        TransposeUbTile(ub.bWorkUb, ub.tempUb, ub.idxUb, bs, bs);
        ExpandAstepBrcbAll(ub.brcbUb, ub.bWorkUb, ub.diagBrcbUb, bs, t.diag);
    } else {
        ExpandAstepBrcbAll(ub.brcbUb, ub.tempUb, ub.diagBrcbUb, bs, t.diag);
    }
}

__aicore__ inline bool TrySolveTileScalar(
    AivUbContext& ub, GlobalTensor<float>& bGm, uint64_t bOff, const StrsmbatchedTilingData& t, uint32_t rows,
    uint32_t cols, uint32_t bCopyRows, bool doScale)
{
    SyncMte2ToV();
    if (NeedsScalarSafeSolve(ub.tempUb, ub.bUb, rows, cols, doScale ? t.alpha : 1.0f)) {
        SolveTileScalarNetlib(ub.tempUb, ub.bUb, rows, cols, t, doScale);
        SyncVToMte3();
        StoreMatrixTileUb(bGm, bOff, t.ldb, ub.bUb, bCopyRows, cols);
        SyncMte3ToNext();
        return true;
    }
    return false;
}

__aicore__ inline void ComputePanelStepChunk(
    AivUbContext& ub, GlobalTensor<float>& bGm, GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t,
    const StepGeometry& geo, bool doScale, uint32_t b, uint32_t cIdx, uint32_t rhsLen, uint64_t aOff,
    uint32_t aCopyRows, bool aColMajorStep, bool checkScalarSafe, uint32_t& cachedBatch)
{
    const uint32_t cStart = cIdx * UB_TILE_STRIDE;
    const uint32_t cLen = (cStart + UB_TILE_STRIDE <= rhsLen) ? UB_TILE_STRIDE : (rhsLen - cStart);
    const uint32_t rows = (t.side == 0U) ? geo.bs : cLen;
    const uint32_t bCopyRows =
        ((t.reserved & 2U) != 0U) ? (CeilDivU(rows, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS) : rows;
    const uint32_t cols = (t.side == 0U) ? cLen : geo.bs;
    const uint64_t bOff = (t.side == 0U) ? (static_cast<uint64_t>(cStart) * t.ldb + geo.kPanel) :
                                           (static_cast<uint64_t>(geo.kPanel) * t.ldb + cStart);
    if (b != cachedBatch) {
        GlobalTensor<float> aGm;
        aGm.SetGlobalBuffer(GetBatchPtrA(aPtrArray, b, t));
        bGm.SetGlobalBuffer(GetBatchPtrB(bPtrArray, b, t));
        LoadMatrixTileUb(ub.tempUb, aGm, aOff, t.lda, aCopyRows, geo.bs);
        LoadMatrixTileUb(ub.bUb, bGm, bOff, t.ldb, bCopyRows, cols);
        if (checkScalarSafe) {
            if (TrySolveTileScalar(ub, bGm, bOff, t, rows, cols, bCopyRows, doScale)) {
                return;
            }
        } else {
            SyncMte2ToVOnly();
        }
        PrepareAstepBrcb(ub, t, geo.bs, aColMajorStep);
        cachedBatch = b;
    } else {
        LoadMatrixTileUb(ub.bUb, bGm, bOff, t.ldb, bCopyRows, cols);
        SyncMte2ToVOnly();
    }
    SolveTileVectorized(ub, rows, cols, t, doScale);
    SyncVToMte3();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    StoreMatrixTileUb(bGm, bOff, t.ldb, ub.bUb, bCopyRows, cols);
    SyncMte3ToNext();
}

__aicore__ __attribute__((noinline)) void SolvePanelStepAiv(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t, const StepGeometry& geo, bool doScale,
    AivUbContext& ub)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    const uint32_t rhsLen = (t.side == 0U) ? t.n : t.m;
    const uint32_t numChunks = CeilDivU(rhsLen, UB_TILE_STRIDE);
    if (numChunks == 0U) {
        return;
    }
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(t.batchCount * numChunks, aivNum, blk, taskStart, taskEnd)) {
        return;
    }
    const uint64_t aOff = static_cast<uint64_t>(geo.kPanel) * t.lda + geo.kPanel;
    const uint32_t aCopyRows =
        ((t.reserved & 1U) != 0U) ? (CeilDivU(geo.bs, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS) : geo.bs;
    const bool aColMajorStep = (t.side == 1U) ? (t.trans != 0U) : (t.trans == 0U);
    const bool checkScalarSafe = (t.batchCount <= 2U && t.m <= 32U && t.n <= 32U);
    uint32_t cachedBatch = 0xFFFFFFFFU;
    GlobalTensor<float> bGm;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        const uint32_t b = task / numChunks;
        const uint32_t cIdx = task % numChunks;
        ComputePanelStepChunk(
            ub, bGm, aPtrArray, bPtrArray, t, geo, doScale, b, cIdx, rhsLen, aOff, aCopyRows, aColMajorStep,
            checkScalarSafe, cachedBatch);
    }
}

__aicore__ inline StepGeometry ComputeStepGeometry(
    uint32_t aDim, uint32_t panelNb, uint32_t groupPanels, uint32_t numSubPanels, bool forward, uint32_t s)
{
    (void)groupPanels;
    StepGeometry g{0U, 0U, false, 0U, 0U, 0U, 0U};
    const uint32_t p = forward ? s : (numSubPanels - 1U - s);
    g.kPanel = p * panelNb;
    g.bs = (g.kPanel + panelNb <= aDim) ? panelNb : (aDim - g.kPanel);
    if (s + 1U >= numSubPanels) {
        return g;
    }
    g.hasTrail = true;
    const uint32_t nextS = s + 1U;
    const uint32_t rawSpan = nextS & (~nextS + 1U);
    const uint32_t maxSpan = (panelNb > 0U) ? (STRSMBATCHED_GEMM_SINGLE_K / panelNb) : 4U;
    const uint32_t span = (rawSpan >= maxSpan) ? maxSpan : rawSpan;
    const uint32_t groupStart = nextS - span;
    const uint32_t groupEnd = (span >= maxSpan || nextS + span >= numSubPanels) ? numSubPanels : (nextS + span);
    if (forward) {
        g.gemmKPanel = groupStart * panelNb;
        g.kTrail = g.kPanel + g.bs;
        g.gemmBs = g.kTrail - g.gemmKPanel;
        const uint32_t lim = (groupEnd * panelNb <= aDim) ? (groupEnd * panelNb) : aDim;
        g.trailLen = lim - g.kTrail;
    } else {
        g.gemmKPanel = g.kPanel;
        const uint32_t solvedEnd = (groupStart == 0U) ? aDim : ((numSubPanels - groupStart) * panelNb);
        g.gemmBs = solvedEnd - g.gemmKPanel;
        g.kTrail = (numSubPanels - groupEnd) * panelNb;
        g.trailLen = g.kPanel - g.kTrail;
    }
    return g;
}

__aicore__ inline void InitIdentityUb(
    LocalTensor<float>& bUb, LocalTensor<float>& diagUb, const LocalTensor<float>& onesUb, uint32_t bsAligned,
    float scaleVal)
{
    Duplicate(bUb, 0.0f, bsAligned * UB_TILE_STRIDE);
    if (scaleVal != 1.0f) {
        Muls(diagUb, onesUb[UB_TILE_STRIDE], scaleVal, UB_TILE_STRIDE);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> srcDiag = (scaleVal != 1.0f) ? diagUb : onesUb[UB_TILE_STRIDE];
    const uint8_t blkCount = static_cast<uint8_t>(bsAligned / FLOAT_BLOCK_ELEMS);
    for (uint32_t r = 0U; r < FLOAT_BLOCK_ELEMS; ++r) {
        Adds(
            bUb[r * UB_TILE_STRIDE], srcDiag[r * FLOAT_BLOCK_ELEMS], 0.0f, static_cast<uint64_t>(FLOAT_BLOCK_ELEMS),
            blkCount, UnaryRepeatParams(1, 1, 65, 0));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void InitIdentityPairUb(
    LocalTensor<float>& bUb0, LocalTensor<float>& bUb1, LocalTensor<float>& diagUb, const LocalTensor<float>& onesUb,
    uint32_t bsAligned, float scaleVal)
{
    Duplicate(bUb0, 0.0f, bsAligned * UB_TILE_STRIDE);
    Duplicate(bUb1, 0.0f, bsAligned * UB_TILE_STRIDE);
    if (scaleVal != 1.0f) {
        Muls(diagUb, onesUb[UB_TILE_STRIDE], scaleVal, UB_TILE_STRIDE);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> srcDiag = (scaleVal != 1.0f) ? diagUb : onesUb[UB_TILE_STRIDE];
    const uint8_t blkCount = static_cast<uint8_t>(bsAligned / FLOAT_BLOCK_ELEMS);
    for (uint32_t r = 0U; r < FLOAT_BLOCK_ELEMS; ++r) {
        Adds(
            bUb0[r * UB_TILE_STRIDE], srcDiag[r * FLOAT_BLOCK_ELEMS], 0.0f, static_cast<uint64_t>(FLOAT_BLOCK_ELEMS),
            blkCount, UnaryRepeatParams(1, 1, 65, 0));
        Adds(
            bUb1[r * UB_TILE_STRIDE], srcDiag[r * FLOAT_BLOCK_ELEMS], 0.0f, static_cast<uint64_t>(FLOAT_BLOCK_ELEMS),
            blkCount, UnaryRepeatParams(1, 1, 65, 0));
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void InitDiagInvPairUb(
    LocalTensor<float>& temp0, LocalTensor<float>& temp1, LocalTensor<float>& diag1, LocalTensor<float>& diagBrcb1,
    AivUbContext& ub, uint32_t bs)
{
    Gather(ub.diagUb, temp0, ub.diagIdxUb, 0U, static_cast<uint64_t>(bs), 1U, 8U);
    Gather(diag1, temp1, ub.diagIdxUb, 0U, static_cast<uint64_t>(bs), 1U, 8U);
    PipeBarrier<PIPE_V>();
    Div(ub.diagUb, ub.onesUb, ub.diagUb, static_cast<uint64_t>(bs), 1U, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    Div(diag1, ub.onesUb, diag1, static_cast<uint64_t>(bs), 1U, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    PipeBarrier<PIPE_V>();
    const uint8_t diagBlocks = static_cast<uint8_t>(CeilDivU(bs, FLOAT_BLOCK_ELEMS));
    Brcb(ub.diagBrcbUb, ub.diagUb, diagBlocks, BrcbRepeatParams(1, 8));
    Brcb(diagBrcb1, diag1, diagBlocks, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void EliminateBothPanelPairs(
    LocalTensor<float>& work0, LocalTensor<float>& work1, LocalTensor<float>& temp0, LocalTensor<float>& temp1,
    LocalTensor<float>& brcb0, LocalTensor<float>& brcb1, uint32_t k, uint32_t jStart, uint32_t colOff,
    uint32_t brcbBase, uint64_t vecMask, uint8_t rem)
{
    Mul(temp0[jStart * UB_TILE_STRIDE + colOff], brcb0[brcbBase], work0[k * UB_TILE_STRIDE + colOff], vecMask, rem,
        BinaryRepeatParams(1, 0, 1, 8, 1, 0));
    Mul(temp1[jStart * UB_TILE_STRIDE + colOff], brcb1[brcbBase], work1[k * UB_TILE_STRIDE + colOff], vecMask, rem,
        BinaryRepeatParams(1, 0, 1, 8, 1, 0));
    PipeBarrier<PIPE_V>();
    Sub(work0[jStart * UB_TILE_STRIDE + colOff], work0[jStart * UB_TILE_STRIDE + colOff],
        temp0[jStart * UB_TILE_STRIDE + colOff], vecMask, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    Sub(work1[jStart * UB_TILE_STRIDE + colOff], work1[jStart * UB_TILE_STRIDE + colOff],
        temp1[jStart * UB_TILE_STRIDE + colOff], vecMask, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void InvertPanelPairEliminationLoop(
    LocalTensor<float>& work0, LocalTensor<float>& work1, LocalTensor<float>& temp0, LocalTensor<float>& temp1,
    LocalTensor<float>& aStep0, LocalTensor<float>& aStep1, LocalTensor<float>& brcb0, LocalTensor<float>& brcb1,
    uint32_t bs, bool forward)
{
    uint32_t loadedChunkBase = 0xFFFFFFFFU;
    for (uint32_t s = 0U; s + 1U < bs; ++s) {
        const uint32_t k = forward ? s : (bs - 1U - s);
        const uint32_t chunkBase = (k / 16U) * 16U;
        if (chunkBase != loadedChunkBase) {
            const uint32_t cnt = (chunkBase + 16U <= bs) ? 16U : (bs - chunkBase);
            const uint8_t blkCnt = static_cast<uint8_t>(cnt * UB_TILE_BLOCKS);
            Brcb(brcb0, aStep0[chunkBase * UB_TILE_STRIDE], blkCnt, BrcbRepeatParams(1, 8));
            Brcb(brcb1, aStep1[chunkBase * UB_TILE_STRIDE], blkCnt, BrcbRepeatParams(1, 8));
            PipeBarrier<PIPE_V>();
            loadedChunkBase = chunkBase;
        }
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint32_t jEnd = forward ? bs : k;
        const uint8_t rem = static_cast<uint8_t>(jEnd - jStart);
        const uint32_t brcbBase = ((k - chunkBase) * UB_TILE_STRIDE + jStart) * FLOAT_BLOCK_ELEMS;
        const uint32_t colOff = forward ? 0U : ((k / FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
        const uint32_t activeCols = forward ? (k + 1U) : (bs - colOff);
        const uint64_t vecMask = static_cast<uint64_t>(CeilDivU(activeCols, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
        EliminateBothPanelPairs(work0, work1, temp0, temp1, brcb0, brcb1, k, jStart, colOff, brcbBase, vecMask, rem);
    }
}

__aicore__ inline void InvertPanelPairAiv(
    GM_ADDR aPtrArray, GlobalTensor<float>& invGm, const StrsmbatchedTilingData& t, const StepGeometry& geo,
    uint64_t aOff, bool forward, uint32_t bsAligned, float scaleVal, uint32_t b0, uint32_t b1, AivUbContext& ub)
{
    LocalTensor<float> work0 = ub.bUb;
    LocalTensor<float> work1 = ub.bWorkUb;
    LocalTensor<float> temp0 = ub.tempUb;
    LocalTensor<float> temp1 = ub.brcbUb;
    LocalTensor<float> aStep0 = ub.brcbUb[UB_TILE_ELEMS];
    LocalTensor<float> aStep1 = ub.brcbUb[2U * UB_TILE_ELEMS];
    LocalTensor<float> diagBrcb1 = ub.brcbUb[3U * UB_TILE_ELEMS];
    LocalTensor<float> diag1 = ub.brcbUb[3U * UB_TILE_ELEMS + UB_TILE_STRIDE * FLOAT_BLOCK_ELEMS];
    LocalTensor<float> brcb0 = ub.brcbUb[4U * UB_TILE_ELEMS];
    LocalTensor<float> brcb1 = ub.brcbUb[6U * UB_TILE_ELEMS];

    GlobalTensor<float> aGm0;
    GlobalTensor<float> aGm1;
    aGm0.SetGlobalBuffer(GetBatchPtrA(aPtrArray, b0, t));
    aGm1.SetGlobalBuffer(GetBatchPtrA(aPtrArray, b1, t));
    LoadMatrixTileUb(temp0, aGm0, aOff, t.lda, geo.bs, geo.bs);
    LoadMatrixTileUb(temp1, aGm1, aOff, t.lda, geo.bs, geo.bs);
    InitIdentityPairUb(work0, work1, ub.diagUb, ub.onesUb, bsAligned, scaleVal);
    SyncMte2ToVOnly();

    if (t.diag == 0U) {
        InitDiagInvPairUb(temp0, temp1, diag1, diagBrcb1, ub, geo.bs);
    }
    TransposeUbTile(aStep0, temp0, ub.idxUb, geo.bs, geo.bs);
    TransposeUbTile(aStep1, temp1, ub.idxUb, geo.bs, geo.bs);
    const uint64_t colMask = static_cast<uint64_t>(bsAligned);
    if (t.diag == 0U) {
        Mul(aStep0, aStep0, ub.diagBrcbUb, colMask, static_cast<uint8_t>(geo.bs), BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        Mul(aStep1, aStep1, diagBrcb1, colMask, static_cast<uint8_t>(geo.bs), BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        PipeBarrier<PIPE_V>();
    }
    InvertPanelPairEliminationLoop(work0, work1, temp0, temp1, aStep0, aStep1, brcb0, brcb1, geo.bs, forward);
    if (t.diag == 0U) {
        Mul(work0, work0, ub.diagBrcbUb, colMask, static_cast<uint8_t>(geo.bs), BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        Mul(work1, work1, diagBrcb1, colMask, static_cast<uint8_t>(geo.bs), BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        PipeBarrier<PIPE_V>();
    }
    SyncVToMte3();
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopy(invGm[static_cast<uint64_t>(b0) * UB_TILE_ELEMS], work0, bsAligned * UB_TILE_STRIDE);
    DataCopy(invGm[static_cast<uint64_t>(b1) * UB_TILE_ELEMS], work1, bsAligned * UB_TILE_STRIDE);
    SyncMte3ToNext();
}

__aicore__ inline void EliminateColumnsIdentityVec(
    LocalTensor<float>& workUb, AivUbContext& ub, uint32_t numCols, bool forward, uint32_t diag)
{
    if (numCols == 0U) {
        return;
    }
    const uint32_t lastColBlock = (numCols - 1U) / FLOAT_BLOCK_ELEMS;
    for (uint32_t s = 0U; s + 1U < numCols; ++s) {
        const uint32_t k = forward ? s : (numCols - 1U - s);
        const uint32_t jStart = forward ? (k + 1U) : 0U;
        const uint32_t jEnd = forward ? numCols : k;
        const uint8_t rem = static_cast<uint8_t>(jEnd - jStart);
        const uint32_t brcbBase = (k * UB_TILE_STRIDE + jStart) * FLOAT_BLOCK_ELEMS;
        const uint32_t cbStart = forward ? 0U : (k / FLOAT_BLOCK_ELEMS);
        const uint32_t cbEnd = forward ? (k / FLOAT_BLOCK_ELEMS) : lastColBlock;
        const uint32_t colOff = cbStart * FLOAT_BLOCK_ELEMS;
        const uint64_t vecMask = static_cast<uint64_t>((cbEnd - cbStart + 1U) * FLOAT_BLOCK_ELEMS);
        Mul(ub.tempUb[jStart * UB_TILE_STRIDE + colOff], ub.brcbUb[brcbBase], workUb[k * UB_TILE_STRIDE + colOff],
            vecMask, rem, BinaryRepeatParams(1, 0, 1, 8, 1, 0));
        PipeBarrier<PIPE_V>();
        Sub(workUb[jStart * UB_TILE_STRIDE + colOff], workUb[jStart * UB_TILE_STRIDE + colOff],
            ub.tempUb[jStart * UB_TILE_STRIDE + colOff], vecMask, rem, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        PipeBarrier<PIPE_V>();
    }
    if (diag == 0U) {
        const uint64_t fullMask = static_cast<uint64_t>(CeilDivU(numCols, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
        Mul(workUb, workUb, ub.diagBrcbUb, fullMask, static_cast<uint8_t>(numCols),
            BinaryRepeatParams(1, 1, 0, 8, 8, 1));
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void InvertPanelStepAiv(
    GM_ADDR aPtrArray, __gm__ float* invAll, const StrsmbatchedTilingData& t, const StepGeometry& geo, float scaleVal,
    AivUbContext& ub)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    if (aivNum == 0U || geo.bs == 0U) {
        return;
    }
    const uint32_t aivIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t baseTasks = t.batchCount / aivNum;
    const uint32_t remTasks = t.batchCount % aivNum;
    const uint32_t bStart = aivIdx * baseTasks + ((aivIdx < remTasks) ? aivIdx : remTasks);
    const uint32_t bEnd = bStart + baseTasks + ((aivIdx < remTasks) ? 1U : 0U);
    const uint64_t aOff = static_cast<uint64_t>(geo.kPanel) * t.lda + geo.kPanel;
    const bool forward = (t.uplo == 0U);
    const uint32_t bsAligned = CeilDivU(geo.bs, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS;
    GlobalTensor<float> invGm;
    invGm.SetGlobalBuffer(invAll);
    uint32_t b = bStart;
    for (; b + 1U < bEnd; b += 2U) {
        InvertPanelPairAiv(aPtrArray, invGm, t, geo, aOff, forward, bsAligned, scaleVal, b, b + 1U, ub);
    }
    if (b < bEnd) {
        GlobalTensor<float> aGm;
        aGm.SetGlobalBuffer(GetBatchPtrA(aPtrArray, b, t));
        LoadMatrixTileUb(ub.tempUb, aGm, aOff, t.lda, geo.bs, geo.bs);
        InitIdentityUb(ub.bUb, ub.diagUb, ub.onesUb, bsAligned, scaleVal);
        SyncMte2ToVOnly();
        if (t.diag == 0U) {
            PrepareDiagInvBrcb(ub, ub.tempUb, geo.bs);
        }
        TransposeUbTile(ub.bWorkUb, ub.tempUb, ub.idxUb, geo.bs, geo.bs);
        ExpandAstepBrcbAll(ub.brcbUb, ub.bWorkUb, ub.diagBrcbUb, geo.bs, t.diag);
        EliminateColumnsIdentityVec(ub.bUb, ub, geo.bs, forward, t.diag);
        SyncVToMte3();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopy(invGm[static_cast<uint64_t>(b) * UB_TILE_ELEMS], ub.bUb, bsAligned * UB_TILE_STRIDE);
        SyncMte3ToNext();
    }
}

struct GemmTileLayout {
    uint32_t mTileStep;
    uint32_t nTileStep;
    uint32_t mTiles;
    uint32_t nTiles;
    uint32_t tilesPerBatch;
};

__aicore__ inline GemmTileLayout ComputeGemmTileLayout(uint32_t rmM, uint32_t rmN, uint32_t batchCount)
{
    const uint32_t step = (batchCount >= 16U && batchCount < 80U) ? 256U : STRSMBATCHED_GEMM_TILE;
    const uint32_t mTiles = CeilDivU(rmM, step);
    const uint32_t nTiles = CeilDivU(rmN, step);
    return {step, step, mTiles, nTiles, mTiles * nTiles};
}

struct GemmTaskCoord {
    uint32_t b;
    uint32_t row;
    uint32_t col;
    uint32_t singleM;
    uint32_t singleN;
};

__aicore__ inline GemmTaskCoord GetGemmTaskCoord(
    uint32_t task, const GemmTileLayout& layout, uint32_t rmM, uint32_t rmN)
{
    GemmTaskCoord coord{task, 0U, 0U, 0U, 0U};
    if (layout.tilesPerBatch > 1U) {
        coord.b = task / layout.tilesPerBatch;
        const uint32_t tileIdx = task % layout.tilesPerBatch;
        coord.row = (tileIdx / layout.nTiles) * layout.mTileStep;
        coord.col = (tileIdx % layout.nTiles) * layout.nTileStep;
    }
    coord.singleM = (coord.row + layout.mTileStep <= rmM) ? layout.mTileStep : (rmM - coord.row);
    coord.singleN = (coord.col + layout.nTileStep <= rmN) ? layout.nTileStep : (rmN - coord.col);
    return coord;
}

struct GemmTaskIter {
    uint32_t cachedBatch{0xFFFFFFFFU};
    uint32_t prevM{0U};
    uint32_t prevN{0U};
    __gm__ float* leftBase{nullptr};
    __gm__ float* rightBase{nullptr};
};

template <bool L_TRANS, bool R_TRANS>
__aicore__ inline void ComputeGemmTilePtrs(
    __gm__ float* leftBase, __gm__ float* rightBase, uint32_t row, uint32_t col, uint32_t leftStride,
    uint32_t rightStride, __gm__ float*& aPtr, __gm__ float*& bPtr)
{
    if constexpr (L_TRANS) {
        aPtr = leftBase + static_cast<uint64_t>(row);
    } else {
        aPtr = leftBase + static_cast<uint64_t>(row) * leftStride;
    }
    if constexpr (R_TRANS) {
        bPtr = rightBase + static_cast<uint64_t>(col) * rightStride;
    } else {
        bPtr = rightBase + static_cast<uint64_t>(col);
    }
}

template <bool L_TRANS, bool R_TRANS, typename MM>
__aicore__ inline void InitGemmStridesAndOrgShape(
    MM& mm, uint32_t leftStride, uint32_t rightStride, uint32_t rmM, uint32_t rmN, uint32_t rmK, uint32_t cLd)
{
    int leftM = static_cast<int>(rmM);
    int leftK = static_cast<int>(leftStride);
    if constexpr (L_TRANS) {
        leftM = static_cast<int>(leftStride);
        leftK = static_cast<int>(rmK);
    }
    int rightK = static_cast<int>(rightStride);
    int rightN = static_cast<int>(rmN);
    if constexpr (!R_TRANS) {
        rightK = static_cast<int>(rmK);
        rightN = static_cast<int>(rightStride);
    }
    mm.SetOrgShape(leftM, rightN, leftK, rightK, static_cast<int>(cLd));
}

template <bool L_TRANS, bool R_TRANS, typename MM>
__aicore__ inline void IterateGemmTile(
    MM& mm, __gm__ float* aPtr, __gm__ float* bPtr, __gm__ float* cPtr, uint32_t singleM, uint32_t singleN,
    uint32_t rmK, uint32_t& prevM, uint32_t& prevN)
{
    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> cG;
    aG.SetGlobalBuffer(aPtr);
    bG.SetGlobalBuffer(bPtr);
    cG.SetGlobalBuffer(cPtr);
    if (singleM != prevM || singleN != prevN) {
        mm.SetSingleShape(static_cast<int>(singleM), static_cast<int>(singleN), static_cast<int>(rmK));
        prevM = singleM;
        prevN = singleN;
    }
    mm.SetTensorA(aG, L_TRANS);
    mm.SetTensorB(bG, R_TRANS);
    mm.IterateAll(cG, 0);
    mm.End();
}

__aicore__ inline void ResolvePanelGemmPtrs(
    GM_ADDR bPtrArray, __gm__ float* invAll, const StrsmbatchedTilingData& t, uint32_t kPanel, uint32_t b,
    __gm__ float*& leftBase, __gm__ float*& rightBase)
{
    __gm__ float* bBatch = GetBatchPtrB(bPtrArray, b, t);
    __gm__ float* invBatch = invAll + static_cast<uint64_t>(b) * UB_TILE_ELEMS;
    if (t.side == 0U) {
        leftBase = bBatch + kPanel;
        rightBase = invBatch;
    } else {
        leftBase = invBatch;
        rightBase = bBatch + static_cast<uint64_t>(kPanel) * t.ldb;
    }
}

template <bool L_TRANS, bool R_TRANS, typename MM>
__aicore__ inline void RunPanelGemmImpl(
    MM& mm, GM_ADDR bPtrArray, __gm__ float* invAll, const StrsmbatchedTilingData& t, const StepGeometry& geo)
{
    const uint32_t aicNum = static_cast<uint32_t>(GetBlockNum());
    const uint32_t rmM = (t.side == 0U) ? t.n : geo.bs;
    const uint32_t rmN = (t.side == 0U) ? geo.bs : t.m;
    const uint32_t rmK = geo.bs;
    if (aicNum == 0U || rmM == 0U || rmN == 0U || rmK == 0U) {
        return;
    }
    const GemmTileLayout layout = ComputeGemmTileLayout(rmM, rmN, t.batchCount);
    if (layout.tilesPerBatch == 0U || layout.nTiles == 0U) {
        return;
    }
    const uint32_t leftStride = (t.side == 0U) ? t.ldb : UB_TILE_STRIDE;
    const uint32_t rightStride = (t.side == 0U) ? UB_TILE_STRIDE : t.ldb;
    InitGemmStridesAndOrgShape<L_TRANS, R_TRANS>(mm, leftStride, rightStride, rmM, rmN, rmK, t.ldb);
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(t.batchCount * layout.tilesPerBatch, aicNum, blk, taskStart, taskEnd)) {
        return;
    }
    GemmTaskIter iter;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        const GemmTaskCoord coord = GetGemmTaskCoord(task, layout, rmM, rmN);
        if (coord.b != iter.cachedBatch) {
            ResolvePanelGemmPtrs(bPtrArray, invAll, t, geo.kPanel, coord.b, iter.leftBase, iter.rightBase);
            iter.cachedBatch = coord.b;
        }
        __gm__ float* aPtr = nullptr;
        __gm__ float* bPtr = nullptr;
        ComputeGemmTilePtrs<L_TRANS, R_TRANS>(
            iter.leftBase, iter.rightBase, coord.row, coord.col, leftStride, rightStride, aPtr, bPtr);
        __gm__ float* cPtr = (t.side == 0U) ? aPtr : bPtr;
        IterateGemmTile<L_TRANS, R_TRANS>(
            mm, aPtr, bPtr, cPtr, coord.singleM, coord.singleN, rmK, iter.prevM, iter.prevN);
    }
}

__aicore__ inline void ResolveTrailGemmPtrs(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t, const StepGeometry& geo, uint32_t b,
    __gm__ float*& leftBase, __gm__ float*& rightBase)
{
    __gm__ float* aBatch = GetBatchPtrA(aPtrArray, b, t);
    __gm__ float* bBatch = GetBatchPtrB(bPtrArray, b, t);
    if (t.side == 0U) {
        leftBase = bBatch + geo.gemmKPanel;
        rightBase = (t.trans == 0U) ? (aBatch + static_cast<uint64_t>(geo.gemmKPanel) * t.lda + geo.kTrail) :
                                      (aBatch + static_cast<uint64_t>(geo.kTrail) * t.lda + geo.gemmKPanel);
    } else {
        leftBase = (t.trans == 0U) ? (aBatch + static_cast<uint64_t>(geo.kTrail) * t.lda + geo.gemmKPanel) :
                                     (aBatch + static_cast<uint64_t>(geo.gemmKPanel) * t.lda + geo.kTrail);
        rightBase = bBatch + static_cast<uint64_t>(geo.gemmKPanel) * t.ldb;
    }
}

__aicore__ inline bool ComputeTrailingGemmRange(
    uint32_t batchCount, uint32_t rhsLen, uint32_t tilesPerBatch, uint32_t aicNum, uint32_t aicIdx, uint32_t& taskStart,
    uint32_t& taskEnd)
{
    const bool batchFirst = (batchCount >= 16U) && (rhsLen <= 256U);
    const uint32_t partTasks = batchFirst ? batchCount : (batchCount * tilesPerBatch);
    uint32_t partStart = 0U;
    uint32_t partEnd = 0U;
    if (!ComputeLinearTaskRange(partTasks, aicNum, aicIdx, partStart, partEnd)) {
        return false;
    }
    taskStart = batchFirst ? (partStart * tilesPerBatch) : partStart;
    taskEnd = batchFirst ? (partEnd * tilesPerBatch) : partEnd;
    return true;
}

template <bool L_TRANS, bool R_TRANS, typename MM>
__aicore__ inline void RunTrailingGemmImpl(
    MM& mm, GM_ADDR aPtrArray, GM_ADDR bPtrArray, GM_ADDR tempWorkspace, const StrsmbatchedTilingData& t,
    const StepGeometry& geo)
{
    const uint32_t aicNum = static_cast<uint32_t>(GetBlockNum());
    const uint32_t rmM = (t.side == 0U) ? t.n : geo.trailLen;
    const uint32_t rmN = (t.side == 0U) ? geo.trailLen : t.m;
    const uint32_t rmK = geo.gemmBs;
    if (aicNum == 0U || rmM == 0U || rmN == 0U || rmK == 0U) {
        return;
    }
    const GemmTileLayout layout = ComputeGemmTileLayout(rmM, rmN, t.batchCount);
    if (layout.tilesPerBatch == 0U || layout.nTiles == 0U) {
        return;
    }
    const uint32_t leftStride = (t.side == 0U) ? t.ldb : t.lda;
    const uint32_t rightStride = (t.side == 0U) ? t.lda : t.ldb;
    const uint32_t stepTempLd = CeilDivU(rmN, 16U) * 16U;
    const uint64_t stepTempBatchStride = static_cast<uint64_t>(rmM) * stepTempLd;
    InitGemmStridesAndOrgShape<L_TRANS, R_TRANS>(mm, leftStride, rightStride, rmM, rmN, rmK, stepTempLd);
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t rhsLen = (t.side == 0U) ? t.n : t.m;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeTrailingGemmRange(t.batchCount, rhsLen, layout.tilesPerBatch, aicNum, blk, taskStart, taskEnd)) {
        return;
    }
    __gm__ float* tempAll = reinterpret_cast<__gm__ float*>(tempWorkspace);
    GemmTaskIter iter;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        const GemmTaskCoord coord = GetGemmTaskCoord(task, layout, rmM, rmN);
        if (coord.b != iter.cachedBatch) {
            ResolveTrailGemmPtrs(aPtrArray, bPtrArray, t, geo, coord.b, iter.leftBase, iter.rightBase);
            iter.cachedBatch = coord.b;
        }
        __gm__ float* aPtr = nullptr;
        __gm__ float* bPtr = nullptr;
        ComputeGemmTilePtrs<L_TRANS, R_TRANS>(
            iter.leftBase, iter.rightBase, coord.row, coord.col, leftStride, rightStride, aPtr, bPtr);
        const uint64_t cOff = static_cast<uint64_t>(coord.b) * stepTempBatchStride +
                              static_cast<uint64_t>(coord.row) * stepTempLd + coord.col;
        __gm__ float* cPtr = tempAll + cOff;
        IterateGemmTile<L_TRANS, R_TRANS>(
            mm, aPtr, bPtr, cPtr, coord.singleM, coord.singleN, rmK, iter.prevM, iter.prevN);
    }
}

struct SubtractTileLayout {
    uint32_t rows;
    uint32_t stepTempLd;
    uint64_t stepTempBatchStride;
    uint32_t rStep;
    uint32_t cStep;
    uint32_t cTiles;
    uint32_t tilesPerBatch;
};

__aicore__ inline SubtractTileLayout ComputeSubtractTileLayout(
    uint32_t rawRows, uint32_t cols, const StrsmbatchedTilingData& t, const StepGeometry& geo)
{
    constexpr uint32_t bigChunkElems = 4U * UB_TILE_ELEMS;
    const uint32_t stepTempLd = CeilDivU(rawRows, 16U) * 16U;
    const bool useAlignedRows =
        ((t.reserved & 2U) != 0U) && ((t.ldb & 15U) == 0U) &&
        ((t.side == 1U) ? (stepTempLd <= t.ldb) : ((geo.kTrail & 7U) == 0U && geo.kTrail + stepTempLd <= t.ldb));
    const uint32_t rows = useAlignedRows ? stepTempLd : rawRows;
    const uint64_t stepTempBatchStride = static_cast<uint64_t>(cols) * stepTempLd;
    const uint32_t rStep = (rows <= bigChunkElems) ? rows : bigChunkElems;
    const uint32_t rTiles = CeilDivU(rows, rStep);
    const uint32_t colStrideMax = CeilDivU(rStep, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS;
    const uint32_t cStepRaw =
        (colStrideMax > 0U && bigChunkElems >= colStrideMax) ? (bigChunkElems / colStrideMax) : 1U;
    const uint32_t cStep = (cStepRaw > 256U) ? 256U : cStepRaw;
    const uint32_t cTiles = CeilDivU(cols, cStep);
    return {rows, stepTempLd, stepTempBatchStride, rStep, cStep, cTiles, rTiles * cTiles};
}

__aicore__ inline void SubtractTileUb(
    GlobalTensor<float>& bGm, GlobalTensor<float>& tGm, uint64_t bOff, uint64_t tOff, uint32_t ldb, uint32_t stepTempLd,
    uint32_t rCnt, uint32_t cCnt, uint32_t colStride, LocalTensor<float>& bBigUb, LocalTensor<float>& tBigUb)
{
    LoadColChunkUb(bBigUb, bGm, bOff, ldb, rCnt, cCnt);
    LoadColChunkUb(tBigUb, tGm, tOff, stepTempLd, colStride, cCnt);
    SyncMte2ToVOnly();
    Sub(bBigUb, bBigUb, tBigUb, static_cast<int32_t>(cCnt * colStride));
    PipeBarrier<PIPE_V>();
    SyncVToMte3();
    StoreColChunkUb(bGm, bOff, ldb, bBigUb, rCnt, cCnt);
    SyncMte3ToNext();
}

__aicore__ inline void SubtractTileStep(
    uint32_t task, const SubtractTileLayout& layout, uint32_t cols, uint64_t bSubBase, const StrsmbatchedTilingData& t,
    GM_ADDR bPtrArray, uint32_t& cachedBatch, GlobalTensor<float>& bGm, GlobalTensor<float>& tGm,
    LocalTensor<float>& bBigUb, LocalTensor<float>& tBigUb)
{
    uint32_t b = task;
    uint32_t rBase = 0U;
    uint32_t cBase = 0U;
    if (layout.tilesPerBatch > 1U && layout.cTiles > 0U) {
        b = task / layout.tilesPerBatch;
        const uint32_t tileIdx = task % layout.tilesPerBatch;
        rBase = (tileIdx / layout.cTiles) * layout.rStep;
        cBase = (tileIdx % layout.cTiles) * layout.cStep;
    }
    const uint32_t rCnt = (rBase + layout.rStep <= layout.rows) ? layout.rStep : (layout.rows - rBase);
    const uint32_t cCnt = (cBase + layout.cStep <= cols) ? layout.cStep : (cols - cBase);
    const uint32_t colStride = CeilDivU(rCnt, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS;
    const uint64_t bOff = bSubBase + static_cast<uint64_t>(cBase) * t.ldb + rBase;
    const uint64_t tOff = static_cast<uint64_t>(b) * layout.stepTempBatchStride +
                          static_cast<uint64_t>(cBase) * layout.stepTempLd + rBase;
    if (b != cachedBatch) {
        bGm.SetGlobalBuffer(GetBatchPtrB(bPtrArray, b, t));
        cachedBatch = b;
    }
    SubtractTileUb(bGm, tGm, bOff, tOff, t.ldb, layout.stepTempLd, rCnt, cCnt, colStride, bBigUb, tBigUb);
}

__aicore__ inline void SubtractTrailingAiv(
    GM_ADDR bPtrArray, GM_ADDR tempWorkspace, const StrsmbatchedTilingData& t, const StepGeometry& geo,
    AivUbContext& ub)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    const uint32_t rawRows = (t.side == 0U) ? geo.trailLen : t.m;
    const uint32_t cols = (t.side == 0U) ? t.n : geo.trailLen;
    if (aivNum == 0U || rawRows == 0U || cols == 0U) {
        return;
    }
    constexpr uint32_t bigChunkElems = 4U * UB_TILE_ELEMS;
    LocalTensor<float> bBigUb = ub.brcbUb;
    LocalTensor<float> tBigUb = ub.brcbUb[bigChunkElems];
    const SubtractTileLayout layout = ComputeSubtractTileLayout(rawRows, cols, t, geo);
    if (layout.tilesPerBatch == 0U || layout.cTiles == 0U) {
        return;
    }
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(t.batchCount * layout.tilesPerBatch, aivNum, blk, taskStart, taskEnd)) {
        return;
    }
    __gm__ float* tempAll = reinterpret_cast<__gm__ float*>(tempWorkspace);
    uint32_t cachedBatch = 0xFFFFFFFFU;
    GlobalTensor<float> bGm;
    GlobalTensor<float> tGm;
    tGm.SetGlobalBuffer(tempAll);
    const uint64_t bSubBase =
        (t.side == 0U) ? static_cast<uint64_t>(geo.kTrail) : (static_cast<uint64_t>(geo.kTrail) * t.ldb);
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        SubtractTileStep(task, layout, cols, bSubBase, t, bPtrArray, cachedBatch, bGm, tGm, bBigUb, tBigUb);
    }
}

struct MatrixTileTask {
    uint32_t b;
    uint32_t rBase;
    uint32_t cBase;
    uint32_t rCnt;
    uint32_t cCnt;
};

__aicore__ inline MatrixTileTask DecomposeMatrixTask(
    uint32_t task, uint32_t tilesPerBatch, uint32_t cTiles, uint32_t rStep, uint32_t cStep, uint32_t rows,
    uint32_t cols)
{
    uint32_t b = task;
    uint32_t rBase = 0U;
    uint32_t cBase = 0U;
    if (tilesPerBatch > 1U && cTiles > 0U) {
        b = task / tilesPerBatch;
        const uint32_t tileIdx = task % tilesPerBatch;
        rBase = (tileIdx / cTiles) * rStep;
        cBase = (tileIdx % cTiles) * cStep;
    }
    const uint32_t rCnt = (rBase + rStep <= rows) ? rStep : (rows - rBase);
    const uint32_t cCnt = (cBase + cStep <= cols) ? cStep : (cols - cBase);
    return {b, rBase, cBase, rCnt, cCnt};
}

__aicore__ inline bool ClipTriangularTile(
    int32_t triUplo, uint32_t cBase, uint32_t cCnt, uint32_t rows, uint32_t& rBase, uint32_t& rCnt)
{
    const uint32_t pColFirst = cBase / STRSMBATCHED_PANEL_NB;
    const uint32_t pColLast = (cBase + cCnt - 1U) / STRSMBATCHED_PANEL_NB;
    if (triUplo == 0) {
        const uint32_t rMaxLim = (pColLast + 1U) * STRSMBATCHED_PANEL_NB;
        const uint32_t rMax = (rMaxLim < rows) ? rMaxLim : rows;
        if (rBase >= rMax) {
            return false;
        }
        if (rBase + rCnt > rMax) {
            rCnt = rMax - rBase;
        }
    } else if (triUplo == 1) {
        const uint32_t rMin = pColFirst * STRSMBATCHED_PANEL_NB;
        if (rBase + rCnt <= rMin) {
            return false;
        }
        if (rBase < rMin) {
            rCnt = (rBase + rCnt) - rMin;
            rBase = rMin;
        }
    }
    return true;
}

__aicore__ inline void PackTilePingPong(
    GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, uint64_t srcOff, uint64_t dstOff, uint32_t cCnt,
    uint32_t dstLd, uint32_t ubColStride, bool rTilesSingle, int32_t triUplo, const DataCopyExtParams& inParams,
    const DataCopyPadExtParams<float>& padParams, LocalTensor<float>& brcbUb, bool bufBusy[2], uint32_t& curBuf)
{
    const event_t ev = (curBuf == 0U) ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<float> curUb = (curBuf == 0U) ? brcbUb : brcbUb[4U * UB_TILE_ELEMS];
    if (bufBusy[curBuf]) {
        WaitFlag<HardEvent::MTE3_MTE2>(ev);
    }
    DataCopyPad(curUb, srcGm[srcOff], inParams, padParams);
    SetFlag<HardEvent::MTE2_MTE3>(ev);
    WaitFlag<HardEvent::MTE2_MTE3>(ev);
    if (rTilesSingle && triUplo < 0) {
        DataCopy(dstGm[dstOff], curUb, cCnt * dstLd);
    } else {
        StoreColChunkUb(dstGm, dstOff, dstLd, curUb, ubColStride, cCnt);
    }
    SetFlag<HardEvent::MTE3_MTE2>(ev);
    bufBusy[curBuf] = true;
    curBuf ^= 1U;
}

__aicore__ inline void PackTileScaled(
    GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, uint64_t srcOff, uint64_t dstOff, uint32_t cCnt,
    uint32_t dstLd, uint32_t ubColStride, bool rTilesSingle, int32_t triUplo, float scaleVal,
    const DataCopyExtParams& inParams, const DataCopyPadExtParams<float>& padParams, LocalTensor<float>& bufUb)
{
    DataCopyPad(bufUb, srcGm[srcOff], inParams, padParams);
    SyncMte2ToVOnly();
    Muls(bufUb, bufUb, scaleVal, static_cast<int32_t>(cCnt * ubColStride));
    PipeBarrier<PIPE_V>();
    SyncVToMte3();
    if (rTilesSingle && triUplo < 0) {
        DataCopy(dstGm[dstOff], bufUb, cCnt * dstLd);
    } else {
        StoreColChunkUb(dstGm, dstOff, dstLd, bufUb, ubColStride, cCnt);
    }
    SyncMte3ToNext();
}

struct MatrixPackingPlan {
    uint32_t rStep;
    uint32_t rTiles;
    uint32_t cStep;
    uint32_t cTiles;
    uint32_t tilesPerBatch;
};

__aicore__ inline MatrixPackingPlan ComputeMatrixPackingPlan(
    uint32_t rows, uint32_t cols, uint32_t alignLd, uint32_t bigBufElems, uint32_t rStepLim, uint32_t cStepLim)
{
    const uint32_t rStep = (rows <= rStepLim) ? rows : rStepLim;
    const uint32_t rTiles = CeilDivU(rows, rStep);
    const uint32_t rStrideMax = (rTiles == 1U) ? alignLd : (CeilDivU(rStep, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS);
    const uint32_t cStepRaw = (rStrideMax > 0U && bigBufElems >= rStrideMax) ? (bigBufElems / rStrideMax) : 1U;
    const uint32_t cStep = (cStepRaw > cStepLim) ? cStepLim : cStepRaw;
    const uint32_t cTiles = CeilDivU(cols, cStep);
    return {rStep, rTiles, cStep, cTiles, rTiles * cTiles};
}

__aicore__ inline void WaitPingPongFinished(bool bufBusy[2])
{
    if (bufBusy[0]) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (bufBusy[1]) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
}

__aicore__ inline void PackMatrixTileStep(
    GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, GM_ADDR srcPtrArray, const MatrixTileTask& mt,
    const MatrixPackingPlan& plan, uint32_t srcLd, uint32_t dstLd, uint64_t dstBatchStride, float scaleVal,
    int32_t triUplo, bool usePingPong, AivUbContext& ub, bool bufBusy[2], uint32_t& curBuf, uint32_t& cachedBatch)
{
    const uint32_t rBlocks = CeilDivU(mt.rCnt, FLOAT_BLOCK_ELEMS);
    const uint32_t ubColStride = (plan.rTiles == 1U && triUplo < 0) ? dstLd : (rBlocks * FLOAT_BLOCK_ELEMS);
    const uint32_t ubGapBlocks = (ubColStride / FLOAT_BLOCK_ELEMS) - rBlocks;
    const uint32_t srcGapBytes = (srcLd >= mt.rCnt) ? ((srcLd - mt.rCnt) * static_cast<uint32_t>(sizeof(float))) : 0U;
    if (mt.b != cachedBatch) {
        srcGm.SetGlobalBuffer(GetBatchPtr(srcPtrArray, mt.b));
        cachedBatch = mt.b;
    }
    const uint64_t srcOff = static_cast<uint64_t>(mt.cBase) * srcLd + mt.rBase;
    const uint64_t dstOff =
        static_cast<uint64_t>(mt.b) * dstBatchStride + static_cast<uint64_t>(mt.cBase) * dstLd + mt.rBase;
    const DataCopyExtParams inParams{
        static_cast<uint16_t>(mt.cCnt), mt.rCnt * static_cast<uint32_t>(sizeof(float)), srcGapBytes, ubGapBlocks, 0U};
    const DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0f};
    if (usePingPong) {
        PackTilePingPong(
            srcGm, dstGm, srcOff, dstOff, mt.cCnt, dstLd, ubColStride, (plan.rTiles == 1U), triUplo, inParams,
            padParams, ub.brcbUb, bufBusy, curBuf);
    } else {
        PackTileScaled(
            srcGm, dstGm, srcOff, dstOff, mt.cCnt, dstLd, ubColStride, (plan.rTiles == 1U), triUplo, scaleVal, inParams,
            padParams, ub.brcbUb);
    }
}

__aicore__ inline void PackMatrixToAlignedAiv(
    GM_ADDR srcPtrArray, __gm__ float* dstContigAll, uint32_t rows, uint32_t cols, uint32_t srcLd, uint32_t dstLd,
    uint32_t batchCount, float scaleVal, int32_t triUplo, AivUbContext& ub)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    if (aivNum == 0U || rows == 0U || cols == 0U) {
        return;
    }
    const bool usePingPong = (scaleVal == 1.0f);
    const uint32_t bigBufElems = usePingPong ? (4U * UB_TILE_ELEMS) : (8U * UB_TILE_ELEMS);
    const uint32_t rStepLim = (triUplo >= 0) ? 512U : 2048U;
    const uint32_t cStepLim = (triUplo >= 0) ? 32U : 256U;
    const MatrixPackingPlan plan = ComputeMatrixPackingPlan(rows, cols, dstLd, bigBufElems, rStepLim, cStepLim);
    if (plan.tilesPerBatch == 0U || plan.cTiles == 0U) {
        return;
    }
    const uint64_t dstBatchStride = static_cast<uint64_t>(dstLd) * cols;
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(batchCount * plan.tilesPerBatch, aivNum, blk, taskStart, taskEnd)) {
        return;
    }
    uint32_t cachedBatch = 0xFFFFFFFFU;
    GlobalTensor<float> srcGm;
    GlobalTensor<float> dstGm;
    dstGm.SetGlobalBuffer(dstContigAll);
    bool bufBusy[2] = {false, false};
    uint32_t curBuf = 0U;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        MatrixTileTask mt =
            DecomposeMatrixTask(task, plan.tilesPerBatch, plan.cTiles, plan.rStep, plan.cStep, rows, cols);
        if (!ClipTriangularTile(triUplo, mt.cBase, mt.cCnt, rows, mt.rBase, mt.rCnt)) {
            continue;
        }
        PackMatrixTileStep(
            srcGm, dstGm, srcPtrArray, mt, plan, srcLd, dstLd, dstBatchStride, scaleVal, triUplo, usePingPong, ub,
            bufBusy, curBuf, cachedBatch);
    }
    WaitPingPongFinished(bufBusy);
}

__aicore__ inline void UnpackTilePingPong(
    GlobalTensor<float>& srcGm, GlobalTensor<float>& dstGm, uint64_t srcOff, uint64_t dstOff, uint32_t srcLd,
    uint32_t ubColStride, uint32_t cCnt, bool rTilesSingle, const DataCopyExtParams& outParams,
    LocalTensor<float>& brcbUb, bool bufBusy[2], uint32_t& curBuf)
{
    const event_t ev = (curBuf == 0U) ? EVENT_ID0 : EVENT_ID1;
    LocalTensor<float> curUb = (curBuf == 0U) ? brcbUb : brcbUb[4U * UB_TILE_ELEMS];
    if (bufBusy[curBuf]) {
        WaitFlag<HardEvent::MTE3_MTE2>(ev);
    }
    if (rTilesSingle) {
        DataCopy(curUb, srcGm[srcOff], cCnt * srcLd);
    } else {
        LoadColChunkUb(curUb, srcGm, srcOff, srcLd, ubColStride, cCnt);
    }
    SetFlag<HardEvent::MTE2_MTE3>(ev);
    WaitFlag<HardEvent::MTE2_MTE3>(ev);
    DataCopyPad(dstGm[dstOff], curUb, outParams);
    SetFlag<HardEvent::MTE3_MTE2>(ev);
    bufBusy[curBuf] = true;
    curBuf ^= 1U;
}

__aicore__ inline void UnpackMatrixFromAlignedAiv(
    const __gm__ float* srcContigAll, GM_ADDR dstPtrArray, uint32_t rows, uint32_t cols, uint32_t srcLd, uint32_t dstLd,
    uint32_t batchCount, AivUbContext& ub)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    if (aivNum == 0U || rows == 0U || cols == 0U) {
        return;
    }
    const MatrixPackingPlan plan = ComputeMatrixPackingPlan(rows, cols, srcLd, 4U * UB_TILE_ELEMS, 2048U, 256U);
    if (plan.tilesPerBatch == 0U || plan.cTiles == 0U) {
        return;
    }
    const uint64_t srcBatchStride = static_cast<uint64_t>(srcLd) * cols;
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(batchCount * plan.tilesPerBatch, aivNum, blk, taskStart, taskEnd)) {
        return;
    }
    uint32_t cachedBatch = 0xFFFFFFFFU;
    GlobalTensor<float> srcGm;
    GlobalTensor<float> dstGm;
    srcGm.SetGlobalBuffer(const_cast<__gm__ float*>(srcContigAll));
    bool bufBusy[2] = {false, false};
    uint32_t curBuf = 0U;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        MatrixTileTask mt =
            DecomposeMatrixTask(task, plan.tilesPerBatch, plan.cTiles, plan.rStep, plan.cStep, rows, cols);
        const uint32_t rBlocks = CeilDivU(mt.rCnt, FLOAT_BLOCK_ELEMS);
        const uint32_t ubColStride = (plan.rTiles == 1U) ? srcLd : (rBlocks * FLOAT_BLOCK_ELEMS);
        const uint32_t ubGapBlocks = (ubColStride / FLOAT_BLOCK_ELEMS) - rBlocks;
        const uint32_t dstGapBytes =
            (dstLd >= mt.rCnt) ? ((dstLd - mt.rCnt) * static_cast<uint32_t>(sizeof(float))) : 0U;
        if (mt.b != cachedBatch) {
            dstGm.SetGlobalBuffer(GetBatchPtr(dstPtrArray, mt.b));
            cachedBatch = mt.b;
        }
        const uint64_t srcOff =
            static_cast<uint64_t>(mt.b) * srcBatchStride + static_cast<uint64_t>(mt.cBase) * srcLd + mt.rBase;
        const uint64_t dstOff = static_cast<uint64_t>(mt.cBase) * dstLd + mt.rBase;
        const DataCopyExtParams outParams{
            static_cast<uint16_t>(mt.cCnt), mt.rCnt * static_cast<uint32_t>(sizeof(float)), ubGapBlocks, dstGapBytes,
            0U};
        UnpackTilePingPong(
            srcGm, dstGm, srcOff, dstOff, srcLd, ubColStride, mt.cCnt, (plan.rTiles == 1U), outParams, ub.brcbUb,
            bufBusy, curBuf);
    }
    WaitPingPongFinished(bufBusy);
}

__aicore__ inline void ScaleOrZeroChunk(
    GlobalTensor<float>& bGm, uint64_t bOff, uint32_t ldb, uint32_t rCnt, uint32_t cCnt, uint32_t colStride,
    float alpha, LocalTensor<float>& bUb, bool zeroMode)
{
    if (!zeroMode) {
        LoadColChunkUb(bUb, bGm, bOff, ldb, rCnt, cCnt);
        SyncMte2ToVOnly();
        Muls(bUb, bUb, alpha, static_cast<int32_t>(cCnt * colStride));
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
    }
    StoreColChunkUb(bGm, bOff, ldb, bUb, rCnt, cCnt);
    SyncMte3ToNext();
}

__aicore__ inline void ScaleOrZeroAllBatchesAiv(
    GM_ADDR bPtrArray, const StrsmbatchedTilingData& t, LocalTensor<float>& bUb, bool zeroMode)
{
    const uint32_t aivNum = static_cast<uint32_t>(GetBlockNum() * GetTaskRation());
    if (aivNum == 0U || t.m == 0U || t.n == 0U) {
        return;
    }
    const uint32_t rStep = (t.m <= UB_TILE_ELEMS) ? t.m : UB_TILE_ELEMS;
    const uint32_t rTiles = CeilDivU(t.m, rStep);
    const uint32_t colStrideMax = CeilDivU(rStep, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS;
    const uint32_t cStepRaw =
        (colStrideMax > 0U && UB_TILE_ELEMS >= colStrideMax) ? (UB_TILE_ELEMS / colStrideMax) : 1U;
    const uint32_t cStep = (cStepRaw > 64U) ? 64U : cStepRaw;
    const uint32_t cTiles = CeilDivU(t.n, cStep);
    const uint32_t tilesPerBatch = rTiles * cTiles;
    if (tilesPerBatch == 0U || cTiles == 0U) {
        return;
    }
    if (zeroMode) {
        Duplicate(bUb, 0.0f, UB_TILE_ELEMS);
        PipeBarrier<PIPE_V>();
        SyncVToMte3();
    }
    uint32_t taskStart = 0U;
    uint32_t taskEnd = 0U;
    const uint32_t blk = static_cast<uint32_t>(GetBlockIdx());
    if (!ComputeLinearTaskRange(t.batchCount * tilesPerBatch, aivNum, blk, taskStart, taskEnd)) {
        return;
    }
    uint32_t cachedBatch = 0xFFFFFFFFU;
    GlobalTensor<float> bGm;
    for (uint32_t task = taskStart; task < taskEnd; ++task) {
        MatrixTileTask mt = DecomposeMatrixTask(task, tilesPerBatch, cTiles, rStep, cStep, t.m, t.n);
        const uint32_t colStride = CeilDivU(mt.rCnt, FLOAT_BLOCK_ELEMS) * FLOAT_BLOCK_ELEMS;
        const uint64_t bOff = static_cast<uint64_t>(mt.cBase) * t.ldb + mt.rBase;
        if (mt.b != cachedBatch) {
            bGm.SetGlobalBuffer(GetBatchPtrB(bPtrArray, mt.b, t));
            cachedBatch = mt.b;
        }
        ScaleOrZeroChunk(bGm, bOff, t.ldb, mt.rCnt, mt.cCnt, colStride, t.alpha, bUb, zeroMode);
    }
}

__aicore__ inline void InitAivBuffers(
    TPipe& pipe, TBuf<TPosition::VECCALC>& floatBuf, TBuf<TPosition::VECCALC>& idxBuf, AivUbContext& ub)
{
    constexpr uint32_t diagBrcbElems = UB_TILE_STRIDE * FLOAT_BLOCK_ELEMS;
    constexpr uint32_t totalFloatElems = 11U * UB_TILE_ELEMS + diagBrcbElems + 3U * UB_TILE_STRIDE;
    constexpr uint32_t idxElems = 32U * UB_TILE_STRIDE;
    pipe.InitBuffer(floatBuf, totalFloatElems * sizeof(float));
    pipe.InitBuffer(idxBuf, (idxElems + UB_TILE_STRIDE) * sizeof(uint32_t));
    LocalTensor<float> base = floatBuf.Get<float>();
    ub.bUb = base;
    ub.bWorkUb = base[UB_TILE_ELEMS];
    ub.tempUb = base[2U * UB_TILE_ELEMS];
    ub.brcbUb = base[3U * UB_TILE_ELEMS];
    ub.diagBrcbUb = base[11U * UB_TILE_ELEMS];
    ub.diagUb = base[11U * UB_TILE_ELEMS + diagBrcbElems];
    ub.onesUb = base[11U * UB_TILE_ELEMS + diagBrcbElems + UB_TILE_STRIDE];
    LocalTensor<uint32_t> idxBase = idxBuf.Get<uint32_t>();
    ub.idxUb = idxBase;
    ub.diagIdxUb = idxBase[idxElems];
    InitTransposeIdxUb(ub.idxUb, ub.diagIdxUb, ub.onesUb);
}

__aicore__ inline bool ShouldUsePanelGemm(
    const StrsmbatchedTilingData& t, const StepGeometry& geo, uint32_t numChunks, uint32_t numSubPanels,
    uint32_t stepIdx, bool hasPreSync)
{
    if (numSubPanels > 1U) {
        if (stepIdx > 0U || hasPreSync) {
            return ((geo.bs >= 32U) && (t.batchCount >= 16U)) ||
                   ((geo.bs >= 8U) && ((t.reserved & 2U) != 0U) && (t.batchCount * numChunks >= 512U));
        }
        return (geo.bs >= 32U) && (t.batchCount * numChunks >= 64U);
    }
    return (geo.bs >= 24U) && (numChunks >= 3U) && (t.batchCount * numChunks >= 96U);
}

struct PackPlan {
    GM_ADDR aWorkPtr;
    GM_ADDR bWorkPtr;
    __gm__ float* invAll;
    __gm__ float* aAlignedAll;
    __gm__ float* bAlignedAll;
    uint32_t ldaAligned;
    uint32_t ldbAligned;
    bool packA;
    bool packB;
    bool needPreScale;
    bool hasPreSync;
};

__aicore__ inline PackPlan InitPackPlan(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, GM_ADDR tempWorkspace, const StrsmbatchedTilingData& tIn, uint32_t aDim,
    uint32_t rhsLen, uint32_t numSubPanels, StrsmbatchedTilingData& t)
{
    PackPlan plan{};
    plan.invAll =
        reinterpret_cast<__gm__ float*>(tempWorkspace) + static_cast<uint64_t>(tIn.batchCount) * tIn.tempBatchStride;
    plan.packA = (numSubPanels > 1U) && ((tIn.lda & 7U) != 0U);
    plan.packB = (numSubPanels > 1U) && ((tIn.ldb & 7U) != 0U) && (rhsLen > 64U);
    plan.aAlignedAll = plan.invAll + static_cast<uint64_t>(tIn.batchCount) * 2ULL * UB_TILE_ELEMS;
    plan.ldaAligned = plan.packA ? (CeilDivU(aDim, 16U) * 16U) : tIn.lda;
    plan.ldbAligned = plan.packB ? (CeilDivU(tIn.m, 16U) * 16U) : tIn.ldb;
    plan.bAlignedAll =
        plan.aAlignedAll + (plan.packA ? (static_cast<uint64_t>(tIn.batchCount) * plan.ldaAligned * aDim) : 0ULL);
    plan.aWorkPtr = aPtrArray;
    plan.bWorkPtr = bPtrArray;
    if (plan.packB) {
        t.ldb = plan.ldbAligned;
        t.reserved |= 2U;
        t.alpha = 1.0f;
        plan.bWorkPtr = reinterpret_cast<GM_ADDR>(plan.bAlignedAll);
    }
    if (plan.packA) {
        t.lda = plan.ldaAligned;
        t.reserved |= 1U;
        plan.aWorkPtr = reinterpret_cast<GM_ADDR>(plan.aAlignedAll);
    }
    plan.needPreScale = (numSubPanels > 1U) && !plan.packB && (t.alpha != 1.0f);
    plan.hasPreSync = plan.packA || plan.packB || plan.needPreScale;
    return plan;
}

__aicore__ inline void ExecutePreSyncPack(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const PackPlan& plan, const StrsmbatchedTilingData& tIn,
    const StrsmbatchedTilingData& t, const StepGeometry& geo0, uint32_t aDim, uint32_t numSubPanels, bool usePanelGemm0,
    AivUbContext& ub)
{
    const bool doScale0 = (numSubPanels == 1U) && (t.alpha != 1.0f);
    StrsmbatchedTilingData tPre = tIn;
    if (plan.packB) {
        tPre.ldb = plan.ldbAligned;
        tPre.reserved |= 2U;
        tPre.alpha = 1.0f;
    }
    if ASCEND_IS_AIV {
        if (plan.packA) {
            PackMatrixToAlignedAiv(
                aPtrArray, plan.aAlignedAll, aDim, aDim, tIn.lda, plan.ldaAligned, tIn.batchCount, 1.0f,
                static_cast<int32_t>(tIn.uplo), ub);
        }
        if (plan.packB) {
            PackMatrixToAlignedAiv(
                bPtrArray, plan.bAlignedAll, tIn.m, tIn.n, tIn.ldb, plan.ldbAligned, tIn.batchCount, tIn.alpha, -1, ub);
        }
        if (plan.needPreScale) {
            ScaleOrZeroAllBatchesAiv(plan.bWorkPtr, t, ub.bUb, false);
        }
        if (usePanelGemm0) {
            InvertPanelStepAiv(aPtrArray, plan.invAll, tPre, geo0, doScale0 ? tPre.alpha : 1.0f, ub);
        }
    }
    SyncAll<false>();
}

template <bool L_TRANS, bool R_TRANS, typename MatmulT>
__aicore__ inline void ExecuteStepPanelSolve(
    MatmulT& mm, GM_ADDR aWorkPtr, GM_ADDR bWorkPtr, __gm__ float* invCur, const StrsmbatchedTilingData& t,
    const StepGeometry& geo, bool usePanelGemm, bool doScale, AivUbContext& ub)
{
    if (usePanelGemm) {
        if ASCEND_IS_AIC {
            RunPanelGemmImpl<L_TRANS, R_TRANS>(mm, bWorkPtr, invCur, t, geo);
        }
    } else {
        if ASCEND_IS_AIV {
            SolvePanelStepAiv(aWorkPtr, bWorkPtr, t, geo, doScale, ub);
        }
    }
}

template <bool L_TRANS, bool R_TRANS, typename MatmulT>
__aicore__ inline void ExecuteStepTrailingUpdate(
    MatmulT& mm, GM_ADDR aWorkPtr, GM_ADDR bWorkPtr, GM_ADDR tempWorkspace, __gm__ float* invNext,
    const StrsmbatchedTilingData& t, const StepGeometry& geo, const StepGeometry& nextGeo, bool useNextPanelGemm,
    bool fusePanelTrail, AivUbContext& ub)
{
    if (!fusePanelTrail) {
        SyncAll<false>();
    } else {
        if ASCEND_IS_AIC {
            PipeBarrier<PIPE_ALL>();
        }
    }
    if ASCEND_IS_AIC {
        RunTrailingGemmImpl<L_TRANS, R_TRANS>(mm, aWorkPtr, bWorkPtr, tempWorkspace, t, geo);
    }
    if (useNextPanelGemm) {
        if ASCEND_IS_AIV {
            InvertPanelStepAiv(aWorkPtr, invNext, t, nextGeo, 1.0f, ub);
        }
    }
    SyncAll<false>();
    if ASCEND_IS_AIV {
        SubtractTrailingAiv(bWorkPtr, tempWorkspace, t, geo, ub);
    }
    SyncAll<false>();
}

template <typename MM>
__aicore__ inline void InitMatmulInstance(MM& mm, TPipe& pipe)
{
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
        mm.DisableBias();
        mm.SetHF32(false);
    }
}

template <bool L_TRANS, bool R_TRANS>
__aicore__ inline void RunBlockedMixLoop(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, GM_ADDR tempWorkspace, const StrsmbatchedTilingData& tIn, TPipe& pipe,
    AivUbContext& ub)
{
    MatmulImpl<MmInput<L_TRANS>, MmInput<R_TRANS>, MmOutput, MmOutput, MM_TILING<L_TRANS, R_TRANS>> mm;
    InitMatmulInstance(mm, pipe);
    const uint32_t aDim = (tIn.side == 0U) ? tIn.m : tIn.n;
    const uint32_t rhsLen = (tIn.side == 0U) ? tIn.n : tIn.m;
    const uint32_t numChunks = CeilDivU(rhsLen, UB_TILE_STRIDE);
    const uint32_t panelNb = (tIn.panelNb == 0U) ? STRSMBATCHED_PANEL_NB : tIn.panelNb;
    const uint32_t groupPanels = (tIn.groupPanels == 0U) ? 1U : tIn.groupPanels;
    const uint32_t numSubPanels = CeilDivU(aDim, panelNb);
    const bool forward = IsForwardSweep(tIn);
    StrsmbatchedTilingData t = tIn;
    const PackPlan plan = InitPackPlan(aPtrArray, bPtrArray, tempWorkspace, tIn, aDim, rhsLen, numSubPanels, t);
    const StepGeometry geo0 = ComputeStepGeometry(aDim, panelNb, groupPanels, numSubPanels, forward, 0U);
    const bool usePanelGemm0 = ShouldUsePanelGemm(t, geo0, numChunks, numSubPanels, 0U, plan.hasPreSync);
    if (plan.hasPreSync || usePanelGemm0) {
        ExecutePreSyncPack(aPtrArray, bPtrArray, plan, tIn, t, geo0, aDim, numSubPanels, usePanelGemm0, ub);
    }
    for (uint32_t s = 0U; s < numSubPanels; ++s) {
        const StepGeometry geo = ComputeStepGeometry(aDim, panelNb, groupPanels, numSubPanels, forward, s);
        const bool doScale = (numSubPanels == 1U) && (t.alpha != 1.0f);
        const bool usePanelGemm = ShouldUsePanelGemm(t, geo, numChunks, numSubPanels, s, plan.hasPreSync);
        __gm__ float* invCur = plan.invAll + static_cast<uint64_t>(s & 1U) * t.batchCount * UB_TILE_ELEMS;
        __gm__ float* invNext = plan.invAll + static_cast<uint64_t>((s + 1U) & 1U) * t.batchCount * UB_TILE_ELEMS;
        ExecuteStepPanelSolve<L_TRANS, R_TRANS>(
            mm, plan.aWorkPtr, plan.bWorkPtr, invCur, t, geo, usePanelGemm, doScale, ub);
        if (!geo.hasTrail) {
            break;
        }
        const bool fusePanelTrail = usePanelGemm && (t.batchCount >= 16U) &&
                                    ((rhsLen <= 256U) || (t.batchCount >= 80U && rhsLen <= STRSMBATCHED_GEMM_TILE &&
                                                          geo.trailLen <= STRSMBATCHED_GEMM_TILE));
        const StepGeometry nextGeo = ComputeStepGeometry(aDim, panelNb, groupPanels, numSubPanels, forward, s + 1U);
        const bool useNextPanelGemm = ShouldUsePanelGemm(t, nextGeo, numChunks, numSubPanels, s + 1U, plan.hasPreSync);
        ExecuteStepTrailingUpdate<L_TRANS, R_TRANS>(
            mm, plan.aWorkPtr, plan.bWorkPtr, tempWorkspace, invNext, t, geo, nextGeo, useNextPanelGemm, fusePanelTrail,
            ub);
    }
    if (plan.packB) {
        SyncAll<false>();
        if ASCEND_IS_AIV {
            UnpackMatrixFromAlignedAiv(
                plan.bAlignedAll, bPtrArray, tIn.m, tIn.n, plan.ldbAligned, tIn.ldb, tIn.batchCount, ub);
        }
    }
}

__aicore__ __attribute__((noinline)) void SolveSmallFallbackAiv(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& t)
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    AivUbContext ub;
    InitAivBuffers(pipe, floatBuf, idxBuf, ub);
    const uint32_t aDim = (t.side == 0U) ? t.m : t.n;
    const StepGeometry geo{0U, aDim, false, 0U, 0U, 0U, 0U};
    SolvePanelStepAiv(aPtrArray, bPtrArray, t, geo, true, ub);
    PipeBarrier<PIPE_ALL>();
}

} // namespace

extern "C" __global__ __aicore__ void strsmbatched_zero_kernel(GM_ADDR bPtrArray, StrsmbatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> bBuf;
    pipe.InitBuffer(bBuf, UB_TILE_ELEMS * sizeof(float));
    LocalTensor<float> bUb = bBuf.Get<float>();
    ScaleOrZeroAllBatchesAiv(bPtrArray, t, bUb, true);
    PipeBarrier<PIPE_ALL>();
}

extern "C" __global__ __aicore__ void strsmbatched_fallback_aiv_kernel(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, StrsmbatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SolveSmallFallbackAiv(aPtrArray, bPtrArray, t);
}

extern "C" __global__ __aicore__ void strsmbatched_blocked_mix_kernel(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, GM_ADDR tempWorkspace, StrsmbatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    TBuf<TPosition::VECCALC> floatBuf;
    TBuf<TPosition::VECCALC> idxBuf;
    AivUbContext ub;
    if ASCEND_IS_AIV {
        InitAivBuffers(pipe, floatBuf, idxBuf, ub);
    }
    if (t.trans == 0U) {
        RunBlockedMixLoop<false, false>(aPtrArray, bPtrArray, tempWorkspace, t, pipe, ub);
    } else if (t.side == 0U) {
        RunBlockedMixLoop<false, true>(aPtrArray, bPtrArray, tempWorkspace, t, pipe, ub);
    } else {
        RunBlockedMixLoop<true, false>(aPtrArray, bPtrArray, tempWorkspace, t, pipe, ub);
    }
    PipeBarrier<PIPE_ALL>();
}

void strsmbatched_zero_kernel_do(
    GM_ADDR bPtrArray, const StrsmbatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsmbatched_zero_kernel<<<numBlocks, nullptr, stream>>>(bPtrArray, tiling);
}

void strsmbatched_fallback_aiv_kernel_do(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, const StrsmbatchedTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsmbatched_fallback_aiv_kernel<<<numBlocks, nullptr, stream>>>(aPtrArray, bPtrArray, tiling);
}

void strsmbatched_blocked_mix_kernel_do(
    GM_ADDR aPtrArray, GM_ADDR bPtrArray, GM_ADDR tempWorkspace, const StrsmbatchedTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    strsmbatched_blocked_mix_kernel<<<numBlocks, nullptr, stream>>>(aPtrArray, bPtrArray, tempWorkspace, tiling);
}
