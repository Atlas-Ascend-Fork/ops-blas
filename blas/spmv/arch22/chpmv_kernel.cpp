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
 * \file chpmv_kernel.cpp
 * \brief Device-side two-stage SIMD AIV kernels for aclblasChpmv (arch22 / dav-2201).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "chpmv_kernel.h"

using namespace AscendC;

namespace {

constexpr uint32_t CHPMV_BUF_FLOATS = 2048U;
constexpr uint32_t CHPMV_PACK_STRIDE = 1024U;
constexpr uint32_t CHPMV_RED_TMP = 512U;

__aicore__ inline void ChpmvWaitEvent(HardEvent evt, TPipe* pipe)
{
    const event_t eid = static_cast<event_t>(pipe->FetchEventID(evt));
    if (evt == HardEvent::MTE2_V) {
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
    } else if (evt == HardEvent::V_MTE2) {
        SetFlag<HardEvent::V_MTE2>(eid);
        WaitFlag<HardEvent::V_MTE2>(eid);
    } else if (evt == HardEvent::V_MTE3) {
        SetFlag<HardEvent::V_MTE3>(eid);
        WaitFlag<HardEvent::V_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_V) {
        SetFlag<HardEvent::MTE3_V>(eid);
        WaitFlag<HardEvent::MTE3_V>(eid);
    } else if (evt == HardEvent::MTE2_S) {
        SetFlag<HardEvent::MTE2_S>(eid);
        WaitFlag<HardEvent::MTE2_S>(eid);
    } else if (evt == HardEvent::S_V) {
        SetFlag<HardEvent::S_V>(eid);
        WaitFlag<HardEvent::S_V>(eid);
    } else if (evt == HardEvent::V_S) {
        SetFlag<HardEvent::V_S>(eid);
        WaitFlag<HardEvent::V_S>(eid);
    } else if (evt == HardEvent::S_MTE3) {
        SetFlag<HardEvent::S_MTE3>(eid);
        WaitFlag<HardEvent::S_MTE3>(eid);
    } else if (evt == HardEvent::MTE3_MTE2) {
        SetFlag<HardEvent::MTE3_MTE2>(eid);
        WaitFlag<HardEvent::MTE3_MTE2>(eid);
    }
}

__aicore__ inline uint64_t ChpmvVecPhysIdx(uint32_t logical, uint32_t n, int64_t inc)
{
    if (inc >= 0) {
        return static_cast<uint64_t>(logical) * static_cast<uint64_t>(inc);
    }
    return static_cast<uint64_t>(n - 1U - logical) * static_cast<uint64_t>(-inc);
}

__aicore__ inline uint32_t ChpmvColOf(uint32_t blockIdx, uint32_t ordinal, uint32_t numBlocks)
{
    const uint32_t base = ordinal * numBlocks;
    return base + ((ordinal & 1U) != 0U ? (numBlocks - 1U - blockIdx) : blockIdx);
}

__aicore__ inline uint64_t ChpmvColBase(bool upper, uint32_t j, uint32_t n)
{
    const uint64_t col = static_cast<uint64_t>(j);
    if (upper) {
        return col * (col + 1ULL) / 2ULL;
    }
    return col * (2ULL * static_cast<uint64_t>(n) - col + 1ULL) / 2ULL;
}

__aicore__ inline void BuildChpmvAccumTables(const LocalTensor<int32_t>& shiftIdx, const LocalTensor<int32_t>& splitIdx)
{
    CreateVecIndex(shiftIdx, 0, CHPMV_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(shiftIdx, shiftIdx, static_cast<int32_t>(sizeof(float)), CHPMV_BUF_FLOATS);
    CreateVecIndex(splitIdx, 0, CHPMV_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
    Muls(splitIdx, splitIdx, 8, CHPMV_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildChpmvTables(
    const LocalTensor<int32_t>& shiftIdx, const LocalTensor<int32_t>& splitIdx, const LocalTensor<int32_t>& packIdx,
    const LocalTensor<int32_t>& tmpHalf, const LocalTensor<int32_t>& tmpOdd)
{
    CreateVecIndex(shiftIdx, 0, CHPMV_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    ShiftRight(tmpHalf, shiftIdx, 1, CHPMV_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(tmpOdd, tmpHalf, 2, CHPMV_BUF_FLOATS);
    Sub(tmpOdd, shiftIdx, tmpOdd, CHPMV_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Muls(tmpHalf, tmpHalf, static_cast<int32_t>(sizeof(float)), CHPMV_BUF_FLOATS);
    Muls(tmpOdd, tmpOdd, static_cast<int32_t>(CHPMV_PACK_STRIDE * sizeof(float)), CHPMV_BUF_FLOATS);
    PipeBarrier<PIPE_V>();
    Add(packIdx, tmpHalf, tmpOdd, CHPMV_BUF_FLOATS);
    Muls(shiftIdx, shiftIdx, static_cast<int32_t>(sizeof(float)), CHPMV_BUF_FLOATS);
    CreateVecIndex(splitIdx, 0, CHPMV_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
    Muls(splitIdx, splitIdx, 8, CHPMV_ARCH22_TILE_ROWS);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void LoadInterleavedVec(
    GlobalTensor<float>& gm, int64_t inc, uint32_t n, uint32_t start, uint32_t count, const LocalTensor<float>& ubIo,
    const LocalTensor<float>& ubWin, const LocalTensor<float>& ubScalar, TPipe* pipe)
{
    DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
    if (inc == 1) {
        DataCopyExtParams cp{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPad(ubIo, gm[2ULL * start], cp, noPad);
        ChpmvWaitEvent(HardEvent::MTE2_V, pipe);
        return;
    }
    const uint64_t first = ChpmvVecPhysIdx(start, n, inc);
    const uint64_t last = ChpmvVecPhysIdx(start + count - 1U, n, inc);
    const uint64_t winLo = (first < last) ? first : last;
    const uint64_t winSpan = ((first < last) ? (last - first) : (first - last)) + 1ULL;
    if (winSpan <= CHPMV_ARCH22_WIN_MAX_CPLX) {
        DataCopyExtParams cpWin{1U, static_cast<uint32_t>(winSpan * 8ULL), 0U, 0U, 0U};
        DataCopyPad(ubWin, gm[2ULL * winLo], cpWin, noPad);
        ChpmvWaitEvent(HardEvent::MTE2_S, pipe);
        for (uint32_t i = 0U; i < count; ++i) {
            const uint64_t phys = ChpmvVecPhysIdx(start + i, n, inc);
            const uint32_t wOff = static_cast<uint32_t>(phys - winLo) * 2U;
            ubIo.SetValue(2U * i, ubWin.GetValue(wOff));
            ubIo.SetValue(2U * i + 1U, ubWin.GetValue(wOff + 1U));
        }
    } else {
        for (uint32_t i = 0U; i < count; ++i) {
            const uint64_t phys = ChpmvVecPhysIdx(start + i, n, inc);
            DataCopyExtParams cp1{1U, 8U, 0U, 0U, 0U};
            DataCopyPad(ubScalar, gm[2ULL * phys], cp1, noPad);
            ChpmvWaitEvent(HardEvent::MTE2_S, pipe);
            ubIo.SetValue(2U * i, ubScalar.GetValue(0));
            ubIo.SetValue(2U * i + 1U, ubScalar.GetValue(1));
        }
    }
    ChpmvWaitEvent(HardEvent::S_V, pipe);
}

__aicore__ inline void StoreInterleavedVec(
    GlobalTensor<float>& gm, int64_t inc, uint32_t n, uint32_t start, uint32_t count, const LocalTensor<float>& ubIo,
    const LocalTensor<float>& ubScalar, TPipe* pipe)
{
    if (inc == 1) {
        ChpmvWaitEvent(HardEvent::V_MTE3, pipe);
        DataCopyExtParams cp{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPad(gm[2ULL * start], ubIo, cp);
        ChpmvWaitEvent(HardEvent::MTE3_V, pipe);
        return;
    }
    ChpmvWaitEvent(HardEvent::V_S, pipe);
    for (uint32_t i = 0U; i < count; ++i) {
        const uint64_t phys = ChpmvVecPhysIdx(start + i, n, inc);
        ubScalar.SetValue(0, ubIo.GetValue(2U * i));
        ubScalar.SetValue(1, ubIo.GetValue(2U * i + 1U));
        ChpmvWaitEvent(HardEvent::S_MTE3, pipe);
        DataCopyExtParams cp1{1U, 8U, 0U, 0U, 0U};
        DataCopyPad(gm[2ULL * phys], ubScalar, cp1);
        ChpmvWaitEvent(HardEvent::MTE3_V, pipe);
    }
}

class ChpmvAccumAIV {
public:
    __aicore__ inline ChpmvAccumAIV() = default;

    __aicore__ inline void Init(GM_ADDR ap, GM_ADDR x, GM_ADDR workSpace, const ChpmvTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        active_ = (tiling_.accumBlocks > 0U) && (blockIdx_ < tiling_.accumBlocks) && (tiling_.n > 0U);
        if (!active_) {
            return;
        }
        apGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ap));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));

        pipe_->InitBuffer(bufXRe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufXIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufAccRe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufAccIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufIo_, CHPMV_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufARe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufAIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufT1_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufT2_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufT3_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufT4_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufXSegRe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufXSegIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufShiftIdx_, CHPMV_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufSplitIdx_, CHPMV_ARCH22_TILE_ROWS * sizeof(int32_t));
        pipe_->InitBuffer(bufRedOut_, 64U * sizeof(float));
        pipe_->InitBuffer(bufRedTmp_, CHPMV_RED_TMP * sizeof(float));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incx != 1) {
            pipe_->InitBuffer(bufWin_, CHPMV_ARCH22_WIN_MAX_CPLX * 2U * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        LocalTensor<int32_t> shiftIdx = bufShiftIdx_.Get<int32_t>();
        LocalTensor<int32_t> splitIdx = bufSplitIdx_.Get<int32_t>();
        BuildChpmvAccumTables(shiftIdx, splitIdx);

        const uint64_t wsCoreBase = static_cast<uint64_t>(blockIdx_) * 2ULL * static_cast<uint64_t>(tiling_.nPad);
        const bool multiTile = (tiling_.n > tiling_.tileRows);
        if (multiTile) {
            ZeroWorkspaceSlice(wsCoreBase);
        }
        const uint32_t totalOrds = (tiling_.n + tiling_.accumBlocks - 1U) / tiling_.accumBlocks;
        for (uint32_t tileStart = 0U; tileStart < tiling_.n; tileStart += tiling_.tileRows) {
            const uint32_t tileEnd =
                (tileStart + tiling_.tileRows < tiling_.n) ? (tileStart + tiling_.tileRows) : tiling_.n;
            const uint32_t curTileLen = tileEnd - tileStart;
            PrepareTile(tileStart, curTileLen);
            ProcessColumnsForTile(tileStart, tileEnd, totalOrds, wsCoreBase, multiTile);
            StoreTileAccum(wsCoreBase, tileStart, curTileLen, multiTile);
        }
    }

private:
    __aicore__ inline void ZeroWorkspaceSlice(uint64_t wsCoreBase)
    {
        LocalTensor<float> ubZero = bufAccRe_.Get<float>();
        Duplicate(ubZero, 0.0f, CHPMV_ARCH22_TILE_ROWS);
        ChpmvWaitEvent(HardEvent::V_MTE3, pipe_);
        for (uint32_t off = 0U; off < tiling_.n; off += CHPMV_ARCH22_TILE_ROWS) {
            const uint32_t cur =
                ((tiling_.n - off) > CHPMV_ARCH22_TILE_ROWS) ? CHPMV_ARCH22_TILE_ROWS : (tiling_.n - off);
            DataCopyExtParams cp{1U, cur * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(wsGm_[wsCoreBase + off], ubZero, cp);
            DataCopyPad(wsGm_[wsCoreBase + tiling_.nPad + off], ubZero, cp);
        }
        ChpmvWaitEvent(HardEvent::MTE3_MTE2, pipe_);
    }

    __aicore__ inline void PrepareTile(uint32_t tileStart, uint32_t curTileLen)
    {
        LocalTensor<float> ubIo = bufIo_.Get<float>();
        LocalTensor<float> ubWin = (tiling_.incx != 1) ? bufWin_.Get<float>() : ubIo;
        LoadInterleavedVec(
            xGm_, tiling_.incx, tiling_.n, tileStart, curTileLen, ubIo, ubWin, bufScalar_.Get<float>(), pipe_);
        LocalTensor<uint32_t> splitIdx = bufSplitIdx_.Get<uint32_t>();
        Gather(bufXRe_.Get<float>(), ubIo, splitIdx, 0U, curTileLen);
        Gather(bufXIm_.Get<float>(), ubIo, splitIdx, sizeof(float), curTileLen);
        Duplicate(bufAccRe_.Get<float>(), 0.0f, curTileLen);
        Duplicate(bufAccIm_.Get<float>(), 0.0f, curTileLen);
        PipeBarrier<PIPE_V>();
        ChpmvWaitEvent(HardEvent::V_S, pipe_);
    }

    __aicore__ inline void StoreTileAccum(uint64_t wsCoreBase, uint32_t tileStart, uint32_t curTileLen, bool multiTile)
    {
        LocalTensor<float> ubAccRe = bufAccRe_.Get<float>();
        LocalTensor<float> ubAccIm = bufAccIm_.Get<float>();
        DataCopyExtParams cp{1U, curTileLen * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        if (multiTile) {
            LocalTensor<float> ubTmpRe = bufT1_.Get<float>();
            LocalTensor<float> ubTmpIm = bufT2_.Get<float>();
            DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
            DataCopyPad(ubTmpRe, wsGm_[wsCoreBase + tileStart], cp, noPad);
            DataCopyPad(ubTmpIm, wsGm_[wsCoreBase + tiling_.nPad + tileStart], cp, noPad);
            ChpmvWaitEvent(HardEvent::MTE2_V, pipe_);
            Add(ubAccRe, ubAccRe, ubTmpRe, curTileLen);
            Add(ubAccIm, ubAccIm, ubTmpIm, curTileLen);
            PipeBarrier<PIPE_V>();
        }
        ChpmvWaitEvent(HardEvent::V_MTE3, pipe_);
        DataCopyPad(wsGm_[wsCoreBase + tileStart], ubAccRe, cp);
        DataCopyPad(wsGm_[wsCoreBase + tiling_.nPad + tileStart], ubAccIm, cp);
        ChpmvWaitEvent(HardEvent::MTE3_V, pipe_);
        ChpmvWaitEvent(HardEvent::MTE3_MTE2, pipe_);
    }

    __aicore__ inline void ReadXScalar(uint32_t j, uint32_t tileStart, uint32_t tileEnd, float& xr, float& xi)
    {
        if (j >= tileStart && j < tileEnd) {
            xr = bufXRe_.Get<float>().GetValue(j - tileStart);
            xi = bufXIm_.Get<float>().GetValue(j - tileStart);
            return;
        }
        LocalTensor<float> ubScalar = bufScalar_.Get<float>();
        const uint64_t phys = ChpmvVecPhysIdx(j, tiling_.n, tiling_.incx);
        DataCopyExtParams cp{1U, 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubScalar, xGm_[2ULL * phys], cp, noPad);
        ChpmvWaitEvent(HardEvent::MTE2_S, pipe_);
        xr = ubScalar.GetValue(0);
        xi = ubScalar.GetValue(1);
    }

    __aicore__ inline void AccumScalarAt(
        uint32_t row, uint32_t tileStart, uint32_t tileEnd, uint64_t wsCoreBase, float addRe, float addIm)
    {
        if (row >= tileStart && row < tileEnd) {
            const uint32_t loc = row - tileStart;
            LocalTensor<float> accRe = bufAccRe_.Get<float>();
            LocalTensor<float> accIm = bufAccIm_.Get<float>();
            accRe.SetValue(loc, accRe.GetValue(loc) + addRe);
            accIm.SetValue(loc, accIm.GetValue(loc) + addIm);
            ChpmvWaitEvent(HardEvent::S_V, pipe_);
            return;
        }
        LocalTensor<float> ubScalar = bufScalar_.Get<float>();
        DataCopyExtParams cp1{1U, 4U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubScalar, wsGm_[wsCoreBase + row], cp1, noPad);
        DataCopyPad(ubScalar[8], wsGm_[wsCoreBase + tiling_.nPad + row], cp1, noPad);
        ChpmvWaitEvent(HardEvent::MTE2_S, pipe_);
        ubScalar.SetValue(0, ubScalar.GetValue(0) + addRe);
        ubScalar.SetValue(8, ubScalar.GetValue(8) + addIm);
        ChpmvWaitEvent(HardEvent::S_MTE3, pipe_);
        DataCopyPad(wsGm_[wsCoreBase + row], ubScalar, cp1);
        DataCopyPad(wsGm_[wsCoreBase + tiling_.nPad + row], ubScalar[8], cp1);
        ChpmvWaitEvent(HardEvent::MTE3_MTE2, pipe_);
    }

    __aicore__ inline void AddVectorToAcc(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t localOff, uint32_t count)
    {
        const uint32_t h = localOff & 7U;
        if (h == 0U) {
            Add(dst[localOff], dst[localOff], src, count);
            return;
        }
        const uint32_t lead = ((8U - h) < count) ? (8U - h) : count;
        ChpmvWaitEvent(HardEvent::V_S, pipe_);
        for (uint32_t k = 0U; k < lead; ++k) {
            dst.SetValue(localOff + k, dst.GetValue(localOff + k) + src.GetValue(k));
        }
        ChpmvWaitEvent(HardEvent::S_V, pipe_);
        if (count > lead) {
            LocalTensor<float> ubWork = bufT4_.Get<float>();
            Gather(
                ubWork, src, bufShiftIdx_.Get<uint32_t>(), static_cast<uint32_t>(lead * sizeof(float)), count - lead);
            PipeBarrier<PIPE_V>();
            Add(dst[localOff + lead], dst[localOff + lead], ubWork, count - lead);
        }
    }

    __aicore__ inline void ProcessOffDiagSlice(
        uint32_t j, uint64_t apGmOff, uint32_t tileStart, uint32_t tileEnd, uint32_t offLo, uint32_t count,
        uint64_t wsCoreBase, float xjRe, float xjIm)
    {
        LocalTensor<float> ubIo = bufIo_.Get<float>();
        DataCopyExtParams cpAP{1U, count * 8U, 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        DataCopyPad(ubIo, apGm_[apGmOff], cpAP, noPad);
        ChpmvWaitEvent(HardEvent::MTE2_V, pipe_);

        LocalTensor<float> aRe = bufARe_.Get<float>();
        LocalTensor<float> aIm = bufAIm_.Get<float>();
        LocalTensor<uint32_t> splitIdx = bufSplitIdx_.Get<uint32_t>();
        Gather(aRe, ubIo, splitIdx, 0U, count);
        Gather(aIm, ubIo, splitIdx, sizeof(float), count);
        PipeBarrier<PIPE_V>();

        // AXPY: acc[offLo..offHi) += AP_off * x[j]
        LocalTensor<float> t1 = bufT1_.Get<float>();
        LocalTensor<float> t2 = bufT2_.Get<float>();
        LocalTensor<float> t3 = bufT3_.Get<float>();
        LocalTensor<float> t4 = bufT4_.Get<float>();
        Muls(t1, aRe, xjRe, static_cast<int32_t>(count));
        Muls(t2, aIm, xjIm, static_cast<int32_t>(count));
        Muls(t3, aRe, xjIm, static_cast<int32_t>(count));
        Muls(t4, aIm, xjRe, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        Sub(t1, t1, t2, count);
        Add(t3, t3, t4, count);
        PipeBarrier<PIPE_V>();
        const uint32_t localOff = offLo - tileStart;
        AddVectorToAcc(bufAccRe_.Get<float>(), t1, localOff, count);
        AddVectorToAcc(bufAccIm_.Get<float>(), t3, localOff, count);
        PipeBarrier<PIPE_V>();

        // DOT: sum += conj(AP_off) * x[offLo..offHi)
        LocalTensor<float> xSegRe = bufXSegRe_.Get<float>();
        LocalTensor<float> xSegIm = bufXSegIm_.Get<float>();
        Gather(xSegRe, bufXRe_.Get<float>(), bufShiftIdx_.Get<uint32_t>(), localOff * 4U, count);
        Gather(xSegIm, bufXIm_.Get<float>(), bufShiftIdx_.Get<uint32_t>(), localOff * 4U, count);
        PipeBarrier<PIPE_V>();
        Mul(t1, aRe, xSegRe, count);
        Mul(t2, aIm, xSegIm, count);
        Mul(t3, aRe, xSegIm, count);
        Mul(t4, aIm, xSegRe, count);
        PipeBarrier<PIPE_V>();
        Add(t1, t1, t2, count);
        Sub(t3, t3, t4, count);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> redOut = bufRedOut_.Get<float>();
        LocalTensor<float> redTmp = bufRedTmp_.Get<float>();
        ReduceSum(redOut, t1, redTmp, static_cast<int32_t>(count));
        ReduceSum(redOut[8], t3, redTmp, static_cast<int32_t>(count));
        ChpmvWaitEvent(HardEvent::V_S, pipe_);
        AccumScalarAt(j, tileStart, tileEnd, wsCoreBase, redOut.GetValue(0), redOut.GetValue(8));
    }

    __aicore__ inline void ProcessColumnsForTile(
        uint32_t tileStart, uint32_t tileEnd, uint32_t totalOrds, uint64_t wsCoreBase, bool multiTile)
    {
        (void)multiTile;
        const bool upper = (tiling_.uplo == ACLBLAS_UPPER);
        for (uint32_t ord = 0U; ord < totalOrds; ++ord) {
            const uint32_t j = ChpmvColOf(blockIdx_, ord, tiling_.accumBlocks);
            if (j >= tiling_.n) {
                continue;
            }
            const uint64_t colBase = ChpmvColBase(upper, j, tiling_.n);
            const uint32_t colRowStart = upper ? 0U : j;
            float xjRe = 0.0f;
            float xjIm = 0.0f;
            ReadXScalar(j, tileStart, tileEnd, xjRe, xjIm);

            if (j >= tileStart && j < tileEnd) {
                // Read diagonal real only (never read diagonal imag!)
                const uint64_t diagGmIdx = 2ULL * (colBase + static_cast<uint64_t>(upper ? j : 0U));
                const float diagRe = apGm_.GetValue(diagGmIdx);
                AccumScalarAt(j, tileStart, tileEnd, wsCoreBase, diagRe * xjRe, diagRe * xjIm);
            }
            const uint32_t offBegin = upper ? 0U : (j + 1U);
            const uint32_t offEnd = upper ? j : tiling_.n;
            const uint32_t offLo = (offBegin > tileStart) ? offBegin : tileStart;
            const uint32_t offHi = (offEnd < tileEnd) ? offEnd : tileEnd;
            if (offLo < offHi) {
                const uint64_t apGmOff = 2ULL * (colBase + static_cast<uint64_t>(offLo - colRowStart));
                ProcessOffDiagSlice(j, apGmOff, tileStart, tileEnd, offLo, offHi - offLo, wsCoreBase, xjRe, xjIm);
            }
        }
    }

    TPipe* pipe_{nullptr};
    GlobalTensor<float> apGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> wsGm_;
    TBuf<TPosition::VECCALC> bufXRe_;
    TBuf<TPosition::VECCALC> bufXIm_;
    TBuf<TPosition::VECCALC> bufAccRe_;
    TBuf<TPosition::VECCALC> bufAccIm_;
    TBuf<TPosition::VECCALC> bufIo_;
    TBuf<TPosition::VECCALC> bufARe_;
    TBuf<TPosition::VECCALC> bufAIm_;
    TBuf<TPosition::VECCALC> bufT1_;
    TBuf<TPosition::VECCALC> bufT2_;
    TBuf<TPosition::VECCALC> bufT3_;
    TBuf<TPosition::VECCALC> bufT4_;
    TBuf<TPosition::VECCALC> bufXSegRe_;
    TBuf<TPosition::VECCALC> bufXSegIm_;
    TBuf<TPosition::VECCALC> bufShiftIdx_;
    TBuf<TPosition::VECCALC> bufSplitIdx_;
    TBuf<TPosition::VECCALC> bufRedOut_;
    TBuf<TPosition::VECCALC> bufRedTmp_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    ChpmvTilingData tiling_{};
    uint32_t blockIdx_{0U};
    bool active_{false};
};

class ChpmvReduceAIV {
public:
    __aicore__ inline ChpmvReduceAIV() = default;

    __aicore__ inline void Init(GM_ADDR workSpace, GM_ADDR y, const ChpmvTilingData& tiling, TPipe* pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        const uint32_t blockIdx = GetBlockIdx();
        active_ = (tiling_.reduceBlocks > 0U) && (blockIdx < tiling_.reduceBlocks) && (tiling_.n > 0U);
        if (!active_) {
            return;
        }
        const uint32_t perCore = tiling_.n / tiling_.reduceBlocks;
        const uint32_t rem = tiling_.n % tiling_.reduceBlocks;
        start_ = blockIdx * perCore + ((blockIdx < rem) ? blockIdx : rem);
        count_ = perCore + ((blockIdx < rem) ? 1U : 0U);
        active_ = (count_ > 0U);
        if (!active_) {
            return;
        }
        if (workSpace != nullptr) {
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
        }
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

        pipe_->InitBuffer(bufSumRe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufSumIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufTmpRe_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufTmpIm_, CHPMV_ARCH22_TILE_ROWS * sizeof(float));
        pipe_->InitBuffer(bufOutPlanes_, CHPMV_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufIo_, CHPMV_BUF_FLOATS * sizeof(float));
        pipe_->InitBuffer(bufShiftIdx_, CHPMV_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufSplitIdx_, CHPMV_ARCH22_TILE_ROWS * sizeof(int32_t));
        pipe_->InitBuffer(bufPackIdx_, CHPMV_BUF_FLOATS * sizeof(int32_t));
        pipe_->InitBuffer(bufScalar_, 64U * sizeof(float));
        if (tiling_.incy != 1) {
            pipe_->InitBuffer(bufWin_, CHPMV_ARCH22_WIN_MAX_CPLX * 2U * sizeof(float));
        }
    }

    __aicore__ inline void Process(bool scaleOnly)
    {
        if (!active_) {
            return;
        }
        BuildChpmvTables(
            bufShiftIdx_.Get<int32_t>(), bufSplitIdx_.Get<int32_t>(), bufPackIdx_.Get<int32_t>(), bufIo_.Get<int32_t>(),
            bufOutPlanes_.Get<int32_t>());
        for (uint32_t off = 0U; off < count_; off += CHPMV_ARCH22_TILE_ROWS) {
            const uint32_t cur = ((count_ - off) > CHPMV_ARCH22_TILE_ROWS) ? CHPMV_ARCH22_TILE_ROWS : (count_ - off);
            if (scaleOnly) {
                ScaleSegment(start_ + off, cur);
            } else {
                ReduceSegment(start_ + off, cur);
            }
        }
    }

private:
    __aicore__ inline void ApplyBetaAndStore(
        uint32_t segStart, uint32_t cur, LocalTensor<float>& outRe, LocalTensor<float>& outIm)
    {
        const bool betaZero = (tiling_.betaReal == 0.0f && tiling_.betaImag == 0.0f);
        LocalTensor<float> ubIo = bufIo_.Get<float>();
        if (!betaZero) {
            LocalTensor<float> ubWin = (tiling_.incy != 1) ? bufWin_.Get<float>() : ubIo;
            LoadInterleavedVec(
                yGm_, tiling_.incy, tiling_.n, segStart, cur, ubIo, ubWin, bufScalar_.Get<float>(), pipe_);
            LocalTensor<float> yRe = bufSumRe_.Get<float>();
            LocalTensor<float> yIm = bufSumIm_.Get<float>();
            LocalTensor<float> tmpRe = bufTmpRe_.Get<float>();
            LocalTensor<float> tmpIm = bufTmpIm_.Get<float>();
            Gather(yRe, ubIo, bufSplitIdx_.Get<uint32_t>(), 0U, cur);
            Gather(yIm, ubIo, bufSplitIdx_.Get<uint32_t>(), sizeof(float), cur);
            PipeBarrier<PIPE_V>();
            Muls(tmpRe, yRe, tiling_.betaReal, static_cast<int32_t>(cur));
            Muls(tmpIm, yIm, tiling_.betaImag, static_cast<int32_t>(cur));
            PipeBarrier<PIPE_V>();
            Sub(tmpRe, tmpRe, tmpIm, cur);
            Add(outRe, outRe, tmpRe, cur);

            Muls(tmpRe, yRe, tiling_.betaImag, static_cast<int32_t>(cur));
            Muls(tmpIm, yIm, tiling_.betaReal, static_cast<int32_t>(cur));
            PipeBarrier<PIPE_V>();
            Add(tmpRe, tmpRe, tmpIm, cur);
            Add(outIm, outIm, tmpRe, cur);
            PipeBarrier<PIPE_V>();
        }
        Gather(ubIo, outRe, bufPackIdx_.Get<uint32_t>(), 0U, 2U * cur);
        StoreInterleavedVec(yGm_, tiling_.incy, tiling_.n, segStart, cur, ubIo, bufScalar_.Get<float>(), pipe_);
    }

    __aicore__ inline void ReduceSegment(uint32_t segStart, uint32_t cur)
    {
        LocalTensor<float> sumRe = bufSumRe_.Get<float>();
        LocalTensor<float> sumIm = bufSumIm_.Get<float>();
        LocalTensor<float> tmpRe = bufTmpRe_.Get<float>();
        LocalTensor<float> tmpIm = bufTmpIm_.Get<float>();
        Duplicate(sumRe, 0.0f, cur);
        Duplicate(sumIm, 0.0f, cur);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams cp{1U, cur * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0F};
        for (uint32_t c = 0U; c < tiling_.accumBlocks; ++c) {
            const uint64_t wsOff = static_cast<uint64_t>(c) * 2ULL * static_cast<uint64_t>(tiling_.nPad) + segStart;
            DataCopyPad(tmpRe, wsGm_[wsOff], cp, noPad);
            DataCopyPad(tmpIm, wsGm_[wsOff + tiling_.nPad], cp, noPad);
            ChpmvWaitEvent(HardEvent::MTE2_V, pipe_);
            Add(sumRe, sumRe, tmpRe, cur);
            Add(sumIm, sumIm, tmpIm, cur);
            PipeBarrier<PIPE_V>();
            ChpmvWaitEvent(HardEvent::V_MTE2, pipe_);
        }
        LocalTensor<float> outRe = bufOutPlanes_.Get<float>();
        LocalTensor<float> outIm = outRe[CHPMV_PACK_STRIDE];
        Muls(outRe, sumRe, tiling_.alphaReal, static_cast<int32_t>(cur));
        Muls(tmpRe, sumIm, tiling_.alphaImag, static_cast<int32_t>(cur));
        Muls(outIm, sumRe, tiling_.alphaImag, static_cast<int32_t>(cur));
        Muls(tmpIm, sumIm, tiling_.alphaReal, static_cast<int32_t>(cur));
        PipeBarrier<PIPE_V>();
        Sub(outRe, outRe, tmpRe, cur);
        Add(outIm, outIm, tmpIm, cur);
        PipeBarrier<PIPE_V>();
        ApplyBetaAndStore(segStart, cur, outRe, outIm);
    }

    __aicore__ inline void ScaleSegment(uint32_t segStart, uint32_t cur)
    {
        LocalTensor<float> outRe = bufOutPlanes_.Get<float>();
        LocalTensor<float> outIm = outRe[CHPMV_PACK_STRIDE];
        Duplicate(outRe, 0.0f, cur);
        Duplicate(outIm, 0.0f, cur);
        PipeBarrier<PIPE_V>();
        ApplyBetaAndStore(segStart, cur, outRe, outIm);
    }

    TPipe* pipe_{nullptr};
    GlobalTensor<float> wsGm_;
    GlobalTensor<float> yGm_;
    TBuf<TPosition::VECCALC> bufSumRe_;
    TBuf<TPosition::VECCALC> bufSumIm_;
    TBuf<TPosition::VECCALC> bufTmpRe_;
    TBuf<TPosition::VECCALC> bufTmpIm_;
    TBuf<TPosition::VECCALC> bufOutPlanes_;
    TBuf<TPosition::VECCALC> bufIo_;
    TBuf<TPosition::VECCALC> bufShiftIdx_;
    TBuf<TPosition::VECCALC> bufSplitIdx_;
    TBuf<TPosition::VECCALC> bufPackIdx_;
    TBuf<TPosition::VECCALC> bufScalar_;
    TBuf<TPosition::VECCALC> bufWin_;
    ChpmvTilingData tiling_{};
    uint32_t start_{0U};
    uint32_t count_{0U};
    bool active_{false};
};

} // namespace

extern "C" __global__ __aicore__ void chpmv_accum_arch22_kernel(
    GM_ADDR ap, GM_ADDR x, GM_ADDR workSpace, const ChpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ChpmvAccumAIV op;
    op.Init(ap, x, workSpace, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void chpmv_reduce_arch22_kernel(
    GM_ADDR workSpace, GM_ADDR y, const ChpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ChpmvReduceAIV op;
    op.Init(workSpace, y, tiling, &pipe);
    op.Process(false);
}

extern "C" __global__ __aicore__ void chpmv_scale_arch22_kernel(GM_ADDR y, const ChpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ChpmvReduceAIV op;
    op.Init(nullptr, y, tiling, &pipe);
    op.Process(true);
}

aclError chpmv_accum_kernel_do(
    const uint8_t* ap, const uint8_t* x, uint8_t* workSpace, const ChpmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // Drain any stale thread-level error so the post-launch check only reports
    // this launch's own failure (issue #1181 precedent).
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    chpmv_accum_arch22_kernel<<<numBlocks, nullptr, stream>>>(
        const_cast<uint8_t*>(ap), const_cast<uint8_t*>(x), workSpace, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

aclError chpmv_reduce_kernel_do(
    const uint8_t* workSpace, uint8_t* y, const ChpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    chpmv_reduce_arch22_kernel<<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(workSpace), y, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}

aclError chpmv_scale_kernel_do(uint8_t* y, const ChpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
    chpmv_scale_arch22_kernel<<<numBlocks, nullptr, stream>>>(y, tiling);
    return aclrtGetLastError(ACL_RT_THREAD_LEVEL);
}
