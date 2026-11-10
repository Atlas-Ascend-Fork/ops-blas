/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

// Half specialization of GEMM strided-batch native ND->NZ / LoadData2D
// movement, using GEMM's double-buffer event schedule. Ex keeps its defaults.
namespace hgemm_pipeline {
constexpr uint32_t C0 = HGEMM_CUBE_C0_ELEMENTS;

template <uint32_t BM, uint32_t CHUNK, bool TA, bool TB>
__aicore__ inline void CopyPanel(
    const AscendC::GlobalTensor<half>& ga, const AscendC::GlobalTensor<half>& gb,
    const GemmStridedBatchedExTilingData& tiling, uint64_t mo, uint64_t no, uint32_t cm, uint32_t cn, uint32_t ko,
    uint64_t slot)
{
    constexpr uint64_t A_BYTES = BM * CHUNK * sizeof(half), B_BYTES = BM * CHUNK * sizeof(half);
    const uint64_t ao = slot * HGEMM_L1_SLOT_BYTES, bo = ao + A_BYTES;
    const uint32_t kc = static_cast<uint32_t>(tiling.k) - ko < CHUNK ? static_cast<uint32_t>(tiling.k) - ko : CHUNK;
    const uint32_t ma = (cm + C0 - 1) / C0 * C0, na = (cn + C0 - 1) / C0 * C0, ka = (kc + C0 - 1) / C0 * C0;
    const uint32_t mas = ma, nas = na, kas = ka;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    AscendC::LocalTensor<half> a1(AscendC::TPosition::A1, ao, A_BYTES / sizeof(half));
    AscendC::LocalTensor<half> b1(AscendC::TPosition::B1, bo, B_BYTES / sizeof(half));
    AscendC::Nd2NzParams pa{};
    pa.ndNum = 1;
    pa.nValue = TA ? kc : cm;
    pa.dValue = TA ? cm : kc;
    pa.srcDValue = tiling.lda;
    pa.dstNzC0Stride = TA ? kas : mas;
    pa.dstNzNStride = 1;
    const uint64_t ai = TA ? static_cast<uint64_t>(ko) * tiling.lda + mo : mo * tiling.lda + ko;
    AscendC::DataCopy(a1, ga[ai], pa);
    AscendC::Nd2NzParams pb{};
    pb.ndNum = 1;
    pb.nValue = TB ? cn : kc;
    pb.dValue = TB ? kc : cn;
    pb.srcDValue = tiling.ldb;
    pb.dstNzC0Stride = TB ? nas : kas;
    pb.dstNzNStride = 1;
    const uint64_t bi = TB ? no * tiling.ldb + ko : static_cast<uint64_t>(ko) * tiling.ldb + no;
    AscendC::DataCopy(b1, gb[bi], pb);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
}

template <uint32_t BM, uint32_t BK, typename C, QuantMode_t MODE, bool TA, bool TB>
struct Pipeline {
    static constexpr uint32_t CHUNK = HGEMM_K_PANEL_ELEMENTS;
    static constexpr uint32_t L0_SLOTS =
        BM * BK * sizeof(half) > HGEMM_L0_AB_HALF_BYTES ? 1 : HGEMM_DOUBLE_BUFFER_SLOTS;
    static constexpr uint64_t L0_SLOT = HGEMM_L0_AB_CAPACITY_BYTES / L0_SLOTS;
    static constexpr uint32_t C_SLOTS = BM == HGEMM_TILE_SMALL ?
                                            HGEMM_SMALL_TILE_C_SLOTS :
                                            (BM == HGEMM_TILE_STANDARD ? HGEMM_STANDARD_TILE_C_SLOTS : 1);
    static constexpr uint64_t C_BYTES = BM * BM * sizeof(float);
    static constexpr uint64_t A_BYTES = BM * CHUNK * sizeof(half);
    static constexpr uint64_t B_BYTES = BM * CHUNK * sizeof(half);
    static_assert(A_BYTES + B_BYTES <= HGEMM_L1_SLOT_BYTES, "HGEMM double L1 capacity");
    static_assert(BM * BK * sizeof(half) <= L0_SLOT, "HGEMM double L0 capacity");
    static_assert(C_SLOTS * C_BYTES <= HGEMM_L0_C_CAPACITY_BYTES, "HGEMM L0C capacity");
    GM_ADDR a;
    GM_ADDR b;
    GM_ADDR c;
    GemmStridedBatchedExTilingData tiling;
    GemmStridedCubeState st{};
    uint64_t l1Index = 0, l0Index = 0, cIndex = 0;
    bool singleTile = false, hasNextPanel = false;
    uint64_t nextPanelSlot = 0;

    __aicore__ inline void InitializeEventsAndShape()
    {
        if (singleTile) {
            // Balanced splitK=1 tiling: task id is the batch id. Fixpipe only
            // needs these shape fields; retain them across batches rather than
            // repeating generic split-K and multi-tile scheduling arithmetic.
            st.actualM = tiling.m;
            st.actualN = tiling.n;
            st.mBlockIdx = 0;
            st.nBlockIdx = 0;
            st.cBaseOffset = 0;
            st.nBlockSize = 1;
            st.baseMCount = tiling.m / BM;
            st.tailM = tiling.m % BM;
            st.tailMAlign = (st.tailM + C0 - 1) / C0 * C0;
            st.baseNCount = tiling.n / BM;
            st.tailN = tiling.n % BM;
            st.tailNAlign = (st.tailN + C0 - 1) / C0 * C0;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
        if constexpr (L0_SLOTS == HGEMM_DOUBLE_BUFFER_SLOTS)
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
        for (uint32_t i = 0; i < C_SLOTS; ++i)
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(i);
    }

    __aicore__ inline void MultiplyInner(
        AscendC::LocalTensor<half> a1, AscendC::LocalTensor<half> b1, uint32_t cm, uint32_t cn, uint32_t kc,
        uint32_t ko, uint32_t inner, uint32_t ma, uint32_t na, uint32_t mas, uint32_t nas, uint32_t kas,
        uint64_t cOffset)
    {
        const uint32_t ck = kc - inner < BK ? kc - inner : BK;
        const uint64_t l0 = l0Index++ & (L0_SLOTS - 1), offset = l0 * L0_SLOT;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0);
        AscendC::LocalTensor<half> a2(AscendC::TPosition::A2, offset, BM * BK);
        AscendC::LocalTensor<half> b2(AscendC::TPosition::B2, offset, BK * BM);
        const uint64_t aInner = TA ? static_cast<uint64_t>(inner) * C0 : static_cast<uint64_t>(inner / C0) * mas * C0;
        const uint64_t bInner = TB ? static_cast<uint64_t>(inner / C0) * nas * C0 : static_cast<uint64_t>(inner) * C0;
        AscendC::LoadData2DParamsV2 la{};
        la.mStep = TA ? (ck + C0 - 1) / C0 : ma / C0;
        la.kStep = TA ? ma / C0 : (ck + C0 - 1) / C0;
        la.srcStride = TA ? kas / C0 : mas / C0;
        la.dstStride = ma / C0;
        la.ifTranspose = TA;
        AscendC::LoadData(a2, a1[aInner], la);
        AscendC::LoadData2DParamsV2 lb{};
        lb.mStep = TB ? na / C0 : (ck + C0 - 1) / C0;
        lb.kStep = TB ? (ck + C0 - 1) / C0 : na / C0;
        lb.srcStride = TB ? nas / C0 : kas / C0;
        lb.dstStride = na / C0;
        lb.ifTranspose = !TB;
        AscendC::LoadData(b2, b1[bInner], lb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0);
        AscendC::LocalTensor<half> al(AscendC::TPosition::A2, offset, BM * BK);
        AscendC::LocalTensor<half> bl(AscendC::TPosition::B2, offset, BK * BM);
        AscendC::LocalTensor<float> cl(AscendC::TPosition::CO1, cOffset, BM * BM);
        // Small L0C accumulations need a matrix-pipe dependency as well
        // as the L0 source-buffer lifetime events.
        if (ko + inner != 0 && (cm / C0) * (cn / C0) < HGEMM_MMAD_DEPENDENCY_BLOCKS)
            AscendC::PipeBarrier<PIPE_M>();
        AscendC::MmadParams mp{};
        mp.m = cm;
        mp.n = cn;
        mp.k = ck;
        mp.cmatrixInitVal = (ko + inner == 0);
        mp.disableGemv = true;         // inputs are NZ/ZN, including M=1
        AscendC::Mmad(cl, al, bl, mp); // unitFlag=0 matches shared Fixpipe
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0);
    }

    __aicore__ inline void ComputePanel(
        uint32_t cm, uint32_t cn, uint32_t kc, uint32_t ko, uint64_t ao, uint64_t bo, uint64_t cOffset)
    {
        AscendC::LocalTensor<half> a1(AscendC::TPosition::A1, ao, A_BYTES / sizeof(half));
        AscendC::LocalTensor<half> b1(AscendC::TPosition::B1, bo, B_BYTES / sizeof(half));
        const uint32_t ma = (cm + C0 - 1) / C0 * C0, na = (cn + C0 - 1) / C0 * C0, ka = (kc + C0 - 1) / C0 * C0;
        const uint32_t mas = ma, nas = na, kas = ka;
        for (uint32_t inner = 0; inner < kc; inner += BK) {
            MultiplyInner(a1, b1, cm, cn, kc, ko, inner, ma, na, mas, nas, kas, cOffset);
        }
    }

    __aicore__ inline void ProcessPanel(
        AscendC::GlobalTensor<half> ga, AscendC::GlobalTensor<half> gb, uint64_t task, uint64_t mo, uint64_t no,
        uint32_t cm, uint32_t cn, uint32_t ko, uint64_t cOffset, uint64_t& currentSlot)
    {
        const uint32_t remaining = static_cast<uint32_t>(tiling.k) - ko;
        const uint32_t kc = remaining < CHUNK ? remaining : CHUNK;
        const uint64_t slot = currentSlot, ao = slot * HGEMM_L1_SLOT_BYTES, bo = ao + A_BYTES;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        if (remaining > CHUNK) {
            currentSlot = l1Index++ & (HGEMM_DOUBLE_BUFFER_SLOTS - 1);
            CopyPanel<BM, CHUNK, TA, TB>(ga, gb, tiling, mo, no, cm, cn, ko + CHUNK, currentSlot);
        } else if (BM != HGEMM_TILE_SMALL && task + AscendC::GetBlockNum() < tiling.totalTasks) {
            // On the final K panel, use the other L1 slot to fetch the
            // following grid-stride task while this task consumes L0/Mmad.
            // The same MTE1_MTE2 lifetime event guards every slot reuse.
            GemmStridedCubeState next = st;
            InitStridedTaskState<BM, BK, BM, C0>(next, task + AscendC::GetBlockNum());
            AscendC::GlobalTensor<half> nextA, nextB;
            nextA.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(a) + static_cast<int64_t>(next.batchIdx) * tiling.strideA);
            nextB.SetGlobalBuffer(
                reinterpret_cast<__gm__ half*>(b) + static_cast<int64_t>(next.batchIdx) * tiling.strideB);
            nextPanelSlot = l1Index++ & (HGEMM_DOUBLE_BUFFER_SLOTS - 1);
            CopyPanel<BM, CHUNK, TA, TB>(
                nextA, nextB, tiling, static_cast<uint64_t>(next.mBlockIdx) * BM,
                static_cast<uint64_t>(next.nBlockIdx) * BM, next.actualM, next.actualN, 0, nextPanelSlot);
            hasNextPanel = true;
        }
        ComputePanel(cm, cn, kc, ko, ao, bo, cOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
    }

    __aicore__ inline void ProcessTask(uint64_t task)
    {
        if (singleTile)
            st.batchIdx = static_cast<uint32_t>(task);
        else
            InitStridedTaskState<BM, BK, BM, C0>(st, task);
        const uint32_t cm = st.actualM, cn = st.actualN;
        const uint64_t mo = static_cast<uint64_t>(st.mBlockIdx) * BM;
        const uint64_t no = static_cast<uint64_t>(st.nBlockIdx) * BM;
        AscendC::GlobalTensor<half> ga, gb;
        ga.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a) + static_cast<int64_t>(st.batchIdx) * tiling.strideA);
        gb.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b) + static_cast<int64_t>(st.batchIdx) * tiling.strideB);
        const uint64_t cSlot = cIndex++ & (C_SLOTS - 1);
        const uint64_t cOffset = cSlot * C_BYTES;
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(cSlot);
        uint64_t currentSlot;
        if (hasNextPanel) {
            currentSlot = nextPanelSlot;
            hasNextPanel = false;
        } else {
            currentSlot = l1Index++ & (HGEMM_DOUBLE_BUFFER_SLOTS - 1);
            CopyPanel<BM, CHUNK, TA, TB>(ga, gb, tiling, mo, no, cm, cn, 0, currentSlot);
        }
        for (uint32_t ko = 0; ko < static_cast<uint32_t>(tiling.k); ko += CHUNK)
            ProcessPanel(ga, gb, task, mo, no, cm, cn, ko, cOffset, currentSlot);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(cSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(cSlot);
        AscendC::GlobalTensor<C> gc;
        gc.SetGlobalBuffer(
            reinterpret_cast<__gm__ C*>(c) + static_cast<int64_t>(st.batchIdx) * tiling.workspaceBatchStride);
        WriteStridedFixpipe<C, float, BM, BM, MODE, false>(st, gc, 0, 1, 0, 1, cOffset);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(cSlot);
    }

    __aicore__ inline void Run()
    {
        if (!InitStridedTilingBase(st, tiling))
            return;
        singleTile = st.mnTasks == 1;
        InitializeEventsAndShape();
        for (uint64_t task = AscendC::GetBlockIdx(); task < tiling.totalTasks; task += AscendC::GetBlockNum())
            ProcessTask(task);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
        if constexpr (L0_SLOTS == HGEMM_DOUBLE_BUFFER_SLOTS)
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
        for (uint32_t i = 0; i < C_SLOTS; ++i)
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(i);
    }
};

template <uint32_t BM, uint32_t BK, typename C, QuantMode_t MODE, bool TA, bool TB>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData tiling)
{
    Pipeline<BM, BK, C, MODE, TA, TB> kernel{a, b, c, tiling};
    kernel.Run();
}

template <uint32_t BM, uint32_t BK, typename C, QuantMode_t MODE>
__aicore__ inline void Dispatch(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmStridedBatchedExTilingData t)
{
    if (t.isTransA) {
        if (t.isTransB)
            Run<BM, BK, C, MODE, true, true>(a, b, c, t);
        else
            Run<BM, BK, C, MODE, true, false>(a, b, c, t);
    } else {
        if (t.isTransB)
            Run<BM, BK, C, MODE, false, true>(a, b, c, t);
        else
            Run<BM, BK, C, MODE, false, false>(a, b, c, t);
    }
}
} // namespace hgemm_pipeline
