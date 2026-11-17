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
 * Direct-H cube producer.  Included inside Cher2kKernelLayout so it can
 * access the configured planar tensors and event ring without wrappers.
 */
__aicore__ inline void ProcessDirectHermitian(uint32_t taskIndex, uint32_t taskStride)
{
    constexpr uint32_t directM = Cher2kDirectHermitianState::directM;
    constexpr uint32_t directN = Cher2kDirectHermitianState::directN;
    const uint32_t mTiles = n_ / directM;
    const uint32_t nTiles = n_ / directN;
    const bool directOpc = !transA_ && transB_;
    const bool upper = tilingUplo_ == ACLBLAS_UPPER;
    const uint64_t totalTasks = (static_cast<uint64_t>(mTiles) * (mTiles + 1U)) / 2ULL;
    const uint32_t safeTaskStride = taskStride == 0U ? 1U : taskStride;
    const uint64_t taskBegin = (static_cast<uint64_t>(taskIndex) * totalTasks) / safeTaskStride;
    const uint64_t taskEnd = (static_cast<uint64_t>(taskIndex + 1U) * totalTasks) / safeTaskStride;
    Cher2kDirectHermitianState state;
    state.Init(directOpc, n_, k_);

    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetHF32Mode(HF32Mode::DISABLE);

    uint32_t mTile = 0U;
    uint64_t rowTaskBase = 0ULL;
    uint32_t cachedMBase = UINT32_MAX;
    uint32_t slot0Slice = 0U;
    while (mTile < mTiles) {
        const uint32_t nBegin = upper ? 0U : mTile;
        const uint32_t nEnd = upper ? mTile + 1U : nTiles;
        const uint32_t rowTasks = nEnd - nBegin;
        if (taskBegin < rowTaskBase + rowTasks) {
            break;
        }
        rowTaskBase += rowTasks;
        ++mTile;
    }

    for (uint64_t task = taskBegin; task < taskEnd; ++task) {
        const uint32_t nTile = ResolveTriangularTile(task, upper, nTiles, mTile, rowTaskBase);
        const uint32_t mBase = mTile * directM;
        const uint32_t nBase = nTile * directN;
        const bool sameMRow = mBase == cachedMBase;
        ProcessDirectHermitianTile(state, mBase, nBase, sameMRow, slot0Slice);
        cachedMBase = mBase;
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ inline void AccumulateDirectTerm(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase,
    bool initialize)
{
    for (uint32_t kBase = 0U; kBase < k_; kBase += CHER2K_CUBE_K) {
        CopySingleToL1(a1, b1, left, right, mBase, nBase, kBase);
        LoadSingleToL0(a1, b1, a2, b2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, initialize && kBase == 0U ? firstMmadParams_ : accumulateMmadParams_);
        // Match the proven Chemm lifecycle before either L0 operand is
        // overwritten by the next K slice or algebraic term.
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void CopySingleToL1(
    LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
    uint32_t mBase, uint32_t nBase, uint32_t kBase)
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    const uint64_t aOffset =
        transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + kBase;
    DataCopy(a1, left[aOffset], aCopyParams_);
    const uint64_t bOffset =
        transB_ ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
    DataCopy(b1, right[bOffset], bNzParams_);
    // A and B copies form one panel transaction. A single completion
    // token is sufficient because both destinations are consumed only
    // after this point by LoadSingleToL0.
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
}

__aicore__ inline void CopySingleToL1AtOffsets(
    LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
    uint64_t aOffset, uint64_t bOffset)
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    DataCopy(a1, left[aOffset], aCopyParams_);
    DataCopy(b1, right[bOffset], bNzParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
}

__aicore__ inline void LoadSingleToL0(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2)
{
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData(b2, b1, bLoadParams_);
    // The packed OP_C path now uses the same NxK-left/KxN-right
    // contract as Cherk.  Its validated A2 lifecycle is the regular
    // LoadData3D path; the old hand-written 2D reconstruction targeted
    // the rejected NxK-transposed-B layout and produced wrong products.
    LoadData(a2, a1, aLoadParams_);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
}

// Arch22's 3D A-load contract is not valid for the direct OP_C ND
// workspace.  The documented 2D lifecycle (one repeat per 16-row block,
// K-direction repeats strided by the M-block count) reconstructs the same
// MxK A2 tile from the ND2NZ A1 layout without relying on transpose flags.
__aicore__ inline void LoadDirectA2D(LocalTensor<float>& a2, LocalTensor<float>& a1, uint32_t slice)
{
    constexpr uint32_t cubeBlock = 16U;
    constexpr uint32_t c0Size = 8U; // float elements in one 32-byte block
    constexpr uint32_t mBlocks = CHER2K_CUBE_M / cubeBlock;
    constexpr uint32_t kBlocks = CHER2K_CUBE_K / c0Size;
    LoadData2DParams params{};
    params.repeatTimes = kBlocks;
    params.srcStride = mBlocks;
    params.ifTranspose = false;
    const uint32_t sliceOffset = slice * kBlocks * mBlocks * cubeBlock * cubeBlock;
    for (uint32_t mBlock = 0U; mBlock < mBlocks; ++mBlock) {
        const uint32_t srcOffset = sliceOffset + mBlock * cubeBlock * cubeBlock;
        const uint32_t dstOffset = mBlock * CHER2K_CUBE_K * cubeBlock;
        LoadData(a2[dstOffset], a1[srcOffset], params);
    }
}

__aicore__ inline void ProcessProductSingle(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    if constexpr (CHER2K_L1_PINGPONG_EXPERIMENT) {
        ProcessProductPanelPingPong(a1, b1, a2, b2, c1, left, right, output, mBase, nBase);
        return;
    }
    // B-transpose direct OP_C stores NxK panels.  Its documented L0B
    // contract is one K=64 slice at a time; applying panel kStartPt to
    // this layout raises LOAD3DV2 K_M_START_POS on A2.  Keep the regular
    // KxN route panelized and stream the direct route in valid slices.
    if (!transB_ && k_ >= CHER2K_L1_PANEL_K && k_ % CHER2K_L1_PANEL_K == 0U) {
        ProcessProductPanel(a1, b1, a2, b2, c1, left, right, output, mBase, nBase);
        return;
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    CopySingleToL1(a1, b1, left, right, mBase, nBase, 0U);
    LoadSingleToL0(a1, b1, a2, b2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    Mmad(c1, a2, b2, firstMmadParams_);
    // L0A/L0B are single-buffered on the active route.  The event token
    // only orders MTE1 against M; it does not stop O2 from issuing the
    // next LoadData while the previous MMAD still reads L0.  Chemm's
    // lifecycle closes this producer/consumer edge with a full barrier.
    PipeBarrier<PIPE_ALL>();
    for (uint32_t kBase = CHER2K_CUBE_K; kBase < k_; kBase += CHER2K_CUBE_K) {
        CopySingleToL1(a1, b1, left, right, mBase, nBase, kBase);
        LoadSingleToL0(a1, b1, a2, b2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, accumulateMmadParams_);
        PipeBarrier<PIPE_ALL>();
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

__aicore__ inline void ProcessProductSinglePrefetch(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase, bool firstPanelReady, bool prefetchNext, GlobalTensor<float>& nextLeft,
    GlobalTensor<float>& nextRight, uint32_t nextMBase, uint32_t nextNBase)
{
    // The cross-product contract is only valid for complete K=256 panels;
    // all irregular/tail shapes retain the proven single-product path.
    if (k_ >= CHER2K_L1_PANEL_K && k_ % CHER2K_L1_PANEL_K == 0U) {
        ProcessProductPanel(
            a1, b1, a2, b2, c1, left, right, output, mBase, nBase, firstPanelReady, prefetchNext, nextLeft, nextRight,
            nextMBase, nextNBase);
    } else {
        ProcessProductSingle(a1, b1, a2, b2, c1, left, right, output, mBase, nBase);
    }
}

// Direct OP_C uses an ND A workspace (N x K) and therefore cannot reuse
// the legacy K=256 panel slice offsets. Process each K=64 tile with its
// own ND2NZ transfer; this is the correctness contract for the direct
// format and can be fused/blocked after the contract is proven.
__aicore__ inline void ProcessDirectOpcSlices(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    CopySingleToL1(a1, b1, left, right, mBase, nBase, 0U);
    LoadSingleToL0(a1, b1, a2, b2);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    Mmad(c1, a2, b2, firstMmadParams_);
    for (uint32_t kBase = CHER2K_CUBE_K; kBase < k_; kBase += CHER2K_CUBE_K) {
        if (!CHER2K_SKIP_K_PIPE_BARRIER_EXPERIMENT) {
            PipeBarrier<PIPE_M>();
        }
        CopySingleToL1(a1, b1, left, right, mBase, nBase, kBase);
        LoadSingleToL0(a1, b1, a2, b2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, accumulateMmadParams_);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

#if 1
// Experimental slice prefetch, direct-OP_C and L0 ping-pong variants are
// retired.  They are kept out of the active translation path so the
// production lifecycle below stays readable and deterministic.
// Two L1 slots are sufficient for a producer/consumer ring at K=64. The
// current slice is loaded into L0, then the next GM slice is issued before
// the MMAD. This preserves one L0C accumulator and the existing numerical
// ordering while moving the MTE2 wait to the next iteration.
__aicore__ inline void BeginSlicePrefetchEvents()
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetHF32Mode(HF32Mode::DISABLE);
}

__aicore__ inline void CopyProductSlice(
    LocalTensor<float>& aSlot, LocalTensor<float>& bSlot, GlobalTensor<float>& left, GlobalTensor<float>& right,
    uint32_t mBase, uint32_t nBase, uint32_t slice, uint16_t slot)
{
    const uint32_t kBase = slice * CHER2K_CUBE_K;
    const uint64_t aOffset =
        transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + kBase;
    const uint64_t bOffset =
        transB_ ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
    WaitFlag<HardEvent::MTE1_MTE2>(slot);
    DataCopy(aSlot, left[aOffset], aCopyParams_);
    DataCopy(bSlot, right[bOffset], bNzParams_);
    SetFlag<HardEvent::MTE2_MTE1>(slot);
}

__aicore__ inline void ProcessProductSlicePrefetch(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    constexpr uint32_t aSize = CHER2K_CUBE_M * CHER2K_CUBE_K;
    constexpr uint32_t bSize = CHER2K_CUBE_K * CHER2K_CUBE_N;
    LocalTensor<float> aSlot0{TPosition::A1, 0U, aSize};
    LocalTensor<float> aSlot1{TPosition::A1, static_cast<uint64_t>(aSize) * sizeof(float), aSize};
    LocalTensor<float> bSlot0{TPosition::B1, static_cast<uint64_t>(2U * aSize) * sizeof(float), bSize};
    LocalTensor<float> bSlot1{TPosition::B1, static_cast<uint64_t>(2U * aSize + bSize) * sizeof(float), bSize};
    LocalTensor<float>* aSlots[2] = {&aSlot0, &aSlot1};
    LocalTensor<float>* bSlots[2] = {&bSlot0, &bSlot1};

    BeginSlicePrefetchEvents();

    const uint32_t sliceCount = (k_ + CHER2K_CUBE_K - 1U) / CHER2K_CUBE_K;
    CopyProductSlice(aSlot0, bSlot0, left, right, mBase, nBase, 0U, EVENT_ID0);
    for (uint32_t slice = 0U; slice < sliceCount; ++slice) {
        const uint32_t slot = slice & 1U;
        WaitFlag<HardEvent::MTE2_MTE1>(static_cast<uint16_t>(slot));
        LoadData(b2, *bSlots[slot], bLoadParams_);
        LoadData(a2, *aSlots[slot], aLoadParams_);
        SetFlag<HardEvent::MTE1_MTE2>(static_cast<uint16_t>(slot));

        const uint32_t nextSlice = slice + 1U;
        if (nextSlice < sliceCount) {
            const uint32_t nextSlot = nextSlice & 1U;
            CopyProductSlice(
                *aSlots[nextSlot], *bSlots[nextSlot], left, right, mBase, nBase, nextSlice,
                static_cast<uint16_t>(nextSlot));
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, slice == 0U ? firstMmadParams_ : accumulateMmadParams_);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
}

__aicore__ inline void CopyProductPanelSlot(
    LocalTensor<float>& aSlot, LocalTensor<float>& bSlot, GlobalTensor<float>& left, GlobalTensor<float>& right,
    uint32_t mBase, uint32_t nBase, uint32_t panel, uint16_t slot)
{
    WaitFlag<HardEvent::MTE1_MTE2>(slot);
    const uint32_t panelBase = panel * CHER2K_L1_PANEL_K;
    const uint64_t aOffset =
        transA_ ? static_cast<uint64_t>(panelBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + panelBase;
    const uint64_t bOffset =
        transB_ ? static_cast<uint64_t>(nBase) * k_ + panelBase : static_cast<uint64_t>(panelBase) * n_ + nBase;
    DataCopy(aSlot, left[aOffset], aPanelCopyParams_);
    DataCopy(bSlot, right[bOffset], bPanelCopyParams_);
    SetFlag<HardEvent::MTE2_MTE1>(slot);
}

__aicore__ inline void ComputeProductPanelSlot(
    LocalTensor<float>& aSlot, LocalTensor<float>& bSlot, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint32_t panel, uint16_t slot)
{
    constexpr uint32_t slices = CHER2K_L1_PANEL_K / CHER2K_CUBE_K;
    for (uint32_t slice = 0U; slice < slices; ++slice) {
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        const uint32_t panelK = slice * CHER2K_CUBE_K;
        aPanelLoadParams_.mStartPt = transA_ ? panelK : 0U;
        aPanelLoadParams_.kStartPt = transA_ ? 0U : panelK;
        bPanelLoadParams_.mStartPt = transB_ ? 0U : panelK;
        bPanelLoadParams_.kStartPt = transB_ ? panelK : 0U;
        LoadData(b2, bSlot, bPanelLoadParams_);
        LoadData(a2, aSlot, aPanelLoadParams_);
        if (slice + 1U == slices) {
            SetFlag<HardEvent::MTE1_MTE2>(slot);
        }
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, panel == 0U && slice == 0U ? firstMmadParams_ : accumulateMmadParams_);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    }
}

// L1 double-buffered panel route.  Two 128-wide K panels occupy
// 2*(M*128 + 128*N) floats, leaving headroom in the 512-KB L1.  The next
// panel's GM copies are issued before the current panel's final MMAD;
// M_MTE1 protects the single L0 slot from premature overwrite.
__aicore__ inline void ProcessProductPanelPingPong(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    const uint32_t panelCount = k_ / CHER2K_L1_PANEL_K;
    const uint32_t aL1Size = CHER2K_CUBE_M * CHER2K_L1_PANEL_K;
    const uint32_t bL1Size = CHER2K_L1_PANEL_K * CHER2K_CUBE_N;
    const uint32_t panelSlots = 2U;
    (void)panelSlots;
    LocalTensor<float> a1Slot0{TPosition::A1, 0U, aL1Size};
    LocalTensor<float> a1Slot1{TPosition::A1, static_cast<uint64_t>(aL1Size) * sizeof(float), aL1Size};
    LocalTensor<float> b1Slot0{TPosition::B1, static_cast<uint64_t>(2U * aL1Size) * sizeof(float), bL1Size};
    LocalTensor<float> b1Slot1{TPosition::B1, static_cast<uint64_t>(2U * aL1Size + bL1Size) * sizeof(float), bL1Size};
    LocalTensor<float>* aSlots[2] = {&a1Slot0, &a1Slot1};
    LocalTensor<float>* bSlots[2] = {&b1Slot0, &b1Slot1};

    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    CopyProductPanelSlot(*aSlots[0], *bSlots[0], left, right, mBase, nBase, 0U, EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    for (uint32_t panel = 0U; panel < panelCount; ++panel) {
        const uint16_t slot = static_cast<uint16_t>(panel & 1U);
        if (panel != 0U) {
            CopyProductPanelSlot(*aSlots[slot], *bSlots[slot], left, right, mBase, nBase, panel, slot);
            WaitFlag<HardEvent::MTE2_MTE1>(slot);
        }
        ComputeProductPanelSlot(*aSlots[slot], *bSlots[slot], a2, b2, c1, panel, slot);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

#endif

__aicore__ inline void ProcessProductPanel(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    ProcessProductPanel(a1, b1, a2, b2, c1, left, right, output, mBase, nBase, false, false, left, right, mBase, nBase);
}

__aicore__ inline void CopyPanelToL1(
    LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
    uint32_t mBase, uint32_t nBase, uint32_t panelBase)
{
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    const uint64_t aOffset =
        transA_ ? static_cast<uint64_t>(panelBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + panelBase;
    const uint64_t bOffset =
        transB_ ? static_cast<uint64_t>(nBase) * k_ + panelBase : static_cast<uint64_t>(panelBase) * n_ + nBase;
    DataCopy(a1, left[aOffset], aPanelCopyParams_);
    DataCopy(b1, right[bOffset], bPanelCopyParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
}

// One independent K-panel reduction unit. C1 is initialized for every
// panel, so the result is a partial plane that can be reduced by AIV.
// The input pointers already identify the selected real-product stream;
// panelBase is applied explicitly to both GM operands.
__aicore__ inline void ProcessPanelTile(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase, uint32_t panelBase)
{
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    CopyPanelToL1(a1, b1, left, right, mBase, nBase, panelBase);
    for (uint32_t iter = 0U; iter < CHER2K_L1_PANEL_K / CHER2K_CUBE_K; ++iter) {
        const uint32_t slice =
            CHER2K_REVERSE_PANEL_K_EXPERIMENT ? (CHER2K_L1_PANEL_K / CHER2K_CUBE_K - 1U - iter) : iter;
        const uint32_t panelK = slice * CHER2K_CUBE_K;
        aPanelLoadParams_.mStartPt = transA_ ? panelK : 0U;
        aPanelLoadParams_.kStartPt = transA_ ? 0U : panelK;
        bPanelLoadParams_.mStartPt = transB_ ? 0U : panelK;
        bPanelLoadParams_.kStartPt = transB_ ? panelK : 0U;
        LoadData(b2, b1, bPanelLoadParams_);
        LoadData(a2, a1, aPanelLoadParams_);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(c1, a2, b2, iter == 0U ? firstMmadParams_ : accumulateMmadParams_);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

__aicore__ inline void ProcessProductPanelSlices(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, uint32_t mBase, uint32_t nBase,
    bool& panelPrefetched, bool prefetchNext, GlobalTensor<float>& nextLeft, GlobalTensor<float>& nextRight,
    uint32_t nextMBase, uint32_t nextNBase)
{
    for (uint32_t panelBase = 0U; panelBase < k_; panelBase += CHER2K_L1_PANEL_K) {
        if (panelPrefetched) {
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            panelPrefetched = false;
        } else {
            CopyPanelToL1(a1, b1, left, right, mBase, nBase, panelBase);
        }
        for (uint32_t iter = 0U; iter < CHER2K_L1_PANEL_K / CHER2K_CUBE_K; ++iter) {
            const uint32_t slice =
                CHER2K_REVERSE_PANEL_K_EXPERIMENT ? (CHER2K_L1_PANEL_K / CHER2K_CUBE_K - 1U - iter) : iter;
            const uint32_t panelK = slice * CHER2K_CUBE_K;
            if constexpr (CHER2K_PRECOMPUTE_PANEL_LOAD_PARAMS_EXPERIMENT) {
                LoadData(b2, b1, bPanelLoadParamsSlices_[slice]);
                LoadData(a2, a1, aPanelLoadParamsSlices_[slice]);
            } else {
                aPanelLoadParams_.mStartPt = transA_ ? panelK : 0U;
                aPanelLoadParams_.kStartPt = transA_ ? 0U : panelK;
                bPanelLoadParams_.mStartPt = transB_ ? 0U : panelK;
                bPanelLoadParams_.kStartPt = transB_ ? panelK : 0U;
                LoadData(b2, b1, bPanelLoadParams_);
                LoadData(a2, a1, aPanelLoadParams_);
            }
            const bool lastSlice =
                CHER2K_REVERSE_PANEL_K_EXPERIMENT ? (slice == 0U) : (slice + 1U == CHER2K_L1_PANEL_K / CHER2K_CUBE_K);
            if (lastSlice) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            }
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            Mmad(c1, a2, b2, panelBase == 0U && iter == 0U ? firstMmadParams_ : accumulateMmadParams_);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ inline void ProcessProductPanel(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase, bool firstPanelReady, bool prefetchNext, GlobalTensor<float>& nextLeft,
    GlobalTensor<float>& nextRight, uint32_t nextMBase, uint32_t nextNBase)
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    bool panelPrefetched = firstPanelReady;
    if constexpr (CHER2K_UNROLL_PANEL_LOAD_EXPERIMENT) {
        for (uint32_t panelBase = 0U; panelBase < k_; panelBase += CHER2K_L1_PANEL_K) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            const uint64_t aOffset =
                transA_ ? static_cast<uint64_t>(panelBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + panelBase;
            const uint64_t bOffset =
                transB_ ? static_cast<uint64_t>(nBase) * k_ + panelBase : static_cast<uint64_t>(panelBase) * n_ + nBase;
            DataCopy(a1, left[aOffset], aPanelCopyParams_);
            DataCopy(b1, right[bOffset], bPanelCopyParams_);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            IssuePanelSlice<0U, false>(a1, b1, a2, b2, c1, panelBase == 0U);
            IssuePanelSlice<CHER2K_CUBE_K, false>(a1, b1, a2, b2, c1, false);
            IssuePanelSlice<2U * CHER2K_CUBE_K, false>(a1, b1, a2, b2, c1, false);
            IssuePanelSlice<3U * CHER2K_CUBE_K, true>(a1, b1, a2, b2, c1, false);
        }
    } else {
        ProcessProductPanelSlices(
            a1, b1, a2, b2, c1, left, right, mBase, nBase, panelPrefetched, prefetchNext, nextLeft, nextRight,
            nextMBase, nextNBase);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

// Unit-flag variant of the panel path.  This is intentionally separate
// from ProcessProductPanel: unitFlag=2/3 owns the L0C lifetime and must
// never be mixed with the legacy M_FIX/FIX_M protocol.  Every product
// ends with a unitFlag=3 MMAD followed by a unitFlag=3 Fixpipe, allowing
// the next product in the MIX producer to reuse C1 without an event wait.
__aicore__ inline void ProcessProductPanelUnitFlag(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    for (uint32_t panelBase = 0U; panelBase < k_; panelBase += CHER2K_L1_PANEL_K) {
        CopyPanelToL1(a1, b1, left, right, mBase, nBase, panelBase);

        const bool firstPanel = panelBase == 0U;
        for (uint32_t slice = 0U; slice < CHER2K_L1_PANEL_K / CHER2K_CUBE_K; ++slice) {
            const uint32_t panelK = CHER2K_REVERSE_PANEL_K_EXPERIMENT ?
                                        (CHER2K_L1_PANEL_K / CHER2K_CUBE_K - 1U - slice) * CHER2K_CUBE_K :
                                        slice * CHER2K_CUBE_K;
            aPanelLoadParams_.mStartPt = transA_ ? panelK : 0U;
            aPanelLoadParams_.kStartPt = transA_ ? 0U : panelK;
            bPanelLoadParams_.mStartPt = transB_ ? 0U : panelK;
            bPanelLoadParams_.kStartPt = transB_ ? panelK : 0U;
            LoadData(b2, b1, bPanelLoadParams_);
            LoadData(a2, a1, aPanelLoadParams_);
            const bool lastSlice =
                CHER2K_REVERSE_PANEL_K_EXPERIMENT ? (panelK == 0U) : (slice + 1U == CHER2K_L1_PANEL_K / CHER2K_CUBE_K);
            if (lastSlice) {
                // L0 owns the loaded slice after this point; the L1 slot
                // can be refilled while MMAD consumes L0A/L0B.
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            }
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            MmadParams params = firstPanel && slice == 0U ? firstMmadParams_ : accumulateMmadParams_;
            const bool finalMmad = (panelBase + CHER2K_L1_PANEL_K >= k_) && lastSlice;
            params.unitFlag = finalMmad ? 3U : 2U;
            Mmad(c1, a2, b2, params);
        }
    }
    StoreSingleToGmUnitFlag(c1, output, mBase, nBase);
}

template <uint32_t PANEL_K, bool LAST>
__aicore__ inline void IssuePanelSlice(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, bool initialize)
{
    // transA_=true/transB_=false on the production route.  Keep the
    // conditional assignments for the generic C path while making the
    // common panel offsets compile-time constants.
    aPanelLoadParams_.mStartPt = transA_ ? PANEL_K : 0U;
    aPanelLoadParams_.kStartPt = transA_ ? 0U : PANEL_K;
    bPanelLoadParams_.mStartPt = transB_ ? 0U : PANEL_K;
    bPanelLoadParams_.kStartPt = transB_ ? PANEL_K : 0U;
    LoadData(b2, b1, bPanelLoadParams_);
    if (!transA_ && !transB_) {
        LoadDirectA2D(a2, a1, PANEL_K / CHER2K_CUBE_K);
    } else {
        LoadData(a2, a1, aPanelLoadParams_);
    }
    if constexpr (LAST) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    Mmad(c1, a2, b2, initialize ? firstMmadParams_ : accumulateMmadParams_);
}

// L0 ping-pong experiment for the 128x128x64 geometry.  The next panel
// is copied/loaded into the alternate slot while the previous MMAD is in
// flight; M_MTE1 ownership prevents overwriting an active L0 slot.
__aicore__ inline void ProcessProductSinglePingPong(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    constexpr uint32_t aSize = CHER2K_CUBE_M * CHER2K_CUBE_K;
    constexpr uint32_t bSize = CHER2K_CUBE_K * CHER2K_CUBE_N;
    LocalTensor<float> a1Alt{TPosition::A1, aSize * sizeof(float), aSize};
    // A1 and B1 are independent local address spaces; each gets two
    // buffers starting at offset zero in its own space.
    LocalTensor<float> b1Base{TPosition::B1, 0U, bSize};
    LocalTensor<float> b1Alt{TPosition::B1, bSize * sizeof(float), bSize};
    LocalTensor<float> a2Alt{TPosition::A2, aSize * sizeof(float), aSize};
    LocalTensor<float> b2Base{TPosition::B2, 0U, bSize};
    LocalTensor<float> b2Alt{TPosition::B2, bSize * sizeof(float), bSize};
    for (uint32_t kBase = 0U; kBase < k_; kBase += CHER2K_CUBE_K) {
        const uint32_t slot = (kBase / CHER2K_CUBE_K) & 1U;
        LocalTensor<float>& a1Slot = slot == 0U ? a1 : a1Alt;
        LocalTensor<float>& b1Slot = slot == 0U ? b1Base : b1Alt;
        LocalTensor<float>& a2Slot = slot == 0U ? a2 : a2Alt;
        LocalTensor<float>& b2Slot = slot == 0U ? b2Base : b2Alt;
        WaitFlag<HardEvent::M_MTE1>(slot);
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
        const uint64_t aOffset =
            transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase : static_cast<uint64_t>(mBase) * k_ + kBase;
        const uint64_t bOffset =
            transB_ ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
        DataCopy(a1Slot, left[aOffset], aCopyParams_);
        DataCopy(b1Slot, right[bOffset], bNzParams_);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        LoadData(b2Slot, b1Slot, bLoadParams_);
        LoadData(a2Slot, a1Slot, aLoadParams_);
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        SetFlag<HardEvent::MTE1_M>(slot);
        WaitFlag<HardEvent::MTE1_M>(slot);
        Mmad(c1, a2Slot, b2Slot, kBase == 0U ? firstMmadParams_ : accumulateMmadParams_);
        SetFlag<HardEvent::M_MTE1>(slot);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreSingleToGm(c1, output, mBase, nBase);
}

__aicore__ inline void StoreSingleToGm(
    LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
{
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 params = {};
    params.mSize = CHER2K_CUBE_M;
    params.nSize = CHER2K_CUBE_N;
    params.srcStride = CHER2K_CUBE_M;
    params.dstStride = n_;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}

__aicore__ inline void StoreSingleToGmUnitFlag(
    LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
{
    FixpipeParamsV220 params = {};
    params.mSize = CHER2K_CUBE_M;
    params.nSize = CHER2K_CUBE_N;
    params.srcStride = CHER2K_CUBE_M;
    params.dstStride = n_;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    params.unitFlag = 3U;
    Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
}

__aicore__ inline void CopyPairToL1(
    LocalTensor<float>& a10, LocalTensor<float>& a11, LocalTensor<float>& b1, GlobalTensor<float>& left,
    GlobalTensor<float>& right, uint32_t mBase0, uint32_t mBase1, uint32_t nBase, uint32_t kBase)
{
    const uint64_t aOffset0 =
        transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase0 : static_cast<uint64_t>(mBase0) * k_ + kBase;
    DataCopy(a10, left[aOffset0], aCopyParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    const uint64_t aOffset1 =
        transA_ ? static_cast<uint64_t>(kBase) * n_ + mBase1 : static_cast<uint64_t>(mBase1) * k_ + kBase;
    DataCopy(a11, left[aOffset1], aCopyParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    const uint64_t bOffset =
        transB_ ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
    DataCopy(b1, right[bOffset], bNzParams_);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
}

__aicore__ inline void LoadPairToL0(
    LocalTensor<float>& a10, LocalTensor<float>& a11, LocalTensor<float>& b1, LocalTensor<float>& a20,
    LocalTensor<float>& a21, LocalTensor<float>& b2)
{
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    LoadData(b2, b1, bLoadParams_);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData(a20, a10, aLoadParams_);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    LoadData(a21, a11, aLoadParams_);
    // B, A0, and A1 now form one MTE1 batch. Do not overwrite the L1
    // panels until the next CopyPairToL1 consumes this completion token.
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
}

__aicore__ inline void Compute(
    LocalTensor<float>& c1, LocalTensor<float>& a2, LocalTensor<float>& b2, const MmadParams& params)
{
    // The caller selects the matching MTE1_M token before each Mmad.
    Mmad(c1, a2, b2, params);
}

__aicore__ inline void StoreToGm(
    LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase, uint32_t eventId)
{
    if (eventId == EVENT_ID0) {
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    } else {
        WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
    }
    FixpipeParamsV220 params = {};
    params.mSize = CHER2K_CUBE_M;
    params.nSize = CHER2K_CUBE_N;
    params.srcStride = CHER2K_CUBE_M;
    params.dstStride = n_;
    params.ndNum = 1;
    params.srcNdStride = 0;
    params.dstNdStride = 0;
    Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
    if (eventId == EVENT_ID0) {
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    } else {
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }
}

__aicore__ inline void ProcessProductPair(
    LocalTensor<float>& a10, LocalTensor<float>& a11, LocalTensor<float>& b1, LocalTensor<float>& a20,
    LocalTensor<float>& a21, LocalTensor<float>& b2, LocalTensor<float>& c10, LocalTensor<float>& c11,
    GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output, uint32_t mBase0,
    uint32_t mBase1, uint32_t nBase)
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
    for (uint32_t kBase = 0U; kBase < k_; kBase += CHER2K_CUBE_K) {
        if (kBase != 0U && !CHER2K_SKIP_K_PIPE_BARRIER_EXPERIMENT) {
            PipeBarrier<PIPE_M>();
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        CopyPairToL1(a10, a11, b1, left, right, mBase0, mBase1, nBase, kBase);
        LoadPairToL0(a10, a11, b1, a20, a21, b2);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Compute(c10, a20, b2, kBase == 0U ? firstMmadParams_ : accumulateMmadParams_);
        PipeBarrier<PIPE_M>();
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);
        Compute(c11, a21, b2, kBase == 0U ? firstMmadParams_ : accumulateMmadParams_);
    }
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    StoreToGm(c10, output, mBase0, nBase, EVENT_ID0);
    if constexpr (CHER2K_PAIRED_SECOND_C_DIAGNOSTIC) {
        SetFlag<HardEvent::M_FIX>(EVENT_ID1);
        StoreToGm(c11, output, mBase1, nBase, EVENT_ID1);
    }
}
GlobalTensor<float> ar_;
GlobalTensor<float> ai_;
GlobalTensor<float> br_;
GlobalTensor<float> bi_;
GlobalTensor<float> sumA_;
GlobalTensor<float> sumB_;
GlobalTensor<float> rr_;
GlobalTensor<float> ii_;
GlobalTensor<float> ri_;
GlobalTensor<float> ir_;
uint32_t n_ = 0U;
uint32_t k_ = 0U;
bool transA_ = false;
bool transB_ = false;
uint32_t tilingUplo_ = ACLBLAS_UPPER;
Nd2NzParams aCopyParams_;
Nd2NzParams bNzParams_;
Nd2NzParams aPanelCopyParams_;
Nd2NzParams bPanelCopyParams_;
LoadData3DParamsV2<float> aLoadParams_;
LoadData3DParamsV2<float> bLoadParams_;
LoadData3DParamsV2<float> aPanelLoadParams_;
LoadData3DParamsV2<float> bPanelLoadParams_;
LoadData3DParamsV2<float> aPanelLoadParamsSlices_[CHER2K_L1_PANEL_K / CHER2K_CUBE_K];
LoadData3DParamsV2<float> bPanelLoadParamsSlices_[CHER2K_L1_PANEL_K / CHER2K_CUBE_K];
MmadParams firstMmadParams_;
MmadParams accumulateMmadParams_;

private:
__aicore__ inline void InitCopyParams()
{
    aCopyParams_ = {};
    aCopyParams_.ndNum = 1;
    aCopyParams_.nValue = transA_ ? CHER2K_CUBE_K : CHER2K_CUBE_M;
    aCopyParams_.dValue = transA_ ? CHER2K_CUBE_M : CHER2K_CUBE_K;
    aCopyParams_.srcNdMatrixStride = 0;
    aCopyParams_.srcDValue = transA_ ? n_ : k_;
    aCopyParams_.dstNzC0Stride = transA_ ? CHER2K_CUBE_K : CHER2K_CUBE_M;
    aCopyParams_.dstNzNStride = 1;
    aCopyParams_.dstNzMatrixStride = 0;
    bNzParams_ = {};
    bNzParams_.ndNum = 1;
    bNzParams_.nValue = transB_ ? CHER2K_CUBE_N : CHER2K_CUBE_K;
    bNzParams_.dValue = transB_ ? CHER2K_CUBE_K : CHER2K_CUBE_N;
    bNzParams_.srcNdMatrixStride = 0;
    bNzParams_.srcDValue = transB_ ? k_ : n_;
    bNzParams_.dstNzC0Stride = transB_ ? CHER2K_CUBE_N : CHER2K_CUBE_K;
    bNzParams_.dstNzNStride = 1;
    bNzParams_.dstNzMatrixStride = 0;
}

__aicore__ inline void InitLoadParams()
{
    aLoadParams_ = {};
    aLoadParams_.l1H = 1U;
    aLoadParams_.l1W = transA_ ? CHER2K_CUBE_K : CHER2K_CUBE_M;
    aLoadParams_.channelSize = transA_ ? CHER2K_CUBE_M : CHER2K_CUBE_K;
    aLoadParams_.kExtension = transA_ ? CHER2K_CUBE_M : CHER2K_CUBE_K;
    aLoadParams_.mExtension = transA_ ? CHER2K_CUBE_K : CHER2K_CUBE_M;
    cube_load::InitUnitFilter(aLoadParams_);
    aLoadParams_.enTranspose = transA_;
    aLoadParams_.fMatrixCtrl = false;
    bLoadParams_ = aLoadParams_;
    bLoadParams_.l1W = transB_ ? CHER2K_CUBE_N : CHER2K_CUBE_K;
    bLoadParams_.channelSize = transB_ ? CHER2K_CUBE_K : CHER2K_CUBE_N;
    bLoadParams_.kExtension = transB_ ? CHER2K_CUBE_K : CHER2K_CUBE_N;
    bLoadParams_.mExtension = transB_ ? CHER2K_CUBE_N : CHER2K_CUBE_K;
    bLoadParams_.enTranspose = true;
}

__aicore__ inline void InitPanelParams()
{
    aPanelCopyParams_ = aCopyParams_;
    aPanelCopyParams_.nValue = transA_ ? CHER2K_L1_PANEL_K : CHER2K_CUBE_M;
    aPanelCopyParams_.dValue = transA_ ? CHER2K_CUBE_M : CHER2K_L1_PANEL_K;
    aPanelCopyParams_.dstNzC0Stride = transA_ ? CHER2K_L1_PANEL_K : CHER2K_CUBE_M;
    bPanelCopyParams_ = bNzParams_;
    bPanelCopyParams_.nValue = transB_ ? CHER2K_CUBE_N : CHER2K_L1_PANEL_K;
    bPanelCopyParams_.dValue = transB_ ? CHER2K_L1_PANEL_K : CHER2K_CUBE_N;
    bPanelCopyParams_.dstNzC0Stride = transB_ ? CHER2K_CUBE_N : CHER2K_L1_PANEL_K;
    aPanelLoadParams_ = aLoadParams_;
    aPanelLoadParams_.l1W = transA_ ? CHER2K_L1_PANEL_K : CHER2K_CUBE_M;
    aPanelLoadParams_.channelSize = transA_ ? CHER2K_CUBE_M : CHER2K_L1_PANEL_K;
    bPanelLoadParams_ = bLoadParams_;
    bPanelLoadParams_.l1W = transB_ ? CHER2K_CUBE_N : CHER2K_L1_PANEL_K;
    for (uint32_t slice = 0U; slice < CHER2K_L1_PANEL_K / CHER2K_CUBE_K; ++slice) {
        const uint32_t panelK = slice * CHER2K_CUBE_K;
        aPanelLoadParamsSlices_[slice] = aPanelLoadParams_;
        aPanelLoadParamsSlices_[slice].mStartPt = transA_ ? panelK : 0U;
        aPanelLoadParamsSlices_[slice].kStartPt = transA_ ? 0U : panelK;
        bPanelLoadParamsSlices_[slice] = bPanelLoadParams_;
        bPanelLoadParamsSlices_[slice].mStartPt = transB_ ? 0U : panelK;
        bPanelLoadParamsSlices_[slice].kStartPt = transB_ ? panelK : 0U;
    }
}

__aicore__ inline void InitMmadParams()
{
    firstMmadParams_ = {};
    firstMmadParams_.m = CHER2K_CUBE_M;
    firstMmadParams_.n = CHER2K_CUBE_N;
    firstMmadParams_.k = CHER2K_CUBE_K;
    firstMmadParams_.cmatrixInitVal = true;
    firstMmadParams_.cmatrixSource = false;
    firstMmadParams_.kDirectionAlign = false;
    accumulateMmadParams_ = firstMmadParams_;
    accumulateMmadParams_.cmatrixInitVal = false;
}

// Keep the product stream selection in one place.  ProcessImpl used to
// duplicate the 3M/4M dispatch (and the optional L0 ping-pong branch) in
// its hot loop, which made the scheduler difficult to audit.  The helper
// only selects the operand planes; all copies, barriers and MMAD ordering
// remain in ProcessProductSingle/ProcessProductSinglePingPong.
__aicore__ inline void ProcessSelectedProduct(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
    uint32_t mBase, uint32_t nBase)
{
    if constexpr (CHER2K_ENABLE_L0_PINGPONG_EXPERIMENT) {
        ProcessProductSinglePingPong(a1, b1, a2, b2, c1, left, right, output, mBase, nBase);
    } else {
        ProcessProductSingle(a1, b1, a2, b2, c1, left, right, output, mBase, nBase);
    }
}

template <bool USE_THREE_M>
__aicore__ inline void ProcessProductByIndex(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint32_t product, uint32_t mBase, uint32_t nBase)
{
    if constexpr (USE_THREE_M) {
        if (product == 0U) {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, sumA_, sumB_, rr_, mBase, nBase);
        } else if (product == 1U) {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ar_, br_, ri_, mBase, nBase);
        } else {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ai_, bi_, ir_, mBase, nBase);
        }
    } else {
        if (product == 0U) {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ar_, br_, rr_, mBase, nBase);
        } else if (product == 1U) {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ai_, bi_, ii_, mBase, nBase);
        } else if (product == 2U) {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ar_, bi_, ri_, mBase, nBase);
        } else {
            ProcessSelectedProduct(a1, b1, a2, b2, c1, ai_, br_, ir_, mBase, nBase);
        }
    }
}

__aicore__ inline bool IsCubeTriangleTile(uint32_t tileRow, uint32_t tileCol)
{
    // Cube tiles are 256x128 while postprocess tiles are 64x64.  A cube
    // tile is retained when any contained 64x64 tile belongs to uplo.
    constexpr uint32_t cubeRowTiles = CHER2K_CUBE_M / 64U;
    constexpr uint32_t cubeColTiles = CHER2K_CUBE_N / 64U;
    const uint32_t rowFirst = tileRow * cubeRowTiles;
    const uint32_t rowLast = rowFirst + cubeRowTiles - 1U;
    const uint32_t colFirst = tileCol * cubeColTiles;
    const uint32_t colLast = colFirst + cubeColTiles - 1U;
    return tilingUplo_ == ACLBLAS_UPPER ? rowFirst <= colLast : rowLast >= colFirst;
}

__aicore__ inline void AdvanceProductTile(
    uint32_t& product, uint32_t& tileRow, uint32_t& tileCol, uint32_t nTiles, uint32_t productCount, uint32_t taskStep)
{
    const uint32_t advancedProduct = product + taskStep;
    const uint32_t tileAdvance = advancedProduct / (productCount == 0U ? 1U : productCount);
    product = advancedProduct - tileAdvance * productCount;
    for (uint32_t advance = 0U; advance < tileAdvance; ++advance) {
        ++tileCol;
        if (tileCol == nTiles) {
            tileCol = 0U;
            ++tileRow;
        }
    }
}

template <bool USE_THREE_M>
__aicore__ inline void ProcessTileStream(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint64_t taskBegin, uint64_t taskEnd, uint64_t taskStep, uint32_t nTiles,
    uint32_t productCount)
{
    uint64_t task = taskBegin;
    const uint32_t safeProductCount = productCount == 0U ? 1U : productCount;
    const uint32_t safeNTiles = nTiles == 0U ? 1U : nTiles;
    uint64_t tile = task / safeProductCount;
    uint32_t product = static_cast<uint32_t>(task - tile * safeProductCount);
    uint32_t tileRow = static_cast<uint32_t>(tile / safeNTiles);
    uint32_t tileCol = static_cast<uint32_t>(tile - static_cast<uint64_t>(tileRow) * safeNTiles);
    for (; task < taskEnd; task += taskStep) {
        const uint32_t mBase = tileRow * CHER2K_CUBE_M;
        const uint32_t nBase = tileCol * CHER2K_CUBE_N;
        if constexpr (CHER2K_TRIANGLE_CUBE_EXPERIMENT) {
            if (!IsCubeTriangleTile(tileRow, tileCol)) {
                AdvanceProductTile(product, tileRow, tileCol, nTiles, productCount, static_cast<uint32_t>(taskStep));
                continue;
            }
        }
        ProcessProductByIndex<USE_THREE_M>(a1, b1, a2, b2, c1, product, mBase, nBase);
        AdvanceProductTile(product, tileRow, tileCol, nTiles, productCount, static_cast<uint32_t>(taskStep));
    }
}

template <bool USE_THREE_M>
__aicore__ inline void ProcessTileMajor(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint64_t tileCount, uint32_t nTiles, uint32_t taskIndex, uint32_t taskStride)
{
    for (uint64_t tile = taskIndex; tile < tileCount; tile += taskStride) {
        const uint32_t safeNTiles = nTiles == 0U ? 1U : nTiles;
        const uint32_t mBase = static_cast<uint32_t>((tile / safeNTiles) * CHER2K_CUBE_M);
        const uint32_t nBase = static_cast<uint32_t>((tile % safeNTiles) * CHER2K_CUBE_N);
        if constexpr (USE_THREE_M) {
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 0U, mBase, nBase);
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 1U, mBase, nBase);
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 2U, mBase, nBase);
        } else {
            ProcessProductByIndex<false>(a1, b1, a2, b2, c1, 0U, mBase, nBase);
            ProcessProductByIndex<false>(a1, b1, a2, b2, c1, 1U, mBase, nBase);
            ProcessProductByIndex<false>(a1, b1, a2, b2, c1, 2U, mBase, nBase);
            ProcessProductByIndex<false>(a1, b1, a2, b2, c1, 3U, mBase, nBase);
        }
    }
}

__aicore__ inline void ProcessGroupedThreeM(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint64_t tileCount, uint32_t nTiles, uint32_t taskIndex, uint32_t taskStride)
{
    const uint64_t groupCount = (tileCount + CHER2K_TILE_GROUP_SIZE - 1U) / CHER2K_TILE_GROUP_SIZE;
    for (uint64_t group = taskIndex; group < groupCount; group += taskStride) {
        const uint64_t firstTile = group * CHER2K_TILE_GROUP_SIZE;
        const uint64_t lastTile = min(tileCount, firstTile + CHER2K_TILE_GROUP_SIZE);
        for (uint64_t tile = firstTile; tile < lastTile; ++tile) {
            const uint32_t mBase = static_cast<uint32_t>((tile / (nTiles == 0U ? 1U : nTiles)) * CHER2K_CUBE_M);
            const uint32_t nBase = static_cast<uint32_t>((tile % (nTiles == 0U ? 1U : nTiles)) * CHER2K_CUBE_N);
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 0U, mBase, nBase);
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 1U, mBase, nBase);
            ProcessProductByIndex<true>(a1, b1, a2, b2, c1, 2U, mBase, nBase);
        }
    }
}

__aicore__ inline void BeginProductStreamEvents()
{
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    if constexpr (CHER2K_L1_PINGPONG_EXPERIMENT) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
    } else if constexpr (CHER2K_ENABLE_L0_PINGPONG_EXPERIMENT) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    } else if constexpr (!CHER2K_L1_SLICE_PREFETCH_EXPERIMENT) {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    SetHF32Mode(HF32Mode::DISABLE);
}

__aicore__ inline void EndProductStreamEvents()
{
    if constexpr (!CHER2K_L1_SLICE_PREFETCH_EXPERIMENT) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

template <bool USE_THREE_M>
__aicore__ inline bool ProcessSpecialProductSchedule(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, uint64_t tileCount, uint32_t nTiles, uint32_t taskIndex, uint32_t taskStride)
{
    if constexpr (CHER2K_TILE_GROUP_EXPERIMENT && USE_THREE_M) {
        ProcessGroupedThreeM(a1, b1, a2, b2, c1, tileCount, nTiles, taskIndex, taskStride);
        EndProductStreamEvents();
        return true;
    }
    if constexpr (CHER2K_TILE_MAJOR_EXPERIMENT) {
        ProcessTileMajor<USE_THREE_M>(a1, b1, a2, b2, c1, tileCount, nTiles, taskIndex, taskStride);
        EndProductStreamEvents();
        return true;
    }
    return false;
}
}
;
struct Cher2kPreprocessContext {
    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> ws;
    bool transN = false;
    bool directOpc = false;
    bool directOutputOpc = false;
    uint32_t physicalRows = 0U;
    uint32_t physicalCols = 0U;
    uint32_t packedStride = 0U;
    uint32_t packedRows = 0U;
    uint64_t arBase = 0ULL;
    uint64_t aiBase = 0ULL;
    uint64_t sumABase = 0ULL;
    uint64_t brBase = 0ULL;
    uint64_t biBase = 0ULL;
    uint64_t sumBBase = 0ULL;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const Cher2kTilingData& tiling)
    {
        transN = tiling.trans == ACLBLAS_OP_N;
        directOpc = CHER2K_DIRECT_OPC_CUBE_LAYOUT_EXPERIMENT && tiling.directHermitianPath != 0U && !transN;
        directOutputOpc = directOpc && tiling.directOutputPath != 0U;
        physicalRows = transN ? tiling.n : tiling.k;
        physicalCols = transN ? tiling.k : tiling.n;
        packedStride = transN ? tiling.nAligned : tiling.kAligned;
        packedRows = transN ? tiling.kAligned : tiling.nAligned;
        aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * physicalCols * 2ULL);
        bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * physicalCols * 2ULL);
        const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
        const bool packedSg3Input = tiling.sg3MmadPath != 0U || directOpc;
        ws.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace),
            packedSg3Input ? Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned) :
                             WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, tiling.useThreeM != 0U));
        aiBase = nk;
        sumABase = packedSg3Input ? 2ULL * nk : OffsetSumA(tiling.nAligned, tiling.kAligned);
        brBase = packedSg3Input ? 3ULL * nk : OffsetBr(tiling.nAligned, tiling.kAligned);
        biBase = packedSg3Input ? 4ULL * nk : OffsetBi(tiling.nAligned, tiling.kAligned);
        sumBBase = packedSg3Input ? 5ULL * nk : OffsetSumB(tiling.nAligned, tiling.kAligned);
    }
};

__aicore__ inline void Cher2kPreprocessImpl(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GM_ADDR sharedOffsets, const Cher2kTilingData& tiling, uint32_t blockIdx,
    uint32_t blockNum, TPipe& pipe)
{
    TBuf<TPosition::VECCALC> aosBuf;
    TBuf<TPosition::VECCALC> realBuf;
    TBuf<TPosition::VECCALC> imagBuf;
    pipe.InitBuffer(
        aosBuf, CHER2K_ROW_BATCH_ELEMENTS * (CHER2K_FUSE_OPC_A_B_LOADS_EXPERIMENT ? 4U : 2U) * sizeof(float));
    pipe.InitBuffer(realBuf, CHER2K_ROW_BATCH_ELEMENTS * sizeof(float));
    pipe.InitBuffer(imagBuf, CHER2K_ROW_BATCH_ELEMENTS * sizeof(float));
    LocalTensor<float> aosLocal = aosBuf.Get<float>();
    LocalTensor<float> aosSecondLocal = aosLocal[CHER2K_ROW_BATCH_ELEMENTS * 2U];
    LocalTensor<float> realLocal = realBuf.Get<float>();
    LocalTensor<float> imagLocal = imagBuf.Get<float>();

    Cher2kPreprocessContext context;
    context.Init(a, b, workspace, tiling);
    if (blockNum == 0U) {
        blockNum = 1U;
    }

    // Event IDs preserve the GM -> UB -> vector -> GM buffer lifetime.
    const event_t eventMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    const event_t eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    const event_t eventMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

    if (!context.transN) {
        Cher2kPreprocessTransposePath(
            sharedOffsets, context.aG, context.bG, context.ws, aosLocal, aosSecondLocal, realLocal, imagLocal, tiling,
            blockIdx, blockNum, context.directOpc, context.directOutputOpc, context.arBase, context.aiBase,
            context.sumABase, context.brBase, context.biBase, context.sumBBase, pipe, eventVToMte3);
        return;
    }
    Cher2kPreprocessNormalBatches(
        context.aG, context.bG, context.ws, aosLocal, realLocal, imagLocal, tiling, blockIdx, blockNum,
        context.physicalRows, context.physicalCols, context.packedStride, context.packedRows, context.arBase,
        context.aiBase, context.brBase, context.biBase, context.sumABase, context.sumBBase, eventMte2ToV, eventVToMte3,
        eventMte3ToV);
}

extern "C" __global__ __aicore__ void cher2k_preprocess_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GM_ADDR sharedOffsets, GM_ADDR alpha, GM_ADDR beta,
    uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    // Value-agnostic when dispatched with ALL; the MIX fallback queues it with
    // NON_UNIT so the unit-scalar producer keeps exclusive ownership.
    if (Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }
    TPipe pipe;
    Cher2kPreprocessImpl(
        a, b, workspace, sharedOffsets, tiling, static_cast<uint32_t>(GetBlockIdx()),
        static_cast<uint32_t>(GetBlockNum()), pipe);
}

using Cher2kNormalMatmulType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
using Cher2kTransMatmulType = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
constexpr MatmulShapeParams kCher2kMatmulShape{256, 256, 256, 128, 128, 64};
constexpr MatmulFuncParams kCher2kMatmulFunc{
    false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true};
constexpr MatmulBiasParams kCher2kMatmulBias{false};
constexpr MatmulConfig kCher2kMatmulConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_MDL>(kCher2kMatmulShape, kCher2kMatmulFunc, kCher2kMatmulBias);
constexpr MatmulApiStaticTiling kCher2kNStatic =
    GetMatmulApiTiling<Cher2kTransMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType>(
        kCher2kMatmulConfig);
constexpr MatmulApiStaticTiling kCher2kCStatic =
    GetMatmulApiTiling<Cher2kNormalMatmulType, Cher2kTransMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType>(
        kCher2kMatmulConfig);
using Cher2kNMatmul = MatmulImpl<
    Cher2kTransMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType, kCher2kNStatic>;
using Cher2kCMatmul = MatmulImpl<
    Cher2kNormalMatmulType, Cher2kTransMatmulType, Cher2kNormalMatmulType, Cher2kNormalMatmulType, kCher2kCStatic>;
template <typename MatmulT>
__aicore__ inline void Cher2kInitBandMatmul(MatmulT& mm, TPipe& pipe, const Cher2kTilingData& tiling, uint32_t rows)
{
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetOrgShape(rows, tiling.nAligned, tiling.kAligned);
    mm.SetSingleShape(rows, CHER2K_CUBE_N, tiling.kAligned);
}

template <typename MatmulT>
__aicore__ inline void Cher2kRunCube(__gm__ float* base, const Cher2kTilingData& tiling, TPipe& pipe)
{
    constexpr uint32_t bandRows = 256U;
    const uint32_t bandCount = tiling.nAligned / bandRows;
    const uint32_t nTileCount = tiling.nAligned / CHER2K_CUBE_N;
    const uint32_t productCount = tiling.useThreeM != 0U ? 3U : 4U;
    const uint64_t tasksPerProduct = static_cast<uint64_t>(bandCount) * nTileCount;
    // nAligned below one output tile leaves no band/N tile to divide over.
    if (tasksPerProduct == 0U) {
        return;
    }
    const uint64_t taskCount = tasksPerProduct * productCount;
    // One logical task owns one MatmulImpl lifecycle.
    const uint64_t task = static_cast<uint32_t>(GetBlockIdx());
    if (task >= taskCount) {
        return;
    }

    GlobalTensor<float> left;
    GlobalTensor<float> right;
    GlobalTensor<float> output;
    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
    const uint64_t leftOffsets[4] = {
        OffsetSumA(tiling.nAligned, tiling.kAligned), OffsetAr(tiling.nAligned, tiling.kAligned),
        OffsetAi(tiling.nAligned, tiling.kAligned), OffsetAi(tiling.nAligned, tiling.kAligned)};
    const uint64_t rightOffsets[4] = {
        OffsetSumB(tiling.nAligned, tiling.kAligned), OffsetBr(tiling.nAligned, tiling.kAligned),
        OffsetBi(tiling.nAligned, tiling.kAligned), OffsetBr(tiling.nAligned, tiling.kAligned)};
    const uint64_t outputOffsets[4] = {
        OffsetRr(tiling.nAligned, tiling.kAligned), OffsetRi(tiling.nAligned, tiling.kAligned),
        OffsetIr(tiling.nAligned, tiling.kAligned), OffsetIr(tiling.nAligned, tiling.kAligned)};
    const uint32_t product = static_cast<uint32_t>(task / tasksPerProduct);
    const uint64_t tile = task - static_cast<uint64_t>(product) * tasksPerProduct;
    const uint32_t band = static_cast<uint32_t>(tile / nTileCount);
    const uint32_t nTile = static_cast<uint32_t>(tile - static_cast<uint64_t>(band) * nTileCount);
    const uint32_t mBase = band * bandRows;
    const uint64_t leftTileOffset = mBase;
    const uint64_t outputTileOffset = static_cast<uint64_t>(mBase) * tiling.nAligned;
    const uint32_t nBase = nTile * CHER2K_CUBE_N;
    MatmulT mm;
    Cher2kInitBandMatmul(mm, pipe, tiling, bandRows);
    left.SetGlobalBuffer(base + leftOffsets[product] + leftTileOffset, nk - leftTileOffset);
    right.SetGlobalBuffer(base + rightOffsets[product] + nBase, nk - nBase);
    output.SetGlobalBuffer(base + outputOffsets[product] + outputTileOffset + nBase, nn - outputTileOffset - nBase);
    mm.SetTensorA(left, true);
    mm.SetTensorB(right, false);
    mm.IterateAll(output, 0);
    mm.End();
}

// OP_C low-K direct producer.  Unlike the retired NxK-transposed-B probe, this
// class uses the exact operand contracts already proven by Cherk and the packed
// SG3 path: N x K for A-side tiles and K x N for B-side tiles.
class Cher2kKernelCube {
public:
    __aicore__ inline void Init(
        __gm__ float* base, uint32_t n, uint32_t validN, uint32_t k, uint32_t validK, uint32_t uplo)
    {
        const uint64_t nk = static_cast<uint64_t>(n) * k;
        const uint64_t nn = static_cast<uint64_t>(n) * n;
        leftAr_.SetGlobalBuffer(base + DirectOpcLeftAr(n, k), nk);
        leftAi_.SetGlobalBuffer(base + DirectOpcLeftAi(n, k), nk);
        leftBr_.SetGlobalBuffer(base + DirectOpcLeftBr(n, k), nk);
        leftBi_.SetGlobalBuffer(base + DirectOpcLeftBi(n, k), nk);
        rightAr_.SetGlobalBuffer(base + DirectOpcRightAr(n, k), nk);
        rightAi_.SetGlobalBuffer(base + DirectOpcRightAi(n, k), nk);
        rightBr_.SetGlobalBuffer(base + DirectOpcRightBr(n, k), nk);
        rightBi_.SetGlobalBuffer(base + DirectOpcRightBi(n, k), nk);
        leftNegAr_.SetGlobalBuffer(base + DirectOpcLeftNegAr(n, k), nk);
        leftNegBr_.SetGlobalBuffer(base + DirectOpcLeftNegBr(n, k), nk);
        real_.SetGlobalBuffer(base + DirectOpcReal(n, k), nn);
        imag_.SetGlobalBuffer(base + DirectOpcImag(n, k), nn);
        n_ = n;
        validN_ = validN;
        k_ = k;
        computeK_ = (validK + 15U) & ~15U;
        uplo_ = uplo;
    }

    __aicore__ inline void Process(uint32_t taskIndex, uint32_t taskStride)
    {
        ProcessImpl<false>(taskIndex, taskStride);
    }

    __aicore__ inline void ProcessMixed(uint32_t taskIndex, uint32_t taskStride)
    {
        ProcessImpl<true>(taskIndex, taskStride);
    }

private:
    struct DirectOpcState {
        static constexpr uint32_t M = 128U;
        static constexpr uint32_t N = 128U;
        static constexpr uint32_t maxK = CHER2K_CUBE_K;
        static constexpr uint32_t aSize = M * maxK;
        static constexpr uint32_t bSize = maxK * N;
        static constexpr uint32_t cSize = M * N;
        uint32_t panelCapacity = 0U;
        uint32_t bCacheSlots = 0U;
        LocalTensor<float> aAr1;
        LocalTensor<float> aAi1;
        LocalTensor<float> aBr1;
        LocalTensor<float> aBi1;
        LocalTensor<float> aNegAr1;
        LocalTensor<float> aNegBr1;
        LocalTensor<float> bAr10;
        LocalTensor<float> bAi10;
        LocalTensor<float> bBr10;
        LocalTensor<float> bBi10;
        LocalTensor<float> bAr11;
        LocalTensor<float> bAi11;
        LocalTensor<float> bBr11;
        LocalTensor<float> bBi11;
        LocalTensor<float> bAr12;
        LocalTensor<float> bAi12;
        LocalTensor<float> bBr12;
        LocalTensor<float> bBi12;
        LocalTensor<float> a2;
        LocalTensor<float> a2Alt;
        LocalTensor<float> b2;
        LocalTensor<float> b2Alt;
        LocalTensor<float> realC1;
        LocalTensor<float> imagC1;
        Nd2NzParams aCopy{};
        Nd2NzParams bCopyParams{};
        LoadData3DParamsV2<float> aLoad{};
        LoadData3DParamsV2<float> bLoad{};
        MmadParams first{};
        MmadParams accum{};

        __aicore__ inline void Init(uint32_t n, uint32_t computeK)
        {
            panelCapacity = min(maxK, computeK);
            const uint32_t aPanelSize = M * panelCapacity;
            const uint32_t bPanelSize = panelCapacity * N;
            bCacheSlots = computeK <= maxK && panelCapacity <= 48U ? 3U : 2U;
            aAr1 = LocalTensor<float>(TPosition::A1, 0ULL * aPanelSize * sizeof(float), aPanelSize);
            aAi1 = LocalTensor<float>(TPosition::A1, 1ULL * aPanelSize * sizeof(float), aPanelSize);
            aBr1 = LocalTensor<float>(TPosition::A1, 2ULL * aPanelSize * sizeof(float), aPanelSize);
            aBi1 = LocalTensor<float>(TPosition::A1, 3ULL * aPanelSize * sizeof(float), aPanelSize);
            aNegAr1 = LocalTensor<float>(TPosition::A1, 4ULL * aPanelSize * sizeof(float), aPanelSize);
            aNegBr1 = LocalTensor<float>(TPosition::A1, 5ULL * aPanelSize * sizeof(float), aPanelSize);
            const uint64_t bBase = 6ULL * aPanelSize * sizeof(float);
            bAr10 = LocalTensor<float>(TPosition::B1, bBase + 0ULL * bPanelSize * sizeof(float), bPanelSize);
            bAi10 = LocalTensor<float>(TPosition::B1, bBase + 1ULL * bPanelSize * sizeof(float), bPanelSize);
            bBr10 = LocalTensor<float>(TPosition::B1, bBase + 2ULL * bPanelSize * sizeof(float), bPanelSize);
            bBi10 = LocalTensor<float>(TPosition::B1, bBase + 3ULL * bPanelSize * sizeof(float), bPanelSize);
            bAr11 = LocalTensor<float>(TPosition::B1, bBase + 4ULL * bPanelSize * sizeof(float), bPanelSize);
            bAi11 = LocalTensor<float>(TPosition::B1, bBase + 5ULL * bPanelSize * sizeof(float), bPanelSize);
            bBr11 = LocalTensor<float>(TPosition::B1, bBase + 6ULL * bPanelSize * sizeof(float), bPanelSize);
            bBi11 = LocalTensor<float>(TPosition::B1, bBase + 7ULL * bPanelSize * sizeof(float), bPanelSize);
            bAr12 = LocalTensor<float>(TPosition::B1, bBase + 8ULL * bPanelSize * sizeof(float), bPanelSize);
            bAi12 = LocalTensor<float>(TPosition::B1, bBase + 9ULL * bPanelSize * sizeof(float), bPanelSize);
            bBr12 = LocalTensor<float>(TPosition::B1, bBase + 10ULL * bPanelSize * sizeof(float), bPanelSize);
            bBi12 = LocalTensor<float>(TPosition::B1, bBase + 11ULL * bPanelSize * sizeof(float), bPanelSize);
            a2 = LocalTensor<float>(TPosition::A2, 0U, aSize);
            a2Alt = LocalTensor<float>(TPosition::A2, static_cast<uint64_t>(aSize) * sizeof(float), aSize);
            b2 = LocalTensor<float>(TPosition::B2, 0U, bSize);
            b2Alt = LocalTensor<float>(TPosition::B2, static_cast<uint64_t>(bSize) * sizeof(float), bSize);
            realC1 = LocalTensor<float>(TPosition::CO1, 0U, cSize);
            imagC1 = LocalTensor<float>(TPosition::CO1, static_cast<uint64_t>(cSize) * sizeof(float), cSize);
            InitCopyAndMmadParams(n);
        }

        __aicore__ inline void InitCopyAndMmadParams(uint32_t n)
        {
            aCopy.ndNum = 1U;
            aCopy.nValue = panelCapacity;
            aCopy.dValue = M;
            aCopy.srcDValue = n;
            aCopy.dstNzC0Stride = panelCapacity;
            aCopy.dstNzNStride = 1U;
            bCopyParams.ndNum = 1U;
            bCopyParams.nValue = panelCapacity;
            bCopyParams.dValue = N;
            bCopyParams.srcDValue = n;
            bCopyParams.dstNzC0Stride = panelCapacity;
            bCopyParams.dstNzNStride = 1U;
            Cher2kInitLoadData3D(aLoad, panelCapacity, M, M, true);
            Cher2kInitLoadData3D(bLoad, panelCapacity, N, N, true);
            first.m = M;
            first.n = N;
            first.k = panelCapacity;
            first.cmatrixInitVal = true;
            first.cmatrixSource = false;
            first.kDirectionAlign = false;
            accum = first;
            accum.cmatrixInitVal = false;
        }
    };

    __aicore__ inline void ConfigureDirectOpcTile(DirectOpcState& state, uint32_t tileHeight, uint32_t tileWidth)
    {
        state.aCopy.dValue = tileHeight;
        state.aLoad.channelSize = tileHeight;
        state.aLoad.kExtension = tileHeight;
        state.first.m = tileHeight;
        state.accum.m = tileHeight;
        state.bCopyParams.dValue = tileWidth;
        state.bLoad.channelSize = tileWidth;
        state.bLoad.kExtension = tileWidth;
        state.first.n = tileWidth;
        state.accum.n = tileWidth;
    }

    __aicore__ inline void IssueDirectOpcTerm(
        DirectOpcState& state, uint32_t& termIndex, LocalTensor<float>& output, LocalTensor<float>& left,
        LocalTensor<float>& right, bool initialize)
    {
        const uint16_t slot = static_cast<uint16_t>(termIndex & 1U);
        LocalTensor<float>& a2Slot = slot == 0U ? state.a2 : state.a2Alt;
        LocalTensor<float>& b2Slot = slot == 0U ? state.b2 : state.b2Alt;
        WaitFlag<HardEvent::M_MTE1>(slot);
        LoadData(b2Slot, right, state.bLoad);
        LoadData(a2Slot, left, state.aLoad);
        SetFlag<HardEvent::MTE1_M>(slot);
        WaitFlag<HardEvent::MTE1_M>(slot);
        Mmad(output, a2Slot, b2Slot, initialize ? state.first : state.accum);
        SetFlag<HardEvent::M_MTE1>(slot);
        ++termIndex;
    }

    __aicore__ inline void ComputeDirectOpcTerms(
        DirectOpcState& state, LocalTensor<float>& bAr1, LocalTensor<float>& bAi1, LocalTensor<float>& bBr1,
        LocalTensor<float>& bBi1, uint32_t kBase, uint32_t& termIndex)
    {
        IssueDirectOpcTerm(state, termIndex, state.realC1, state.aAr1, bBr1, kBase == 0U);
        IssueDirectOpcTerm(state, termIndex, state.realC1, state.aAi1, bBi1, false);
        IssueDirectOpcTerm(state, termIndex, state.realC1, state.aBr1, bAr1, false);
        IssueDirectOpcTerm(state, termIndex, state.realC1, state.aBi1, bAi1, false);
        IssueDirectOpcTerm(state, termIndex, state.imagC1, state.aAi1, bBr1, kBase == 0U);
        IssueDirectOpcTerm(state, termIndex, state.imagC1, state.aNegAr1, bBi1, false);
        IssueDirectOpcTerm(state, termIndex, state.imagC1, state.aBi1, bAr1, false);
        IssueDirectOpcTerm(state, termIndex, state.imagC1, state.aNegBr1, bAi1, false);
    }

    __aicore__ inline void StoreDirectOpcTile(
        DirectOpcState& state, uint32_t mBase, uint32_t nBase, uint32_t tileHeight, uint32_t tileWidth)
    {
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 store{};
        store.mSize = tileHeight;
        store.nSize = tileWidth;
        store.srcStride = tileHeight;
        store.dstStride = n_;
        store.ndNum = 1U;
        const uint64_t outputOffset = static_cast<uint64_t>(mBase) * n_ + nBase;
        Fixpipe(real_[outputOffset], state.realC1, store);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::M_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
        Fixpipe(imag_[outputOffset], state.imagC1, store);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }

    struct DirectOpcCache {
        uint32_t mBase = UINT32_MAX;
        uint32_t aKBase = UINT32_MAX;
        uint32_t bBase[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
        uint32_t bWidth[3] = {0U, 0U, 0U};
        uint32_t bKBase[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
        uint32_t nextBSlot = 0U;
    };

    struct DirectOpcSchedule {
        uint64_t taskBegin = 0ULL;
        uint64_t taskEnd = 0ULL;
        uint64_t rowTaskBase = 0ULL;
        uint32_t mTiles = 0U;
        uint32_t nTiles = 0U;
        uint32_t mTile = 0U;
        uint32_t localIteration = 0U;
        bool upper = false;
    };

    struct DirectOpcWork {
        uint32_t mBase = 0U;
        uint32_t nBase = 0U;
        uint32_t tileHeight = 0U;
        uint32_t tileWidth = 0U;
        uint16_t slot = 0U;
        bool mergeHalves = false;
    };

    __aicore__ inline void LoadDirectOpcA(DirectOpcState& state, DirectOpcCache& cache, uint32_t mBase, uint32_t kBase)
    {
        if (mBase == cache.mBase && kBase == cache.aKBase) {
            return;
        }
        const uint64_t rowOffset = static_cast<uint64_t>(kBase) * n_ + mBase;
        DataCopy(state.aAr1, leftAr_[rowOffset], state.aCopy);
        DataCopy(state.aAi1, leftAi_[rowOffset], state.aCopy);
        DataCopy(state.aBr1, leftBr_[rowOffset], state.aCopy);
        DataCopy(state.aBi1, leftBi_[rowOffset], state.aCopy);
        DataCopy(state.aNegAr1, leftNegAr_[rowOffset], state.aCopy);
        DataCopy(state.aNegBr1, leftNegBr_[rowOffset], state.aCopy);
        cache.mBase = mBase;
        cache.aKBase = kBase;
    }

    __aicore__ inline uint32_t SelectDirectOpcBSlot(
        DirectOpcState& state, DirectOpcCache& cache, uint32_t nBase, uint32_t tileWidth, uint32_t kBase,
        bool& resident)
    {
        for (uint32_t slot = 0U; slot < state.bCacheSlots; ++slot) {
            if (cache.bBase[slot] == nBase && cache.bWidth[slot] == tileWidth && cache.bKBase[slot] == kBase) {
                resident = true;
                return slot;
            }
        }
        resident = false;
        const uint32_t slot = cache.nextBSlot;
        cache.nextBSlot = (cache.nextBSlot + 1U) % state.bCacheSlots;
        return slot;
    }

    __aicore__ inline void LoadDirectOpcB(
        DirectOpcState& state, DirectOpcCache& cache, uint32_t slot, uint32_t nBase, uint32_t tileWidth, uint32_t kBase,
        LocalTensor<float>& bAr1, LocalTensor<float>& bAi1, LocalTensor<float>& bBr1, LocalTensor<float>& bBi1)
    {
        const uint64_t colOffset = static_cast<uint64_t>(kBase) * n_ + nBase;
        DataCopy(bAr1, rightAr_[colOffset], state.bCopyParams);
        DataCopy(bAi1, rightAi_[colOffset], state.bCopyParams);
        DataCopy(bBr1, rightBr_[colOffset], state.bCopyParams);
        DataCopy(bBi1, rightBi_[colOffset], state.bCopyParams);
        cache.bBase[slot] = nBase;
        cache.bWidth[slot] = tileWidth;
        cache.bKBase[slot] = kBase;
    }

    __aicore__ inline void ProcessDirectOpcPanel(
        DirectOpcState& state, DirectOpcCache& cache, uint32_t mBase, uint32_t nBase, uint32_t tileWidth,
        uint32_t kBase, uint32_t& termIndex)
    {
        const uint32_t panelK = min(DirectOpcState::maxK, computeK_ - kBase);
        state.aCopy.nValue = panelK;
        state.aCopy.dstNzC0Stride = panelK;
        state.bCopyParams.nValue = panelK;
        state.bCopyParams.dstNzC0Stride = panelK;
        state.aLoad.l1W = panelK;
        state.aLoad.mExtension = panelK;
        state.bLoad.l1W = panelK;
        state.bLoad.mExtension = panelK;
        state.first.k = panelK;
        state.accum.k = panelK;
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        LoadDirectOpcA(state, cache, mBase, kBase);
        bool resident = false;
        const uint32_t slot = SelectDirectOpcBSlot(state, cache, nBase, tileWidth, kBase, resident);
        LocalTensor<float>& bAr1 = slot == 0U ? state.bAr10 : (slot == 1U ? state.bAr11 : state.bAr12);
        LocalTensor<float>& bAi1 = slot == 0U ? state.bAi10 : (slot == 1U ? state.bAi11 : state.bAi12);
        LocalTensor<float>& bBr1 = slot == 0U ? state.bBr10 : (slot == 1U ? state.bBr11 : state.bBr12);
        LocalTensor<float>& bBi1 = slot == 0U ? state.bBi10 : (slot == 1U ? state.bBi11 : state.bBi12);
        if (!resident) {
            LoadDirectOpcB(state, cache, slot, nBase, tileWidth, kBase, bAr1, bAi1, bBr1, bBi1);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        ComputeDirectOpcTerms(state, bAr1, bAi1, bBr1, bBi1, kBase, termIndex);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline DirectOpcSchedule BuildDirectOpcSchedule(uint32_t taskIndex, uint32_t taskStride)
    {
        DirectOpcSchedule schedule;
        schedule.mTiles = (validN_ + DirectOpcState::M - 1U) / DirectOpcState::M;
        schedule.nTiles = (validN_ + DirectOpcState::N - 1U) / DirectOpcState::N;
        schedule.upper = uplo_ == ACLBLAS_UPPER;
        const uint64_t totalTasks = (static_cast<uint64_t>(schedule.mTiles) * (schedule.mTiles + 1U)) / 2ULL;
        const uint64_t totalWorkUnits = MIX_CONSUMER ? 2ULL * totalTasks : totalTasks;
        schedule.taskBegin = (static_cast<uint64_t>(taskIndex) * totalWorkUnits) / (taskStride == 0U ? 1U : taskStride);
        schedule.taskEnd =
            (static_cast<uint64_t>(taskIndex + 1U) * totalWorkUnits) / (taskStride == 0U ? 1U : taskStride);
        const uint64_t firstTask = MIX_CONSUMER ? schedule.taskBegin / 2ULL : schedule.taskBegin;
        while (schedule.mTile < schedule.mTiles) {
            const uint32_t nBegin = schedule.upper ? 0U : schedule.mTile;
            const uint32_t nEnd = schedule.upper ? schedule.mTile + 1U : schedule.nTiles;
            const uint32_t rowTasks = nEnd - nBegin;
            if (firstTask < schedule.rowTaskBase + rowTasks) {
                break;
            }
            schedule.rowTaskBase += rowTasks;
            ++schedule.mTile;
        }
        return schedule;
    }

    __aicore__ inline void BeginDirectOpcEvents()
    {
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetHF32Mode(HF32Mode::DISABLE);
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline DirectOpcWork ResolveDirectOpcWork(DirectOpcSchedule& schedule, uint64_t workUnit)
    {
        DirectOpcWork work;
        const uint64_t task = MIX_CONSUMER ? workUnit / 2ULL : workUnit;
        const uint32_t halfPart = MIX_CONSUMER ? static_cast<uint32_t>(workUnit & 1ULL) : 0U;
        work.mergeHalves = !MIX_CONSUMER || (halfPart == 0U && workUnit + 1ULL < schedule.taskEnd);
        work.slot = static_cast<uint16_t>(schedule.localIteration % CHER2K_DIRECT_MIX_FLAG_COUNT);
        const uint32_t nTile =
            ResolveTriangularTile(task, schedule.upper, schedule.nTiles, schedule.mTile, schedule.rowTaskBase);
        work.mBase = schedule.mTile * DirectOpcState::M;
        work.nBase = nTile * DirectOpcState::N + (work.mergeHalves ? 0U : halfPart * (DirectOpcState::N / 2U));
        work.tileHeight = min(DirectOpcState::M, n_ - work.mBase);
        work.tileWidth = work.nBase < validN_ ?
                             min(work.mergeHalves ? DirectOpcState::N : DirectOpcState::N / 2U, n_ - work.nBase) :
                             0U;
        return work;
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline void WaitDirectOpcSlot(const DirectOpcSchedule& schedule, uint16_t slot)
    {
        if constexpr (MIX_CONSUMER) {
            if (schedule.localIteration >= CHER2K_DIRECT_MIX_FLAG_COUNT) {
                CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(CHER2K_DIRECT_MIX_ACK_BASE + slot));
            }
        }
    }

    __aicore__ inline void ComputeDirectOpcTile(DirectOpcState& state, DirectOpcCache& cache, const DirectOpcWork& work)
    {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        uint32_t termIndex = 0U;
        for (uint32_t kBase = 0U; kBase < computeK_; kBase += DirectOpcState::maxK) {
            ProcessDirectOpcPanel(state, cache, work.mBase, work.nBase, work.tileWidth, kBase, termIndex);
        }
        StoreDirectOpcTile(state, work.mBase, work.nBase, work.tileHeight, work.tileWidth);
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline uint64_t ProcessDirectOpcWorkUnit(
        DirectOpcState& state, DirectOpcCache& cache, DirectOpcSchedule& schedule, uint64_t workUnit)
    {
        const DirectOpcWork work = ResolveDirectOpcWork<MIX_CONSUMER>(schedule, workUnit);
        WaitDirectOpcSlot<MIX_CONSUMER>(schedule, work.slot);
        ConfigureDirectOpcTile(state, work.tileHeight, work.tileWidth);
        if (work.tileWidth != 0U) {
            ComputeDirectOpcTile(state, cache, work);
        }
        if constexpr (MIX_CONSUMER) {
            CrossCoreSetFlag<2, PIPE_FIX>(work.slot);
        }
        ++schedule.localIteration;
        return (MIX_CONSUMER && work.mergeHalves) ? 2ULL : 1ULL;
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline void EndDirectOpcEvents(const DirectOpcSchedule& schedule)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        SetHF32Mode(HF32Mode::DISABLE);
        if constexpr (MIX_CONSUMER) {
            const uint32_t drainCount = min(schedule.localIteration, CHER2K_DIRECT_MIX_FLAG_COUNT);
            for (uint32_t drain = 0U; drain < drainCount; ++drain) {
                const uint16_t slot =
                    static_cast<uint16_t>((schedule.localIteration - 1U - drain) % CHER2K_DIRECT_MIX_FLAG_COUNT);
                CrossCoreWaitFlag<2, PIPE_MTE2>(static_cast<uint16_t>(CHER2K_DIRECT_MIX_ACK_BASE + slot));
            }
        }
    }

    template <bool MIX_CONSUMER>
    __aicore__ inline void ProcessImpl(uint32_t taskIndex, uint32_t taskStride)
    {
        DirectOpcState state;
        state.Init(n_, computeK_);
        DirectOpcSchedule schedule = BuildDirectOpcSchedule<MIX_CONSUMER>(taskIndex, taskStride);
        DirectOpcCache cache;
        BeginDirectOpcEvents();
        for (uint64_t workUnit = schedule.taskBegin; workUnit < schedule.taskEnd;) {
            workUnit += ProcessDirectOpcWorkUnit<MIX_CONSUMER>(state, cache, schedule, workUnit);
        }
        EndDirectOpcEvents<MIX_CONSUMER>(schedule);
    }

    GlobalTensor<float> leftAr_;
    GlobalTensor<float> leftAi_;
    GlobalTensor<float> leftBr_;
    GlobalTensor<float> leftBi_;
    GlobalTensor<float> rightAr_;
    GlobalTensor<float> rightAi_;
    GlobalTensor<float> rightBr_;
    GlobalTensor<float> rightBi_;
    GlobalTensor<float> leftNegAr_;
    GlobalTensor<float> leftNegBr_;
    GlobalTensor<float> real_;
    GlobalTensor<float> imag_;
    uint32_t n_ = 0U;
    uint32_t validN_ = 0U;
    uint32_t k_ = 0U;
    uint32_t computeK_ = CHER2K_CUBE_K;
    uint32_t uplo_ = ACLBLAS_UPPER;
};

// Unit-alpha/zero-beta 3M epilogue used by the mixed-kernel probe.  It keeps
// the same workspace algebra as the production postprocess but omits scalar
// alpha/beta handling and C reads.
// Entire off-diagonal tiles are selected. Diagonal tiles use the exact
// prefix/suffix helper below, with aligned UB scratch for LOWER suffixes.
__aicore__ inline bool Cher2kStoreOffDiagonalUnitTile(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAos, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    if (rowBase == colBase) {
        return false;
    }
    constexpr uint32_t tileSize = 64U;
    const uint32_t blockLen = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
    const DataCopyExtParams outputCopy{
        static_cast<uint16_t>(validCols), blockLen,
        static_cast<uint32_t>((tileSize * 2U * sizeof(float) - alignedBlockLen) / 32U),
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
    const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    DataCopyPad(cG[cOffset], cAos, outputCopy);
    return true;
}

// The caller provides dead planar/index buffers and fences their reuse after
// this store. No DMA destination includes the unselected triangle.
__aicore__ inline void Cher2kStorePlanarUnitTile(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAos, LocalTensor<float>& scratch,
    LocalTensor<float>& indexBuffer, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    if (Cher2kStoreOffDiagonalUnitTile(cG, tiling, cAos, rowBase, colBase, validRows, validCols)) {
        return;
    }
    triangle_store::StoreComplexDiagonal<64U, true>(
        cG, cAos, scratch, indexBuffer.ReinterpretCast<int32_t>(), tiling.ldc, rowBase, colBase, validRows, validCols,
        tiling.uplo == ACLBLAS_UPPER, static_cast<event_t>(0));
}

__aicore__ inline void Cher2kPostprocessApplyBeta(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAos, LocalTensor<float>& realIj,
    LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi, uint32_t tileSize,
    uint32_t tileElements, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, float betaValue)
{
    if (betaValue == 0.0f) {
        return;
    }
    const uint32_t blockLen = validRows * 2U * sizeof(float);
    const uint32_t alignedLen = (blockLen + 31U) & ~31U;
    const DataCopyExtParams copy{
        static_cast<uint16_t>(validCols), blockLen,
        static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)),
        static_cast<uint32_t>((tileSize * 2U * sizeof(float) - alignedLen) / 32U), 0U};
    const DataCopyPadExtParams<float> pad{
        alignedLen != blockLen, 0U, static_cast<uint8_t>((alignedLen - blockLen) / sizeof(float)), 0.0f};
    const uint64_t offset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
    if (rowBase == colBase) {
        triangle_store::LoadComplexTile(
            cAos, cG, tileSize, tiling.ldc, rowBase, colBase, validRows, validCols, tiling.uplo == ACLBLAS_UPPER,
            static_cast<event_t>(0));
    } else {
        DataCopyPad(cAos, cG[offset], copy, pad);
    }
    PipeBarrier<PIPE_ALL>();
    uint64_t reserved = 0ULL;
    GatherMask<float>(
        realJi, cAos, 1U, false, 0U, {1U, static_cast<uint16_t>(tileElements * 2U / 64U), 8U, 8U}, reserved);
    GatherMask<float>(
        imagIj, cAos, 2U, false, 0U, {1U, static_cast<uint16_t>(tileElements * 2U / 64U), 8U, 8U}, reserved);
    PipeBarrier<PIPE_ALL>();
    Muls(realJi, realJi, betaValue, tileElements);
    Muls(imagIj, imagIj, betaValue, tileElements);
    PipeBarrier<PIPE_ALL>();
    Add(realIj, realIj, realJi, tileElements);
    Add(imagJi, imagJi, imagIj, tileElements);
}

__aicore__ inline void Cher2kPublishPlanarTile(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& imagJi, LocalTensor<float>& planar,
    LocalTensor<float>& cAos, LocalTensor<uint32_t>& interleaveOffset, uint32_t tileSize, uint32_t tileElements,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, bool diagonal, bool waitBeforeDiagonal)
{
    if (diagonal) {
        if (waitBeforeDiagonal) {
            PipeBarrier<PIPE_ALL>();
        }
        for (uint32_t index = 0U; index < min(validRows, validCols); ++index)
            imagJi.SetValue(index * tileSize + index, 0.0f);
    }
    PipeBarrier<PIPE_ALL>();
    Adds(planar[tileElements], imagJi, 0.0f, tileElements);
    PipeBarrier<PIPE_ALL>();
    Gather(cAos, planar, interleaveOffset, 0U, tileElements * 2U);
    PipeBarrier<PIPE_ALL>();
    Cher2kStorePlanarUnitTile(cG, tiling, cAos, planar, imagJi, rowBase, colBase, validRows, validCols);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kPostprocessStore(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& imagJi, LocalTensor<float>& planar,
    LocalTensor<float>& cAos, LocalTensor<uint32_t>& interleaveOffset, uint32_t tileSize, uint32_t tileElements,
    uint32_t rowTile, uint32_t colTile, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    Cher2kPublishPlanarTile(
        cG, tiling, imagJi, planar, cAos, interleaveOffset, tileSize, tileElements, rowBase, colBase, validRows,
        validCols, rowTile == colTile, true);
}

__aicore__ inline void Cher2kPostprocessTransposeTile(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffset, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& raw,
    LocalTensor<float>& realIj, LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi,
    LocalTensor<float>& temp, LocalTensor<float>& planar, LocalTensor<float>& cAos, uint64_t rrBase, uint64_t riBase,
    uint64_t irBase, uint64_t ijOffset, uint64_t jiOffset, uint32_t rowTile, uint32_t colTile, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    constexpr uint32_t tileSize = 64U;
    constexpr uint32_t tileElements = tileSize * tileSize;
    const DataCopyExtParams copy{
        static_cast<uint16_t>(tileSize), static_cast<uint32_t>(tileSize * sizeof(float)),
        static_cast<uint32_t>((tiling.nAligned - tileSize) * sizeof(float)), 0U, 0U};
    CopyWorkspaceTile<false>(realIj, ws, riBase + jiOffset, copy);
    CopyWorkspaceTile<false>(imagIj, ws, irBase + jiOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Add(temp, realIj, imagIj, tileElements);
    PipeBarrier<PIPE_V>();
    Sub(realIj, realIj, imagIj, tileElements);
    CopyWorkspaceTile<false>(raw, ws, rrBase + jiOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Sub(imagIj, raw, temp, tileElements);
    CopyWorkspaceTile<false>(realJi, ws, riBase + ijOffset, copy);
    CopyWorkspaceTile<false>(imagJi, ws, irBase + ijOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Add(temp, realJi, imagJi, tileElements);
    PipeBarrier<PIPE_V>();
    Sub(realJi, realJi, imagJi, tileElements);
    CopyWorkspaceTile<false>(raw, ws, rrBase + ijOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Sub(imagJi, raw, temp, tileElements);
    Add(realIj, realIj, realJi, tileElements);
    Sub(imagJi, imagIj, imagJi, tileElements);
    Cher2kPublishPlanarTile(
        cG, tiling, imagJi, planar, cAos, interleaveOffset, tileSize, tileElements, rowBase, colBase, validRows,
        validCols, rowTile == colTile, false);
    (void)transposeOffset;
}

__aicore__ inline void Cher2kPostprocessLoadPlanes(
    GlobalTensor<float>& ws, LocalTensor<uint32_t>& transposeOffset, LocalTensor<float>& raw,
    LocalTensor<float>& realIj, LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi,
    LocalTensor<float>& temp, uint64_t rrBase, uint64_t riBase, uint64_t irBase, uint64_t ijOffset, uint64_t jiOffset,
    const DataCopyExtParams& copy, uint32_t tileElements)
{
    CopyWorkspaceTile<false>(raw, ws, riBase + ijOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Gather(realIj, raw, transposeOffset, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    CopyWorkspaceTile<false>(raw, ws, irBase + ijOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Gather(imagIj, raw, transposeOffset, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    Add(temp, realIj, imagIj, tileElements);
    PipeBarrier<PIPE_V>();
    Sub(realIj, realIj, imagIj, tileElements);
    CopyWorkspaceTile<false>(raw, ws, rrBase + ijOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Gather(imagIj, raw, transposeOffset, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    Sub(imagIj, imagIj, temp, tileElements);
    CopyWorkspaceTile<false>(realJi, ws, riBase + jiOffset, copy);
    CopyWorkspaceTile<false>(imagJi, ws, irBase + jiOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Add(temp, realJi, imagJi, tileElements);
    PipeBarrier<PIPE_V>();
    Sub(realJi, realJi, imagJi, tileElements);
    PipeBarrier<PIPE_ALL>();
    CopyWorkspaceTile<false>(imagJi, ws, rrBase + jiOffset, copy);
    PipeBarrier<PIPE_ALL>();
    Sub(imagJi, imagJi, temp, tileElements);
}

__aicore__ inline void Cher2kPostprocessCombine(
    LocalTensor<float>& realIj, LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi,
    LocalTensor<float>& raw, LocalTensor<float>& temp, uint32_t tileElements, float alphaReal, float alphaImag,
    float betaValue)
{
    if (alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f) {
        Add(realIj, realIj, realJi, tileElements);
        Sub(imagJi, imagIj, imagJi, tileElements);
        return;
    }
    Add(temp, realIj, realJi, tileElements);
    Sub(realJi, realIj, realJi, tileElements);
    Add(raw, imagIj, imagJi, tileElements);
    Sub(imagJi, imagIj, imagJi, tileElements);
    PipeBarrier<PIPE_V>();
    Muls(realIj, temp, alphaReal, tileElements);
    Muls(raw, raw, alphaImag, tileElements);
    Muls(imagIj, imagJi, alphaReal, tileElements);
    Muls(temp, realJi, alphaImag, tileElements);
    PipeBarrier<PIPE_V>();
    Sub(realIj, realIj, raw, tileElements);
    Add(imagJi, imagIj, temp, tileElements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kPostprocessUnitTile(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffset, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& raw,
    LocalTensor<float>& realIj, LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi,
    LocalTensor<float>& temp, LocalTensor<float>& planar, LocalTensor<float>& cAos, uint32_t rowTile, uint32_t colTile,
    float alphaReal = 1.0f, float alphaImag = 0.0f, float betaValue = 0.0f)
{
    constexpr uint32_t tileSize = 64U;
    constexpr uint32_t tileElements = tileSize * tileSize;
    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
    const bool sgemm3Batch =
        tiling.sg3MmadPath != 0U || (tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C);
    const uint64_t sgemm3OutputBase = 6ULL * nk;
    const uint64_t rrBase = sgemm3Batch ? sgemm3OutputBase + 2ULL * nn : OffsetRr(tiling.nAligned, tiling.kAligned);
    const uint64_t riBase = sgemm3Batch ? sgemm3OutputBase : OffsetRi(tiling.nAligned, tiling.kAligned);
    const uint64_t irBase = sgemm3Batch ? sgemm3OutputBase + nn : OffsetIr(tiling.nAligned, tiling.kAligned);
    const uint32_t rowBase = rowTile * tileSize;
    const uint32_t colBase = colTile * tileSize;
    if ((tiling.uplo == ACLBLAS_UPPER && rowTile > colTile) || (tiling.uplo == ACLBLAS_LOWER && rowTile < colTile)) {
        return;
    }
    if (rowBase >= tiling.n || colBase >= tiling.n) {
        return;
    }
    const uint32_t validRows = min(tileSize, tiling.n - rowBase);
    const uint32_t validCols = min(tileSize, tiling.n - colBase);
    const uint64_t ijOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
    const uint64_t jiOffset = static_cast<uint64_t>(colBase) * tiling.nAligned + rowBase;
    const DataCopyExtParams copyParams{
        static_cast<uint16_t>(tileSize), static_cast<uint32_t>(tileSize * sizeof(float)),
        static_cast<uint32_t>((tiling.nAligned - tileSize) * sizeof(float)), 0U, 0U};
    if (tiling.outputTransposePath != 0U) {
        Cher2kPostprocessTransposeTile(
            cG, ws, tiling, transposeOffset, interleaveOffset, raw, realIj, imagIj, realJi, imagJi, temp, planar, cAos,
            rrBase, riBase, irBase, ijOffset, jiOffset, rowTile, colTile, rowBase, colBase, validRows, validCols);
        return;
    }
    Cher2kPostprocessLoadPlanes(
        ws, transposeOffset, raw, realIj, imagIj, realJi, imagJi, temp, rrBase, riBase, irBase, ijOffset, jiOffset,
        copyParams, tileElements);
    Cher2kPostprocessCombine(realIj, imagIj, realJi, imagJi, raw, temp, tileElements, alphaReal, alphaImag, betaValue);
    Cher2kPostprocessApplyBeta(
        cG, tiling, cAos, realIj, imagIj, realJi, imagJi, tileSize, tileElements, rowBase, colBase, validRows,
        validCols, betaValue);
    Cher2kPostprocessStore(
        cG, tiling, imagJi, planar, cAos, interleaveOffset, tileSize, tileElements, rowTile, colTile, rowBase, colBase,
        validRows, validCols);
}
__aicore__ inline void Cher2kRunDirectOpcCube(__gm__ float* base, const Cher2kTilingData& tiling)
{
    Cher2kKernelCube direct;
    direct.Init(base, tiling.nAligned, tiling.n, tiling.kAligned, tiling.k, tiling.uplo);
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    direct.Process(static_cast<uint32_t>(GetBlockIdx()), blockNum);
}

__aicore__ inline void Cher2kDispatchConfiguredCube(
    Cher2kKernelLayout& mm, const Cher2kTilingData& tiling, bool directOpc, bool directOutput, uint32_t blockNum)
{
    const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
    if (directOpc && !directOutput) {
        mm.Process(block, blockNum, true);
    } else if (tiling.directHermitianPath != 0U) {
        mm.ProcessDirectHermitian(block, blockNum);
    } else {
        mm.Process(block, blockNum, tiling.useThreeM != 0U);
    }
}

__aicore__ inline void Cher2kRunConfiguredCube(
    __gm__ float* base, const Cher2kTilingData& tiling, bool directOpc, uint64_t nk, uint64_t nn, bool directOutput)
{
    Cher2kKernelLayout mm;
    mm.Init(
        base + (directOpc ? 0ULL : OffsetAr(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? nk : OffsetAi(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 3ULL * nk : OffsetBr(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 4ULL * nk : OffsetBi(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 2ULL * nk : OffsetSumA(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 5ULL * nk : OffsetSumB(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 6ULL * nk + 2ULL * nn : OffsetRr(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 6ULL * nk + 2ULL * nn : OffsetIi(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 6ULL * nk : OffsetRi(tiling.nAligned, tiling.kAligned)),
        base + (directOpc ? 6ULL * nk + nn : OffsetIr(tiling.nAligned, tiling.kAligned)), tiling.nAligned,
        tiling.kAligned, directOpc ? false : true, directOpc && directOutput, tiling.useThreeM != 0U, tiling.uplo,
        tiling.outputTransposePath != 0U);
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    Cher2kDispatchConfiguredCube(mm, tiling, directOpc, directOutput, blockNum);

    GlobalTensor<float> ws;
    ws.SetGlobalBuffer(
        base, directOpc ? Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned) :
                          WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, tiling.useThreeM != 0U));
    if constexpr (!CHER2K_SKIP_WORKSPACE_CACHE_CLEAN_EXPERIMENT) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kRunStandardCube(__gm__ float* base, const Cher2kTilingData& tiling)
{
    const bool directOpc = CHER2K_DIRECT_OPC_CUBE_LAYOUT_EXPERIMENT && tiling.directHermitianPath != 0U &&
                           tiling.trans == static_cast<uint32_t>(ACLBLAS_OP_C);
    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
    const bool directOutput = tiling.directOutputPath != 0U;
    if (directOpc && directOutput) {
        Cher2kRunDirectOpcCube(base, tiling);
        return;
    }
    Cher2kRunConfiguredCube(base, tiling, directOpc, nk, nn, directOutput);
}

__aicore__ inline bool Cher2kMatmulEligible(const Cher2kTilingData& tiling)
{
    return tiling.n == tiling.nAligned && tiling.k == tiling.kAligned && tiling.kAligned == 256U &&
           tiling.nAligned == 256U && tiling.nAligned >= CHER2K_CUBE_M && (tiling.nAligned % CHER2K_CUBE_M) == 0U &&
           (tiling.nAligned % CHER2K_CUBE_N) == 0U;
}

extern "C" __global__ __aicore__ void cher2k_cube_kernel(
    GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }
    AscendC::InitSocState();

    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspace);
    // MatmulImpl's static contract is a full, aligned 256-row band.  Do not
    // feed padded tails (for example n=65 -> nAligned=128) into that route:
    // its band scheduler would emit no work and silently lose the tail.
    const bool matmulEligible = Cher2kMatmulEligible(tiling);
    if constexpr (CHER2K_ENABLE_MATMUL_EXPERIMENT) {
        if (matmulEligible) {
            TPipe pipe;
            Cher2kRunCube<Cher2kNMatmul>(base, tiling, pipe);
            GlobalTensor<float> ws;
            ws.SetGlobalBuffer(base, WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, tiling.useThreeM != 0U));
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
            PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    Cher2kRunStandardCube(base, tiling);
}
