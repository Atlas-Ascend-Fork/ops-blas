/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ inline void Cher2kSmallCubeStore(
    GlobalTensor<float>& output, LocalTensor<float>& c1, const Cher2kTilingData& tiling, uint32_t mBase, uint32_t nBase)
{
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    FixpipeParamsV220 fix{};
    fix.mSize = CHER2K_SMALL_CUBE_M;
    fix.nSize = CHER2K_SMALL_CUBE_N;
    fix.srcStride = CHER2K_SMALL_CUBE_M;
    fix.dstStride = tiling.nAligned;
    fix.ndNum = 1;
    fix.srcNdStride = 0;
    fix.dstNdStride = 0;
    Fixpipe(output[static_cast<uint64_t>(mBase) * tiling.nAligned + nBase], c1, fix);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}

__aicore__ inline void Cher2kLoadAndMmad(
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, const LoadData3DParamsV2<float>& aLoad, const LoadData3DParamsV2<float>& bLoad,
    const MmadParams& mmad)
{
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    LoadData(a2, a1, aLoad);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    LoadData(b2, b1, bLoad);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
    Mmad(c1, a2, b2, mmad);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
}

__aicore__ inline void Cher2kSmallCubeReduceK(
    const Cher2kTilingData& tiling, GlobalTensor<float>& left, GlobalTensor<float>& right, uint32_t mBase,
    uint32_t nBase, LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, const Nd2NzParams& aCopy, const Nd2NzParams& bNzParams,
    const LoadData3DParamsV2<float>& aLoad, const LoadData3DParamsV2<float>& bLoad, const MmadParams& first,
    const MmadParams& accum)
{
    for (uint32_t kBase = 0U; kBase < tiling.kAligned; kBase += CHER2K_SMALL_CUBE_K) {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        const uint64_t aOffset = tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C ?
                                     static_cast<uint64_t>(mBase) * tiling.kAligned + kBase :
                                     static_cast<uint64_t>(kBase) * tiling.nAligned + mBase;
        DataCopy(a1, left[aOffset], aCopy);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        DataCopy(b1, right[static_cast<uint64_t>(kBase) * tiling.nAligned + nBase], bNzParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        Cher2kLoadAndMmad(a1, b1, a2, b2, c1, aLoad, bLoad, kBase == 0U ? first : accum);
    }
}

__aicore__ inline void Cher2kSmallCubeRunProduct(
    __gm__ float* base, const Cher2kTilingData& tiling, const uint64_t* leftBase, const uint64_t* rightBase,
    const uint64_t* outBase, uint32_t product, uint64_t nk, uint64_t nn, uint32_t mBase, uint32_t nBase,
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, const Nd2NzParams& aCopy, const Nd2NzParams& bNzParams,
    const LoadData3DParamsV2<float>& aLoad, const LoadData3DParamsV2<float>& bLoad, const MmadParams& first,
    const MmadParams& accum)
{
    GlobalTensor<float> left;
    left.SetGlobalBuffer(base + leftBase[product], nk);
    GlobalTensor<float> right;
    right.SetGlobalBuffer(base + rightBase[product], nk);
    GlobalTensor<float> output;
    output.SetGlobalBuffer(base + outBase[product], nn);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    Cher2kSmallCubeReduceK(
        tiling, left, right, mBase, nBase, a1, b1, a2, b2, c1, aCopy, bNzParams, aLoad, bLoad, first, accum);
    Cher2kSmallCubeStore(output, c1, tiling, mBase, nBase);
}

__aicore__ inline void Cher2kSmallCubeProcessTiles(
    __gm__ float* base, const Cher2kTilingData& tiling, const uint64_t* leftBase, const uint64_t* rightBase,
    const uint64_t* outBase, uint64_t nk, uint64_t nn, uint32_t nTiles, uint64_t tileCount, uint32_t blockNum,
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, const Nd2NzParams& aCopy, const Nd2NzParams& bNzParams,
    const LoadData3DParamsV2<float>& aLoad, const LoadData3DParamsV2<float>& bLoad, const MmadParams& first,
    const MmadParams& accum)
{
    const uint32_t safeNTiles = nTiles == 0U ? 1U : nTiles;
    for (uint64_t tile = static_cast<uint64_t>(GetBlockIdx()); tile < tileCount; tile += blockNum) {
        const uint32_t mBase = static_cast<uint32_t>(tile / safeNTiles) * CHER2K_SMALL_CUBE_M;
        const uint32_t nBase = static_cast<uint32_t>(tile % safeNTiles) * CHER2K_SMALL_CUBE_N;
        for (uint32_t product = 0U; product < 3U; ++product) {
            Cher2kSmallCubeRunProduct(
                base, tiling, leftBase, rightBase, outBase, product, nk, nn, mBase, nBase, a1, b1, a2, b2, c1, aCopy,
                bNzParams, aLoad, bLoad, first, accum);
        }
    }
}

__aicore__ inline void Cher2kSmallCubeBuildBases(
    const Cher2kTilingData& tiling, uint64_t nk, uint64_t nn, uint64_t (&leftBase)[3], uint64_t (&rightBase)[3],
    uint64_t (&outBase)[3])
{
    const bool packed = tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C;
    leftBase[0] = packed ? 2ULL * nk : OffsetSumA(tiling.nAligned, tiling.kAligned);
    leftBase[1] = packed ? 0ULL : OffsetAr(tiling.nAligned, tiling.kAligned);
    leftBase[2] = packed ? nk : OffsetAi(tiling.nAligned, tiling.kAligned);
    rightBase[0] = packed ? 5ULL * nk : OffsetSumB(tiling.nAligned, tiling.kAligned);
    rightBase[1] = packed ? 3ULL * nk : OffsetBr(tiling.nAligned, tiling.kAligned);
    rightBase[2] = packed ? 4ULL * nk : OffsetBi(tiling.nAligned, tiling.kAligned);
    outBase[0] = packed ? 6ULL * nk + 2ULL * nn : OffsetRr(tiling.nAligned, tiling.kAligned);
    outBase[1] = packed ? 6ULL * nk : OffsetRi(tiling.nAligned, tiling.kAligned);
    outBase[2] = packed ? 6ULL * nk + nn : OffsetIr(tiling.nAligned, tiling.kAligned);
}

__aicore__ inline void Cher2kSmallCubeFinalize(__gm__ float* base, const Cher2kTilingData& tiling)
{
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    GlobalTensor<float> workspaceG;
    workspaceG.SetGlobalBuffer(base, Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned));
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(workspaceG);
    PipeBarrier<PIPE_ALL>();
}

struct Cher2kSmallCubeKernelParams {
    Nd2NzParams aCopy{};
    Nd2NzParams bCopyParams{};
    LoadData3DParamsV2<float> aLoad{};
    LoadData3DParamsV2<float> bLoad{};
    MmadParams first{};
    MmadParams accum{};

    __aicore__ inline void Init(const Cher2kTilingData& tiling, bool packed)
    {
        constexpr uint32_t M = CHER2K_SMALL_CUBE_M;
        constexpr uint32_t N = CHER2K_SMALL_CUBE_N;
        constexpr uint32_t K = CHER2K_SMALL_CUBE_K;
        aCopy.ndNum = 1;
        aCopy.nValue = packed ? M : K;
        aCopy.dValue = packed ? K : M;
        aCopy.srcNdMatrixStride = 0;
        aCopy.srcDValue = packed ? tiling.kAligned : tiling.nAligned;
        aCopy.dstNzC0Stride = packed ? M : K;
        aCopy.dstNzNStride = 1;
        aCopy.dstNzMatrixStride = 0;
        bCopyParams.ndNum = 1;
        bCopyParams.nValue = K;
        bCopyParams.dValue = N;
        bCopyParams.srcNdMatrixStride = 0;
        bCopyParams.srcDValue = tiling.nAligned;
        bCopyParams.dstNzC0Stride = K;
        bCopyParams.dstNzNStride = 1;
        bCopyParams.dstNzMatrixStride = 0;
        Cher2kInitLoadData3D(aLoad, packed ? M : K, packed ? K : M, packed ? K : M, !packed);
        Cher2kInitLoadData3D(bLoad, K, N, N, true);
        first.m = M;
        first.n = N;
        first.k = K;
        first.cmatrixInitVal = true;
        accum = first;
        accum.cmatrixInitVal = false;
    }
};

extern "C" __global__ __aicore__ void cher2k_small_cube_kernel(GM_ADDR workspace, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    constexpr uint32_t M = CHER2K_SMALL_CUBE_M;
    constexpr uint32_t N = CHER2K_SMALL_CUBE_N;
    constexpr uint32_t K = CHER2K_SMALL_CUBE_K;
    const uint32_t mTiles = tiling.nAligned / M;
    const uint32_t nTiles = tiling.nAligned / N;
    const uint64_t tileCount = static_cast<uint64_t>(mTiles) * nTiles;
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
    const bool packedOpc = tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C;
    uint64_t leftBase[3] = {};
    uint64_t rightBase[3] = {};
    uint64_t outBase[3] = {};
    Cher2kSmallCubeBuildBases(tiling, nk, nn, leftBase, rightBase, outBase);
    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspace);
    // A1 and B1 are views of the same L1 allocation on A2.  Keep the two
    // source panels disjoint: using offset 0 for both lets the second GM
    // copy overwrite A before MTE1 consumes it, which corrupts every Cube
    // product while leaving the event protocol apparently healthy.
    LocalTensor<float> a1{TPosition::A1, 0U, M * K};
    LocalTensor<float> b1{TPosition::B1, static_cast<uint64_t>(M * K) * sizeof(float), K * N};
    LocalTensor<float> a2{TPosition::A2, 0U, M * K};
    LocalTensor<float> b2{TPosition::B2, 0U, K * N};
    LocalTensor<float> c1{TPosition::CO1, 0U, M * N};
    Cher2kSmallCubeKernelParams params;
    params.Init(tiling, packedOpc);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetHF32Mode(HF32Mode::DISABLE);
    Cher2kSmallCubeProcessTiles(
        base, tiling, leftBase, rightBase, outBase, nk, nn, nTiles, tileCount, blockNum, a1, b1, a2, b2, c1,
        params.aCopy, params.bCopyParams, params.aLoad, params.bLoad, params.first, params.accum);
    Cher2kSmallCubeFinalize(base, tiling);
}

__aicore__ inline void Cher2kSmallMixConvert(
    LocalTensor<float>& real, LocalTensor<float>& imag, LocalTensor<float>& sum, LocalTensor<float>& aos, bool packB)
{
    constexpr uint32_t matrixElements = 64U * 64U;
    constexpr uint16_t repeats = matrixElements * 2U / 64U;
    uint64_t reserved = 0ULL;
    GatherMask<float>(real, aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(imag, aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
    if (packB) {
        Muls(imag, imag, -1.0f, matrixElements);
        PipeBarrier<PIPE_V>();
    }
    Add(sum, real, imag, matrixElements);
    PipeBarrier<PIPE_V>();
}

// Chemm-style tile-owned SG3 producer. One AIC block owns a complete output
// tile and serializes P1/P2/P3 through one CO1 tile. Each product follows the
// proven FIX_M -> MTE2 -> MTE1 -> M -> FIX lifecycle and fully accumulates K
// before publishing its GM plane. There are no K-panel partials, reductions,
// cross-core flags, or KFC system-workspace state.
__aicore__ inline void Cher2kSmallMixPublish(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const Cher2kTilingData& tiling)
{
    constexpr uint32_t matrixSize = 64U;
    constexpr uint32_t matrixElements = matrixSize * matrixSize;
    const bool packB = static_cast<uint32_t>(GetSubBlockIdx()) != 0U;
    TPipe pipe;
    TBuf<TPosition::VECCALC> aosBuf;
    TBuf<TPosition::VECCALC> planarBuf;
    pipe.InitBuffer(aosBuf, matrixElements * 2U * sizeof(float));
    pipe.InitBuffer(planarBuf, matrixElements * 3U * sizeof(float));
    LocalTensor<float> aos = aosBuf.Get<float>();
    LocalTensor<float> real = planarBuf.Get<float>();
    LocalTensor<float> imag = real[matrixElements];
    LocalTensor<float> sum = imag[matrixElements];
    GlobalTensor<float> input;
    GlobalTensor<float> ws;
    const uint32_t ld = packB ? tiling.ldb : tiling.lda;
    input.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(packB ? b : a), static_cast<uint64_t>(ld) * matrixSize * 2ULL);
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), WorkspaceFloatCount(matrixSize, matrixSize, true));
    const event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    const event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    const event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    const DataCopyExtParams copy{
        matrixSize, matrixSize * 2U * sizeof(float), static_cast<uint32_t>((ld - matrixSize) * 2U * sizeof(float)), 0U,
        0U};
    const DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
    DataCopyPad(aos, input, copy, pad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    Cher2kSmallMixConvert(real, imag, sum, aos, packB);
    const uint64_t realBase = packB ? OffsetBr(matrixSize, matrixSize) : OffsetAr(matrixSize, matrixSize);
    const uint64_t imagBase = packB ? OffsetBi(matrixSize, matrixSize) : OffsetAi(matrixSize, matrixSize);
    const uint64_t sumBase = packB ? OffsetSumB(matrixSize, matrixSize) : OffsetSumA(matrixSize, matrixSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    DataCopy(ws[realBase], real, matrixElements);
    DataCopy(ws[imagBase], imag, matrixElements);
    DataCopy(ws[sumBase], sum, matrixElements);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    pipe.Reset();
}

__aicore__ inline void Cher2kSmallMixProducts(
    __gm__ float* base, const uint64_t* leftBase, const uint64_t* rightBase, const uint64_t* outBase,
    LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
    LocalTensor<float>& c1, const Nd2NzParams& aCopy, const Nd2NzParams& bCopyParams,
    const LoadData3DParamsV2<float>& aLoad, const LoadData3DParamsV2<float>& bLoad, const MmadParams& first,
    const MmadParams& accum)
{
    constexpr uint32_t matrix = 64U;
    for (uint32_t product = 0U; product < 3U; ++product) {
        GlobalTensor<float> left;
        GlobalTensor<float> right;
        GlobalTensor<float> output;
        left.SetGlobalBuffer(base + leftBase[product], matrix * matrix);
        right.SetGlobalBuffer(base + rightBase[product], matrix * matrix);
        output.SetGlobalBuffer(base + outBase[product], matrix * matrix);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (uint32_t kBase = 0U; kBase < matrix; kBase += matrix) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
            DataCopy(a1, left[kBase * matrix], aCopy);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            DataCopy(b1, right[kBase * matrix], bCopyParams);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
            Cher2kLoadAndMmad(a1, b1, a2, b2, c1, aLoad, bLoad, kBase == 0U ? first : accum);
        }
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 fix{};
        fix.mSize = matrix;
        fix.nSize = matrix;
        fix.srcStride = matrix;
        fix.dstStride = matrix;
        fix.ndNum = 1U;
        Fixpipe(output, c1, fix);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }
}

__aicore__ inline void Cher2kSmallMixCopyParams(Nd2NzParams& params, uint32_t rows)
{
    constexpr uint32_t matrixSize = 64U;
    params.ndNum = 1U;
    params.nValue = matrixSize;
    params.dValue = rows;
    params.srcNdMatrixStride = 0U;
    params.srcDValue = matrixSize;
    params.dstNzC0Stride = matrixSize;
    params.dstNzNStride = 1U;
    params.dstNzMatrixStride = 0U;
}

__aicore__ inline void Cher2kSmallMixMmadParams(
    MmadParams& first, MmadParams& accum, uint32_t rows, uint32_t cols, uint32_t layers)
{
    first.m = rows;
    first.n = cols;
    first.k = layers;
    first.cmatrixInitVal = true;
    accum = first;
    accum.cmatrixInitVal = false;
}

__aicore__ inline void Cher2kSmallMixCube(
    GM_ADDR workspace, const Cher2kTilingData& tiling, uint16_t readyFlag, uint16_t ackFlag)
{
    constexpr uint32_t matrixSize = 64U;
    AscendC::InitSocState();
    constexpr uint32_t M = CHER2K_SMALL_CUBE_M;
    constexpr uint32_t N = CHER2K_SMALL_CUBE_N;
    // The fixed n=k=64 bucket fits one complete operand panel in L1/L0.
    // Keeping Chemm's generic K=32 split here doubled MTE2/MTE1/MMAD
    // control without adding overlap, so each SG3 product uses one panel.
    constexpr uint32_t K = matrixSize;
    constexpr uint32_t aSize = M * K;
    constexpr uint32_t bSize = K * N;
    constexpr uint32_t cSize = M * N;
    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspace);
    const uint64_t leftBase[3] = {
        OffsetSumA(matrixSize, matrixSize), OffsetAr(matrixSize, matrixSize), OffsetAi(matrixSize, matrixSize)};
    const uint64_t rightBase[3] = {
        OffsetSumB(matrixSize, matrixSize), OffsetBr(matrixSize, matrixSize), OffsetBi(matrixSize, matrixSize)};
    const uint64_t outBase[3] = {
        OffsetRr(matrixSize, matrixSize), OffsetRi(matrixSize, matrixSize), OffsetIr(matrixSize, matrixSize)};

    LocalTensor<float> a1{TPosition::A1, 0U, aSize};
    LocalTensor<float> b1{TPosition::B1, static_cast<uint64_t>(aSize) * sizeof(float), bSize};
    LocalTensor<float> a2{TPosition::A2, 0U, aSize};
    LocalTensor<float> b2{TPosition::B2, 0U, bSize};
    LocalTensor<float> c1{TPosition::CO1, 0U, cSize};
    Nd2NzParams aCopy{};
    Cher2kSmallMixCopyParams(aCopy, M);
    Nd2NzParams bNzParams{};
    Cher2kSmallMixCopyParams(bNzParams, N);
    LoadData3DParamsV2<float> aLoad;
    LoadData3DParamsV2<float> bLoad;
    Cher2kInitLoadData3D(aLoad, K, M, M, true);
    Cher2kInitLoadData3D(bLoad, K, N, N, true);
    MmadParams first{};
    MmadParams accum{};
    Cher2kSmallMixMmadParams(first, accum, M, N, K);

    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    SetHF32Mode(HF32Mode::DISABLE);
    Cher2kSmallMixProducts(
        base, leftBase, rightBase, outBase, a1, b1, a2, b2, c1, aCopy, bNzParams, aLoad, bLoad, first, accum);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    CrossCoreSetFlag<2, PIPE_FIX>(readyFlag);
    CrossCoreWaitFlag<2, PIPE_MTE2>(ackFlag);
}

struct Cher2kSmallMixPostContext {
    TBuf<TPosition::VECCALC> transposeOffsetBuf;
    TBuf<TPosition::VECCALC> interleaveOffsetBuf;
    TBuf<TPosition::VECCALC> realIjBuf;
    TBuf<TPosition::VECCALC> imagIjBuf;
    TBuf<TPosition::VECCALC> realJiBuf;
    TBuf<TPosition::VECCALC> imagJiBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> cAosBuf;
    __aicore__ inline void Init(TPipe& pipe, uint32_t elements)
    {
        pipe.InitBuffer(transposeOffsetBuf, elements * sizeof(uint32_t));
        pipe.InitBuffer(interleaveOffsetBuf, elements * 2U * sizeof(uint32_t));
        pipe.InitBuffer(realIjBuf, elements * 2U * sizeof(float));
        pipe.InitBuffer(imagIjBuf, elements * sizeof(float));
        pipe.InitBuffer(realJiBuf, elements * sizeof(float));
        pipe.InitBuffer(imagJiBuf, elements * sizeof(float));
        pipe.InitBuffer(tempBuf, elements * sizeof(float));
        pipe.InitBuffer(cAosBuf, elements * 2U * sizeof(float));
    }
};

__aicore__ inline void Cher2kSmallMixPostStore(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffset, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& raw,
    LocalTensor<float>& realIj, LocalTensor<float>& imagIj, LocalTensor<float>& realJi, LocalTensor<float>& imagJi,
    LocalTensor<float>& temp, LocalTensor<float>& planar, LocalTensor<float>& cAos, float alphaReal, float alphaImag,
    float betaValue)
{
    GlobalTensor<uint32_t> offsetsG;
    offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleaveOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
    DataCopy(interleaveOffset, offsetsG, CHER2K_DIRECT_INTERLEAVE_COUNT);
    DataCopy(transposeOffset, offsetsG[CHER2K_POST_TRANSPOSE_OFFSET], CHER2K_POST_TRANSPOSE_COUNT);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> cG;
    GlobalTensor<float> ws;
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * 64U * 2ULL);
    ws.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace), WorkspaceFloatCount(64U, 64U, true));
    Cher2kPostprocessUnitTile(
        cG, ws, tiling, transposeOffset, interleaveOffset, raw, realIj, imagIj, realJi, imagJi, temp, planar, cAos, 0U,
        0U, alphaReal, alphaImag, betaValue);
}

__aicore__ inline void Cher2kSmallMixPostprocess(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, GM_ADDR alpha, GM_ADDR beta,
    const Cher2kTilingData& tiling)
{
    constexpr uint32_t matrixSize = 64U;
    constexpr uint32_t matrixElements = matrixSize * matrixSize;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    constexpr uint32_t tileElements = matrixElements;
    TPipe postprocessPipe;
    Cher2kSmallMixPostContext context;
    context.Init(postprocessPipe, tileElements);
    auto& transposeOffsetBuf = context.transposeOffsetBuf;
    auto& interleaveOffsetBuf = context.interleaveOffsetBuf;
    auto& realIjBuf = context.realIjBuf;
    auto& imagIjBuf = context.imagIjBuf;
    auto& realJiBuf = context.realJiBuf;
    auto& imagJiBuf = context.imagJiBuf;
    auto& tempBuf = context.tempBuf;
    auto& cAosBuf = context.cAosBuf;

    LocalTensor<uint32_t> transposeOffset = transposeOffsetBuf.Get<uint32_t>();
    LocalTensor<uint32_t> interleaveOffset = interleaveOffsetBuf.Get<uint32_t>();
    LocalTensor<float> planar = realIjBuf.Get<float>();
    LocalTensor<float> realIj = planar;
    LocalTensor<float> imagIj = imagIjBuf.Get<float>();
    LocalTensor<float> realJi = realJiBuf.Get<float>();
    LocalTensor<float> imagJi = imagJiBuf.Get<float>();
    LocalTensor<float> temp = tempBuf.Get<float>();
    LocalTensor<float> cAos = cAosBuf.Get<float>();
    LocalTensor<float> raw = cAos;

    Cher2kSmallMixPostStore(
        c, workspace, interleaveOffsets, tiling, transposeOffset, interleaveOffset, raw, realIj, imagIj, realJi, imagJi,
        temp, planar, cAos, alphaReal, alphaImag, betaValue);
}

__aicore__ inline void Cher2kSmallMixConsume(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, GM_ADDR alpha, GM_ADDR beta,
    const Cher2kTilingData& tiling, uint16_t readyFlag, uint16_t ackFlag)
{
    CrossCoreWaitFlag<2, PIPE_MTE2>(readyFlag);
    const uint32_t subBlock = static_cast<uint32_t>(GetSubBlockIdx());
    if (subBlock == 0U) {
        Cher2kSmallMixPostprocess(c, workspace, interleaveOffsets, alpha, beta, tiling);
    }
    CrossCoreSetFlag<2, PIPE_MTE3>(ackFlag);
}

// Single-launch SG3 lifecycle for the 64x64 OP_N bucket. Two AIV sub-blocks
// unpack A/B concurrently, one AIC owns all three products, and one AIV
// consumes the completed tile. The cross-core flag pair publishes Fixpipe
// output and acknowledges the final C write without separate kernel launches.
extern "C" __global__ __aicore__ __schedmode__(1) void cher2k_small_sg3_mix_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, GM_ADDR alpha, GM_ADDR beta,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta)) {
        return;
    }
    constexpr uint16_t readyFlag = 0U;
    constexpr uint16_t ackFlag = CHER2K_MIX_ACK_BASE;

    if ASCEND_IS_AIV {
        Cher2kSmallMixPublish(a, b, workspace, tiling);
    }
    SyncAll<false>();
    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        Cher2kSmallMixCube(workspace, tiling, readyFlag, ackFlag);
    }
    if ASCEND_IS_AIV {
        Cher2kSmallMixConsume(c, workspace, interleaveOffsets, alpha, beta, tiling, readyFlag, ackFlag);
    }
}
__aicore__ inline void Cher2kSmallFusedAdd(float& sum, float& correction, float value, bool compensated)
{
    if (!compensated) {
        sum += value;
        return;
    }
    const float corrected = value - correction;
    const float next = sum + corrected;
    correction = (next - sum) - corrected;
    sum = next;
}

__aicore__ inline void Cher2kSmallFusedElementN(
    __gm__ const float* aPtr, __gm__ const float* bPtr, const Cher2kTilingData& tiling, uint32_t row, uint32_t col,
    bool compensated, float& p1r, float& p1i, float& p2r, float& p2i, float& p1rCorrection, float& p1iCorrection,
    float& p2rCorrection, float& p2iCorrection)
{
    for (uint32_t l = 0U; l < tiling.k; ++l) {
        const uint64_t a1 = (static_cast<uint64_t>(row) + static_cast<uint64_t>(l) * tiling.lda) * 2ULL;
        const uint64_t a2 = (static_cast<uint64_t>(col) + static_cast<uint64_t>(l) * tiling.lda) * 2ULL;
        const uint64_t b1 = (static_cast<uint64_t>(row) + static_cast<uint64_t>(l) * tiling.ldb) * 2ULL;
        const uint64_t b2 = (static_cast<uint64_t>(col) + static_cast<uint64_t>(l) * tiling.ldb) * 2ULL;
        const float ar = aPtr[a1];
        const float ai = aPtr[a1 + 1ULL];
        const float br = bPtr[b2];
        const float bi = bPtr[b2 + 1ULL];
        Cher2kSmallFusedAdd(p1r, p1rCorrection, ar * br, compensated);
        Cher2kSmallFusedAdd(p1r, p1rCorrection, ai * bi, compensated);
        Cher2kSmallFusedAdd(p1i, p1iCorrection, ai * br, compensated);
        Cher2kSmallFusedAdd(p1i, p1iCorrection, -ar * bi, compensated);
        const float br2 = bPtr[b1];
        const float bi2 = bPtr[b1 + 1ULL];
        const float ar2 = aPtr[a2];
        const float ai2 = aPtr[a2 + 1ULL];
        Cher2kSmallFusedAdd(p2r, p2rCorrection, br2 * ar2, compensated);
        Cher2kSmallFusedAdd(p2r, p2rCorrection, bi2 * ai2, compensated);
        Cher2kSmallFusedAdd(p2i, p2iCorrection, bi2 * ar2, compensated);
        Cher2kSmallFusedAdd(p2i, p2iCorrection, -br2 * ai2, compensated);
    }
}

__aicore__ inline void Cher2kSmallFusedElementC(
    __gm__ const float* aPtr, __gm__ const float* bPtr, const Cher2kTilingData& tiling, uint32_t row, uint32_t col,
    bool compensated, float& p1r, float& p1i, float& p2r, float& p2i, float& p1rCorrection, float& p1iCorrection,
    float& p2rCorrection, float& p2iCorrection)
{
    for (uint32_t l = 0U; l < tiling.k; ++l) {
        const uint64_t a1 = (static_cast<uint64_t>(l) + static_cast<uint64_t>(row) * tiling.lda) * 2ULL;
        const uint64_t a2 = (static_cast<uint64_t>(l) + static_cast<uint64_t>(col) * tiling.lda) * 2ULL;
        const uint64_t b1 = (static_cast<uint64_t>(l) + static_cast<uint64_t>(row) * tiling.ldb) * 2ULL;
        const uint64_t b2 = (static_cast<uint64_t>(l) + static_cast<uint64_t>(col) * tiling.ldb) * 2ULL;
        const float ar = aPtr[a1];
        const float ai = -aPtr[a1 + 1ULL];
        const float br = bPtr[b2];
        const float bi = bPtr[b2 + 1ULL];
        Cher2kSmallFusedAdd(p1r, p1rCorrection, ar * br, compensated);
        Cher2kSmallFusedAdd(p1r, p1rCorrection, -ai * bi, compensated);
        Cher2kSmallFusedAdd(p1i, p1iCorrection, ar * bi, compensated);
        Cher2kSmallFusedAdd(p1i, p1iCorrection, ai * br, compensated);
        const float br2 = bPtr[b1];
        const float bi2 = -bPtr[b1 + 1ULL];
        const float ar2 = aPtr[a2];
        const float ai2 = aPtr[a2 + 1ULL];
        Cher2kSmallFusedAdd(p2r, p2rCorrection, br2 * ar2, compensated);
        Cher2kSmallFusedAdd(p2r, p2rCorrection, -bi2 * ai2, compensated);
        Cher2kSmallFusedAdd(p2i, p2iCorrection, br2 * ai2, compensated);
        Cher2kSmallFusedAdd(p2i, p2iCorrection, bi2 * ar2, compensated);
    }
}

struct Cher2kSmallFusedScalars {
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    bool unitAlphaNoBeta = false;

    __aicore__ inline void Init(const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta)
    {
        Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
        unitAlphaNoBeta = alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f;
    }
};

__aicore__ inline void Cher2kSmallFusedValue(
    __gm__ const float* aPtr, __gm__ const float* bPtr, GlobalTensor<float>& cG, const Cher2kTilingData& tiling,
    const Cher2kSmallFusedScalars& scalars, uint32_t row, uint32_t col, float& outR, float& outI)
{
    float p1r = 0.0f;
    float p1i = 0.0f;
    float p2r = 0.0f;
    float p2i = 0.0f;
    float p1rCorrection = 0.0f;
    float p1iCorrection = 0.0f;
    float p2rCorrection = 0.0f;
    float p2iCorrection = 0.0f;
    const bool compensated = tiling.n == 32U;
    if (tiling.trans == ACLBLAS_OP_N) {
        Cher2kSmallFusedElementN(
            aPtr, bPtr, tiling, row, col, compensated, p1r, p1i, p2r, p2i, p1rCorrection, p1iCorrection, p2rCorrection,
            p2iCorrection);
    } else {
        Cher2kSmallFusedElementC(
            aPtr, bPtr, tiling, row, col, compensated, p1r, p1i, p2r, p2i, p1rCorrection, p1iCorrection, p2rCorrection,
            p2iCorrection);
    }
    outR = p1r;
    outI = p1i;
    if (scalars.unitAlphaNoBeta) {
        outR += p2r;
        outI += p2i;
    } else {
        outR = scalars.alphaReal * (p1r + p2r) + scalars.alphaImag * (p2i - p1i);
        outI = scalars.alphaReal * (p1i + p2i) + scalars.alphaImag * (p1r - p2r);
        const uint64_t offset = (static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * tiling.ldc) * 2ULL;
        if (scalars.betaValue != 0.0f) {
            outR += scalars.betaValue * cG.GetValue(offset);
            outI += scalars.betaValue * cG.GetValue(offset + 1ULL);
        }
    }
    if (row == col) {
        outI = 0.0f;
    }
}

__aicore__ inline void Cher2kSmallFusedColumn(
    __gm__ const float* aPtr, __gm__ const float* bPtr, GlobalTensor<float>& cG, LocalTensor<float>& columnOutput,
    const Cher2kTilingData& tiling, const Cher2kSmallFusedScalars& scalars, uint32_t col, event_t scalarToMte3,
    event_t mte3ToScalar)
{
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    const uint32_t rowBegin = upper ? 0U : col;
    const uint32_t rowEnd = upper ? col + 1U : tiling.n;
    WaitFlag<HardEvent::MTE3_S>(mte3ToScalar);
    for (uint32_t row = rowBegin; row < rowEnd; ++row) {
        float outR = 0.0f;
        float outI = 0.0f;
        Cher2kSmallFusedValue(aPtr, bPtr, cG, tiling, scalars, row, col, outR, outI);
        const uint32_t localOffset = (row - rowBegin) * 2U;
        columnOutput.SetValue(localOffset, outR);
        columnOutput.SetValue(localOffset + 1U, outI);
    }
    SetFlag<HardEvent::S_MTE3>(scalarToMte3);
    WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
    const uint64_t columnOffset = (static_cast<uint64_t>(col) * tiling.ldc + rowBegin) * 2ULL;
    const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>((rowEnd - rowBegin) * 2U * sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(cG[columnOffset], columnOutput, outputCopy);
    SetFlag<HardEvent::MTE3_S>(mte3ToScalar);
}

extern "C" __global__ __aicore__ void cher2k_small_fused_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta)) {
        return;
    }
    constexpr uint32_t maxFallbackN = 2048U;
    TPipe pipe;
    TBuf<TPosition::VECCALC> columnOutputBuf;
    pipe.InitBuffer(columnOutputBuf, maxFallbackN * 2U * sizeof(float));
    LocalTensor<float> columnOutput = columnOutputBuf.Get<float>();
    const event_t scalarToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    const event_t mte3ToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
    if (blocks == 0U) {
        blocks = 1U;
    }
    Cher2kSmallFusedScalars scalars;
    scalars.Init(tiling, alpha, beta);
    __gm__ const float* aPtr = reinterpret_cast<__gm__ const float*>(a);
    __gm__ const float* bPtr = reinterpret_cast<__gm__ const float*>(b);
    GlobalTensor<float> cG;
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    const uint32_t n = tiling.n;
    const uint32_t ldcRemainder = tiling.ldc & 3U;
    const uint32_t columnGroup = ldcRemainder == 0U ? 1U : (ldcRemainder == 2U ? 2U : 4U);
    const uint32_t columnGroupCount = (n + columnGroup - 1U) / columnGroup;
    const uint32_t groupsPerBlock = (columnGroupCount + blocks - 1U) / blocks;
    const uint32_t groupBegin = block * groupsPerBlock;
    const uint32_t groupEnd = min(columnGroupCount, groupBegin + groupsPerBlock);
    const uint32_t colBegin = groupBegin * columnGroup;
    const uint32_t colEnd = min(n, groupEnd * columnGroup);
    SetFlag<HardEvent::MTE3_S>(mte3ToScalar);
    for (uint32_t col = colBegin; col < colEnd; ++col) {
        Cher2kSmallFusedColumn(aPtr, bPtr, cG, columnOutput, tiling, scalars, col, scalarToMte3, mte3ToScalar);
    }
    WaitFlag<HardEvent::MTE3_S>(mte3ToScalar);
}
__aicore__ inline void Cher2kTinySquareAccumulate(
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& p1r, LocalTensor<float>& p1i, LocalTensor<float>& p2r, LocalTensor<float>& p2i, uint32_t nPad,
    uint32_t col, uint32_t rowBegin, uint32_t count, uint32_t k)
{
    for (uint32_t l = 0U; l < k; ++l) {
        const uint32_t matrixOffset = l * nPad;
        const float acr = ar.GetValue(matrixOffset + col);
        const float aci = ai.GetValue(matrixOffset + col);
        const float bcr = br.GetValue(matrixOffset + col);
        const float bci = bi.GetValue(matrixOffset + col);
        const LocalTensor<float> arRows = ar[matrixOffset + rowBegin];
        const LocalTensor<float> aiRows = ai[matrixOffset + rowBegin];
        const LocalTensor<float> brRows = br[matrixOffset + rowBegin];
        const LocalTensor<float> biRows = bi[matrixOffset + rowBegin];
        Axpy(p1r, arRows, bcr, count);
        Axpy(p1i, arRows, -bci, count);
        PipeBarrier<PIPE_V>();
        Axpy(p1r, aiRows, bci, count);
        Axpy(p1i, aiRows, bcr, count);
        PipeBarrier<PIPE_V>();
        Axpy(p2r, brRows, acr, count);
        Axpy(p2i, brRows, -aci, count);
        PipeBarrier<PIPE_V>();
        Axpy(p2r, biRows, aci, count);
        Axpy(p2i, biRows, acr, count);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void Cher2kTinySquareApplyAlpha(
    LocalTensor<float>& p1r, LocalTensor<float>& p1i, LocalTensor<float>& p2r, LocalTensor<float>& p2i,
    LocalTensor<float>& outR, LocalTensor<float>& outI, uint32_t nPad, uint32_t count, bool unitAlphaNoBeta,
    float alphaReal, float alphaImag)
{
    if (unitAlphaNoBeta) {
        Add(outR, p1r, p2r, count);
        Add(outI, p1i, p2i, count);
    } else {
        Duplicate(outR, 0.0f, nPad);
        Duplicate(outI, 0.0f, nPad);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p1r, alphaReal, count);
        Axpy(outI, p1i, alphaReal, count);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p2r, alphaReal, count);
        Axpy(outI, p2i, alphaReal, count);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p1i, -alphaImag, count);
        Axpy(outI, p1r, alphaImag, count);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p2i, alphaImag, count);
        Axpy(outI, p2r, -alphaImag, count);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kTinySquareApplyBeta(
    GlobalTensor<float>& cG, LocalTensor<float>& aAos, LocalTensor<float>& p1r, LocalTensor<float>& p1i,
    LocalTensor<float>& outR, LocalTensor<float>& outI, const Cher2kTilingData& tiling, uint32_t col, uint32_t rowBegin,
    uint32_t count, float betaValue, uint64_t& reserved, event_t mte2ToV)
{
    if (betaValue == 0.0f) {
        return;
    }
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + rowBegin) * 2ULL;
    const uint32_t oldBytes = count * 2U * sizeof(float);
    const uint32_t oldAlignedBytes = (oldBytes + 31U) & ~31U;
    const DataCopyExtParams oldCopy{1U, oldBytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> oldPad{
        oldBytes != oldAlignedBytes, 0U, static_cast<uint8_t>((oldAlignedBytes - oldBytes) / sizeof(float)), 0.0f};
    DataCopyPad(aAos, cG[cOffset], oldCopy, oldPad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    const uint16_t oldRepeats = static_cast<uint16_t>((count * 2U + 63U) / 64U);
    GatherMask<float>(p1r, aAos, 1U, false, 0U, {1U, oldRepeats, 8U, 8U}, reserved);
    GatherMask<float>(p1i, aAos, 2U, false, 0U, {1U, oldRepeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
    Axpy(outR, p1r, betaValue, count);
    Axpy(outI, p1i, betaValue, count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kTinySquareStoreColumn(
    GlobalTensor<float>& cG, LocalTensor<float>& outR, LocalTensor<float>& outI, LocalTensor<float>& outputAos,
    LocalTensor<uint32_t>& interleaveOffset, const Cher2kTilingData& tiling, uint32_t col, uint32_t rowBegin,
    uint32_t count, uint32_t nPad, event_t vToScalar, event_t scalarToV, event_t vToMte3, event_t mte3ToV)
{
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    outI.SetValue(col - rowBegin, 0.0f);
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
    Gather(outputAos, outR, interleaveOffset, 0U, nPad * 2U);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    const uint64_t outputOffset = (static_cast<uint64_t>(col) * tiling.ldc + rowBegin) * 2ULL;
    const DataCopyExtParams outputCopy{1U, static_cast<uint32_t>(count * 2U * sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(cG[outputOffset], outputAos, outputCopy);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

__aicore__ inline void Cher2kTinySquareProcessColumn(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& ar, LocalTensor<float>& ai,
    LocalTensor<float>& br, LocalTensor<float>& bi, LocalTensor<float>& p1r, LocalTensor<float>& p1i,
    LocalTensor<float>& p2r, LocalTensor<float>& p2i, LocalTensor<float>& outR, LocalTensor<float>& outI,
    LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& outputAos, LocalTensor<float>& aAos, uint32_t n,
    uint32_t k, uint32_t nPad, uint32_t col, bool upper, bool unitAlphaNoBeta, float alphaReal, float alphaImag,
    float betaValue, uint64_t& reserved, event_t vToMte2, event_t mte2ToV, event_t vToScalar, event_t scalarToV,
    event_t vToMte3, event_t mte3ToV)
{
    const uint32_t rowBegin = upper ? 0U : col;
    const uint32_t rowEnd = upper ? col + 1U : n;
    const uint32_t count = rowEnd - rowBegin;
    Duplicate(p1r, 0.0f, nPad);
    Duplicate(p1i, 0.0f, nPad);
    Duplicate(p2r, 0.0f, nPad);
    Duplicate(p2i, 0.0f, nPad);
    PipeBarrier<PIPE_V>();
    Cher2kTinySquareAccumulate(ar, ai, br, bi, p1r, p1i, p2r, p2i, nPad, col, rowBegin, count, k);
    Cher2kTinySquareApplyAlpha(p1r, p1i, p2r, p2i, outR, outI, nPad, count, unitAlphaNoBeta, alphaReal, alphaImag);
    Cher2kTinySquareApplyBeta(
        cG, aAos, p1r, p1i, outR, outI, tiling, col, rowBegin, count, betaValue, reserved, mte2ToV);
    (void)vToMte2;
    Cher2kTinySquareStoreColumn(
        cG, outR, outI, outputAos, interleaveOffset, tiling, col, rowBegin, count, nPad, vToScalar, scalarToV, vToMte3,
        mte3ToV);
}

/** Execute the fixed-shape column ownership loop. */
__aicore__ inline void Cher2kTinySquareProcessColumns(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& ar, LocalTensor<float>& ai,
    LocalTensor<float>& br, LocalTensor<float>& bi, LocalTensor<float>& p1r, LocalTensor<float>& p1i,
    LocalTensor<float>& p2r, LocalTensor<float>& p2i, LocalTensor<float>& outR, LocalTensor<float>& outI,
    LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& outputAos, LocalTensor<float>& aAos, uint32_t n,
    uint32_t k, uint32_t nPad, uint32_t colBegin, uint32_t colEnd, bool upper, bool unitAlphaNoBeta, float alphaReal,
    float alphaImag, float betaValue, uint64_t& reserved, event_t vToMte2, event_t mte2ToV, event_t vToScalar,
    event_t scalarToV, event_t vToMte3, event_t mte3ToV)
{
    for (uint32_t col = colBegin; col < colEnd; ++col) {
        Cher2kTinySquareProcessColumn(
            cG, tiling, ar, ai, br, bi, p1r, p1i, p2r, p2i, outR, outI, interleaveOffset, outputAos, aAos, n, k, nPad,
            col, upper, unitAlphaNoBeta, alphaReal, alphaImag, betaValue, reserved, vToMte2, mte2ToV, vToScalar,
            scalarToV, vToMte3, mte3ToV);
    }
}
struct Cher2kOperandViews {
    LocalTensor<float> aAos;
    LocalTensor<float> bAos;
    LocalTensor<float> ar;
    LocalTensor<float> ai;
    LocalTensor<float> br;
    LocalTensor<float> bi;
    LocalTensor<float> p1r;
    LocalTensor<float> p1i;
    LocalTensor<float> p2r;
    LocalTensor<float> p2i;
    LocalTensor<float> outR;
    LocalTensor<float> outI;

    __aicore__ inline void BindOperands(
        LocalTensor<float> input, LocalTensor<float> planar, LocalTensor<float> accumulator, uint32_t matrix,
        uint32_t rows)
    {
        aAos = input;
        bAos = aAos[2U * matrix];
        ar = planar;
        ai = ar[matrix];
        br = ai[matrix];
        bi = br[matrix];
        p1r = accumulator;
        p1i = p1r[rows];
        p2r = p1i[rows];
        p2i = p2r[rows];
        outR = p2i[rows];
        outI = outR[rows];
    }
};
struct Cher2kColumnStorage : Cher2kOperandViews {
    TBuf<TPosition::VECCALC> aosBuf;
    TBuf<TPosition::VECCALC> planarBuf;
    TBuf<TPosition::VECCALC> accumulatorBuf;
    TBuf<TPosition::VECCALC> interleaveOffsetBuf;
    TBuf<TPosition::VECCALC> outputAosBuf;
    LocalTensor<uint32_t> interleaveOffset;
    LocalTensor<float> outputAos;

    __aicore__ inline void Allocate(TPipe& pipe, uint32_t matrix, uint32_t rows)
    {
        pipe.InitBuffer(aosBuf, matrix * 4U * sizeof(float));
        pipe.InitBuffer(planarBuf, matrix * 4U * sizeof(float));
        pipe.InitBuffer(accumulatorBuf, rows * 6U * sizeof(float));
        pipe.InitBuffer(interleaveOffsetBuf, rows * 2U * sizeof(uint32_t));
        pipe.InitBuffer(outputAosBuf, rows * 2U * sizeof(float));
    }
    __aicore__ inline void Bind(uint32_t matrix, uint32_t rows)
    {
        BindOperands(aosBuf.Get<float>(), planarBuf.Get<float>(), accumulatorBuf.Get<float>(), matrix, rows);
        interleaveOffset = interleaveOffsetBuf.Get<uint32_t>();
        outputAos = outputAosBuf.Get<float>();
    }
};

struct Cher2kTinySquareContext : Cher2kColumnStorage {
    __aicore__ inline void Init(TPipe& pipe)
    {
        constexpr uint32_t rows = 32U;
        constexpr uint32_t matrix = rows * rows;
        Allocate(pipe, matrix, rows);
        Bind(matrix, rows);
    }
};

__aicore__ inline void Cher2kTinySquareLoad(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& cG, GlobalTensor<uint32_t>& offsetsG,
    LocalTensor<float>& aAos, LocalTensor<float>& bAos, LocalTensor<float>& ar, LocalTensor<float>& ai,
    LocalTensor<float>& br, LocalTensor<float>& bi, LocalTensor<uint32_t>& interleaveOffset,
    const Cher2kTilingData& tiling, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR interleaveOffsets, uint32_t n, uint32_t k,
    event_t vToMte2, event_t mte2ToV)
{
    aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * k * 2ULL);
    bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * k * 2ULL);
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * n * 2ULL);
    offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleaveOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
    const uint32_t matrix = n * k;
    Duplicate(aAos, 0.0f, matrix * 2U);
    Duplicate(bAos, 0.0f, matrix * 2U);
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
    const uint32_t blockLen = n * 2U * sizeof(float);
    const uint32_t alignedLen = (blockLen + 31U) & ~31U;
    const uint32_t stride = (n * 2U * sizeof(float) - alignedLen) / 32U;
    const DataCopyExtParams aCopy{
        static_cast<uint16_t>(k), blockLen, static_cast<uint32_t>((tiling.lda - n) * 2U * sizeof(float)), stride, 0U};
    const DataCopyExtParams bCopyParams{
        static_cast<uint16_t>(k), blockLen, static_cast<uint32_t>((tiling.ldb - n) * 2U * sizeof(float)), stride, 0U};
    const uint32_t padding = (alignedLen - blockLen) / sizeof(float);
    const DataCopyPadExtParams<float> pad{padding != 0U, 0U, static_cast<uint8_t>(padding), 0.0f};
    DataCopyPad(aAos, aG, aCopy, pad);
    DataCopyPad(bAos, bG, bCopyParams, pad);
    DataCopy(interleaveOffset, offsetsG[CHER2K_TINY_INTERLEAVE_OFFSET], CHER2K_TINY_INTERLEAVE_COUNT);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    uint64_t reserved = 0ULL;
    const uint16_t repeats = static_cast<uint16_t>(matrix * 2U / 64U);
    GatherMask<float>(ar, aAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(ai, aAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(br, bAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(bi, bAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kTinySquareEvents(
    event_t& vToMte2, event_t& mte2ToV, event_t& vToScalar, event_t& scalarToV, event_t& vToMte3, event_t& mte3ToV)
{
    vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

__aicore__ inline void Cher2kTinySquareCompute(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta, LocalTensor<float>& aAos,
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& p1r, LocalTensor<float>& p1i, LocalTensor<float>& p2r, LocalTensor<float>& p2i,
    LocalTensor<float>& outR, LocalTensor<float>& outI, LocalTensor<uint32_t>& interleaveOffset,
    LocalTensor<float>& outputAos, uint32_t n, uint32_t k, uint32_t nPad, event_t vToMte2, event_t mte2ToV,
    event_t vToScalar, event_t scalarToV, event_t vToMte3, event_t mte3ToV)
{
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
    if (blocks == 0U) {
        blocks = 1U;
    }
    const uint32_t groups = (n + blocks - 1U) / blocks;
    const uint32_t colBegin = block * groups;
    const uint32_t colEnd = min(n, colBegin + groups);
    uint64_t reserved = 0ULL;
    Cher2kTinySquareProcessColumns(
        cG, tiling, ar, ai, br, bi, p1r, p1i, p2r, p2i, outR, outI, interleaveOffset, outputAos, aAos, n, k, nPad,
        colBegin, colEnd, upper, alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f, alphaReal, alphaImag,
        betaValue, reserved, vToMte2, mte2ToV, vToScalar, scalarToV, vToMte3, mte3ToV);
}

// Fixed-shape OP_N path for n=k=16/32. Each AIV owns complete C columns,
// matching the general small-GEMV ownership while removing its n=128 UB
// graph and unused low-rank panel scratch.
extern "C" __global__ __aicore__ void cher2k_tiny_square_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta, GM_ADDR interleaveOffsets,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta)) {
        return;
    }

    TPipe pipe;
    Cher2kTinySquareContext context;
    context.Init(pipe);

    const uint32_t n = tiling.n;
    const uint32_t k = tiling.k;
    if ((n != 16U && n != 32U) || k != n || interleaveOffsets == nullptr) {
        return;
    }
    const uint32_t nPad = n;

    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> cG;
    GlobalTensor<uint32_t> offsetsG;

    event_t vToMte2;
    event_t mte2ToV;
    event_t vToScalar;
    event_t scalarToV;
    event_t vToMte3;
    event_t mte3ToV;
    Cher2kTinySquareEvents(vToMte2, mte2ToV, vToScalar, scalarToV, vToMte3, mte3ToV);

    Cher2kTinySquareLoad(
        aG, bG, cG, offsetsG, context.aAos, context.bAos, context.ar, context.ai, context.br, context.bi,
        context.interleaveOffset, tiling, a, b, c, interleaveOffsets, n, k, vToMte2, mte2ToV);
    // The column accumulation reads planar coefficients through GetValue.
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    Cher2kTinySquareCompute(
        cG, tiling, alpha, beta, context.aAos, context.ar, context.ai, context.br, context.bi, context.p1r, context.p1i,
        context.p2r, context.p2i, context.outR, context.outI, context.interleaveOffset, context.outputAos, n, k, nPad,
        vToMte2, mte2ToV, vToScalar, scalarToV, vToMte3, mte3ToV);
}
// Low-rank OP_N fast path built from the proven arch22 CGEMV dataflow.  Each
// AIV owns complete C columns, loads the two small input matrices once, and
// expresses both rank-k terms as complex AXPY chains.  This removes the
// output-element x K scalar loop from the tiny path while retaining a single
// kernel launch and a workspace-free contract.
__aicore__ inline void Cher2kSmallGemvOuterRow(
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& outReal, LocalTensor<float>& outImag, LocalTensor<float>& tmp0, LocalTensor<float>& tmp1,
    LocalTensor<float>& coeffBr, LocalTensor<float>& coeffBi, LocalTensor<float>& coeffAr, LocalTensor<float>& coeffAi,
    uint32_t srcBase, uint32_t dstBase, uint32_t rowChunk, uint8_t batchRepeats, uint8_t dstRepeatStride)
{
    const BinaryRepeatParams mulParams{1U, 1U, 0U, dstRepeatStride, 0U, 1U};
    const BinaryRepeatParams addParams{1U, 1U, 1U, dstRepeatStride, dstRepeatStride, dstRepeatStride};
    Mul(tmp0[dstBase], ar[srcBase], coeffBr, rowChunk, batchRepeats, mulParams);
    Mul(tmp1[dstBase], ai[srcBase], coeffBi, rowChunk, batchRepeats, mulParams);
    PipeBarrier<PIPE_V>();
    Add(tmp0[dstBase], tmp0[dstBase], tmp1[dstBase], rowChunk, batchRepeats, addParams);
    PipeBarrier<PIPE_V>();
    Add(outReal[dstBase], outReal[dstBase], tmp0[dstBase], rowChunk, batchRepeats, addParams);
    Mul(tmp0[dstBase], ai[srcBase], coeffBr, rowChunk, batchRepeats, mulParams);
    Mul(tmp1[dstBase], ar[srcBase], coeffBi, rowChunk, batchRepeats, mulParams);
    PipeBarrier<PIPE_V>();
    Sub(tmp0[dstBase], tmp0[dstBase], tmp1[dstBase], rowChunk, batchRepeats, addParams);
    PipeBarrier<PIPE_V>();
    Add(outImag[dstBase], outImag[dstBase], tmp0[dstBase], rowChunk, batchRepeats, addParams);
    Mul(tmp0[dstBase], br[srcBase], coeffAr, rowChunk, batchRepeats, mulParams);
    Mul(tmp1[dstBase], bi[srcBase], coeffAi, rowChunk, batchRepeats, mulParams);
    PipeBarrier<PIPE_V>();
    Add(tmp0[dstBase], tmp0[dstBase], tmp1[dstBase], rowChunk, batchRepeats, addParams);
    PipeBarrier<PIPE_V>();
    Add(outReal[dstBase], outReal[dstBase], tmp0[dstBase], rowChunk, batchRepeats, addParams);
    Mul(tmp0[dstBase], bi[srcBase], coeffAr, rowChunk, batchRepeats, mulParams);
    Mul(tmp1[dstBase], br[srcBase], coeffAi, rowChunk, batchRepeats, mulParams);
    PipeBarrier<PIPE_V>();
    Sub(tmp0[dstBase], tmp0[dstBase], tmp1[dstBase], rowChunk, batchRepeats, addParams);
    PipeBarrier<PIPE_V>();
    Add(outImag[dstBase], outImag[dstBase], tmp0[dstBase], rowChunk, batchRepeats, addParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kSmallGemvOuterOffsets(
    LocalTensor<uint32_t>& outerOffset, LocalTensor<int32_t>& outerOffsetI32, GM_ADDR interleaveOffsets,
    event_t scalarToV)
{
    constexpr uint32_t maxRows = 128U;
    constexpr uint32_t batchCols = 8U;
    constexpr uint32_t outerTileElements = batchCols * maxRows;
    if (interleaveOffsets != nullptr) {
        GlobalTensor<uint32_t> offsetsG;
        offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleaveOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
        DataCopy(outerOffset, offsetsG[CHER2K_SMALL_INTERLEAVE_OFFSET], CHER2K_SMALL_INTERLEAVE_COUNT);
        return;
    }
    for (uint32_t index = 0U; index < maxRows * 2U; ++index) {
        const uint32_t sourceIndex = (index & 1U) == 0U ? index / 2U : outerTileElements + index / 2U;
        outerOffset.SetValue(index, sourceIndex * sizeof(float));
    }
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
    for (uint32_t col = 1U; col < batchCols; ++col) {
        Adds(
            outerOffsetI32[col * maxRows * 2U], outerOffsetI32, static_cast<int32_t>(col * maxRows * sizeof(float)),
            maxRows * 2U);
    }
}

__aicore__ inline void Cher2kSmallGemvOuterCompute(
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& outReal, LocalTensor<float>& outImag, LocalTensor<float>& tmp0, LocalTensor<float>& tmp1,
    LocalTensor<float>& coeffBr, LocalTensor<float>& coeffBi, LocalTensor<float>& coeffAr, LocalTensor<float>& coeffAi,
    uint32_t colBase, uint32_t k, uint32_t nPad)
{
    constexpr uint32_t maxRows = 128U;
    constexpr uint32_t rowChunk = 64U;
    constexpr uint8_t repeats = 8U;
    constexpr uint8_t repeatStride = maxRows / 8U;
    for (uint32_t l = 0U; l < k; ++l) {
        const uint32_t matrixOffset = l * nPad;
        Brcb(coeffBr, br[matrixOffset + colBase], 1U, {1U, 8U});
        Brcb(coeffBi, bi[matrixOffset + colBase], 1U, {1U, 8U});
        Brcb(coeffAr, ar[matrixOffset + colBase], 1U, {1U, 8U});
        Brcb(coeffAi, ai[matrixOffset + colBase], 1U, {1U, 8U});
        PipeBarrier<PIPE_V>();
        for (uint32_t rowBase = 0U; rowBase < maxRows; rowBase += rowChunk) {
            Cher2kSmallGemvOuterRow(
                ar, ai, br, bi, outReal, outImag, tmp0, tmp1, coeffBr, coeffBi, coeffAr, coeffAi,
                matrixOffset + rowBase, rowBase, rowChunk, repeats, repeatStride);
        }
    }
}

__aicore__ inline void Cher2kSmallGemvOuterStore(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& outReal, LocalTensor<float>& outImag,
    LocalTensor<float>& outerAos, LocalTensor<uint32_t>& outerOffset, uint32_t colBase, uint32_t validCols,
    event_t vToScalar, event_t scalarToV, event_t mte2ToV, event_t vToMte3, event_t mte3ToV)
{
    constexpr uint32_t maxRows = 128U;
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    for (uint32_t col = 0U; col < validCols; ++col) {
        outImag.SetValue(col * maxRows + colBase + col, 0.0f);
    }
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    Gather(outerAos, outReal, outerOffset, 0U, 16U * maxRows);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    for (uint32_t col = 0U; col < validCols; ++col) {
        const uint32_t absoluteCol = colBase + col;
        const uint64_t offset = static_cast<uint64_t>(absoluteCol) * tiling.ldc * 2ULL;
        const DataCopyExtParams outputCopy{1U, (absoluteCol + 1U) * 8U, 0U, 0U, 0U};
        DataCopyPad(cG[offset], outerAos[col * maxRows * 2U], outputCopy);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

struct Cher2kSmallGemvOuterContext {
    static constexpr uint32_t tileElements = 8U * 128U;
    static constexpr uint32_t accFloats = 4U * tileElements;
    static constexpr uint32_t coeffFloats = 4U * 64U;
    static constexpr uint32_t offsetWords = 2U * tileElements;
    LocalTensor<float> outReal;
    LocalTensor<float> outImag;
    LocalTensor<float> tmp0;
    LocalTensor<float> tmp1;
    LocalTensor<float> coeffBr;
    LocalTensor<float> coeffBi;
    LocalTensor<float> coeffAr;
    LocalTensor<float> coeffAi;
    LocalTensor<uint32_t> offset;
    LocalTensor<int32_t> offsetI32;
    LocalTensor<float> aos;

    __aicore__ inline void Init(LocalTensor<float>& storage)
    {
        outReal = storage;
        outImag = outReal[tileElements];
        tmp0 = outImag[tileElements];
        tmp1 = tmp0[tileElements];
        coeffBr = storage[accFloats];
        coeffBi = coeffBr[64U];
        coeffAr = coeffBi[64U];
        coeffAi = coeffAr[64U];
        offset = storage[accFloats + coeffFloats].ReinterpretCast<uint32_t>();
        offsetI32 = offset.ReinterpretCast<int32_t>();
        aos = storage[accFloats + coeffFloats + offsetWords];
    }
};

__aicore__ inline bool Cher2kSmallGemvOuterBatch(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& ar, LocalTensor<float>& ai,
    LocalTensor<float>& br, LocalTensor<float>& bi, LocalTensor<float>& outerStorage, GM_ADDR interleaveOffsets,
    uint32_t n, uint32_t k, uint32_t nPad, uint32_t block, bool unitAlphaNoBeta, bool upper, event_t scalarToV,
    event_t mte2ToV, event_t vToScalar, event_t vToMte3, event_t mte3ToV)
{
    if (!(unitAlphaNoBeta && upper && n > 64U && k <= 4U)) {
        return false;
    }
    constexpr uint32_t maxRows = 128U;
    constexpr uint32_t batchCols = 8U;
    const uint32_t colBase = block * batchCols;
    if (colBase >= n) {
        return true;
    }
    const uint32_t validCols = min(batchCols, n - colBase);
    Cher2kSmallGemvOuterContext context;
    context.Init(outerStorage);
    Cher2kSmallGemvOuterOffsets(context.offset, context.offsetI32, interleaveOffsets, scalarToV);
    Duplicate(context.outReal, 0.0f, Cher2kSmallGemvOuterContext::tileElements);
    Duplicate(context.outImag, 0.0f, Cher2kSmallGemvOuterContext::tileElements);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    Cher2kSmallGemvOuterCompute(
        ar, ai, br, bi, context.outReal, context.outImag, context.tmp0, context.tmp1, context.coeffBr, context.coeffBi,
        context.coeffAr, context.coeffAi, colBase, k, nPad);
    Cher2kSmallGemvOuterStore(
        cG, tiling, context.outReal, context.outImag, context.aos, context.offset, colBase, validCols, vToScalar,
        scalarToV, mte2ToV, vToMte3, mte3ToV);
    return true;
}

struct Cher2kSmallGemvContext : Cher2kColumnStorage {
    static constexpr uint32_t maxRows = 128U;
    static constexpr uint32_t maxK = 32U;
    static constexpr uint32_t maxMatrixComplex = maxRows * maxK;
    TBuf<TPosition::VECCALC> outerBatchBuf;
    LocalTensor<float> outerStorage;

    __aicore__ inline void Init(TPipe& pipe)
    {
        Allocate(pipe, maxMatrixComplex, maxRows);
        constexpr uint32_t outerTile = 8U * maxRows;
        constexpr uint32_t outerBytes = (4U * outerTile + 4U * 64U + 2U * outerTile + 2U * outerTile) * sizeof(float);
        pipe.InitBuffer(outerBatchBuf, outerBytes);
        Bind(maxMatrixComplex, maxRows);
        outerStorage = outerBatchBuf.Get<float>();
    }
};

struct Cher2kSmallGemvRuntime {
    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> cG;
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToScalar;
    event_t scalarToV;
    event_t vToMte3;
    event_t mte3ToV;
    uint32_t n = 0U;
    uint32_t k = 0U;
    uint32_t nPad = 0U;
    uint32_t matrixComplex = 0U;
    uint32_t block = 0U;
    uint32_t blocks = 1U;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    uint64_t reserved = 0ULL;
    bool unitAlphaNoBeta = false;
    bool upper = false;

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta, const Cher2kTilingData& tiling)
    {
        n = tiling.n;
        k = tiling.k;
        nPad = (n + 63U) & ~63U;
        matrixComplex = nPad * k;
        block = static_cast<uint32_t>(GetBlockIdx());
        blocks = static_cast<uint32_t>(GetBlockNum());
        if (blocks == 0U) {
            blocks = 1U;
        }
        aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * k * 2ULL);
        bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * k * 2ULL);
        cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * n * 2ULL);
        vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
        unitAlphaNoBeta = alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f;
        upper = tiling.uplo == ACLBLAS_UPPER;
    }
};

__aicore__ inline void Cher2kSmallGemvLoadInputs(
    Cher2kSmallGemvContext& context, Cher2kSmallGemvRuntime& runtime, const Cher2kTilingData& tiling)
{
    Duplicate(context.aAos, 0.0f, runtime.matrixComplex * 2U);
    Duplicate(context.bAos, 0.0f, runtime.matrixComplex * 2U);
    SetFlag<HardEvent::V_MTE2>(runtime.vToMte2);
    WaitFlag<HardEvent::V_MTE2>(runtime.vToMte2);
    const uint32_t blockLen = runtime.n * 2U * sizeof(float);
    const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
    const uint32_t dstStride = (runtime.nPad * 2U * sizeof(float) - alignedBlockLen) / 32U;
    const DataCopyExtParams aCopy{
        static_cast<uint16_t>(runtime.k), blockLen,
        static_cast<uint32_t>((tiling.lda - runtime.n) * 2U * sizeof(float)), dstStride, 0U};
    const DataCopyExtParams bCopyParams{
        static_cast<uint16_t>(runtime.k), blockLen,
        static_cast<uint32_t>((tiling.ldb - runtime.n) * 2U * sizeof(float)), dstStride, 0U};
    const uint32_t rightPadding = (alignedBlockLen - blockLen) / sizeof(float);
    const DataCopyPadExtParams<float> pad{rightPadding != 0U, 0U, static_cast<uint8_t>(rightPadding), 0.0f};
    DataCopyPad(context.aAos, runtime.aG, aCopy, pad);
    DataCopyPad(context.bAos, runtime.bG, bCopyParams, pad);
    SetFlag<HardEvent::MTE2_V>(runtime.mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(runtime.mte2ToV);
    const uint16_t repeats = static_cast<uint16_t>(runtime.matrixComplex * 2U / 64U);
    GatherMask<float>(context.ar, context.aAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, runtime.reserved);
    GatherMask<float>(context.ai, context.aAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, runtime.reserved);
    GatherMask<float>(context.br, context.bAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, runtime.reserved);
    GatherMask<float>(context.bi, context.bAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, runtime.reserved);
    PipeBarrier<PIPE_V>();
}

// Use extended-range accumulation when the FP32 product/sum bound can overflow.
// Ordinary finite input ranges continue through the existing vector path.
__aicore__ inline bool Cher2kSmallGemvNeedsWide(Cher2kSmallGemvContext& context, Cher2kSmallGemvRuntime& runtime)
{
    const uint32_t count = runtime.matrixComplex;
    AscendC::Abs(context.aAos, context.ar, count);
    AscendC::Abs(context.aAos[count], context.ai, count);
    AscendC::Abs(context.aAos[count * 2U], context.br, count);
    AscendC::Abs(context.aAos[count * 3U], context.bi, count);
    PipeBarrier<PIPE_V>();
    // VALUE_INDEX reduction scratch needs two floats per 64 input floats.
    // The outer-product scratch is unused until the range check completes.
    AscendC::ReduceMax(context.outR, context.aAos, context.outerStorage, count * 4U, false);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(runtime.vToScalar);
    WaitFlag<HardEvent::V_S>(runtime.vToScalar);
    const float maximum = context.outR.GetValue(0U);
    const float alphaMagnitude = __builtin_fabsf(runtime.alphaReal) + __builtin_fabsf(runtime.alphaImag);
    const float bound = maximum * maximum * (4.0F * runtime.k * alphaMagnitude);
    return !__builtin_isfinite(maximum) || !__builtin_isfinite(bound);
}

__aicore__ inline scaled_scalar::Complex Cher2kSmallGemvWideAccumulate(
    Cher2kSmallGemvContext& context, const Cher2kSmallGemvRuntime& runtime, uint32_t row, uint32_t col,
    scaled_scalar::Complex sum)
{
    for (uint32_t layer = 0U; layer < runtime.k && (runtime.alphaReal != 0.0F || runtime.alphaImag != 0.0F); ++layer) {
        const uint32_t r = layer * runtime.nPad + row;
        const uint32_t c = layer * runtime.nPad + col;
        const scaled_scalar::Complex aColumn{
            scaled_scalar::FromFloat(context.ar.GetValue(c)), scaled_scalar::FromFloat(-context.ai.GetValue(c))};
        const scaled_scalar::Complex bColumn{
            scaled_scalar::FromFloat(context.br.GetValue(c)), scaled_scalar::FromFloat(-context.bi.GetValue(c))};
        const auto firstCoefficient = scaled_scalar::ScaleComplex(bColumn, runtime.alphaReal, runtime.alphaImag);
        const auto secondCoefficient = scaled_scalar::ScaleComplex(aColumn, runtime.alphaReal, -runtime.alphaImag);
        const auto first =
            scaled_scalar::ScaleComplex(firstCoefficient, context.ar.GetValue(r), context.ai.GetValue(r));
        const auto second =
            scaled_scalar::ScaleComplex(secondCoefficient, context.br.GetValue(r), context.bi.GetValue(r));
        if (row == col) {
            sum.real = scaled_scalar::Add(sum.real, scaled_scalar::Add(first.real, second.real));
        } else {
            sum = scaled_scalar::AddComplex(scaled_scalar::AddComplex(sum, first), second);
        }
    }
    return sum;
}

struct Cher2kColumnRange {
    uint32_t begin;
    uint32_t end;
};
__aicore__ inline Cher2kColumnRange Cher2kSmallGemvColumnRange(
    const Cher2kSmallGemvRuntime& runtime, const Cher2kTilingData& tiling)
{
    const uint32_t remainder = tiling.ldc & 3U;
    const uint32_t columnGroup = remainder == 0U ? 1U : (remainder == 2U ? 2U : 4U);
    const uint32_t groupCount = (runtime.n + columnGroup - 1U) / columnGroup;
    const uint32_t groupsPerBlock = (groupCount + runtime.blocks - 1U) / runtime.blocks;
    const uint32_t begin = runtime.block * groupsPerBlock * columnGroup;
    return {begin, min(runtime.n, begin + groupsPerBlock * columnGroup)};
}

__aicore__ inline void Cher2kSmallGemvWideColumns(
    Cher2kSmallGemvContext& context, Cher2kSmallGemvRuntime& runtime, const Cher2kTilingData& tiling)
{
    __gm__ float* c = const_cast<__gm__ float*>(runtime.cG.GetPhyAddr());
    const auto range = Cher2kSmallGemvColumnRange(runtime, tiling);
    for (uint32_t col = range.begin; col < range.end; ++col) {
        const uint32_t begin = runtime.upper ? 0U : col;
        const uint32_t end = runtime.upper ? col + 1U : runtime.n;
        for (uint32_t row = begin; row < end; ++row) {
            scaled_scalar::Complex sum;
            const uint64_t index = (static_cast<uint64_t>(col) * tiling.ldc + row) * 2ULL;
            if (runtime.betaValue != 0.0F) {
                sum.real = scaled_scalar::Add(sum.real, scaled_scalar::Product(runtime.betaValue, c[index]));
                if (row != col) {
                    sum.imag = scaled_scalar::Add(sum.imag, scaled_scalar::Product(runtime.betaValue, c[index + 1U]));
                }
            }
            sum = Cher2kSmallGemvWideAccumulate(context, runtime, row, col, sum);
            c[index] = scaled_scalar::ToFloat(sum.real);
            c[index + 1U] = row == col ? 0.0F : scaled_scalar::ToFloat(sum.imag);
        }
    }
}

__aicore__ inline void Cher2kSmallGemvRunColumns(
    Cher2kSmallGemvContext& context, Cher2kSmallGemvRuntime& runtime, const Cher2kTilingData& tiling)
{
    for (uint32_t index = 0U; index < runtime.nPad * 2U; ++index) {
        const uint32_t source = (index & 1U) == 0U ? index / 2U : Cher2kSmallGemvContext::maxRows + index / 2U;
        context.interleaveOffset.SetValue(index, source * sizeof(float));
    }
    SetFlag<HardEvent::S_V>(runtime.scalarToV);
    WaitFlag<HardEvent::S_V>(runtime.scalarToV);
    const auto range = Cher2kSmallGemvColumnRange(runtime, tiling);
    for (uint32_t col = range.begin; col < range.end; ++col) {
        Cher2kTinySquareProcessColumn(
            runtime.cG, tiling, context.ar, context.ai, context.br, context.bi, context.p1r, context.p1i, context.p2r,
            context.p2i, context.outR, context.outI, context.interleaveOffset, context.outputAos, context.aAos,
            runtime.n, runtime.k, runtime.nPad, col, runtime.upper, runtime.unitAlphaNoBeta, runtime.alphaReal,
            runtime.alphaImag, runtime.betaValue, runtime.reserved, runtime.vToMte2, runtime.mte2ToV, runtime.vToScalar,
            runtime.scalarToV, runtime.vToMte3, runtime.mte3ToV);
    }
}

extern "C" __global__ __aicore__ void cher2k_small_gemv_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta, GM_ADDR interleaveOffsets,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta)) {
        return;
    }
    TPipe pipe;
    Cher2kSmallGemvContext context;
    context.Init(pipe);
    Cher2kSmallGemvRuntime runtime;
    runtime.Init(a, b, c, alpha, beta, tiling);
    if (runtime.n == 0U || runtime.n > Cher2kSmallGemvContext::maxRows || runtime.k == 0U ||
        runtime.k > Cher2kSmallGemvContext::maxK) {
        return;
    }
    Cher2kSmallGemvLoadInputs(context, runtime, tiling);
    if (Cher2kSmallGemvNeedsWide(context, runtime)) {
        Cher2kSmallGemvWideColumns(context, runtime, tiling);
        return;
    }
    SetFlag<HardEvent::V_S>(runtime.vToScalar);
    WaitFlag<HardEvent::V_S>(runtime.vToScalar);
    const bool handledOuter = Cher2kSmallGemvOuterBatch(
        runtime.cG, tiling, context.ar, context.ai, context.br, context.bi, context.outerStorage, interleaveOffsets,
        runtime.n, runtime.k, runtime.nPad, runtime.block, runtime.unitAlphaNoBeta, runtime.upper, runtime.scalarToV,
        runtime.mte2ToV, runtime.vToScalar, runtime.vToMte3, runtime.mte3ToV);
    if (handledOuter) {
        return;
    }
    Cher2kSmallGemvRunColumns(context, runtime, tiling);
}
struct Cher2kK2Context : Cher2kOperandViews {
    TBuf<TPosition::VECCALC> inputBuf;
    TBuf<TPosition::VECCALC> planarBuf;
    TBuf<TPosition::VECCALC> accumBuf;
    TBuf<TPosition::VECCALC> ioBuf;
    LocalTensor<float> coeff0;
    LocalTensor<float> coeff1;
    LocalTensor<float> coeff2;
    LocalTensor<float> coeff3;
    LocalTensor<float> outputAos;
    LocalTensor<float> oldAos;
    LocalTensor<uint32_t> offsets;
    __aicore__ inline void Init(TPipe& pipe)
    {
        constexpr uint32_t rows = 128U;
        constexpr uint32_t rank = 2U;
        constexpr uint32_t cols = 8U;
        constexpr uint32_t panel = rows * cols;
        pipe.InitBuffer(inputBuf, 2U * rank * rows * 2U * sizeof(float));
        pipe.InitBuffer(planarBuf, 4U * rank * rows * sizeof(float));
        pipe.InitBuffer(accumBuf, 6U * panel * sizeof(float));
        pipe.InitBuffer(ioBuf, (4U * 64U + CHER2K_SMALL_INTERLEAVE_COUNT + 4U * panel) * sizeof(float));
        BindOperands(inputBuf.Get<float>(), planarBuf.Get<float>(), accumBuf.Get<float>(), rank * rows, panel);
        coeff0 = ioBuf.Get<float>();
        coeff1 = coeff0[64U];
        coeff2 = coeff1[64U];
        coeff3 = coeff2[64U];
        offsets = ioBuf.Get<float>()[4U * 64U].ReinterpretCast<uint32_t>();
        outputAos = ioBuf.Get<float>()[4U * 64U + CHER2K_SMALL_INTERLEAVE_COUNT];
        oldAos = outputAos[panel * 2U];
    }
};
__aicore__ inline void Cher2kK2AccumulateRow(
    LocalTensor<float>& ar, LocalTensor<float>& ai, LocalTensor<float>& br, LocalTensor<float>& bi,
    LocalTensor<float>& p1r, LocalTensor<float>& p1i, LocalTensor<float>& p2r, LocalTensor<float>& p2i,
    LocalTensor<float>& outR, LocalTensor<float>& outI, LocalTensor<float>& coeff0, LocalTensor<float>& coeff1,
    LocalTensor<float>& coeff2, LocalTensor<float>& coeff3, uint32_t rowSrc, uint32_t rowBase, uint32_t coefficientLane,
    const BinaryRepeatParams& mul, const BinaryRepeatParams& add)
{
    constexpr uint8_t repeats = 8U;
    Mul(outR[rowBase], ar[rowSrc], coeff0[coefficientLane], 64U, repeats, mul);
    Mul(outI[rowBase], ai[rowSrc], coeff1[coefficientLane], 64U, repeats, mul);
    PipeBarrier<PIPE_V>();
    Add(outR[rowBase], outR[rowBase], outI[rowBase], 64U, repeats, add);
    PipeBarrier<PIPE_V>();
    Add(p1r[rowBase], p1r[rowBase], outR[rowBase], 64U, repeats, add);
    Mul(outR[rowBase], ai[rowSrc], coeff0[coefficientLane], 64U, repeats, mul);
    Mul(outI[rowBase], ar[rowSrc], coeff1[coefficientLane], 64U, repeats, mul);
    PipeBarrier<PIPE_V>();
    Sub(outR[rowBase], outR[rowBase], outI[rowBase], 64U, repeats, add);
    PipeBarrier<PIPE_V>();
    Add(p1i[rowBase], p1i[rowBase], outR[rowBase], 64U, repeats, add);
    Mul(outR[rowBase], br[rowSrc], coeff2[coefficientLane], 64U, repeats, mul);
    Mul(outI[rowBase], bi[rowSrc], coeff3[coefficientLane], 64U, repeats, mul);
    PipeBarrier<PIPE_V>();
    Add(outR[rowBase], outR[rowBase], outI[rowBase], 64U, repeats, add);
    PipeBarrier<PIPE_V>();
    Add(p2r[rowBase], p2r[rowBase], outR[rowBase], 64U, repeats, add);
    Mul(outR[rowBase], bi[rowSrc], coeff2[coefficientLane], 64U, repeats, mul);
    Mul(outI[rowBase], br[rowSrc], coeff3[coefficientLane], 64U, repeats, mul);
    PipeBarrier<PIPE_V>();
    Sub(outR[rowBase], outR[rowBase], outI[rowBase], 64U, repeats, add);
    PipeBarrier<PIPE_V>();
    Add(p2i[rowBase], p2i[rowBase], outR[rowBase], 64U, repeats, add);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kK2Accumulate(Cher2kK2Context& context, uint32_t colBase, uint32_t panelElements)
{
    constexpr uint32_t rows = 128U;
    const uint32_t alignedColBase = colBase & ~7U;
    const uint32_t coefficientLane = (colBase - alignedColBase) * 8U;
    const BinaryRepeatParams mul{1U, 1U, 0U, rows / 8U, 0U, 1U};
    const BinaryRepeatParams add{1U, 1U, 1U, rows / 8U, rows / 8U, rows / 8U};
    for (uint32_t layer = 0U; layer < 2U; ++layer) {
        const uint32_t src = layer * rows;
        Brcb(context.coeff0, context.br[src + alignedColBase], 1U, {1U, 8U});
        Brcb(context.coeff1, context.bi[src + alignedColBase], 1U, {1U, 8U});
        Brcb(context.coeff2, context.ar[src + alignedColBase], 1U, {1U, 8U});
        Brcb(context.coeff3, context.ai[src + alignedColBase], 1U, {1U, 8U});
        PipeBarrier<PIPE_V>();
        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += 64U) {
            Cher2kK2AccumulateRow(
                context.ar, context.ai, context.br, context.bi, context.p1r, context.p1i, context.p2r, context.p2i,
                context.outR, context.outI, context.coeff0, context.coeff1, context.coeff2, context.coeff3,
                src + rowBase, rowBase, coefficientLane, mul, add);
        }
    }
    (void)panelElements;
}

__aicore__ inline void Cher2kK2ApplyScalars(
    Cher2kK2Context& context, uint32_t panelElements, float alphaReal, float alphaImag, float betaValue,
    uint64_t& reserved, event_t mte2ToV)
{
    auto& p1r = context.p1r;
    auto& p1i = context.p1i;
    auto& p2r = context.p2r;
    auto& p2i = context.p2i;
    auto& outR = context.outR;
    auto& outI = context.outI;
    if (alphaReal == 1.0f && alphaImag == 0.0f) {
        Add(outR, p1r, p2r, panelElements);
        Add(outI, p1i, p2i, panelElements);
    } else {
        Muls(outR, p1r, alphaReal, panelElements);
        Muls(outI, p1i, alphaReal, panelElements);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p1i, -alphaImag, panelElements);
        Axpy(outI, p1r, alphaImag, panelElements);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p2r, alphaReal, panelElements);
        Axpy(outI, p2i, alphaReal, panelElements);
        PipeBarrier<PIPE_V>();
        Axpy(outR, p2i, alphaImag, panelElements);
        Axpy(outI, p2r, -alphaImag, panelElements);
    }
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    if (betaValue == 0.0f) {
        return;
    }
    constexpr uint16_t oldRepeats = 128U * 8U * 2U / 64U;
    GatherMask<float>(p1r, context.oldAos, 1U, false, 0U, {1U, oldRepeats, 8U, 8U}, reserved);
    GatherMask<float>(p1i, context.oldAos, 2U, false, 0U, {1U, oldRepeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
    Axpy(outR, p1r, betaValue, panelElements);
    Axpy(outI, p1i, betaValue, panelElements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kK2Store(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, Cher2kK2Context& context, uint32_t colBase,
    uint32_t panelElements, event_t vToMte3, event_t mte3ToV, event_t vToScalar, event_t scalarToV)
{
    constexpr uint32_t rows = 128U;
    constexpr uint32_t panelCols = 8U;
    auto& outI = context.outI;
    auto& outputAos = context.outputAos;
    auto& outR = context.outR;
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    for (uint32_t col = 0U; col < panelCols; ++col) {
        outI.SetValue(col * rows + colBase + col, 0.0f);
    }
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
    Gather(outputAos, outR, context.offsets, 0U, panelElements * 2U);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    for (uint32_t col = 0U; col < panelCols; ++col) {
        const uint32_t absoluteCol = colBase + col;
        const uint64_t offset = static_cast<uint64_t>(absoluteCol) * tiling.ldc * 2ULL;
        const DataCopyExtParams outputCopy{1U, (absoluteCol + 1U) * 8U, 0U, 0U, 0U};
        DataCopyPad(cG[offset], outputAos[col * rows * 2U], outputCopy);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

// Fixed-rank OP_N path for the n=120, K=2 performance bucket.  This is kept
// separate from the general small GEMV kernel so the compiler sees fixed loop
// bounds and a compact UB layout instead of the maxRows x maxK workspace.
__aicore__ inline void Cher2kK2Finalize(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta, Cher2kK2Context& context,
    uint32_t colBase, uint32_t panelElements, event_t mte2ToV, event_t vToMte3, event_t mte3ToV, event_t vToScalar,
    event_t scalarToV)
{
    uint64_t reserved = 0ULL;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    Cher2kK2ApplyScalars(context, panelElements, alphaReal, alphaImag, betaValue, reserved, mte2ToV);
    Cher2kK2Store(cG, tiling, context, colBase, panelElements, vToMte3, mte3ToV, vToScalar, scalarToV);
}

struct Cher2kK2Runtime {
    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> cG;
    event_t vToMte2;
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToV;
    event_t vToScalar;
    event_t scalarToV;
    uint32_t n = 0U;
    uint32_t colBase = 0U;
    uint32_t inputBytes = 0U;
    uint64_t cOffset = 0ULL;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const Cher2kTilingData& tiling)
    {
        constexpr uint32_t rank = 2U;
        constexpr uint32_t panelCols = 8U;
        n = tiling.n;
        colBase = static_cast<uint32_t>(GetBlockIdx()) * panelCols;
        inputBytes = n * 2U * sizeof(float);
        cOffset = static_cast<uint64_t>(colBase) * tiling.ldc * 2ULL;
        aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * rank * 2ULL);
        bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * rank * 2ULL);
        cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * n * 2ULL);
        vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }
};

__aicore__ inline void Cher2kK2LoadInputs(
    Cher2kK2Context& context, Cher2kK2Runtime& runtime, const Cher2kTilingData& tiling)
{
    constexpr uint32_t rows = 128U;
    constexpr uint32_t rank = 2U;
    Duplicate(context.aAos, 0.0f, rank * rows * 2U);
    Duplicate(context.bAos, 0.0f, rank * rows * 2U);
    SetFlag<HardEvent::V_MTE2>(runtime.vToMte2);
    WaitFlag<HardEvent::V_MTE2>(runtime.vToMte2);
    const uint32_t dstStride = (rows * 2U * sizeof(float) - runtime.inputBytes) / 32U;
    const DataCopyExtParams inputCopy{
        rank, runtime.inputBytes, static_cast<uint32_t>((tiling.lda - runtime.n) * 2U * sizeof(float)), dstStride, 0U};
    const DataCopyExtParams inputCopyB{
        rank, runtime.inputBytes, static_cast<uint32_t>((tiling.ldb - runtime.n) * 2U * sizeof(float)), dstStride, 0U};
    const DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0f};
    DataCopyPad(context.aAos, runtime.aG, inputCopy, noPad);
    DataCopyPad(context.bAos, runtime.bG, inputCopyB, noPad);
    SetFlag<HardEvent::MTE2_V>(runtime.mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(runtime.mte2ToV);
}

__aicore__ inline void Cher2kK2SplitInputs(Cher2kK2Context& context, GM_ADDR interleaveOffsets)
{
    constexpr uint32_t rows = 128U;
    constexpr uint32_t rank = 2U;
    constexpr uint16_t repeats = rank * rows * 2U / 64U;
    uint64_t reserved = 0ULL;
    GatherMask<float>(context.ar, context.aAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(context.ai, context.aAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(context.br, context.bAos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(context.bi, context.bAos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GlobalTensor<uint32_t> offsetsG;
    offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleaveOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
    DataCopy(context.offsets, offsetsG[CHER2K_SMALL_INTERLEAVE_OFFSET], CHER2K_SMALL_INTERLEAVE_COUNT);
}

__aicore__ inline void Cher2kK2PrepareOutput(
    Cher2kK2Context& context, Cher2kK2Runtime& runtime, const Cher2kTilingData& tiling)
{
    constexpr uint32_t rows = 128U;
    constexpr uint32_t panelCols = 8U;
    triangle_store::LoadComplexSelected<rows>(
        context.oldAos, runtime.cG, tiling.ldc, 0U, runtime.colBase, runtime.n, panelCols, tiling.uplo == ACLBLAS_UPPER,
        runtime.vToMte2);
    SetFlag<HardEvent::MTE2_V>(runtime.mte2ToV);
    constexpr uint32_t panelElements = rows * panelCols;
    Duplicate(context.p1r, 0.0f, panelElements);
    Duplicate(context.p1i, 0.0f, panelElements);
    Duplicate(context.p2r, 0.0f, panelElements);
    Duplicate(context.p2i, 0.0f, panelElements);
    PipeBarrier<PIPE_V>();
}

extern "C" __global__ __aicore__ void cher2k_k2_outer_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR alpha, GM_ADDR beta, GM_ADDR interleaveOffsets,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta)) {
        return;
    }
    constexpr uint32_t rows = 128U;
    constexpr uint32_t rank = 2U;
    constexpr uint32_t panelCols = 8U;
    constexpr uint32_t panelElements = rows * panelCols;
    TPipe pipe;
    Cher2kK2Context context;
    context.Init(pipe);
    Cher2kK2Runtime runtime;
    runtime.Init(a, b, c, tiling);
    if (runtime.n != 120U || tiling.k != rank || runtime.colBase >= runtime.n) {
        return;
    }
    Cher2kK2LoadInputs(context, runtime, tiling);
    Cher2kK2SplitInputs(context, interleaveOffsets);
    Cher2kK2PrepareOutput(context, runtime, tiling);
    Cher2kK2Accumulate(context, runtime.colBase, panelElements);
    Cher2kK2Finalize(
        runtime.cG, tiling, alpha, beta, context, runtime.colBase, panelElements, runtime.mte2ToV, runtime.vToMte3,
        runtime.mte3ToV, runtime.vToScalar, runtime.scalarToV);
}
// Keep GM pointers before the by-value tiling aggregate in device entry points.
// Its 84-byte size must not affect the alignment of subsequent pointer arguments.
inline uint32_t Cher2kPostprocessBlockCount(const Cher2kTilingData& tiling)
{
    constexpr uint32_t postprocessTileSize = 64U;
    const uint32_t postprocessTileAxis = (tiling.n + postprocessTileSize - 1U) / postprocessTileSize;
    const uint64_t postprocessTileCount =
        CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT ?
            (static_cast<uint64_t>(postprocessTileAxis) * (postprocessTileAxis + 1U)) / 2ULL :
            static_cast<uint64_t>(postprocessTileAxis) * postprocessTileAxis;
    return postprocessTileCount < tiling.aivCoreNum ?
               (postprocessTileCount == 0U ? 1U : static_cast<uint32_t>(postprocessTileCount)) :
               tiling.aivCoreNum;
}

inline uint32_t Cher2kSmallPathBlockCount(const Cher2kTilingData& tiling)
{
    const uint32_t ldcRemainder = tiling.ldc & 3U;
    const uint32_t columnGroup = ldcRemainder == 0U ? 1U : (ldcRemainder == 2U ? 2U : 4U);
    const uint32_t columnGroupCount = (tiling.n + columnGroup - 1U) / columnGroup;
    const bool outerBatchCandidate =
        tiling.trans == ACLBLAS_OP_N && tiling.n > 64U && tiling.n <= 128U && tiling.k > 0U && tiling.k <= 4U;
    const uint32_t requestedBlocks = outerBatchCandidate ? (tiling.n + 7U) / 8U : columnGroupCount;
    return requestedBlocks < tiling.aivCoreNum ? requestedBlocks : tiling.aivCoreNum;
}

inline void LaunchCher2kSmallPath(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const Cher2kTilingData& tiling, void* stream, GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR interleaveOffsets)
{
    const uint32_t smallBlocks = Cher2kSmallPathBlockCount(tiling);
    if (tiling.trans == ACLBLAS_OP_N && tiling.n <= 128U && tiling.k > 0U && tiling.k <= 32U) {
        cher2k_small_gemv_kernel<<<smallBlocks, nullptr, stream>>>(a, b, c, alpha, beta, interleaveOffsets, tiling);
    } else {
        cher2k_small_fused_kernel<<<smallBlocks, nullptr, stream>>>(a, b, c, alpha, beta, tiling);
    }
}

inline void LaunchCher2kSmallCubePath(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream,
    bool skipPreprocess, GM_ADDR alpha, GM_ADDR beta, GM_ADDR interleaveOffsets)
{
    if (tiling.computeProduct != 0U && !skipPreprocess) {
        const uint32_t safeNAligned = tiling.nAligned == 0U ? 1U : tiling.nAligned;
        const uint32_t rowsPerBatch =
            safeNAligned <= CHER2K_ROW_BATCH_ELEMENTS ? CHER2K_ROW_BATCH_ELEMENTS / safeNAligned : 1U;
        const uint32_t rowBatchCount = (tiling.kAligned + rowsPerBatch - 1U) / rowsPerBatch;
        const uint32_t preprocessTasks = rowBatchCount * 2U;
        const uint32_t preprocessBlocks = tiling.aivCoreNum < preprocessTasks ? tiling.aivCoreNum : preprocessTasks;
        cher2k_preprocess_normal_kernel<<<preprocessBlocks == 0U ? 1U : preprocessBlocks, nullptr, stream>>>(
            a, b, workspace, nullptr, nullptr, CHER2K_SCALAR_DISPATCH_ALL, tiling);
    }
    const uint32_t smallMtiles = (tiling.nAligned + 63U) / 64U;
    const uint64_t smallTileCount = static_cast<uint64_t>(smallMtiles) * smallMtiles;
    const uint32_t smallCubeBlocks = smallTileCount < tiling.aicCoreNum ?
                                         (smallTileCount == 0ULL ? 1U : static_cast<uint32_t>(smallTileCount)) :
                                         tiling.aicCoreNum;
    cher2k_small_cube_kernel<<<smallCubeBlocks, nullptr, stream>>>(workspace, tiling);
    const uint32_t postBlocks = smallTileCount < tiling.aivCoreNum ?
                                    (smallTileCount == 0ULL ? 1U : static_cast<uint32_t>(smallTileCount)) :
                                    tiling.aivCoreNum;
    cher2k_postprocess_kernel<<<postBlocks, nullptr, stream>>>(
        c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
}

inline uint32_t Cher2kNormalPreprocessBlockCount(const Cher2kTilingData& tiling)
{
    const uint32_t safeNAligned = tiling.nAligned == 0U ? 1U : tiling.nAligned;
    const uint32_t rowsPerBatch =
        safeNAligned <= CHER2K_ROW_BATCH_ELEMENTS ? CHER2K_ROW_BATCH_ELEMENTS / safeNAligned : 1U;
    const uint32_t preprocessTasks = ((tiling.kAligned + rowsPerBatch - 1U) / rowsPerBatch) * 2U;
    return tiling.aivCoreNum < preprocessTasks ? tiling.aivCoreNum : preprocessTasks;
}

inline uint32_t Cher2kDirectPreprocessBlockCount(const Cher2kTilingData& tiling)
{
    const uint32_t directPreprocessTasks = ((tiling.kAligned + CHER2K_TRANSPOSE_ROWS - 1U) / CHER2K_TRANSPOSE_ROWS) *
                                           ((tiling.nAligned + CHER2K_TRANSPOSE_COLS - 1U) / CHER2K_TRANSPOSE_COLS);
    if (tiling.directHermitianPath == 0U) {
        return tiling.aivCoreNum;
    }
    return directPreprocessTasks < tiling.aivCoreNum ? directPreprocessTasks : tiling.aivCoreNum;
}

inline uint32_t Cher2kSg3ChunkBlockCount(const Cher2kTilingData& tiling)
{
    const uint32_t rowTiles = (tiling.n + 63U) / 64U;
    uint64_t taskCount = 0ULL;
    for (uint32_t rowTile = 0U; rowTile < rowTiles; ++rowTile) {
        const uint32_t rowBase = rowTile * 64U;
        const uint32_t colBegin = tiling.uplo == ACLBLAS_UPPER ? rowBase : 0U;
        const uint32_t colEnd =
            tiling.uplo == ACLBLAS_UPPER ? tiling.n : (tiling.n < rowBase + 64U ? tiling.n : rowBase + 64U);
        taskCount += (colEnd - colBegin + 31U) / 32U;
    }
    return taskCount < tiling.aivCoreNum ? static_cast<uint32_t>(taskCount) : tiling.aivCoreNum;
}

inline void LaunchCher2kPreprocess(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream, GM_ADDR alpha, GM_ADDR beta,
    GM_ADDR interleaveOffsets, bool skipPreprocess, bool directOpcMix)
{
    if (skipPreprocess || directOpcMix) {
        return;
    }
    if (tiling.trans == ACLBLAS_OP_N) {
        cher2k_preprocess_normal_kernel<<<Cher2kNormalPreprocessBlockCount(tiling), nullptr, stream>>>(
            a, b, workspace, nullptr, nullptr, CHER2K_SCALAR_DISPATCH_ALL, tiling);
        return;
    }
    cher2k_preprocess_kernel<<<Cher2kDirectPreprocessBlockCount(tiling), nullptr, stream>>>(
        a, b, workspace, interleaveOffsets, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
}

// The OP_C direct-output geometry can be served by the fused MIX producer,
// which publishes the unit-alpha Hermitian result itself and declines every
// non-unit call.  The packed graph queued behind it is restricted to non-unit
// scalars (CHER2K_SCALAR_DISPATCH_NON_UNIT), so the two launches cover
// complementary scalar domains and the real classification runs on device.
// Only when the values were resolved on the host do the tiling scalars have to
// be exactly unit for the fused producer to be selected.
inline bool IsCher2kDirectOpcMix(const Cher2kTilingData& tiling)
{
    if (tiling.directHermitianPath == 0U || tiling.directOutputPath == 0U || tiling.trans != ACLBLAS_OP_C ||
        tiling.uplo != ACLBLAS_UPPER) {
        return false;
    }
    return tiling.deviceScalars != 0U || (tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f && tiling.beta == 0.0f);
}

inline bool TryLaunchCher2kDeviceScalarDirectOpc(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream, GM_ADDR alpha,
    GM_ADDR beta, GM_ADDR interleaveOffsets, bool directOpcMix)
{
    if (!directOpcMix || alpha == nullptr || beta == nullptr) {
        return false;
    }
    cher2k_direct_opc_mix_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
        a, b, c, workspace, interleaveOffsets, alpha, beta, tiling);
    Cher2kTilingData packedTiling = tiling;
    packedTiling.directOutputPath = 0U;
    const uint32_t preprocessBlocks = Cher2kDirectPreprocessBlockCount(packedTiling);
    cher2k_preprocess_kernel<<<preprocessBlocks == 0U ? 1U : preprocessBlocks, nullptr, stream>>>(
        a, b, workspace, interleaveOffsets, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, packedTiling);
    cher2k_packed_opc_cube_kernel<<<packedTiling.aicCoreNum, nullptr, stream>>>(
        workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, packedTiling);
    const uint32_t chunkBlocks = Cher2kSg3ChunkBlockCount(packedTiling);
    cher2k_sg3_chunk_postprocess_kernel<<<chunkBlocks == 0U ? 1U : chunkBlocks, nullptr, stream>>>(
        c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, packedTiling);
    return true;
}

inline uint32_t Cher2kGeneralCubeBlockCount(const Cher2kTilingData& tiling)
{
    const bool matmulEligible = tiling.n == tiling.nAligned && tiling.k == tiling.kAligned && tiling.kAligned == 256U &&
                                tiling.nAligned == 256U && tiling.nAligned >= CHER2K_CUBE_M &&
                                (tiling.nAligned % CHER2K_CUBE_M) == 0U && (tiling.nAligned % CHER2K_CUBE_N) == 0U;
    const uint64_t matmulTasks = (static_cast<uint64_t>(tiling.nAligned) / CHER2K_CUBE_M) *
                                 (static_cast<uint64_t>(tiling.nAligned) / CHER2K_CUBE_N) *
                                 (tiling.useThreeM != 0U ? 3ULL : 4ULL);
    const bool oneTaskPerBlock = CHER2K_ONE_TASK_PER_BLOCK_EXPERIMENT && matmulTasks <= UINT32_MAX;
    const uint32_t cubeBlocks =
        oneTaskPerBlock ?
            static_cast<uint32_t>(matmulTasks) :
            (matmulEligible && matmulTasks <= UINT32_MAX ? static_cast<uint32_t>(matmulTasks) : tiling.aicCoreNum);
    return tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C ? tiling.aicCoreNum : cubeBlocks;
}

inline bool UsesCher2kPackedOpcCube(const Cher2kTilingData& tiling)
{
    return tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C && tiling.directOutputPath == 0U &&
           tiling.nAligned % 64U == 0U;
}

inline void LaunchCher2kCube(const Cher2kTilingData& tiling, GM_ADDR workspace, void* stream)
{
    if (tiling.sg3MmadPath != 0U) {
        cher2k_packed_opc_cube_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
            workspace, nullptr, nullptr, CHER2K_SCALAR_DISPATCH_ALL, tiling);
        return;
    }
    if (UsesCher2kPackedOpcCube(tiling)) {
        cher2k_packed_opc_cube_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
            workspace, nullptr, nullptr, CHER2K_SCALAR_DISPATCH_ALL, tiling);
        return;
    }
    const uint32_t blocks = Cher2kGeneralCubeBlockCount(tiling);
    cher2k_cube_kernel<<<blocks, nullptr, stream>>>(workspace, nullptr, nullptr, CHER2K_SCALAR_DISPATCH_ALL, tiling);
}

inline void LaunchCher2kEpilogue(
    GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream, GM_ADDR alpha, GM_ADDR beta,
    uint32_t postprocessBlocks)
{
    if (tiling.directHermitianPath == 0U) {
        cher2k_postprocess_kernel<<<postprocessBlocks, nullptr, stream>>>(
            c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
        return;
    }
    if (tiling.trans == ACLBLAS_OP_C && tiling.directOutputPath == 0U) {
        const uint32_t chunkBlocks = Cher2kSg3ChunkBlockCount(tiling);
        cher2k_sg3_chunk_postprocess_kernel<<<chunkBlocks == 0U ? 1U : chunkBlocks, nullptr, stream>>>(
            c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
        return;
    }
    cher2k_direct_hermitian_postprocess_kernel<<<postprocessBlocks, nullptr, stream>>>(
        c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
}

// A single K panel fits the direct-H signed-operand producer. It forms only
// the selected Hermitian tiles and publishes two planes instead of three
// full-square products. Require enough triangular tiles to use every AIC.
// Device scalars select mutually exclusive graphs on device; non-unit inputs
// retain the original three-product producer and scalar-aware epilogue.
inline bool TryLaunchCher2kDirectNormalPanel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream, GM_ADDR alpha,
    GM_ADDR beta)
{
    const uint64_t axis = tiling.nAligned / 128U;
    const uint64_t tasks = axis * (axis + 1ULL) / 2ULL;
    if (tiling.trans != ACLBLAS_OP_N || tiling.computeProduct == 0U || tiling.useThreeM == 0U ||
        tiling.kAligned > CHER2K_CUBE_K || tiling.nAligned % 128U != 0U || tasks < tiling.aicCoreNum) {
        return false;
    }
    const bool device = tiling.deviceScalars != 0U;
    if (!device && !(tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f && tiling.beta == 0.0f)) {
        return false;
    }
    Cher2kTilingData direct = tiling;
    direct.directHermitianPath = 1U;
    direct.directOutputPath = 1U;
    const uint32_t dispatch = device ? CHER2K_SCALAR_DISPATCH_UNIT : CHER2K_SCALAR_DISPATCH_ALL;
    const uint32_t preBlocks = Cher2kNormalPreprocessBlockCount(direct);
    const uint32_t postBlocks = Cher2kPostprocessBlockCount(direct);
    cher2k_preprocess_normal_kernel<<<preBlocks, nullptr, stream>>>(a, b, workspace, alpha, beta, dispatch, direct);
    cher2k_cube_kernel<<<direct.aicCoreNum, nullptr, stream>>>(workspace, alpha, beta, dispatch, direct);
    cher2k_direct_hermitian_postprocess_kernel<<<postBlocks, nullptr, stream>>>(
        c, workspace, alpha, beta, dispatch, direct);
    if (device) {
        cher2k_preprocess_normal_kernel<<<preBlocks, nullptr, stream>>>(
            a, b, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, tiling);
        cher2k_cube_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
            workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, tiling);
        cher2k_postprocess_kernel<<<postBlocks, nullptr, stream>>>(
            c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_NON_UNIT, tiling);
    }
    return true;
}

void cher2k_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream,
    bool skipPreprocess, GM_ADDR alpha, GM_ADDR beta, GM_ADDR interleaveOffsets)
{
    const uint32_t postprocessBlocks = Cher2kPostprocessBlockCount(tiling);
    // Tiny shapes avoid workspace and Cube launch overhead.
    if (tiling.smallPath == 1U) {
        LaunchCher2kSmallPath(a, b, c, tiling, stream, alpha, beta, interleaveOffsets);
        return;
    }
    if (tiling.smallPath == CHER2K_SMALL_CUBE_PATH) {
        LaunchCher2kSmallCubePath(a, b, c, workspace, tiling, stream, skipPreprocess, alpha, beta, interleaveOffsets);
        return;
    }
    if (TryLaunchCher2kDirectNormalPanel(a, b, c, workspace, tiling, stream, alpha, beta)) {
        return;
    }
    if (tiling.computeProduct != 0U) {
        const bool directOpcMix = IsCher2kDirectOpcMix(tiling);
        const bool launchedDirect = TryLaunchCher2kDeviceScalarDirectOpc(
            a, b, c, workspace, tiling, stream, alpha, beta, interleaveOffsets, directOpcMix);
        if (launchedDirect) {
            return;
        }
        LaunchCher2kPreprocess(
            a, b, workspace, tiling, stream, alpha, beta, interleaveOffsets, skipPreprocess, directOpcMix);
        if (directOpcMix) {
            cher2k_direct_opc_mix_kernel<<<tiling.aicCoreNum, nullptr, stream>>>(
                a, b, c, workspace, interleaveOffsets, alpha, beta, tiling);
            return;
        }
        LaunchCher2kCube(tiling, workspace, stream);
        if (tiling.sg3MmadPath != 0U) {
            cher2k_sg3_chunk_postprocess_kernel<<<Cher2kSg3ChunkBlockCount(tiling), nullptr, stream>>>(
                c, workspace, alpha, beta, CHER2K_SCALAR_DISPATCH_ALL, tiling);
            return;
        }
    }
    LaunchCher2kEpilogue(c, workspace, tiling, stream, alpha, beta, postprocessBlocks);
}
