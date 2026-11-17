/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

class Cher2kKernelPostprocess {
public:
    __aicore__ inline void Init(__gm__ float* base, const Cher2kTilingData& tiling)
    {
        const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
        const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
        sumA_.SetGlobalBuffer(base + 2ULL * nk, nk);
        sumB_.SetGlobalBuffer(base + 5ULL * nk, nk);
        ar_.SetGlobalBuffer(base, nk);
        br_.SetGlobalBuffer(base + 3ULL * nk, nk);
        ai_.SetGlobalBuffer(base + nk, nk);
        bi_.SetGlobalBuffer(base + 4ULL * nk, nk);
        sum_.SetGlobalBuffer(base + 6ULL * nk + 2ULL * nn, nn);
        rr_.SetGlobalBuffer(base + 6ULL * nk, nn);
        ii_.SetGlobalBuffer(base + 6ULL * nk + nn, nn);
        n_ = tiling.nAligned;
        k_ = tiling.kAligned;
        opC_ = tiling.trans == ACLBLAS_OP_C;
    }

    __aicore__ inline void Process(uint32_t firstTile, uint32_t tileStride)
    {
        constexpr uint32_t M = 128U;
        constexpr uint32_t N = 128U;
        constexpr uint32_t K = CHER2K_CUBE_K;
        constexpr uint32_t aCount = M * K;
        constexpr uint32_t bCount = K * N;
        constexpr uint32_t cCount = M * N;
        LocalTensor<float> a1{TPosition::A1, 0U, aCount};
        LocalTensor<float> b1{TPosition::B1, aCount * sizeof(float), bCount};
        LocalTensor<float> a2{TPosition::A2, 0U, aCount};
        LocalTensor<float> b2{TPosition::B2, 0U, bCount};
        LocalTensor<float> c1{TPosition::CO1, 0U, cCount};
        const uint32_t tilesPerDim = n_ / M;
        const uint64_t tileCount = static_cast<uint64_t>(tilesPerDim) * tilesPerDim;

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetHF32Mode(HF32Mode::DISABLE);

        for (uint64_t tile = firstTile; tile < tileCount; tile += tileStride) {
            const uint32_t mBase = static_cast<uint32_t>(tile / tilesPerDim) * M;
            const uint32_t nBase = static_cast<uint32_t>(tile % tilesPerDim) * N;
            Product(a1, b1, a2, b2, c1, sumA_, sumB_, sum_, mBase, nBase);
            Product(a1, b1, a2, b2, c1, ar_, br_, rr_, mBase, nBase);
            Product(a1, b1, a2, b2, c1, ai_, bi_, ii_, mBase, nBase);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

private:
    __aicore__ inline void CopyToL1(
        LocalTensor<float>& a1, LocalTensor<float>& b1, GlobalTensor<float>& left, GlobalTensor<float>& right,
        uint32_t mBase, uint32_t nBase, uint32_t kBase)
    {
        constexpr uint32_t M = 128U;
        constexpr uint32_t N = 128U;
        constexpr uint32_t K = CHER2K_CUBE_K;
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);

        Nd2NzParams aParams{};
        aParams.ndNum = 1;
        aParams.nValue = opC_ ? M : K;
        aParams.dValue = opC_ ? K : M;
        aParams.srcDValue = opC_ ? k_ : n_;
        aParams.dstNzC0Stride = opC_ ? M : K;
        aParams.dstNzNStride = 1;
        const uint64_t aOffset =
            opC_ ? static_cast<uint64_t>(mBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + mBase;
        DataCopy(a1, left[aOffset], aParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        Nd2NzParams bParams{};
        bParams.ndNum = 1;
        bParams.nValue = K;
        bParams.dValue = N;
        bParams.srcDValue = n_;
        bParams.dstNzC0Stride = K;
        bParams.dstNzNStride = 1;
        DataCopy(b1, right[static_cast<uint64_t>(kBase) * n_ + nBase], bParams);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
    }

    __aicore__ inline void LoadToL0(
        LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2)
    {
        constexpr uint32_t M = 128U;
        constexpr uint32_t N = 128U;
        constexpr uint32_t K = CHER2K_CUBE_K;
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        LoadData3DParamsV2<float> aParams{};
        aParams.l1H = 1U;
        aParams.l1W = opC_ ? M : K;
        aParams.channelSize = opC_ ? K : M;
        aParams.kExtension = opC_ ? K : M;
        aParams.mExtension = opC_ ? M : K;
        cube_load::InitUnitFilter(aParams);
        aParams.enTranspose = !opC_;
        aParams.fMatrixCtrl = false;
        LoadData(a2, a1, aParams);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);
        LoadData3DParamsV2<float> bParams{};
        bParams.l1H = 1U;
        bParams.l1W = K;
        bParams.channelSize = N;
        bParams.kExtension = N;
        bParams.mExtension = K;
        cube_load::InitUnitFilter(bParams);
        bParams.enTranspose = true;
        bParams.fMatrixCtrl = false;
        LoadData(b2, b1, bParams);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
    }

    __aicore__ inline void Compute(
        LocalTensor<float>& c1, LocalTensor<float>& a2, LocalTensor<float>& b2, bool initialize)
    {
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        MmadParams params{};
        params.m = 128U;
        params.n = 128U;
        params.k = CHER2K_CUBE_K;
        params.cmatrixInitVal = initialize;
        params.cmatrixSource = false;
        params.kDirectionAlign = false;
        Mmad(c1, a2, b2, params);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void Store(LocalTensor<float>& c1, GlobalTensor<float>& output, uint32_t mBase, uint32_t nBase)
    {
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 params{};
        params.mSize = 128U;
        params.nSize = 128U;
        params.srcStride = 128U;
        params.dstStride = n_;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(output[static_cast<uint64_t>(mBase) * n_ + nBase], c1, params);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void Product(
        LocalTensor<float>& a1, LocalTensor<float>& b1, LocalTensor<float>& a2, LocalTensor<float>& b2,
        LocalTensor<float>& c1, GlobalTensor<float>& left, GlobalTensor<float>& right, GlobalTensor<float>& output,
        uint32_t mBase, uint32_t nBase)
    {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (uint32_t kBase = 0U; kBase < k_; kBase += CHER2K_CUBE_K) {
            CopyToL1(a1, b1, left, right, mBase, nBase, kBase);
            LoadToL0(a1, b1, a2, b2);
            Compute(c1, a2, b2, kBase == 0U);
        }
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        Store(c1, output, mBase, nBase);
    }

    GlobalTensor<float> sumA_;
    GlobalTensor<float> sumB_;
    GlobalTensor<float> ar_;
    GlobalTensor<float> br_;
    GlobalTensor<float> ai_;
    GlobalTensor<float> bi_;
    GlobalTensor<float> sum_;
    GlobalTensor<float> rr_;
    GlobalTensor<float> ii_;
    uint32_t n_ = 0U;
    uint32_t k_ = 0U;
    bool opC_ = true;
};

extern "C" __global__ __aicore__ void cher2k_packed_opc_cube_kernel(
    GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }
    AscendC::InitSocState();
    __gm__ float* base = reinterpret_cast<__gm__ float*>(workspace);
    Cher2kKernelPostprocess mm;
    mm.Init(base, tiling);
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    mm.Process(static_cast<uint32_t>(GetBlockIdx()), blockNum);

    GlobalTensor<float> workspaceG;
    workspaceG.SetGlobalBuffer(base, Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned));
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(workspaceG);
    PipeBarrier<PIPE_ALL>();
}

// Chemm-compatible small geometry. One AIC block owns complete 64x64
// output tiles and runs the three 3M products through a single CO1 lifetime.
// This avoids padding n=64/128 into the 256x128 production geometry.
__aicore__ inline void Cher2kSg3LoadPlanes(
    GlobalTensor<float>& ws, LocalTensor<float>& first, LocalTensor<float>& second, LocalTensor<float>& temp,
    uint64_t p1Offset, uint64_t p2Offset, uint64_t p3Offset, const DataCopyExtParams& copy,
    const DataCopyPadExtParams<float>& pad, uint32_t elements, event_t mte2ToV, event_t vToMte2)
{
    DataCopyPad(first, ws[p1Offset], copy, pad);
    DataCopyPad(second, ws[p2Offset], copy, pad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    Add(temp, first, second, elements);
    PipeBarrier<PIPE_V>();
    Sub(first, first, second, elements);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
    DataCopyPad(second, ws[p3Offset], copy, pad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    Sub(second, second, temp, elements);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void Cher2kSg3LoadChunk(
    GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<uint32_t>& transposeOffset,
    LocalTensor<float>& realIj, LocalTensor<float>& imagOut, LocalTensor<float>& imagIj, LocalTensor<float>& realJi,
    LocalTensor<float>& imagJi, LocalTensor<float>& raw, LocalTensor<float>& temp, uint64_t p1Base, uint64_t p2Base,
    uint64_t p3Base, uint32_t rowBase, uint32_t colBase, event_t mte2ToV, event_t vToMte2)
{
    constexpr uint32_t rows = 64U;
    constexpr uint32_t cols = 32U;
    constexpr uint32_t elements = rows * cols;
    const uint64_t ijOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
    const uint64_t jiOffset = static_cast<uint64_t>(colBase) * tiling.nAligned + rowBase;
    const DataCopyExtParams ijCopy{
        static_cast<uint16_t>(rows), cols * sizeof(float),
        static_cast<uint32_t>((tiling.nAligned - cols) * sizeof(float)), 0U, 0U};
    const DataCopyExtParams jiCopy{
        static_cast<uint16_t>(cols), rows * sizeof(float),
        static_cast<uint32_t>((tiling.nAligned - rows) * sizeof(float)), 0U, 0U};
    const DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0f};
    Cher2kSg3LoadPlanes(
        ws, raw, imagIj, temp, p1Base + ijOffset, p2Base + ijOffset, p3Base + ijOffset, ijCopy, noPad, elements,
        mte2ToV, vToMte2);
    Gather(realIj, raw, transposeOffset, 0U, elements);
    Gather(imagOut, imagIj, transposeOffset, 0U, elements);
    PipeBarrier<PIPE_V>();
    Cher2kSg3LoadPlanes(
        ws, realJi, imagJi, temp, p1Base + jiOffset, p2Base + jiOffset, p3Base + jiOffset, jiCopy, noPad, elements,
        mte2ToV, vToMte2);
}

__aicore__ inline void Cher2kAddSelectedOldC(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& real, LocalTensor<float>& imag,
    uint32_t blockRows, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, float betaValue,
    event_t vToScalar, event_t scalarToV)
{
    if (betaValue == 0.0f) {
        return;
    }
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    for (uint32_t col = 0U; col < validCols; ++col) {
        for (uint32_t row = 0U; row < validRows; ++row) {
            const bool selected =
                tiling.uplo == ACLBLAS_UPPER ? rowBase + row <= colBase + col : rowBase + row >= colBase + col;
            if (!selected) {
                continue;
            }
            const uint32_t local = col * blockRows + row;
            const uint64_t cOffset = (static_cast<uint64_t>(colBase + col) * tiling.ldc + rowBase + row) * 2ULL;
            real.SetValue(local, real.GetValue(local) + betaValue * cG.GetValue(cOffset));
            imag.SetValue(local, imag.GetValue(local) + betaValue * cG.GetValue(cOffset + 1ULL));
        }
    }
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
}

__aicore__ inline void Cher2kSg3CombineChunk(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& realIj, LocalTensor<float>& imagOut,
    LocalTensor<float>& realJi, LocalTensor<float>& imagJi, LocalTensor<float>& temp, LocalTensor<float>& raw,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, bool unitAlphaNoBeta, float alphaReal,
    float alphaImag, float betaValue, event_t vToScalar, event_t scalarToV)
{
    constexpr uint32_t blockRows = 64U;
    constexpr uint32_t elements = blockRows * 32U;
    if (unitAlphaNoBeta) {
        Add(realIj, realIj, realJi, elements);
        Sub(imagOut, imagOut, imagJi, elements);
        PipeBarrier<PIPE_V>();
        return;
    }
    Add(temp, realIj, realJi, elements);
    Sub(realJi, realIj, realJi, elements);
    Add(raw, imagOut, imagJi, elements);
    Sub(imagJi, imagOut, imagJi, elements);
    PipeBarrier<PIPE_V>();
    Muls(realIj, temp, alphaReal, elements);
    Muls(raw, raw, alphaImag, elements);
    Muls(imagOut, imagJi, alphaReal, elements);
    Muls(temp, realJi, alphaImag, elements);
    PipeBarrier<PIPE_V>();
    Sub(realIj, realIj, raw, elements);
    Add(imagOut, imagOut, temp, elements);
    PipeBarrier<PIPE_V>();
    Cher2kAddSelectedOldC(
        cG, tiling, realIj, imagOut, blockRows, rowBase, colBase, validRows, validCols, betaValue, vToScalar,
        scalarToV);
}

__aicore__ inline void Cher2kSg3StoreChunk(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAos, LocalTensor<float>& imagOut,
    LocalTensor<float>& planar, LocalTensor<float>& indexBuffer, LocalTensor<uint32_t>& interleaveOffset,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, bool upper, event_t vToMte2,
    event_t vToMte3, event_t vToScalar, event_t scalarToV)
{
    constexpr uint32_t blockRows = 64U;
    constexpr uint32_t elements = blockRows * 32U;
    if (colBase < rowBase + validRows && colBase + validCols > rowBase) {
        SetFlag<HardEvent::V_S>(vToScalar);
        WaitFlag<HardEvent::V_S>(vToScalar);
        for (uint32_t col = 0U; col < validCols; ++col) {
            const uint32_t absoluteCol = colBase + col;
            if (absoluteCol >= rowBase && absoluteCol < rowBase + validRows) {
                imagOut.SetValue(col * blockRows + absoluteCol - rowBase, 0.0f);
            }
        }
        SetFlag<HardEvent::S_V>(scalarToV);
        WaitFlag<HardEvent::S_V>(scalarToV);
    }
    Gather(cAos, planar, interleaveOffset, 0U, elements * 2U);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    const bool fullySelected = upper ? colBase >= rowBase + validRows : colBase + validCols <= rowBase;
    if (fullySelected) {
        const uint32_t blockLen = validRows * 2U * sizeof(float);
        const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
        const DataCopyExtParams outputCopy{
            static_cast<uint16_t>(validCols), blockLen,
            static_cast<uint32_t>((blockRows * 2U * sizeof(float) - alignedBlockLen) / 32U),
            static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
        const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
        DataCopyPad(cG[cOffset], cAos, outputCopy);
    } else {
        triangle_store::StoreComplexDiagonal<64U>(
            cG, cAos, planar, indexBuffer.ReinterpretCast<int32_t>(), tiling.ldc, rowBase, colBase, validRows,
            validCols, upper, vToMte3);
    }
    // Index construction uses a dead computation buffer; release it only
    // after the vector gather has finished.
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
}

/** Process one SG3 64x32 output chunk. */
__aicore__ inline void Cher2kSg3ProcessChunk(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffset, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& planar,
    LocalTensor<float>& realIj, LocalTensor<float>& imagOut, LocalTensor<float>& imagIj, LocalTensor<float>& realJi,
    LocalTensor<float>& imagJi, LocalTensor<float>& temp, LocalTensor<float>& cAos, LocalTensor<float>& raw,
    uint64_t p1Base, uint64_t p2Base, uint64_t p3Base, uint32_t rowBase, uint32_t colBase, uint32_t validRows,
    uint32_t validCols, bool upper, bool unitAlphaNoBeta, float alphaReal, float alphaImag, float betaValue,
    event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV, event_t vToScalar, event_t scalarToV,
    event_t mte3ToMte2)
{
    Cher2kSg3LoadChunk(
        ws, tiling, transposeOffset, realIj, imagOut, imagIj, realJi, imagJi, raw, temp, p1Base, p2Base, p3Base,
        rowBase, colBase, mte2ToV, vToMte2);

    Cher2kSg3CombineChunk(
        cG, tiling, realIj, imagOut, realJi, imagJi, temp, raw, rowBase, colBase, validRows, validCols, unitAlphaNoBeta,
        alphaReal, alphaImag, betaValue, vToScalar, scalarToV);

    Cher2kSg3StoreChunk(
        cG, tiling, cAos, imagOut, planar, temp, interleaveOffset, rowBase, colBase, validRows, validCols, upper,
        vToMte2, vToMte3, vToScalar, scalarToV);
    // Both cAos (also the next raw input) and planar can be MTE3 sources.
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
}
struct Cher2kOutputStorage {
    TBuf<TPosition::VECCALC> transposeOffsetBuf;
    TBuf<TPosition::VECCALC> interleaveOffsetBuf;
    TBuf<TPosition::VECCALC> realIjBuf;
    TBuf<TPosition::VECCALC> imagIjBuf;
    TBuf<TPosition::VECCALC> realJiBuf;
    TBuf<TPosition::VECCALC> imagJiBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> cAosBuf;

    LocalTensor<uint32_t> transposeOffset;
    LocalTensor<uint32_t> interleaveOffset;
    LocalTensor<float> realIj;
    LocalTensor<float> imagIj;
    LocalTensor<float> realJi;
    LocalTensor<float> imagJi;
    LocalTensor<float> temp;
    LocalTensor<float> cAos;

    __aicore__ inline void InitStorage(TPipe& pipe, uint32_t elements)
    {
        const uint32_t aosElements = elements * 2U;
        pipe.InitBuffer(transposeOffsetBuf, elements * sizeof(uint32_t));
        pipe.InitBuffer(interleaveOffsetBuf, aosElements * sizeof(uint32_t));
        pipe.InitBuffer(realIjBuf, aosElements * sizeof(float));
        pipe.InitBuffer(imagIjBuf, elements * sizeof(float));
        pipe.InitBuffer(realJiBuf, elements * sizeof(float));
        pipe.InitBuffer(imagJiBuf, elements * sizeof(float));
        pipe.InitBuffer(tempBuf, elements * sizeof(float));
        pipe.InitBuffer(cAosBuf, aosElements * sizeof(float));
        transposeOffset = transposeOffsetBuf.Get<uint32_t>();
        interleaveOffset = interleaveOffsetBuf.Get<uint32_t>();
        realIj = realIjBuf.Get<float>();
        imagIj = imagIjBuf.Get<float>();
        realJi = realJiBuf.Get<float>();
        imagJi = imagJiBuf.Get<float>();
        temp = tempBuf.Get<float>();
        cAos = cAosBuf.Get<float>();
    }
};

struct Cher2kSg3ConsumerContext : Cher2kOutputStorage {
    LocalTensor<int32_t> transposeOffsetI32;
    LocalTensor<int32_t> interleaveOffsetI32;
    LocalTensor<float> planar;
    LocalTensor<float> imagOut;
    LocalTensor<float> raw;

    __aicore__ inline void Init(TPipe& pipe, uint32_t blockElements, uint32_t blockRows, uint32_t blockCols)
    {
        const uint32_t aosElements = blockElements * 2U;
        InitStorage(pipe, blockElements);
        transposeOffsetI32 = transposeOffsetBuf.Get<int32_t>();
        interleaveOffsetI32 = interleaveOffsetBuf.Get<int32_t>();
        planar = realIj;
        imagOut = planar[blockElements];
        raw = cAos;
        CreateVecIndex(transposeOffsetI32, 0, blockRows);
        PipeBarrier<PIPE_V>();
        Muls(transposeOffsetI32, transposeOffsetI32, static_cast<int32_t>(blockCols * sizeof(float)), blockRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t col = 1U; col < blockCols; ++col) {
            Adds(
                transposeOffsetI32[col * blockRows], transposeOffsetI32, static_cast<int32_t>(col * sizeof(float)),
                blockRows);
        }
        for (uint32_t index = 0U; index < blockRows * 2U; ++index) {
            const uint32_t source = (index & 1U) == 0U ? index / 2U : blockElements + index / 2U;
            interleaveOffset.SetValue(index, source * sizeof(float));
        }
        const auto seedToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(seedToVector);
        WaitFlag<HardEvent::S_V>(seedToVector);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t group = 1U; group < blockCols; ++group) {
            Adds(
                interleaveOffsetI32[group * blockRows * 2U], interleaveOffsetI32,
                static_cast<int32_t>(group * blockRows * sizeof(float)), blockRows * 2U);
        }
        PipeBarrier<PIPE_ALL>();
    }
};

struct Cher2kSg3ConsumerState {
    GlobalTensor<float> cG;
    GlobalTensor<float> ws;
    uint64_t p1Base = 0ULL;
    uint64_t p2Base = 0ULL;
    uint64_t p3Base = 0ULL;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    bool unitAlphaNoBeta = false;
    event_t mte2ToV;
    event_t vToMte2;
    event_t vToMte3;
    event_t mte3ToV;
    event_t mte3ToMte2;
    event_t vToScalar;
    event_t scalarToV;

    __aicore__ inline void Init(
        GM_ADDR c, GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, const Cher2kTilingData& tiling)
    {
        const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
        const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
        const bool packed =
            tiling.sg3MmadPath != 0U || (tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C);
        p1Base = packed ? 6ULL * nk : OffsetRi(tiling.nAligned, tiling.kAligned);
        p2Base = packed ? 6ULL * nk + nn : OffsetIr(tiling.nAligned, tiling.kAligned);
        p3Base = packed ? 6ULL * nk + 2ULL * nn : OffsetRr(tiling.nAligned, tiling.kAligned);
        cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
        ws.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace),
            packed ? Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned) :
                     WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, true));
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
        PipeBarrier<PIPE_ALL>();
        Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
        unitAlphaNoBeta = alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f;
        mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        mte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }
};

struct Cher2kOutputTile {
    uint32_t rowBase = 0U;
    uint32_t colBase = 0U;
    uint32_t validRows = 0U;
    uint32_t validCols = 0U;
};
struct Cher2kAssignedTriangle {
    uint64_t begin;
    uint64_t end;
    Cher2kTriangleCursor cursor;

    __aicore__ inline void Init(
        const Cher2kTilingData& tiling, uint32_t blockIdx, uint32_t blockNum, uint32_t rowTileCount, uint32_t rows,
        uint32_t cols, bool upper)
    {
        const uint64_t totalTasks = Cher2kTriangleTaskCount(tiling, rowTileCount, rows, cols, upper);
        const uint32_t safeBlockNum = blockNum == 0U ? 1U : blockNum;
        begin = (static_cast<uint64_t>(blockIdx) * totalTasks) / safeBlockNum;
        end = (static_cast<uint64_t>(blockIdx + 1U) * totalTasks) / safeBlockNum;
        Cher2kSeekTriangleTask(begin, tiling, rowTileCount, rows, cols, upper, cursor);
    }
    __aicore__ inline Cher2kOutputTile Resolve(
        uint64_t task, const Cher2kTilingData& tiling, uint32_t rowTileCount, uint32_t rows, uint32_t cols, bool upper)
    {
        Cher2kOutputTile tile;
        Cher2kResolveTriangleTask(
            task, tiling, rowTileCount, rows, cols, upper, cursor, tile.rowBase, tile.colBase, tile.validRows,
            tile.validCols);
        return tile;
    }
};

__aicore__ inline void Cher2kSg3ConsumerRun(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffset, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& planar,
    LocalTensor<float>& realIj, LocalTensor<float>& imagOut, LocalTensor<float>& imagIj, LocalTensor<float>& realJi,
    LocalTensor<float>& imagJi, LocalTensor<float>& temp, LocalTensor<float>& cAos, LocalTensor<float>& raw,
    uint64_t p1Base, uint64_t p2Base, uint64_t p3Base, float alphaReal, float alphaImag, float betaValue,
    bool unitAlphaNoBeta, event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV, event_t vToScalar,
    event_t scalarToV, event_t mte3ToMte2)
{
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    constexpr uint32_t blockRows = 64U;
    constexpr uint32_t blockCols = 32U;
    const uint32_t rowTileCount = (tiling.n + blockRows - 1U) / blockRows;
    const bool upper = tiling.uplo == ACLBLAS_UPPER;
    Cher2kAssignedTriangle range;
    range.Init(tiling, blockIdx, blockNum, rowTileCount, blockRows, blockCols, upper);
    for (uint64_t task = range.begin; task < range.end; ++task) {
        const Cher2kOutputTile tile = range.Resolve(task, tiling, rowTileCount, blockRows, blockCols, upper);
        Cher2kSg3ProcessChunk(
            cG, ws, tiling, transposeOffset, interleaveOffset, planar, realIj, imagOut, imagIj, realJi, imagJi, temp,
            cAos, raw, p1Base, p2Base, p3Base, tile.rowBase, tile.colBase, tile.validRows, tile.validCols, upper,
            unitAlphaNoBeta, alphaReal, alphaImag, betaValue, mte2ToV, vToMte2, vToMte3, mte3ToV, vToScalar, scalarToV,
            mte3ToMte2);
        SetFlag<HardEvent::MTE3_V>(mte3ToV);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2);
    }
}

// SG3 performance epilogue. Unlike the legacy 64x64 consumer
// keeps one 64x32 micro-tile in UB and gives every AIV a contiguous, balanced
// range of triangular micro-tiles.
// Its MTE2 -> V -> MTE3 ownership follows the stable Chemm/Cherk epilogues.
extern "C" __global__ __aicore__ void cher2k_sg3_chunk_postprocess_kernel(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta) || Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }

    constexpr uint32_t blockRows = 64U;
    // A 64x32 micro-tile keeps the eight-buffer consumer below 96 KiB UB and
    // doubles scheduling granularity.  This kernel owns its reconstruction
    // and aligned triangular writeback; it does not call the legacy 64x64
    // helper, so all copy/gather extents follow blockCols directly.
    constexpr uint32_t blockCols = 32U;
    constexpr uint32_t blockElements = blockRows * blockCols;
    constexpr uint32_t aosElements = blockElements * 2U;

    TPipe pipe;
    Cher2kSg3ConsumerContext context;
    context.Init(pipe, blockElements, blockRows, blockCols);
    auto& transposeOffset = context.transposeOffset;
    auto& transposeOffsetI32 = context.transposeOffsetI32;
    auto& interleaveOffset = context.interleaveOffset;
    auto& interleaveOffsetI32 = context.interleaveOffsetI32;
    auto& planar = context.planar;
    auto& realIj = context.realIj;
    auto& imagOut = context.imagOut;
    auto& imagIj = context.imagIj;
    auto& realJi = context.realJi;
    auto& imagJi = context.imagJi;
    auto& temp = context.temp;
    auto& cAos = context.cAos;
    auto& raw = context.raw;

    Cher2kSg3ConsumerState state;
    state.Init(c, workspace, alpha, beta, tiling);

    Cher2kSg3ConsumerRun(
        state.cG, state.ws, tiling, transposeOffset, interleaveOffset, planar, realIj, imagOut, imagIj, realJi, imagJi,
        temp, cAos, raw, state.p1Base, state.p2Base, state.p3Base, state.alphaReal, state.alphaImag, state.betaValue,
        state.unitAlphaNoBeta, state.mte2ToV, state.vToMte2, state.vToMte3, state.mte3ToV, state.vToScalar,
        state.scalarToV, state.mte3ToMte2);
    // Every micro-tile waits for MTE3 completion before its UB storage is
    // reused. Stream ordering publishes those GM writes to the caller; a
    // per-AIV whole-C cache operation is both redundant and unsafe here.
}
__aicore__ inline void Cher2kLoadPostprocessScalars(
    const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta, float& alphaReal, float& alphaImag, float& betaValue)
{
    alphaReal = tiling.alphaReal;
    alphaImag = tiling.alphaImag;
    betaValue = tiling.beta;
    if (alpha == nullptr || beta == nullptr) {
        return;
    }
    __gm__ const float* alphaPtr = reinterpret_cast<__gm__ const float*>(alpha);
    __gm__ const float* betaPtr = reinterpret_cast<__gm__ const float*>(beta);
    alphaReal = alphaPtr[0];
    alphaImag = alphaPtr[1];
    betaValue = betaPtr[0];
}

__aicore__ inline void Cher2kZeroDiagonal(
    LocalTensor<float>& imagLocal, uint32_t tileSize, uint32_t validRows, uint32_t validCols, event_t eventVToScalar,
    event_t eventScalarToV)
{
    SetFlag<HardEvent::V_S>(eventVToScalar);
    WaitFlag<HardEvent::V_S>(eventVToScalar);
    for (uint32_t diagonal = 0U; diagonal < min(validRows, validCols); ++diagonal) {
        imagLocal.SetValue(diagonal * tileSize + diagonal, 0.0f);
    }
    SetFlag<HardEvent::S_V>(eventScalarToV);
    WaitFlag<HardEvent::S_V>(eventScalarToV);
}

__aicore__ inline void Cher2kLoadOldSelected(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAosLocal, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols, uint32_t tileSize, uint32_t tileElements, event_t vToMte2,
    event_t vToScalar, event_t scalarToV)
{
    Duplicate(cAosLocal, 0.0f, tileElements * 2U);
    if constexpr (CHER2K_SCALAR_C_LOAD_EXPERIMENT) {
        SetFlag<HardEvent::V_S>(vToScalar);
        WaitFlag<HardEvent::V_S>(vToScalar);
        for (uint32_t col = 0U; col < validCols; ++col) {
            for (uint32_t row = 0U; row < validRows; ++row) {
                const bool selected =
                    tiling.uplo == ACLBLAS_UPPER ? rowBase + row <= colBase + col : rowBase + row >= colBase + col;
                if (!selected) {
                    continue;
                }
                const uint64_t cIndex = (static_cast<uint64_t>(colBase + col) * tiling.ldc + rowBase + row) * 2ULL;
                const uint32_t ubIndex = (col * tileSize + row) * 2U;
                cAosLocal.SetValue(ubIndex, cG.GetValue(cIndex));
                cAosLocal.SetValue(ubIndex + 1U, cG.GetValue(cIndex + 1ULL));
            }
        }
        SetFlag<HardEvent::S_V>(scalarToV);
        WaitFlag<HardEvent::S_V>(scalarToV);
    } else {
        // Finish the zero fill before MTE2 overwrites the valid C lanes.
        SetFlag<HardEvent::V_MTE2>(vToMte2);
        WaitFlag<HardEvent::V_MTE2>(vToMte2);
        const uint32_t cBlockLen = validRows * 2U * sizeof(float);
        const uint32_t cAlignedBlockLen = (cBlockLen + 31U) & ~31U;
        const uint32_t cRightPadding = (cAlignedBlockLen - cBlockLen) / sizeof(float);
        DataCopyExtParams cCopy{
            static_cast<uint16_t>(validCols), cBlockLen,
            static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)),
            static_cast<uint32_t>((tileSize * 2U * sizeof(float) - cAlignedBlockLen) / 32U), 0U};
        DataCopyPadExtParams<float> cPad{cRightPadding != 0U, 0U, static_cast<uint8_t>(cRightPadding), 0.0f};
        const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
        if (rowBase == colBase) {
            triangle_store::LoadComplexTile(
                cAosLocal, cG, tileSize, tiling.ldc, rowBase, colBase, validRows, validCols,
                tiling.uplo == ACLBLAS_UPPER, vToMte2);
        } else {
            DataCopyPad(cAosLocal, cG[cOffset], cCopy, cPad);
        }
    }
}

__aicore__ inline void Cher2kPostprocessApplyBetaGeneric(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& cAosLocal,
    LocalTensor<float>& cRealLocal, LocalTensor<float>& cImagLocal, LocalTensor<float>& realIjLocal,
    LocalTensor<float>& imagJiLocal, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols,
    uint32_t tileSize, uint32_t tileElements, float betaValue, event_t vToMte2, event_t vToScalar, event_t scalarToV)
{
    if (betaValue == 0.0f) {
        return;
    }
    Cher2kLoadOldSelected(
        cG, tiling, cAosLocal, rowBase, colBase, validRows, validCols, tileSize, tileElements, vToMte2, vToScalar,
        scalarToV);
    PipeBarrier<PIPE_ALL>();
    uint64_t reservedCount = 0ULL;
    GatherMask<float>(
        cRealLocal, cAosLocal, 1U, false, 0U, {1U, static_cast<uint16_t>(tileElements * 2U / 64U), 8U, 8U},
        reservedCount);
    GatherMask<float>(
        cImagLocal, cAosLocal, 2U, false, 0U, {1U, static_cast<uint16_t>(tileElements * 2U / 64U), 8U, 8U},
        reservedCount);
    PipeBarrier<PIPE_ALL>();
    Muls(cRealLocal, cRealLocal, betaValue, tileElements);
    Muls(cImagLocal, cImagLocal, betaValue, tileElements);
    PipeBarrier<PIPE_ALL>();
    Add(realIjLocal, realIjLocal, cRealLocal, tileElements);
    Add(imagJiLocal, imagJiLocal, cImagLocal, tileElements);
}

template <bool PAIR_LOAD>
__aicore__ inline void Cher2kPostprocessLoadThreeMIJ(
    GlobalTensor<float>& ws, LocalTensor<uint32_t>& transposeOffsetLocal, LocalTensor<float>& rawLocal,
    LocalTensor<float>& realIjLocal, LocalTensor<float>& imagIjLocal, LocalTensor<float>& realJiLocal,
    LocalTensor<float>& imagJiLocal, LocalTensor<float>& tempLocal, uint64_t rrBase, uint64_t riBase, uint64_t irBase,
    uint64_t ijOffset, uint32_t tileElements, const DataCopyExtParams& workspaceCopy)
{
    if constexpr (PAIR_LOAD) {
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, rrBase + ijOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(imagIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
        PipeBarrier<PIPE_ALL>();
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, riBase + ijOffset, workspaceCopy);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(imagJiLocal, ws, irBase + ijOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(realIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
        Gather(realJiLocal, imagJiLocal, transposeOffsetLocal, 0U, tileElements);
        PipeBarrier<PIPE_ALL>();
        Add(tempLocal, realIjLocal, realJiLocal, tileElements);
        Sub(realIjLocal, realIjLocal, realJiLocal, tileElements);
        Sub(imagIjLocal, imagIjLocal, tempLocal, tileElements);
    } else {
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, riBase + ijOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(realIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
        PipeBarrier<PIPE_ALL>();
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, irBase + ijOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(imagIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
        PipeBarrier<PIPE_ALL>();
        Add(tempLocal, realIjLocal, imagIjLocal, tileElements);
        Sub(realIjLocal, realIjLocal, imagIjLocal, tileElements);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, rrBase + ijOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Gather(imagIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
        PipeBarrier<PIPE_ALL>();
        Sub(imagIjLocal, imagIjLocal, tempLocal, tileElements);
    }
}
__aicore__ inline void Cher2kAdvanceCompactPostprocessTile(
    uint32_t tileAxis, bool upper, uint32_t& compactRow, uint32_t& compactCol);

template <bool TRIPLE_LOAD>
__aicore__ inline void Cher2kPostprocessLoadThreeMJI(
    GlobalTensor<float>& ws, LocalTensor<float>& rawLocal, LocalTensor<float>& realJiLocal,
    LocalTensor<float>& imagJiLocal, LocalTensor<float>& tempLocal, uint64_t rrBase, uint64_t riBase, uint64_t irBase,
    uint64_t jiOffset, uint32_t tileElements, const DataCopyExtParams& workspaceCopy, event_t eventVToMte2)
{
    if constexpr (TRIPLE_LOAD) {
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(realJiLocal, ws, riBase + jiOffset, workspaceCopy);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, irBase + jiOffset, workspaceCopy);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(imagJiLocal, ws, rrBase + jiOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Add(tempLocal, realJiLocal, rawLocal, tileElements);
        Sub(realJiLocal, realJiLocal, rawLocal, tileElements);
        Sub(imagJiLocal, imagJiLocal, tempLocal, tileElements);
    } else {
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(realJiLocal, ws, riBase + jiOffset, workspaceCopy);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(imagJiLocal, ws, irBase + jiOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Add(tempLocal, realJiLocal, imagJiLocal, tileElements);
        Sub(realJiLocal, realJiLocal, imagJiLocal, tileElements);
        SetFlag<HardEvent::V_MTE2>(eventVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2);
        CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(imagJiLocal, ws, rrBase + jiOffset, workspaceCopy);
        PipeBarrier<PIPE_ALL>();
        Sub(imagJiLocal, imagJiLocal, tempLocal, tileElements);
    }
}

__aicore__ inline void Cher2kPostprocessLoadTwoM(
    GlobalTensor<float>& ws, LocalTensor<uint32_t>& transposeOffsetLocal, LocalTensor<float>& rawLocal,
    LocalTensor<float>& realIjLocal, LocalTensor<float>& imagIjLocal, LocalTensor<float>& realJiLocal,
    LocalTensor<float>& imagJiLocal, LocalTensor<float>& tempLocal, uint64_t rrBase, uint64_t iiBase, uint64_t riBase,
    uint64_t irBase, uint64_t ijOffset, uint64_t jiOffset, uint32_t tileElements,
    const DataCopyExtParams& workspaceCopy)
{
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, rrBase + ijOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Gather(realIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, iiBase + ijOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Gather(tempLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    Sub(realIjLocal, realIjLocal, tempLocal, tileElements);
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, riBase + ijOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Gather(imagIjLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(rawLocal, ws, irBase + ijOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Gather(tempLocal, rawLocal, transposeOffsetLocal, 0U, tileElements);
    PipeBarrier<PIPE_ALL>();
    Add(imagIjLocal, imagIjLocal, tempLocal, tileElements);
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(realJiLocal, ws, rrBase + jiOffset, workspaceCopy);
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(tempLocal, ws, iiBase + jiOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Sub(realJiLocal, realJiLocal, tempLocal, tileElements);
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(imagJiLocal, ws, riBase + jiOffset, workspaceCopy);
    CopyWorkspaceTile<CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT>(tempLocal, ws, irBase + jiOffset, workspaceCopy);
    PipeBarrier<PIPE_ALL>();
    Add(imagJiLocal, imagJiLocal, tempLocal, tileElements);
}

__aicore__ inline void Cher2kPostprocessCombine(
    LocalTensor<float>& realIjLocal, LocalTensor<float>& imagIjLocal, LocalTensor<float>& realJiLocal,
    LocalTensor<float>& imagJiLocal, uint32_t tileElements, float alphaReal, float alphaImag, bool unitAlphaNoBeta)
{
    if (unitAlphaNoBeta) {
        Add(realIjLocal, realIjLocal, realJiLocal, tileElements);
        Sub(imagJiLocal, imagIjLocal, imagJiLocal, tileElements);
        return;
    }
    Sub(realJiLocal, realIjLocal, realJiLocal, tileElements);
    Sub(imagJiLocal, imagIjLocal, imagJiLocal, tileElements);
    if constexpr (CHER2K_POSTPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
    Muls(realIjLocal, realIjLocal, 2.0f, tileElements);
    Muls(imagIjLocal, imagIjLocal, 2.0f, tileElements);
    if constexpr (CHER2K_POSTPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
    Sub(realIjLocal, realIjLocal, realJiLocal, tileElements);
    Sub(imagIjLocal, imagIjLocal, imagJiLocal, tileElements);
    if constexpr (CHER2K_POSTPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
    Muls(realIjLocal, realIjLocal, alphaReal, tileElements);
    Muls(imagIjLocal, imagIjLocal, alphaImag, tileElements);
    Muls(imagJiLocal, imagJiLocal, alphaReal, tileElements);
    Muls(realJiLocal, realJiLocal, alphaImag, tileElements);
    if constexpr (CHER2K_POSTPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
    Sub(realIjLocal, realIjLocal, imagIjLocal, tileElements);
    Add(imagJiLocal, imagJiLocal, realJiLocal, tileElements);
}

__aicore__ inline void Cher2kPostprocessStoreTile(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, LocalTensor<float>& planarOutputLocal,
    LocalTensor<float>& imagJiLocal, LocalTensor<uint32_t>& interleaveOffsetLocal, LocalTensor<float>& cAosLocal,
    uint32_t rowTile, uint32_t colTile, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols,
    uint32_t tileSize, uint32_t tileElements, event_t eventVToScalar, event_t eventScalarToV)
{
    if (rowTile == colTile) {
        Cher2kZeroDiagonal(imagJiLocal, tileSize, validRows, validCols, eventVToScalar, eventScalarToV);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (CHER2K_POSTPROCESS_UB_COPY_EXPERIMENT) {
        DataCopy(planarOutputLocal[tileElements], imagJiLocal, tileElements);
    } else {
        Adds(planarOutputLocal[tileElements], imagJiLocal, 0.0f, tileElements);
    }
    PipeBarrier<PIPE_ALL>();
    Gather(cAosLocal, planarOutputLocal, interleaveOffsetLocal, 0U, tileElements * 2U);
    PipeBarrier<PIPE_ALL>();
    Cher2kStorePlanarUnitTile(
        cG, tiling, cAosLocal, planarOutputLocal, imagJiLocal, rowBase, colBase, validRows, validCols);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool Cher2kResolvePostprocessTile(
    uint64_t tile, uint32_t tileAxis, bool upper, uint32_t blockNum, uint32_t& compactRow, uint32_t& compactCol,
    uint32_t& incrementalRow, uint32_t& incrementalCol, uint32_t& rowTile, uint32_t& colTile)
{
    if constexpr (CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT) {
        if constexpr (CHER2K_POSTPROCESS_COMPACT_CONTIGUOUS_EXPERIMENT) {
            rowTile = compactRow;
            colTile = compactCol;
        } else {
            uint32_t ordinal = static_cast<uint32_t>(tile);
            if (upper) {
                uint32_t rowWidth = tileAxis;
                while (ordinal >= rowWidth && rowWidth > 0U) {
                    ordinal -= rowWidth;
                    ++rowTile;
                    --rowWidth;
                }
                colTile = rowTile + ordinal;
            } else {
                while (ordinal > rowTile) {
                    ordinal -= rowTile + 1U;
                    ++rowTile;
                }
                colTile = ordinal;
            }
        }
    } else {
        if constexpr (CHER2K_POSTPROCESS_INCREMENTAL_SCHEDULE_EXPERIMENT) {
            rowTile = incrementalRow;
            colTile = incrementalCol;
            uint32_t nextCol = incrementalCol + blockNum;
            while (nextCol >= tileAxis) {
                nextCol -= tileAxis;
                ++incrementalRow;
            }
            incrementalCol = nextCol;
        } else {
            rowTile = static_cast<uint32_t>(tile / (tileAxis == 0U ? 1U : tileAxis));
            colTile = static_cast<uint32_t>(tile - static_cast<uint64_t>(rowTile) * tileAxis);
        }
        if ((upper && rowTile > colTile) || (!upper && rowTile < colTile)) {
            return false;
        }
    }
    Cher2kAdvanceCompactPostprocessTile(tileAxis, upper, compactRow, compactCol);
    return true;
}

__aicore__ inline void Cher2kPostprocessExecuteTile(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffsetLocal, LocalTensor<uint32_t>& interleaveOffsetLocal,
    LocalTensor<float>& planarOutputLocal, LocalTensor<float>& realIjLocal, LocalTensor<float>& imagIjLocal,
    LocalTensor<float>& realJiLocal, LocalTensor<float>& imagJiLocal, LocalTensor<float>& tempLocal,
    LocalTensor<float>& rawLocal, LocalTensor<float>& cAosLocal, LocalTensor<float>& cRealLocal,
    LocalTensor<float>& cImagLocal, uint32_t rowTile, uint32_t colTile, uint32_t tileSize, uint32_t tileElements,
    uint32_t validRows, uint32_t validCols, uint64_t ijOffset, uint64_t jiOffset, uint64_t rrBase, uint64_t iiBase,
    uint64_t riBase, uint64_t irBase, const DataCopyExtParams& workspaceCopy, event_t eventVToMte2,
    event_t eventVToScalar, event_t eventScalarToV, float alphaReal, float alphaImag, float betaValue,
    bool unitAlphaNoBeta)
{
    if (tiling.computeProduct != 0U) {
        if (tiling.useThreeM != 0U) {
            Cher2kPostprocessLoadThreeMIJ<CHER2K_POSTPROCESS_IJ_PAIR_LOAD_EXPERIMENT>(
                ws, transposeOffsetLocal, rawLocal, realIjLocal, imagIjLocal, realJiLocal, imagJiLocal, tempLocal,
                rrBase, riBase, irBase, ijOffset, tileElements, workspaceCopy);
            Cher2kPostprocessLoadThreeMJI<CHER2K_POSTPROCESS_JI_TRIPLE_LOAD_EXPERIMENT>(
                ws, rawLocal, realJiLocal, imagJiLocal, tempLocal, rrBase, riBase, irBase, jiOffset, tileElements,
                workspaceCopy, eventVToMte2);
        } else {
            Cher2kPostprocessLoadTwoM(
                ws, transposeOffsetLocal, rawLocal, realIjLocal, imagIjLocal, realJiLocal, imagJiLocal, tempLocal,
                rrBase, iiBase, riBase, irBase, ijOffset, jiOffset, tileElements, workspaceCopy);
        }
        PipeBarrier<PIPE_ALL>();
        Cher2kPostprocessCombine(
            realIjLocal, imagIjLocal, realJiLocal, imagJiLocal, tileElements, alphaReal, alphaImag, unitAlphaNoBeta);
    } else {
        Duplicate(realIjLocal, 0.0f, tileElements);
        Duplicate(imagJiLocal, 0.0f, tileElements);
    }
    Cher2kPostprocessApplyBetaGeneric(
        cG, tiling, cAosLocal, cRealLocal, cImagLocal, realIjLocal, imagJiLocal, rowTile * tileSize, colTile * tileSize,
        validRows, validCols, tileSize, tileElements, betaValue, eventVToMte2, eventVToScalar, eventScalarToV);
    Cher2kPostprocessStoreTile(
        cG, tiling, planarOutputLocal, imagJiLocal, interleaveOffsetLocal, cAosLocal, rowTile, colTile,
        rowTile * tileSize, colTile * tileSize, validRows, validCols, tileSize, tileElements, eventVToScalar,
        eventScalarToV);
}

__aicore__ inline void Cher2kPostprocessRunSg3(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& transposeOffsetLocal, LocalTensor<uint32_t>& interleaveOffsetLocal,
    LocalTensor<float>& rawLocal, LocalTensor<float>& realIjLocal, LocalTensor<float>& imagIjLocal,
    LocalTensor<float>& realJiLocal, LocalTensor<float>& imagJiLocal, LocalTensor<float>& tempLocal,
    LocalTensor<float>& planarOutputLocal, LocalTensor<float>& cAosLocal, uint32_t blockIdx, uint32_t blockNum,
    uint32_t tileSize)
{
    const uint32_t tileAxis = (tiling.n + tileSize - 1U) / (tileSize == 0U ? 1U : tileSize);
    const uint64_t tileCount = static_cast<uint64_t>(tileAxis) * tileAxis;
    for (uint64_t tile = blockIdx; tile < tileCount; tile += blockNum) {
        const uint32_t rowTile = static_cast<uint32_t>(tile / (tileAxis == 0U ? 1U : tileAxis));
        const uint32_t colTile = static_cast<uint32_t>(tile - static_cast<uint64_t>(rowTile) * tileAxis);
        Cher2kPostprocessUnitTile(
            cG, ws, tiling, transposeOffsetLocal, interleaveOffsetLocal, rawLocal, realIjLocal, imagIjLocal,
            realJiLocal, imagJiLocal, tempLocal, planarOutputLocal, cAosLocal, rowTile, colTile);
    }
    PipeBarrier<PIPE_ALL>();
}

struct Cher2kPostprocessSchedule {
    uint64_t begin;
    uint64_t end;
    uint32_t step;
    uint32_t compactRow;
    uint32_t compactCol;
    uint32_t incrementalRow;
    uint32_t incrementalCol;
};

__aicore__ inline void Cher2kAdvanceCompactPostprocessTile(
    uint32_t tileAxis, bool upper, uint32_t& compactRow, uint32_t& compactCol)
{
    if constexpr (CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT && CHER2K_POSTPROCESS_COMPACT_CONTIGUOUS_EXPERIMENT) {
        ++compactCol;
        if (upper && compactCol == tileAxis) {
            ++compactRow;
            compactCol = compactRow;
        } else if (!upper && compactCol > compactRow) {
            ++compactRow;
            compactCol = 0U;
        }
    }
}

__aicore__ inline Cher2kPostprocessSchedule Cher2kInitPostprocessSchedule(
    uint32_t tileAxis, uint64_t scheduledTiles, uint32_t blockIdx, uint32_t blockNum, bool upper)
{
    Cher2kPostprocessSchedule schedule{blockIdx, scheduledTiles, blockNum, 0U, 0U, 0U, 0U};
    if constexpr (CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT && CHER2K_POSTPROCESS_COMPACT_CONTIGUOUS_EXPERIMENT) {
        const uint32_t safeBlockNum = blockNum == 0U ? 1U : blockNum;
        const uint64_t tilesPerCore = (scheduledTiles + safeBlockNum - 1U) / safeBlockNum;
        schedule.begin = static_cast<uint64_t>(blockIdx) * tilesPerCore;
        schedule.end = min(scheduledTiles, schedule.begin + tilesPerCore);
        schedule.step = 1U;
        uint32_t ordinal = static_cast<uint32_t>(schedule.begin);
        if (upper) {
            uint32_t rowWidth = tileAxis;
            while (ordinal >= rowWidth && rowWidth > 0U) {
                ordinal -= rowWidth;
                ++schedule.compactRow;
                --rowWidth;
            }
            schedule.compactCol = schedule.compactRow + ordinal;
        } else {
            while (ordinal > schedule.compactRow) {
                ordinal -= schedule.compactRow + 1U;
                ++schedule.compactRow;
            }
            schedule.compactCol = ordinal;
        }
    }
    if constexpr (
        CHER2K_POSTPROCESS_INCREMENTAL_SCHEDULE_EXPERIMENT && !CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT) {
        schedule.incrementalRow = blockIdx / (tileAxis == 0U ? 1U : tileAxis);
        schedule.incrementalCol = blockIdx - schedule.incrementalRow * tileAxis;
    }
    return schedule;
}

struct Cher2kPostprocessBuffers : Cher2kOutputStorage {
    LocalTensor<float> planarOutput;
    event_t vToMte2;
    event_t vToScalar;
    event_t scalarToV;

    __aicore__ inline void Init(TPipe& pipe, uint32_t tileSize, uint32_t tileElements)
    {
        InitStorage(pipe, tileElements);
        planarOutput = realIj;
        LocalTensor<int32_t> transposeI32 = transposeOffsetBuf.Get<int32_t>();
        LocalTensor<int32_t> interleaveI32 = interleaveOffsetBuf.Get<int32_t>();
        CreateVecIndex(transposeI32, 0, tileSize);
        PipeBarrier<PIPE_V>();
        Muls(transposeI32, transposeI32, static_cast<int32_t>(tileSize * sizeof(float)), tileSize);
        PipeBarrier<PIPE_V>();
        for (uint32_t group = 1U; group < tileSize; ++group) {
            Adds(transposeI32[group * tileSize], transposeI32, static_cast<int32_t>(group * sizeof(float)), tileSize);
        }
        Cher2kBuildInterleaveOffsets(interleaveOffset, interleaveI32, tileSize, tileSize, tileElements);
        vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }
};

__aicore__ inline void Cher2kPostprocessRunGeneric(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling, Cher2kPostprocessBuffers& buffers,
    uint64_t rrBase, uint64_t iiBase, uint64_t riBase, uint64_t irBase, uint32_t blockIdx, uint32_t blockNum,
    float alphaReal, float alphaImag, float betaValue, bool unitAlphaNoBeta)
{
    constexpr uint32_t tileSize = 64U;
    constexpr uint32_t tileElements = tileSize * tileSize;
    const uint32_t tileAxis = (tiling.n + tileSize - 1U) / (tileSize == 0U ? 1U : tileSize);
    const uint64_t totalTiles = static_cast<uint64_t>(tileAxis) * tileAxis;
    const uint64_t scheduledTiles = CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT ?
                                        (static_cast<uint64_t>(tileAxis) * (tileAxis + 1U)) / 2ULL :
                                        totalTiles;
    const DataCopyExtParams workspaceCopy{
        static_cast<uint16_t>(tileSize), static_cast<uint32_t>(tileSize * sizeof(float)),
        static_cast<uint32_t>((tiling.nAligned - tileSize) * sizeof(float)), 0U, 0U};
    Cher2kPostprocessSchedule schedule =
        Cher2kInitPostprocessSchedule(tileAxis, scheduledTiles, blockIdx, blockNum, tiling.uplo == ACLBLAS_UPPER);
    for (uint64_t tile = schedule.begin; tile < schedule.end; tile += schedule.step) {
        uint32_t rowTile = 0U;
        uint32_t colTile = 0U;
        const bool resolved = Cher2kResolvePostprocessTile(
            tile, tileAxis, tiling.uplo == ACLBLAS_UPPER, blockNum, schedule.compactRow, schedule.compactCol,
            schedule.incrementalRow, schedule.incrementalCol, rowTile, colTile);
        if (!resolved) {
            continue;
        }
        const uint32_t rowBase = rowTile * tileSize;
        const uint32_t colBase = colTile * tileSize;
        const uint32_t validRows = min(tileSize, tiling.n - rowBase);
        const uint32_t validCols = min(tileSize, tiling.n - colBase);
        const uint64_t ijOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
        const uint64_t jiOffset = static_cast<uint64_t>(colBase) * tiling.nAligned + rowBase;
        Cher2kPostprocessExecuteTile(
            cG, ws, tiling, buffers.transposeOffset, buffers.interleaveOffset, buffers.planarOutput, buffers.realIj,
            buffers.imagIj, buffers.realJi, buffers.imagJi, buffers.temp, buffers.cAos, buffers.cAos, buffers.realJi,
            buffers.imagIj, rowTile, colTile, tileSize, tileElements, validRows, validCols, ijOffset, jiOffset, rrBase,
            iiBase, riBase, irBase, workspaceCopy, buffers.vToMte2, buffers.vToScalar, buffers.scalarToV, alphaReal,
            alphaImag, betaValue, unitAlphaNoBeta);
    }
}

__aicore__ inline void Cher2kFinishPostprocessOutput(
    GlobalTensor<float>& cG, const Cher2kTilingData& tiling, bool unitAlphaNoBeta)
{
    if constexpr (!CHER2K_SKIP_C_OUTPUT_CACHE_CLEAN_EXPERIMENT) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cG);
    } else if (!(unitAlphaNoBeta && tiling.uplo == ACLBLAS_UPPER)) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cG);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kRunPostprocessKernel(
    GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta,
    Cher2kPostprocessBuffers& buffers)
{
    GlobalTensor<float> cG;
    GlobalTensor<float> ws;
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    const bool sgemm3Batch =
        tiling.sg3MmadPath != 0U || (tiling.directHermitianPath != 0U && tiling.trans == ACLBLAS_OP_C);
    if (tiling.computeProduct != 0U) {
        ws.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace),
            sgemm3Batch ? Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned) :
                          WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, tiling.useThreeM != 0U));
        if constexpr (!CHER2K_SKIP_WORKSPACE_CACHE_CLEAN_EXPERIMENT) {
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
        }
    }
    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    const uint64_t nn = static_cast<uint64_t>(tiling.nAligned) * tiling.nAligned;
    const uint64_t rrBase = sgemm3Batch ? 6ULL * nk + 2ULL * nn : OffsetRr(tiling.nAligned, tiling.kAligned);
    const uint64_t iiBase = OffsetIi(tiling.nAligned, tiling.kAligned);
    const uint64_t riBase = sgemm3Batch ? 6ULL * nk : OffsetRi(tiling.nAligned, tiling.kAligned);
    const uint64_t irBase = sgemm3Batch ? 6ULL * nk + nn : OffsetIr(tiling.nAligned, tiling.kAligned);
    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    Cher2kLoadPostprocessScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    const bool unitAlphaNoBeta =
        tiling.computeProduct != 0U && alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f;
    if (sgemm3Batch && unitAlphaNoBeta) {
        Cher2kPostprocessRunSg3(
            cG, ws, tiling, buffers.transposeOffset, buffers.interleaveOffset, buffers.cAos, buffers.realIj,
            buffers.imagIj, buffers.realJi, buffers.imagJi, buffers.temp, buffers.planarOutput, buffers.cAos, blockIdx,
            blockNum, 64U);
        return;
    }
    Cher2kPostprocessRunGeneric(
        cG, ws, tiling, buffers, rrBase, iiBase, riBase, irBase, blockIdx, blockNum, alphaReal, alphaImag, betaValue,
        unitAlphaNoBeta);
    Cher2kFinishPostprocessOutput(cG, tiling, unitAlphaNoBeta);
}

extern "C" __global__ __aicore__ void cher2k_postprocess_kernel(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta) || Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }
    constexpr uint32_t tileSize = 64U;
    constexpr uint32_t tileElements = tileSize * tileSize;
    TPipe pipe;
    Cher2kPostprocessBuffers buffers;
    buffers.Init(pipe, tileSize, tileElements);
    Cher2kRunPostprocessKernel(c, workspace, tiling, alpha, beta, buffers);
}

// Low-rank direct-Hermitian consumer. Cube already published real(H) and
// imag(H^T) in row-major storage, which is exactly the byte order required by
// column-major C.  A 64x64 unit therefore needs two planar reads and one
// interleave, with no mirrored workspace reads and no matrix transpose.
struct Cher2kDirectEvents {
    event_t mte2ToV;
    event_t vToMte2;
    event_t vToMte3;
    event_t mte3ToV;
    event_t vToScalar;
    event_t scalarToV;

    __aicore__ inline void Init()
    {
        mte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        vToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        vToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        mte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        vToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        scalarToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }
};

template <uint32_t ROWS = 64U>
__aicore__ inline void Cher2kDirectZeroDiagonal(
    LocalTensor<float>& imagLocal, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols,
    event_t vToScalar, event_t scalarToV)
{
    if (colBase >= rowBase + validRows || colBase + validCols <= rowBase) {
        return;
    }
    SetFlag<HardEvent::V_S>(vToScalar);
    WaitFlag<HardEvent::V_S>(vToScalar);
    for (uint32_t col = 0U; col < validCols; ++col) {
        const uint32_t absoluteCol = colBase + col;
        if (absoluteCol >= rowBase && absoluteCol < rowBase + validRows) {
            imagLocal.SetValue(col * ROWS + absoluteCol - rowBase, 0.0f);
        }
    }
    SetFlag<HardEvent::S_V>(scalarToV);
    WaitFlag<HardEvent::S_V>(scalarToV);
}
__aicore__ inline void Cher2kDirectMixLoad(
    GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, uint32_t rowBase, uint32_t colBase, event_t mte2ToV)
{
    constexpr uint32_t rows = 64U;
    constexpr uint32_t cols = 64U;
    const uint32_t validRows = min(rows, tiling.n - rowBase);
    const uint32_t validCols = min(cols, tiling.n - colBase);
    const uint64_t sourceOffset = static_cast<uint64_t>(colBase) * tiling.nAligned + rowBase;
    const uint32_t blockLen = validRows * sizeof(float);
    const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
    const uint32_t rightPadding = (alignedBlockLen - blockLen) / sizeof(float);
    const DataCopyExtParams inputCopy{
        static_cast<uint16_t>(validCols), blockLen,
        static_cast<uint32_t>((tiling.nAligned - validRows) * sizeof(float)),
        static_cast<uint32_t>((rows * sizeof(float) - alignedBlockLen) / 32U), 0U};
    const DataCopyPadExtParams<float> inputPad{rightPadding != 0U, 0U, static_cast<uint8_t>(rightPadding), 0.0f};
    const uint64_t realBase = DirectOpcReal(tiling.nAligned, tiling.kAligned);
    const uint64_t imagBase = DirectOpcImag(tiling.nAligned, tiling.kAligned);
    DataCopyPad(realLocal, ws[realBase + sourceOffset], inputCopy, inputPad);
    DataCopyPad(imagLocal, ws[imagBase + sourceOffset], inputCopy, inputPad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
}

__aicore__ inline void Cher2kDirectMixStore(
    GlobalTensor<float>& cG, LocalTensor<float>& cAos, const Cher2kTilingData& tiling, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols, event_t vToMte3, event_t mte3ToV)
{
    constexpr uint32_t rows = 64U;
    const bool fullySelected = colBase >= rowBase + validRows;
    if (fullySelected) {
        const uint32_t outputBlockLen = validRows * 2U * sizeof(float);
        const uint32_t alignedOutputBlockLen = (outputBlockLen + 31U) & ~31U;
        const DataCopyExtParams outputCopy{
            static_cast<uint16_t>(validCols), outputBlockLen,
            static_cast<uint32_t>((rows * 2U * sizeof(float) - alignedOutputBlockLen) / 32U),
            static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
        const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
        DataCopyPad(cG[cOffset], cAos, outputCopy);
    } else {
        PipeBarrier<PIPE_ALL>();
        for (uint32_t col = 0U; col < validCols; ++col) {
            const uint32_t absoluteCol = colBase + col;
            if (absoluteCol < rowBase) {
                continue;
            }
            const uint32_t selectedRows = min(validRows, absoluteCol - rowBase + 1U);
            if (selectedRows == 0U) {
                continue;
            }
            const uint64_t cOffset = (static_cast<uint64_t>(absoluteCol) * tiling.ldc + rowBase) * 2ULL;
            const DataCopyExtParams outputCopy{
                1U, static_cast<uint32_t>(selectedRows * 2U * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(cG[cOffset], cAos[col * rows * 2U], outputCopy);
        }
        PipeBarrier<PIPE_ALL>();
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

__aicore__ inline void Cher2kDirectMixUpperTile(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& planar, LocalTensor<float>& cAos, uint32_t rowBase,
    uint32_t colBase, event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV, event_t vToScalar,
    event_t scalarToV)
{
    constexpr uint32_t rows = 64U;
    constexpr uint32_t cols = 64U;
    constexpr uint32_t elements = rows * cols;
    if (rowBase >= tiling.n || colBase >= tiling.n || rowBase > colBase + cols - 1U) {
        return;
    }

    LocalTensor<float> realLocal = planar;
    LocalTensor<float> imagLocal = planar[elements];
    const uint32_t validRows = min(rows, tiling.n - rowBase);
    const uint32_t validCols = min(cols, tiling.n - colBase);
    Cher2kDirectMixLoad(ws, tiling, realLocal, imagLocal, rowBase, colBase, mte2ToV);
    Cher2kDirectZeroDiagonal(imagLocal, rowBase, colBase, validRows, validCols, vToScalar, scalarToV);

    Gather(cAos, planar, interleaveOffset, 0U, elements * 2U);
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);

    Cher2kDirectMixStore(cG, cAos, tiling, rowBase, colBase, validRows, validCols, vToMte3, mte3ToV);
}

__aicore__ inline void Cher2kDirectMixBuildOffsets(
    GM_ADDR interleaveOffsets, LocalTensor<uint32_t>& interleaveOffset, LocalTensor<int32_t>& interleaveOffsetI32,
    uint32_t rows, uint32_t cols, uint32_t elements)
{
    if (interleaveOffsets != nullptr) {
        GlobalTensor<uint32_t> offsetsG;
        offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(interleaveOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
        DataCopy(interleaveOffset, offsetsG, CHER2K_DIRECT_INTERLEAVE_COUNT);
        PipeBarrier<PIPE_ALL>();
        return;
    }
    for (uint32_t index = 0U; index < rows * 2U; ++index) {
        const uint32_t sourceIndex = (index & 1U) == 0U ? index / 2U : elements + index / 2U;
        interleaveOffset.SetValue(index, sourceIndex * sizeof(float));
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t col = 1U; col < cols; ++col) {
        Adds(
            interleaveOffsetI32[col * rows * 2U], interleaveOffsetI32, static_cast<int32_t>(col * rows * sizeof(float)),
            rows * 2U);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kDirectMixRunTasks(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling,
    LocalTensor<uint32_t>& interleaveOffset, LocalTensor<float>& planar, LocalTensor<float>& cAos, uint32_t groupIndex,
    uint32_t groupCount, uint32_t subBlock, event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV,
    event_t vToScalar, event_t scalarToV)
{
    constexpr uint32_t rows = 64U;
    constexpr uint32_t cols = 64U;
    constexpr uint32_t cubeTile = 128U;
    const uint32_t tileAxis = (tiling.n + cubeTile - 1U) / cubeTile;
    const uint64_t totalTasks = (static_cast<uint64_t>(tileAxis) * (tileAxis + 1U)) / 2ULL;
    const uint64_t totalWorkUnits = 2ULL * totalTasks;
    const uint32_t safeGroupCount = groupCount == 0U ? 1U : groupCount;
    const uint64_t taskBegin = (static_cast<uint64_t>(groupIndex) * totalWorkUnits) / safeGroupCount;
    const uint64_t taskEnd = (static_cast<uint64_t>(groupIndex + 1U) * totalWorkUnits) / safeGroupCount;
    uint32_t mTile = 0U;
    uint64_t rowTaskBase = 0ULL;
    ResolveTriangularTile(taskBegin / 2ULL, true, tileAxis, mTile, rowTaskBase);
    uint32_t localIteration = 0U;
    for (uint64_t workUnit = taskBegin; workUnit < taskEnd;) {
        const uint64_t task = workUnit / 2ULL;
        const uint32_t halfPart = static_cast<uint32_t>(workUnit & 1ULL);
        const bool mergeHalves = halfPart == 0U && workUnit + 1ULL < taskEnd;
        const uint32_t nTile = ResolveTriangularTile(task, true, tileAxis, mTile, rowTaskBase);
        const uint16_t slot = static_cast<uint16_t>(localIteration % CHER2K_DIRECT_MIX_FLAG_COUNT);
        CrossCoreWaitFlag<2, PIPE_MTE2>(slot);
        const uint32_t logicalRowBase = nTile * cubeTile + subBlock * rows;
        const uint32_t logicalColBase = mTile * cubeTile;
        if (mergeHalves || subBlock == halfPart) {
            for (uint32_t colPart = 0U; colPart < 2U; ++colPart) {
                Cher2kDirectMixUpperTile(
                    cG, ws, tiling, interleaveOffset, planar, cAos, logicalRowBase, logicalColBase + colPart * cols,
                    mte2ToV, vToMte2, vToMte3, mte3ToV, vToScalar, scalarToV);
            }
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(static_cast<uint16_t>(CHER2K_DIRECT_MIX_ACK_BASE + slot));
        ++localIteration;
        workUnit += mergeHalves ? 2ULL : 1ULL;
    }
}

__aicore__ inline void Cher2kRunDirectMixConsumer(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, const Cher2kTilingData& tiling, uint32_t blockIndex,
    uint32_t groupCount)
{
    constexpr uint32_t rows = 64U;
    constexpr uint32_t cols = 64U;
    constexpr uint32_t elements = rows * cols;
    const uint32_t groupIndex = blockIndex / 2U;
    const uint32_t subBlock = static_cast<uint32_t>(GetSubBlockIdx());
    TPipe pipe;
    TBuf<TPosition::VECCALC> interleaveOffsetBuf;
    TBuf<TPosition::VECCALC> planarBuf;
    TBuf<TPosition::VECCALC> cAosBuf;
    pipe.InitBuffer(interleaveOffsetBuf, elements * 2U * sizeof(uint32_t));
    pipe.InitBuffer(planarBuf, elements * 2U * sizeof(float));
    pipe.InitBuffer(cAosBuf, elements * 2U * sizeof(float));
    LocalTensor<uint32_t> interleaveOffset = interleaveOffsetBuf.Get<uint32_t>();
    LocalTensor<float> planar = planarBuf.Get<float>();
    LocalTensor<float> cAos = cAosBuf.Get<float>();
    LocalTensor<int32_t> interleaveOffsetI32 = interleaveOffsetBuf.Get<int32_t>();
    Cher2kDirectMixBuildOffsets(interleaveOffsets, interleaveOffset, interleaveOffsetI32, rows, cols, elements);
    GlobalTensor<float> cG;
    GlobalTensor<float> ws;
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    ws.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace), DirectOpcWorkspaceFloatCount(tiling.nAligned, tiling.kAligned));
    Cher2kDirectEvents events;
    events.Init();
    Cher2kDirectMixRunTasks(
        cG, ws, tiling, interleaveOffset, planar, cAos, groupIndex, groupCount, subBlock, events.mte2ToV,
        events.vToMte2, events.vToMte3, events.mte3ToV, events.vToScalar, events.scalarToV);
}

extern "C" __global__ __aicore__ __schedmode__(1) void cher2k_direct_opc_mix_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR interleaveOffsets, GM_ADDR alpha, GM_ADDR beta,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Cher2kVectorMaskGuard maskGuard;
    // This producer publishes only the unit-alpha Hermitian result.  Device
    // scalars are classified independently by every participating AIC/AIV;
    // non-unit calls return before the first cross-core synchronization and
    // are handled by the scalar-aware packed graph queued on the same stream.
    const bool unitAlphaNoBeta = alpha != nullptr && beta != nullptr ?
                                     Cher2kIsDeviceUnitScalar(alpha, beta) :
                                     (tiling.alphaReal == 1.0f && tiling.alphaImag == 0.0f && tiling.beta == 0.0f);
    if (!unitAlphaNoBeta) {
        return;
    }
    const uint32_t blockIndex = static_cast<uint32_t>(GetBlockIdx());
    uint32_t groupCount = tiling.aicCoreNum;
    if (groupCount == 0U) {
        groupCount = 1U;
    }

    // Phase 0: both AIV sub-blocks publish the direct-H operand planes.  Its
    // 152 KiB UB layout is dead before the consumer's 96 KiB layout is built.
    if ASCEND_IS_AIV {
        TPipe preprocessPipe;
        Cher2kPreprocessImpl(a, b, workspace, interleaveOffsets, tiling, blockIndex, groupCount * 2U, preprocessPipe);
        preprocessPipe.Reset();
    }

    // The sole whole-kernel boundary publishes all MTE3 writes to AIC. Batch
    // scheduling keeps every participating MIX group resident at the barrier.
    SyncAll<false>();

    if ASCEND_IS_AIC {
        AscendC::InitSocState();
        Cher2kKernelCube producer;
        producer.Init(
            reinterpret_cast<__gm__ float*>(workspace), tiling.nAligned, tiling.n, tiling.kAligned, tiling.k,
            tiling.uplo);
        producer.ProcessMixed(blockIndex, groupCount);
    }

    if ASCEND_IS_AIV {
        Cher2kRunDirectMixConsumer(c, workspace, interleaveOffsets, tiling, blockIndex, groupCount);
    }
}

// Small-shape fast path.  For tiny matrices the three-kernel planar pipeline
// costs more than the arithmetic itself.  One AIV block stream computes each
// selected C element directly from the original AoS BLAS operands, avoiding
// workspace allocation, preprocess, Cube launch, and postprocess launch.
template <uint32_t ROWS, uint32_t COLS>
__aicore__ inline void Cher2kDirectApplyScaleAndBeta(
    GlobalTensor<float>& cG, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    LocalTensor<float>& alphaImagScratch, LocalTensor<float>& alphaRealScratch, const Cher2kTilingData& tiling,
    uint32_t validRows, uint32_t validCols, uint32_t rowBase, uint32_t colBase, bool unitAlphaNoBeta, float alphaReal,
    float alphaImag, float betaValue, event_t vToScalar, event_t scalarToV)
{
    if (unitAlphaNoBeta) {
        return;
    }
    constexpr uint32_t blockElements = ROWS * COLS;
    Muls(realLocal, realLocal, alphaReal, blockElements);
    Muls(imagLocal, imagLocal, alphaReal, blockElements);
    Muls(alphaImagScratch, imagLocal, alphaImag, blockElements);
    Muls(alphaRealScratch, realLocal, alphaImag, blockElements);
    PipeBarrier<PIPE_ALL>();
    Sub(realLocal, realLocal, alphaImagScratch, blockElements);
    Add(imagLocal, imagLocal, alphaRealScratch, blockElements);
    Cher2kAddSelectedOldC(
        cG, tiling, realLocal, imagLocal, ROWS, rowBase, colBase, validRows, validCols, betaValue, vToScalar,
        scalarToV);
}

template <uint32_t ROWS, uint32_t COLS>
__aicore__ inline void Cher2kDirectStoreTile(
    GlobalTensor<float>& cG, LocalTensor<float>& cAos, LocalTensor<float>& oldCAos, const Cher2kTilingData& tiling,
    uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols, bool upper, bool fullySelected,
    event_t mte2ToV, event_t vToMte3, LocalTensor<float> indexScratch)
{
    constexpr uint32_t blockRows = ROWS;
    if (fullySelected) {
        const uint32_t blockLen = validRows * 2U * sizeof(float);
        const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
        const DataCopyExtParams outputCopy{
            static_cast<uint16_t>(validCols), blockLen,
            static_cast<uint32_t>((blockRows * 2U * sizeof(float) - alignedBlockLen) / 32U),
            static_cast<uint32_t>((tiling.ldc - validRows) * 2U * sizeof(float)), 0U};
        const uint64_t cOffset = (static_cast<uint64_t>(colBase) * tiling.ldc + rowBase) * 2ULL;
        DataCopyPad(cG[cOffset], cAos, outputCopy);
        return;
    }
    // The product planes are dead after the AoS Gather. Reuse the real
    // plane for integer byte-offset tables and oldCAos for aligned output.
    // No old C is loaded or written solely to preserve the other triangle.
    triangle_store::StoreComplexDiagonal<blockRows>(
        cG, cAos, oldCAos, indexScratch.ReinterpretCast<int32_t>(), tiling.ldc, rowBase, colBase, validRows, validCols,
        upper, vToMte3);
}
template <uint32_t ROWS, uint32_t COLS>
__aicore__ inline void Cher2kDirectPostprocessOneTask(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, LocalTensor<float>& cAos, LocalTensor<float>& oldCAos,
    LocalTensor<float>& alphaImagScratch, LocalTensor<float>& alphaRealScratch, LocalTensor<float>& planar,
    LocalTensor<uint32_t>& interleaveOffset, uint64_t realBase, uint64_t imagBase, bool upper, bool unitAlphaNoBeta,
    float alphaReal, float alphaImag, float betaValue, uint32_t rowBase, uint32_t colBase, uint32_t validRows,
    uint32_t validCols, event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV, event_t vToScalar,
    event_t scalarToV)
{
    constexpr uint32_t blockRows = ROWS;
    constexpr uint32_t blockCols = COLS;
    constexpr uint32_t aosElements = blockRows * blockCols * 2U;
    const uint64_t sourceOffset = static_cast<uint64_t>(colBase) * tiling.nAligned + rowBase;
    const uint32_t inputBlockLen = validRows * sizeof(float);
    const uint32_t alignedInputBlockLen = (inputBlockLen + 31U) & ~31U;
    const uint32_t inputRightPadding = (alignedInputBlockLen - inputBlockLen) / sizeof(float);
    const DataCopyExtParams inputCopy{
        static_cast<uint16_t>(validCols), inputBlockLen,
        static_cast<uint32_t>((tiling.nAligned - validRows) * sizeof(float)),
        static_cast<uint32_t>((blockRows * sizeof(float) - alignedInputBlockLen) / 32U), 0U};
    const DataCopyPadExtParams<float> inputPad{
        inputRightPadding != 0U, 0U, static_cast<uint8_t>(inputRightPadding), 0.0f};
    DataCopyPad(realLocal, ws[realBase + sourceOffset], inputCopy, inputPad);
    DataCopyPad(imagLocal, ws[imagBase + sourceOffset], inputCopy, inputPad);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);
    Cher2kDirectApplyScaleAndBeta<ROWS, COLS>(
        cG, realLocal, imagLocal, alphaImagScratch, alphaRealScratch, tiling, validRows, validCols, rowBase, colBase,
        unitAlphaNoBeta, alphaReal, alphaImag, betaValue, vToScalar, scalarToV);
    Cher2kDirectZeroDiagonal<ROWS>(imagLocal, rowBase, colBase, validRows, validCols, vToScalar, scalarToV);
    Gather(cAos, planar, interleaveOffset, 0U, aosElements);
    SetFlag<HardEvent::V_MTE3>(vToMte3);
    WaitFlag<HardEvent::V_MTE3>(vToMte3);
    const bool fullySelected = upper ? colBase >= rowBase + validRows : colBase + validCols <= rowBase;
    Cher2kDirectStoreTile<ROWS, COLS>(
        cG, cAos, oldCAos, tiling, rowBase, colBase, validRows, validCols, upper, fullySelected, mte2ToV, vToMte3,
        realLocal);
    // realLocal holds the packing index tables until the suffix Gathers
    // finish. Only now may the next product load reuse it on MTE2.
    SetFlag<HardEvent::V_MTE2>(vToMte2);
    WaitFlag<HardEvent::V_MTE2>(vToMte2);
    SetFlag<HardEvent::MTE3_V>(mte3ToV);
    WaitFlag<HardEvent::MTE3_V>(mte3ToV);
}

/** Execute the assigned direct-H triangular tile range. */
template <uint32_t ROWS, uint32_t COLS>
__aicore__ inline void Cher2kDirectPostprocessTasks(
    GlobalTensor<float>& cG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, LocalTensor<float>& cAos, LocalTensor<float>& oldCAos,
    LocalTensor<float>& alphaImagScratch, LocalTensor<float>& alphaRealScratch, LocalTensor<float>& planar,
    LocalTensor<uint32_t>& interleaveOffset, uint64_t realBase, uint64_t imagBase, bool upper, bool unitAlphaNoBeta,
    float alphaReal, float alphaImag, float betaValue, uint32_t blockIdx, uint32_t blockNum, uint32_t rowTileCount,
    event_t mte2ToV, event_t vToMte2, event_t vToMte3, event_t mte3ToV, event_t vToScalar, event_t scalarToV)
{
    constexpr uint32_t blockRows = ROWS;
    constexpr uint32_t blockCols = COLS;
    Cher2kAssignedTriangle range;
    range.Init(tiling, blockIdx, blockNum, rowTileCount, blockRows, blockCols, upper);
    for (uint64_t task = range.begin; task < range.end; ++task) {
        const Cher2kOutputTile tile = range.Resolve(task, tiling, rowTileCount, blockRows, blockCols, upper);
        Cher2kDirectPostprocessOneTask<ROWS, COLS>(
            cG, ws, tiling, realLocal, imagLocal, cAos, oldCAos, alphaImagScratch, alphaRealScratch, planar,
            interleaveOffset, realBase, imagBase, upper, unitAlphaNoBeta, alphaReal, alphaImag, betaValue, tile.rowBase,
            tile.colBase, tile.validRows, tile.validCols, mte2ToV, vToMte2, vToMte3, mte3ToV, vToScalar, scalarToV);
    }
}
template <uint32_t ROWS, uint32_t COLS>
struct Cher2kDirectPostprocessBuffers {
    TBuf<TPosition::VECCALC> interleaveOffsetBuf;
    TBuf<TPosition::VECCALC> planarBuf;
    TBuf<TPosition::VECCALC> cAosBuf;
    TBuf<TPosition::VECCALC> oldCAosBuf;
    LocalTensor<uint32_t> interleaveOffset;
    LocalTensor<int32_t> interleaveOffsetI32;
    LocalTensor<float> planar;
    LocalTensor<float> realLocal;
    LocalTensor<float> imagLocal;
    LocalTensor<float> cAos;
    LocalTensor<float> oldCAos;
    LocalTensor<float> alphaImagScratch;
    LocalTensor<float> alphaRealScratch;

    __aicore__ inline void Init(TPipe& pipe)
    {
        constexpr uint32_t elements = ROWS * COLS;
        pipe.InitBuffer(interleaveOffsetBuf, elements * 2U * sizeof(uint32_t));
        pipe.InitBuffer(planarBuf, elements * 2U * sizeof(float));
        pipe.InitBuffer(cAosBuf, elements * 2U * sizeof(float));
        pipe.InitBuffer(oldCAosBuf, elements * 2U * sizeof(float));
        interleaveOffset = interleaveOffsetBuf.Get<uint32_t>();
        interleaveOffsetI32 = interleaveOffsetBuf.Get<int32_t>();
        planar = planarBuf.Get<float>();
        realLocal = planar;
        imagLocal = planar[elements];
        cAos = cAosBuf.Get<float>();
        oldCAos = oldCAosBuf.Get<float>();
        alphaImagScratch = cAos;
        alphaRealScratch = cAos[elements];
    }
};

template <uint32_t ROWS, uint32_t COLS>
__aicore__ inline void Cher2kRunDirectPostprocess(
    GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta,
    Cher2kDirectPostprocessBuffers<ROWS, COLS>& buffers)
{
    GlobalTensor<float> cG;
    GlobalTensor<float> ws;
    const bool directOutputOpc = tiling.directOutputPath != 0U && tiling.trans == ACLBLAS_OP_C;
    const uint64_t realBase =
        directOutputOpc ? DirectOpcReal(tiling.nAligned, tiling.kAligned) : OffsetRr(tiling.nAligned, tiling.kAligned);
    const uint64_t imagBase =
        directOutputOpc ? DirectOpcImag(tiling.nAligned, tiling.kAligned) : OffsetRi(tiling.nAligned, tiling.kAligned);
    cG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c), static_cast<uint64_t>(tiling.ldc) * tiling.n * 2ULL);
    ws.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace), directOutputOpc ?
                                                        DirectOpcWorkspaceFloatCount(tiling.nAligned, tiling.kAligned) :
                                                        WorkspaceFloatCount(tiling.nAligned, tiling.kAligned, true));
    Cher2kDirectEvents events;
    events.Init();
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaValue = 0.0f;
    Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    Cher2kDirectPostprocessTasks<ROWS, COLS>(
        cG, ws, tiling, buffers.realLocal, buffers.imagLocal, buffers.cAos, buffers.oldCAos, buffers.alphaImagScratch,
        buffers.alphaRealScratch, buffers.planar, buffers.interleaveOffset, realBase, imagBase,
        tiling.uplo == ACLBLAS_UPPER, alphaReal == 1.0f && alphaImag == 0.0f && betaValue == 0.0f, alphaReal, alphaImag,
        betaValue, static_cast<uint32_t>(GetBlockIdx()), blockNum, (tiling.n + ROWS - 1U) / ROWS, events.mte2ToV,
        events.vToMte2, events.vToMte3, events.mte3ToV, events.vToScalar, events.scalarToV);
}

extern "C" __global__ __aicore__ void cher2k_direct_hermitian_postprocess_kernel(
    GM_ADDR c, GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch, const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kIsNoOp(tiling, alpha, beta) || Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }
    TPipe pipe;
    if (tiling.trans == ACLBLAS_OP_N) {
        // Keep 4096 elements in UB, but make each column DMA four times
        // longer than the narrow chunk consumer. Leading-dimension gaps
        // then amortize over 2 KiB output bursts instead of short bursts.
        Cher2kDirectPostprocessBuffers<256U, 16U> buffers;
        buffers.Init(pipe);
        Cher2kBuildInterleaveOffsets(buffers.interleaveOffset, buffers.interleaveOffsetI32, 256U, 16U, 4096U);
        Cher2kRunDirectPostprocess(c, workspace, tiling, alpha, beta, buffers);
    } else {
        Cher2kDirectPostprocessBuffers<64U, 64U> buffers;
        buffers.Init(pipe);
        Cher2kBuildInterleaveOffsets(buffers.interleaveOffset, buffers.interleaveOffsetI32, 64U, 64U, 4096U);
        Cher2kRunDirectPostprocess(c, workspace, tiling, alpha, beta, buffers);
    }
}
