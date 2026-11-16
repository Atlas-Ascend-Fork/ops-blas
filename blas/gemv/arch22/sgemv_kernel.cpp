/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "sgemv_tiling_data.h"

namespace {
constexpr uint32_t UB_A_FLOATS = 32768U;
constexpr uint32_t UB_VEC_FLOATS = 2048U;
constexpr uint32_t F32_BLOCK_LEN = 8U;
constexpr uint32_t MAX_2D_STRIDE_BYTES = 65535U;
constexpr uint32_t FAST_BATCH_COLS = 64U;
constexpr uint32_t EXACT_TRANS_BATCH = 8U;
constexpr int32_t FAST_TRANS_MIN_M = 65;
constexpr int32_t FAST_TRANS_MAX_M = 65536;

union SgemvFp32Bits {
    float f;
    uint32_t u;
};

__aicore__ inline uint32_t AlignUp8(uint32_t count)
{
    return (count + F32_BLOCK_LEN - 1U) & ~(F32_BLOCK_LEN - 1U);
}

__aicore__ inline uint32_t SafeDivU32(uint32_t num, uint32_t den)
{
    if (den == 0U) {
        return 1U;
    }
    return num / den;
}

__aicore__ inline uint64_t ShiftRightJam64(uint64_t val, uint32_t dist)
{
    if (dist == 0U) {
        return val;
    }
    if (dist >= 64U) {
        return static_cast<uint64_t>(val != 0U);
    }
    return (val >> dist) | static_cast<uint64_t>((val << (64U - dist)) != 0U);
}

__aicore__ inline float PackNearestEvenFp32(uint64_t mag, uint32_t signBit, int32_t expVal)
{
    if (mag == 0U) {
        return 0.0f;
    }
    int32_t leadBit = 63 - static_cast<int32_t>(__builtin_clzll(mag));
    expVal += leadBit - 61;
    int32_t rshift = leadBit - 23;
    if (expVal < -126) {
        rshift += -126 - expVal;
        expVal = -126;
    }
    uint64_t sig = 0U;
    if (rshift < 64) {
        if (rshift > 0) {
            sig = mag >> static_cast<uint32_t>(rshift);
            uint64_t rem = mag & ((uint64_t{1} << static_cast<uint32_t>(rshift)) - 1U);
            uint64_t half = uint64_t{1} << static_cast<uint32_t>(rshift - 1);
            sig += static_cast<uint64_t>(rem > half || (rem == half && (sig & 1U) != 0U));
        } else {
            sig = mag << static_cast<uint32_t>(-rshift);
        }
    }
    if (sig >= 0x1000000U) {
        sig >>= 1U;
        expVal++;
    }
    SgemvFp32Bits res;
    if (expVal > 127) {
        res.u = signBit | 0x7F800000U;
    } else if (sig < 0x800000U) {
        res.u = signBit | static_cast<uint32_t>(sig);
    } else {
        res.u = signBit | (static_cast<uint32_t>(expVal + 127) << 23U) | (static_cast<uint32_t>(sig) & 0x7FFFFFU);
    }
    return res.f;
}

__aicore__ inline uint32_t UnpackMantissaFp32(uint32_t bits, uint32_t expField, int32_t *outExp)
{
    *outExp = static_cast<int32_t>(expField) - 127 + static_cast<int32_t>(expField == 0U);
    return (bits & 0x7FFFFFU) | (static_cast<uint32_t>(expField != 0U) << 23U);
}

__aicore__ inline float SgemvExactFmaGeneral(SgemvFp32Bits ua, SgemvFp32Bits ub, SgemvFp32Bits uc)
{
    uint32_t ea = (ua.u >> 23U) & 0xFFU;
    uint32_t eb = (ub.u >> 23U) & 0xFFU;
    uint32_t ec = (uc.u >> 23U) & 0xFFU;
    if (ec == 0xFFU && ea != 0xFFU && eb != 0xFFU) {
        return uc.f + uc.f;
    }
    bool hasSpecial = (ea == 0xFFU) || (eb == 0xFFU) || (ec == 0xFFU);
    bool prodZero = ((ua.u & 0x7FFFFFFFU) == 0U) || ((ub.u & 0x7FFFFFFFU) == 0U);
    if (hasSpecial || prodZero) {
        return ua.f * ub.f + uc.f;
    }
    if ((uc.u & 0x7FFFFFFFU) == 0U) {
        return ua.f * ub.f;
    }
    int32_t expA = 0;
    int32_t expB = 0;
    int32_t expC = 0;
    uint32_t ma = UnpackMantissaFp32(ua.u, ea, &expA);
    uint32_t mb = UnpackMantissaFp32(ub.u, eb, &expB);
    uint32_t mc = UnpackMantissaFp32(uc.u, ec, &expC);
    uint64_t prod = static_cast<uint64_t>(ma) * static_cast<uint64_t>(mb);
    int32_t topP = 63 - static_cast<int32_t>(__builtin_clzll(prod));
    int32_t expP = expA + expB + topP - 46;
    prod <<= static_cast<uint32_t>(61 - topP);
    int32_t topC = 31 - static_cast<int32_t>(__builtin_clz(mc));
    expC += topC - 23;
    uint64_t addend = static_cast<uint64_t>(mc) << static_cast<uint32_t>(61 - topC);
    int32_t refExp = expP;
    if (expP > expC) {
        addend = ShiftRightJam64(addend, static_cast<uint32_t>(expP - expC));
    } else if (expP < expC) {
        prod = ShiftRightJam64(prod, static_cast<uint32_t>(expC - expP));
        refExp = expC;
    }
    uint32_t signP = (ua.u ^ ub.u) & 0x80000000U;
    uint32_t signC = uc.u & 0x80000000U;
    if (signP == signC) {
        return PackNearestEvenFp32(prod + addend, signP, refExp);
    }
    if (prod >= addend) {
        return PackNearestEvenFp32(prod - addend, signP, refExp);
    }
    return PackNearestEvenFp32(addend - prod, signC, refExp);
}

__aicore__ inline float SgemvExactFma(float a, float b, float c)
{
    SgemvFp32Bits ua{a};
    SgemvFp32Bits ub{b};
    SgemvFp32Bits uc{c};
    uint32_t ea = (ua.u >> 23U) & 0xFFU;
    uint32_t eb = (ub.u >> 23U) & 0xFFU;
    uint32_t ec = (uc.u >> 23U) & 0xFFU;
    if (ea != 0U && eb != 0U && ec != 0U && ea != 0xFFU && eb != 0xFFU && ec != 0xFFU && (ea + eb) >= 194U) {
        int32_t cShift = static_cast<int32_t>(ec) - static_cast<int32_t>(ea + eb) + 150;
        if (cShift >= 0 && cShift <= 37) {
            uint64_t prod = static_cast<uint64_t>((ua.u & 0x7FFFFFU) | 0x800000U) *
                            static_cast<uint64_t>((ub.u & 0x7FFFFFU) | 0x800000U);
            uint64_t shiftedC = static_cast<uint64_t>((uc.u & 0x7FFFFFU) | 0x800000U) << static_cast<uint32_t>(cShift);
            int64_t sProd = ((ua.u ^ ub.u) & 0x80000000U) ? -static_cast<int64_t>(prod) : static_cast<int64_t>(prod);
            int64_t sAdd = (uc.u & 0x80000000U) ? -static_cast<int64_t>(shiftedC) : static_cast<int64_t>(shiftedC);
            int64_t total = sProd + sAdd;
            if (total == 0) {
                return 0.0f;
            }
            uint32_t outSign = (total < 0) ? 0x80000000U : 0U;
            uint64_t mag = (total < 0) ? static_cast<uint64_t>(-total) : static_cast<uint64_t>(total);
            int32_t top = 63 - static_cast<int32_t>(__builtin_clzll(mag));
            int32_t rshift = top - 23;
            uint32_t rounded = 0U;
            if (rshift > 0) {
                uint64_t bias = (uint64_t{1} << static_cast<uint32_t>(rshift - 1)) - 1U +
                                ((mag >> static_cast<uint32_t>(rshift)) & 1U);
                rounded = static_cast<uint32_t>((mag + bias) >> static_cast<uint32_t>(rshift));
            } else {
                rounded = static_cast<uint32_t>(mag << static_cast<uint32_t>(-rshift));
            }
            uint32_t expField = static_cast<uint32_t>(static_cast<int32_t>(ea + eb) + top - 174) * 0x800000U + rounded;
            SgemvFp32Bits out;
            out.u = outSign | ((expField >= 0x7F800000U) ? 0x7F800000U : expField);
            return out.f;
        }
    }
    return SgemvExactFmaGeneral(ua, ub, uc);
}

__aicore__ inline void CopyGmToUb1D(const AscendC::LocalTensor<float> &dst, const AscendC::GlobalTensor<float> &src,
                                    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyUbToGm1D(const AscendC::GlobalTensor<float> &dst, const AscendC::LocalTensor<float> &src,
                                    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

__aicore__ inline void CopyGmToUb2D(const AscendC::LocalTensor<float> &dst, const AscendC::GlobalTensor<float> &src,
                                    uint32_t rows, uint32_t cols, uint32_t lda)
{
    if (lda == rows && ((rows & 7U) == 0U)) {
        CopyGmToUb1D(dst, src, rows * cols);
        return;
    }
    uint32_t strideBytes = (lda - rows) * static_cast<uint32_t>(sizeof(float));
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (strideBytes <= MAX_2D_STRIDE_BYTES) {
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(cols),
                                              static_cast<uint32_t>(rows * sizeof(float)),
                                              strideBytes, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
        return;
    }
    uint32_t pitch = AlignUp8(rows);
    for (uint32_t c = 0; c < cols; ++c) {
        CopyGmToUb1D(dst[c * pitch], src[static_cast<int64_t>(c) * static_cast<int64_t>(lda)], rows);
    }
}

class SgemvKernel {
public:
    __aicore__ inline SgemvKernel() = default;

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SgemvTilingData &tiling, AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        kernelMode_ = tiling.kernelMode;
        usedCoreNum_ = tiling.usedCoreNum;
        workPerCore_ = tiling.workPerCore;
        m_ = tiling.m;
        n_ = tiling.n;
        lda_ = tiling.lda;
        incx_ = tiling.incx;
        incy_ = tiling.incy;
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;

        const bool isNoTrans = (kernelMode_ != SGEMV_MODE_TRANS);
        const int64_t xLen = isNoTrans ? static_cast<int64_t>(n_) : static_cast<int64_t>(m_);
        const int64_t yLen = isNoTrans ? static_cast<int64_t>(m_) : static_cast<int64_t>(n_);
        const int64_t absIncx = (incx_ > 0) ? static_cast<int64_t>(incx_) : -static_cast<int64_t>(incx_);
        const int64_t absIncy = (incy_ > 0) ? static_cast<int64_t>(incy_) : -static_cast<int64_t>(incy_);

        xBase_ = (incx_ > 0) ? 0 : (1 - xLen) * static_cast<int64_t>(incx_);
        yBase_ = (incy_ > 0) ? 0 : (1 - yLen) * static_cast<int64_t>(incy_);

        const uint64_t aSpan = static_cast<uint64_t>(lda_) * static_cast<uint64_t>(n_);
        const uint64_t xSpan = static_cast<uint64_t>((xLen - 1) * absIncx + 1);
        const uint64_t ySpan = static_cast<uint64_t>((yLen - 1) * absIncy + 1);

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a), aSpan);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xSpan);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), ySpan);

        pipe_->InitBuffer(aBuf_, UB_A_FLOATS * sizeof(float));
        pipe_->InitBuffer(xBuf_, UB_VEC_FLOATS * sizeof(float));
        pipe_->InitBuffer(yBuf_, UB_VEC_FLOATS * sizeof(float));
        pipe_->InitBuffer(workBuf_, UB_VEC_FLOATS * sizeof(float));

        ubA_ = aBuf_.Get<float>();
        ubX_ = xBuf_.Get<float>();
        ubY_ = yBuf_.Get<float>();
        ubWork_ = workBuf_.Get<float>();
    }

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (coreIdx >= usedCoreNum_) {
            return;
        }
        if (kernelMode_ == SGEMV_MODE_NO_TRANS_SPLIT_N) {
            ProcessNoTransSplitN(coreIdx);
            return;
        }
        if (kernelMode_ == SGEMV_MODE_NO_TRANS) {
            ProcessNoTrans(coreIdx);
            return;
        }
        ProcessTrans(coreIdx);
    }

private:
    __aicore__ inline bool CanUseFastVec() const
    {
        return (incx_ == 1) && (incy_ == 1) && (alpha_ == 1.0f) && (beta_ == 0.0f);
    }

    __aicore__ inline void WriteScalarToGm(int64_t yIdx, float val)
    {
        ubWork_.SetValue(0, val);
        AscendC::PipeBarrier<PIPE_ALL>();
        CopyUbToGm1D(yGm_[yIdx], ubWork_, 1U);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ScaleYAlphaZero(uint32_t startIdx, uint32_t count)
    {
        if (incy_ == 1) {
            for (uint32_t off = 0; off < count; off += UB_VEC_FLOATS) {
                uint32_t cur = ((count - off) < UB_VEC_FLOATS) ? (count - off) : UB_VEC_FLOATS;
                uint32_t aligned = AlignUp8(cur);
                if (beta_ == 0.0f) {
                    AscendC::Duplicate<float>(ubY_, 0.0f, static_cast<int32_t>(aligned));
                } else {
                    CopyGmToUb1D(ubY_, yGm_[startIdx + off], cur);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Muls(ubY_, ubY_, beta_, static_cast<int32_t>(aligned));
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyUbToGm1D(yGm_[startIdx + off], ubY_, cur);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            return;
        }
        for (uint32_t i = 0; i < count; ++i) {
            int64_t yIdx = yBase_ + static_cast<int64_t>(startIdx + i) * static_cast<int64_t>(incy_);
            WriteScalarToGm(yIdx, (beta_ == 0.0f) ? 0.0f : (beta_ * yGm_.GetValue(yIdx)));
        }
    }

    __aicore__ inline void AccumulateSplitNBatch(uint32_t tileM, uint32_t tileMAligned,
                                                 uint32_t curBatch, uint32_t maxBatch)
    {
        if (curBatch == maxBatch) {
            uint16_t rowBlocks = static_cast<uint16_t>(tileMAligned >> 3U);
            uint8_t repeatTimes = static_cast<uint8_t>(curBatch >> 3U);
            AscendC::BrcbRepeatParams brcbParams(rowBlocks, static_cast<uint16_t>(rowBlocks * 8U));
            for (uint32_t b = 0; b < rowBlocks; ++b) {
                AscendC::Brcb(ubWork_[b * 8U], ubX_, repeatTimes, brcbParams);
            }
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t totalElems = curBatch * tileMAligned;
            AscendC::Mul(ubA_, ubA_, ubWork_, static_cast<int32_t>(totalElems));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t half = curBatch >> 1U; half >= 1U; half >>= 1U) {
                uint32_t halfElems = half * tileMAligned;
                AscendC::Add(ubA_, ubA_, ubA_[halfElems], static_cast<int32_t>(halfElems));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(ubY_, ubY_, ubA_, static_cast<int32_t>(tileMAligned));
            AscendC::PipeBarrier<PIPE_V>();
            return;
        }
        for (uint32_t k = 0; k < curBatch; ++k) {
            float xScaled = ubX_.GetValue(k);
            AscendC::Axpy(ubY_, ubA_[k * tileMAligned], xScaled, static_cast<int32_t>(tileM));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessNoTransSplitN(uint32_t coreIdx)
    {
        uint32_t colStart = coreIdx * workPerCore_;
        uint32_t totalN = static_cast<uint32_t>(n_);
        if (colStart >= totalN) {
            return;
        }
        uint32_t colEnd = ((colStart + workPerCore_) < totalN) ? (colStart + workPerCore_) : totalN;
        uint32_t tileM = static_cast<uint32_t>(m_);
        uint32_t tileMAligned = AlignUp8(tileM);
        uint32_t maxBatch = (tileMAligned <= 8U) ? 256U : ((tileMAligned <= 16U) ? 128U : 64U);

        AscendC::Duplicate<float>(ubY_, 0.0f, static_cast<int32_t>(tileMAligned));
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t c = colStart; c < colEnd; c += maxBatch) {
            uint32_t curBatch = ((colEnd - c) < maxBatch) ? (colEnd - c) : maxBatch;
            CopyGmToUb1D(ubX_, xGm_[c], curBatch);
            CopyGmToUb2D(ubA_, aGm_[static_cast<int64_t>(c) * static_cast<int64_t>(lda_)],
                         tileM, curBatch, static_cast<uint32_t>(lda_));
            AscendC::PipeBarrier<PIPE_ALL>();
            AccumulateSplitNBatch(tileM, tileMAligned, curBatch, maxBatch);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::SetAtomicAdd<float>();
        CopyUbToGm1D(yGm_, ubY_, tileM);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetAtomicNone();
    }

    __aicore__ inline void ProcessNoTransFastTile(uint32_t rowBase, uint32_t tileM)
    {
        uint32_t tileMAligned = AlignUp8(tileM);
        uint32_t maxBatchN = SafeDivU32(UB_A_FLOATS, tileMAligned);
        if (maxBatchN > FAST_BATCH_COLS) {
            maxBatchN = FAST_BATCH_COLS;
        }
        uint32_t totalN = static_cast<uint32_t>(n_);

        AscendC::Duplicate<float>(ubY_, 0.0f, static_cast<int32_t>(tileMAligned));
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t colBase = 0; colBase < totalN; colBase += maxBatchN) {
            uint32_t curBatchN = ((totalN - colBase) < maxBatchN) ? (totalN - colBase) : maxBatchN;
            CopyGmToUb1D(ubX_, xGm_[colBase], curBatchN);
            int64_t aOffset = static_cast<int64_t>(rowBase) +
                              static_cast<int64_t>(colBase) * static_cast<int64_t>(lda_);
            CopyGmToUb2D(ubA_, aGm_[aOffset], tileM, curBatchN, static_cast<uint32_t>(lda_));
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t k = 0; k < curBatchN; ++k) {
                float xVal = ubX_.GetValue(k);
                AscendC::Axpy(ubY_, ubA_[k * tileMAligned], xVal, static_cast<int32_t>(tileM));
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        CopyUbToGm1D(yGm_[rowBase], ubY_, tileM);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitRowTileY(uint32_t rowBase, uint32_t tileM)
    {
        uint32_t aligned = AlignUp8(tileM);
        if (beta_ == 0.0f) {
            AscendC::Duplicate<float>(ubY_, 0.0f, static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        if (incy_ == 1) {
            CopyGmToUb1D(ubY_, yGm_[rowBase], tileM);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(ubY_, ubY_, beta_, static_cast<int32_t>(aligned));
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t r = 0; r < tileM; ++r) {
            int64_t yIdx = yBase_ + static_cast<int64_t>(rowBase + r) * static_cast<int64_t>(incy_);
            ubY_.SetValue(r, beta_ * yGm_.GetValue(yIdx));
        }
    }

    __aicore__ inline void StoreRowTileY(uint32_t rowBase, uint32_t tileM)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (incy_ == 1) {
            CopyUbToGm1D(yGm_[rowBase], ubY_, tileM);
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t r = 0; r < tileM; ++r) {
            int64_t yIdx = yBase_ + static_cast<int64_t>(rowBase + r) * static_cast<int64_t>(incy_);
            WriteScalarToGm(yIdx, ubY_.GetValue(r));
        }
    }

    __aicore__ inline void LoadScaledXBatch(uint32_t colBase, uint32_t count)
    {
        if (incx_ == 1) {
            CopyGmToUb1D(ubX_, xGm_[colBase], count);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t k = 0; k < count; ++k) {
                ubX_.SetValue(k, alpha_ * ubX_.GetValue(k));
            }
            return;
        }
        for (uint32_t k = 0; k < count; ++k) {
            int64_t xIdx = xBase_ + static_cast<int64_t>(colBase + k) * static_cast<int64_t>(incx_);
            ubX_.SetValue(k, alpha_ * xGm_.GetValue(xIdx));
        }
    }

    __aicore__ inline void ProcessNoTransExactTile(uint32_t rowBase, uint32_t tileM)
    {
        uint32_t tileMAligned = AlignUp8(tileM);
        uint32_t maxBatchN = SafeDivU32(UB_A_FLOATS, tileMAligned);
        if (maxBatchN > FAST_BATCH_COLS) {
            maxBatchN = FAST_BATCH_COLS;
        }
        uint32_t totalN = static_cast<uint32_t>(n_);

        InitRowTileY(rowBase, tileM);

        for (uint32_t colBase = 0; colBase < totalN; colBase += maxBatchN) {
            uint32_t curBatchN = ((totalN - colBase) < maxBatchN) ? (totalN - colBase) : maxBatchN;
            LoadScaledXBatch(colBase, curBatchN);
            int64_t aOffset = static_cast<int64_t>(rowBase) +
                              static_cast<int64_t>(colBase) * static_cast<int64_t>(lda_);
            CopyGmToUb2D(ubA_, aGm_[aOffset], tileM, curBatchN, static_cast<uint32_t>(lda_));
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t k = 0; k < curBatchN; ++k) {
                float temp = ubX_.GetValue(k);
                uint32_t colOff = k * tileMAligned;
                for (uint32_t r = 0; r < tileM; ++r) {
                    float prev = ubY_.GetValue(r);
                    ubY_.SetValue(r, SgemvExactFma(ubA_.GetValue(colOff + r), temp, prev));
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        StoreRowTileY(rowBase, tileM);
    }

    __aicore__ inline void ProcessNoTrans(uint32_t coreIdx)
    {
        uint32_t rowStart = coreIdx * workPerCore_;
        uint32_t totalM = static_cast<uint32_t>(m_);
        if (rowStart >= totalM) {
            return;
        }
        uint32_t coreRows = ((totalM - rowStart) < workPerCore_) ? (totalM - rowStart) : workPerCore_;
        if (alpha_ == 0.0f) {
            ScaleYAlphaZero(rowStart, coreRows);
            return;
        }
        const bool fastVec = CanUseFastVec() && (n_ <= FAST_TRANS_MAX_M);
        for (uint32_t rOff = 0; rOff < coreRows; rOff += UB_VEC_FLOATS) {
            uint32_t tileM = ((coreRows - rOff) < UB_VEC_FLOATS) ? (coreRows - rOff) : UB_VEC_FLOATS;
            if (fastVec) {
                ProcessNoTransFastTile(rowStart + rOff, tileM);
            } else {
                ProcessNoTransExactTile(rowStart + rOff, tileM);
            }
        }
    }

    __aicore__ inline void ReduceFastSubBatch(uint32_t colOff, uint32_t subCols, uint32_t tileM,
                                              uint32_t tileMAligned)
    {
        for (uint32_t c = 0; c < subCols; ++c) {
            AscendC::LocalTensor<float> aCol = ubA_[c * tileMAligned];
            AscendC::Mul(aCol, aCol, ubX_, static_cast<int32_t>(tileM));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum<float>(ubWork_, aCol, ubWork_[F32_BLOCK_LEN], static_cast<int32_t>(tileM));
            AscendC::PipeBarrier<PIPE_ALL>();
            ubY_.SetValue(colOff + c, ubY_.GetValue(colOff + c) + ubWork_.GetValue(0));
        }
    }

    __aicore__ inline void ProcessTransFastBatch(uint32_t colBase, uint32_t curCols)
    {
        uint32_t totalM = static_cast<uint32_t>(m_);
        AscendC::Duplicate<float>(ubY_, 0.0f, static_cast<int32_t>(AlignUp8(curCols)));
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t rowBase = 0; rowBase < totalM; rowBase += UB_VEC_FLOATS) {
            uint32_t tileM = ((totalM - rowBase) < UB_VEC_FLOATS) ? (totalM - rowBase) : UB_VEC_FLOATS;
            uint32_t tileMAligned = AlignUp8(tileM);
            uint32_t maxSubCols = SafeDivU32(UB_A_FLOATS, tileMAligned);
            CopyGmToUb1D(ubX_, xGm_[rowBase], tileM);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t cOff = 0; cOff < curCols; cOff += maxSubCols) {
                uint32_t subCols = ((curCols - cOff) < maxSubCols) ? (curCols - cOff) : maxSubCols;
                int64_t aOffset = static_cast<int64_t>(rowBase) +
                                  static_cast<int64_t>(colBase + cOff) * static_cast<int64_t>(lda_);
                CopyGmToUb2D(ubA_, aGm_[aOffset], tileM, subCols, static_cast<uint32_t>(lda_));
                AscendC::PipeBarrier<PIPE_ALL>();
                ReduceFastSubBatch(cOff, subCols, tileM, tileMAligned);
            }
        }

        CopyUbToGm1D(yGm_[colBase], ubY_, curCols);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadRawXTile(uint32_t rowBase, uint32_t tileM)
    {
        if (incx_ == 1) {
            CopyGmToUb1D(ubX_, xGm_[rowBase], tileM);
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t i = 0; i < tileM; ++i) {
            int64_t xIdx = xBase_ + static_cast<int64_t>(rowBase + i) * static_cast<int64_t>(incx_);
            ubX_.SetValue(i, xGm_.GetValue(xIdx));
        }
    }

    __aicore__ inline void AccumulateTransUnitX32(uint32_t colOff, uint32_t limit32, float (&v16)[16])
    {
        for (uint32_t i = 0; i < limit32; i += 32U) {
            for (uint32_t k = 0; k < 16U; ++k) {
                v16[k] = SgemvExactFma(ubA_.GetValue(colOff + i + k), ubX_.GetValue(i + k), v16[k]);
            }
            for (uint32_t k = 0; k < 16U; ++k) {
                v16[k] = SgemvExactFma(ubA_.GetValue(colOff + i + 16U + k), ubX_.GetValue(i + 16U + k), v16[k]);
            }
        }
    }

    __aicore__ inline float FinalizeTransUnitXCol(uint32_t colOff, uint32_t tileM, float (&v16)[16])
    {
        uint32_t i = tileM & ~31U;
        AccumulateTransUnitX32(colOff, i, v16);
        if (m_ >= 32) {
            for (uint32_t k = 0; k < 4U; ++k) {
                v16[k] = ((v16[k] + v16[k + 4U]) + v16[k + 8U]) + v16[k + 12U];
            }
        }
        for (; (i + 3U) < tileM; i += 4U) {
            for (uint32_t k = 0; k < 4U; ++k) {
                v16[k] = SgemvExactFma(ubA_.GetValue(colOff + i + k), ubX_.GetValue(i + k), v16[k]);
            }
        }
        float colAcc = (v16[0] + v16[2]) + (v16[1] + v16[3]);
        for (; i < tileM; ++i) {
            colAcc = SgemvExactFma(ubA_.GetValue(colOff + i), ubX_.GetValue(i), colAcc);
        }
        return colAcc;
    }

    __aicore__ inline void WriteTransColOutput(uint32_t colIdx, float colAcc)
    {
        int64_t yIdx = yBase_ + static_cast<int64_t>(colIdx) * static_cast<int64_t>(incy_);
        float yTerm = (beta_ == 0.0f) ? 0.0f : (beta_ * yGm_.GetValue(yIdx));
        WriteScalarToGm(yIdx, SgemvExactFma(alpha_, colAcc, yTerm));
    }

    __aicore__ inline void ProcessTransExactBatch(uint32_t colBase, uint32_t subCols)
    {
        float v16[EXACT_TRANS_BATCH][16] = {{0.0f}};
        float sAcc[EXACT_TRANS_BATCH] = {0.0f};
        uint32_t totalM = static_cast<uint32_t>(m_);

        for (uint32_t rowBase = 0; rowBase < totalM; rowBase += UB_VEC_FLOATS) {
            uint32_t tileM = ((totalM - rowBase) < UB_VEC_FLOATS) ? (totalM - rowBase) : UB_VEC_FLOATS;
            uint32_t tileMAligned = AlignUp8(tileM);
            const bool isLastTile = ((rowBase + tileM) >= totalM);

            LoadRawXTile(rowBase, tileM);
            int64_t aOffset = static_cast<int64_t>(rowBase) +
                              static_cast<int64_t>(colBase) * static_cast<int64_t>(lda_);
            CopyGmToUb2D(ubA_, aGm_[aOffset], tileM, subCols, static_cast<uint32_t>(lda_));
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t c = 0; c < subCols; ++c) {
                uint32_t colOff = c * tileMAligned;
                if (incx_ != 1) {
                    for (uint32_t i = 0; i < tileM; ++i) {
                        sAcc[c] = SgemvExactFma(ubA_.GetValue(colOff + i), ubX_.GetValue(i), sAcc[c]);
                    }
                } else if (!isLastTile) {
                    AccumulateTransUnitX32(colOff, tileM, v16[c]);
                } else {
                    sAcc[c] = FinalizeTransUnitXCol(colOff, tileM, v16[c]);
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        for (uint32_t c = 0; c < subCols; ++c) {
            WriteTransColOutput(colBase + c, sAcc[c]);
        }
    }

    __aicore__ inline void ProcessTrans(uint32_t coreIdx)
    {
        uint32_t colStart = coreIdx * workPerCore_;
        uint32_t totalN = static_cast<uint32_t>(n_);
        if (colStart >= totalN) {
            return;
        }
        uint32_t coreCols = ((totalN - colStart) < workPerCore_) ? (totalN - colStart) : workPerCore_;
        if (alpha_ == 0.0f) {
            ScaleYAlphaZero(colStart, coreCols);
            return;
        }
        const bool fastVec = CanUseFastVec() && (m_ >= FAST_TRANS_MIN_M) && (m_ <= FAST_TRANS_MAX_M);
        if (fastVec) {
            for (uint32_t cOff = 0; cOff < coreCols; cOff += FAST_BATCH_COLS) {
                uint32_t curCols = ((coreCols - cOff) < FAST_BATCH_COLS) ? (coreCols - cOff) : FAST_BATCH_COLS;
                ProcessTransFastBatch(colStart + cOff, curCols);
            }
            return;
        }
        for (uint32_t cOff = 0; cOff < coreCols; cOff += EXACT_TRANS_BATCH) {
            uint32_t subCols = ((coreCols - cOff) < EXACT_TRANS_BATCH) ? (coreCols - cOff) : EXACT_TRANS_BATCH;
            ProcessTransExactBatch(colStart + cOff, subCols);
        }
    }

    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> dummy_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf_;
    AscendC::GlobalTensor<float> aGm_;
    AscendC::GlobalTensor<float> xGm_;
    AscendC::GlobalTensor<float> yGm_;
    AscendC::LocalTensor<float> ubA_;
    AscendC::LocalTensor<float> ubX_;
    AscendC::LocalTensor<float> ubY_;
    AscendC::LocalTensor<float> ubWork_;
    int64_t xBase_ = 0;
    int64_t yBase_ = 0;
    uint32_t kernelMode_ = 0;
    uint32_t usedCoreNum_ = 1;
    uint32_t workPerCore_ = 1;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t lda_ = 1;
    int32_t incx_ = 1;
    int32_t incy_ = 1;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};
} // namespace

extern "C" __global__ __aicore__ void sgemv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, SgemvTilingData tiling)
{
    AscendC::TPipe pipe;
    SgemvKernel op;
    op.Init(a, x, y, tiling, &pipe);
    op.Process();
}

void aclblasSgemv_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *x, uint8_t *y, SgemvTilingData *tiling)
{
    sgemv_kernel<<<blockDim, nullptr, stream>>>(a, x, y, *tiling);
}
