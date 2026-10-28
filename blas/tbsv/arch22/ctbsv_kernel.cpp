/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "ctbsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "ctbsv_complex_arith.h"

using namespace AscendC;

enum class CtbsvUplo { UPPER, LOWER };
enum class CtbsvTrans { NO_TRANS, TRANS, CONJ };
enum class CtbsvDiag { UNIT, NON_UNIT };

// Ascend C DMA blocks are 32 bytes; one FP32 vector repeat is 256 bytes.
constexpr uint32_t CTBSV_DMA_BLOCK_BYTES = 32U;
constexpr uint32_t CTBSV_VECTOR_BYTES = 256U;
constexpr uint32_t CTBSV_FLOATS_PER_BLOCK = CTBSV_DMA_BLOCK_BYTES / sizeof(float);
constexpr uint32_t CTBSV_BLOCK_ALIGN_BIAS = CTBSV_FLOATS_PER_BLOCK - 1U;
constexpr uint32_t CTBSV_VECTOR_PACK_MIN = 64;
constexpr uint32_t CTBSV_FLOAT_BYTE_SHIFT = 2; // sizeof(float) == 1 << 2
// Preserve the existing scalar/vector crossover points and buffer padding.
constexpr uint32_t CTBSV_TRANS_VECTOR_MIN = 8U;
constexpr uint32_t CTBSV_UNIT_VECTOR_MIN = 8U;
constexpr uint32_t CTBSV_NON_UNIT_VECTOR_MIN = 16U;
constexpr uint32_t CTBSV_SHORT_UNIT_VECTOR_MAX = 16U;
constexpr uint32_t CTBSV_REDUCE_WORK_PAD_FLOATS = 256U;
constexpr uint32_t CTBSV_REDUCE_OUTPUT_FLOATS = 32U;
constexpr uint32_t CTBSV_REFERENCE_MIN_N = 1024U;
constexpr uint32_t CTBSV_SMALL_AIC_MAX_N = 2U;
// Large intermediates use explicit products instead of the ordinary Axpy path.
constexpr float CTBSV_LARGE_INTERMEDIATE_MAGNITUDE = 1.0e30f;

// A nonzero band with one RHS has dependencies between substitution steps.
// Keep that chain on one core; use vector instructions within each band update.
// Simply increasing blockDim would duplicate the solve and race on in-place x.
// The diagonal-only case currently reuses this dispatch and Smith division.
constexpr uint32_t CTBSV_SOLVE_BLOCKS = 1;

__aicore__ inline uint32_t CtbsvAlignBlock(uint32_t n)
{
    return (n + CTBSV_FLOATS_PER_BLOCK - 1U) / CTBSV_FLOATS_PER_BLOCK * CTBSV_FLOATS_PER_BLOCK;
}

__aicore__ inline void CtbsvCopyGm2UbIssue(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t gmOff, uint32_t count)
{
    const uint32_t aligned = (count / CTBSV_FLOATS_PER_BLOCK) * CTBSV_FLOATS_PER_BLOCK;
    const uint32_t tail = count - aligned;
    if (aligned > 0U) {
        DataCopy(dst, src[gmOff], aligned);
    }
    if (tail > 0U) {
        const uint8_t pad = static_cast<uint8_t>(CTBSV_FLOATS_PER_BLOCK - tail);
        DataCopyExtParams params{1, static_cast<uint32_t>(tail * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, pad, 0.0f};
        DataCopyPad(dst[aligned], src[gmOff + static_cast<int64_t>(aligned)], params, padParams);
    }
}

template <bool SINGLE_COPY>
__aicore__ inline void CtbsvCopyAColGm2UbIssue(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t gmOff, uint32_t count)
{
    if constexpr (SINGLE_COPY) {
        const uint32_t tail = count % CTBSV_FLOATS_PER_BLOCK;
        // Replace only the two-command case. Keep aligned and tiny copies.
        if (count > CTBSV_FLOATS_PER_BLOCK && tail != 0U) {
            const uint8_t pad = static_cast<uint8_t>(CTBSV_FLOATS_PER_BLOCK - tail);
            DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, pad, 0.0f};
            DataCopyPad(dst, src[gmOff], params, padParams);
            return;
        }
    }
    CtbsvCopyGm2UbIssue(dst, src, gmOff, count);
}

__aicore__ inline void CtbsvCopyUb2Gm(
    const GlobalTensor<float>& dst, int64_t gmOff, const LocalTensor<float>& src, uint32_t count)
{
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    const uint32_t aligned = (count / CTBSV_FLOATS_PER_BLOCK) * CTBSV_FLOATS_PER_BLOCK;
    const uint32_t tail = count - aligned;
    if (aligned > 0U) {
        DataCopy(dst[gmOff], src, aligned);
    }
    if (tail > 0U) {
        DataCopyExtParams params{1, static_cast<uint32_t>(tail * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst[gmOff + static_cast<int64_t>(aligned)], src[aligned], params);
    }
}

__aicore__ inline int64_t CtbsvXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    } else {
        return static_cast<int64_t>(n - 1 - idx) * static_cast<int64_t>(-incx);
    }
}

__aicore__ inline int64_t CtbsvComplexFloatIdx(int64_t complexIdx) { return complexIdx * 2; }

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
class CtbsvGmKernel {
public:
    __aicore__ inline void Init(const CtbsvTilingData& tiling)
    {
        n = tiling.n;
        k = tiling.k;
        lda = tiling.lda;
        incx = tiling.incx;
        const int64_t aCount = static_cast<int64_t>(lda) * static_cast<int64_t>(n) * 2;
        const int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
        const int64_t xCount = (absIncx * static_cast<int64_t>(n - 1U) + 1) * 2;
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), aCount);
        xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
    }

    __aicore__ inline void Process()
    {
        if constexpr (TRANS == CtbsvTrans::NO_TRANS) {
            ProcessNoTrans();
        } else {
            ProcessTrans();
        }
    }

private:
    __aicore__ inline int64_t XFloatOffset(uint32_t logicalIndex) const
    {
        return CtbsvComplexFloatIdx(CtbsvXOffset(logicalIndex, n, incx));
    }

    __aicore__ inline int64_t AFloatOffset(uint32_t row, uint32_t col) const
    {
        const uint32_t bandRow = (UPLO == CtbsvUplo::UPPER) ? (k + row - col) : (row - col);
        return (static_cast<int64_t>(bandRow) + static_cast<int64_t>(col) * static_cast<int64_t>(lda)) * 2;
    }

    __aicore__ inline void ReadX(uint32_t index, float& real, float& imag) const
    {
        const int64_t offset = XFloatOffset(index);
        real = xGM.GetValue(offset);
        imag = xGM.GetValue(offset + 1);
    }

    __aicore__ inline void WriteX(uint32_t index, float real, float imag)
    {
        const int64_t offset = XFloatOffset(index);
        xGM.SetValue(offset, real);
        xGM.SetValue(offset + 1, imag);
    }

    __aicore__ inline void ReadA(uint32_t row, uint32_t col, float& real, float& imag) const
    {
        const int64_t offset = AFloatOffset(row, col);
        real = aGM.GetValue(offset);
        imag = aGM.GetValue(offset + 1);
    }

    __aicore__ inline void DivideByDiag(uint32_t index, float& real, float& imag) const
    {
        float diagReal = 0.0f;
        float diagImag = 0.0f;
        ReadA(index, index, diagReal, diagImag);
        float outReal = 0.0f;
        float outImag = 0.0f;
        if constexpr (TRANS == CtbsvTrans::CONJ) {
            CtbsvComplexDivByConj(real, imag, diagReal, diagImag, outReal, outImag);
        } else {
            CtbsvComplexDiv(real, imag, diagReal, diagImag, outReal, outImag);
        }
        real = outReal;
        imag = outImag;
    }

    __aicore__ inline void Scatter(uint32_t row, uint32_t col, float valueReal, float valueImag)
    {
        float aReal = 0.0f;
        float aImag = 0.0f;
        ReadA(row, col, aReal, aImag);
        float productReal = 0.0f;
        float productImag = 0.0f;
        CtbsvCmul(valueReal, valueImag, aReal, aImag, productReal, productImag);
        float xReal = 0.0f;
        float xImag = 0.0f;
        ReadX(row, xReal, xImag);
        WriteX(row, xReal - productReal, xImag - productImag);
    }

    __aicore__ inline void AccumulateTrans(uint32_t row, uint32_t col, float& valueReal, float& valueImag) const
    {
        float aReal = 0.0f;
        float aImag = 0.0f;
        ReadA(row, col, aReal, aImag);
        if constexpr (TRANS == CtbsvTrans::CONJ) {
            aImag = -aImag;
        }
        float xReal = 0.0f;
        float xImag = 0.0f;
        ReadX(row, xReal, xImag);
        float productReal = 0.0f;
        float productImag = 0.0f;
        CtbsvCmul(aReal, aImag, xReal, xImag, productReal, productImag);
        valueReal -= productReal;
        valueImag -= productImag;
    }

    __aicore__ inline void ProcessNoTransColumn(uint32_t col)
    {
        float valueReal = 0.0f;
        float valueImag = 0.0f;
        ReadX(col, valueReal, valueImag);
        if (CtbsvIsCzero(valueReal, valueImag)) {
            return;
        }
        if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
            DivideByDiag(col, valueReal, valueImag);
            WriteX(col, valueReal, valueImag);
        }
        if constexpr (UPLO == CtbsvUplo::LOWER) {
            const uint32_t rowEnd = (col + k + 1U < n) ? (col + k + 1U) : n;
            for (uint32_t row = col + 1U; row < rowEnd; ++row) {
                Scatter(row, col, valueReal, valueImag);
            }
        } else {
            const uint32_t rowStart = (col >= k) ? (col - k) : 0U;
            for (uint32_t row = col; row-- > rowStart;) {
                Scatter(row, col, valueReal, valueImag);
            }
        }
    }

    __aicore__ inline void ProcessNoTrans()
    {
        if constexpr (UPLO == CtbsvUplo::LOWER) {
            for (uint32_t col = 0; col < n; ++col) {
                ProcessNoTransColumn(col);
            }
        } else {
            for (uint32_t col = n; col-- > 0;) {
                ProcessNoTransColumn(col);
            }
        }
    }

    __aicore__ inline void ProcessTransColumn(uint32_t col)
    {
        float valueReal = 0.0f;
        float valueImag = 0.0f;
        ReadX(col, valueReal, valueImag);
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            const uint32_t rowStart = (col >= k) ? (col - k) : 0U;
            for (uint32_t row = rowStart; row < col; ++row) {
                AccumulateTrans(row, col, valueReal, valueImag);
            }
        } else {
            const uint32_t rowEnd = (col + k + 1U < n) ? (col + k + 1U) : n;
            for (uint32_t row = col + 1U; row < rowEnd; ++row) {
                AccumulateTrans(row, col, valueReal, valueImag);
            }
        }
        if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
            DivideByDiag(col, valueReal, valueImag);
        }
        WriteX(col, valueReal, valueImag);
    }

    __aicore__ inline void ProcessTrans()
    {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            for (uint32_t col = 0; col < n; ++col) {
                ProcessTransColumn(col);
            }
        } else {
            for (uint32_t col = n; col-- > 0;) {
                ProcessTransColumn(col);
            }
        }
    }

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    int32_t incx;
};

struct CtbsvUbBuffers {
    TBuf<TPosition::VECCALC> xUbRe;
    TBuf<TPosition::VECCALC> xPack;
    TBuf<TPosition::VECCALC> aColPing;
    TBuf<TPosition::VECCALC> aColPong;
    TBuf<TPosition::VECCALC> aRe;
    TBuf<TPosition::VECCALC> aIm;
    TBuf<TPosition::VECCALC> prodRe;
    TBuf<TPosition::VECCALC> prodIm;
    TBuf<TPosition::VECCALC> vecTmp;
    TBuf<TPosition::VECCALC> vecWork;
    TBuf<TPosition::VECCALC> vecOut;
    TBuf<TPosition::VECCALC> gatherOff;
    TBuf<TPosition::VECCALC> complexGatherOff;
};

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
class CtbsvKernel {
public:
    __aicore__ inline CtbsvKernel() {}
    __aicore__ inline void Init(const CtbsvTilingData& tiling);
    __aicore__ inline void Process(TPipe& pipe);

private:
    __aicore__ inline void ProcessRows(TPipe& pipe);
    __aicore__ inline void InitPrimaryBuffers(
        TPipe& pipe, CtbsvUbBuffers& buffers, LocalTensor<float>& xUbRe, LocalTensor<float>& xUbIm);
    __aicore__ inline uint32_t InitWorkBuffers(TPipe& pipe, CtbsvUbBuffers& buffers);
    __aicore__ inline void InitGatherOffsets(uint32_t pairPad);
    __aicore__ inline void RunSolve(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void LoadXUb(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void StoreXUb(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void PackXVector(const LocalTensor<float>& xUbRe);
    __aicore__ inline void PackXScalar(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void GetDiag(uint32_t row, float& diagRe, float& diagIm) const;
    __aicore__ inline void DivByDiag(uint32_t j, float& tRe, float& tIm) const;
    __aicore__ inline bool ScaleNoTransCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t j, float& tRe, float& tIm);
    __aicore__ inline void ScatterCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float tRe, float tIm);
    __aicore__ inline void ScatterColReference(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float tRe, float tIm);
    __aicore__ inline void ScatterColVector(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
        float tRe, float tIm);
    __aicore__ inline void ScatterColSmallVector(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
        float tRe, float tIm);
    __aicore__ inline void GatherTransTerm(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float& tRe,
        float& tIm);
    __aicore__ inline void GatherTransTermReference(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float& tRe,
        float& tIm);
    __aicore__ inline void StoreTransCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t j, float tRe, float tIm);
    __aicore__ inline void ProcessNoTransUpper(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessNoTransLower(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessTransUpper(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessTransLower(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline int64_t XOffset(uint32_t idx) const;
    __aicore__ inline void IssueACol(const LocalTensor<float>& dst, uint32_t j, bool ping);
    __aicore__ inline void WaitACol(bool ping);
    __aicore__ inline void LoadACol(uint32_t j);
    __aicore__ inline void BindACol(bool usePing);
    __aicore__ inline void BindAColVec(bool usePing);
    __aicore__ inline void PrefetchACol(uint32_t j, bool usePing);
    __aicore__ inline void GetAFromCol(uint32_t row, uint32_t col, float& aRe, float& aIm) const;
    __aicore__ inline void GetOpAFromCol(uint32_t row, uint32_t col, float& aRe, float& aIm) const;
    __aicore__ inline void GatherTransVec(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
        float& tRe, float& tIm);

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    LocalTensor<float> aColUb;
    LocalTensor<float> aColPing;
    LocalTensor<float> aColPong;
    LocalTensor<float> xPackUb;
    LocalTensor<float> aReUb;
    LocalTensor<float> aImUb;
    LocalTensor<float> prodReUb;
    LocalTensor<float> prodImUb;
    LocalTensor<float> vecTmpUb;
    LocalTensor<float> vecWorkUb;
    LocalTensor<float> vecOutUb;
    LocalTensor<uint32_t> gatherOffUb;
    LocalTensor<uint32_t> complexGatherOffUb;
    uint32_t aColAlign;
    uint32_t complexGatherCount;
    uint32_t xPlaneStride;

    uint32_t n;
    uint32_t k;
    uint32_t lda;
    int32_t incx;
};

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline int64_t CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::XOffset(uint32_t idx) const
{
    return CtbsvXOffset(idx, n, incx);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::Init(const CtbsvTilingData& tiling)
{
    this->n = tiling.n;
    this->k = tiling.k;
    this->lda = tiling.lda;
    this->incx = tiling.incx;

    int64_t aCount = static_cast<int64_t>(tiling.lda) * static_cast<int64_t>(n) * 2;
    int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    int64_t xCount = (n > 0) ? (absIncx * static_cast<int64_t>(n - 1) + 1) * 2 : 0;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), aCount);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::IssueACol(
    const LocalTensor<float>& dst, uint32_t j, bool ping)
{
    uint32_t firstBandRow = 0U;
    uint32_t lastBandRow = k;
    if constexpr (UPLO == CtbsvUplo::UPPER) {
        firstBandRow = (j < k) ? (k - j) : 0U;
    } else {
        lastBandRow = (j + k < n) ? k : (n - 1U - j);
    }
    const uint32_t firstFloat = firstBandRow * 2U / CTBSV_FLOATS_PER_BLOCK * CTBSV_FLOATS_PER_BLOCK;
    const uint32_t floatCount = (lastBandRow + 1U) * 2U - firstFloat;
    const int64_t gmOff = static_cast<int64_t>(j) * static_cast<int64_t>(lda) * 2 + firstFloat;
    constexpr bool singleCopy = DIAG == CtbsvDiag::UNIT && !REFERENCE;
    CtbsvCopyAColGm2UbIssue<singleCopy>(dst[firstFloat], aGM, gmOff, floatCount);
    constexpr bool scalarReferenceRead = REFERENCE && TRANS != CtbsvTrans::NO_TRANS;
    if (ping) {
        if constexpr (DIAG == CtbsvDiag::NON_UNIT || TRANS == CtbsvTrans::NO_TRANS) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else if (scalarReferenceRead) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    } else {
        if constexpr (DIAG == CtbsvDiag::NON_UNIT || TRANS == CtbsvTrans::NO_TRANS) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        } else if (scalarReferenceRead) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::WaitACol(bool ping)
{
    constexpr bool scalarReferenceRead = REFERENCE && TRANS != CtbsvTrans::NO_TRANS;
    if (ping) {
        if constexpr (DIAG == CtbsvDiag::NON_UNIT || TRANS == CtbsvTrans::NO_TRANS) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        } else if (scalarReferenceRead) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    } else {
        if constexpr (DIAG == CtbsvDiag::NON_UNIT || TRANS == CtbsvTrans::NO_TRANS) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        } else if (scalarReferenceRead) {
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::LoadACol(uint32_t j)
{
    IssueACol(aColUb, j, true);
    WaitACol(true);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::BindACol(bool usePing)
{
    WaitACol(usePing);
    aColUb = usePing ? aColPing : aColPong;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::BindAColVec(bool usePing)
{
    BindACol(usePing);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::PrefetchACol(uint32_t j, bool usePing)
{
    IssueACol(usePing ? aColPing : aColPong, j, usePing);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GatherTransVec(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
    float& tRe, float& tIm)
{
    const uint32_t count = iEnd - iStart;
    if (count == 0U) {
        return;
    }
    // The wide-band UNIT reference path preserves CBLAS scalar rounding order;
    // ordinary bands below use vector products/reductions, not this path.
    if constexpr (REFERENCE) {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            for (uint32_t i = iStart; i < iEnd; ++i) {
                GatherTransTermReference(xUbRe, xUbIm, i, j, tRe, tIm);
            }
        } else {
            for (uint32_t i = iEnd; i-- > iStart;) {
                GatherTransTermReference(xUbRe, xUbIm, i, j, tRe, tIm);
            }
        }
        return;
    }
    if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
        if (count < CTBSV_TRANS_VECTOR_MIN) {
            for (uint32_t i = iStart; i < iEnd; ++i) {
                GatherTransTerm(xUbRe, xUbIm, i, j, tRe, tIm);
            }
            return;
        }
    }
    const uint32_t bandOff = (UPLO == CtbsvUplo::UPPER) ? (k + iStart - j) : (iStart - j);
    const uint32_t aBase = bandOff * 2U * static_cast<uint32_t>(sizeof(float));
    const uint32_t xBase = iStart * static_cast<uint32_t>(sizeof(float));
    Gather(prodReUb, aColUb, complexGatherOffUb, aBase, count);
    Gather(prodImUb, aColUb, complexGatherOffUb, aBase + static_cast<uint32_t>(sizeof(float)), count);
    const bool xAligned = (iStart % CTBSV_FLOATS_PER_BLOCK) == 0U;
    if (!xAligned) {
        Gather(vecTmpUb, xUbRe, gatherOffUb, xBase, count);
        Gather(aReUb, xUbIm, gatherOffUb, xBase, count);
    }
    PipeBarrier<PIPE_V>();
    const int32_t nElem = static_cast<int32_t>(count);
    if (xAligned) {
        Mul(vecWorkUb, prodReUb, xUbRe[iStart], nElem);
        Mul(aImUb, prodReUb, xUbIm[iStart], nElem);
        if constexpr (TRANS == CtbsvTrans::CONJ) {
            Muls(vecTmpUb, xUbRe[iStart], -1.0f, nElem);
            MulAddDst(vecWorkUb, prodImUb, xUbIm[iStart], nElem);
            MulAddDst(aImUb, prodImUb, vecTmpUb, nElem);
        } else {
            Muls(aReUb, xUbIm[iStart], -1.0f, nElem);
            MulAddDst(vecWorkUb, prodImUb, aReUb, nElem);
            MulAddDst(aImUb, prodImUb, xUbRe[iStart], nElem);
        }
    } else {
        Mul(vecWorkUb, prodReUb, vecTmpUb, nElem);
        if constexpr (TRANS == CtbsvTrans::CONJ) {
            Mul(aImUb, prodReUb, aReUb, nElem);
            Muls(vecTmpUb, vecTmpUb, -1.0f, nElem);
            MulAddDst(vecWorkUb, prodImUb, aReUb, nElem);
            MulAddDst(aImUb, prodImUb, vecTmpUb, nElem);
        } else {
            Mul(aImUb, prodReUb, aReUb, nElem);
            Muls(aReUb, aReUb, -1.0f, nElem);
            MulAddDst(vecWorkUb, prodImUb, aReUb, nElem);
            MulAddDst(aImUb, prodImUb, vecTmpUb, nElem);
        }
    }
    ReduceSum<float>(vecOutUb, vecWorkUb, aReUb, nElem);
    ReduceSum<float>(vecOutUb[CTBSV_FLOATS_PER_BLOCK], aImUb, aReUb, nElem);
    tRe -= vecOutUb.GetValue(0);
    tIm -= vecOutUb.GetValue(CTBSV_FLOATS_PER_BLOCK);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GetAFromCol(
    uint32_t row, uint32_t col, float& aRe, float& aIm) const
{
    const uint32_t bandRow = (UPLO == CtbsvUplo::UPPER) ? (k + row - col) : (row - col);
    const uint32_t fIdx = bandRow * 2U;
    aRe = aColUb.GetValue(fIdx);
    aIm = aColUb.GetValue(fIdx + 1U);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GetOpAFromCol(
    uint32_t row, uint32_t col, float& aRe, float& aIm) const
{
    GetAFromCol(row, col, aRe, aIm);
    if constexpr (TRANS == CtbsvTrans::CONJ) {
        aIm = -aIm;
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GetDiag(
    uint32_t row, float& diagRe, float& diagIm) const
{
    (void)row;
    const uint32_t bandRow = (UPLO == CtbsvUplo::UPPER) ? k : 0U;
    const uint32_t fIdx = bandRow * 2U;
    diagRe = aColUb.GetValue(fIdx);
    diagIm = aColUb.GetValue(fIdx + 1U);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::DivByDiag(uint32_t j, float& tRe, float& tIm) const
{
    float diagRe = 0.0f;
    float diagIm = 0.0f;
    GetDiag(j, diagRe, diagIm);
    float outRe = 0.0f;
    float outIm = 0.0f;
    if constexpr (TRANS == CtbsvTrans::CONJ) {
        CtbsvComplexDivByConj(tRe, tIm, diagRe, diagIm, outRe, outIm);
    } else {
        CtbsvComplexDiv(tRe, tIm, diagRe, diagIm, outRe, outIm);
    }
    tRe = outRe;
    tIm = outIm;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline bool CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ScaleNoTransCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t j, float& tRe, float& tIm)
{
    tRe = xUbRe.GetValue(j);
    tIm = xUbIm.GetValue(j);
    if (CtbsvIsCzero(tRe, tIm)) {
        return false;
    }
    if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
        DivByDiag(j, tRe, tIm);
        xUbRe.SetValue(j, tRe);
        xUbIm.SetValue(j, tIm);
    }
    return true;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ScatterCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float tRe, float tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetAFromCol(i, j, aRe, aIm);
    CtbsvCmul(tRe, tIm, aRe, aIm, pRe, pIm);
    xUbRe.SetValue(i, xUbRe.GetValue(i) - pRe);
    xUbIm.SetValue(i, xUbIm.GetValue(i) - pIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ScatterColReference(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float tRe, float tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetAFromCol(i, j, aRe, aIm);
    CtbsvCmulReferenceSoft(tRe, tIm, aRe, aIm, pRe, pIm);
    xUbRe.SetValue(i, xUbRe.GetValue(i) - pRe);
    xUbIm.SetValue(i, xUbIm.GetValue(i) - pIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ScatterColSmallVector(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
    float tRe, float tIm)
{
    const uint32_t count = iEnd - iStart;
    const uint32_t bandStart = (UPLO == CtbsvUplo::UPPER) ? (k + iStart - j) : (iStart - j);
    const uint32_t packedBase = bandStart * 2U * static_cast<uint32_t>(sizeof(float));
    const uint32_t xBase = iStart * static_cast<uint32_t>(sizeof(float));
    Gather(aReUb, aColUb, complexGatherOffUb, packedBase, count);
    Gather(aImUb, aColUb, complexGatherOffUb, packedBase + static_cast<uint32_t>(sizeof(float)), count);
    Gather(prodReUb, xUbRe, gatherOffUb, xBase, count);
    Gather(prodImUb, xUbIm, gatherOffUb, xBase, count);
    PipeBarrier<PIPE_V>();
    Axpy(prodReUb, aReUb, -tRe, static_cast<int32_t>(count));
    Axpy(prodImUb, aReUb, -tIm, static_cast<int32_t>(count));
    PipeBarrier<PIPE_V>();
    Axpy(prodReUb, aImUb, tIm, static_cast<int32_t>(count));
    Axpy(prodImUb, aImUb, -tRe, static_cast<int32_t>(count));
    SetFlag<HardEvent::V_S>(EVENT_ID4);
    WaitFlag<HardEvent::V_S>(EVENT_ID4);
    for (uint32_t i = 0; i < count; ++i) {
        xUbRe.SetValue(iStart + i, prodReUb.GetValue(i));
        xUbIm.SetValue(iStart + i, prodImUb.GetValue(i));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ScatterColVector(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t iStart, uint32_t iEnd, uint32_t j,
    float tRe, float tIm)
{
    if constexpr (REFERENCE) {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            for (uint32_t i = iEnd; i-- > iStart;) {
                ScatterColReference(xUbRe, xUbIm, i, j, tRe, tIm);
            }
        } else {
            for (uint32_t i = iStart; i < iEnd; ++i) {
                ScatterColReference(xUbRe, xUbIm, i, j, tRe, tIm);
            }
        }
        return;
    }
    // Full-band unit-triangular systems can grow beyond FP32 range. Preserve
    // the BLAS column-update order in that numerically sensitive case.
    if (k + 1U >= n) {
        for (uint32_t i = iStart; i < iEnd; ++i) {
            ScatterCol(xUbRe, xUbIm, i, j, tRe, tIm);
        }
        return;
    }
    const uint32_t totalCount = iEnd - iStart;
    if constexpr (DIAG == CtbsvDiag::UNIT) {
        if (totalCount >= CTBSV_UNIT_VECTOR_MIN && totalCount <= CTBSV_SHORT_UNIT_VECTOR_MAX) {
            const uint32_t alignedStart =
                (iStart + CTBSV_BLOCK_ALIGN_BIAS) / CTBSV_FLOATS_PER_BLOCK * CTBSV_FLOATS_PER_BLOCK;
            if (alignedStart >= iEnd || iEnd - alignedStart < CTBSV_UNIT_VECTOR_MIN) {
                ScatterColSmallVector(xUbRe, xUbIm, iStart, iEnd, j, tRe, tIm);
                return;
            }
        }
    }
    while (iStart < iEnd && (iStart % CTBSV_FLOATS_PER_BLOCK) != 0U) {
        ScatterCol(xUbRe, xUbIm, iStart, j, tRe, tIm);
        ++iStart;
    }
    const uint32_t count = iEnd - iStart;
    constexpr uint32_t minVectorCount = (DIAG == CtbsvDiag::UNIT) ? CTBSV_UNIT_VECTOR_MIN : CTBSV_NON_UNIT_VECTOR_MIN;
    if (count < minVectorCount) {
        for (uint32_t i = iStart; i < iEnd; ++i) {
            ScatterCol(xUbRe, xUbIm, i, j, tRe, tIm);
        }
        return;
    }

    const uint32_t bandStart = (UPLO == CtbsvUplo::UPPER) ? (k + iStart - j) : (iStart - j);
    const uint32_t packedBase = bandStart * 2U * static_cast<uint32_t>(sizeof(float));
    Gather(aReUb, aColUb, complexGatherOffUb, packedBase, count);
    Gather(aImUb, aColUb, complexGatherOffUb, packedBase + static_cast<uint32_t>(sizeof(float)), count);
    PipeBarrier<PIPE_V>();

    const int32_t vectorCount = static_cast<int32_t>(count);
    if (CtbsvAbsf(tRe) > CTBSV_LARGE_INTERMEDIATE_MAGNITUDE || CtbsvAbsf(tIm) > CTBSV_LARGE_INTERMEDIATE_MAGNITUDE) {
        Muls(prodReUb, aReUb, tRe, vectorCount);
        Muls(prodImUb, aImUb, tRe, vectorCount);
        Muls(vecTmpUb, aReUb, tIm, vectorCount);
        Muls(vecWorkUb, aImUb, tIm, vectorCount);
        PipeBarrier<PIPE_V>();
        Sub(prodReUb, prodReUb, vecWorkUb, vectorCount);
        Add(prodImUb, prodImUb, vecTmpUb, vectorCount);
        PipeBarrier<PIPE_V>();
        Sub(xUbRe[iStart], xUbRe[iStart], prodReUb, vectorCount);
        Sub(xUbIm[iStart], xUbIm[iStart], prodImUb, vectorCount);
        PipeBarrier<PIPE_V>();
        return;
    }
    Axpy(xUbRe[iStart], aReUb, -tRe, vectorCount);
    Axpy(xUbIm[iStart], aReUb, -tIm, vectorCount);
    PipeBarrier<PIPE_V>();
    Axpy(xUbRe[iStart], aImUb, tIm, vectorCount);
    Axpy(xUbIm[iStart], aImUb, -tRe, vectorCount);
    PipeBarrier<PIPE_V>();
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GatherTransTerm(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float& tRe, float& tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetOpAFromCol(i, j, aRe, aIm);
    CtbsvCmul(aRe, aIm, xUbRe.GetValue(i), xUbIm.GetValue(i), pRe, pIm);
    tRe -= pRe;
    tIm -= pIm;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::GatherTransTermReference(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t i, uint32_t j, float& tRe, float& tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetOpAFromCol(i, j, aRe, aIm);
    CtbsvCmulReferenceSoft(xUbRe.GetValue(i), xUbIm.GetValue(i), aRe, aIm, pRe, pIm);
    tRe -= pRe;
    tIm -= pIm;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::StoreTransCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm, uint32_t j, float tRe, float tIm)
{
    if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
        DivByDiag(j, tRe, tIm);
    }
    xUbRe.SetValue(j, tRe);
    xUbIm.SetValue(j, tIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::LoadXUb(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (incx == 1) {
        CtbsvCopyGm2UbIssue(xPackUb, xGM, 0, n * 2U);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        Gather(xUbRe, xPackUb, complexGatherOffUb, 0U, n);
        Gather(xUbIm, xPackUb, complexGatherOffUb, static_cast<uint32_t>(sizeof(float)), n);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID4);
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int64_t off = CtbsvComplexFloatIdx(XOffset(i));
        // Strided views must not read outside their logical elements.
        xUbRe.SetValue(i, xGM.GetValue(off));
        xUbIm.SetValue(i, xGM.GetValue(off + 1));
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::StoreXUb(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (incx == 1) {
        if (n >= CTBSV_VECTOR_PACK_MIN) {
            PackXVector(xUbRe);
        } else {
            PackXScalar(xUbRe, xUbIm);
        }
        CtbsvCopyUb2Gm(xGM, 0, xPackUb, n * 2U);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int64_t off = CtbsvComplexFloatIdx(XOffset(i));
        // Preserve stride holes for both positive and negative increments.
        xGM.SetValue(off, xUbRe.GetValue(i));
        xGM.SetValue(off + 1, xUbIm.GetValue(i));
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::PackXVector(const LocalTensor<float>& xUbRe)
{
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
    const uint32_t packedLanes = CtbsvAlignBlock(n);
    const uint32_t complexTile = packedLanes / 2U;
    const int32_t count = static_cast<int32_t>(packedLanes);
    auto offsets = complexGatherOffUb.ReinterpretCast<int32_t>();
    auto scratch = xPackUb.ReinterpretCast<uint32_t>();
    // For output lane q, read planar x[q / 2 + (q % 2) * xPlaneStride].
    // Build byte offsets with integer vector operations, preserving all FP32
    // payload bits (including signed zero and NaN) in the final Gather.
    ArithProgression<int32_t>(offsets, 0, 1, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scratch, complexGatherOffUb, static_cast<uint32_t>(CtbsvFp32::SIGN_SHIFT), count);
    PipeBarrier<PIPE_V>();
    ShiftRight(scratch, scratch, static_cast<uint32_t>(CtbsvFp32::SIGN_SHIFT), count);
    ShiftRight(complexGatherOffUb, complexGatherOffUb, 1U, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(complexGatherOffUb, complexGatherOffUb, CTBSV_FLOAT_BYTE_SHIFT, count);
    Muls(
        scratch.ReinterpretCast<int32_t>(), scratch.ReinterpretCast<int32_t>(),
        static_cast<int32_t>(xPlaneStride * sizeof(float)), count);
    PipeBarrier<PIPE_V>();
    Add(offsets, offsets, scratch.ReinterpretCast<int32_t>(), count);
    PipeBarrier<PIPE_V>();
    // Reuse the existing offset buffer in two tiles; do not increase peak UB.
    for (uint32_t start = 0U; start < n; start += complexTile) {
        const uint32_t complexCount = n - start < complexTile ? n - start : complexTile;
        Gather(
            xPackUb[start * 2U], xUbRe, complexGatherOffUb, start * static_cast<uint32_t>(sizeof(float)),
            complexCount * 2U);
    }
    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::PackXScalar(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    // Tiny vectors avoid the fixed setup cost of the vector index pipeline.
    LocalTensor<uint64_t> xPack64 = xPackUb.ReinterpretCast<uint64_t>();
    LocalTensor<uint32_t> xUbReBits = xUbRe.ReinterpretCast<uint32_t>();
    LocalTensor<uint32_t> xUbImBits = xUbIm.ReinterpretCast<uint32_t>();
    constexpr uint32_t componentBits = sizeof(float) * 8U;
    uint32_t i = 0U;
    for (; i + 3U < n; i += 4U) {
        const uint64_t real0 = static_cast<uint64_t>(xUbReBits.GetValue(i));
        const uint64_t imag0 = static_cast<uint64_t>(xUbImBits.GetValue(i));
        const uint64_t real1 = static_cast<uint64_t>(xUbReBits.GetValue(i + 1U));
        const uint64_t imag1 = static_cast<uint64_t>(xUbImBits.GetValue(i + 1U));
        const uint64_t real2 = static_cast<uint64_t>(xUbReBits.GetValue(i + 2U));
        const uint64_t imag2 = static_cast<uint64_t>(xUbImBits.GetValue(i + 2U));
        const uint64_t real3 = static_cast<uint64_t>(xUbReBits.GetValue(i + 3U));
        const uint64_t imag3 = static_cast<uint64_t>(xUbImBits.GetValue(i + 3U));
        xPack64.SetValue(i, real0 | (imag0 << componentBits));
        xPack64.SetValue(i + 1U, real1 | (imag1 << componentBits));
        xPack64.SetValue(i + 2U, real2 | (imag2 << componentBits));
        xPack64.SetValue(i + 3U, real3 | (imag3 << componentBits));
    }
    for (; i < n; ++i) {
        const uint64_t realBits = static_cast<uint64_t>(xUbReBits.GetValue(i));
        const uint64_t imagBits = static_cast<uint64_t>(xUbImBits.GetValue(i));
        xPack64.SetValue(i, realBits | (imagBits << componentBits));
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ProcessNoTransUpper(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(n - 1U, true);
    for (uint32_t j = n; j-- > 0;) {
        if (j > 0U) {
            PrefetchACol(j - 1U, !usePing);
        }
        BindACol(usePing);
        usePing = !usePing;
        float tRe = 0.0f;
        float tIm = 0.0f;
        if (!ScaleNoTransCol(xUbRe, xUbIm, j, tRe, tIm)) {
            continue;
        }
        const uint32_t iStart = (j >= k) ? (j - k) : 0;
        ScatterColVector(xUbRe, xUbIm, iStart, j, j, tRe, tIm);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ProcessNoTransLower(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(0U, true);
    for (uint32_t j = 0; j < n; ++j) {
        if (j + 1U < n) {
            PrefetchACol(j + 1U, !usePing);
        }
        BindACol(usePing);
        usePing = !usePing;
        float tRe = 0.0f;
        float tIm = 0.0f;
        if (!ScaleNoTransCol(xUbRe, xUbIm, j, tRe, tIm)) {
            continue;
        }
        const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
        ScatterColVector(xUbRe, xUbIm, j + 1U, iEnd, j, tRe, tIm);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ProcessTransUpper(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(0U, true);
    for (uint32_t j = 0; j < n; ++j) {
        BindAColVec(usePing);
        if (j + 1U < n) {
            PrefetchACol(j + 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = xUbRe.GetValue(j);
        float tIm = xUbIm.GetValue(j);
        const uint32_t iStart = (j >= k) ? (j - k) : 0;
        GatherTransVec(xUbRe, xUbIm, iStart, j, j, tRe, tIm);
        StoreTransCol(xUbRe, xUbIm, j, tRe, tIm);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ProcessTransLower(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(n - 1U, true);
    for (uint32_t j = n; j-- > 0;) {
        BindAColVec(usePing);
        if (j > 0U) {
            PrefetchACol(j - 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = xUbRe.GetValue(j);
        float tIm = xUbIm.GetValue(j);
        const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
        GatherTransVec(xUbRe, xUbIm, j + 1U, iEnd, j, tRe, tIm);
        StoreTransCol(xUbRe, xUbIm, j, tRe, tIm);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::InitPrimaryBuffers(
    TPipe& pipe, CtbsvUbBuffers& buffers, LocalTensor<float>& xUbRe, LocalTensor<float>& xUbIm)
{
    const uint32_t xAlign = CtbsvAlignBlock(n);
    const uint32_t packAlign = CtbsvAlignBlock(n * 2U);
    aColAlign = CtbsvAlignBlock((k + 1U) * 2U);
    constexpr uint32_t kVecPad = CTBSV_VECTOR_BYTES / sizeof(float);
    const uint32_t xPad = (TRANS != CtbsvTrans::NO_TRANS) ? kVecPad : 0U;
    xPlaneStride = xAlign + xPad;
    pipe.InitBuffer(buffers.xUbRe, 2U * xPlaneStride * sizeof(float));
    pipe.InitBuffer(buffers.xPack, packAlign * sizeof(float));
    pipe.InitBuffer(buffers.aColPing, aColAlign * sizeof(float));
    pipe.InitBuffer(buffers.aColPong, aColAlign * sizeof(float));
    xUbRe = buffers.xUbRe.Get<float>();
    xUbIm = xUbRe[xPlaneStride];
    xPackUb = buffers.xPack.Get<float>();
    aColPing = buffers.aColPing.Get<float>();
    aColPong = buffers.aColPong.Get<float>();
    aColUb = aColPing;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline uint32_t CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::InitWorkBuffers(
    TPipe& pipe, CtbsvUbBuffers& buffers)
{
    constexpr uint32_t kVecPad = CTBSV_VECTOR_BYTES / sizeof(float);
    const uint32_t pairAlign = CtbsvAlignBlock(aColAlign / 2U);
    const uint32_t workAlign = CtbsvAlignBlock(k + 1U + CTBSV_REDUCE_WORK_PAD_FLOATS);
    const uint32_t pairPad = pairAlign + kVecPad;
    complexGatherCount = CtbsvAlignBlock(n);
    if (complexGatherCount < pairPad) {
        complexGatherCount = pairPad;
    }
    pipe.InitBuffer(buffers.aRe, pairPad * sizeof(float));
    pipe.InitBuffer(buffers.aIm, pairPad * sizeof(float));
    pipe.InitBuffer(buffers.prodRe, pairPad * sizeof(float));
    pipe.InitBuffer(buffers.prodIm, pairPad * sizeof(float));
    pipe.InitBuffer(buffers.vecTmp, pairPad * sizeof(float));
    pipe.InitBuffer(buffers.vecWork, workAlign * sizeof(float));
    pipe.InitBuffer(buffers.vecOut, CTBSV_REDUCE_OUTPUT_FLOATS * sizeof(float));
    pipe.InitBuffer(buffers.gatherOff, pairPad * sizeof(uint32_t));
    pipe.InitBuffer(buffers.complexGatherOff, complexGatherCount * sizeof(uint32_t));
    aReUb = buffers.aRe.Get<float>();
    aImUb = buffers.aIm.Get<float>();
    prodReUb = buffers.prodRe.Get<float>();
    prodImUb = buffers.prodIm.Get<float>();
    vecTmpUb = buffers.vecTmp.Get<float>();
    vecWorkUb = buffers.vecWork.Get<float>();
    vecOutUb = buffers.vecOut.Get<float>();
    gatherOffUb = buffers.gatherOff.Get<uint32_t>();
    complexGatherOffUb = buffers.complexGatherOff.Get<uint32_t>();
    return pairPad;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::InitGatherOffsets(uint32_t pairPad)
{
    Duplicate(aReUb, 0.0f, static_cast<int32_t>(pairPad));
    Duplicate(aImUb, 0.0f, static_cast<int32_t>(pairPad));
    ArithProgression<int32_t>(
        gatherOffUb.ReinterpretCast<int32_t>(), 0, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(pairPad));
    ArithProgression<int32_t>(
        complexGatherOffUb.ReinterpretCast<int32_t>(), 0, static_cast<int32_t>(2U * sizeof(float)),
        static_cast<int32_t>(complexGatherCount));
    PipeBarrier<PIPE_V>();
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::RunSolve(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    constexpr uint32_t kVecPad = CTBSV_VECTOR_BYTES / sizeof(float);
    if constexpr (TRANS != CtbsvTrans::NO_TRANS) {
        for (uint32_t i = n; i < n + kVecPad; ++i) {
            xUbRe.SetValue(i, 0.0f);
            xUbIm.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
    }
    if constexpr (TRANS == CtbsvTrans::NO_TRANS) {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            ProcessNoTransUpper(xUbRe, xUbIm);
        } else {
            ProcessNoTransLower(xUbRe, xUbIm);
        }
    } else {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            ProcessTransUpper(xUbRe, xUbIm);
        } else {
            ProcessTransLower(xUbRe, xUbIm);
        }
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::ProcessRows(TPipe& pipe)
{
    CtbsvUbBuffers buffers;
    LocalTensor<float> xUbRe;
    LocalTensor<float> xUbIm;
    InitPrimaryBuffers(pipe, buffers, xUbRe, xUbIm);
    const uint32_t pairPad = InitWorkBuffers(pipe, buffers);
    InitGatherOffsets(pairPad);
    LoadXUb(xUbRe, xUbIm);
    RunSolve(xUbRe, xUbIm);
    StoreXUb(xUbRe, xUbIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG, bool REFERENCE>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG, REFERENCE>::Process(TPipe& pipe)
{
    ProcessRows(pipe);
}

#define DEFINE_CTBSV_KERNEL(uplo, trans, diag, name)                                 \
    __global__ __aicore__ void name(CtbsvTilingData tiling)                          \
    {                                                                                \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                              \
        if (tiling.useGmFallback != 0U) {                                            \
            CtbsvGmKernel<CtbsvUplo::uplo, CtbsvTrans::trans, CtbsvDiag::diag> gmOp; \
            gmOp.Init(tiling);                                                       \
            gmOp.Process();                                                          \
            return;                                                                  \
        }                                                                            \
        TPipe pipe;                                                                  \
        CtbsvKernel<CtbsvUplo::uplo, CtbsvTrans::trans, CtbsvDiag::diag, false> op;  \
        op.Init(tiling);                                                             \
        op.Process(pipe);                                                            \
    }

#define DEFINE_CTBSV_REFERENCE_KERNEL(uplo, trans, name)                           \
    __global__ __aicore__ void name(CtbsvTilingData tiling)                        \
    {                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                            \
        TPipe pipe;                                                                \
        CtbsvKernel<CtbsvUplo::uplo, CtbsvTrans::trans, CtbsvDiag::UNIT, true> op; \
        op.Init(tiling);                                                           \
        op.Process(pipe);                                                          \
    }

DEFINE_CTBSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, ctbsv_kernel_lower_no_trans_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, NO_TRANS, UNIT, ctbsv_kernel_lower_no_trans_unit)
DEFINE_CTBSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, ctbsv_kernel_upper_no_trans_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, NO_TRANS, UNIT, ctbsv_kernel_upper_no_trans_unit)
DEFINE_CTBSV_KERNEL(LOWER, TRANS, NON_UNIT, ctbsv_kernel_lower_trans_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, TRANS, UNIT, ctbsv_kernel_lower_trans_unit)
DEFINE_CTBSV_KERNEL(UPPER, TRANS, NON_UNIT, ctbsv_kernel_upper_trans_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, TRANS, UNIT, ctbsv_kernel_upper_trans_unit)
DEFINE_CTBSV_KERNEL(LOWER, CONJ, NON_UNIT, ctbsv_kernel_lower_conj_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, CONJ, UNIT, ctbsv_kernel_lower_conj_unit)
DEFINE_CTBSV_KERNEL(UPPER, CONJ, NON_UNIT, ctbsv_kernel_upper_conj_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, CONJ, UNIT, ctbsv_kernel_upper_conj_unit)

DEFINE_CTBSV_REFERENCE_KERNEL(LOWER, NO_TRANS, ctbsv_kernel_lower_no_trans_unit_reference)
DEFINE_CTBSV_REFERENCE_KERNEL(UPPER, NO_TRANS, ctbsv_kernel_upper_no_trans_unit_reference)
DEFINE_CTBSV_REFERENCE_KERNEL(LOWER, TRANS, ctbsv_kernel_lower_trans_unit_reference)
DEFINE_CTBSV_REFERENCE_KERNEL(UPPER, TRANS, ctbsv_kernel_upper_trans_unit_reference)
DEFINE_CTBSV_REFERENCE_KERNEL(LOWER, CONJ, ctbsv_kernel_lower_conj_unit_reference)
DEFINE_CTBSV_REFERENCE_KERNEL(UPPER, CONJ, ctbsv_kernel_upper_conj_unit_reference)

#undef DEFINE_CTBSV_KERNEL
#undef DEFINE_CTBSV_REFERENCE_KERNEL

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvRunSmallGm(const CtbsvTilingData& tiling)
{
    CtbsvGmKernel<UPLO, TRANS, DIAG> op;
    op.Init(tiling);
    op.Process();
}

template <CtbsvUplo UPLO>
__aicore__ inline void CtbsvDispatchSmallTrans(const CtbsvTilingData& tiling)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::NO_TRANS, CtbsvDiag::NON_UNIT>(tiling);
        } else {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::NO_TRANS, CtbsvDiag::UNIT>(tiling);
        }
    } else if (tiling.trans == ACLBLAS_OP_T) {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::TRANS, CtbsvDiag::NON_UNIT>(tiling);
        } else {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::TRANS, CtbsvDiag::UNIT>(tiling);
        }
    } else {
        if (tiling.diag == ACLBLAS_NON_UNIT) {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::CONJ, CtbsvDiag::NON_UNIT>(tiling);
        } else {
            CtbsvRunSmallGm<UPLO, CtbsvTrans::CONJ, CtbsvDiag::UNIT>(tiling);
        }
    }
}

__global__ __aicore__ void ctbsv_kernel_small_aic(CtbsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (tiling.uplo == ACLBLAS_LOWER) {
        CtbsvDispatchSmallTrans<CtbsvUplo::LOWER>(tiling);
    } else {
        CtbsvDispatchSmallTrans<CtbsvUplo::UPPER>(tiling);
    }
}

namespace {
bool CtbsvUseReferenceOrder(const CtbsvTilingData& tiling)
{
    if (tiling.useGmFallback != 0U || tiling.diag != ACLBLAS_UNIT || tiling.n < CTBSV_REFERENCE_MIN_N) {
        return false;
    }
    const uint32_t effectiveK = tiling.k < tiling.n - 1U ? tiling.k : tiling.n - 1U;
    // Preserve reference order when the effective band covers at least half n.
    return 2ULL * (static_cast<uint64_t>(effectiveK) + 1ULL) >= static_cast<uint64_t>(tiling.n);
}

void CtbsvLaunchLowerNoTrans(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_no_trans_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_lower_no_trans_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_no_trans_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLowerTrans(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_trans_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_lower_trans_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_trans_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLowerConj(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_conj_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_lower_conj_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_conj_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperNoTrans(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_no_trans_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_upper_no_trans_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_no_trans_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperTrans(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_trans_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_upper_trans_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_trans_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperConj(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_conj_non_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else if (CtbsvUseReferenceOrder(tiling)) {
        ctbsv_kernel_upper_conj_unit_reference<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_conj_unit<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLower(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        CtbsvLaunchLowerNoTrans(tiling, stream);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        CtbsvLaunchLowerTrans(tiling, stream);
    } else {
        CtbsvLaunchLowerConj(tiling, stream);
    }
}

void CtbsvLaunchUpper(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        CtbsvLaunchUpperNoTrans(tiling, stream);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        CtbsvLaunchUpperTrans(tiling, stream);
    } else {
        CtbsvLaunchUpperConj(tiling, stream);
    }
}
} // namespace

void ctbsv_kernel_do(const CtbsvTilingData& tiling, void* stream)
{
    if (tiling.n <= CTBSV_SMALL_AIC_MAX_N) {
        ctbsv_kernel_small_aic<<<CTBSV_SOLVE_BLOCKS, nullptr, stream>>>(tiling);
        return;
    }
    if (tiling.uplo == ACLBLAS_LOWER) {
        CtbsvLaunchLower(tiling, stream);
    } else {
        CtbsvLaunchUpper(tiling, stream);
    }
}
