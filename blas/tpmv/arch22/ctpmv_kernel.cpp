/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "cann_ops_blas_common.h"
#include "common/compute/simd.h"
#include "common/helper/kernel_utils.h"
#include "ctpmv_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t MAX_COMPLEX_PER_TILE = 4096U;
constexpr uint32_t FLOATS_PER_COMPLEX = 2U;

constexpr uint32_t AOS_BYTES = MAX_COMPLEX_PER_TILE * FLOATS_PER_COMPLEX * sizeof(float);
constexpr uint32_t PLANE_BYTES = MAX_COMPLEX_PER_TILE * sizeof(float);
constexpr uint32_t X_AOS_OFFSET = AOS_BYTES + 2U * PLANE_BYTES;

// Keep the working set within the 192-KiB UB. Reuse xAos for gather indices after SplitComplex.
constexpr uint32_t OUT_REAL_OFFSET = 2U * AOS_BYTES + 4U * PLANE_BYTES;
constexpr uint32_t OUT_IMAG_OFFSET = OUT_REAL_OFFSET + PLANE_BYTES;
constexpr uint32_t OUTPUT_AOS_OFFSET = OUT_IMAG_OFFSET + PLANE_BYTES;

constexpr float CTPMV_FLOAT_MAX = 3.402823466e38F;

struct CtpmvUb {
    LocalTensor<float> aAos;
    LocalTensor<float> aReal;
    LocalTensor<float> aImag;
    LocalTensor<float> xAos;
    LocalTensor<float> xReal;
    LocalTensor<float> xImag;
    LocalTensor<float> outReal;
    LocalTensor<float> outImag;
    LocalTensor<uint32_t> gatherMask;
};

struct CtpmvScalarUb {
    LocalTensor<float> fmaLhs;
    LocalTensor<float> fmaRhs;
    LocalTensor<float> fmaAcc;
};

// FusedMulAdd computes lhs * rhs + acc with rhs in dst.
// Scalar SetValue writes require an S/V hand-off before the vector instruction.
__aicore__ inline float CtpmvScalarMulAdd(
    LocalTensor<float> lhsBuf, LocalTensor<float> rhsBuf, LocalTensor<float> accBuf, float lhs, float rhs, float acc)
{
    // Preserve an infinite accumulator for finite multiplicands, matching ARM FMLA.
    // Let input Inf/NaN reach vmadd for propagation.
    const bool accInfinite = acc > CTPMV_FLOAT_MAX || acc < -CTPMV_FLOAT_MAX;
    const bool lhsFinite = lhs <= CTPMV_FLOAT_MAX && lhs >= -CTPMV_FLOAT_MAX;
    const bool rhsFinite = rhs <= CTPMV_FLOAT_MAX && rhs >= -CTPMV_FLOAT_MAX;
    if (accInfinite && lhsFinite && rhsFinite) {
        return acc;
    }

    // Scale lhs and acc together to avoid intermediate product overflow.
    // Undo scaling after the fused operation to preserve the finite FMLA result.
    if (lhsFinite && rhsFinite && !accInfinite) {
        const float lhsAbs = lhs >= 0.0F ? lhs : -lhs;
        const float rhsAbs = rhs >= 0.0F ? rhs : -rhs;
        const float accAbs = acc >= 0.0F ? acc : -acc;
        if (rhsAbs != 0.0F && lhsAbs > CTPMV_FLOAT_MAX / rhsAbs) {
            float scale = 1.0F;
            float scaledLhsAbs = lhsAbs;
            float scaledAccAbs = accAbs;
            for (uint32_t i = 0U;
                 i < 16U && (scaledLhsAbs > CTPMV_FLOAT_MAX / rhsAbs || scaledAccAbs > CTPMV_FLOAT_MAX); ++i) {
                scale *= 0.5F;
                scaledLhsAbs *= 0.5F;
                scaledAccAbs *= 0.5F;
            }
            const float scaledLhs = lhs * scale;
            const float scaledAcc = acc * scale;
            lhsBuf.SetValue(0U, scaledLhs);
            rhsBuf.SetValue(0U, rhs);
            accBuf.SetValue(0U, scaledAcc);
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
            AscendC::FusedMulAdd<float>(rhsBuf, lhsBuf, accBuf, 1);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            return rhsBuf.GetValue(0U) / scale;
        }
    }
    lhsBuf.SetValue(0U, lhs);
    rhsBuf.SetValue(0U, rhs);
    accBuf.SetValue(0U, acc);
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
    AscendC::FusedMulAdd<float>(rhsBuf, lhsBuf, accBuf, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    return rhsBuf.GetValue(0U);
}

// A separate vector multiply preserves the diagonal product rounding boundary.
__aicore__ inline float CtpmvScalarMul(
    LocalTensor<float> lhsBuf, LocalTensor<float> rhsBuf, LocalTensor<float> outBuf, float lhs, float rhs)
{
    lhsBuf.SetValue(0U, lhs);
    rhsBuf.SetValue(0U, rhs);
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
    Mul<float>(outBuf, lhsBuf, rhsBuf, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    return outBuf.GetValue(0U);
}

__aicore__ inline float CtpmvScalarAdd(
    LocalTensor<float> lhsBuf, LocalTensor<float> rhsBuf, LocalTensor<float> outBuf, float lhs, float rhs)
{
    lhsBuf.SetValue(0U, lhs);
    rhsBuf.SetValue(0U, rhs);
    SetFlag<HardEvent::S_V>(EVENT_ID2);
    WaitFlag<HardEvent::S_V>(EVENT_ID2);
    Add<float>(outBuf, lhsBuf, rhsBuf, 1);
    SetFlag<HardEvent::V_S>(EVENT_ID2);
    WaitFlag<HardEvent::V_S>(EVENT_ID2);
    return outBuf.GetValue(0U);
}

__aicore__ inline CtpmvUb GetCtpmvUb()
{
    AsdopsBuffer<ArchType::ASCEND_V220> buffer;
    CtpmvUb ub{};
    ub.aAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
    ub.aReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(AOS_BYTES);
    ub.aImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(AOS_BYTES + PLANE_BYTES);
    ub.xAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(X_AOS_OFFSET);
    ub.xReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(2U * AOS_BYTES + 2U * PLANE_BYTES);
    ub.xImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(2U * AOS_BYTES + 3U * PLANE_BYTES);
    ub.outReal = buffer.GetBuffer<BufferType::ASCEND_UB, float>(OUT_REAL_OFFSET);
    ub.outImag = buffer.GetBuffer<BufferType::ASCEND_UB, float>(OUT_IMAG_OFFSET);
    // Keep address scratch separate from the Gather source and destination.
    ub.gatherMask = buffer.GetBuffer<BufferType::ASCEND_UB, uint32_t>(OUTPUT_AOS_OFFSET);
    return ub;
}

__aicore__ inline CtpmvScalarUb GetCtpmvScalarUb()
{
    AsdopsBuffer<ArchType::ASCEND_V220> buffer;
    return {
        buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U), buffer.GetBuffer<BufferType::ASCEND_UB, float>(32U),
        buffer.GetBuffer<BufferType::ASCEND_UB, float>(64U)};
}

__aicore__ inline uint64_t UpperColumnBase(uint32_t column)
{
    return static_cast<uint64_t>(column) * (column + 1ULL) / 2ULL;
}

__aicore__ inline uint64_t LowerColumnBase(uint32_t n, uint32_t column)
{
    return static_cast<uint64_t>(column) * (2ULL * n - column + 1ULL) / 2ULL;
}

__aicore__ inline uint32_t XPhysicalPos(uint32_t logical, uint32_t n, int64_t incx)
{
    const uint32_t absIncx = incx >= 0 ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    return incx >= 0 ? logical * absIncx : (n - 1U - logical) * absIncx;
}

__aicore__ inline uint64_t PackedIndex(uint32_t row, uint32_t column, uint32_t n, uint32_t uplo)
{
    if (uplo == ACLBLAS_UPPER) {
        return UpperColumnBase(column) + row;
    }
    return LowerColumnBase(n, column) + row - column;
}

__aicore__ inline void SplitComplex(
    LocalTensor<float> real, LocalTensor<float> imag, LocalTensor<float> aos, uint32_t count)
{
    const uint16_t repeats = static_cast<uint16_t>((count * FLOATS_PER_COMPLEX + 63U) / 64U);
    uint64_t reserved = 0U;
    // GatherMask uses isSetMask=false; restore all lanes before taking the AoS snapshot.
    SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    GatherMask<float>(real, aos, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(imag, aos, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void AxpyAligned(LocalTensor<float> dst, LocalTensor<float> src, float scalar, uint32_t count)
{
    const AscendC::UnaryRepeatParams repeatParams{1U, 1U, 8U, 8U};
    const uint32_t fullRepeats = count / 64U;
    const uint32_t tail = count % 64U;
    if (fullRepeats != 0U) {
        AscendC::Axpy<float, float, true>(dst, src, scalar, 64U, static_cast<uint8_t>(fullRepeats), repeatParams);
    }
    if (tail != 0U) {
        const uint32_t offset = fullRepeats * 64U;
        AscendC::Axpy<float, float, true>(dst[offset], src[offset], scalar, tail, 1U, repeatParams);
    }
}

// The short-transpose path uses at most 32 lanes under one explicit mask.
__aicore__ inline void MultiplyShortLanes(LocalTensor<float> dst, LocalTensor<float> src, float scalar, uint32_t count)
{
    SetMaskNorm();
    const uint32_t tail = count;
    if (tail != 0U) {
        AscendC::SetVectorMask<float>(0U, (1ULL << tail) - 1ULL);
        AscendC::Muls<float, false>(
            dst, src, scalar, static_cast<uint64_t>(0U), 1U, AscendC::UnaryRepeatParams(1U, 1U, 8U, 8U));
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

// TwoProduct/TwoSum recover ARM FMLA rounding for ordinary finite operands.
// Volatile prevents contraction; the scalar helper handles extreme values and Inf/NaN.
__aicore__ inline float CtpmvStrictFmaEmulated(float lhs, float rhs, float acc)
{
    constexpr float split = 4097.0F;
    volatile float lhsSplit = split * lhs;
    volatile float lhsHi = lhsSplit - (lhsSplit - lhs);
    volatile float lhsLo = lhs - lhsHi;
    volatile float rhsSplit = split * rhs;
    volatile float rhsHi = rhsSplit - (rhsSplit - rhs);
    volatile float rhsLo = rhs - rhsHi;
    volatile float product = lhs * rhs;
    volatile float productError = ((lhsHi * rhsHi - product) + lhsHi * rhsLo + lhsLo * rhsHi) + lhsLo * rhsLo;
    volatile float sum = product + acc;
    volatile float sumDelta = sum - product;
    volatile float correction = ((acc - sumDelta) + (product - (sum - sumDelta))) + productError;
    return sum + correction;
}

__aicore__ inline bool CtpmvStrictFmaRange(float value)
{
    const bool finite = value == value && value <= CTPMV_FLOAT_MAX && value >= -CTPMV_FLOAT_MAX;
    const float magnitude = value >= 0.0F ? value : -value;
    return finite && magnitude <= 1.0e19F;
}

// Shared by the ordered path and the resident small-matrix path.
__aicore__ inline float CtpmvStrictMulAdd(
    LocalTensor<float> lhsBuf, LocalTensor<float> rhsBuf, LocalTensor<float> accBuf, float lhs, float rhs, float acc)
{
    if (CtpmvStrictFmaRange(lhs) && CtpmvStrictFmaRange(rhs) && CtpmvStrictFmaRange(acc)) {
        return CtpmvStrictFmaEmulated(lhs, rhs, acc);
    }
    return CtpmvScalarMulAdd(lhsBuf, rhsBuf, accBuf, lhs, rhs, acc);
}

__aicore__ inline void CtpmvStrictComplexMul(
    const CtpmvScalarUb& ub, float& dstReal, float& dstImag, float ar, float ai, float xr, float xi, bool conjugate)
{
    const float aiXi = CtpmvScalarMul(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, ai, xi);
    const float aiXr = CtpmvScalarMul(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, ai, xr);
    dstReal = CtpmvStrictMulAdd(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, ar, xr, conjugate ? aiXi : -aiXi);
    dstImag = CtpmvStrictMulAdd(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, ar, xi, conjugate ? -aiXr : aiXr);
}

__aicore__ inline void CtpmvStrictComplexAxpy(
    const CtpmvScalarUb& ub, float& dstReal, float& dstImag, float ar, float ai, float xr, float xi, bool conjugate)
{
    float productReal;
    float productImag;
    CtpmvStrictComplexMul(ub, productReal, productImag, ar, ai, xr, xi, conjugate);
    dstReal = CtpmvScalarAdd(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, dstReal, productReal);
    dstImag = CtpmvScalarAdd(ub.fmaLhs, ub.fmaRhs, ub.fmaAcc, dstImag, productImag);
}

__aicore__ inline void CtpmvStrictLoad(
    GlobalTensor<float> x, uint32_t logical, const CtpmvTilingData& tiling, float& real, float& imag)
{
    const uint32_t pos = XPhysicalPos(logical, tiling.n, tiling.incx) * FLOATS_PER_COMPLEX;
    real = x.GetValue(pos);
    imag = x.GetValue(pos + 1U);
}

__aicore__ inline void CtpmvStrictStore(
    GlobalTensor<float> x, uint32_t logical, const CtpmvTilingData& tiling, float real, float imag)
{
    const uint32_t pos = XPhysicalPos(logical, tiling.n, tiling.incx) * FLOATS_PER_COMPLEX;
    x.SetValue(pos, real);
    x.SetValue(pos + 1U, imag);
}

template <bool upper>
__aicore__ inline void CtpmvStrictNotrans(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvScalarUb& ub)
{
    // Traverse the source columns before their in-place destinations.
    for (uint32_t step = 0U; step < tiling.n; ++step) {
        const uint32_t j = upper ? step : tiling.n - 1U - step;
        float xr, xi;
        CtpmvStrictLoad(x, j, tiling, xr, xi);
        if (xr == 0.0F && xi == 0.0F) {
            continue;
        }
        const uint32_t count = upper ? j : tiling.n - j - 1U;
        for (uint32_t off = 0U; off < count; ++off) {
            const uint32_t i = upper ? off : tiling.n - 1U - off;
            const uint64_t packed = PackedIndex(i, j, tiling.n, tiling.uplo) * FLOATS_PER_COMPLEX;
            float yr, yi;
            CtpmvStrictLoad(x, i, tiling, yr, yi);
            CtpmvStrictComplexAxpy(ub, yr, yi, xr, xi, ap.GetValue(packed), ap.GetValue(packed + 1U), false);
            CtpmvStrictStore(x, i, tiling, yr, yi);
        }
        if (tiling.diag != ACLBLAS_UNIT) {
            const uint64_t diagonal = PackedIndex(j, j, tiling.n, tiling.uplo) * FLOATS_PER_COMPLEX;
            CtpmvStrictComplexMul(ub, xr, xi, xr, xi, ap.GetValue(diagonal), ap.GetValue(diagonal + 1U), false);
            CtpmvStrictStore(x, j, tiling, xr, xi);
        }
    }
}

template <bool upper>
__aicore__ inline void CtpmvStrictTranspose(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvScalarUb& ub)
{
    const bool conjugate = tiling.trans == ACLBLAS_OP_C;
    for (uint32_t step = 0U; step < tiling.n; ++step) {
        const uint32_t j = upper ? tiling.n - 1U - step : step;
        float yr, yi;
        CtpmvStrictLoad(x, j, tiling, yr, yi);
        if (tiling.diag != ACLBLAS_UNIT) {
            const uint64_t diagonal = PackedIndex(j, j, tiling.n, tiling.uplo) * FLOATS_PER_COMPLEX;
            CtpmvStrictComplexMul(ub, yr, yi, ap.GetValue(diagonal), ap.GetValue(diagonal + 1U), yr, yi, conjugate);
        }
        const uint32_t count = upper ? j : tiling.n - j - 1U;
        for (uint32_t off = 0U; off < count; ++off) {
            const uint32_t i = upper ? j - 1U - off : j + 1U + off;
            const uint64_t packed = PackedIndex(i, j, tiling.n, tiling.uplo) * FLOATS_PER_COMPLEX;
            float xr, xi;
            CtpmvStrictLoad(x, i, tiling, xr, xi);
            CtpmvStrictComplexAxpy(ub, yr, yi, ap.GetValue(packed), ap.GetValue(packed + 1U), xr, xi, conjugate);
        }
        CtpmvStrictStore(x, j, tiling, yr, yi);
    }
}

__aicore__ inline void ComputeStrict(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvScalarUb& ub)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        if (tiling.uplo == ACLBLAS_UPPER) {
            CtpmvStrictNotrans<true>(ap, x, tiling, ub);
        } else {
            CtpmvStrictNotrans<false>(ap, x, tiling, ub);
        }
    } else if (tiling.uplo == ACLBLAS_UPPER) {
        CtpmvStrictTranspose<true>(ap, x, tiling, ub);
    } else {
        CtpmvStrictTranspose<false>(ap, x, tiling, ub);
    }
}

__aicore__ inline void ComputeOrdered(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvScalarUb& ub)
{
    const uint32_t n = tiling.n;
    // The unit upper triangle updates y0 += A01*x1 and leaves y1 unchanged.
    if (n == 2U && tiling.incx == 1 && tiling.uplo == ACLBLAS_UPPER && tiling.trans == ACLBLAS_OP_N &&
        tiling.diag == ACLBLAS_UNIT) {
        const float x0r = x.GetValue(0U);
        const float x0i = x.GetValue(1U);
        const float x1r = x.GetValue(2U);
        const float x1i = x.GetValue(3U);
        const float ar = ap.GetValue(2U);
        const float ai = ap.GetValue(3U);
        const float y0r = x0r + ar * x1r - ai * x1i;
        const float y0i = x0i + ar * x1i + ai * x1r;
        x.SetValue(0U, y0r);
        x.SetValue(1U, y0i);
        x.SetValue(2U, x1r);
        x.SetValue(3U, x1i);
        return;
    }
    ComputeStrict(ap, x, tiling, ub);
}

// Each core accumulates a packed column range. Reduce its output planes in
// groups of eight producers and aligned 64-row tiles.
constexpr uint32_t CTPMV_PARTIAL_GROUP = 8U;

__aicore__ inline void ReduceProducerGroup(LocalTensor<float> real, LocalTensor<float> imag, uint32_t planeRows)
{
    Add(real, real, real[planeRows], planeRows);
    Add(real[2U * planeRows], real[2U * planeRows], real[3U * planeRows], planeRows);
    Add(real[4U * planeRows], real[4U * planeRows], real[5U * planeRows], planeRows);
    Add(real[6U * planeRows], real[6U * planeRows], real[7U * planeRows], planeRows);
    Add(imag, imag, imag[planeRows], planeRows);
    Add(imag[2U * planeRows], imag[2U * planeRows], imag[3U * planeRows], planeRows);
    Add(imag[4U * planeRows], imag[4U * planeRows], imag[5U * planeRows], planeRows);
    Add(imag[6U * planeRows], imag[6U * planeRows], imag[7U * planeRows], planeRows);
    PipeBarrier<PIPE_V>();

    Add(real, real, real[2U * planeRows], planeRows);
    Add(real[4U * planeRows], real[4U * planeRows], real[6U * planeRows], planeRows);
    Add(imag, imag, imag[2U * planeRows], planeRows);
    Add(imag[4U * planeRows], imag[4U * planeRows], imag[6U * planeRows], planeRows);
    PipeBarrier<PIPE_V>();

    Add(real, real, real[4U * planeRows], planeRows);
    Add(imag, imag, imag[4U * planeRows], planeRows);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceProducerBatches(
    GlobalTensor<float> workspace, const CtpmvTilingData& tiling, const CtpmvUb& ub, uint32_t planeRows,
    uint32_t rowBegin, uint32_t rowCount)
{
    constexpr uint32_t rowTile = 64U;
    const uint32_t partialStride = planeRows * FLOATS_PER_COMPLEX;
    const uint32_t blockBytes = rowCount * sizeof(float);
    const uint32_t alignedBlockBytes = (blockBytes + 31U) & ~31U;
    const uint32_t sourceStrideBytes = (partialStride - rowCount) * sizeof(float);
    const uint32_t destinationStrideBlocks = (rowTile * sizeof(float) - alignedBlockBytes) / 32U;
    const uint8_t rightPadding = static_cast<uint8_t>((8U - (rowCount & 7U)) & 7U);
    const DataCopyPadExtParams<float> partialPad{true, 0U, rightPadding, 0.0f};

    for (uint32_t producerBegin = 0U; producerBegin < tiling.coreNum; producerBegin += CTPMV_PARTIAL_GROUP) {
        const uint32_t producerCount =
            tiling.coreNum - producerBegin < CTPMV_PARTIAL_GROUP ? tiling.coreNum - producerBegin : CTPMV_PARTIAL_GROUP;
        Duplicate<float>(ub.aReal, 0.0f, CTPMV_PARTIAL_GROUP * rowTile);
        Duplicate<float>(ub.aImag, 0.0f, CTPMV_PARTIAL_GROUP * rowTile);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);

        const uint64_t producerBase = static_cast<uint64_t>(producerBegin) * partialStride + rowBegin;
        const DataCopyExtParams partialCopy{
            static_cast<uint16_t>(producerCount), blockBytes, sourceStrideBytes, destinationStrideBlocks, 0U};
        DataCopyPad(ub.aReal, workspace[producerBase], partialCopy, partialPad);
        DataCopyPad(ub.aImag, workspace[producerBase + planeRows], partialCopy, partialPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        ReduceProducerGroup(ub.aReal, ub.aImag, rowTile);
        Add(ub.outReal, ub.outReal, ub.aReal, rowTile);
        Add(ub.outImag, ub.outImag, ub.aImag, rowTile);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void StorePartialOutput(
    GlobalTensor<float> x, const CtpmvUb& ub, uint32_t rowBegin, uint32_t rowCount)
{
    const uint32_t count = rowCount * FLOATS_PER_COMPLEX;
    auto offsets = ub.aAos.template ReinterpretCast<int32_t>();
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t i = 0U; i < 8U; ++i) {
        offsets.SetValue(i, (i / 2U) * sizeof(float) + (i % 2U) * PLANE_BYTES);
    }
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    for (uint32_t span = 8U; span < count; span *= 2U) {
        const uint32_t length = count - span < span ? count - span : span;
        Adds(offsets[span], offsets, static_cast<int32_t>(span / 2U * sizeof(float)), length);
        PipeBarrier<PIPE_V>();
    }
    Gather(ub.xAos, ub.outReal, offsets.template ReinterpretCast<uint32_t>(), 0U, count);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        x[rowBegin * FLOATS_PER_COMPLEX], ub.xAos,
        DataCopyExtParams{1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void ReduceColumnPartials(
    GlobalTensor<float> x, GlobalTensor<float> workspace, const CtpmvTilingData& tiling, const CtpmvUb& ub,
    uint32_t coreIdx, uint32_t planeStride)
{
    constexpr uint32_t rowTile = 64U;
    const uint32_t rowBlocks = (tiling.n + rowTile - 1U) / rowTile;
    SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    for (uint32_t rowBlock = coreIdx; rowBlock < rowBlocks; rowBlock += tiling.coreNum) {
        const uint32_t rowBegin = rowBlock * rowTile;
        const uint32_t rowCount = tiling.n - rowBegin < rowTile ? tiling.n - rowBegin : rowTile;
        Duplicate(ub.outReal, 0.0f, rowTile);
        Duplicate(ub.outImag, 0.0f, rowTile);
        PipeBarrier<PIPE_V>();
        ReduceProducerBatches(workspace, tiling, ub, planeStride, rowBegin, rowCount);
        StorePartialOutput(x, ub, rowBegin, rowCount);
    }
}

__aicore__ inline void PublishColumnPartials(
    GlobalTensor<float> workspace, const CtpmvUb& ub, uint32_t coreIdx, uint32_t planeStride)
{
    // Every producer initializes complete aligned planes, including the tail.
    const uint64_t partialBase = static_cast<uint64_t>(coreIdx) * planeStride * FLOATS_PER_COMPLEX;
    const uint32_t planeBytes = planeStride * sizeof(float);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        workspace[partialBase], ub.outReal, 0U, 1U, planeBytes, 0U, 0U, 0U, 0U);
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        workspace[partialBase + planeStride], ub.outImag, 0U, 1U, planeBytes, 0U, 0U, 0U, 0U);
    // Drain publication even when only one core reuses the output UB.
    SyncAll();
}

template <bool upper>
__aicore__ inline void AccumulateColumn(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvUb& ub, uint32_t column)
{
    const uint32_t first = upper ? 0U : column;
    const uint32_t alignedFirst = first & ~7U;
    const uint32_t left = first - alignedFirst;
    const uint32_t count = upper ? column + 1U : tiling.n - column;
    const uint32_t vectorCount = left + count;
    const uint32_t copyOffset = (left & ~3U) * FLOATS_PER_COMPLEX;
    const uint8_t leftPad = static_cast<uint8_t>((left & 3U) * FLOATS_PER_COMPLEX);
    const uint8_t rightPad = static_cast<uint8_t>((8U - ((leftPad + count * FLOATS_PER_COMPLEX) & 7U)) & 7U);
    if (copyOffset != 0U) {
        Duplicate<float, true>(ub.aAos, 0.0f, 8U, 1U, 1U, 8U);
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    const uint64_t packed = upper ? UpperColumnBase(column) : LowerColumnBase(tiling.n, column);
    DataCopyPad(
        ub.aAos[copyOffset], ap[packed * FLOATS_PER_COMPLEX],
        DataCopyExtParams{1U, count * FLOATS_PER_COMPLEX * sizeof(float), 0U, 0U, 0U},
        DataCopyPadExtParams<float>{true, leftPad, rightPad, 0.0f});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    SplitComplex(ub.aReal, ub.aImag, ub.aAos, vectorCount);
    if (tiling.diag == ACLBLAS_UNIT) {
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
        const uint32_t diagonal = upper ? column : left;
        ub.aReal.SetValue(diagonal, 1.0f);
        ub.aImag.SetValue(diagonal, 0.0f);
    }
    const float xr = x.GetValue(column * FLOATS_PER_COMPLEX);
    const float xi = x.GetValue(column * FLOATS_PER_COMPLEX + 1U);
    SetFlag<HardEvent::S_V>(EVENT_ID1);
    WaitFlag<HardEvent::S_V>(EVENT_ID1);
    LocalTensor<float> outputReal = ub.outReal[alignedFirst];
    LocalTensor<float> outputImag = ub.outImag[alignedFirst];
    AxpyAligned(outputReal, ub.aReal, xr, vectorCount);
    AxpyAligned(outputImag, ub.aReal, xi, vectorCount);
    PipeBarrier<PIPE_V>();
    AxpyAligned(outputReal, ub.aImag, -xi, vectorCount);
    AxpyAligned(outputImag, ub.aImag, xr, vectorCount);
    PipeBarrier<PIPE_V>();
}

template <bool upper>
__aicore__ inline void ComputeColumnPartials(
    GlobalTensor<float> ap, GlobalTensor<float> x, GlobalTensor<float> workspace, const CtpmvTilingData& tiling,
    const CtpmvUb& ub, uint32_t coreIdx)
{
    const uint32_t n = tiling.n;
    const uint32_t planeStride = (n + 7U) & ~7U;
    SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    const uint8_t repeats = static_cast<uint8_t>((n + 63U) / 64U);
    Duplicate<float, false>(ub.outReal, 0.0F, 64U, repeats, 1U, 8U);
    Duplicate<float, false>(ub.outImag, 0.0F, 64U, repeats, 1U, 8U);
    PipeBarrier<PIPE_V>();
    const uint32_t begin = static_cast<uint64_t>(n) * coreIdx / tiling.coreNum;
    const uint32_t end = static_cast<uint64_t>(n) * (coreIdx + 1U) / tiling.coreNum;
    for (uint32_t offset = 0U; offset < end - begin; ++offset) {
        const uint32_t column = upper ? begin + offset : end - 1U - offset;
        AccumulateColumn<upper>(ap, x, tiling, ub, column);
    }
    PublishColumnPartials(workspace, ub, coreIdx, planeStride);
    ReduceColumnPartials(x, workspace, tiling, ub, coreIdx, planeStride);
}

// A false comparison bit includes NaN, Inf, and overflow-prone magnitudes.
// Inspect packed words instead of loading every operand through Scalar.
constexpr float CTPMV_PANEL_SAFE_ABS = 1.0e18F;

__aicore__ inline bool CtpmvPanelValuesAreSafe(LocalTensor<float> src, uint32_t count, const CtpmvUb& ub)
{
    const uint32_t paddedCount = (count + 63U) & ~63U;
    LocalTensor<uint8_t> mask = ub.outImag.template ReinterpretCast<uint8_t>();
    LocalTensor<uint32_t> words = mask.template ReinterpretCast<uint32_t>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    Abs(ub.outReal, src, paddedCount);
    PipeBarrier<PIPE_V>();
    Compares(mask, ub.outReal, CTPMV_PANEL_SAFE_ABS, CMPMODE::LE, paddedCount);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (uint32_t i = 0U; i < count / 32U; ++i) {
        if (words.GetValue(i) != 0xFFFFFFFFU) {
            return false;
        }
    }
    const uint32_t tail = count % 32U;
    const uint32_t tailMask = (1U << tail) - 1U;
    return tail == 0U || (words.GetValue(count / 32U) & tailMask) == tailMask;
}

// Stage the short upper triangle once, then gather and reduce aligned column panels.
// Keep the input snapshot and outputs resident until the final in-place store.
#if __DAV_C220_VEC__
__aicore__ inline bool LoadShortTranspose(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
    const uint32_t n = tiling.n;
    const uint32_t apCount = n * (n + 1U) / 2U;
    const uint32_t inputBytes = n * FLOATS_PER_COMPLEX * sizeof(float);
    const uint32_t apBytes = apCount * FLOATS_PER_COMPLEX * sizeof(float);
    const uint8_t inputPad = static_cast<uint8_t>(((32U - (inputBytes & 31U)) & 31U) / sizeof(float));
    const uint8_t apPad = static_cast<uint8_t>(((32U - (apBytes & 31U)) & 31U) / sizeof(float));

    // Clear a full AoS repeat so padded lanes contain zero instead of stale NaNs.
    SetMaskNorm();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    Duplicate<float, false>(ub.xAos, 0.0F, 64U, 1U, 1U, 8U);
    PipeBarrier<PIPE_V>();
    DataCopyPad(
        ub.xAos, x, DataCopyExtParams{1U, inputBytes, 0U, 0U, 0U},
        DataCopyPadExtParams<float>{true, 0U, inputPad, 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    SplitComplex(ub.xReal, ub.xImag, ub.xAos, n);
    LocalTensor<float> inputReal = ub.xReal;
    LocalTensor<float> inputImag = ub.xImag;

    LocalTensor<float> signedX = ub.xAos[4096U];
    // Clear signedX padding for the fixed 32-lane panel mask.
    const uint32_t signedLanes = n > 8U ? 32U : n;
    if (tiling.trans == ACLBLAS_OP_C) {
        MultiplyShortLanes(signedX, inputReal, -1.0F, signedLanes);
    } else {
        MultiplyShortLanes(signedX, inputImag, -1.0F, signedLanes);
    }
    PipeBarrier<PIPE_V>();

    DataCopyPad(
        ub.aAos, ap, DataCopyExtParams{1U, apBytes, 0U, 0U, 0U}, DataCopyPadExtParams<float>{true, 0U, apPad, 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    SplitComplex(ub.aReal, ub.aImag, ub.aAos, apCount);

    // Exceptional operands require the reference scalar/FMA order instead of tree reduction.
    const bool safeAp = CtpmvPanelValuesAreSafe(ub.aAos, apCount * FLOATS_PER_COMPLEX, ub);
    const bool safeX = CtpmvPanelValuesAreSafe(ub.xAos, n * FLOATS_PER_COMPLEX, ub);
    return safeAp && safeX;
}

__aicore__ inline void StageShortTranspose(const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
    constexpr uint32_t panelBegin = 0U;
    const uint32_t panelCols = tiling.n;
    const bool fixedShortMask = tiling.n > 8U;
    LocalTensor<uint32_t> offsets = ub.gatherMask;
    LocalTensor<float> panelAReal = ub.xAos;
    LocalTensor<float> panelAImag = ub.xAos[2048U];
    LocalTensor<float> productReal = ub.outReal;
    LocalTensor<float> productImag = ub.outImag;
    if (fixedShortMask) {
        SetMaskNorm();
        AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        Duplicate<float, false>(panelAReal, 0.0F, 64U, static_cast<uint8_t>(panelCols), 1U, 8U);
        Duplicate<float, false>(panelAImag, 0.0F, 64U, static_cast<uint8_t>(panelCols), 1U, 8U);
        PipeBarrier<PIPE_V>();
    }
    Duplicate<float, false>(productReal, 0.0F, 64U, static_cast<uint8_t>(panelCols), 1U, 8U);
    Duplicate<float, false>(productImag, 0.0F, 64U, static_cast<uint8_t>(panelCols), 1U, 8U);
    PipeBarrier<PIPE_V>();

    for (uint32_t k = 0U; k < panelCols; ++k) {
        const uint32_t column = panelBegin + k;
        const uint32_t len = tiling.diag == ACLBLAS_UNIT ? column : column + 1U;
        if (len == 0U) {
            continue;
        }
        const uint32_t slab = k * 64U;
        const uint32_t apOffset = column * (column + 1U) / 2U;
        Gather(panelAReal[slab], ub.aReal, offsets, apOffset * sizeof(float), len);
        Gather(panelAImag[slab], ub.aImag, offsets, apOffset * sizeof(float), len);
    }
    PipeBarrier<PIPE_V>();
}

template <bool accumulate>
__aicore__ inline void ProductShortTranspose(const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
    const bool fixedMask = tiling.n > 8U;
    const bool conjugate = tiling.trans == ACLBLAS_OP_C;
    auto matrix = accumulate ? ub.xAos[2048U] : ub.xAos;
    auto realInput = accumulate ? (conjugate ? ub.xImag : ub.xAos[4096U]) : ub.xReal;
    auto imagInput = accumulate ? (conjugate ? ub.xAos[4096U] : ub.xReal) : ub.xImag;
    const auto xr = reinterpret_cast<__ubuf__ float*>(realInput.GetPhyAddr());
    const auto xi = reinterpret_cast<__ubuf__ float*>(imagInput.GetPhyAddr());
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    if (fixedMask) {
        AscendC::SetVectorMask<float>(0U, 0xFFFFFFFFULL);
    }
    for (uint32_t column = 0U; column < tiling.n; ++column) {
        const uint32_t count = tiling.diag == ACLBLAS_UNIT ? column : column + 1U;
        if (count == 0U) {
            continue;
        }
        if (!fixedMask) {
            AscendC::SetVectorMask<float>(0U, (1ULL << count) - 1ULL);
        }
        const uint32_t slab = column * 64U;
        const auto a = reinterpret_cast<__ubuf__ float*>(matrix[slab].GetPhyAddr());
        const auto yr = reinterpret_cast<__ubuf__ float*>(ub.outReal[slab].GetPhyAddr());
        const auto yi = reinterpret_cast<__ubuf__ float*>(ub.outImag[slab].GetPhyAddr());
        if constexpr (accumulate) {
            vmla(yr, a, xr, 1U, 1U, 1U, 1U, 0U, 8U, 8U);
            vmla(yi, a, xi, 1U, 1U, 1U, 1U, 0U, 8U, 8U);
        } else {
            vmul(yr, a, xr, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
            vmul(yi, a, xi, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        }
    }
    PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

__aicore__ inline void ReduceShortTranspose(const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
    constexpr uint32_t panelBegin = 0U;
    const uint32_t panelCols = tiling.n;
    LocalTensor<uint32_t> offsets = ub.gatherMask;
    LocalTensor<int32_t> offsetsI32 = ub.gatherMask.template ReinterpretCast<int32_t>();
    LocalTensor<float> inputReal = ub.xReal;
    LocalTensor<float> inputImag = ub.xImag;
    LocalTensor<float> productReal = ub.outReal;
    LocalTensor<float> productImag = ub.outImag;
    LocalTensor<float> reduceReal = ub.aAos;
    LocalTensor<float> reduceImag = ub.aAos[1024U];
    LocalTensor<float> outputReal = ub.aAos[2048U];
    LocalTensor<float> outputImag = ub.aAos[2112U];

    PipeBarrier<PIPE_V>();
    AscendC::SetVectorMask<float>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    WholeReduceSum<float, false>(reduceReal, productReal, 64, static_cast<int32_t>(panelCols), 1U, 1U, 8U);
    WholeReduceSum<float, false>(reduceImag, productImag, 64, static_cast<int32_t>(panelCols), 1U, 1U, 8U);
    PipeBarrier<PIPE_V>();
    CreateVecIndex<int32_t>(offsetsI32, 0, panelCols);
    Muls<int32_t>(offsetsI32, offsetsI32, static_cast<int32_t>(sizeof(float)), panelCols);
    PipeBarrier<PIPE_V>();
    Gather(outputReal[panelBegin], reduceReal, offsets, 0U, panelCols);
    Gather(outputImag[panelBegin], reduceImag, offsets, 0U, panelCols);
    PipeBarrier<PIPE_V>();
    if (tiling.diag == ACLBLAS_UNIT) {
        Add(outputReal[panelBegin], outputReal[panelBegin], inputReal[panelBegin], panelCols);
        Add(outputImag[panelBegin], outputImag[panelBegin], inputImag[panelBegin], panelCols);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void StoreShortTranspose(GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
    const uint32_t n = tiling.n;
    const uint32_t inputBytes = tiling.n * FLOATS_PER_COMPLEX * sizeof(float);
    LocalTensor<uint32_t> offsets = ub.gatherMask;
    LocalTensor<int32_t> offsetsI32 = ub.gatherMask.template ReinterpretCast<int32_t>();

    // Vector indices interleave the real and imaginary planes using byte offsets from ub.aAos.
    SetMaskNorm();
    AscendC::SetVectorMask<int32_t>(0U, static_cast<uint64_t>(-1));
    CreateVecIndex<int32_t>(offsetsI32, 0, n * FLOATS_PER_COMPLEX);
    // Both lanes of a complex value share one source index; reuse the lane-index tensor.
    ShiftRight<int32_t>(offsetsI32, offsetsI32, 1, n * FLOATS_PER_COMPLEX);
    Muls<int32_t>(offsetsI32, offsetsI32, static_cast<int32_t>(sizeof(float)), n * FLOATS_PER_COMPLEX);

    AscendC::SetVectorMask<int32_t>(0U, 0x5555555555555555ULL);
    Adds<int32_t, false>(
        offsetsI32, offsetsI32, static_cast<int32_t>(2048U * sizeof(float)), static_cast<uint64_t>(0U),
        static_cast<uint8_t>(1U), AscendC::UnaryRepeatParams(1U, 1U, 8U, 8U));
    AscendC::SetVectorMask<int32_t>(0U, 0xAAAAAAAAAAAAAAAAULL);
    Adds<int32_t, false>(
        offsetsI32, offsetsI32, static_cast<int32_t>(2112U * sizeof(float)), static_cast<uint64_t>(0U),
        static_cast<uint8_t>(1U), AscendC::UnaryRepeatParams(1U, 1U, 8U, 8U));
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Gather(ub.xAos, ub.aAos, offsets, 0U, n * FLOATS_PER_COMPLEX);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    ub_to_gm_align<ArchType::ASCEND_V220, float>(x, ub.xAos, 0U, 1U, inputBytes, 0U, 0U, 0U, 0U);
}

#endif

__aicore__ inline void ComputeShortUpperTransposeVector(
    GlobalTensor<float> ap, GlobalTensor<float> x, const CtpmvTilingData& tiling, const CtpmvUb& ub)
{
#if __DAV_C220_VEC__
    if (!LoadShortTranspose(ap, x, tiling, ub)) {
        ComputeOrdered(ap, x, tiling, GetCtpmvScalarUb());
        return;
    }
    auto offsets = ub.gatherMask.template ReinterpretCast<int32_t>();
    CreateVecIndex<int32_t>(offsets, 0, 64U);
    Muls<int32_t>(offsets, offsets, static_cast<int32_t>(sizeof(float)), 64U);
    PipeBarrier<PIPE_V>();
    StageShortTranspose(tiling, ub);
    ProductShortTranspose<false>(tiling, ub);
    ProductShortTranspose<true>(tiling, ub);
    ReduceShortTranspose(tiling, ub);
    StoreShortTranspose(x, tiling, ub);
#endif
}

} // namespace

__aicore__ inline void FinishCtpmvKernel()
{
    PipeBarrier<PIPE_ALL>();
    SetSyncBaseAddr(0);
    ResetMask();
}

template <CtpmvPath path, bool upper = true>
__global__ __aicore__ void ctpmv_compute_kernel(
    GM_ADDR apAddr, GM_ADDR xAddr, GM_ADDR workspaceAddr, const CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> ap;
    GlobalTensor<float> x;
    ap.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(apAddr));
    x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xAddr));
    if constexpr (path == CTPMV_PATH_STRICT_SCALAR) {
        ComputeOrdered(ap, x, tiling, GetCtpmvScalarUb());
        ResetMask();
    } else if constexpr (path == CTPMV_PATH_UPPER_TRANSPOSE_PANEL) {
        TPipe pipe;
        ComputeShortUpperTransposeVector(ap, x, tiling, GetCtpmvUb());
        pipe.Destroy();
        ResetMask();
    } else {
        SetSyncBaseAddr(tiling.syncAddress);
        constexpr uint32_t alignment = 8U;
        const uint64_t rows = (static_cast<uint64_t>(tiling.n) + alignment - 1U) & ~(alignment - 1ULL);
        GlobalTensor<float> workspace;
        workspace.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspaceAddr), tiling.coreNum * rows * FLOATS_PER_COMPLEX);
        const CtpmvUb ub = GetCtpmvUb();
        ComputeColumnPartials<upper>(ap, x, workspace, tiling, ub, GetBlockIdx());
        FinishCtpmvKernel();
    }
}

void ctpmv_kernel_do(
    GM_ADDR ap, GM_ADDR x, GM_ADDR workspace, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    switch (static_cast<CtpmvPath>(tiling.path)) {
        case CTPMV_PATH_COLUMN_PARTIALS:
            if (tiling.uplo == ACLBLAS_UPPER) {
                ctpmv_compute_kernel<CTPMV_PATH_COLUMN_PARTIALS, true>
                    <<<numBlocks, nullptr, stream>>>(ap, x, workspace, tiling);
            } else {
                ctpmv_compute_kernel<CTPMV_PATH_COLUMN_PARTIALS, false>
                    <<<numBlocks, nullptr, stream>>>(ap, x, workspace, tiling);
            }
            return;
        case CTPMV_PATH_STRICT_SCALAR:
            ctpmv_compute_kernel<CTPMV_PATH_STRICT_SCALAR><<<numBlocks, nullptr, stream>>>(ap, x, workspace, tiling);
            return;
        case CTPMV_PATH_UPPER_TRANSPOSE_PANEL:
            ctpmv_compute_kernel<CTPMV_PATH_UPPER_TRANSPOSE_PANEL>
                <<<numBlocks, nullptr, stream>>>(ap, x, workspace, tiling);
            return;
        case CTPMV_PATH_BATCHED_REFERENCE:
            ctpmv_batched_reference_kernel_do(ap, x, tiling, numBlocks, stream);
            return;
    }
}
