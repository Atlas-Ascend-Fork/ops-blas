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
 * \file sgemv_kernel.cpp
 * \brief FLOAT32 Sgemv kernel for Atlas A2/A3.
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "common/compute/simd.h"
#include "common/iterator/iterator.h"
#include "common/helper/kernel_utils.h"
#include "../../trsv/arch22/ctrsv_kernel_utils.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sgemv_tiling_data.h"

using namespace AscendC;
using namespace matmul;

namespace {
union SgemvFloatBits {
    float value;
    uint32_t bits;
};

__aicore__ inline bool IsFiniteFp32(float value)
{
    SgemvFloatBits data{value};
    return (data.bits & 0x7f800000U) != 0x7f800000U;
}

__aicore__ inline uint32_t ExponentFp32(float value)
{
    SgemvFloatBits data{value};
    return (data.bits >> 23U) & 0xffU;
}

__aicore__ inline float RoundAddFp32(float lhs, float rhs)
{
    volatile float result = lhs + rhs;
    return result;
}

__aicore__ inline float RoundSubFp32(float lhs, float rhs)
{
    volatile float result = lhs - rhs;
    return result;
}

__aicore__ inline float RoundMulFp32(float lhs, float rhs)
{
    volatile float result = lhs * rhs;
    return result;
}

__aicore__ inline float TwoSumHigh(float lhs, float rhs) { return RoundAddFp32(lhs, rhs); }

__aicore__ inline float TwoSumError(float lhs, float rhs, float sum)
{
    const float rhsVirtual = RoundSubFp32(sum, lhs);
    const float lhsError = RoundSubFp32(lhs, RoundSubFp32(sum, rhsVirtual));
    const float rhsError = RoundSubFp32(rhs, rhsVirtual);
    return RoundAddFp32(lhsError, rhsError);
}

__aicore__ inline void TwoProduct(float lhs, float rhs, float& product, float& error)
{
    constexpr float splitter = 4097.0f;
    product = RoundMulFp32(lhs, rhs);
    const float lhsScaled = RoundMulFp32(lhs, splitter);
    const float lhsHigh = RoundSubFp32(lhsScaled, RoundSubFp32(lhsScaled, lhs));
    const float lhsLow = RoundSubFp32(lhs, lhsHigh);
    const float rhsScaled = RoundMulFp32(rhs, splitter);
    const float rhsHigh = RoundSubFp32(rhsScaled, RoundSubFp32(rhsScaled, rhs));
    const float rhsLow = RoundSubFp32(rhs, rhsHigh);
    const float highProduct = RoundSubFp32(RoundMulFp32(lhsHigh, rhsHigh), product);
    const float crossLow = RoundAddFp32(RoundMulFp32(lhsHigh, rhsLow), RoundMulFp32(lhsLow, rhsHigh));
    error = RoundAddFp32(RoundAddFp32(highProduct, crossLow), RoundMulFp32(lhsLow, rhsLow));
}

__aicore__ inline float FmaFp32(float lhs, float rhs, float addend)
{
    if (!IsFiniteFp32(lhs) || !IsFiniteFp32(rhs) || !IsFiniteFp32(addend)) {
        return lhs * rhs + addend;
    }
    const float roundedProduct = RoundMulFp32(lhs, rhs);
    if (!IsFiniteFp32(roundedProduct)) {
        return ctrsv_netlib::Fma(lhs, rhs, addend);
    }
    if (ExponentFp32(lhs) >= 116U || ExponentFp32(rhs) >= 116U) {
        return ctrsv_netlib::Fma(lhs, rhs, addend);
    }
    float product = 0.0f;
    float productError = 0.0f;
    TwoProduct(lhs, rhs, product, productError);

    const float sum = TwoSumHigh(product, addend);
    const float sumError = TwoSumError(product, addend, sum);
    const float residualSum = TwoSumHigh(sumError, productError);
    const float residualError = TwoSumError(sumError, productError, residualSum);
    const float high = TwoSumHigh(sum, residualSum);
    const float low = TwoSumError(sum, residualSum, high);
    return RoundAddFp32(high, RoundAddFp32(low, residualError));
}

constexpr uint32_t SGEMMV_NORMAL_T_COL_CHUNK = 16;
constexpr uint32_t SGEMMV_BYTES_PER_FLOAT = 4;
constexpr uint32_t SGEMMV_ACC_FLOATS = 2048;
constexpr uint32_t SGEMMV_A_FLOATS = 16384;
constexpr uint32_t SGEMMV_X_FLOATS = 4608;
constexpr uint32_t SGEMMV_WORK_FLOATS = SGEMMV_SMALL_N_COL_TILE;
constexpr uint32_t SGEMMV_ROW_FLOATS = SGEMMV_SMALL_N_COL_TILE;
constexpr uint32_t SGEMMV_REDUCE_FLOATS = 64;
constexpr uint32_t SGEMMV_OUTPUT_FLOATS = 256;
constexpr uint32_t SGEMMV_OFFSET_COUNT = SGEMMV_SMALL_N_COL_TILE;
constexpr uint32_t SGEMMV_FAST_N_A_FLOATS = 16384;
constexpr uint32_t SGEMMV_FAST_N_WORK_FLOATS = 16384;
constexpr uint32_t SGEMMV_FAST_N_VEC_FLOATS = 2048;
constexpr uint32_t SGEMMV_FAST_N_COL_BATCH = 64;
constexpr int64_t SGEMMV_UINT16_STRIDE_LIMIT = 65536;
constexpr int32_t SGEMMV_CUBE_COL_TILE = SGEMMV_CUBE_MAX_COLS;
constexpr uint32_t SGEMMV_SPLIT_MAX = SGEMMV_CUBE_SPLIT_HARD_MAX;
constexpr uint32_t SGEMMV_SPLIT_CHUNK = SGEMMV_CUBE_SPLIT_REDUCE_TILE;

using SgemvDenseFp32 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams SGEMMV_CUBE_SHAPE{16, SGEMMV_CUBE_COL_TILE, 8192, 16, 512, 32};
constexpr MatmulFuncParams SGEMMV_CUBE_FUNC{
    false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true};
constexpr MatmulBiasParams SGEMMV_CUBE_BIAS{false};
constexpr MatmulConfig SGEMMV_CUBE_CONFIG =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(SGEMMV_CUBE_SHAPE, SGEMMV_CUBE_FUNC, SGEMMV_CUBE_BIAS);
constexpr auto SGEMMV_CUBE_TILING =
    GetMatmulApiTiling<SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32>(SGEMMV_CUBE_CONFIG);
constexpr MatmulShapeParams SGEMMV_ODD_CUBE_SHAPE{16, SGEMMV_CUBE_ODD_COLS, 8192, 16, SGEMMV_CUBE_BASE_COLS, 32};
constexpr MatmulConfig SGEMMV_ODD_CUBE_CONFIG =
    GetMMConfig<MatmulConfigMode::CONFIG_NORM>(SGEMMV_ODD_CUBE_SHAPE, SGEMMV_CUBE_FUNC, SGEMMV_CUBE_BIAS);
constexpr auto SGEMMV_ODD_CUBE_TILING =
    GetMatmulApiTiling<SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32>(SGEMMV_ODD_CUBE_CONFIG);
using SgemvCubeMatmul = MatmulImpl<SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SGEMMV_CUBE_TILING>;
using SgemvOddCubeMatmul =
    MatmulImpl<SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SgemvDenseFp32, SGEMMV_ODD_CUBE_TILING>;

template <typename MatmulType>
__aicore__ inline void ComputeCubeTile(
    GlobalTensor<float> left, GlobalTensor<float> right, GlobalTensor<float> output, uint32_t outDim, uint32_t lda,
    uint32_t cols, uint32_t kLen)
{
    TPipe pipe;
    MatmulType matmul;
    matmul.SetSubBlockIdx(0);
    matmul.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    matmul.DisableBias();
    matmul.SetOrgShape(1, lda, kLen, kLen, outDim);
    matmul.SetSingleShape(1, cols, kLen);
    matmul.SetTensorA(left, false);
    matmul.SetTensorB(right, false);
    matmul.IterateAll(output, 0);
    matmul.End();
}

__aicore__ inline void CopyMatrixTile(
    LocalTensor<float> dst, GlobalTensor<float> src, int64_t rowsActual, int64_t colsActual, int64_t srcStride,
    int64_t dstStride, bool batchPadding = false)
{
    const int64_t rowsAligned = (rowsActual + SGEMMV_ALIGN - 1) / SGEMMV_ALIGN * SGEMMV_ALIGN;
    if (rowsActual % SGEMMV_ALIGN == 0 && srcStride % SGEMMV_ALIGN == 0 && srcStride < SGEMMV_UINT16_STRIDE_LIMIT) {
        gm_to_ub<ArchType::ASCEND_V220, float>(
            dst, src, 0, colsActual, rowsAligned / SGEMMV_ALIGN, (srcStride - rowsAligned) / SGEMMV_ALIGN,
            (dstStride - rowsAligned) / SGEMMV_ALIGN);
    } else if (rowsActual % SGEMMV_ALIGN == 0 && srcStride * SGEMMV_ALIGN < SGEMMV_UINT16_STRIDE_LIMIT) {
        const int64_t outer = colsActual / SGEMMV_ALIGN;
        const int64_t tail = colsActual % SGEMMV_ALIGN;
        for (int64_t i = 0; i < SGEMMV_ALIGN; ++i) {
            if (outer > 0) {
                gm_to_ub<ArchType::ASCEND_V220, float>(
                    dst[i * dstStride], src[i * srcStride], 0, outer, rowsAligned / SGEMMV_ALIGN,
                    (srcStride * SGEMMV_ALIGN - rowsAligned) / SGEMMV_ALIGN,
                    (dstStride * SGEMMV_ALIGN - rowsAligned) / SGEMMV_ALIGN);
            }
        }
        for (int64_t j = 0; j < tail; ++j) {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                dst[outer * SGEMMV_ALIGN * dstStride + j * dstStride],
                src[outer * SGEMMV_ALIGN * srcStride + j * srcStride], 0, 1, rowsActual * SGEMMV_BYTES_PER_FLOAT, 0, 0,
                0, 0);
        }
    } else if (
        batchPadding && colsActual <= 4095 && dstStride % SGEMMV_ALIGN == 0 &&
        srcStride - rowsActual <= 0xffffffffLL / SGEMMV_BYTES_PER_FLOAT) {
        DataCopyExtParams copy{
            static_cast<uint16_t>(colsActual), static_cast<uint32_t>(rowsActual * SGEMMV_BYTES_PER_FLOAT),
            static_cast<uint32_t>((srcStride - rowsActual) * SGEMMV_BYTES_PER_FLOAT),
            static_cast<uint32_t>((dstStride - rowsAligned) / SGEMMV_ALIGN), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(rowsAligned - rowsActual), 0.0f};
        DataCopyPad(dst, src, copy, pad);
    } else {
        for (int64_t col = 0; col < colsActual; ++col) {
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                dst[col * dstStride], src[col * srcStride], 0, 1, rowsActual * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0, 0);
        }
    }
}
} // namespace

class SgemvKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SgemvTilingData& tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        m_ = tiling.m;
        n_ = tiling.n;
        lda_ = tiling.lda;
        outPerBlock_ = tiling.outPerBlock;
        path_ = tiling.path;
        const bool isNPath =
            (path_ == static_cast<uint32_t>(SgemvPath::NORMAL_N) ||
             path_ == static_cast<uint32_t>(SgemvPath::SMALL_M_N));
        xLength_ = isNPath ? n_ : m_;
        yLength_ = isNPath ? m_ : n_;
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;
        incx_ = tiling.incx;
        incy_ = tiling.incy;
        absIncX_ = static_cast<uint32_t>(incx_ > 0 ? incx_ : -incx_);
        absIncY_ = static_cast<uint32_t>(incy_ > 0 ? incy_ : -incy_);

        if (path_ == static_cast<uint32_t>(SgemvPath::NORMAL_N) && UseFastPath()) {
            pipe_->InitBuffer(aBuf_, SGEMMV_FAST_N_A_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(workBuf_, SGEMMV_FAST_N_WORK_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(xBuf_, SGEMMV_FAST_N_VEC_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(accBuf_, SGEMMV_FAST_N_VEC_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            FetchEvents();
            return;
        }
        if (path_ == static_cast<uint32_t>(SgemvPath::NORMAL_T) && UseFastPath() && m_ <= SGEMMV_FAST_T_MAX_ROWS) {
            pipe_->InitBuffer(aBuf_, 40960U * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(xBuf_, SGEMMV_X_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(rowBuf_, 640U * SGEMMV_BYTES_PER_FLOAT);
            pipe_->InitBuffer(outputBuf_, SGEMMV_OUTPUT_FLOATS * SGEMMV_BYTES_PER_FLOAT);
            FetchEvents();
            return;
        }
        pipe_->InitBuffer(accBuf_, SGEMMV_ACC_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(aBuf_, SGEMMV_A_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(xBuf_, SGEMMV_X_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(productBuf_, SGEMMV_X_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(rowBuf_, SGEMMV_ROW_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(workBuf_, SGEMMV_WORK_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(reduceBuf_, SGEMMV_REDUCE_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(outputBuf_, SGEMMV_OUTPUT_FLOATS * SGEMMV_BYTES_PER_FLOAT);
        pipe_->InitBuffer(offsetBuf_, SGEMMV_OFFSET_COUNT * sizeof(uint32_t));
        FetchEvents();
    }

    __aicore__ inline void Process()
    {
        const uint32_t block = static_cast<uint32_t>(GetBlockIdx());
        if (path_ == static_cast<uint32_t>(SgemvPath::NORMAL_N)) {
            ProcessNormalN(block);
        } else if (path_ == static_cast<uint32_t>(SgemvPath::NORMAL_T)) {
            ProcessNormalT(block);
        } else if (path_ == static_cast<uint32_t>(SgemvPath::SMALL_M_T)) {
            ProcessSmallMT(block);
        }
    }

    __aicore__ inline void ProcessSmallMN(uint32_t block) { ProcessSmallMCompute(block); }

    __aicore__ inline void InitOutput()
    {
        const uint32_t count = m_;
        const uint32_t countAligned = AlignUp8(count);
        LocalTensor<float> acc = accBuf_.Get<float>();
        if (beta_ == 0.0f) {
            Duplicate(acc, 0.0f, countAligned);
        } else {
            LoadY(0U, count, acc);
            Muls(acc, acc, beta_, countAligned);
            SyncVToMte3();
        }
        if (beta_ == 0.0f) {
            SyncVToMte3();
        }
        StoreYContiguous(0U, count, acc);
    }

private:
    __aicore__ static inline uint32_t AlignUp8(uint32_t value)
    {
        return (value + SGEMMV_ALIGN - 1U) / SGEMMV_ALIGN * SGEMMV_ALIGN;
    }

    __aicore__ inline uint32_t XPhysical(uint32_t logical) const
    {
        if (incx_ > 0) {
            return logical * absIncX_;
        }
        return (xLength_ - 1U - logical) * absIncX_;
    }

    __aicore__ inline uint32_t YPhysical(uint32_t logical) const
    {
        if (incy_ > 0) {
            return logical * absIncY_;
        }
        return (yLength_ - 1U - logical) * absIncY_;
    }

    __aicore__ inline bool UseFastPath() const { return alpha_ == 1.0f && beta_ == 0.0f && incx_ == 1 && incy_ == 1; }

    __aicore__ inline void FetchEvents()
    {
        eventMte2ToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVToMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventVToS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        eventSToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        eventVToMte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        eventMte2ToS_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        eventMte3ToMte2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        eventMte3ToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void SyncMte2ToV()
    {
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV_);
    }

    __aicore__ inline void SyncVToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(eventVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3_);
    }

    __aicore__ inline void SyncVToS()
    {
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
    }

    __aicore__ inline void SyncSToV()
    {
        SetFlag<HardEvent::S_V>(eventSToV_);
        WaitFlag<HardEvent::S_V>(eventSToV_);
    }

    __aicore__ inline void SyncVToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(eventVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventVToMte2_);
    }

    __aicore__ inline void SyncMte2ToS()
    {
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    }

    __aicore__ inline void SyncMte3ToV()
    {
        SetFlag<HardEvent::MTE3_V>(eventMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventMte3ToV_);
    }

    __aicore__ inline void LoadX(uint32_t logicalStart, uint32_t count, LocalTensor<float> dst)
    {
        const uint32_t countAligned = AlignUp8(count);
        if (incx_ == 1) {
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(countAligned - count), 0.0f};
            DataCopyExtParams copy{1, count * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(dst, xGm_[logicalStart], copy, pad);
            SyncMte2ToV();
        } else {
            Duplicate(dst, 0.0f, countAligned);
            SyncVToS();
            for (uint32_t i = 0; i < count; ++i) {
                dst.SetValue(i, xGm_.GetValue(XPhysical(logicalStart + i)));
            }
            SyncSToV();
        }
    }

    __aicore__ inline void LoadY(uint32_t logicalStart, uint32_t count, LocalTensor<float> dst)
    {
        const uint32_t countAligned = AlignUp8(count);
        if (incy_ == 1 && beta_ == 0.0f) {
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(countAligned - count), 0.0f};
            DataCopyExtParams copy{1, count * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(dst, yGm_[logicalStart], copy, pad);
            SyncMte2ToV();
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                dst.SetValue(i, yGm_.GetValue(YPhysical(logicalStart + i)));
            }
            SyncSToV();
        }
    }

    __aicore__ inline void StoreYContiguous(uint32_t logicalStart, uint32_t count, LocalTensor<float> src)
    {
        if (incy_ == 1 && beta_ == 0.0f) {
            DataCopyExtParams copy{1, count * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(yGm_[logicalStart], src, copy);
        } else {
            SyncVToS();
            for (uint32_t i = 0; i < count; ++i) {
                yGm_.SetValue(YPhysical(logicalStart + i), src.GetValue(i));
            }
        }
    }

    __aicore__ inline void FinishYChunk()
    {
        if (incy_ == 1 && beta_ == 0.0f) {
            SyncMte3ToMte2();
        }
    }

    __aicore__ inline void LoadA2D(
        GlobalTensor<float> src, uint32_t srcOffset, uint32_t rows, uint32_t cols, uint32_t dstStride,
        LocalTensor<float> dst)
    {
        if (cols == 1U) {
            for (uint32_t i = 0; i < rows; ++i) {
                dst.SetValue(i, src.GetValue(srcOffset + i));
            }
            SyncSToV();
        } else {
            CopyMatrixTile(
                dst, src[srcOffset], static_cast<int64_t>(rows), static_cast<int64_t>(cols), static_cast<int64_t>(lda_),
                static_cast<int64_t>(dstStride));
            SyncMte2ToV();
        }
    }

    __aicore__ inline void LoadAColumn(uint32_t column, uint32_t rowStart, uint32_t count, LocalTensor<float> dst)
    {
        const uint32_t countAligned = AlignUp8(count);
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(countAligned - count), 0.0f};
        DataCopyExtParams copy{1, count * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
        DataCopyPad(dst, aGm_[static_cast<uint64_t>(column) * lda_ + rowStart], copy, pad);
        SyncMte2ToV();
    }

    __aicore__ inline void LoadAContiguous(uint32_t srcOffset, uint32_t count, LocalTensor<float> dst)
    {
        constexpr uint32_t maxPiece = 8192;
        uint32_t done = 0;
        while (done < count) {
            const uint32_t piece = (count - done < maxPiece) ? (count - done) : maxPiece;
            const uint32_t pieceAligned = AlignUp8(piece);
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(pieceAligned - piece), 0.0f};
            DataCopyExtParams copy{1, piece * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(dst[done], aGm_[srcOffset + done], copy, pad);
            done += piece;
        }
        SyncMte2ToV();
    }

    __aicore__ inline void ProcessNormalN(uint32_t block)
    {
        if (UseFastPath()) {
            ProcessFastN(block);
            return;
        }
        ProcessReferenceN(block);
    }

    __aicore__ inline void ProcessFastN(uint32_t block)
    {
        const uint32_t rowBegin = block * outPerBlock_;
        const uint32_t rowEnd = (rowBegin + outPerBlock_ < m_) ? (rowBegin + outPerBlock_) : m_;
        if (rowBegin >= rowEnd) {
            return;
        }
        LocalTensor<float> aTile = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        const bool useTree = n_ > SGEMMV_ALIGN;
        const uint32_t rowTile = useTree ? 128U : SGEMMV_FAST_N_VEC_FLOATS;
        for (uint32_t rowStart = rowBegin; rowStart < rowEnd; rowStart += rowTile) {
            const uint32_t rows = (rowStart + rowTile < rowEnd) ? rowTile : (rowEnd - rowStart);
            const uint32_t rowsAligned = AlignUp8(rows);
            if (rowsAligned == 0U) {
                return;
            }
            const uint32_t capacityBatch =
                useTree ? SGEMMV_FAST_N_WORK_FLOATS / rowsAligned : SGEMMV_FAST_N_A_FLOATS / rowsAligned;
            const uint32_t colBatch =
                (capacityBatch < SGEMMV_FAST_N_COL_BATCH) ? capacityBatch : SGEMMV_FAST_N_COL_BATCH;
            Duplicate(acc, 0.0f, rowsAligned);
            bool firstBatch = true;
            for (uint32_t colStart = 0; colStart < n_; colStart += colBatch) {
                const uint32_t cols = (colStart + colBatch < n_) ? colBatch : (n_ - colStart);
                if (!firstBatch) {
                    SyncVToMte2();
                }
                firstBatch = false;
                CopyMatrixTile(
                    aTile, aGm_[static_cast<uint64_t>(colStart) * lda_ + rowStart], rows, cols, lda_, rowsAligned,
                    true);
                LoadX(colStart, cols, xTile);
                SyncMte2ToS();
                AccumulateFastNBatch(acc, aTile, xTile, rows, rowsAligned, cols);
            }
            SyncVToMte3();
            StoreYContiguous(rowStart, rows, acc);
            SyncMte3ToV();
        }
    }

    __aicore__ inline void AccumulateFastNBatch(
        LocalTensor<float> acc, LocalTensor<float> aTile, LocalTensor<float> xTile, uint32_t rows, uint32_t rowsAligned,
        uint32_t cols)
    {
        LocalTensor<float> work = workBuf_.Get<float>();
        if (n_ <= SGEMMV_ALIGN || cols % SGEMMV_ALIGN != 0U) {
            for (uint32_t col = 0; col < cols; ++col) {
                Axpy(acc, aTile[col * rowsAligned], xTile.GetValue(col), rows);
                PIPE_BARRIER(V);
            }
            return;
        }
        const uint16_t rowBlocks = static_cast<uint16_t>(rowsAligned / SGEMMV_ALIGN);
        const uint8_t repeatTimes = static_cast<uint8_t>(cols / SGEMMV_ALIGN);
        BrcbRepeatParams brcbParams(rowBlocks, static_cast<uint16_t>(rowBlocks * SGEMMV_ALIGN));
        for (uint32_t rowBlock = 0; rowBlock < rowBlocks; ++rowBlock) {
            Brcb(work[rowBlock * SGEMMV_ALIGN], xTile, repeatTimes, brcbParams);
        }
        PIPE_BARRIER(V);
        Mul(aTile, aTile, work, cols * rowsAligned);
        PIPE_BARRIER(V);
        for (uint32_t half = cols / 2U; half >= 1U; half >>= 1U) {
            Add(aTile, aTile, aTile[half * rowsAligned], half * rowsAligned);
            PIPE_BARRIER(V);
        }
        Add(acc, acc, aTile, rowsAligned);
    }

    __aicore__ inline void ProcessReferenceN(uint32_t block)
    {
        const uint32_t outBegin = block * outPerBlock_;
        const uint32_t outEnd = (outBegin + outPerBlock_ < m_) ? (outBegin + outPerBlock_) : m_;
        if (outBegin >= outEnd) {
            return;
        }
        if (alpha_ == 0.0f) {
            if (beta_ == 0.0f) {
                for (uint32_t row = outBegin; row < outEnd; ++row) {
                    yGm_.SetValue(YPhysical(row), 0.0f);
                }
            } else {
                for (uint32_t row = outBegin; row < outEnd; ++row) {
                    yGm_.SetValue(YPhysical(row), beta_ * yGm_.GetValue(YPhysical(row)));
                }
            }
            return;
        }
        for (uint32_t row = outBegin; row < outEnd; ++row) {
            float result = 0.0f;
            if (beta_ != 0.0f) {
                result = RoundMulFp32(beta_, yGm_.GetValue(YPhysical(row)));
            }
            for (uint32_t col = 0; col < n_; ++col) {
                const float scaledX = RoundMulFp32(alpha_, xGm_.GetValue(XPhysical(col)));
                result = FmaFp32(aGm_.GetValue(static_cast<uint64_t>(col) * lda_ + row), scaledX, result);
            }
            yGm_.SetValue(YPhysical(row), result);
        }
    }

    __aicore__ inline void ProcessReferenceT(uint32_t block)
    {
        const uint32_t outBegin = block * outPerBlock_;
        const uint32_t outEnd = (outBegin + outPerBlock_ < n_) ? (outBegin + outPerBlock_) : n_;
        if (outBegin >= outEnd) {
            return;
        }
        if (alpha_ == 0.0f) {
            if (beta_ == 0.0f) {
                for (uint32_t col = outBegin; col < outEnd; ++col) {
                    yGm_.SetValue(YPhysical(col), 0.0f);
                }
            } else {
                for (uint32_t col = outBegin; col < outEnd; ++col) {
                    yGm_.SetValue(YPhysical(col), beta_ * yGm_.GetValue(YPhysical(col)));
                }
            }
            return;
        }
        for (uint32_t col = outBegin; col < outEnd; ++col) {
            float sum = 0.0f;
            for (uint32_t row = 0; row < m_; ++row) {
                sum =
                    FmaFp32(aGm_.GetValue(static_cast<uint64_t>(col) * lda_ + row), xGm_.GetValue(XPhysical(row)), sum);
            }
            float result = 0.0f;
            if (beta_ != 0.0f) {
                const float betaProduct = RoundMulFp32(beta_, yGm_.GetValue(YPhysical(col)));
                result = FmaFp32(alpha_, sum, betaProduct);
            } else {
                result = alpha_ * sum;
            }
            yGm_.SetValue(YPhysical(col), result);
        }
    }

    __aicore__ inline void ReduceFastT(
        LocalTensor<float> aTile, LocalTensor<float> output, uint32_t cols, uint32_t rowTileRows)
    {
        LocalTensor<float> sums = rowBuf_.Get<float>();
        constexpr uint32_t lanes = 64;
        // Output slot offset of the second-pass reduce: 2 * lanes keeps the two
        // partial results (the first pass starts at slot 0) from overlapping.
        constexpr uint32_t splitSlot = 2U * lanes;
        const uint32_t segments = rowTileRows / lanes;
        const uint32_t repeats = cols * segments;
        // A single WholeReduceSum supports at most 248 repeat times, so cap each call.
        for (uint32_t done = 0; done < repeats; done += 248U) {
            const uint32_t count = repeats - done < 248U ? repeats - done : 248U;
            WholeReduceSum(sums[done], aTile[done * lanes], lanes, count, 1, 1, lanes / SGEMMV_ALIGN);
        }
        PIPE_BARRIER(V);
        if (segments <= lanes) {
            WholeReduceSum(output, sums, segments, cols, 1, 1, segments / SGEMMV_ALIGN);
        } else {
            WholeReduceSum(output, sums, lanes, cols, 1, 1, segments / SGEMMV_ALIGN);
            WholeReduceSum(output[splitSlot], sums[lanes], segments - lanes, cols, 1, 1, segments / SGEMMV_ALIGN);
            PIPE_BARRIER(V);
            Add(output, output, output[splitSlot], cols);
        }
    }

    __aicore__ inline void ProcessPipelinedT(uint32_t outBegin, uint32_t outEnd)
    {
        constexpr uint32_t slotSize = 20480;
        const uint32_t rowStride = CeilDiv(m_, 512U) * 512U;
        const uint32_t colTile = slotSize / rowStride;
        LocalTensor<float> aTile = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        if (m_ != rowStride) {
            Duplicate(aTile, 0.0f, 2U * slotSize);
            Duplicate(xTile, 0.0f, rowStride);
            SyncVToMte2();
        }
        LoadX(0U, m_, xTile);
        const uint32_t firstCols = outEnd - outBegin < colTile ? outEnd - outBegin : colTile;
        CopyMatrixTile(aTile, aGm_[static_cast<uint64_t>(outBegin) * lda_], m_, firstCols, lda_, rowStride, true);
        SyncMte2ToV();
        uint32_t slot = 0;
        for (uint32_t start = outBegin; start < outEnd; start += colTile) {
            const uint32_t cols = outEnd - start < colTile ? outEnd - start : colTile;
            const uint32_t next = start + colTile;
            if (next < outEnd) {
                const uint32_t nextCols = outEnd - next < colTile ? outEnd - next : colTile;
                SyncVToMte2();
                CopyMatrixTile(
                    aTile[(1U - slot) * slotSize], aGm_[static_cast<uint64_t>(next) * lda_], m_, nextCols, lda_,
                    rowStride, true);
            }
            LocalTensor<float> current = aTile[slot * slotSize];
            for (uint32_t col = 0; col < cols; ++col) {
                Mul(current[col * rowStride], current[col * rowStride], xTile, rowStride);
            }
            PIPE_BARRIER(V);
            ReduceFastT(current, output, cols, rowStride);
            SyncVToMte3();
            StoreYContiguous(start, cols, output);
            SyncMte3ToV();
            if (next < outEnd) {
                SyncMte2ToV();
            }
            slot = 1U - slot;
        }
    }

    __aicore__ inline void ProcessFastT(uint32_t block)
    {
        const uint32_t groups = CeilDiv(n_, SGEMMV_ALIGN);
        const uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
        if (blocks == 0U) {
            return;
        }
        const uint32_t outBegin = static_cast<uint64_t>(block) * groups / blocks * SGEMMV_ALIGN;
        const uint32_t alignedEnd = static_cast<uint64_t>(block + 1U) * groups / blocks * SGEMMV_ALIGN;
        const uint32_t outEnd = alignedEnd < n_ ? alignedEnd : n_;
        if (outBegin >= outEnd) {
            return;
        }
        if (m_ >= 3072U) {
            ProcessPipelinedT(outBegin, outEnd);
            return;
        }
        LocalTensor<float> aTile = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();
        constexpr uint32_t rowAlign = 512;
        const uint32_t rowTileRows = CeilDiv(m_, rowAlign) * rowAlign;
        const uint32_t colTile = 40960U / rowTileRows;
        if (m_ != rowTileRows) {
            Duplicate(xTile, 0.0f, rowTileRows);
            SyncVToMte2();
        }
        LoadX(0U, m_, xTile);
        for (uint32_t colStart = outBegin; colStart < outEnd; colStart += colTile) {
            const uint32_t cols = (colStart + colTile < outEnd) ? colTile : (outEnd - colStart);
            if (m_ != rowTileRows) {
                Duplicate(aTile, 0.0f, cols * rowTileRows);
                SyncVToMte2();
            }
            CopyMatrixTile(aTile, aGm_[static_cast<uint64_t>(colStart) * lda_], m_, cols, lda_, rowTileRows, true);
            SyncMte2ToV();
            for (uint32_t col = 0; col < cols; ++col) {
                const uint32_t offset = col * rowTileRows;
                Mul(aTile[offset], aTile[offset], xTile, rowTileRows);
            }
            PIPE_BARRIER(V);
            ReduceFastT(aTile, output, cols, rowTileRows);
            SyncVToMte3();
            StoreYContiguous(colStart, cols, output);
            FinishYChunk();
            SyncMte3ToV();
        }
    }

    __aicore__ inline bool PrepareTransposeBlock(uint32_t block, uint32_t& outBegin, uint32_t& outEnd)
    {
        if (!UseFastPath()) {
            ProcessReferenceT(block);
            return false;
        }
        outBegin = block * outPerBlock_;
        outEnd = (outBegin + outPerBlock_ < n_) ? (outBegin + outPerBlock_) : n_;
        return outBegin < outEnd;
    }

    __aicore__ inline void AccumulateNormalT(uint32_t colStart, uint32_t cols, LocalTensor<float> output)
    {
        LocalTensor<float> aCol = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> product = productBuf_.Get<float>();
        LocalTensor<float> reduceTmp = workBuf_.Get<float>();
        LocalTensor<float> reduceOut = reduceBuf_.Get<float>();
        bool firstRowTile = true;
        for (uint32_t rowStart = 0; rowStart < m_; rowStart += SGEMMV_T_REDUCE_TILE) {
            const uint32_t rows = (rowStart + SGEMMV_T_REDUCE_TILE < m_) ? SGEMMV_T_REDUCE_TILE : (m_ - rowStart);
            const uint32_t rowsAligned = AlignUp8(rows);
            if (!firstRowTile) {
                SyncVToMte2();
            }
            LoadX(rowStart, rows, xTile);
            firstRowTile = false;
            for (uint32_t col = 0; col < cols; ++col) {
                if (col != 0U) {
                    SyncVToMte2();
                }
                LoadAColumn(colStart + col, rowStart, rows, aCol);
                Mul(product, aCol, xTile, rowsAligned);
                PIPE_BARRIER(V);
                ReduceSum(reduceOut, product, reduceTmp, static_cast<int32_t>(rowsAligned));
                SyncVToS();
                const uint32_t compensationOffset = SGEMMV_NORMAL_T_COL_CHUNK + col;
                const float corrected = reduceOut.GetValue(0) - output.GetValue(compensationOffset);
                const float previous = output.GetValue(col);
                const float current = previous + corrected;
                output.SetValue(compensationOffset, (current - previous) - corrected);
                output.SetValue(col, current);
            }
        }
    }

    __aicore__ inline void ProcessNormalT(uint32_t block)
    {
        if (UseFastPath() && m_ <= SGEMMV_FAST_T_MAX_ROWS) {
            ProcessFastT(block);
            return;
        }
        uint32_t outBegin;
        uint32_t outEnd;
        if (!PrepareTransposeBlock(block, outBegin, outEnd)) {
            return;
        }

        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> output = outputBuf_.Get<float>();

        for (uint32_t colStart = outBegin; colStart < outEnd; colStart += SGEMMV_NORMAL_T_COL_CHUNK) {
            const uint32_t cols =
                (colStart + SGEMMV_NORMAL_T_COL_CHUNK < outEnd) ? SGEMMV_NORMAL_T_COL_CHUNK : (outEnd - colStart);
            Duplicate(output, 0.0f, SGEMMV_OUTPUT_FLOATS);
            SyncVToS();

            AccumulateNormalT(colStart, cols, output);

            if (beta_ == 0.0f) {
                Duplicate(acc, 0.0f, SGEMMV_NORMAL_T_COL_CHUNK);
            } else {
                LoadY(colStart, cols, acc);
                Muls(acc, acc, beta_, SGEMMV_NORMAL_T_COL_CHUNK);
            }
            SyncSToV();
            Muls(output, output, alpha_, SGEMMV_NORMAL_T_COL_CHUNK);
            PIPE_BARRIER(V);
            if (beta_ != 0.0f) {
                Add(acc, acc, output, SGEMMV_NORMAL_T_COL_CHUNK);
            } else {
                Muls(acc, output, 1.0f, SGEMMV_NORMAL_T_COL_CHUNK);
            }
            SyncVToMte3();
            StoreYContiguous(colStart, cols, acc);
            FinishYChunk();
            SyncVToMte2();
        }
    }

    __aicore__ inline void BuildGatherOffsets(uint32_t stride)
    {
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        LocalTensor<int32_t> table = offsets.ReinterpretCast<int32_t>();
        constexpr uint32_t offsetBlock = 64;
        const uint32_t head = (SGEMMV_OFFSET_COUNT < offsetBlock) ? SGEMMV_OFFSET_COUNT : offsetBlock;
        for (uint32_t i = 0; i < head; ++i) {
            table.SetValue(i, static_cast<int32_t>(i * stride * SGEMMV_BYTES_PER_FLOAT));
        }
        SyncSToV();
        for (uint32_t off = head; off < SGEMMV_OFFSET_COUNT; off += offsetBlock) {
            const uint32_t lanes =
                (SGEMMV_OFFSET_COUNT - off < offsetBlock) ? (SGEMMV_OFFSET_COUNT - off) : offsetBlock;
            Adds(
                table[off], table[off - offsetBlock],
                static_cast<int32_t>(offsetBlock * stride * SGEMMV_BYTES_PER_FLOAT), static_cast<int32_t>(lanes));
        }
    }

    __aicore__ inline void ProcessSmallMT(uint32_t block)
    {
        uint32_t outBegin;
        uint32_t outEnd;
        if (!PrepareTransposeBlock(block, outBegin, outEnd)) {
            return;
        }

        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> aTile = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> row = rowBuf_.Get<float>();
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        const uint32_t rowsAligned = AlignUp8(m_);

        LoadX(0U, m_, xTile);
        BuildGatherOffsets(rowsAligned);
        bool firstTile = true;
        for (uint32_t colStart = outBegin; colStart < outEnd; colStart += SGEMMV_T_COL_TILE) {
            const uint32_t cols = (colStart + SGEMMV_T_COL_TILE < outEnd) ? SGEMMV_T_COL_TILE : (outEnd - colStart);
            const uint32_t colsAligned = AlignUp8(cols);
            if (!firstTile) {
                SyncVToMte2();
            }
            LoadA2D(aGm_, static_cast<uint64_t>(colStart) * lda_, m_, cols, rowsAligned, aTile);
            firstTile = false;
            if (beta_ == 0.0f) {
                Duplicate(acc, 0.0f, colsAligned);
            } else {
                LoadY(colStart, cols, acc);
                Muls(acc, acc, beta_, colsAligned);
            }
            for (uint32_t rowIndex = 0; rowIndex < m_; ++rowIndex) {
                Gather(row, aTile, offsets, rowIndex * SGEMMV_BYTES_PER_FLOAT, colsAligned);
                const float xValue = xTile.GetValue(rowIndex);
                SyncSToV();
                Axpy(acc, row, alpha_ * xValue, colsAligned);
            }
            SyncVToMte3();
            StoreYContiguous(colStart, cols, acc);
            FinishYChunk();
        }
    }

    __aicore__ inline void AccumulateSmallMRows(
        uint32_t colsAligned, LocalTensor<float> aTile, LocalTensor<float> xTile, LocalTensor<float> partial)
    {
        // partial layout: Kahan sum area [0, m), compensation area [SGEMMV_SMALL_M_LIMIT, SGEMMV_SMALL_M_LIMIT + m).
        // The areas must not overlap: m <= SGEMMV_SMALL_M_LIMIT keeps compensation beyond the sum area, so
        // multi-tile accumulation never reads a partial sum as a compensation term.
        LocalTensor<float> row = rowBuf_.Get<float>();
        LocalTensor<float> product = productBuf_.Get<float>();
        LocalTensor<float> reduceTmp = workBuf_.Get<float>();
        LocalTensor<float> reduceOut = reduceBuf_.Get<float>();
        LocalTensor<uint32_t> offsets = offsetBuf_.Get<uint32_t>();
        for (uint32_t rowIndex = 0; rowIndex < m_; ++rowIndex) {
            Gather(row, aTile, offsets, rowIndex * SGEMMV_BYTES_PER_FLOAT, colsAligned);
            Mul(product, row, xTile, colsAligned);
            ReduceSum(reduceOut, product, reduceTmp, static_cast<int32_t>(colsAligned));
            SyncVToS();
            const uint32_t compensationOffset = SGEMMV_SMALL_M_LIMIT + rowIndex;
            const float corrected = reduceOut.GetValue(0) - partial.GetValue(compensationOffset);
            const float previous = partial.GetValue(rowIndex);
            const float current = previous + corrected;
            partial.SetValue(compensationOffset, (current - previous) - corrected);
            partial.SetValue(rowIndex, current);
        }
    }

    __aicore__ inline void ProcessSmallMCompute(uint32_t block)
    {
        const uint32_t colBegin = block * outPerBlock_;
        const uint32_t colEnd = (colBegin + outPerBlock_ < n_) ? (colBegin + outPerBlock_) : n_;
        if (colBegin >= colEnd) {
            return;
        }

        LocalTensor<float> aTile = aBuf_.Get<float>();
        LocalTensor<float> xTile = xBuf_.Get<float>();
        LocalTensor<float> partial = outputBuf_.Get<float>();

        const bool flatA = lda_ == m_ && m_ <= SGEMMV_SMALL_M_LIMIT;
        const uint32_t aStride = flatA ? m_ : AlignUp8(m_);
        uint32_t colTile = SGEMMV_A_FLOATS / aStride;
        if (colTile > SGEMMV_SMALL_N_COL_TILE) {
            colTile = SGEMMV_SMALL_N_COL_TILE;
        }
        colTile = colTile / SGEMMV_ALIGN * SGEMMV_ALIGN;
        BuildGatherOffsets(aStride);
        Duplicate(partial, 0.0f, SGEMMV_OUTPUT_FLOATS);
        SyncVToS();
        if (flatA) {
            Duplicate(aTile, 0.0f, SGEMMV_A_FLOATS);
            SyncVToMte2();
        }
        bool firstTile = true;
        for (uint32_t colStart = colBegin; colStart < colEnd; colStart += colTile) {
            const uint32_t cols = (colStart + colTile < colEnd) ? colTile : (colEnd - colStart);
            const uint32_t colsAligned = AlignUp8(cols);
            if (!firstTile) {
                SyncVToMte2();
            }
            if (flatA) {
                LoadAContiguous(colStart * lda_, m_ * cols, aTile);
            } else {
                LoadA2D(aGm_, static_cast<uint64_t>(colStart) * lda_, m_, cols, aStride, aTile);
            }
            LoadX(colStart, cols, xTile);
            firstTile = false;
            AccumulateSmallMRows(colsAligned, aTile, xTile, partial);
        }
        SyncSToV();
        Muls(partial, partial, alpha_, m_);
        SyncVToMte3();
        SetAtomicAdd<float>();
        DataCopyExtParams copy{1, m_ * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
        DataCopyPad(yGm_[0], partial, copy);
        SetAtomicNone();
    }

    TPipe* pipe_ = nullptr;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> aBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> productBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> outputBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;

    event_t eventMte2ToV_;
    event_t eventVToMte3_;
    event_t eventVToS_;
    event_t eventSToV_;
    event_t eventVToMte2_;
    event_t eventMte2ToS_;
    event_t eventMte3ToMte2_;
    event_t eventMte3ToV_;

    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t xLength_ = 0;
    uint32_t yLength_ = 0;
    uint32_t lda_ = 0;
    uint32_t outPerBlock_ = 0;
    uint32_t path_ = 0;
    uint32_t absIncX_ = 1;
    uint32_t absIncY_ = 1;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int64_t incx_ = 1;
    int64_t incy_ = 1;
};

extern "C" __global__ __aicore__ void sgemv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemvKernel op;
    op.Init(a, x, y, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void sgemv_init_kernel(GM_ADDR y, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemvKernel op;
    op.Init(nullptr, nullptr, y, tiling, &pipe);
    op.InitOutput();
}

extern "C" __global__ __aicore__ void sgemv_small_m_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemvKernel op;
    op.Init(a, x, y, tiling, &pipe);
    op.ProcessSmallMN(static_cast<uint32_t>(GetBlockIdx()));
}

extern "C" __global__ __aicore__ __cube__ void sgemv_cube_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);

    const bool isNPath =
        (tiling.path == static_cast<uint32_t>(SgemvPath::NORMAL_N) ||
         tiling.path == static_cast<uint32_t>(SgemvPath::SMALL_M_N));
    const uint32_t outDim = isNPath ? tiling.m : tiling.n;
    const uint32_t inDim = isNPath ? tiling.n : tiling.m;
    const uint32_t splitK = tiling.cubeSplitK == 0U ? 1U : tiling.cubeSplitK;
    const uint32_t outBlockNum = static_cast<uint32_t>(GetBlockNum()) / splitK;
    const uint32_t task = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t outBlock = task / splitK;
    const uint32_t kSlice = task % splitK;
    const uint32_t colsPerBlock =
        ((outDim + outBlockNum - 1U) / outBlockNum + SGEMMV_ALIGN - 1U) / SGEMMV_ALIGN * SGEMMV_ALIGN;
    const uint32_t colBase = outBlock * colsPerBlock;
    if (colBase >= outDim) {
        return;
    }
    const uint32_t cols = (colBase + colsPerBlock < outDim) ? colsPerBlock : (outDim - colBase);
    const uint32_t kStart = kSlice * tiling.cubeKPerSlice;
    if (kStart >= inDim) {
        return;
    }
    const uint32_t kRemain = inDim - kStart;
    const uint32_t kLen = (kRemain < tiling.cubeKPerSlice) ? kRemain : tiling.cubeKPerSlice;

    GlobalTensor<float> left;
    GlobalTensor<float> right;
    GlobalTensor<float> output;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x) + kStart, kLen);
    const uint64_t aOffset = isNPath ? static_cast<uint64_t>(kStart) * tiling.lda + colBase :
                                       static_cast<uint64_t>(colBase) * tiling.lda + kStart;
    const uint64_t rightLength = isNPath ? static_cast<uint64_t>(kLen - 1U) * tiling.lda + cols :
                                           static_cast<uint64_t>(cols - 1U) * tiling.lda + kLen;
    right.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a) + aOffset, rightLength);
    const uint64_t outputOffset = static_cast<uint64_t>(splitK == 1U ? 0U : kSlice) * outDim + colBase;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y) + outputOffset, outDim - colBase);

    if (isNPath && outDim >= 3840U && outDim % SGEMMV_CUBE_MAX_COLS != 0U) {
        ComputeCubeTile<SgemvOddCubeMatmul>(left, right, output, outDim, tiling.lda, cols, kLen);
    } else {
        ComputeCubeTile<SgemvCubeMatmul>(left, right, output, outDim, tiling.lda, cols, kLen);
    }
}

// Reduce the split-K partial sums of this block's column range and store the block result.
// colsPerBlock may exceed SGEMMV_SPLIT_CHUNK when fewer cores than CeilDiv(outDim, SGEMMV_SPLIT_CHUNK)
// are available, so the block's column range is reduced in fixed-size chunks to keep every slice
// write within its own SGEMMV_SPLIT_CHUNK slot of the buffer (acc aliases slot 0 of base).
__aicore__ inline void SplitReduceChunks(
    LocalTensor<float> base, GlobalTensor<float> partial, GlobalTensor<float> y, uint32_t outDim, uint32_t colBase,
    uint32_t cols, uint32_t splitK)
{
    LocalTensor<float> acc = base;
    for (uint32_t chunkStart = 0; chunkStart < cols; chunkStart += SGEMMV_SPLIT_CHUNK) {
        const uint32_t chunkLen = (chunkStart + SGEMMV_SPLIT_CHUNK < cols) ? SGEMMV_SPLIT_CHUNK : (cols - chunkStart);
        const uint32_t chunkAligned = (chunkLen + SGEMMV_ALIGN - 1U) / SGEMMV_ALIGN * SGEMMV_ALIGN;
        const uint8_t padding = static_cast<uint8_t>(chunkAligned - chunkLen);
        for (uint32_t offset = 0; offset < splitK; ++offset) {
            DataCopyPadExtParams<float> pad{true, 0, padding, 0.0f};
            DataCopyExtParams copy{1, chunkLen * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(
                base[offset * SGEMMV_SPLIT_CHUNK],
                partial[static_cast<uint64_t>(offset) * outDim + colBase + chunkStart], copy, pad);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t offset = 1; offset < splitK; ++offset) {
            Add(acc, acc, base[offset * SGEMMV_SPLIT_CHUNK], chunkAligned);
        }
        // splitBuf slots are reused across chunks: the next chunk's MTE2 loads must wait
        // for this chunk's Add reads (V) and y write (MTE3) to complete, matching the
        // cross-iteration discipline used by ProcessNormalT/ProcessSmallMT.
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
        DataCopyExtParams copy{1, chunkLen * SGEMMV_BYTES_PER_FLOAT, 0, 0, 0};
        DataCopyPad(y[colBase + chunkStart], acc, copy);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
    }
}

extern "C" __global__ __aicore__ void sgemv_split_reduce_kernel(GM_ADDR partialGm, GM_ADDR yGm, SgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.cubeSplitK <= 1U) {
        return;
    }

    const bool isNPath =
        (tiling.path == static_cast<uint32_t>(SgemvPath::NORMAL_N) ||
         tiling.path == static_cast<uint32_t>(SgemvPath::SMALL_M_N));
    const uint32_t outDim = isNPath ? tiling.m : tiling.n;
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    const uint32_t colsPerBlock =
        ((outDim + blockNum - 1U) / blockNum + SGEMMV_ALIGN - 1U) / SGEMMV_ALIGN * SGEMMV_ALIGN;
    const uint32_t colBase = static_cast<uint32_t>(GetBlockIdx()) * colsPerBlock;
    if (colBase >= outDim) {
        return;
    }
    const uint32_t cols = (colBase + colsPerBlock < outDim) ? colsPerBlock : (outDim - colBase);

    TPipe pipe;
    TBuf<TPosition::VECCALC> splitBuf;
    pipe.InitBuffer(splitBuf, SGEMMV_SPLIT_MAX * SGEMMV_SPLIT_CHUNK * SGEMMV_BYTES_PER_FLOAT);
    LocalTensor<float> base = splitBuf.Get<float>();
    GlobalTensor<float> partial;
    GlobalTensor<float> y;
    partial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partialGm));
    y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yGm));
    SplitReduceChunks(base, partial, y, outDim, colBase, cols, tiling.cubeSplitK);
}

void sgemv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sgemv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
}

void sgemv_init_kernel_do(uint8_t* y, const SgemvTilingData& tiling, void* stream)
{
    sgemv_init_kernel<<<1, nullptr, stream>>>(y, tiling);
}

void sgemv_small_m_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sgemv_small_m_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
}

void sgemv_cube_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, void* stream)
{
    sgemv_cube_kernel<<<tiling.numBlocks, nullptr, stream>>>(a, x, y, tiling);
}

void sgemv_split_reduce_kernel_do(
    uint8_t* partial, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    sgemv_split_reduce_kernel<<<numBlocks, nullptr, stream>>>(partial, y, tiling);
}
