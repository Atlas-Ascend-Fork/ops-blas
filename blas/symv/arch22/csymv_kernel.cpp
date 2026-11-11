/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <type_traits>

#include "kernel_operator.h"
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "csymv_tiling_data.h"

template <AscendC::HardEvent event>
__aicore__ inline void CsymvSync()
{
    AscendC::SetFlag<event>(0);
    AscendC::WaitFlag<event>(0);
}

__aicore__ inline CsymvTilingData CsymvResolveScalars(const CsymvTilingData& tiling, GM_ADDR alpha, GM_ADDR beta)
{
    CsymvTilingData value = tiling;
    if (value.alphaIsDevice) {
        AscendC::GlobalTensor<float> scalar;
        scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
        value.alphaReal = scalar.GetValue(0);
        value.alphaImag = scalar.GetValue(1);
    }
    if (value.betaIsDevice) {
        AscendC::GlobalTensor<float> scalar;
        scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        value.betaReal = scalar.GetValue(0);
        value.betaImag = scalar.GetValue(1);
    }
    return value;
}

__aicore__ inline void CsymvStoreScalar(
    const CsymvTilingData& tiling, AscendC::GlobalTensor<float>& y, AscendC::LocalTensor<float> output, uint32_t row,
    float real, float imag, uint32_t slot)
{
    const int64_t logical = tiling.incy > 0 ? static_cast<int64_t>(row) * tiling.incy :
                                              static_cast<int64_t>(tiling.n - 1 - row) * (-tiling.incy);
    const bool betaZero = tiling.betaReal == 0.0f && tiling.betaImag == 0.0f;
    const bool betaOne = tiling.betaReal == 1.0f && tiling.betaImag == 0.0f;
    float yReal = 0.0f;
    float yImag = 0.0f;
    // beta == 0 skips old y; beta == 1 preserves Inf without evaluating 0 * Inf.
    if (!betaZero) {
        yReal = y.GetValue(logical * 2);
        yImag = y.GetValue(logical * 2 + 1);
    }
    const float betaContributionReal =
        betaOne ? yReal : (betaZero ? 0.0f : tiling.betaReal * yReal - tiling.betaImag * yImag);
    const float betaContributionImag =
        betaOne ? yImag : (betaZero ? 0.0f : tiling.betaReal * yImag + tiling.betaImag * yReal);
    const bool alphaZero = tiling.alphaReal == 0.0f && tiling.alphaImag == 0.0f;
    const uint32_t offset = tiling.incy == 1 ? slot * 2U : 0U;
    output.SetValue(
        offset, (alphaZero ? 0.0f : tiling.alphaReal * real - tiling.alphaImag * imag) + betaContributionReal);
    output.SetValue(
        offset + 1U, (alphaZero ? 0.0f : tiling.alphaReal * imag + tiling.alphaImag * real) + betaContributionImag);
    if (tiling.incy != 1) {
        CsymvSync<AscendC::HardEvent::S_MTE3>();
        AscendC::DataCopyExtParams copy{1U, 8U, 0U, 0U, 0U};
        AscendC::DataCopyPad(y[logical * 2], output, copy);
        CsymvSync<AscendC::HardEvent::MTE3_S>();
    }
}

// Tensor views into each path's single UB allocation; unused views need not be bound.
struct CsymvPlanarBuffers {
    AscendC::GlobalTensor<float> a_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> y_;
    AscendC::LocalTensor<float> raw_; // Interleaved real/imag DMA staging.
    AscendC::LocalTensor<float> matrixReal_;
    AscendC::LocalTensor<float> matrixImag_;
    AscendC::LocalTensor<float> vectorReal_; // Owner reuses this view for a product temporary.
    AscendC::LocalTensor<float> vectorImag_;
    AscendC::LocalTensor<float> scratch_;
    AscendC::LocalTensor<float> productReal_;
    AscendC::LocalTensor<float> productImag_;
    AscendC::LocalTensor<float> sums_;
    AscendC::LocalTensor<float> output_;
};

// General path for strided x and sizes outside the owner bounds.
class CsymvAiv {
    static constexpr uint32_t CHUNK = 256U;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
    CsymvPlanarBuffers buffers_;
    AscendC::LocalTensor<uint32_t> offsets_;
    CsymvTilingData tiling_{};

    __aicore__ inline int64_t VectorIndex(uint32_t i, int64_t inc) const
    {
        return inc > 0 ? static_cast<int64_t>(i) * inc : (static_cast<int64_t>(tiling_.n) - 1 - i) * (-inc);
    }

    __aicore__ inline void Load(
        AscendC::GlobalTensor<float> source, int64_t start, int64_t stride, uint32_t count,
        AscendC::LocalTensor<float> realPlane, AscendC::LocalTensor<float> imagPlane)
    {
        // A strided complex DMA occupies one 32-byte UB block per element.
        // Negative vector strides are gathered in logical order using scalar loads.
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        CsymvSync<AscendC::HardEvent::V_S>();
        if (stride > 0 && static_cast<uint64_t>(stride - 1) * 8U <= 0xffffffffULL) {
            const uint32_t spacing = stride == 1 ? 2U : 8U;
            AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(stride == 1 ? 1U : count), stride == 1 ? count * 8U : 8U,
                stride == 1 ? 0U : static_cast<uint32_t>((stride - 1) * 8), 0U, 0U};
            AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(buffers_.raw_, source[start * 2], copy, pad);
            const uint32_t gatherCount = (count + 7U) / 8U * 8U;
            for (uint32_t i = 0; i < gatherCount; ++i) {
                offsets_.SetValue(i, i < count ? i * spacing * sizeof(float) : 0U);
            }
            CsymvSync<AscendC::HardEvent::MTE2_V>();
            CsymvSync<AscendC::HardEvent::S_V>();
            AscendC::Gather(realPlane, buffers_.raw_, offsets_, 0U, gatherCount);
            AscendC::Gather(imagPlane, buffers_.raw_, offsets_, sizeof(float), gatherCount);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                const int64_t pos = 2 * (start + static_cast<int64_t>(i) * stride);
                realPlane.SetValue(i, source.GetValue(pos));
                imagPlane.SetValue(i, source.GetValue(pos + 1));
            }
            CsymvSync<AscendC::HardEvent::S_V>();
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(
        uint32_t row, uint32_t begin, uint32_t end, bool direct, float& sumReal, float& sumImag)
    {
        for (uint32_t col = begin; col < end;) {
            const uint32_t count = end - col < CHUNK ? end - col : CHUNK;
            const int64_t aIndex =
                direct ? row + static_cast<int64_t>(col) * tiling_.lda : col + static_cast<int64_t>(row) * tiling_.lda;
            Load(buffers_.a_, aIndex, direct ? tiling_.lda : 1, count, buffers_.matrixReal_, buffers_.matrixImag_);
            Load(
                buffers_.x_, VectorIndex(col, tiling_.incx), tiling_.incx, count, buffers_.vectorReal_,
                buffers_.vectorImag_);
            AscendC::Mul(buffers_.productReal_, buffers_.matrixReal_, buffers_.vectorReal_, count);
            AscendC::Mul(buffers_.scratch_, buffers_.matrixImag_, buffers_.vectorImag_, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(buffers_.productReal_, buffers_.productReal_, buffers_.scratch_, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(buffers_.productImag_, buffers_.matrixReal_, buffers_.vectorImag_, count);
            AscendC::Mul(buffers_.scratch_, buffers_.matrixImag_, buffers_.vectorReal_, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(buffers_.productImag_, buffers_.productImag_, buffers_.scratch_, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(buffers_.sums_, buffers_.productReal_, buffers_.scratch_, count);
            CsymvSync<AscendC::HardEvent::V_S>();
            sumReal += buffers_.sums_.GetValue(0);
            CsymvSync<AscendC::HardEvent::S_V>();
            AscendC::ReduceSum(buffers_.sums_, buffers_.productImag_, buffers_.scratch_, count);
            CsymvSync<AscendC::HardEvent::V_S>();
            sumImag += buffers_.sums_.GetValue(0);
            CsymvSync<AscendC::HardEvent::S_V>();
            col += count;
        }
    }

public:
    __aicore__ inline void Init(
        AscendC::TPipe& pipe, GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y,
        const CsymvTilingData& tiling)
    {
        tiling_ = CsymvResolveScalars(tiling, alpha, beta);
        buffers_.a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        buffers_.x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        buffers_.y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        // 8 floats/element for strided DMA, nine planar arrays, one offset array.
        pipe.InitBuffer(storage_, (8U + 9U + 1U) * CHUNK * sizeof(float));
        auto data = storage_.Get<float>();
        buffers_.raw_ = data;
        buffers_.matrixReal_ = data[8U * CHUNK];
        buffers_.matrixImag_ = data[9U * CHUNK];
        buffers_.vectorReal_ = data[10U * CHUNK];
        buffers_.vectorImag_ = data[11U * CHUNK];
        buffers_.scratch_ = data[12U * CHUNK];
        buffers_.productReal_ = data[13U * CHUNK];
        buffers_.productImag_ = data[14U * CHUNK];
        buffers_.sums_ = data[15U * CHUNK];
        buffers_.output_ = data[16U * CHUNK];
        offsets_ = data[17U * CHUNK].template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void Process()
    {
        const bool alphaZero = tiling_.alphaReal == 0.0f && tiling_.alphaImag == 0.0f;
        const bool betaOne = tiling_.betaReal == 1.0f && tiling_.betaImag == 0.0f;
        if (alphaZero && betaOne) {
            return;
        }
        for (uint32_t row = AscendC::GetBlockIdx(); row < tiling_.n; row += tiling_.useCoreNum) {
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            if (!alphaZero) {
                Accumulate(row, 0U, row, tiling_.uplo == ACLBLAS_LOWER, sumReal, sumImag);
                Accumulate(row, row, tiling_.n, tiling_.uplo == ACLBLAS_UPPER, sumReal, sumImag);
            }
            CsymvStoreScalar(tiling_, buffers_.y_, buffers_.output_, row, sumReal, sumImag, 0U);
            if (tiling_.incy == 1) {
                CsymvSync<AscendC::HardEvent::S_MTE3>();
                const AscendC::DataCopyExtParams copy{1U, 8U, 0U, 0U, 0U};
                AscendC::DataCopyPad(buffers_.y_[row * 2U], buffers_.output_, copy);
                CsymvSync<AscendC::HardEvent::MTE3_S>();
            }
        }
    }
};

// One output-owner path covers aligned and tail shapes without workspace.
template <bool PREFETCH = false>
class CsymvOwnerAiv {
    static constexpr uint32_t MAX_N = 8192U;
    static constexpr uint32_t OUTPUT_ROWS = 16U;
    static constexpr uint32_t CHUNK_COLS = 128U;
    static constexpr uint32_t PLANE_PITCH = CHUNK_COLS + 16U;
    static constexpr uint32_t RAW_PITCH = 2U * OUTPUT_ROWS + 8U;
    static constexpr uint32_t TILE_ELEMENTS = OUTPUT_ROWS * CHUNK_COLS;
    static constexpr uint32_t RAW_FLOATS = RAW_PITCH * CHUNK_COLS;
    static constexpr uint32_t RAW_SLOTS = PREFETCH ? 2U : 1U;
    // UB offsets count floats, not bytes. RAW_SLOTS preserves the prefetch allocation.
    static constexpr uint32_t MATRIX_REAL_OFFSET = RAW_SLOTS * RAW_FLOATS;
    static constexpr uint32_t MATRIX_IMAG_OFFSET = MATRIX_REAL_OFFSET + OUTPUT_ROWS * PLANE_PITCH;
    static constexpr uint32_t PRODUCT_OFFSET = MATRIX_IMAG_OFFSET + OUTPUT_ROWS * PLANE_PITCH;
    static constexpr uint32_t ACCUMULATOR_OFFSET = PRODUCT_OFFSET + 4U * TILE_ELEMENTS;
    static constexpr uint32_t X_CACHE_OFFSET = ACCUMULATOR_OFFSET + 2U * TILE_ELEMENTS;
    static constexpr uint32_t AUX_OFFSET = X_CACHE_OFFSET + 2U * MAX_N;
    static constexpr uint32_t UB_FLOATS = AUX_OFFSET + 8U * OUTPUT_ROWS + 16U;
    static_assert(UB_FLOATS * sizeof(float) < 184U * 1024U, "Owner UB overlaps API scratch");
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
    CsymvPlanarBuffers buffers_;
    AscendC::LocalTensor<float> acc_;
    AscendC::LocalTensor<float> cache_;
    AscendC::LocalTensor<uint8_t> masks_;
    AscendC::LocalTensor<uint32_t> offsets_;
    CsymvTilingData tiling_{};
    AscendC::LocalTensor<float> rawStorage_;
    uint32_t rawSlot_ = 0U;
    uint64_t transposeSrc_[RAW_SLOTS][OUTPUT_ROWS / 4U][16U];
    uint64_t transposeDst_[OUTPUT_ROWS / 4U][16U];
    uint32_t cacheStride_ = 0U;

    __aicore__ inline void PrepareConstants()
    {
        for (uint32_t group = 0U; group < OUTPUT_ROWS / 4U; ++group) {
            for (uint32_t item = 0U; item < 16U; ++item) {
                for (uint32_t slot = 0U; slot < RAW_SLOTS; ++slot) {
                    transposeSrc_[slot][group][item] = reinterpret_cast<uint64_t>(
                        rawStorage_[slot * RAW_FLOATS + item * RAW_PITCH + group * 8U].GetPhyAddr());
                }
                const uint32_t row = group * 4U + item / 4U;
                auto plane = (item / 2U) % 2U == 0U ? buffers_.matrixReal_ : buffers_.matrixImag_;
                transposeDst_[group][item] =
                    reinterpret_cast<uint64_t>(plane[row * PLANE_PITCH + (item % 2U) * 8U].GetPhyAddr());
            }
        }
        auto mask = masks_.ReinterpretCast<uint64_t>();
        const bool upper = tiling_.uplo == ACLBLAS_UPPER;
        mask.SetValue(0U, upper ? 0xfff8fffcfffeffffULL : 0x000f000700030001ULL);
        mask.SetValue(1U, upper ? 0xff80ffc0ffe0fff0ULL : 0x00ff007f003f001fULL);
        mask.SetValue(2U, upper ? 0xf800fc00fe00ff00ULL : 0x0fff07ff03ff01ffULL);
        mask.SetValue(3U, upper ? 0x8000c000e000f000ULL : 0xffff7fff3fff1fffULL);
        mask.SetValue(4U, upper ? 0x0007000300010000ULL : 0xfff0fff8fffcfffeULL);
        mask.SetValue(5U, upper ? 0x007f003f001f000fULL : 0xff00ff80ffc0ffe0ULL);
        mask.SetValue(6U, upper ? 0x07ff03ff01ff00ffULL : 0xf000f800fc00fe00ULL);
        mask.SetValue(7U, upper ? 0x7fff3fff1fff0fffULL : 0x00008000c000e000ULL);
        for (uint32_t i = 0U; i < 2U * OUTPUT_ROWS; ++i) {
            offsets_.SetValue(i, ((i % 2U) * OUTPUT_ROWS + i / 2U) * sizeof(float));
        }
        CsymvSync<AscendC::HardEvent::S_V>();
    }

    __aicore__ inline void PrepareX()
    {
        const uint32_t rounded = (tiling_.n + 63U) / 64U * 64U;
        AscendC::Duplicate(buffers_.raw_, 0.0f, 2U * rounded);
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        const AscendC::DataCopyExtParams copy{1U, tiling_.n * 8U, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        AscendC::DataCopyPad(buffers_.raw_, buffers_.x_, copy, pad);
        CsymvSync<AscendC::HardEvent::MTE2_V>();
        uint64_t reserved = 0U;
        for (uint32_t begin = 0U; begin < rounded; begin += 4096U) {
            const uint32_t count = rounded - begin < 4096U ? rounded - begin : 4096U;
            const uint16_t repeats = static_cast<uint16_t>(count / 32U);
            AscendC::GatherMask(
                cache_[begin], buffers_.raw_[2U * begin], 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
            AscendC::GatherMask(
                cache_[cacheStride_ + begin], buffers_.raw_[2U * begin], 2U, false, 0U, {1U, repeats, 8U, 8U},
                reserved);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadRaw(
        uint32_t row, uint32_t begin, uint32_t count, uint32_t rows, bool direct, uint32_t slot)
    {
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        const uint32_t width = (count + 15U) / 16U * 16U;
        const uint32_t blockFloats = direct ? rows * 2U : count * 2U;
        const uint32_t pitch = direct ? RAW_PITCH : 2U * width;
        const uint32_t rounded = (blockFloats + 7U) / 8U * 8U;
        const AscendC::DataCopyExtParams copy{
            static_cast<uint16_t>(direct ? count : rows), blockFloats * static_cast<uint32_t>(sizeof(float)),
            (tiling_.lda - (direct ? rows : count)) * 8U, (pitch - rounded) / 8U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        const uint64_t index = direct ? static_cast<uint64_t>(begin) * tiling_.lda + row :
                                        static_cast<uint64_t>(row) * tiling_.lda + begin;
        AscendC::DataCopyPad(rawStorage_[slot * RAW_FLOATS], buffers_.a_[index * 2U], copy, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void Deinterleave(uint32_t width, bool direct, bool diagonal)
    {
        if (direct) {
            AscendC::TransDataTo5HDParams p{};
            p.repeatTimes = static_cast<uint8_t>(width / 16U);
            p.srcRepStride = width == 16U ? 0U : 2U * RAW_PITCH;
            p.dstRepStride = width == 16U ? 0U : 2U;
            for (uint32_t group = 0U; group < OUTPUT_ROWS / 4U; ++group) {
                AscendC::TransDataTo5HD<float>(transposeDst_[group], transposeSrc_[rawSlot_][group], p);
            }
        } else {
            uint64_t reserved = 0U;
            if (diagonal) {
                for (uint32_t r = 0U; r < OUTPUT_ROWS; ++r) {
                    AscendC::GatherMask(
                        buffers_.matrixReal_[r * PLANE_PITCH], buffers_.raw_[r * RAW_PITCH], 1U, false, 0U,
                        {1U, 1U, 8U, 8U}, reserved);
                    AscendC::GatherMask(
                        buffers_.matrixImag_[r * PLANE_PITCH], buffers_.raw_[r * RAW_PITCH], 2U, false, 0U,
                        {1U, 1U, 8U, 8U}, reserved);
                }
            } else {
                const uint16_t repeats = static_cast<uint16_t>(OUTPUT_ROWS * width / 32U);
                AscendC::GatherMask(
                    buffers_.matrixReal_, buffers_.raw_, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
                AscendC::GatherMask(
                    buffers_.matrixImag_, buffers_.raw_, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Products(uint32_t begin, uint32_t width, uint32_t pitch)
    {
        const AscendC::BinaryRepeatParams p{
            1U, 1U, 1U, static_cast<uint8_t>(width / 8U), static_cast<uint8_t>(pitch / 8U), 0U};
        for (uint32_t col = 0U; col < width; col += 64U) {
            const uint32_t lanes = width - col < 64U ? width - col : 64U;
            AscendC::Mul<float>(
                buffers_.productReal_[col], buffers_.matrixReal_[col], cache_[begin + col], uint64_t{lanes},
                OUTPUT_ROWS, p);
            AscendC::Mul<float>(
                buffers_.scratch_[col], buffers_.matrixImag_[col], cache_[cacheStride_ + begin + col], uint64_t{lanes},
                OUTPUT_ROWS, p);
            AscendC::Mul<float>(
                buffers_.productImag_[col], buffers_.matrixReal_[col], cache_[cacheStride_ + begin + col],
                uint64_t{lanes}, OUTPUT_ROWS, p);
            AscendC::Mul<float>(
                buffers_.vectorReal_[col], buffers_.matrixImag_[col], cache_[begin + col], uint64_t{lanes}, OUTPUT_ROWS,
                p);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(buffers_.productReal_, buffers_.productReal_, buffers_.scratch_, OUTPUT_ROWS * width);
        AscendC::Add(buffers_.productImag_, buffers_.productImag_, buffers_.vectorReal_, OUTPUT_ROWS * width);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(uint32_t width, uint32_t valid)
    {
        const AscendC::BinaryRepeatParams p{
            1U, 1U, 1U, CHUNK_COLS / 8U, CHUNK_COLS / 8U, static_cast<uint8_t>(width / 8U)};
        for (uint32_t col = 0U; col < valid; col += 64U) {
            const uint32_t lanes = valid - col < 64U ? valid - col : 64U;
            AscendC::Add<float>(acc_[col], acc_[col], buffers_.productReal_[col], uint64_t{lanes}, OUTPUT_ROWS, p);
            AscendC::Add<float>(
                acc_[TILE_ELEMENTS + col], acc_[TILE_ELEMENTS + col], buffers_.productImag_[col], uint64_t{lanes},
                OUTPUT_ROWS, p);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Segment(uint32_t row, uint32_t begin, uint32_t end, uint32_t rows, bool direct)
    {
        for (uint32_t col = begin; col < end; col += CHUNK_COLS) {
            const uint32_t count = end - col < CHUNK_COLS ? end - col : CHUNK_COLS;
            const uint32_t width = (count + 15U) / 16U * 16U;
            LoadRaw(row, col, count, rows, direct, 0U);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Deinterleave(width, direct, false);
            Products(col, width, direct ? PLANE_PITCH : width);
            Accumulate(width, count);
        }
    }

    __aicore__ inline void Diagonal(uint32_t row, uint32_t rows)
    {
        LoadRaw(row, row, rows, rows, true, 0U);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        DiagonalViews(row, rows);
    }

    __aicore__ inline void DiagonalViews(uint32_t row, uint32_t rows)
    {
        for (uint32_t view = 0U; view < 2U; ++view) {
            Deinterleave(OUTPUT_ROWS, view == 0U, true);
            Products(row, OUTPUT_ROWS, PLANE_PITCH);
            AscendC::Select(
                buffers_.productReal_, masks_[view * 32U], buffers_.productReal_, 0.0f,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, OUTPUT_ROWS * OUTPUT_ROWS);
            AscendC::Select(
                buffers_.productImag_, masks_[view * 32U], buffers_.productImag_, 0.0f,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, OUTPUT_ROWS * OUTPUT_ROWS);
            AscendC::PipeBarrier<PIPE_V>();
            Accumulate(OUTPUT_ROWS, rows);
        }
    }

    struct SegmentInfo {
        uint32_t begin;
        uint32_t count;
        bool direct;
        bool diagonal;
    };

    __aicore__ inline SegmentInfo RowSegment(uint32_t index, uint32_t row, uint32_t rows) const
    {
        const uint32_t before = (row + CHUNK_COLS - 1U) / CHUNK_COLS;
        if (index < before) {
            const uint32_t begin = index * CHUNK_COLS;
            return {begin, row - begin < CHUNK_COLS ? row - begin : CHUNK_COLS, tiling_.uplo == ACLBLAS_LOWER, false};
        }
        if (index == before) {
            return {row, rows, true, true};
        }
        const uint32_t begin = row + rows + (index - before - 1U) * CHUNK_COLS;
        return {
            begin, tiling_.n - begin < CHUNK_COLS ? tiling_.n - begin : CHUNK_COLS, tiling_.uplo == ACLBLAS_UPPER,
            false};
    }

    __aicore__ inline void PrefetchRow(uint32_t row, uint32_t rows)
    {
        const uint32_t count =
            (row + CHUNK_COLS - 1U) / CHUNK_COLS + 1U + (tiling_.n - row - rows + CHUNK_COLS - 1U) / CHUNK_COLS;
        auto current = RowSegment(0U, row, rows);
        rawSlot_ = 0U;
        LoadRaw(row, current.begin, current.count, rows, current.direct, rawSlot_);
        for (uint32_t index = 0U; index < count; ++index) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            buffers_.raw_ = rawStorage_[rawSlot_ * RAW_FLOATS];
            SegmentInfo next{};
            if (index + 1U < count) {
                next = RowSegment(index + 1U, row, rows);
                LoadRaw(row, next.begin, next.count, rows, next.direct, 1U - rawSlot_);
            }
            if (current.diagonal) {
                // Both views consume this raw slot before it can be reused.
                DiagonalViews(row, rows);
            } else {
                const uint32_t width = (current.count + 15U) / 16U * 16U;
                Deinterleave(width, current.direct, false);
                Products(current.begin, width, current.direct ? PLANE_PITCH : width);
                Accumulate(width, current.count);
            }
            current = next;
            rawSlot_ = 1U - rawSlot_;
        }
        rawSlot_ = 0U;
        buffers_.raw_ = rawStorage_;
    }

    __aicore__ inline void Reduce()
    {
        AscendC::BlockReduceSum<float>(buffers_.matrixReal_, acc_, OUTPUT_ROWS * CHUNK_COLS / 64U, 64U, 1U, 1U, 8U);
        AscendC::BlockReduceSum<float>(
            buffers_.matrixImag_, acc_[TILE_ELEMENTS], OUTPUT_ROWS * CHUNK_COLS / 64U, 64U, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float>(
            buffers_.productReal_, buffers_.matrixReal_, OUTPUT_ROWS * CHUNK_COLS / 512U, 64U, 1U, 1U, 8U);
        AscendC::BlockReduceSum<float>(
            buffers_.productImag_, buffers_.matrixImag_, OUTPUT_ROWS * CHUNK_COLS / 512U, 64U, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum<float>(buffers_.sums_, buffers_.productReal_, 1U, 2U * OUTPUT_ROWS, 1U, 1U, 8U);
        AscendC::PairReduceSum<float>(
            buffers_.sums_[OUTPUT_ROWS], buffers_.productImag_, 1U, 2U * OUTPUT_ROWS, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Store(uint32_t row, uint32_t rows, bool alphaZero)
    {
        if (tiling_.incy == 1 && tiling_.betaReal == 0.0f && tiling_.betaImag == 0.0f) {
            if (alphaZero) {
                AscendC::Duplicate(buffers_.output_, 0.0f, 2U * OUTPUT_ROWS);
            } else {
                AscendC::Muls(buffers_.productReal_, buffers_.sums_, tiling_.alphaReal, 2U * OUTPUT_ROWS);
                AscendC::Muls(buffers_.scratch_, buffers_.sums_, tiling_.alphaImag, 2U * OUTPUT_ROWS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(buffers_.productReal_, buffers_.productReal_, buffers_.scratch_[OUTPUT_ROWS], OUTPUT_ROWS);
                AscendC::Add(
                    buffers_.productReal_[OUTPUT_ROWS], buffers_.productReal_[OUTPUT_ROWS], buffers_.scratch_,
                    OUTPUT_ROWS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Adds(buffers_.productReal_, buffers_.productReal_, 0.0f, 2U * OUTPUT_ROWS);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(buffers_.output_, buffers_.productReal_, offsets_, 0U, 2U * rows);
            }
            CsymvSync<AscendC::HardEvent::V_MTE3>();
        } else {
            CsymvSync<AscendC::HardEvent::V_S>();
            for (uint32_t r = 0U; r < rows; ++r) {
                CsymvStoreScalar(
                    tiling_, buffers_.y_, buffers_.output_, row + r, alphaZero ? 0.0f : buffers_.sums_.GetValue(r),
                    alphaZero ? 0.0f : buffers_.sums_.GetValue(OUTPUT_ROWS + r), r);
            }
            CsymvSync<AscendC::HardEvent::S_MTE3>();
        }
        if (tiling_.incy == 1) {
            const AscendC::DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
            AscendC::DataCopyPad(buffers_.y_[row * 2U], buffers_.output_, copy);
            CsymvSync<AscendC::HardEvent::MTE3_V>();
        }
    }

public:
    __aicore__ inline void Init(
        AscendC::TPipe& pipe, GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y,
        const CsymvTilingData& tiling)
    {
        pipe.InitBuffer(storage_, UB_FLOATS * sizeof(float));
        InitFromStorage(storage_.Get<float>(), alpha, a, x, beta, y, tiling);
    }

    __aicore__ inline void InitFromStorage(
        AscendC::LocalTensor<float> data, GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y,
        const CsymvTilingData& tiling)
    {
        tiling_ = CsymvResolveScalars(tiling, alpha, beta);
        cacheStride_ = (tiling_.n + 63U) / 64U * 64U;
        buffers_.a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        buffers_.x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        buffers_.y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        rawStorage_ = data;
        buffers_.raw_ = data;
        buffers_.matrixReal_ = data[MATRIX_REAL_OFFSET];
        buffers_.matrixImag_ = data[MATRIX_IMAG_OFFSET];
        buffers_.productReal_ = data[PRODUCT_OFFSET];
        buffers_.productImag_ = data[PRODUCT_OFFSET + TILE_ELEMENTS];
        buffers_.scratch_ = data[PRODUCT_OFFSET + 2U * TILE_ELEMENTS];
        buffers_.vectorReal_ = data[PRODUCT_OFFSET + 3U * TILE_ELEMENTS];
        acc_ = data[ACCUMULATOR_OFFSET];
        cache_ = data[X_CACHE_OFFSET];
        buffers_.sums_ = data[AUX_OFFSET];
        buffers_.output_ = data[AUX_OFFSET + 2U * OUTPUT_ROWS];
        offsets_ = data[AUX_OFFSET + 4U * OUTPUT_ROWS].template ReinterpretCast<uint32_t>();
        masks_ = data[AUX_OFFSET + 6U * OUTPUT_ROWS].template ReinterpretCast<uint8_t>();
    }

    __aicore__ inline void AccumulateRow(uint32_t row, uint32_t rows)
    {
        AscendC::Duplicate(acc_, 0.0f, 2U * TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (PREFETCH) {
            PrefetchRow(row, rows);
        } else {
            Segment(row, 0U, row, rows, tiling_.uplo == ACLBLAS_LOWER);
            Diagonal(row, rows);
            Segment(row, row + rows, tiling_.n, rows, tiling_.uplo == ACLBLAS_UPPER);
        }
        Reduce();
    }

    __aicore__ inline void ReplayRows(uint32_t begin, uint32_t end)
    {
        PrepareConstants();
        PrepareX();
        for (uint32_t row = begin; row < end && row < tiling_.n; row += OUTPUT_ROWS) {
            const uint32_t rows = tiling_.n - row < OUTPUT_ROWS ? tiling_.n - row : OUTPUT_ROWS;
            AccumulateRow(row, rows);
            Store(row, rows, false);
        }
    }

    __aicore__ inline void Process()
    {
        const bool alphaZero = tiling_.alphaReal == 0.0f && tiling_.alphaImag == 0.0f;
        if (alphaZero && tiling_.betaReal == 1.0f && tiling_.betaImag == 0.0f) {
            return;
        }
        if (!alphaZero) {
            PrepareConstants();
            PrepareX();
        }
        for (uint32_t row = AscendC::GetBlockIdx() * OUTPUT_ROWS; row < tiling_.n;
             row += tiling_.useCoreNum * OUTPUT_ROWS) {
            const uint32_t rows = tiling_.n - row < OUTPUT_ROWS ? tiling_.n - row : OUTPUT_ROWS;
            if (!alphaZero) {
                AccumulateRow(row, rows);
            }
            Store(row, rows, alphaZero);
        }
    }
};
// T64/T128 producer with two raw AoS matrix buffers. Each scalar multiply is rounded before
// PairReduceSum forms one complex product; reduction starts only afterwards.
// Source tiles are padded to the macro width; matrix DMA uses logical extents.
// Masking products excludes padding even for Inf/NaN inputs.
template <uint32_t TILE>
class CsymvTriangleAiv {
    static constexpr uint32_t MACRO_TILE = TILE;
    static constexpr uint32_t MICRO_TILE = 64U;
    static constexpr uint32_t MICRO_ELEMENTS = MICRO_TILE * MICRO_TILE;
    static constexpr uint32_t X_SLOTS = 4U;
    // UB offsets count floats. Keep the padding between live banks and API scratch.
    static constexpr uint32_t EXPANDED_REAL_OFFSET = 4U * MICRO_ELEMENTS + MICRO_TILE;
    static constexpr uint32_t EXPANDED_IMAG_OFFSET = EXPANDED_REAL_OFFSET + 2U * MICRO_ELEMENTS;
    static constexpr uint32_t PRODUCT_OFFSET = EXPANDED_IMAG_OFFSET + 2U * MICRO_ELEMENTS + MICRO_TILE;
    static constexpr uint32_t BROADCAST_OFFSET = PRODUCT_OFFSET + 2U * MICRO_ELEMENTS + MICRO_TILE;
    static constexpr uint32_t X_CACHE_OFFSET = BROADCAST_OFFSET + 2U * MACRO_TILE * 8U;
    static constexpr uint32_t MAX_UB_FLOATS = X_CACHE_OFFSET + X_SLOTS * 2U * MACRO_TILE + 3U * MACRO_TILE +
                                              4U * MACRO_TILE + 4U * MACRO_TILE + 2U * MICRO_ELEMENTS / 32U +
                                              2U * MACRO_TILE + 8U;
    static_assert(
        MAX_UB_FLOATS * sizeof(float) <= 184608U,
        "Recheck the AoS double-buffered lifetime layout when changing sizes");
    static_assert(
        MAX_UB_FLOATS * sizeof(float) <= 184U * 1024U, "The arch2201 Select temporary region must remain unused");
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
    AscendC::GlobalTensor<float> a_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> w_;
    AscendC::LocalTensor<float> raw_;
    AscendC::LocalTensor<float> expanded_;
    AscendC::LocalTensor<float> expandedImag_;
    AscendC::LocalTensor<float> real_;
    AscendC::LocalTensor<float> imag_;
    AscendC::LocalTensor<float> broadcast_;
    AscendC::LocalTensor<float> xv_;
    AscendC::LocalTensor<float> xaos_;
    AscendC::LocalTensor<float> mirror_;
    AscendC::LocalTensor<float> partial_;
    AscendC::LocalTensor<uint8_t> masks_;
    AscendC::LocalTensor<uint32_t> swap_;
    AscendC::LocalTensor<uint64_t> alt_;
    CsymvTilingData tiling_{};
    bool masksReady_ = false;
    bool pendingStore_ = false;
    uint32_t xCoordinates_[X_SLOTS] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t nextXSlot_ = 0U;
    uint32_t coefficientRow_ = UINT32_MAX;
    uint32_t coefficientCol_ = UINT32_MAX;

    __aicore__ inline AscendC::LocalTensor<float> XCoordinate(uint32_t coordinate)
    {
        for (uint32_t slot = 0U; slot < X_SLOTS; ++slot) {
            if (xCoordinates_[slot] == coordinate) {
                return xaos_[slot * 2U * MACRO_TILE];
            }
        }
        const uint32_t slot = nextXSlot_;
        nextXSlot_ = (nextXSlot_ + 1U) % X_SLOTS;
        auto data = xaos_[slot * 2U * MACRO_TILE];
        const uint32_t begin = coordinate * MACRO_TILE;
        const uint32_t count = tiling_.n - begin < MACRO_TILE ? tiling_.n - begin : MACRO_TILE;
        AscendC::Duplicate(data, 0.0f, 2U * MACRO_TILE);
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        const AscendC::DataCopyExtParams copy{1U, count * 8U, 0U, 0U, 0U};
        const AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        AscendC::DataCopyPad(data, x_[begin * 2U], copy, pad);
        // Matrix prefetch owns event 0; a short x load must not consume it.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
        xCoordinates_[slot] = coordinate;
        return data;
    }

    __aicore__ inline void PrepareOffsets()
    {
        // Four packed seeds and bounded doubling build the permanent swap offsets.
        auto seed = swap_.ReinterpretCast<uint64_t>();
        seed.SetValue(0U, 0x0000000000000004ULL);
        seed.SetValue(1U, 0x000000080000000cULL);
        seed.SetValue(2U, 0x0000001000000014ULL);
        seed.SetValue(3U, 0x000000180000001cULL);
        alt_.SetValue(0U, 0x5555555555555555ULL);
        CsymvSync<AscendC::HardEvent::S_V>();
        auto offset = swap_.ReinterpretCast<int32_t>();
        AscendC::Adds(offset[8U], offset, int32_t{32}, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(offset[16U], offset, int32_t{64}, 16U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(offset[32U], offset, int32_t{128}, 32U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(offset[64U], offset, int32_t{256}, 64U);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (MACRO_TILE == 128U) {
            AscendC::Adds(offset[128U], offset, int32_t{512}, 128U);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void EnsureMasks()
    {
        if (masksReady_) {
            return;
        }
        CsymvSync<AscendC::HardEvent::V_S>();
        auto seed = expanded_.ReinterpretCast<uint64_t>();
        seed.SetValue(0U, 0x3f80000000000000ULL);
        seed.SetValue(1U, 0x4040000040000000ULL);
        seed.SetValue(2U, 0x40a0000040800000ULL);
        seed.SetValue(3U, 0x40e0000040c00000ULL);
        CsymvSync<AscendC::HardEvent::S_V>();
        AscendC::Adds(expanded_[8U], expanded_, 8.0f, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(expanded_[16U], expanded_, 16.0f, 16U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(expanded_[32U], expanded_, 32.0f, 32U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(broadcast_, expanded_, 8U, {1U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams compare{1U, 1U, 0U, 8U, 0U, 1U};
        AscendC::Compare<float, uint8_t>(
            masks_, expanded_, broadcast_, tiling_.uplo == ACLBLAS_UPPER ? AscendC::CMPMODE::LE : AscendC::CMPMODE::GE,
            uint64_t{MICRO_TILE}, MICRO_TILE, compare);
        AscendC::Compare<float, uint8_t>(
            masks_[MICRO_ELEMENTS / 8U], expanded_, broadcast_,
            tiling_.uplo == ACLBLAS_UPPER ? AscendC::CMPMODE::LT : AscendC::CMPMODE::GT, uint64_t{MICRO_TILE},
            MICRO_TILE, compare);
        AscendC::PipeBarrier<PIPE_V>();
        masksReady_ = true;
        coefficientCol_ = UINT32_MAX;
    }

    __aicore__ inline void DirectCoefficients(uint32_t col)
    {
        const auto x = XCoordinate(col);
        uint64_t reserved = 0;
        AscendC::GatherMask(xv_, x, 1U, false, 0U, {1U, MACRO_TILE / 32U, 8U, 8U}, reserved);
        AscendC::GatherMask(xv_[MACRO_TILE], x, 2U, false, 0U, {1U, MACRO_TILE / 32U, 8U, 8U}, reserved);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(xv_[2U * MACRO_TILE], xv_[MACRO_TILE], -1.0f, MACRO_TILE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(broadcast_, xv_, MACRO_TILE / 8U, {1U, 8U});
        AscendC::Brcb(broadcast_[MACRO_TILE * 8U], xv_[MACRO_TILE], MACRO_TILE / 8U, {1U, 8U});
        AscendC::Brcb(expanded_, xv_[2U * MACRO_TILE], MACRO_TILE / 8U, {1U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams contiguous{1U, 1U, 1U, 8U, 8U, 8U};
        AscendC::Select(
            broadcast_[MACRO_TILE * 8U], alt_, broadcast_[MACRO_TILE * 8U], broadcast_,
            AscendC::SELMODE::VSEL_CMPMASK_SPR, uint64_t{MICRO_TILE}, MACRO_TILE / 8U, contiguous);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(
            broadcast_, alt_, broadcast_, expanded_, AscendC::SELMODE::VSEL_CMPMASK_SPR, uint64_t{MICRO_TILE},
            MACRO_TILE / 8U, contiguous);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MirrorCoefficients(uint32_t row)
    {
        const auto x = XCoordinate(row);
        AscendC::Muls(mirror_, x, 1.0f, 2U * MACRO_TILE);
        AscendC::Gather(mirror_[2U * MACRO_TILE], x, swap_, 0U, 2U * MACRO_TILE);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        AscendC::Muls<float>(mirror_, mirror_, -1.0f, odd, 2U * MACRO_TILE / MICRO_TILE, {1U, 1U, 8U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProductPlanes(
        AscendC::LocalTensor<float> cr, AscendC::LocalTensor<float> ci, AscendC::LocalTensor<float> matrix, bool mirror)
    {
        const AscendC::BinaryRepeatParams p = mirror ? AscendC::BinaryRepeatParams{1U, 1U, 1U, 16U, 16U, 0U} :
                                                       AscendC::BinaryRepeatParams{1U, 1U, 0U, 16U, 16U, 1U};
        AscendC::Mul<float, false>(expanded_, matrix, cr, uint64_t{MICRO_TILE}, MICRO_TILE, p);
        AscendC::Mul<float, false>(
            expanded_[MICRO_TILE], matrix[MICRO_TILE], cr[mirror ? MICRO_TILE : 0U], uint64_t{MICRO_TILE}, MICRO_TILE,
            p);
        AscendC::Mul<float, false>(expandedImag_, matrix, ci, uint64_t{MICRO_TILE}, MICRO_TILE, p);
        AscendC::Mul<float, false>(
            expandedImag_[MICRO_TILE], matrix[MICRO_TILE], ci[mirror ? MICRO_TILE : 0U], uint64_t{MICRO_TILE},
            MICRO_TILE, p);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum<float, false>(
            real_, expanded_, 2U * MICRO_ELEMENTS / MICRO_TILE, MICRO_TILE, 1U, 1U, 8U);
        AscendC::PairReduceSum<float, false>(
            imag_, expandedImag_, 2U * MICRO_ELEMENTS / MICRO_TILE, MICRO_TILE, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MaskPadding(uint32_t rows, uint32_t cols)
    {
        if (rows == MICRO_TILE && cols == MICRO_TILE) {
            return;
        }
        if (rows < MICRO_TILE) {
            const uint64_t valid = rows == 0U ? 0U : (uint64_t{1} << rows) - 1U;
            uint64_t mask[2] = {~valid, 0U};
            AscendC::Duplicate<float>(real_, 0.0f, mask, MICRO_TILE, 1U, MICRO_TILE / 8U);
            AscendC::Duplicate<float>(imag_, 0.0f, mask, MICRO_TILE, 1U, MICRO_TILE / 8U);
        }
        if (cols < MICRO_TILE) {
            AscendC::Duplicate(real_[cols * MICRO_TILE], 0.0f, (MICRO_TILE - cols) * MICRO_TILE);
            AscendC::Duplicate(imag_[cols * MICRO_TILE], 0.0f, (MICRO_TILE - cols) * MICRO_TILE);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool HAS_TAIL>
    __aicore__ inline void Products(
        AscendC::LocalTensor<float> matrix, uint32_t rowOffset, uint32_t colOffset, bool mirror, bool diagonal,
        uint32_t rows, uint32_t cols)
    {
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        ProductPlanes(
            mirror ? mirror_[2U * rowOffset] : broadcast_[colOffset * 8U],
            mirror ? mirror_[2U * MACRO_TILE + 2U * rowOffset] : broadcast_[MACRO_TILE * 8U + colOffset * 8U], matrix,
            mirror);
        if (diagonal) {
            const auto mask = masks_[mirror ? MICRO_ELEMENTS / 8U : 0U];
            AscendC::Select(real_, mask, real_, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, MICRO_ELEMENTS);
            AscendC::Select(imag_, mask, imag_, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, MICRO_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (HAS_TAIL) {
            MaskPadding(rows, cols);
        }
    }

    template <uint8_t REPEATS>
    __aicore__ inline void ReduceStage(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src)
    {
        const AscendC::BinaryRepeatParams pair{1U, 1U, 1U, 8U, 16U, 16U};
        AscendC::Add<float, false>(dst, src, src[MICRO_TILE], uint64_t{MICRO_TILE}, REPEATS, pair);
        AscendC::PipeBarrier<PIPE_V>();
    }

    template <bool HAS_TAIL>
    __aicore__ inline void Contribute(
        AscendC::LocalTensor<float> matrix, uint32_t rowOffset, uint32_t colOffset, bool sameBlock, bool diagonal,
        uint32_t rows, uint32_t cols)
    {
        Products<HAS_TAIL>(matrix, rowOffset, colOffset, false, diagonal, rows, cols);
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        ReduceStage<64U>(expanded_, real_);
        ReduceStage<32U>(real_, expanded_);
        ReduceStage<16U>(expanded_, real_);
        ReduceStage<8U>(real_, expanded_);
        ReduceStage<4U>(expanded_, real_);
        ReduceStage<2U>(real_, expanded_);
        const AscendC::BinaryRepeatParams accumulate{1U, 1U, 1U, MACRO_TILE / 8U, MACRO_TILE / 8U, MICRO_TILE / 8U};
        AscendC::Add<float, false>(
            partial_[rowOffset], partial_[rowOffset], real_, uint64_t{MICRO_TILE}, 2U, accumulate);
        AscendC::PipeBarrier<PIPE_V>();
        Products<HAS_TAIL>(matrix, rowOffset, colOffset, true, diagonal, rows, cols);
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        AscendC::BlockReduceSum<float, false>(
            expanded_, real_, 2U * MICRO_ELEMENTS / MICRO_TILE, MICRO_TILE, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        const auto reduced = expanded_[2U * MICRO_ELEMENTS / 8U];
        AscendC::BlockReduceSum<float, false>(reduced, expanded_, 2U * MICRO_ELEMENTS / 512U, MICRO_TILE, 1U, 1U, 8U);
        AscendC::PipeBarrier<PIPE_V>();
        const uint32_t dest = sameBlock ? colOffset : 2U * MACRO_TILE + colOffset;
        AscendC::Add<float, false>(partial_[dest], partial_[dest], reduced, uint64_t{MICRO_TILE}, 2U, accumulate);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Publish(uint32_t output, uint32_t source, uint32_t local)
    {
        const uint64_t offset = (static_cast<uint64_t>(output) * tiling_.sourceBlocks + source) * (2U * MACRO_TILE);
        AscendC::DataCopyExtParams copy{1U, 2U * MACRO_TILE * sizeof(float), 0U, 0U, 0U};
        AscendC::DataCopyPad(w_[offset], partial_[local], copy);
    }

    __aicore__ inline uint32_t ValidExtent(uint32_t begin) const
    {
        return begin >= tiling_.n ? 0U : (tiling_.n - begin < MICRO_TILE ? tiling_.n - begin : MICRO_TILE);
    }

    template <bool HAS_TAIL>
    __aicore__ inline void LoadMatrix(uint32_t row, uint32_t col, uint32_t slot)
    {
        const uint32_t rows = HAS_TAIL ? ValidExtent(row) : MICRO_TILE;
        const uint32_t cols = HAS_TAIL ? ValidExtent(col) : MICRO_TILE;
        if constexpr (HAS_TAIL) {
            if (rows != MICRO_TILE || cols != MICRO_TILE) {
                AscendC::Duplicate(raw_[slot * 2U * MICRO_ELEMENTS], 0.0f, 2U * MICRO_ELEMENTS);
                CsymvSync<AscendC::HardEvent::V_MTE2>();
            }
        }
        const uint32_t rounded = (rows * 2U + 7U) / 8U * 8U;
        const AscendC::DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0f};
        if (rows == MICRO_TILE && cols == MICRO_TILE && tiling_.lda % 4U != 0U) {
            // Group equal column-address phases while retaining the original
            // AoS destination, full64-column arithmetic and prefetch lifetime.
            const uint32_t period = tiling_.lda % 2U == 0U ? 2U : 4U;
            const AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(MICRO_TILE / period), MICRO_TILE * 8U, (period * tiling_.lda - MICRO_TILE) * 8U,
                (period - 1U) * 2U * MICRO_TILE / 8U, 0U};
            for (uint32_t phase = 0U; phase < period; ++phase) {
                AscendC::DataCopyPad(
                    raw_[slot * 2U * MICRO_ELEMENTS + phase * 2U * MICRO_TILE],
                    a_[(static_cast<uint64_t>(col + phase) * tiling_.lda + row) * 2U], copy, pad);
            }
        } else {
            const AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(cols), rows * 8U, (tiling_.lda - rows) * 8U, (2U * MICRO_TILE - rounded) / 8U,
                0U};
            AscendC::DataCopyPad(
                raw_[slot * 2U * MICRO_ELEMENTS], a_[(static_cast<uint64_t>(col) * tiling_.lda + row) * 2U], copy, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    template <bool HAS_TAIL>
    __aicore__ inline uint32_t NextMicro(uint32_t current, uint32_t row, uint32_t col) const
    {
        constexpr uint32_t chunks = MACRO_TILE / MICRO_TILE;
        for (uint32_t next = current + 1U; next < chunks * chunks; ++next) {
            const uint32_t rowOffset = next % chunks;
            const uint32_t colOffset = next / chunks;
            if (row == col && (tiling_.uplo == ACLBLAS_UPPER ? rowOffset > colOffset : rowOffset < colOffset)) {
                continue;
            }
            if constexpr (HAS_TAIL) {
                if (ValidExtent(row * MACRO_TILE + rowOffset * MICRO_TILE) == 0U ||
                    ValidExtent(col * MACRO_TILE + colOffset * MICRO_TILE) == 0U) {
                    continue;
                }
            }
            return next;
        }
        return chunks * chunks;
    }

    __aicore__ inline void PrepareCoefficients(uint32_t row, uint32_t col)
    {
        if (row == col) {
            EnsureMasks();
        }
        // Consecutive triangle tasks keep one macro coordinate fixed. Neither
        // product direction overwrites its coefficient bank, so keep it live
        // across those tasks instead of gathering and broadcasting x again.
        if (col != coefficientCol_) {
            DirectCoefficients(col);
            coefficientCol_ = col;
        }
        if (row != coefficientRow_) {
            MirrorCoefficients(row);
            coefficientRow_ = row;
        }
    }

    template <bool HAS_TAIL>
    __aicore__ inline void Tile(
        uint32_t row, uint32_t col, uint32_t nextRow, uint32_t nextCol, bool hasNext, bool prefetched, uint32_t& slot)
    {
        if (!prefetched) {
            CsymvSync<AscendC::HardEvent::V_MTE2>();
            LoadMatrix<HAS_TAIL>(row * MACRO_TILE, col * MACRO_TILE, slot);
        }
        PrepareCoefficients(row, col);
        // The previous output DMA overlaps this tile's independent coefficient
        // work. Only reusing partial_ requires its completion.
        if (pendingStore_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            pendingStore_ = false;
        }
        AscendC::Duplicate(partial_, 0.0f, 4U * MACRO_TILE);
        uint32_t micro = 0U;
        while (micro < (MACRO_TILE / MICRO_TILE) * (MACRO_TILE / MICRO_TILE)) {
            const uint32_t rowOffset = (micro % (MACRO_TILE / MICRO_TILE)) * MICRO_TILE;
            const uint32_t colOffset = (micro / (MACRO_TILE / MICRO_TILE)) * MICRO_TILE;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            const uint32_t next = NextMicro<HAS_TAIL>(micro, row, col);
            if (next < (MACRO_TILE / MICRO_TILE) * (MACRO_TILE / MICRO_TILE)) {
                // Both directions read the current raw AoS tile. A second raw
                // allocation lets the next DMA overlap their arithmetic.
                CsymvSync<AscendC::HardEvent::V_MTE2>();
                LoadMatrix<HAS_TAIL>(
                    row * MACRO_TILE + (next % (MACRO_TILE / MICRO_TILE)) * MICRO_TILE,
                    col * MACRO_TILE + (next / (MACRO_TILE / MICRO_TILE)) * MICRO_TILE, 1U - slot);
            } else if (hasNext) {
                // Keep the double buffer running across macro-tile boundaries.
                // The next macro's first DMA hides under the current last
                // micro's arithmetic, rather than restarting an empty pipe.
                CsymvSync<AscendC::HardEvent::V_MTE2>();
                LoadMatrix<true>(nextRow * MACRO_TILE, nextCol * MACRO_TILE, 1U - slot);
            }
            Contribute<HAS_TAIL>(
                raw_[slot * 2U * MICRO_ELEMENTS], rowOffset, colOffset, row == col,
                row == col && rowOffset == colOffset, HAS_TAIL ? ValidExtent(row * MACRO_TILE + rowOffset) : MICRO_TILE,
                HAS_TAIL ? ValidExtent(col * MACRO_TILE + colOffset) : MICRO_TILE);
            micro = next;
            slot = 1U - slot;
        }
        AscendC::ResetMask();
        CsymvSync<AscendC::HardEvent::V_MTE3>();
        Publish(row, col, 0U);
        if (row != col) {
            Publish(col, row, 2U * MACRO_TILE);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        pendingStore_ = true;
    }

    __aicore__ inline void DispatchTile(
        uint32_t row, uint32_t col, uint32_t nextRow, uint32_t nextCol, bool hasNext, bool prefetched, uint32_t& slot)
    {
        if (tiling_.n % MACRO_TILE != 0U && (row + 1U == tiling_.sourceBlocks || col + 1U == tiling_.sourceBlocks)) {
            Tile<true>(row, col, nextRow, nextCol, hasNext, prefetched, slot);
        } else {
            Tile<false>(row, col, nextRow, nextCol, hasNext, prefetched, slot);
        }
    }

    struct WorkRange {
        uint32_t diagonalBegin;
        uint32_t diagonalCount;
        uint32_t count;
        uint32_t offDiagonalBegin;
    };
    struct Coordinate {
        uint32_t row;
        uint32_t col;
    };

    // A diagonal macro has three micro tiles versus four off diagonal (T128).
    // Distribute diagonals first to bound per-core work, then enumerate the rest.
    __aicore__ inline WorkRange AssignWork() const
    {
        const uint32_t tasks = tiling_.sourceBlocks * (tiling_.sourceBlocks + 1U) / 2U;
        const uint32_t core = AscendC::GetBlockIdx();
        const uint32_t cores = tiling_.useCoreNum;
        const uint32_t base = tasks / cores;
        const uint32_t extra = tasks % cores;
        uint32_t limit = (4U * tasks - tiling_.sourceBlocks + cores - 1U) / cores;
        uint32_t highNeed;
        uint32_t lowNeed;
        uint32_t required;
        bool assigned;
        do {
            highNeed = 4U * (base + 1U) > limit ? 4U * (base + 1U) - limit : 0U;
            lowNeed = 4U * base > limit ? 4U * base - limit : 0U;
            required = extra * highNeed + (cores - extra) * lowNeed;
            assigned = required <= tiling_.sourceBlocks && (extra == 0U || highNeed <= base + 1U) && lowNeed <= base;
            if (!assigned) {
                ++limit;
            }
        } while (!assigned);
        const uint32_t highBefore = core < extra ? core : extra;
        const uint32_t lowBefore = core - highBefore;
        const uint32_t capacityBefore = (base + 1U - highNeed) * highBefore + (base - lowNeed) * lowBefore;
        const uint32_t remaining = tiling_.sourceBlocks - required;
        const uint32_t extraBefore = remaining < capacityBefore ? remaining : capacityBefore;
        const uint32_t leftHere = remaining - extraBefore;
        const uint32_t capacityHere = core < extra ? base + 1U - highNeed : base - lowNeed;
        const uint32_t extraHere = leftHere < capacityHere ? leftHere : capacityHere;
        const uint32_t diagonalBefore = highNeed * highBefore + lowNeed * lowBefore + extraBefore;
        const uint32_t diagonalCount = (core < extra ? highNeed : lowNeed) + extraHere;
        const uint32_t taskCount = base + (core < extra ? 1U : 0U);
        return {diagonalBefore, diagonalCount, taskCount, base * core + highBefore - diagonalBefore};
    }

    __aicore__ inline Coordinate NextCoordinate(
        const WorkRange& work, uint32_t index, uint32_t& row, uint32_t& col) const
    {
        if (index < work.diagonalCount) {
            const uint32_t diagonal = work.diagonalBegin + index;
            return {diagonal, diagonal};
        }
        const Coordinate result = tiling_.uplo == ACLBLAS_UPPER ? Coordinate{row, col} : Coordinate{col, row};
        ++col;
        if (col == tiling_.sourceBlocks) {
            ++row;
            col = row + 1U;
        }
        return result;
    }

public:
    __aicore__ inline AscendC::LocalTensor<float> GetFinalizeBuffer() { return raw_; }

    __aicore__ inline void Init(
        AscendC::TPipe& pipe, GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CsymvTilingData& tiling)
    {
        tiling_ = tiling;
        masksReady_ = false;
        pendingStore_ = false;
        coefficientRow_ = UINT32_MAX;
        coefficientCol_ = UINT32_MAX;
        if (tiling_.alphaIsDevice) {
            AscendC::GlobalTensor<float> scalar;
            scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
            tiling_.alphaReal = scalar.GetValue(0U);
            tiling_.alphaImag = scalar.GetValue(1U);
        }
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        const uint32_t xvOffset = X_CACHE_OFFSET + X_SLOTS * 2U * MACRO_TILE;
        const uint32_t mirrorOffset = xvOffset + 3U * MACRO_TILE;
        const uint32_t partialOffset = mirrorOffset + 4U * MACRO_TILE;
        const uint32_t masksOffset = partialOffset + 4U * MACRO_TILE;
        const uint32_t swapOffset = masksOffset + 2U * MICRO_ELEMENTS / 32U;
        const uint32_t altOffset = swapOffset + 2U * MACRO_TILE;
        pipe.InitBuffer(storage_, (altOffset + 8U) * sizeof(float));
        auto data = storage_.Get<float>();
        raw_ = data;
        expanded_ = data[EXPANDED_REAL_OFFSET];
        expandedImag_ = data[EXPANDED_IMAG_OFFSET];
        real_ = data[PRODUCT_OFFSET];
        imag_ = real_[MICRO_ELEMENTS];
        broadcast_ = data[BROADCAST_OFFSET];
        xaos_ = data[X_CACHE_OFFSET];
        xv_ = data[xvOffset];
        mirror_ = data[mirrorOffset];
        partial_ = data[partialOffset];
        masks_ = data[masksOffset].template ReinterpretCast<uint8_t>();
        swap_ = data[swapOffset].template ReinterpretCast<uint32_t>();
        alt_ = data[altOffset].template ReinterpretCast<uint64_t>();
    }

    __aicore__ inline void Process()
    {
        if (tiling_.alphaReal == 0.0f && tiling_.alphaImag == 0.0f) {
            return;
        }
        PrepareOffsets();
        const WorkRange work = AssignWork();
        if (work.count == 0U) {
            return;
        }
        uint32_t row = 0U;
        uint32_t residual = work.offDiagonalBegin;
        if (work.count > work.diagonalCount) {
            while (residual >= tiling_.sourceBlocks - row - 1U) {
                residual -= tiling_.sourceBlocks - row - 1U;
                ++row;
            }
        }
        uint32_t col = row + 1U + residual;
        uint32_t slot = 0U;
        Coordinate current = NextCoordinate(work, 0U, row, col);
        for (uint32_t index = 0U; index < work.count; ++index) {
            const bool hasNext = index + 1U < work.count;
            const Coordinate next = hasNext ? NextCoordinate(work, index + 1U, row, col) : current;
            DispatchTile(current.row, current.col, next.row, next.col, hasNext, index != 0U, slot);
            current = next;
        }
        if (pendingStore_) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            pendingStore_ = false;
        }
    }
};

// Fixed-q tree reduction. Each finalizer owns consecutive output groups and applies alpha/beta
// exactly once after the fused entry establishes producer completion with SyncAll.
template <uint32_t TILE>
class CsymvFinalizeAiv {
    static constexpr uint32_t SOURCE_TILE = TILE;
    static constexpr uint32_t ROWS_PER_GROUP = 32U;
    static constexpr uint32_t MAX_ROWS = 128U;
    uint32_t rows_ = ROWS_PER_GROUP;
    AscendC::GlobalTensor<float> w_;
    AscendC::GlobalTensor<float> y_;
    AscendC::LocalTensor<float> storage_;
    AscendC::LocalTensor<float> real_;
    AscendC::LocalTensor<float> imag_;
    AscendC::LocalTensor<float> out_;
    AscendC::LocalTensor<uint32_t> offsets_;
    CsymvTilingData tiling_{};
    uint32_t reductionBlocks_ = 1U;

    __aicore__ inline void StoreVector(uint32_t row, bool alphaZero)
    {
        if (alphaZero) {
            // alpha == beta == 0: do not read sums, workspace or old y.
            AscendC::Duplicate(out_, 0.0f, 2U * rows_);
        } else {
            // Both reduced planes stay live; the following four ranges
            // are dead after the q tree and hold the alpha products.
            auto alphaProductReal = real_[2U * rows_];
            auto alphaProductImag = real_[3U * rows_];
            auto alphaImagTimesReal = real_[4U * rows_];
            auto alphaImagTimesImag = real_[5U * rows_];
            // Pack both planes for each scalar multiply. alphaImagTimesReal now holds
            // alphaImag*real and alphaImagTimesImag holds alphaImag*imag; each FP32 multiply
            // and the following subtract/add keep their original rounding.
            AscendC::Muls(alphaProductReal, real_, tiling_.alphaReal, 2U * rows_);
            AscendC::Muls(alphaImagTimesReal, real_, tiling_.alphaImag, 2U * rows_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(alphaProductReal, alphaProductReal, alphaImagTimesImag, rows_);
            AscendC::Add(alphaProductImag, alphaProductImag, alphaImagTimesReal, rows_);
            AscendC::PipeBarrier<PIPE_V>();
            // Match Store's final + beta*y, including signed-zero semantics.
            // Do not simplify alpha == 1: 0 * Inf must still be evaluated.
            AscendC::Adds(alphaProductReal, alphaProductReal, 0.0f, 2U * rows_);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Gather(out_, alphaProductReal, offsets_, 0U, 2U * rows_);
        }
        CsymvSync<AscendC::HardEvent::V_MTE3>();
        const uint32_t rows = tiling_.n - row < rows_ ? tiling_.n - row : rows_;
        AscendC::DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
        AscendC::DataCopyPad(y_[row * 2U], out_, copy);
        CsymvSync<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void Store(uint32_t row, uint32_t slot, bool alphaZero)
    {
        const float real = alphaZero ? 0.0f : real_.GetValue(slot);
        const float imag = alphaZero ? 0.0f : imag_.GetValue(slot);
        CsymvStoreScalar(tiling_, y_, out_, row, real, imag, slot);
    }

    __aicore__ inline void InitState(
        GM_ADDR alpha, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, const CsymvTilingData& tiling)
    {
        tiling_ = tiling;
        reductionBlocks_ = 1U;
        while (reductionBlocks_ < tiling_.sourceBlocks) {
            reductionBlocks_ *= 2U;
        }
        if (tiling_.alphaIsDevice) {
            AscendC::GlobalTensor<float> scalar;
            scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha));
            tiling_.alphaReal = scalar.GetValue(0U);
            tiling_.alphaImag = scalar.GetValue(1U);
        }
        if (tiling_.betaIsDevice) {
            AscendC::GlobalTensor<float> scalar;
            scalar.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
            tiling_.betaReal = scalar.GetValue(0);
            tiling_.betaImag = scalar.GetValue(1);
        }
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    }

    __aicore__ inline void BindStorage(AscendC::LocalTensor<float> data)
    {
        const uint32_t inputFloats = reductionBlocks_ * 2U * rows_;
        real_ = data;
        imag_ = real_[rows_];
        out_ = real_[inputFloats];
        offsets_ = out_[2U * rows_].template ReinterpretCast<uint32_t>();
    }

    __aicore__ inline void InitOffsets()
    {
        auto words = offsets_.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0U; i < 4U; ++i) {
            words.SetValue(i, (static_cast<uint64_t>((rows_ + i) * sizeof(float)) << 32U) | (i * sizeof(float)));
        }
        CsymvSync<AscendC::HardEvent::S_V>();
        auto offsets = offsets_.ReinterpretCast<int32_t>();
        for (uint32_t initialized = 8U; initialized < 2U * rows_;) {
            const uint32_t count = 2U * rows_ - initialized < initialized ? 2U * rows_ - initialized : initialized;
            AscendC::Adds(offsets[initialized], offsets, static_cast<int32_t>(initialized * 2U), count);
            AscendC::PipeBarrier<PIPE_V>();
            initialized += count;
        }
    }

    __aicore__ inline void ReducePartials(uint32_t row, bool vectorize)
    {
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        CsymvSync<AscendC::HardEvent::S_MTE2>();
        // Stitch the same source planes across output-macro boundaries.
        // Each segment is a multiple of 32 rows, so both DMA pitches are aligned.
        for (uint32_t copied = 0U; copied < rows_;) {
            const uint32_t position = row + copied;
            const uint32_t within = position % SOURCE_TILE;
            const uint32_t segment = rows_ - copied < SOURCE_TILE - within ? rows_ - copied : SOURCE_TILE - within;
            const uint64_t offset =
                static_cast<uint64_t>(position / SOURCE_TILE) * tiling_.sourceBlocks * (2U * SOURCE_TILE) + within;
            const AscendC::DataCopyExtParams copy{
                static_cast<uint16_t>(2U * tiling_.sourceBlocks), segment * 4U, (SOURCE_TILE - segment) * 4U,
                (rows_ - segment) / 8U, 0U};
            const AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(real_[copied], w_[offset], copy, pad);
            copied += segment;
        }
        // Only finalizer partials are padded. Matrix loads remain
        // exact, and no arithmetic on nonexistent A/x entries enters
        // the sum. The padding is disjoint from the in-flight DMA.
        if (reductionBlocks_ != tiling_.sourceBlocks) {
            AscendC::Duplicate(
                real_[2U * tiling_.sourceBlocks * rows_], 0.0f, 2U * (reductionBlocks_ - tiling_.sourceBlocks) * rows_);
        }
        CsymvSync<AscendC::HardEvent::MTE2_V>();
        if (reductionBlocks_ != tiling_.sourceBlocks) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (uint32_t width = reductionBlocks_ * rows_; width >= 2U * rows_; width /= 2U) {
            AscendC::Add(real_, real_, real_[width], width);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (!vectorize) {
            CsymvSync<AscendC::HardEvent::V_S>();
        }
    }

public:
    __aicore__ inline void InitWithStorage(
        AscendC::LocalTensor<float> data, GM_ADDR alpha, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace,
        const CsymvTilingData& tiling)
    {
        InitState(alpha, beta, y, workspace, tiling);
        storage_ = data;
    }

    __aicore__ inline void Process()
    {
        const bool alphaZero = tiling_.alphaReal == 0.0f && tiling_.alphaImag == 0.0f;
        if (alphaZero && tiling_.betaReal == 1.0f && tiling_.betaImag == 0.0f) {
            return;
        }
        const bool vectorize =
            tiling_.incy == 1 && tiling_.betaReal == 0.0f && tiling_.betaImag == 0.0f && tiling_.sourceBlocks >= 2U;
        const uint32_t core = AscendC::GetBlockIdx();
        if (core >= tiling_.finalizerBlocks) {
            return;
        }
        const uint32_t groups = (tiling_.n + ROWS_PER_GROUP - 1U) / ROWS_PER_GROUP;
        const uint32_t base = groups / tiling_.finalizerBlocks;
        const uint32_t extra = groups % tiling_.finalizerBlocks;
        const uint32_t count = base + (core < extra ? 1U : 0U);
        const uint32_t start = core * base + (core < extra ? core : extra);
        for (uint32_t done = 0U; done < count;) {
            const uint32_t batch = count - done < MAX_ROWS / ROWS_PER_GROUP ? count - done : MAX_ROWS / ROWS_PER_GROUP;
            rows_ = batch * ROWS_PER_GROUP;
            const uint32_t row = (start + done) * ROWS_PER_GROUP;
            BindStorage(storage_);
            if (vectorize && !alphaZero) {
                InitOffsets();
            }
            if (!alphaZero) {
                ReducePartials(row, vectorize);
            }
            if (vectorize) {
                StoreVector(row, alphaZero);
                done += batch;
                continue;
            }
            for (uint32_t r = 0; r < rows_ && row + r < tiling_.n; ++r) {
                Store(row + r, r, alphaZero);
            }
            if (tiling_.incy == 1) {
                CsymvSync<AscendC::HardEvent::S_MTE3>();
                const uint32_t rows = tiling_.n - row < rows_ ? tiling_.n - row : rows_;
                AscendC::DataCopyExtParams copy{1U, rows * 8U, 0U, 0U, 0U};
                AscendC::DataCopyPad(y_[row * 2U], out_, copy);
                CsymvSync<AscendC::HardEvent::MTE3_S>();
            }
            CsymvSync<AscendC::HardEvent::S_V>();
            done += batch;
        }
    }
};

// Keep packed launch layouts and exported kernel symbols in this compilation unit.
struct CsymvRuntimeArguments {
    uint8_t* alpha;
    uint8_t* a;
    uint8_t* x;
    uint8_t* beta;
    uint8_t* y;
    CsymvTilingData tiling;
};
static_assert(offsetof(CsymvRuntimeArguments, tiling) == 5U * sizeof(void*), "kernel argument offset");
static_assert(sizeof(CsymvRuntimeArguments) == 5U * sizeof(void*) + sizeof(CsymvTilingData), "kernel argument size");

template <typename Arguments>
inline int32_t LaunchCsymvPacked(const void* symbol, uint32_t blocks, void* stream, Arguments& arguments)
{
    static_assert(std::is_standard_layout<Arguments>::value, "kernel argument ABI must be standard layout");
    static_assert(std::is_trivially_copyable<Arguments>::value, "kernel arguments must be byte-copyable");
    // Resolve against the current runtime on every launch: no cached handle
    // survives device reset or runtime finalization.
    aclrtFuncHandle function = nullptr;
    const aclError result = aclrtGetFuncBySymbol(symbol, &function);
    if (result != ACL_SUCCESS) {
        return result;
    }
    if (function == nullptr) {
        return ACL_ERROR_FAILURE;
    }
    return aclrtLaunchKernelWithHostArgs(function, blocks, stream, nullptr, &arguments, sizeof(arguments), nullptr, 0U);
}

__global__ __aicore__ void csymv_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if (tiling.tiled == CSYMV_GENERIC) {
        CsymvAiv op;
        op.Init(pipe, alpha, a, x, beta, y, tiling);
        op.Process();
    } else {
        // Short rows do not hide the double-raw prefetch setup cost.
        if (tiling.n > 64U && tiling.n < 640U) {
            CsymvOwnerAiv<true> op;
            op.Init(pipe, alpha, a, x, beta, y, tiling);
            op.Process();
        } else {
            CsymvOwnerAiv<> op;
            op.Init(pipe, alpha, a, x, beta, y, tiling);
            op.Process();
        }
    }
}

// Two column phases retain their row displacement until the final result merge.
template <uint32_t BLOCKS>
class CsymvPhaseWindowAiv {
    union FloatBits {
        float f;
        uint32_t u;
    };

    static_assert(BLOCKS >= 5U && BLOCKS <= 8U, "phase window grid must fit one AIV wave");
    static constexpr uint32_t TASKS = BLOCKS * (BLOCKS + 1U) / 2U;
    static constexpr uint32_t PLANE_FLOATS = TASKS * 128U;
    static constexpr uint32_t FINAL_START = TASKS - 2U * BLOCKS;
    CsymvTilingData tiling_{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> arena_;
    AscendC::LocalTensor<float> data_;
    AscendC::LocalTensor<float> raw_;
    AscendC::LocalTensor<float> ur_;
    AscendC::LocalTensor<float> ui_;
    AscendC::LocalTensor<float> pr_;
    AscendC::LocalTensor<float> pi_;
    AscendC::LocalTensor<float> coeff_;
    AscendC::LocalTensor<float> mirror_;
    AscendC::LocalTensor<float> xc_;
    AscendC::LocalTensor<float> partial_;
    AscendC::LocalTensor<uint32_t> d_;
    AscendC::LocalTensor<uint32_t> linear_;
    AscendC::LocalTensor<uint32_t> swap_;
    AscendC::LocalTensor<uint8_t> masks_;
    AscendC::LocalTensor<uint64_t> alt_;
    AscendC::GlobalTensor<float> a_;
    AscendC::GlobalTensor<float> x_;
    AscendC::GlobalTensor<float> w_;
    AscendC::GlobalTensor<float> y_;
    uint32_t b_;
    uint32_t c_;

    __aicore__ inline void Linear(AscendC::LocalTensor<uint32_t> dst, bool swap)
    {
        auto seed = dst.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t lo = (2U * i ^ (swap ? 1U : 0U)) * 4U;
            const uint32_t hi = ((2U * i + 1U) ^ (swap ? 1U : 0U)) * 4U;
            seed.SetValue(i, (uint64_t{hi} << 32U) | lo);
        }
        CsymvSync<AscendC::HardEvent::S_V>();
        auto v = dst.ReinterpretCast<int32_t>();
        for (uint32_t n = 8; n < 128; n *= 2) {
            AscendC::Adds(v[n], v, int32_t(n * 4U), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Indices()
    {
        Linear(linear_, false);
        Linear(swap_, true);
        auto seed = d_.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < 4; ++i) {
            seed.SetValue(i, (uint64_t{(2U * i + 1U) * 16U} << 32U) | (2U * i * 16U));
        }
        alt_.SetValue(0, 0x5555555555555555ULL);
        CsymvSync<AscendC::HardEvent::S_V>();
        auto v = d_.ReinterpretCast<int32_t>();
        AscendC::Adds(v[8], v, int32_t{128}, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(v[16], v, int32_t{256}, 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(v[32], v, int32_t{8}, 32);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Load()
    {
        // The cache's logical origin is 64*b-4; its first four values are padding at b=0.
        AscendC::Duplicate(xc_, 0.0f, 272);
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        const uint32_t rowStart = b_ == 0 ? 0 : 64U * b_ - 4U;
        const uint32_t rowCount =
            tiling_.n - rowStart < (b_ == 0 ? 64U : 68U) ? tiling_.n - rowStart : (b_ == 0 ? 64U : 68U);
        const AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xc_[b_ == 0 ? 8 : 0], x_[2U * rowStart], {1, rowCount * 8U, 0, 0, 0}, pad);
        const uint32_t cols = c_ + 1U == BLOCKS ? tiling_.n - 64U * c_ : 64U;
        AscendC::DataCopyPad(xc_[144], x_[128U * c_], {1, cols * 8U, 0, 0, 0}, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
        const uint32_t nc = cols / 2U;
        const uint16_t stride = static_cast<uint16_t>(tiling_.lda / 2U - 16U);
        AscendC::DataCopy(
            raw_, a_[2ULL * (uint64_t{tiling_.lda} * 64U * c_ + 64U * b_)], {static_cast<uint16_t>(nc), 16, stride, 0});
        const bool corner = b_ + 1U == BLOCKS && c_ + 1U == BLOCKS;
        const uint32_t no = corner ? nc - 1U : nc;
        const uint64_t oddStart = uint64_t{tiling_.lda} * (64U * c_ + 1U) + 64U * b_ - 2U;
        if (no != 0U) {
            AscendC::DataCopy(raw_[4096], a_[2ULL * oddStart], {static_cast<uint16_t>(no), 16, stride, 0});
        }
        if (corner) {
            const uint64_t offset = uint64_t{tiling_.lda} * (tiling_.n - 1U) + 64U * b_ - 2U;
            AscendC::DataCopy(
                raw_[4096 + no * 128U], a_[2ULL * offset], {1, static_cast<uint16_t>((cols + 2U) / 4U), 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
    }

    __aicore__ inline void Coefficients()
    {
        AscendC::Gather(partial_, xc_[144], d_, 0, 64);
        AscendC::Gather(partial_[64], xc_[144], d_, 4, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(partial_[128], partial_[64], -1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(coeff_, partial_, 8, {1, 8});
        AscendC::Brcb(coeff_[512], partial_[64], 8, {1, 8});
        AscendC::Brcb(ur_, partial_[128], 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams p{1, 1, 1, 8, 8, 8};
        AscendC::Select(coeff_[512], alt_, coeff_[512], coeff_, AscendC::SELMODE::VSEL_CMPMASK_SPR, uint64_t{64}, 8, p);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(coeff_, alt_, coeff_, ur_, AscendC::SELMODE::VSEL_CMPMASK_SPR, uint64_t{64}, 8, p);
        AscendC::Gather(mirror_, xc_, linear_, 32, 128);
        AscendC::Gather(mirror_[128], xc_, linear_, 16, 128);
        AscendC::Gather(mirror_[256], xc_, swap_, 32, 128);
        AscendC::Gather(mirror_[384], xc_, swap_, 16, 128);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0};
        AscendC::Muls<float>(mirror_, mirror_, -1.0f, odd, 4, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Masks()
    {
        // A 64-wide row ramp and phase-grouped column thresholds create both predicates.
        CsymvSync<AscendC::HardEvent::V_S>();
        auto seed = ur_.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < 4; ++i) {
            FloatBits lo;
            FloatBits hi;
            lo.f = float(int32_t(2U * i));
            hi.f = float(int32_t(2U * i + 1U));
            seed.SetValue(i, (uint64_t{hi.u} << 32U) | lo.u);
        }
        auto thresholds = partial_;
        auto ts = thresholds.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < 4; ++i) {
            FloatBits lo;
            FloatBits hi;
            lo.f = float(int32_t(64U * c_) - int32_t(64U * b_) + int32_t(4U * i));
            hi.f = lo.f + 2.0f;
            ts.SetValue(i, (uint64_t{hi.u} << 32U) | lo.u);
        }
        CsymvSync<AscendC::HardEvent::S_V>();
        for (uint32_t n = 8; n < 64; n *= 2) {
            AscendC::Adds(ur_[n], ur_, float(int32_t(n)), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds(thresholds[8], thresholds, 16.0f, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(thresholds[16], thresholds, 32.0f, 16);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(thresholds[32], thresholds, 3.0f, 32);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(ui_, thresholds, 8, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams p{1, 1, 0, 8, 0, 1};
        AscendC::Compare<float, uint8_t>(masks_, ur_, ui_, AscendC::CMPMODE::GE, uint64_t{64}, 64, p);
        AscendC::Compare<float, uint8_t>(masks_[512], ur_, ui_, AscendC::CMPMODE::GT, uint64_t{64}, 64, p);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MaskProducts(bool mirror)
    {
        if (b_ == c_) {
            auto mask = masks_[mirror ? 512 : 0];
            AscendC::Select(pr_, mask, pr_, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 4096);
            AscendC::Select(pi_, mask, pi_, 0.0f, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 4096);
        }
        if (b_ == c_ + 1U) {
            uint64_t boundary[2] = {mirror ? 3U : 1U, 0U};
            AscendC::Duplicate<float>(pr_[4032], 0.0f, boundary, 1, 1, 8);
            AscendC::Duplicate<float>(pi_[4032], 0.0f, boundary, 1, 1, 8);
        }
        if (b_ + 1U == BLOCKS) {
            const uint32_t tail = tiling_.n - 64U * b_;
            uint64_t even[2] = {~((uint64_t{1} << tail) - 1U), 0};
            uint64_t odd[2] = {tail == 62U ? uint64_t{0} : ~((uint64_t{1} << (tail + 2U)) - 1U), 0};
            AscendC::Duplicate<float>(pr_, 0.0f, even, 32, 1, 8);
            AscendC::Duplicate<float>(pi_, 0.0f, even, 32, 1, 8);
            AscendC::Duplicate<float>(pr_[2048], 0.0f, odd, 32, 1, 8);
            AscendC::Duplicate<float>(pi_[2048], 0.0f, odd, 32, 1, 8);
        }
        if (c_ + 1U == BLOCKS) {
            const uint32_t columns = (tiling_.n - 64U * c_) / 2U;
            for (uint32_t phase = 0; phase < 2; ++phase) {
                AscendC::Duplicate(pr_[phase * 2048U + columns * 64U], 0.0f, (32U - columns) * 64U);
                AscendC::Duplicate(pi_[phase * 2048U + columns * 64U], 0.0f, (32U - columns) * 64U);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Products(bool mirror)
    {
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        if (!mirror) {
            const AscendC::BinaryRepeatParams p{1, 1, 0, 16, 16, 1};
            AscendC::Mul<float, false>(ur_, raw_, coeff_, uint64_t{64}, 64, p);
            AscendC::Mul<float, false>(ur_[64], raw_[64], coeff_, uint64_t{64}, 64, p);
            AscendC::Mul<float, false>(ui_, raw_, coeff_[512], uint64_t{64}, 64, p);
            AscendC::Mul<float, false>(ui_[64], raw_[64], coeff_[512], uint64_t{64}, 64, p);
        } else {
            const AscendC::BinaryRepeatParams p{1, 1, 1, 16, 16, 0};
            for (uint32_t phase = 0; phase < 2; ++phase) {
                for (uint32_t h = 0; h < 2; ++h) {
                    const uint32_t offset = phase * 4096U + h * 64U;
                    AscendC::Mul<float, false>(
                        ur_[offset], raw_[offset], mirror_[phase * 128U + h * 64U], uint64_t{64}, 32, p);
                    AscendC::Mul<float, false>(
                        ui_[offset], raw_[offset], mirror_[256U + phase * 128U + h * 64U], uint64_t{64}, 32, p);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum<float, false>(pr_, ur_, 128, 64, 1, 1, 8);
        AscendC::PairReduceSum<float, false>(pi_, ui_, 128, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        MaskProducts(mirror);
    }

    __aicore__ inline void Produce()
    {
        Indices();
        Load();
        Coefficients();
        if (b_ == c_)
            Masks();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Products(false);
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        auto src = pr_;
        auto dst = ur_;
        const AscendC::BinaryRepeatParams pair{1, 1, 1, 8, 16, 16};
        for (uint32_t repeats = 64; repeats >= 4; repeats /= 2) {
            AscendC::Add<float, false>(dst, src, src[64], uint64_t{64}, repeats, pair);
            AscendC::PipeBarrier<PIPE_V>();
            auto tmp = src;
            src = dst;
            dst = tmp;
        }
        AscendC::Muls(partial_, src, 1.0f, 256);
        AscendC::PipeBarrier<PIPE_V>();
        // src order is D0.Re,D2.Re,D0.Im,D2.Im; publish separate private records.
        CsymvSync<AscendC::HardEvent::V_MTE3>();
        const uint32_t td = b_ * (b_ + 1U) / 2U + c_;
        AscendC::DataCopy(w_[td * 128U], partial_, {2, 8, 8, 0});
        AscendC::DataCopy(w_[PLANE_FLOATS + td * 128U], partial_[64], {2, 8, 8, 0});
        Products(true);
        AscendC::SetVectorMask<float>(uint64_t{0}, ~uint64_t{0});
        AscendC::BlockReduceSum<float, false>(ur_, pr_, 128, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceSum<float, false>(ur_[1024], ur_, 16, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        CsymvSync<AscendC::HardEvent::V_MTE3>();
        const uint32_t tm = c_ * (2U * BLOCKS + 1U - c_) / 2U + b_ - c_;
        AscendC::DataCopy(w_[2U * PLANE_FLOATS + tm * 128U], ur_[1024], 128);
        CsymvSync<AscendC::HardEvent::MTE3_V>();
    }

    __aicore__ inline void FinalIndices(AscendC::LocalTensor<uint32_t> indices, uint32_t mode)
    {
        CsymvSync<AscendC::HardEvent::V_S>();
        auto words = indices.ReinterpretCast<uint64_t>();
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t re = mode == 0 ? i + 2U : (mode == 1 ? (i % 2U) * 16U + i / 2U : i);
            const uint32_t im = re + (mode == 0 ? 40U : 32U);
            words.SetValue(i, (uint64_t{im * 4U} << 32U) | re * 4U);
        }
        CsymvSync<AscendC::HardEvent::S_V>();
        auto v = indices.ReinterpretCast<int32_t>();
        for (uint32_t n = 8; n < 64; n *= 2) {
            AscendC::Adds(v[n], v, int32_t(mode == 1 ? n : 2U * n), n);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadFinalPartials(uint32_t q, uint32_t s)
    {
        auto d0 = data_;
        auto d2 = data_[512];
        auto head = data_[1152];
        auto m = data_[1280];
        // Re/Im of D2 use a 40-float pitch for the two-row phase shift.
        AscendC::Duplicate(data_, 0.0f, 1792);
        CsymvSync<AscendC::HardEvent::V_MTE2>();
        const AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        const uint32_t td = q * (q + 1U) / 2;
        AscendC::DataCopyPad(d0, w_[td * 128U + s], {static_cast<uint16_t>(2U * (q + 1)), 128, 128, 0, 0}, pad);
        if (s == 0) {
            AscendC::DataCopyPad(
                d2, w_[PLANE_FLOATS + td * 128U], {static_cast<uint16_t>(2U * (q + 1)), 160, 96, 0, 0}, pad);
        } else {
            AscendC::DataCopyPad(
                d2, w_[PLANE_FLOATS + td * 128U + 32U], {static_cast<uint16_t>(2U * (q + 1)), 128, 128, 1, 0}, pad);
            if (q + 1U < BLOCKS) {
                const uint32_t next = (q + 1U) * (q + 2U) / 2U;
                AscendC::DataCopyPad(
                    head, w_[PLANE_FLOATS + next * 128U], {static_cast<uint16_t>(2U * (q + 1)), 32, 224, 0, 0}, pad);
            }
        }
        const uint32_t tm = q * (2U * BLOCKS + 1U - q) / 2U;
        // M records contain phase-grouped columns: 16 even/odd outputs per half group.
        AscendC::DataCopyPad(
            m, w_[2U * PLANE_FLOATS + tm * 128U + s / 2], {static_cast<uint16_t>(4U * (BLOCKS - q)), 64, 64, 0, 0},
            pad);
        CsymvSync<AscendC::HardEvent::MTE2_V>();
    }

    __aicore__ inline void ReduceFinalPartials(uint32_t q, uint32_t s)
    {
        auto d0 = data_;
        auto d2 = data_[512];
        auto head = data_[1152];
        auto m = data_[1280];
        for (uint32_t half = 4; half; half /= 2) {
            AscendC::Add(d0, d0, d0[half * 64U], half * 64U);
            AscendC::Add(d2, d2, d2[half * 80U], half * 80U);
            AscendC::Add(m, m, m[half * 64U], half * 64U);
            if (s && q + 1U < BLOCKS)
                AscendC::Add(head, head, head[half * 16U], half * 16U);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (s) {
            AscendC::Muls(d2[32], head, 1.0f, 8);
            AscendC::Muls(d2[72], head[8], 1.0f, 8);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void MergeFinalPlanes()
    {
        auto d0 = data_;
        auto d2 = data_[512];
        auto m = data_[1280];
        auto sums = data_[1792];
        auto temp = data_[1920];
        auto indices = data_[2304].ReinterpretCast<uint32_t>();
        FinalIndices(indices, 0);
        AscendC::Gather(temp, d2, indices, 0, 64);
        AscendC::PipeBarrier<PIPE_V>();
        FinalIndices(indices, 1);
        AscendC::Gather(sums, m, indices, 0, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sums, sums, temp, 64);
        AscendC::PipeBarrier<PIPE_V>();
        FinalIndices(indices, 2);
        AscendC::Gather(temp, d0, indices, 0, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(sums, sums, temp, 64);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool FinalValuesNonfinite(uint32_t rows)
    {
        auto sums = data_[1792];
        auto exponents = data_[2048].ReinterpretCast<uint32_t>();
        auto bad = data_[2112].ReinterpretCast<uint8_t>();
        AscendC::ShiftLeft(exponents, sums.ReinterpretCast<uint32_t>(), uint32_t{1}, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(exponents, exponents, uint32_t{24}, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compares<int32_t, uint8_t>(bad, exponents.ReinterpretCast<int32_t>(), 255, AscendC::CMPMODE::EQ, 64);
        CsymvSync<AscendC::HardEvent::V_S>();
        const uint64_t valid = rows == 32U ? ~uint64_t{0} : ((uint64_t{1} << (2U * rows)) - 1U);
        return (bad.ReinterpretCast<uint64_t>().GetValue(0) & valid) != 0;
    }

public:
    __aicore__ inline void Init(
        AscendC::TPipe& pipe, GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR w, const CsymvTilingData& tiling)
    {
        tiling_ = tiling;
        pipe.InitBuffer(arena_, 176640);
        data_ = arena_.Get<float>();
        raw_ = data_;
        ur_ = data_[16384];
        ui_ = data_[24576];
        pr_ = data_[32768];
        pi_ = data_[36864];
        coeff_ = data_[40960];
        mirror_ = data_[41984];
        xc_ = data_[42496];
        d_ = data_[42880].ReinterpretCast<uint32_t>();
        linear_ = data_[42944].ReinterpretCast<uint32_t>();
        swap_ = data_[43072].ReinterpretCast<uint32_t>();
        alt_ = data_[43200].ReinterpretCast<uint64_t>();
        masks_ = data_[43392].ReinterpretCast<uint8_t>();
        partial_ = data_[43648];
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        const uint32_t core = AscendC::GetBlockIdx();
        if (core < BLOCKS) {
            b_ = core;
            c_ = core;
        } else {
            uint32_t index = core - BLOCKS;
            c_ = 0;
            while (index >= BLOCKS - 1U - c_) {
                index -= BLOCKS - 1U - c_;
                ++c_;
            }
            b_ = c_ + 1U + index;
        }
    }

    __aicore__ inline bool Finalize()
    {
        const uint32_t index = AscendC::GetBlockIdx();
        if (index < FINAL_START)
            return false;
        const uint32_t core = index - FINAL_START;
        const uint32_t row = core * 32U;
        if (row >= tiling_.n)
            return false;
        const uint32_t rows = tiling_.n - row < 32U ? tiling_.n - row : 32U;
        const uint32_t q = core / 2;
        const uint32_t s = (core % 2) * 32;
        LoadFinalPartials(q, s);
        ReduceFinalPartials(q, s);
        MergeFinalPlanes();
        if (FinalValuesNonfinite(rows)) {
            return true;
        }
        auto sums = data_[1792];
        // Nonfinite sums have replayed Owner; this branch has finite AoS values.
        AscendC::Adds(sums, sums, 0.0f, 64);
        CsymvSync<AscendC::HardEvent::V_MTE3>();
        AscendC::DataCopyPad(y_[core * 64U], sums, {1, rows * 8U, 0, 0, 0});
        CsymvSync<AscendC::HardEvent::MTE3_V>();
        return false;
    }

    __aicore__ inline void Process(
        GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, const CsymvTilingData& tiling)
    {
        Produce();
        AscendC::SyncAll<true>();
        if (Finalize()) {
            CsymvOwnerAiv<true> owner;
            owner.InitFromStorage(data_, alpha, a, x, beta, y, tiling);
            owner.ReplayRows(
                (AscendC::GetBlockIdx() - FINAL_START) * 32U, (AscendC::GetBlockIdx() - FINAL_START) * 32U + 32U);
        }
    }
};

template <uint32_t TILE>
__aicore__ inline void CsymvTriangle(
    AscendC::TPipe& pipe, GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace,
    const CsymvTilingData& tiling)
{
    // SyncAll separates workspace publication from reduction; reuse producer UB afterwards.
    CsymvTriangleAiv<TILE> producer;
    producer.Init(pipe, alpha, a, x, workspace, tiling);
    producer.Process();
    auto finalStorage = producer.GetFinalizeBuffer();
    AscendC::SyncAll<true>();
    CsymvFinalizeAiv<TILE> finalizer;
    finalizer.InitWithStorage(finalStorage, alpha, beta, y, workspace, tiling);
    finalizer.Process();
}

__global__ __aicore__ __schedmode__(1) void csymv_triangle_large_fused_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    if (tiling.tiled == CSYMV_TRIANGLE_SMALL) {
        CsymvTriangle<64U>(pipe, alpha, a, x, beta, y, workspace, tiling);
    } else {
        CsymvTriangle<128U>(pipe, alpha, a, x, beta, y, workspace, tiling);
    }
}

__global__ __aicore__ __schedmode__(1) void csymv_phase5_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    CsymvPhaseWindowAiv<5U> op;
    op.Init(pipe, a, x, y, workspace, tiling);
    op.Process(alpha, a, x, beta, y, tiling);
}

__global__ __aicore__ __schedmode__(1) void csymv_phase6_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    CsymvPhaseWindowAiv<6U> op;
    op.Init(pipe, a, x, y, workspace, tiling);
    op.Process(alpha, a, x, beta, y, tiling);
}

__global__ __aicore__ __schedmode__(1) void csymv_phase7_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    CsymvPhaseWindowAiv<7U> op;
    op.Init(pipe, a, x, y, workspace, tiling);
    op.Process(alpha, a, x, beta, y, tiling);
}

__global__ __aicore__ __schedmode__(1) void csymv_phase8_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    CsymvPhaseWindowAiv<8U> op;
    op.Init(pipe, a, x, y, workspace, tiling);
    op.Process(alpha, a, x, beta, y, tiling);
}

const void* CsymvPhaseEntry(uint32_t blocks)
{
    switch (blocks) {
        case 5U:
            return reinterpret_cast<const void*>(csymv_phase5_kernel);
        case 6U:
            return reinterpret_cast<const void*>(csymv_phase6_kernel);
        case 7U:
            return reinterpret_cast<const void*>(csymv_phase7_kernel);
        case 8U:
            return reinterpret_cast<const void*>(csymv_phase8_kernel);
        default:
            return nullptr;
    }
}

int32_t csymv_kernel_do(
    uint8_t* alpha, uint8_t* a, uint8_t* x, uint8_t* beta, uint8_t* y, uint8_t* workspace, uint32_t numBlocks,
    const CsymvTilingData& tiling, void* stream)
{
    if (tiling.triangle == 0U) {
        CsymvRuntimeArguments arguments{alpha, a, x, beta, y, tiling};
        return LaunchCsymvPacked(reinterpret_cast<const void*>(csymv_kernel), numBlocks, stream, arguments);
    }
    struct Arguments {
        uint8_t* alpha;
        uint8_t* a;
        uint8_t* x;
        uint8_t* beta;
        uint8_t* y;
        uint8_t* workspace;
        CsymvTilingData tiling;
    } arguments{alpha, a, x, beta, y, workspace, tiling};
    static_assert(offsetof(Arguments, tiling) == 6U * sizeof(void*), "kernel argument offset");
    static_assert(sizeof(Arguments) == 6U * sizeof(void*) + sizeof(CsymvTilingData), "kernel argument size");
    const void* kernel = tiling.triangle == 2U ? CsymvPhaseEntry(tiling.sourceBlocks) :
                                                 reinterpret_cast<const void*>(csymv_triangle_large_fused_kernel);
    return LaunchCsymvPacked(kernel, numBlocks, stream, arguments);
}
