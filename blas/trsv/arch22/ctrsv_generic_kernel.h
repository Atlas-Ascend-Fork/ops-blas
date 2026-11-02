/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "ctrsv_kernel.h"
#include "ctrsv_kernel_utils.h"
#include "ctrsv_kernel_layout.h"
#include "common/helper/kernel_utils.h"
#include "common/memory/mem.h"

namespace ctrsv_generic {

template <bool TRANSPOSE, bool CONJUGATE, bool UNIT_DIAG, bool MATRIX_RHS = false>
class Kernel {
    using Complex = ctrsv_netlib::Complex;
    static constexpr uint32_t TILE = 64U;
    static constexpr uint32_t BLOCK_FLOATS = 8U;
    AsdopsBuffer<ArchType::ASCEND_V220> buffer_;
    AscendC::GlobalTensor<float> matrixGm_, vectorGm_;
    AscendC::LocalTensor<float> matrixAos_, vectorAos_, output_;
    AscendC::LocalTensor<float> ar_, ai_, xr_, xi_, pr_, pi_, real_, imag_, scratch_;
    AscendC::LocalTensor<float> fmaScratch_;
    AscendC::LocalTensor<uint32_t> inputOffsets_, outputOffsets_;
    int32_t n_, lda_, incx_, uplo_;

    __aicore__ inline bool Upper() const { return (uplo_ == ACLBLAS_UPPER) != TRANSPOSE; }
    __aicore__ inline uint64_t VectorStep() const { return incx_ < 0 ? -static_cast<int64_t>(incx_) : incx_; }
    __aicore__ inline uint64_t VectorOffset(uint32_t index) const
    {
        const uint64_t logical = incx_ < 0 ? static_cast<uint64_t>(n_ - 1U - index) : index;
        return logical * VectorStep() * 2U;
    }
    __aicore__ inline void InitializeOffsets()
    {
        ctrsv_layout::CtrsvIndexRamp64<32U>(inputOffsets_.template ReinterpretCast<int32_t>());
        BuildOutputOffsets(TILE);
    }
    __aicore__ inline void BuildOutputOffsets(uint32_t count)
    {
        auto offsets = scratch_.template ReinterpretCast<int32_t>();
        ctrsv_layout::CtrsvIndexRamp64<sizeof(float)>(offsets);
        if (incx_ < 0) {
            AscendC::Muls(offsets, offsets, int32_t{-1}, TILE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(offsets, offsets, static_cast<int32_t>((count - 1U) * sizeof(float)), TILE);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Brcb(outputOffsets_, offsets.template ReinterpretCast<uint32_t>(), TILE / BLOCK_FLOATS, {1U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t imaginaryLanes[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        auto packed = outputOffsets_.template ReinterpretCast<int32_t>();
        AscendC::Adds(
            packed, packed, static_cast<int32_t>(TILE * sizeof(float)), imaginaryLanes,
            static_cast<uint8_t>(TILE / BLOCK_FLOATS), AscendC::UnaryRepeatParams{1U, 1U, 8U, 8U});
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyVectorIn(uint32_t first, uint32_t count)
    {
        const uint32_t physicalFirst = incx_ < 0 ? first + count - 1U : first;
        const uint64_t offset = VectorOffset(physicalFirst);
        const uint64_t gap = (VectorStep() - 1U) * 2U * sizeof(float);
        const AscendC::DataCopyPadExtParams<float> pad{true, 0U, 6U, 0.0f};
        if (gap <= 0xffffffffULL) {
            AscendC::DataCopyPad(
                vectorAos_, vectorGm_[offset],
                AscendC::DataCopyExtParams{static_cast<uint16_t>(count), 8U, static_cast<uint32_t>(gap), 0U, 0U}, pad);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::DataCopyPad(
                    vectorAos_[i * BLOCK_FLOATS], vectorGm_[offset + i * VectorStep() * 2U],
                    AscendC::DataCopyExtParams{1U, 8U, 0U, 0U, 0U}, pad);
            }
        }
    }
    __aicore__ inline void LoadVector(uint32_t first, uint32_t count)
    {
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        CopyVectorIn(first, count);
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        auto offsets = scratch_.template ReinterpretCast<int32_t>();
        if (incx_ < 0) {
            AscendC::Muls(offsets, inputOffsets_.template ReinterpretCast<int32_t>(), int32_t{-1}, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(offsets, offsets, static_cast<int32_t>((count - 1U) * 32U), count);
        } else {
            AscendC::Adds(offsets, inputOffsets_.template ReinterpretCast<int32_t>(), int32_t{0}, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Gather(xr_, vectorAos_, offsets.template ReinterpretCast<uint32_t>(), 0U, count);
        AscendC::Gather(xi_, vectorAos_, offsets.template ReinterpretCast<uint32_t>(), 4U, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void PackVector(uint32_t count)
    {
        if (incx_ < 0)
            BuildOutputOffsets(count);
        AscendC::Gather(output_, real_, outputOffsets_, 0U, count * BLOCK_FLOATS);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StoreVector(uint32_t first, uint32_t count)
    {
        PackVector(count);
        const uint32_t physicalFirst = incx_ < 0 ? first + count - 1U : first;
        const uint64_t offset = VectorOffset(physicalFirst);
        const uint64_t gap = (VectorStep() - 1U) * 2U * sizeof(float);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        if (gap <= 0xffffffffULL) {
            AscendC::DataCopyPad(
                vectorGm_[offset], output_,
                AscendC::DataCopyExtParams{static_cast<uint16_t>(count), 8U, 0U, static_cast<uint32_t>(gap), 0U});
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                AscendC::DataCopyPad(
                    vectorGm_[offset + i * VectorStep() * 2U], output_[i * BLOCK_FLOATS],
                    AscendC::DataCopyExtParams{1U, 8U, 0U, 0U, 0U});
            }
        }
        SET_FLAG(MTE3, V, EVENT_ID0);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }
    __aicore__ inline void LoadMatrixColumn(uint32_t column, uint32_t first, uint32_t count)
    {
        const uint64_t offset = (static_cast<uint64_t>(column) * lda_ + first) * 2U;
        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        AscendC::DataCopyPad(
            matrixAos_, matrixGm_[offset], AscendC::DataCopyExtParams{1U, count * 8U, 0U, 0U, 0U},
            AscendC::DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((8U - count * 2U % 8U) % 8U), 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        auto offsets = scratch_.template ReinterpretCast<int32_t>();
        // Input offsets are 32-byte blocks; a dense complex column uses 8 bytes.
        ctrsv_layout::CtrsvIndexRamp64<8U>(offsets);
        AscendC::Gather(ar_, matrixAos_, offsets.template ReinterpretCast<uint32_t>(), 0U, count);
        AscendC::Gather(ai_, matrixAos_, offsets.template ReinterpretCast<uint32_t>(), 4U, count);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (CONJUGATE) {
            ConjugateColumn(count);
        }
    }
    __aicore__ inline void ConjugateColumn(uint32_t count)
    {
        // Flip the sign bit without flushing subnormals or changing signed zero.
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        for (uint32_t i = 0; i < count; ++i)
            ai_.SetValue(i, ctrsv_netlib::Value(ctrsv_netlib::Bits(ai_.GetValue(i)) ^ 0x80000000U));
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
    }
    __aicore__ inline Complex DividePivot(uint32_t row, Complex value)
    {
        if constexpr (UNIT_DIAG) {
            return value;
        } else {
            LoadMatrixColumn(row, row, 1U);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            value = ctrsv_netlib::Divide(value, {ar_.GetValue(0U), ai_.GetValue(0U)});
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            return value;
        }
    }
    // Split into non-overlapping high/low parts without changing the original inputs.
    __aicore__ inline void SplitOperand(
        AscendC::LocalTensor<float> hi, AscendC::LocalTensor<float> lo, AscendC::LocalTensor<float> x,
        AscendC::LocalTensor<float> work, uint32_t count)
    {
        AscendC::Muls(work, x, 4097.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(hi, work, x, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(hi, work, hi, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(lo, x, hi, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    // Compensate the separately rounded A2 multiply/add before the column update.
    __aicore__ inline void RoundedFma(
        AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> b,
        AscendC::LocalTensor<float> c, uint32_t count)
    {
        auto ah = fmaScratch_, al = ah[TILE], bh = al[TILE], bl = bh[TILE], p = bl[TILE];
        auto error = p[TILE], hi = error[TILE], z = hi[TILE], lo = z[TILE], work = lo[TILE];
        SplitOperand(ah, al, a, work, count);
        SplitOperand(bh, bl, b, work, count);
        AscendC::Mul(p, a, b, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(error, ah, bh, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(error, error, p, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work, ah, bl, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(error, error, work, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work, al, bh, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(error, error, work, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(work, al, bl, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(error, error, work, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(hi, p, c, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(z, hi, p, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(lo, hi, z, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(lo, p, lo, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(work, c, z, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(lo, lo, work, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(lo, error, lo, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(dst, hi, lo, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void ComputeProducts(uint32_t count)
    {
        // Netlib CTRSM column updates use fused products and the reverse imaginary-product order.
        AscendC::Mul(pr_, ai_, xi_, count);
        AscendC::Mul(pi_, MATRIX_RHS && !TRANSPOSE ? ar_ : ai_, MATRIX_RHS && !TRANSPOSE ? xi_ : xr_, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(pr_, pr_, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (MATRIX_RHS && !TRANSPOSE) {
            RoundedFma(pr_, ar_, xr_, pr_, count);
            RoundedFma(pi_, ai_, xr_, pi_, count);
        } else {
            AscendC::MulAddDst(pr_, ar_, xr_, count);
            AscendC::MulAddDst(pi_, ar_, xi_, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        for (uint32_t i = 0; i < count; ++i) {
            const Complex a{ar_.GetValue(i), ai_.GetValue(i)}, x{xr_.GetValue(i), xi_.GetValue(i)};
            if (ctrsv_netlib::NeedsSoftwareProduct(a) || ctrsv_netlib::NeedsSoftwareProduct(x)) {
                const Complex product =
                    ctrsv_netlib::Product(MATRIX_RHS && !TRANSPOSE ? x : a, MATRIX_RHS && !TRANSPOSE ? a : x);
                pr_.SetValue(i, product.real);
                pi_.SetValue(i, product.imag);
            }
        }
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
    }
    __aicore__ inline Complex AccumulateRow(uint32_t row, Complex sum)
    {
        const uint32_t solved = Upper() ? n_ - 1U - row : row;
        for (uint32_t ordinal = 0; ordinal < solved; ordinal += TILE) {
            const uint32_t count = ctrsv_layout::MinU32(TILE, solved - ordinal);
            const uint32_t first = Upper() ? n_ - ordinal - count : ordinal;
            LoadVector(first, count);
            LoadMatrixColumn(row, first, count);
            ComputeProducts(count);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t lane = Upper() ? count - i - 1U : i;
                volatile float real = sum.real - pr_.GetValue(lane);
                volatile float imag = sum.imag - pi_.GetValue(lane);
                sum = {real, imag};
            }
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
        }
        return sum;
    }
    __aicore__ inline void UpdateColumn(uint32_t pivot, Complex value)
    {
        const uint32_t first = Upper() ? 0U : pivot + 1U;
        const uint32_t end = Upper() ? pivot : n_;
        for (uint32_t start = first; start < end; start += TILE) {
            const uint32_t count = ctrsv_layout::MinU32(TILE, end - start);
            LoadVector(start, count);
            // Keep the old RHS separate while the product engine broadcasts the pivot.
            AscendC::DataCopy(real_, xr_, TILE);
            AscendC::DataCopy(imag_, xi_, TILE);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(xr_, value.real, count);
            AscendC::Duplicate(xi_, value.imag, count);
            AscendC::PipeBarrier<PIPE_V>();
            LoadMatrixColumn(pivot, start, count);
            ComputeProducts(count);
            UpdateResidual(count);
            StoreVector(start, count);
        }
    }

    __aicore__ inline void UpdateResidual(uint32_t count)
    {
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        bool exceptional = false;
        for (uint32_t i = 0; i < count; ++i) {
            exceptional |= ctrsv_netlib::NeedsSoftwareProduct({real_.GetValue(i), imag_.GetValue(i)}) ||
                           ctrsv_netlib::NeedsSoftwareProduct({pr_.GetValue(i), pi_.GetValue(i)});
        }
        if (exceptional) {
            for (uint32_t i = 0; i < count; ++i) {
                volatile float real = real_.GetValue(i) - pr_.GetValue(i);
                volatile float imag = imag_.GetValue(i) - pi_.GetValue(i);
                real_.SetValue(i, real);
                imag_.SetValue(i, imag);
            }
        }
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        if (!exceptional) {
            AscendC::Sub(real_, real_, pr_, count);
            AscendC::Sub(imag_, imag_, pi_, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

public:
    __aicore__ inline void Init(GM_ADDR matrix, GM_ADDR vector, const CtrsvTilingData& td)
    {
        matrixGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        vectorGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector));
        n_ = td.n;
        lda_ = td.lda;
        incx_ = td.incx;
        uplo_ = td.uplo;
        matrixAos_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        vectorAos_ = matrixAos_[2U * TILE];
        output_ = vectorAos_[TILE * BLOCK_FLOATS];
        ar_ = output_[TILE * BLOCK_FLOATS];
        ai_ = ar_[TILE];
        xr_ = ai_[TILE];
        xi_ = xr_[TILE];
        pr_ = xi_[TILE];
        pi_ = pr_[TILE];
        real_ = pi_[TILE];
        imag_ = real_[TILE];
        scratch_ = imag_[TILE];
        inputOffsets_ = scratch_[TILE].template ReinterpretCast<uint32_t>();
        outputOffsets_ = inputOffsets_[TILE];
        // Brcb expands TILE logical indices to TILE * BLOCK_FLOATS packed lanes.
        fmaScratch_ = outputOffsets_[TILE * BLOCK_FLOATS].template ReinterpretCast<float>();
        InitializeOffsets();
    }
    __aicore__ inline void Process()
    {
        for (uint32_t ordinal = 0; ordinal < static_cast<uint32_t>(n_); ++ordinal) {
            const uint32_t row = Upper() ? n_ - ordinal - 1U : ordinal;
            LoadVector(row, 1U);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
            Complex value{xr_.GetValue(0U), xi_.GetValue(0U)};
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            if constexpr (!TRANSPOSE) {
                if (value.real == 0.0f && value.imag == 0.0f)
                    continue;
            } else {
                value = AccumulateRow(row, value);
            }
            value = DividePivot(row, value);
            AscendC::Duplicate(real_, value.real, 1U);
            AscendC::Duplicate(imag_, value.imag, 1U);
            AscendC::PipeBarrier<PIPE_V>();
            StoreVector(row, 1U);
            if constexpr (!TRANSPOSE)
                UpdateColumn(row, value);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
};
} // namespace ctrsv_generic
