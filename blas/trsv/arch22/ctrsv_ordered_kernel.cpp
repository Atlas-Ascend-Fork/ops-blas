/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/** Shared DAV_2201 vector layout helpers for the reachable Ctrsv paths. */
#include "ctrsv_kernel.h"
#include "ctrsv_kernel_utils.h"
#include "ctrsv_kernel_layout.h"
#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/memory/mem.h"
#include "cann_ops_blas_common.h"

using namespace AscendC;
using ctrsv_layout::CtrsvIndexRamp64;
using ctrsv_layout::MinU32;
using ctrsv_layout::SplitComplex;

/** Native-tile UNIT solve with Netlib ordering and a resident RHS. */
namespace {
template <bool TRANSPOSE, bool CONJUGATE>
class CtrsvOrderedUnitKernel {
    static constexpr uint32_t TILE = 64U;
    static constexpr uint32_t MAX_N = 4096U;
    AsdopsBuffer<ArchType::ASCEND_V220> buffer_;
    GlobalTensor<float> matrix_, vector_;
    LocalTensor<float> xr_, xi_, tile_, ar_, ai_, ah_, al_, scratch_;
    uint32_t n_;
    uint32_t firstPanelCount_;
    bool upper_;
    bool tileOrdinary_;
    float pivotLimit_;
    uint32_t activeFirst_, activeEnd_;
    // Included in CtrsvOrderedUnitKernel. The ordinary path has bounded normal
    // operands; its product expansion reproduces the separate rounding stages.
    __aicore__ inline void TransposeOrderedPlane(LocalTensor<float> dst, LocalTensor<float> src)
    {
        for (uint32_t row = 0; row < TILE; row += 16U) {
            uint64_t sources[NCHW_CONV_ADDR_LIST_SIZE], targets[NCHW_CONV_ADDR_LIST_SIZE];
#pragma unroll
            for (uint32_t lane = 0; lane < 16U; ++lane) {
                sources[lane] = reinterpret_cast<uint64_t>(src[(row + lane) * TILE].GetPhyAddr());
            }
#pragma unroll
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                targets[2U * lane] = reinterpret_cast<uint64_t>(dst[lane * TILE + row].GetPhyAddr());
                targets[2U * lane + 1U] = reinterpret_cast<uint64_t>(dst[lane * TILE + row + 8U].GetPhyAddr());
            }
            TransDataTo5HD<float>(targets, sources, TransDataTo5HDParams(false, false, TILE / 8U, TILE, 1U));
        }
        PIPE_BARRIER(V);
    }

    __aicore__ inline void CoefficientBounds(LocalTensor<float> src, uint32_t result)
    {
        constexpr uint32_t plane = TILE * TILE;
        Abs(tile_, src, plane);
        PIPE_BARRIER(V);
        // Integer bit magnitudes converted numerically remain finite even for NaNs.
        Cast(tile_[plane], tile_.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, plane);
        PIPE_BARRIER(V);
        // The count-form ReduceMax/Min requires TPipe-managed scalar events.
        // This path owns its events explicitly; keep both reduction stages on V.
        WholeReduceMax(scratch_[1024U], tile_[plane], 64U, 64U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PIPE_BARRIER(V);
        WholeReduceMax(scratch_[result], scratch_[1024U], 64U, 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PIPE_BARRIER(V);
        auto predicate = scratch_[1536U].template ReinterpretCast<uint8_t>();
        CompareScalar(predicate, tile_[plane], 0.0f, CMPMODE::GT, plane);
        PIPE_BARRIER(V);
        Select(
            tile_[plane], predicate, tile_[plane], static_cast<float>(0x3f800000U), SELMODE::VSEL_TENSOR_SCALAR_MODE,
            plane);
        PIPE_BARRIER(V);
        WholeReduceMin(scratch_[1024U], tile_[plane], 64U, 64U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PIPE_BARRIER(V);
        WholeReduceMin(scratch_[result + 8U], scratch_[1024U], 64U, 1U, 1U, 1U, 8U, ReduceOrder::ORDER_ONLY_VALUE);
        PIPE_BARRIER(V);
    }

    __aicore__ inline void PrepareOrderedExpansion()
    {
        CoefficientBounds(ar_, 768U);
        CoefficientBounds(ai_, 784U);
        Muls(ah_, ar_, 4097.0f, TILE * TILE);
        PIPE_BARRIER(V);
        Sub(al_, ah_, ar_, TILE * TILE);
        PIPE_BARRIER(V);
        Sub(ah_, ah_, al_, TILE * TILE);
        PIPE_BARRIER(V);
        Sub(al_, ar_, ah_, TILE * TILE);
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        const float maxR = scratch_.GetValue(768U), maxI = scratch_.GetValue(784U);
        const float minR = scratch_.GetValue(776U), minI = scratch_.GetValue(792U);
        tileOrdinary_ = maxR <= static_cast<float>(0x47800000U) && maxI <= static_cast<float>(0x47800000U) &&
                        minR >= static_cast<float>(0x30800000U) && minI >= static_cast<float>(0x30800000U);
        const uint32_t maximum =
            tileOrdinary_ ? static_cast<uint32_t>(static_cast<int32_t>(maxR > maxI ? maxR : maxI)) : 0U;
        const int rawExponent = static_cast<int>(maximum >> 23U);
        const int exponent = rawExponent > 126 ? rawExponent - 126 : 0;
        pivotLimit_ = ctrsv_netlib::Value(static_cast<uint32_t>(127 + 122 - exponent) << 23U);
    }

    __aicore__ inline bool OrdinaryPivot(float value) const
    {
        const float magnitude = value < 0.0f ? -value : value;
        return magnitude == 0.0f || (magnitude >= 0x1p-30f && magnitude <= pivotLimit_);
    }

    __aicore__ inline bool ScalablePivot(float value) const
    {
        const float magnitude = value < 0.0f ? -value : value;
        // The minimum keeps the split-product residuals normal after scaling.
        return magnitude == 0.0f || (magnitude >= 0x1p-12f && magnitude <= 0x1.fffffep127f);
    }

    __aicore__ inline void RestoreScaledProduct(LocalTensor<float> result, LocalTensor<float> roundedAddend)
    {
        Muls(result, result, 0x1p32f, TILE);
        Abs(scratch_[832U], roundedAddend, TILE);
        PIPE_BARRIER(V);
        auto predicate = scratch_[1536U].template ReinterpretCast<uint8_t>();
        CompareScalar(predicate, scratch_[832U], ctrsv_netlib::Value(0x7f800000U), CMPMODE::EQ, TILE);
        PIPE_BARRIER(V);
        // An independently rounded infinite addend dominates a finite exact
        // product, even when the latter would overflow if rounded on its own.
        Select(result, predicate, roundedAddend, result, SELMODE::VSEL_CMPMASK_SPR, TILE);
        PIPE_BARRIER(V);
    }

    __aicore__ inline void CompleteOrderedResidual(uint32_t column, float rh, float ih, float rl, float il)
    {
        const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
        const BinaryRepeatParams binary{1U, 1U, 1U, 8U, 8U, 8U};
        auto high = ah_[column * TILE], low = al_[column * TILE];
        auto pr = scratch_[0U], pi = scratch_[64U], cr = scratch_[128U], ci = scratch_[192U];
        auto er = scratch_[256U], ei = scratch_[320U], sr = scratch_[384U], si = scratch_[448U];
        auto dr = scratch_[512U], di = scratch_[576U], tr = scratch_[640U], ti = scratch_[704U];

        Sub<float, false>(er, er, pr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(ei, ei, pi, MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(sr, pr, cr, MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(si, pi, ci, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Axpy<float, float, false>(er, high, rl, MASK_PLACEHOLDER, 1U, unary);
        Axpy<float, float, false>(ei, high, il, MASK_PLACEHOLDER, 1U, unary);
        Sub<float, false>(dr, sr, pr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(di, si, pi, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Axpy<float, float, false>(er, low, rh, MASK_PLACEHOLDER, 1U, unary);
        Axpy<float, float, false>(ei, low, ih, MASK_PLACEHOLDER, 1U, unary);
        Sub<float, false>(tr, sr, dr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(ti, si, di, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Axpy<float, float, false>(er, low, rl, MASK_PLACEHOLDER, 1U, unary);
        Axpy<float, float, false>(ei, low, il, MASK_PLACEHOLDER, 1U, unary);
        Sub<float, false>(tr, pr, tr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(ti, pi, ti, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(dr, cr, dr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(di, ci, di, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Add<float, false>(tr, dr, tr, MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(ti, di, ti, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Add<float, false>(tr, tr, er, MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(ti, ti, ei, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
        Add<float, false>(sr, sr, tr, MASK_PLACEHOLDER, 1U, binary);
        Add<float, false>(si, si, ti, MASK_PLACEHOLDER, 1U, binary);
        PIPE_BARRIER(V);
    }

    __aicore__ inline void UpdateOrdinaryRows(
        uint32_t begin, uint32_t column, uint32_t first, uint32_t last, ctrsv_netlib::Complex pivot, bool scaled)
    {
        const float real = scaled ? pivot.real * 0x1p-32f : pivot.real;
        const float imag = scaled ? pivot.imag * 0x1p-32f : pivot.imag;
        const float rh = ctrsv_netlib::Value(ctrsv_netlib::Bits(real) & 0xfffff000U);
        const float ih = ctrsv_netlib::Value(ctrsv_netlib::Bits(imag) & 0xfffff000U);
        const float rl = real - rh, il = imag - ih;
        const uint64_t active =
            (last == 64U ? ~0ULL : (1ULL << last) - 1ULL) & ~(first == 0U ? 0ULL : (1ULL << first) - 1ULL);
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        SetMaskNorm();
        AscendC::SetVectorMask<float>(0U, active);
        const UnaryRepeatParams unary{1U, 1U, 8U, 8U};
        const BinaryRepeatParams binary{1U, 1U, 1U, 8U, 8U, 8U};
        auto a = ar_[column * TILE], b = ai_[column * TILE];
        auto high = ah_[column * TILE];
        auto pr = scratch_[0U], pi = scratch_[64U], cr = scratch_[128U], ci = scratch_[192U];
        auto er = scratch_[256U], ei = scratch_[320U], sr = scratch_[384U], si = scratch_[448U];
        Muls<float, false>(pr, a, real, MASK_PLACEHOLDER, 1U, unary);
        Muls<float, false>(pi, a, imag, MASK_PLACEHOLDER, 1U, unary);
        Muls<float, false>(cr, b, -pivot.imag, MASK_PLACEHOLDER, 1U, unary);
        Muls<float, false>(ci, b, pivot.real, MASK_PLACEHOLDER, 1U, unary);
        Muls<float, false>(er, high, rh, MASK_PLACEHOLDER, 1U, unary);
        Muls<float, false>(ei, high, ih, MASK_PLACEHOLDER, 1U, unary);
        PIPE_BARRIER(V);
        if (scaled) {
            // Netlib rounds these products before its inner FMA. Scale only
            // afterwards; an overflow here must remain infinity.
            Muls<float, false>(cr, cr, 0x1p-32f, MASK_PLACEHOLDER, 1U, unary);
            Muls<float, false>(ci, ci, 0x1p-32f, MASK_PLACEHOLDER, 1U, unary);
            PIPE_BARRIER(V);
        }
        CompleteOrderedResidual(column, rh, ih, rl, il);
        if (scaled) {
            RestoreScaledProduct(sr, cr);
            RestoreScaledProduct(si, ci);
            SetMaskNorm();
            AscendC::SetVectorMask<float>(0U, active);
        }
        Sub<float, false>(xr_[begin], xr_[begin], sr, MASK_PLACEHOLDER, 1U, binary);
        Sub<float, false>(xi_[begin], xi_[begin], si, MASK_PLACEHOLDER, 1U, binary);
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
        ResetMask();
    }

    __aicore__ inline void UpdateOrderedRows(
        uint32_t begin, uint32_t column, uint32_t first, uint32_t last, ctrsv_netlib::Complex pivot)
    {
        if (begin + first < activeFirst_)
            first = activeFirst_ - begin;
        if (begin + last > activeEnd_)
            last = activeEnd_ > begin ? activeEnd_ - begin : 0U;
        if (first >= last)
            return;
        const bool ordinary = OrdinaryPivot(pivot.real) && OrdinaryPivot(pivot.imag);
        const bool scaled = !ordinary && ScalablePivot(pivot.real) && ScalablePivot(pivot.imag);
        if (!tileOrdinary_ || (!ordinary && !scaled)) {
            for (uint32_t row = first; row < last; ++row) {
                WriteX(begin + row, ctrsv_netlib::SubProduct(ReadX(begin + row), Coefficient(row, column), pivot));
            }
            return;
        }
        UpdateOrdinaryRows(begin, column, first, last, pivot, scaled);
    }

    __aicore__ inline uint32_t Count(uint32_t panel) const
    {
        if (upper_ && panel == 0U)
            return firstPanelCount_;
        const uint32_t remaining = upper_ ? n_ - firstPanelCount_ - (panel - 1U) * TILE : n_ - Start(panel);
        return remaining < TILE ? remaining : TILE;
    }

    __aicore__ inline uint32_t Start(uint32_t panel) const
    {
        if (!upper_)
            return panel * TILE;
        const uint32_t consumed = firstPanelCount_ + panel * TILE;
        return consumed < n_ ? n_ - consumed : 0U;
    }

    __aicore__ inline ctrsv_netlib::Complex ReadX(uint32_t row) const { return {xr_.GetValue(row), xi_.GetValue(row)}; }

    __aicore__ inline void WriteX(uint32_t row, ctrsv_netlib::Complex value)
    {
        xr_.SetValue(row, value.real);
        xi_.SetValue(row, value.imag);
    }

    __aicore__ inline ctrsv_netlib::Complex Coefficient(uint32_t row, uint32_t col) const
    {
        const uint32_t offset = col * TILE + row;
        return {ar_.GetValue(offset), ai_.GetValue(offset)};
    }

    __aicore__ inline void LoadTile(uint32_t target, uint32_t source)
    {
        const uint32_t rowStart = TRANSPOSE ? Start(source) : Start(target);
        const uint32_t colStart = TRANSPOSE ? Start(target) : Start(source);
        const uint32_t rows = TRANSPOSE ? Count(source) : Count(target);
        const uint32_t cols = TRANSPOSE ? Count(target) : Count(source);
        const uint32_t paddedRows = (rows + 3U) & ~3U;
        SET_FLAG(S, MTE2, EVENT_ID0);
        WAIT_FLAG(S, MTE2, EVENT_ID0);
        if (rows != TILE || cols != TILE) {
            Duplicate(tile_, 0.0f, TILE * TILE * 2U);
            SET_FLAG(V, MTE2, EVENT_ID0);
            WAIT_FLAG(V, MTE2, EVENT_ID0);
        }
        DataCopyPad(
            tile_, matrix_[(static_cast<uint64_t>(colStart) * n_ + rowStart) * 2U],
            DataCopyExtParams{static_cast<uint16_t>(cols), rows * 8U, (n_ - rows) * 8U, (TILE - paddedRows) / 4U, 0U},
            DataCopyPadExtParams<float>{true, 0U, static_cast<uint8_t>((paddedRows - rows) * 2U), 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        SplitComplex(ar_, ai_, tile_, TILE * TILE);
        if constexpr (CONJUGATE) {
            Muls(ai_, ai_, -1.0f, TILE * TILE);
            PIPE_BARRIER(V);
        }
        if constexpr (TRANSPOSE) {
            TransposeOrderedPlane(tile_, ar_);
            TransposeOrderedPlane(tile_[TILE * TILE], ai_);
            Adds(ar_, tile_, 0.0f, TILE * TILE);
            Adds(ai_, tile_[TILE * TILE], 0.0f, TILE * TILE);
            PIPE_BARRIER(V);
        }
        PrepareOrderedExpansion();
    }

    __aicore__ inline bool FinishNanSuffix(uint32_t pivot)
    {
        const auto value = ReadX(pivot);
        if (value.real == value.real || value.imag == value.imag)
            return false;
        // Both components of a solved NaN pivot contaminate every later row
        // in Netlib, including zero coefficients. Keep the solved prefix intact.
        uint32_t first = upper_ ? 0U : pivot + 1U;
        const uint32_t end = upper_ ? pivot : n_;
        const float nan = ctrsv_netlib::Value(0x7fc00000U);
        while (first < end && (first & 7U) != 0U)
            WriteX(first++, {nan, nan});
        if (first < end) {
            SET_FLAG(S, V, EVENT_ID0);
            WAIT_FLAG(S, V, EVENT_ID0);
            Duplicate(xr_[first], nan, end - first);
            Duplicate(xi_[first], nan, end - first);
            SET_FLAG(V, S, EVENT_ID0);
            WAIT_FLAG(V, S, EVENT_ID0);
        }
        return true;
    }

    __aicore__ inline void Store()
    {
        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        auto indices = scratch_.template ReinterpretCast<int32_t>();
        CtrsvIndexRamp64<2U>(indices);
        uint64_t odd[2] = {0xaaaaaaaaaaaaaaaaULL, 0U};
        Adds(indices, indices, static_cast<int32_t>(MAX_N * 4U - 2U), odd, 1U, {1U, 1U, 8U, 8U});
        PIPE_BARRIER(V);
        for (uint32_t start = 0; start < n_; start += TILE) {
            const uint32_t count = MinU32(TILE, n_ - start);
            Gather(
                tile_[start * 2U], xr_[start], indices.template ReinterpretCast<uint32_t>(), 0U,
                MinU32(32U, count) * 2U);
            if (count > 32U) {
                Gather(
                    tile_[start * 2U + 64U], xr_[start + 32U], indices.template ReinterpretCast<uint32_t>(), 0U,
                    (count - 32U) * 2U);
            }
        }
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
        DataCopyPad(vector_, tile_, DataCopyExtParams{1U, n_ * 8U, 0U, 0U, 0U});
        SET_FLAG(MTE3, S, EVENT_ID0);
        WAIT_FLAG(MTE3, S, EVENT_ID0);
    }

public:
    __aicore__ inline void Init(GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
    {
        n_ = n;
        upper_ = (uplo == ACLBLAS_UPPER) != TRANSPOSE;
        // Anchor the first solved panel at the upper end, leaving at most
        // seven unused lanes while retaining 32-byte-aligned RHS addresses.
        // The old n%64 leading tail caused avoidable cross-panel updates.
        firstPanelCount_ = (n_ & 7U) != 0U ? 56U + (n_ & 7U) : TILE;
        matrix_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrix));
        vector_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vector));
        xr_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        xi_ = xr_[MAX_N];
        tile_ = buffer_.GetBuffer<BufferType::ASCEND_UB, float>(MAX_N * 8U);
        ar_ = tile_[TILE * TILE * 2U];
        ai_ = ar_[TILE * TILE];
        ah_ = ai_[TILE * TILE];
        al_ = ah_[TILE * TILE];
        scratch_ = al_[TILE * TILE];
        static_assert((MAX_N * 2U + 7U * TILE * TILE) * sizeof(float) <= 144U * 1024U, "Ordered UNIT UB overflow");
        if ((n_ & 31U) != 0U) {
            Duplicate(tile_[(n_ & ~31U) * 2U], 0.0f, 64U);
            SET_FLAG(V, MTE2, EVENT_ID0);
            WAIT_FLAG(V, MTE2, EVENT_ID0);
        }
        DataCopyPad(
            tile_, vector_, DataCopyExtParams{1U, n_ * 8U, 0U, 0U, 0U},
            DataCopyPadExtParams<float>{false, 0U, 0U, 0.0f});
        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);
        SplitComplex(xr_, xi_, tile_, (n_ + 31U) & ~31U);
        SET_FLAG(V, S, EVENT_ID0);
        WAIT_FLAG(V, S, EVENT_ID0);
    }

    __aicore__ inline void FindActiveRange()
    {
        // An RHS row with two NaNs cannot become finite under subtraction.
        // It will contaminate every dependent row when it becomes a pivot.
        // Avoid updating that already-determined suffix, including on
        // subsequent in-place solves. This depends only on input values.
        activeFirst_ = 0U;
        activeEnd_ = n_;
        for (uint32_t ordinal = 0; ordinal < n_; ++ordinal) {
            const uint32_t row = upper_ ? n_ - 1U - ordinal : ordinal;
            const auto value = ReadX(row);
            if (ctrsv_netlib::IsNan(value.real) && ctrsv_netlib::IsNan(value.imag)) {
                if (upper_)
                    activeFirst_ = row + 1U;
                else
                    activeEnd_ = row;
                break;
            }
        }
    }

    __aicore__ inline bool ProbePanel(uint32_t target, uint32_t begin, uint32_t firstPivot)
    {
        // Probe the first row before updating the whole next panel. If it
        // becomes NaN, every dependent row is already determined.
        for (uint32_t source = 0; source < target; ++source) {
            LoadTile(target, source);
            for (uint32_t ordinal = 0; ordinal < Count(source); ++ordinal) {
                const uint32_t col = upper_ ? Count(source) - 1U - ordinal : ordinal;
                const auto solved = ReadX(Start(source) + col);
                if constexpr (!TRANSPOSE) {
                    if (solved.real == 0.0f && solved.imag == 0.0f)
                        continue;
                }
                UpdateOrderedRows(begin, col, firstPivot - begin, firstPivot - begin + 1U, solved);
                if (FinishNanSuffix(firstPivot)) {
                    return true;
                }
            }
        }

        return false;
    }

    __aicore__ inline void UpdatePanel(uint32_t target, uint32_t begin, uint32_t count)
    {
        for (uint32_t source = 0; source < target; ++source) {
            LoadTile(target, source);
            const uint32_t sourceCount = Count(source), sourceStart = Start(source);
            for (uint32_t ordinal = 0; ordinal < sourceCount; ++ordinal) {
                const uint32_t col = upper_ ? sourceCount - 1U - ordinal : ordinal;
                const auto solved = ReadX(sourceStart + col);
                if constexpr (!TRANSPOSE) {
                    if (solved.real == 0.0f && solved.imag == 0.0f)
                        continue;
                }
                UpdateOrderedRows(begin, col, upper_ ? 0U : 1U, upper_ ? count - 1U : count, solved);
            }
        }
    }

    __aicore__ inline void Process()
    {
        FindActiveRange();
        const uint32_t panels = upper_ ? 1U + (n_ - firstPanelCount_ + TILE - 1U) / TILE : (n_ + TILE - 1U) / TILE;
        for (uint32_t target = 0; target < panels; ++target) {
            const uint32_t begin = Start(target), count = Count(target);
            const uint32_t firstPivot = upper_ ? begin + count - 1U : begin;
            if (FinishNanSuffix(firstPivot)) {
                Store();
                return;
            }
            if (ProbePanel(target, begin, firstPivot)) {
                Store();
                return;
            }
            UpdatePanel(target, begin, count);
            LoadTile(target, target);
            for (uint32_t ordinal = 0; ordinal < count; ++ordinal) {
                const uint32_t pivot = upper_ ? count - 1U - ordinal : ordinal;
                const auto solved = ReadX(begin + pivot);
                if (FinishNanSuffix(begin + pivot)) {
                    Store();
                    return;
                }
                if constexpr (!TRANSPOSE) {
                    if (solved.real == 0.0f && solved.imag == 0.0f)
                        continue;
                }
                const uint32_t first = upper_ ? 0U : pivot + 1U;
                const uint32_t last = upper_ ? pivot : count;
                UpdateOrderedRows(begin, pivot, first, last, solved);
            }
        }
        Store();
    }
};
} // namespace

template <bool TRANSPOSE, bool CONJUGATE>
__global__ __aicore__ __vector__ void ctrsv_kernel_ordered_unit(
    GM_ADDR matrix, GM_ADDR vector, uint32_t n, int32_t uplo)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIC {
        return;
    }
    CtrsvOrderedUnitKernel<TRANSPOSE, CONJUGATE> op;
    op.Init(matrix, vector, n, uplo);
    op.Process();
}

void ctrsv_ordered_unit_do(uint8_t* matrix, uint8_t* vector, uint32_t n, int32_t uplo, int32_t trans, void* stream)
{
    if (trans == ACLBLAS_OP_N) {
        ctrsv_kernel_ordered_unit<false, false><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else if (trans == ACLBLAS_OP_T) {
        ctrsv_kernel_ordered_unit<true, false><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    } else {
        ctrsv_kernel_ordered_unit<true, true><<<1, nullptr, stream>>>(matrix, vector, n, uplo);
    }
}
