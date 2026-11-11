/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Finite-row reconstruction preserves Netlib 3.10.0 ZHBMV update order.
#pragma once
#include "finite_soft64.hpp"

namespace chbmv_numeric_range {
struct FloatComplex {
    float real;
    float imag;
};

CHBMV_SOFT_INLINE bool Finite(FloatComplex x) { return Finite(x.real) && Finite(x.imag); }
CHBMV_SOFT_INLINE FiniteComplex64 Wide(FloatComplex x) { return FiniteComplex64::FromFloat(x.real, x.imag); }
CHBMV_SOFT_INLINE FiniteComplex64 Conjugate(const FiniteComplex64& x) { return {x.real, x.imag.Negate()}; }
CHBMV_SOFT_INLINE uint32_t Smaller(uint32_t a, uint32_t b) { return a < b ? a : b; }

struct FiniteRowBand {
    uint32_t declaredK;
    uint32_t lda;
    uint32_t row;
    uint32_t first;
    uint32_t last;
};

template <typename Access>
CHBMV_SOFT_INLINE bool RecomputeUpperFiniteRow(
    const Access& input, const FiniteRowBand& band, const FiniteComplex64& wideAlpha,
    const FiniteComplex64& diagonalTemp1, float diagonal, FiniteComplex64& value)
{
    const uint32_t declaredK = band.declaredK;
    const uint32_t lda = band.lda;
    const uint32_t row = band.row;
    const uint32_t first = band.first;
    const uint32_t last = band.last;
    // At Netlib column j=row, temp2 visits earlier i in ascending order.
    FiniteComplex64 temp2{};
    for (uint32_t i = first; i < row; ++i) {
        const FloatComplex av = input.LoadA(static_cast<int64_t>(row) * lda + declaredK + i - row);
        const FloatComplex xv = input.LoadX(i);
        if (!Finite(av) || !Finite(xv)) {
            return false;
        }
        temp2 = temp2.Add(Conjugate(Wide(av)).Multiply(Wide(xv)));
    }
    // Preserve left associativity: y + temp1*diag, then + alpha*temp2.
    value = value.Add(diagonalTemp1.MultiplyReal(diagonal));
    value = value.Add(wideAlpha.Multiply(temp2));
    // Subsequent columns scatter into this output row in increasing j.
    for (uint32_t j = row + 1U; j <= last; ++j) {
        const FloatComplex av = input.LoadA(static_cast<int64_t>(j) * lda + declaredK + row - j);
        const FloatComplex xv = input.LoadX(j);
        if (!Finite(av) || !Finite(xv)) {
            return false;
        }
        const FiniteComplex64 temp1 = wideAlpha.Multiply(Wide(xv));
        value = value.Add(temp1.Multiply(Wide(av)));
    }
    return true;
}

template <typename Access>
CHBMV_SOFT_INLINE bool RecomputeLowerFiniteRow(
    const Access& input, const FiniteRowBand& band, const FiniteComplex64& wideAlpha,
    const FiniteComplex64& diagonalTemp1, float diagonal, FiniteComplex64& value)
{
    const uint32_t lda = band.lda;
    const uint32_t row = band.row;
    const uint32_t first = band.first;
    const uint32_t last = band.last;
    // Earlier columns have already scattered into this row, in increasing j.
    for (uint32_t j = first; j < row; ++j) {
        const FloatComplex av = input.LoadA(static_cast<int64_t>(j) * lda + row - j);
        const FloatComplex xv = input.LoadX(j);
        if (!Finite(av) || !Finite(xv)) {
            return false;
        }
        const FiniteComplex64 temp1 = wideAlpha.Multiply(Wide(xv));
        value = value.Add(temp1.Multiply(Wide(av)));
    }
    value = value.Add(diagonalTemp1.MultiplyReal(diagonal));
    FiniteComplex64 temp2{};
    for (uint32_t i = row + 1U; i <= last; ++i) {
        const FloatComplex av = input.LoadA(static_cast<int64_t>(row) * lda + i - row);
        const FloatComplex xv = input.LoadX(i);
        if (!Finite(av) || !Finite(xv)) {
            return false;
        }
        temp2 = temp2.Add(Conjugate(Wide(av)).Multiply(Wide(xv)));
    }
    value = value.Add(wideAlpha.Multiply(temp2));
    return true;
}

// Access must provide:
// FloatComplex LoadA(int64_t bandElementIndex) const;
// float LoadDiagonal(int64_t bandElementIndex) const; // real component only
// FloatComplex LoadX(uint32_t logicalIndex) const;    // normalizes signed incx
// FloatComplex LoadY(uint32_t logicalIndex) const;    // normalizes signed incy
// The function performs no stores. On false, output references are unchanged;
// retain the separate IEEE path rather than converting Inf/NaN to finite64.
template <typename Access>
CHBMV_SOFT_INLINE bool RecomputeFiniteNetlibRow(
    const Access& input, uint32_t n, uint32_t declaredK, uint32_t lda, bool upper, uint32_t row, FloatComplex alpha,
    FloatComplex beta, float& outReal, float& outImag)
{
    if (!Finite(alpha) || !Finite(beta)) {
        return false;
    }
    FiniteComplex64 value{};
    // Netlib first forms beta*y; beta==0 never reads old y, beta==1 copies it.
    if (beta.real != 0.0f || beta.imag != 0.0f) {
        const FloatComplex oldY = input.LoadY(row);
        if (!Finite(oldY)) {
            return false;
        }
        value = beta.real == 1.0f && beta.imag == 0.0f ? Wide(oldY) : Wide(beta).Multiply(Wide(oldY));
    }
    if (alpha.real == 0.0f && alpha.imag == 0.0f) {
        outReal = value.real.ToFloat();
        outImag = value.imag.ToFloat();
        return true;
    }
    const uint32_t k = Smaller(declaredK, n - 1U);
    const uint32_t first = row > k ? row - k : 0U;
    const uint32_t last =
        static_cast<uint32_t>(static_cast<uint64_t>(row) + k < n ? static_cast<uint64_t>(row) + k : n - 1U);
    const FiniteComplex64 wideAlpha = Wide(alpha);
    const FloatComplex diagonalX = input.LoadX(row);
    const float diagonal = input.LoadDiagonal(static_cast<int64_t>(row) * lda + (upper ? declaredK : 0U));
    if (!Finite(diagonalX) || !Finite(diagonal)) {
        return false;
    }
    const FiniteComplex64 diagonalTemp1 = wideAlpha.Multiply(Wide(diagonalX));

    const FiniteRowBand band{declaredK, lda, row, first, last};
    const bool valid = upper ? RecomputeUpperFiniteRow(input, band, wideAlpha, diagonalTemp1, diagonal, value) :
                               RecomputeLowerFiniteRow(input, band, wideAlpha, diagonalTemp1, diagonal, value);
    if (!valid) {
        return false;
    }
    outReal = value.real.ToFloat();
    outImag = value.imag.ToFloat();
    return true;
}
} // namespace chbmv_numeric_range
