/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Independent Csyrkx Netlib reference and acceptance rules. */
#pragma once

#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cblas.h>

#include "csyrkx_param.h"

namespace csyrkx_test {
// Scale only the selected triangle; beta=1 preserves every original bit.
inline void ScaleGoldenColumns(const Param& p, std::vector<aclblasComplex>& result, int first, int columns)
{
    if (One(p.beta)) {
        return;
    }
    const float br = Component(p.beta, 0);
    const float bi = Component(p.beta, 1);
    for (int j = 0; j < columns; ++j) {
        for (int i = 0; i < p.n; ++i) {
            if (!p.Selected(i, first + j)) {
                continue;
            }
            auto& z = result[static_cast<size_t>(j) * p.ldc + i];
            if (Zero(p.beta)) {
                z = {0.0f, 0.0f};
                continue;
            }
            const float cr = Component(z, 0);
            const float ci = Component(z, 1);
            z = {br * cr - bi * ci, br * ci + bi * cr};
        }
    }
}

// Bounded-column reference: same Netlib single-precision CGEMM, without a
// second n*n host matrix. This also keeps the largest supplied cases <512 MiB.
inline std::vector<aclblasComplex> GoldenColumns(
    const Param& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    const std::vector<aclblasComplex>& original, int first, int columns)
{
    const size_t count = static_cast<size_t>(p.ldc) * columns;
    std::vector<aclblasComplex> result(count);
    const auto* sourceBytes =
        reinterpret_cast<const unsigned char*>(original.data() + static_cast<size_t>(first) * p.ldc);
    std::copy_n(sourceBytes, count * sizeof(aclblasComplex), reinterpret_cast<unsigned char*>(result.data()));
    if (p.k == 0 || Zero(p.alpha)) {
        // No reference to A/B; neither alpha*0 (possibly NaN) nor beta*oldC
        // when beta=0. Preserve beta=1 exactly, including NaN payloads/-0.
        ScaleGoldenColumns(p, result, first, columns);
        return result;
    }
    const bool normal = p.trans == ACLBLAS_OP_N;
    const auto* bSlice = b.data() + (normal ? static_cast<size_t>(first) : static_cast<size_t>(first) * p.ldb);
    // Compute only the row envelope intersecting the requested triangle in
    // this column tile. No retained dot product or K order changes; columns
    // retain the original lda/ldb/ldc, including the lower-row C offset.
    const int rowStart = p.uplo == ACLBLAS_UPPER ? 0 : first;
    const int rows = p.uplo == ACLBLAS_UPPER ? std::min(p.n, first + columns) : p.n - first;
    const auto* aSlice = a.data() + (normal ? static_cast<size_t>(rowStart) : static_cast<size_t>(rowStart) * p.lda);
    // OP_C is deliberately Trans, NEVER ConjTrans.
    cblas_cgemm(
        CblasColMajor, normal ? CblasNoTrans : CblasTrans, normal ? CblasTrans : CblasNoTrans, rows, columns, p.k,
        &p.alpha, aSlice, p.lda, bSlice, p.ldb, &p.beta, result.data() + rowStart, p.ldc);
    // The dense GEMM reference updates both triangles; restore ignored cells
    // from the exact original bits. Padding was not touched by CGEMM.
    for (int j = 0; j < columns; ++j)
        for (int i = 0; i < p.n; ++i) {
            if (!p.Selected(i, first + j)) {
                const size_t dst = static_cast<size_t>(j) * p.ldc + i;
                const size_t src = static_cast<size_t>(first + j) * p.ldc + i;
                const auto* sourceBytes = reinterpret_cast<const unsigned char*>(&original[src]);
                std::copy_n(sourceBytes, sizeof(aclblasComplex), reinterpret_cast<unsigned char*>(&result[dst]));
            }
        }
    return result;
}

inline bool NonFiniteEquivalent(float actual, float golden)
{
    if (std::isfinite(actual) || std::isfinite(golden))
        return false;
    if (std::isnan(actual) || std::isnan(golden))
        return true;
    return std::signbit(actual) == std::signbit(golden);
}

inline double Ulp(float value)
{
    const float magnitude = std::fabs(value);
    const float next = std::nextafter(magnitude, std::numeric_limits<float>::infinity());
    // At FLT_MAX nextafter is Inf; do not make every error acceptable.
    if (std::isinf(next))
        return static_cast<double>(magnitude) - std::nextafter(magnitude, 0.0f);
    return static_cast<double>(next) - magnitude;
}

struct Metrics {
    uint64_t finite = 0, matched = 0, nonfinite = 0, nonfiniteMatched = 0, hardFailures = 0;
    double maxAbs = 0.0, maxUlp = 0.0, maxPointUlp = 0.0;
    float maxPointActual = 0.0f, maxPointGolden = 0.0f;
    int maxPointRow = -1, maxPointCol = -1, maxPointComponent = -1;
    std::string firstFailure;

    void Add(float actual, float golden, int row, int col, int component)
    {
        bool failure = false;
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++nonfinite;
            if (NonFiniteEquivalent(actual, golden))
                ++nonfiniteMatched;
            else
                failure = true;
        } else {
            ++finite;
            const double error = std::abs(static_cast<double>(actual) - golden);
            const double ulp = Ulp(golden);
            // Official standard 2.1.2: ULP of the golden AT THE MAX-ABS-ERROR
            // POINT, not the largest golden and not an independent per-point
            // hard gate. Ties select the smallest ULP (conservative and stable).
            if (finite == 1 || error > maxAbs || (error == maxAbs && ulp < maxPointUlp)) {
                maxAbs = error;
                maxPointUlp = ulp;
                maxPointActual = actual;
                maxPointGolden = golden;
                maxPointRow = row;
                maxPointCol = col;
                maxPointComponent = component;
            }
            // Ulp() is positive for every finite FP32 value; guard the diagnostic
            // ratio so no zero divisor can reach it.
            if (ulp > 0.0) {
                maxUlp = std::max(maxUlp, error / ulp);
            }
            if (error <= 0x1p-13 + 0x1p-13 * std::abs(static_cast<double>(golden)))
                ++matched;
            // Retain the stricter per-point count for diagnosis ONLY.
            if (!(error <= 0.01 || error <= 32.0 * ulp)) {
                ++hardFailures;
                failure = true;
            }
        }
        if (failure && firstFailure.empty()) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<float>::max_digits10) << "row=" << row << " col=" << col
                    << " component=" << component << " actual=" << actual << " golden=" << golden;
            firstFailure = message.str();
        }
    }
    double MaxAbsLimit() const { return std::max(0.01, 32.0 * maxPointUlp); }
    bool Passed() const
    {
        return (finite == 0 || static_cast<double>(matched) / finite >= 0.99) && maxAbs <= MaxAbsLimit() &&
               nonfinite == nonfiniteMatched;
    }
    std::string Describe() const
    {
        std::ostringstream out;
        out << std::setprecision(std::numeric_limits<float>::max_digits10) << "finite=" << finite
            << " matched=" << matched << " ratio=" << (finite ? static_cast<double>(matched) / finite : 1.0)
            << " max_abs=" << maxAbs << " max_abs_limit=" << MaxAbsLimit() << " max_ulp=" << maxUlp
            << " max_point_row=" << maxPointRow << " max_point_col=" << maxPointCol
            << " max_point_component=" << maxPointComponent << " max_point_actual=" << maxPointActual
            << " max_point_golden=" << maxPointGolden << " max_point_golden_ulp=" << maxPointUlp
            << " stricter_diagnostic_hard_failures=" << hardFailures << " nonfinite=" << nonfinite
            << " nonfinite_matched=" << nonfiniteMatched << " first_stricter_or_nonfinite_failure=" << firstFailure;
        return out.str();
    }
};
} // namespace csyrkx_test
