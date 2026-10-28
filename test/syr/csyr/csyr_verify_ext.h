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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "verify.h" // getUlpsAt

// ─────────────────────────────────────────────────────────────────────────────
// Shared csyr ST verification primitives:
//   - CsyrToleranceGate: per real/imag-component precision accumulator mirroring
//     the frame MixedToleranceStrategy (spec gate) + MereMareStrategy (CSV
//     engineering gate) semantics, with NaN==NaN counted as matched;
//   - untouched-triangle / whole-storage bitwise comparators for the G3
//     snapshot assertion and the quick-return no-write check.
// Kept gtest-free so a plain CPU driver can reuse and cross-check them.
// ─────────────────────────────────────────────────────────────────────────────

namespace csyr_verify {

constexpr double kDefaultMereThreshold = 0.0001220703125; // 2^-13
constexpr double kDefaultMareMultiplier = 10.0;
constexpr double kMereEpsilon = 0.00006103515625; // 2^-14, same as frame verify.h

struct CsyrToleranceGate {
    // spec gates (ACL_FLOAT defaults == applyMixedTolerance(ACL_FLOAT))
    double atol = 1.52587890625e-5; // 2^-16
    double rtol = 0.0009765625;     // 2^-10
    double fixedLimit = 1e-2;
    double requiredRatio = 0.99;
    int mantissaBits = 23;
    int emin = -126;
    // engineering gate (CSV columns)
    double mereThreshold = kDefaultMereThreshold;
    double mareMultiplier = kDefaultMareMultiplier;

    size_t total = 0;
    size_t mixedFail = 0;
    double maxAbsErr = 0.0;
    bool limitFailed = false;
    size_t mmMismatch = 0;
    size_t mmValid = 0;
    double sumRelErr = 0.0;
    double maxRelErr = 0.0;

    void Feed(float outVal, float goldVal)
    {
        total++;
        // matched: bitwise equal (covers +0/-0, +Inf/+Inf) or both NaN (NaN judged equal)
        if (outVal == goldVal || (std::isnan(outVal) && std::isnan(goldVal))) {
            return;
        }
        const double diff = std::abs(static_cast<double>(outVal) - static_cast<double>(goldVal));
        const double goldAbs = std::abs(static_cast<double>(goldVal));
        if (std::isnan(outVal) || std::isnan(goldVal) || std::isinf(outVal) || std::isinf(goldVal)) {
            mixedFail++;
            limitFailed = true;
            mmMismatch++;
            return;
        }
        if (diff > maxAbsErr) {
            maxAbsErr = diff;
        }
        const double tolLimit = atol + rtol * goldAbs;
        const double ulpLimit = std::max(fixedLimit, 32.0 * getUlpsAt(goldAbs, mantissaBits, emin));
        if (diff > tolLimit) {
            mixedFail++;
        }
        if (diff > ulpLimit) {
            limitFailed = true;
        }
        if (goldAbs < mereThreshold) {
            return; // below the noise floor, excluded from MERE/MARE (frame semantics)
        }
        const double relErr = diff / (goldAbs + kMereEpsilon);
        sumRelErr += relErr;
        if (relErr > maxRelErr) {
            maxRelErr = relErr;
        }
        mmValid++;
    }

    double MatchedRatio() const
    {
        return (total == 0) ? 1.0 : static_cast<double>(total - mixedFail) / static_cast<double>(total);
    }
    double Mere() const { return (mmValid == 0) ? 0.0 : sumRelErr / static_cast<double>(mmValid); }
    bool MixedPass() const { return MatchedRatio() >= requiredRatio && !limitFailed; }
    bool MerePass() const
    {
        // Dual gate aligned with the frame MereMareStrategy (verify.h): both the mean
        // (MERE < threshold) and the max (MARE < threshold * multiplier) must hold.
        // Near-zero golden elements are already excluded above the noise floor in
        // Feed(), so relative errors entering this gate are well-defined.
        return mmMismatch == 0 && Mere() < mereThreshold && maxRelErr < mereThreshold * mareMultiplier;
    }
    bool Pass() const { return MixedPass() && MerePass(); }

    void Report(const std::string& caseId, const char* tag) const
    {
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "[" << caseId << "] " << tag << ": matchedRatio=" << MatchedRatio() << " (req=" << requiredRatio
                  << ")"
                  << " maxAbsErr=" << maxAbsErr << " limitFailed=" << (limitFailed ? 1 : 0) << " mere=" << Mere()
                  << " (thr=" << mereThreshold << ") mare=" << maxRelErr << " (lim=" << mereThreshold * mareMultiplier
                  << ") fail=" << mixedFail << "/" << total << " => " << (Pass() ? "PASS" : "FAIL") << std::endl;
        std::cout << std::fixed << std::setprecision(6);
    }
};

inline uint32_t FloatBits(float v)
{
    uint32_t bits = 0;
    const unsigned char* src = reinterpret_cast<const unsigned char*>(&v);
    unsigned char* dst = reinterpret_cast<unsigned char*>(&bits);
    std::copy(src, src + sizeof(bits), dst);
    return bits;
}

inline const char* ClassifyBitDiff(uint32_t before, uint32_t after)
{
    const float fBefore = *reinterpret_cast<const float*>(&before);
    const float fAfter = *reinterpret_cast<const float*>(&after);
    if (std::isnan(fBefore) && std::isnan(fAfter)) {
        return "nan-payload";
    }
    if ((fBefore == 0.0f) && (fAfter == 0.0f)) {
        return "zero-sign-flip";
    }
    return "value";
}

struct BitMismatchInfo {
    int col = 0;
    int row = 0;
    uint32_t beforeRe = 0;
    uint32_t afterRe = 0;
    uint32_t beforeIm = 0;
    uint32_t afterIm = 0;
};

inline bool IsElementReferenced(aclblasFillMode_t uplo, int n, int col, int row)
{
    return (row < n) && ((uplo == ACLBLAS_UPPER) ? (row <= col) : (row >= col));
}

// Bitwise comparison of one complex element: true when either component differs.
inline bool ElementBitsDiffer(const aclblasComplex& lhs, const aclblasComplex& rhs)
{
    return FloatBits(lhs.real) != FloatBits(rhs.real) || FloatBits(lhs.imag) != FloatBits(rhs.imag);
}

// Checks one untouched element (col=j, row=i): returns 1 and records a report
// (bounded by maxReports) when the element differs bitwise, else 0.
inline size_t CheckUntouchedElement(int j, int i, int lda, const std::vector<aclblasComplex>& before,
    const std::vector<aclblasComplex>& after, size_t maxReports, std::vector<BitMismatchInfo>* reports)
{
    const size_t e = static_cast<size_t>(j) * lda + i;
    if (!ElementBitsDiffer(before[e], after[e])) {
        return 0;
    }
    if (reports != nullptr && reports->size() < maxReports) {
        reports->push_back(BitMismatchInfo{j, i, FloatBits(before[e].real), FloatBits(after[e].real),
            FloatBits(before[e].imag), FloatBits(after[e].imag)});
    }
    return 1;
}

// Scans the untouched elements of column j (padding rows included).
inline size_t CountUntouchedColumnMismatch(aclblasFillMode_t uplo, int n, int lda, int j,
    const std::vector<aclblasComplex>& before, const std::vector<aclblasComplex>& after, size_t maxReports,
    std::vector<BitMismatchInfo>* reports)
{
    size_t mismatch = 0;
    for (int i = 0; i < lda; i++) {
        if (IsElementReferenced(uplo, n, j, i)) {
            continue;
        }
        mismatch += CheckUntouchedElement(j, i, lda, before, after, maxReports, reports);
    }
    return mismatch;
}

// G3 snapshot: elements NOT referenced by uplo (padding rows included) must be
// bit-identical (不读不写). Returns the total mismatch count, keeps at most
// `maxReports` first mismatches in `reports` (may be nullptr).
inline size_t CountUntouchedBitMismatch(aclblasFillMode_t uplo, int n, int lda,
    const std::vector<aclblasComplex>& before, const std::vector<aclblasComplex>& after, size_t maxReports,
    std::vector<BitMismatchInfo>* reports)
{
    size_t mismatch = 0;
    for (int j = 0; j < n; j++) {
        mismatch += CountUntouchedColumnMismatch(uplo, n, lda, j, before, after, maxReports, reports);
    }
    return mismatch;
}

// Quick return (n=0 / alpha=(0,0)): the whole storage must be bit-identical.
inline size_t CountWholeBitMismatch(
    const std::vector<aclblasComplex>& before, const std::vector<aclblasComplex>& after)
{
    size_t mismatch = 0;
    const size_t len = std::min(before.size(), after.size());
    for (size_t e = 0; e < len; e++) {
        if (FloatBits(before[e].real) != FloatBits(after[e].real) ||
            FloatBits(before[e].imag) != FloatBits(after[e].imag)) {
            mismatch++;
        }
    }
    return mismatch + (before.size() - len) + (after.size() - len);
}

} // namespace csyr_verify
