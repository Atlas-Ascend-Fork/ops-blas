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

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "fill.h"
#include "verify.h"

// ─────────────────────────────────────────────────────────────────────────────
// Complex banded / strided generation for ctbmv.
//   aclblasComplex = {float real; float imag}. We generate the banded matrix / strided
//   vector from two independent real-valued fills with the SAME value range but INDEPENDENT
//   seeds (real = seed, imag = seed + 1000), matching the fill.h::makeBlasComplexMatrix
//   convention. The banded layout returned by makeBlasBanded matches the ctbmv storage:
//     UPPER: A(i,j) at band[k + i - j + j*lda]; LOWER: A(i,j) at band[i - j + j*lda].
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<aclblasComplex> makeCtbmvComplexBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> realPart = makeBlasBanded(m, n, kl, ku, lda, fill, seed);
    std::vector<float> imagPart = makeBlasBanded(m, n, kl, ku, lda, fill, seed + 1000U);
    std::vector<aclblasComplex> out(realPart.size());
    for (size_t i = 0; i < realPart.size(); ++i) {
        out[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return out;
}

inline std::vector<aclblasComplex> makeCtbmvComplexBanded(
    int m, int n, int kl, int ku, int lda, const std::string& fill, uint32_t seed)
{
    // Same split as fill.h: a string literal cannot reach BlasFillMode in one
    // user-defined conversion, so keep both overloads.
    return makeCtbmvComplexBanded(m, n, kl, ku, lda, BlasFillMode(fill), seed);
}

inline std::vector<aclblasComplex> makeCtbmvComplexStrided(
    int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> realPart = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> imagPart = makeBlasStrided(count, inc, fill, seed + 1000U);
    std::vector<aclblasComplex> out(realPart.size());
    for (size_t i = 0; i < realPart.size(); ++i) {
        out[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return out;
}

inline std::vector<aclblasComplex> makeCtbmvComplexStrided(
    int count, int inc, const std::string& fill, uint32_t seed)
{
    return makeCtbmvComplexStrided(count, inc, BlasFillMode(fill), seed);
}

// ─────────────────────────────────────────────────────────────────────────────
// complex64 accuracy verification (real / imag separately, FLOAT32 tolerance).
//   Per the task: atol = 2^-16, rtol = 2^-10, required_matched_ratio = 0.99,
//   max_abs_error_limit = 1e-2 or 32*ULP. We reuse the framework's MIXED_TOLERANCE
//   (applyMixedTolerance(cfg, ACL_FLOAT, ...)) and call Verifier::verifyVector on
//   the real and imaginary parts. Matching NaN/Inf pairs (both NaN, or both same-sign
//   Inf) are neutralised beforehand so that the task-sanctioned Inf/NaN scenarios do
//   not trip the framework's finite-only tolerance check.
// ─────────────────────────────────────────────────────────────────────────────

inline bool VerifyCtbmvComplexVector(
    const std::vector<aclblasComplex>& output, const std::vector<aclblasComplex>& golden, int n, int incx,
    const std::string& caseId)
{
    if (n == 0) {
        return true;
    }
    const int absInc = std::abs(incx);
    // output / golden are strided host buffers (stride = incx). Logical element i sits at
    //   incx > 0 : i * incx ; incx < 0 : (n - 1 - i) * absInc  (Netlib ctbmv convention).
    std::vector<float> outRe(n), outIm(n), goldRe(n), goldIm(n);
    for (int i = 0; i < n; ++i) {
        const int bufIdx = (incx > 0) ? i * incx : (n - 1 - i) * absInc;
        aclblasComplex o = output[bufIdx];
        aclblasComplex g = golden[bufIdx];
        // Neutralise matching special values so they do not fail the finite-only check.
        bool bothNan = std::isnan(o.real) && std::isnan(g.real);
        bool bothInf = std::isinf(o.real) && std::isinf(g.real) && (o.real == g.real);
        outRe[i] = (bothNan || bothInf) ? 0.0f : o.real;
        goldRe[i] = (bothNan || bothInf) ? 0.0f : g.real;
        bothNan = std::isnan(o.imag) && std::isnan(g.imag);
        bothInf = std::isinf(o.imag) && std::isinf(g.imag) && (o.imag == g.imag);
        outIm[i] = (bothNan || bothInf) ? 0.0f : o.imag;
        goldIm[i] = (bothNan || bothInf) ? 0.0f : g.imag;
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldRe.data(), static_cast<size_t>(n));
    bool realOk = Verifier::verifyVector(outRe.data(), goldRe.data(), static_cast<size_t>(n), 1, cfg, caseId + "_real");
    applyMixedTolerance(cfg, ACL_FLOAT, goldIm.data(), static_cast<size_t>(n));
    bool imagOk = Verifier::verifyVector(outIm.data(), goldIm.data(), static_cast<size_t>(n), 1, cfg, caseId + "_imag");
    return realOk && imagOk;
}

