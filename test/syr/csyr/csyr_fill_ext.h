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
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "fill.h"

// ─────────────────────────────────────────────────────────────────────────────
// csyr ST fill extensions (csyr CSV token table).
//
// test/frame/fill.h only covers uniform/alter/extreme/special-value fills.
// The csyr fill extensions add:
//   RANDOM_GAUSS_5_2        normal distribution, mu ~ U(-5,5), sigma ~ U(0.1,2),
//                           real/imag parts sampled independently
//   VALUE_NEG_ZERO          all elements = (-0.0, -0.0), bitwise negative zero,
//                           exposes a "+0.0 atomic add" sign flip on the untouched
//                           triangle (G3 snapshot contract)
//   INJECT_NAN_ONE_ELEMENT  uniform [-5,5] with exactly one element set to
//                           (NaN, NaN) (extreme input, x column only)
//   SINGLE_POS_INF          uniform [-5,5] with exactly one element set to
//                           (+inf, +inf) (extreme input, x column only)
// Everything else delegates to test/frame/fill.h (RANDOM_NORM_5_5 / VALUE_NORM_0 /
// VALUE_NORM_INF / VALUE_NORM_NAN / RANDOM_ALTER / RANDOM_EXTREME / NULLPTR).
// NOTE: the token "VALUE_NEG_ZERO" would crash stock fill.h (parseNum hits
// std::stof("EG")), hence the csyr-local pre-classification below.
// ─────────────────────────────────────────────────────────────────────────────

enum class CsyrFillKind { FRAME, GAUSS, NEG_ZERO, INJECT_NAN, SINGLE_POS_INF, NULLPTR_TOKEN };

inline CsyrFillKind ClassifyCsyrFill(const std::string& token)
{
    if (token == "NULLPTR" || token == "NULL") {
        return CsyrFillKind::NULLPTR_TOKEN;
    }
    if (token.rfind("RANDOM_GAUSS", 0) == 0) {
        return CsyrFillKind::GAUSS;
    }
    if (token == "VALUE_NEG_ZERO") {
        return CsyrFillKind::NEG_ZERO;
    }
    if (token == "INJECT_NAN_ONE_ELEMENT") {
        return CsyrFillKind::INJECT_NAN;
    }
    if (token == "SINGLE_POS_INF") {
        return CsyrFillKind::SINGLE_POS_INF;
    }
    return CsyrFillKind::FRAME;
}

inline bool IsCsyrNullFill(const std::string& token)
{
    return ClassifyCsyrFill(token) == CsyrFillKind::NULLPTR_TOKEN;
}

// Pack a flat float stream (re, im interleaved) into aclblasComplex elements.
inline std::vector<aclblasComplex> PackCsyrComplex(const std::vector<float>& flat)
{
    const size_t n = flat.size() / 2;
    std::vector<aclblasComplex> dst(n);
    for (size_t i = 0; i < n; ++i) {
        dst[i].real = flat[2 * i];
        dst[i].imag = flat[2 * i + 1];
    }
    return dst;
}

// Generate `count` complex elements according to the fill token.
// seed drives the whole stream; real and imaginary parts are sampled
// independently.
inline std::vector<aclblasComplex> MakeCsyrComplexArray(int64_t count, const std::string& token, uint32_t seed)
{
    if (count <= 0) {
        return {};
    }
    const CsyrFillKind kind = ClassifyCsyrFill(token);
    if (kind == CsyrFillKind::NULLPTR_TOKEN) {
        return {};
    }
    if (kind == CsyrFillKind::FRAME) {
        return PackCsyrComplex(makeBlasArray(count * 2, token, seed));
    }

    std::vector<aclblasComplex> data(static_cast<size_t>(count));
    std::mt19937 rng(seed ? seed : 42);

    if (kind == CsyrFillKind::GAUSS) {
        // RANDOM_GAUSS_5_2: per component mu ~ U(-5,5), sigma ~ U(0.1,2), value ~ N(mu, sigma).
        // Real and imaginary components draw independent mu/sigma/value.
        std::uniform_real_distribution<float> muDist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> sigmaDist(0.1f, 2.0f);
        for (size_t i = 0; i < data.size(); ++i) {
            const float muRe = muDist(rng);
            const float sigmaRe = sigmaDist(rng);
            std::normal_distribution<float> reDist(muRe, sigmaRe);
            const float re = reDist(rng);
            const float muIm = muDist(rng);
            const float sigmaIm = sigmaDist(rng);
            std::normal_distribution<float> imDist(muIm, sigmaIm);
            const float im = imDist(rng);
            data[i] = aclblasComplex{re, im};
        }
        return data;
    }

    if (kind == CsyrFillKind::NEG_ZERO) {
        const float negZero = -0.0f;
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = aclblasComplex{negZero, negZero};
        }
        return data;
    }

    // Inject patterns: uniform [-5,5] base, then exactly one element carries the
    // special value (both components) so the golden/NPU propagation scope is precise.
    data = PackCsyrComplex(makeBlasArray(count * 2, "RANDOM_NORM_5_5", seed));
    const size_t kInject = static_cast<size_t>(count) / 2; // deterministic, mid-vector
    const float special = (kind == CsyrFillKind::INJECT_NAN) ? std::numeric_limits<float>::quiet_NaN() : INFINITY;
    data[kInject] = aclblasComplex{special, special};
    return data;
}
