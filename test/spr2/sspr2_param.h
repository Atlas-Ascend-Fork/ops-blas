/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPR2_PARAM_H
#define SSPR2_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// ── sspr2 extended test vocabulary (below) ─────────────────────────────────
// The 19-column CSV case descriptor, the fill-mark driven data
// generators, and the status/check name helpers used by the arch-specific
// sspr2_test.cpp sources. Enum values follow cann_ops_blas_common.h.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace sspr2_st {

// Status name for evidence-line printing (values per cann_ops_blas_common.h).
inline const char* StatusName(aclblasStatus_t s)
{
    switch (s) {
        case ACLBLAS_STATUS_SUCCESS: return "ACLBLAS_STATUS_SUCCESS";
        case ACLBLAS_STATUS_NOT_INITIALIZED: return "ACLBLAS_STATUS_NOT_INITIALIZED";
        case ACLBLAS_STATUS_ALLOC_FAILED: return "ACLBLAS_STATUS_ALLOC_FAILED";
        case ACLBLAS_STATUS_INVALID_VALUE: return "ACLBLAS_STATUS_INVALID_VALUE";
        case ACLBLAS_STATUS_MAPPING_ERROR: return "ACLBLAS_STATUS_MAPPING_ERROR";
        case ACLBLAS_STATUS_EXECUTION_FAILED: return "ACLBLAS_STATUS_EXECUTION_FAILED";
        case ACLBLAS_STATUS_INTERNAL_ERROR: return "ACLBLAS_STATUS_INTERNAL_ERROR";
        case ACLBLAS_STATUS_NOT_SUPPORTED: return "ACLBLAS_STATUS_NOT_SUPPORTED";
        case ACLBLAS_STATUS_ARCH_MISMATCH: return "ACLBLAS_STATUS_ARCH_MISMATCH";
        case ACLBLAS_STATUS_HANDLE_IS_NULLPTR: return "ACLBLAS_STATUS_HANDLE_IS_NULLPTR";
        case ACLBLAS_STATUS_INVALID_ENUM: return "ACLBLAS_STATUS_INVALID_ENUM";
        case ACLBLAS_STATUS_UNKNOWN: return "ACLBLAS_STATUS_UNKNOWN";
        default: return "ACLBLAS_STATUS_UNKNOWN";
    }
}

// ── 19-column CSV shared case descriptor ────────────────────────────
// Columns:
//   case_id,level,uplo,n,alpha,x_fill,incx,y_fill,incy,ap_fill,
//   expect_result,check,rtol,atol,matched_ratio,max_abs_limit,seed,spec_ref,description
// check in {GOLDEN, AP_UNCHANGED, NAN_PROPAGATE, NO_COMPUTE, STATUS_ONLY, FRAME_GUARD}
//
// FRAME_GUARD: the raw-pointer C API has no length argument, so a
// declared-length-vs-expected-length mismatch is detected at the test-loader
// layer (the case passes without invoking the operator).

enum class CheckKind { GOLDEN, AP_UNCHANGED, NAN_PROPAGATE, NO_COMPUTE, STATUS_ONLY, FRAME_GUARD, UNKNOWN };

inline CheckKind ParseCheck(const std::string& s)
{
    if (s == "GOLDEN") return CheckKind::GOLDEN;
    if (s == "AP_UNCHANGED") return CheckKind::AP_UNCHANGED;
    if (s == "NAN_PROPAGATE") return CheckKind::NAN_PROPAGATE;
    if (s == "NO_COMPUTE") return CheckKind::NO_COMPUTE;
    if (s == "STATUS_ONLY") return CheckKind::STATUS_ONLY;
    if (s == "FRAME_GUARD") return CheckKind::FRAME_GUARD;
    return CheckKind::UNKNOWN;
}

struct Sspr2Case {
    std::string caseId;
    std::string level;
    int uplo = 121;
    int n = 0;
    bool alphaNull = false;
    float alpha = 1.0f;
    std::string xFill;
    int incx = 1;
    std::string yFill;
    int incy = 1;
    std::string apFill;
    std::string expectResult;
    CheckKind check = CheckKind::GOLDEN;
    double rtol = 9.765625e-4;
    double atol = 1.52587890625e-5;
    double matchedRatio = 0.99;
    std::string maxAbsLimit;
    unsigned seed = 0;
    std::string specRef;
    std::string description;

    std::size_t ApLen() const { return (n > 0) ? static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) / 2 : 0; }
    std::size_t XLen() const { return (n > 0 && incx != 0) ? static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(incx < 0 ? -incx : incx) + 1 : 0; }
    std::size_t YLen() const { return (n > 0 && incy != 0) ? static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(incy < 0 ? -incy : incy) + 1 : 0; }
};

// ── fill mark -> test data generation ──────────────────────────────────────
// Vocabulary aligned with test/frame/fill.h:
//   RANDOM_NORM_5_5 : uniform in [-5,5] (default; mt19937 + uniform_real_distribution)
//   RANDOM_GAUSS    : normal distribution (not yet provided by the shared
//                     generators; falls back to uniform [-5,5])
//   VALUE_NORM_0    : all zeros
//   RANDOM_ALTER    : alternating fill (i%2==0 ? i+1 : -(i+1))
//   RANDOM_EXTREME  : extreme-value cycle {1, 0, -1, FLT_MAX, FLT_MIN, -FLT_MAX, denorm_min}
//   VALUE_NORM_INF  : +Inf
//   VALUE_NORM_NAN  : NaN
//   NULLPTR         : null pointer (empty vector; converted by the caller)
// Stride semantics match makeBlasStrided: logical element i lives at
// x[i*inc] (inc>0) or x[(n-1-i)*|inc|] (inc<0).

struct StridedFill {
    bool isNull = false;
    // kind: 0=uniform rand, 1=const zero, 2=alter, 3=extreme, 4=const inf, 5=const nan
    int kind = 0;

    static StridedFill Parse(const std::string& mark)
    {
        StridedFill f;
        if (mark == "NULLPTR") {
            f.isNull = true;
            return f;
        }
        if (mark == "VALUE_NORM_0") f.kind = 1;
        else if (mark == "RANDOM_ALTER") f.kind = 2;
        else if (mark == "RANDOM_EXTREME") f.kind = 3;
        else if (mark == "VALUE_NORM_INF") f.kind = 4;
        else if (mark == "VALUE_NORM_NAN") f.kind = 5;
        else f.kind = 0; // RANDOM_NORM_5_5 / RANDOM_GAUSS (uniform fallback) / empty
        return f;
    }
};

// Extreme-value cycle shared with the frame generators.
inline float ExtremeAt(std::size_t i)
{
    static const float kVals[] = {
        1.0f, 0.0f, -1.0f, FLT_MAX, FLT_MIN, -FLT_MAX, std::numeric_limits<float>::denorm_min()};
    return kVals[i % 7];
}

// Strided vector: count logical elements, stride inc (may be negative),
// physical length 1+(count-1)*|inc|. NULLPTR yields an empty vector.
inline std::vector<float> MakeStrided(int count, int inc, const StridedFill& fill, unsigned seed)
{
    const int absInc = inc < 0 ? -inc : inc;
    const std::size_t size = (count > 0 && absInc > 0) ? static_cast<std::size_t>(count - 1) * static_cast<std::size_t>(absInc) + 1 : 0;
    std::vector<float> data(size, 0.0f);
    if (fill.isNull) {
        data.clear();
        return data;
    }
    if (size == 0) {
        return data;
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (int i = 0; i < count; i++) {
        const std::size_t idx = (inc > 0) ? static_cast<std::size_t>(i) * static_cast<std::size_t>(inc)
                                          : static_cast<std::size_t>(count - 1 - i) * static_cast<std::size_t>(absInc);
        float v = 0.0f;
        switch (fill.kind) {
            case 1: v = 0.0f; break;
            case 2: v = (i % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1); break;
            case 3: v = ExtremeAt(static_cast<std::size_t>(i)); break;
            case 4: v = INFINITY; break;
            case 5: v = NAN; break;
            default: v = dist(rng); break;
        }
        data[idx] = v;
    }
    return data;
}

// Packed AP (length n*(n+1)/2, sequential fill; empty for n<=0).
inline std::vector<float> MakePacked(int n, const StridedFill& fill, unsigned seed)
{
    const std::size_t apLen = (n > 0) ? static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1) / 2 : 0;
    std::vector<float> data(apLen, 0.0f);
    if (fill.isNull) {
        data.clear();
        return data;
    }
    if (apLen == 0) {
        return data;
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (std::size_t k = 0; k < apLen; k++) {
        float v = 0.0f;
        switch (fill.kind) {
            case 1: v = 0.0f; break;
            case 2: v = (k % 2 == 0) ? static_cast<float>(k + 1) : -static_cast<float>(k + 1); break;
            case 3: v = ExtremeAt(k); break;
            case 4: v = INFINITY; break;
            case 5: v = NAN; break;
            default: v = dist(rng); break;
        }
        data[k] = v;
    }
    return data;
}

} // namespace sspr2_st

// ── 19-column CSV -> BlasTestParamBase parameter loading ────────────
struct Sspr2Param : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alpha = 1.0f;
    bool alphaNull = false;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;
    BlasFillMode ap = parseFill("RANDOM");

    // Extended 19-column case descriptor (see sspr2_st::Sspr2Case above).
    sspr2_st::Sspr2Case core;

    Sspr2Param(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        alphaNull = (alphaStr == "NULLPTR");
        if (!alphaNull) {
            alpha = parseFloat(alphaStr, 1.0f);
        }
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        ap = parseFill(ReadMap(map, "ap", "RANDOM"));

        // ── extended 19-column parsing (absent columns keep the
        //    defaults, so 13-column rows parse unchanged) ──
        core.caseId = ReadMap(map, "case_id", "");
        if (core.caseId.empty()) {
            core.caseId = ReadMap(map, "case_name");
        } else {
            // GTest (>= 1.10) parameter names allow only alphanumerics and
            // underscores; evidence lines still print the original case_id.
            caseName = core.caseId;
            std::replace(caseName.begin(), caseName.end(), '-', '_');
        }
        core.level = ReadMap(map, "level");
        // uplo aliases: ACLBLAS_FILL_MODE_* / ACLBLAS_* / numeric
        const std::string uploStr = ReadMap(map, "uplo", "121");
        if (uploStr == "ACLBLAS_FILL_MODE_UPPER" || uploStr == "ACLBLAS_UPPER" || uploStr == "UPPER" || uploStr == "121") {
            core.uplo = 121;
        } else if (uploStr == "ACLBLAS_FILL_MODE_LOWER" || uploStr == "ACLBLAS_LOWER" || uploStr == "LOWER" || uploStr == "122") {
            core.uplo = 122;
        } else {
            core.uplo = parseInt(uploStr, -1); // invalid enum values pass through
        }
        core.n = parseInt(ReadMap(map, "n", "0"));
        core.alphaNull = (alphaStr == "null" || alphaStr == "NULLPTR");
        core.alpha = core.alphaNull ? 0.0f : alpha;
        core.xFill = ReadMap(map, "x_fill", "RANDOM_NORM_5_5");
        core.incx = parseInt(ReadMap(map, "incx", "1"));
        core.yFill = ReadMap(map, "y_fill", "RANDOM_NORM_5_5");
        core.incy = parseInt(ReadMap(map, "incy", "1"));
        core.apFill = ReadMap(map, "ap_fill", "RANDOM_NORM_5_5");
        core.expectResult = ReadMap(map, "expect_result", "ACLBLAS_STATUS_SUCCESS");
        expectResult = parseStatus(core.expectResult);
        core.check = sspr2_st::ParseCheck(ReadMap(map, "check", "GOLDEN"));
        core.rtol = parseDouble(ReadMap(map, "rtol", "9.765625e-4"), 9.765625e-4);
        core.atol = parseDouble(ReadMap(map, "atol", "1.52587890625e-5"), 1.52587890625e-5);
        core.matchedRatio = parseDouble(ReadMap(map, "matched_ratio", "0.99"), 0.99);
        core.maxAbsLimit = ReadMap(map, "max_abs_limit", "");
        core.seed = parseUint(ReadMap(map, "seed", ReadMap(map, "random_seed", "0")));
        randomSeed = core.seed;
        core.specRef = ReadMap(map, "spec_ref", "");
        description = ReadMap(map, "description");
    }
};

#endif // SSPR2_PARAM_H
