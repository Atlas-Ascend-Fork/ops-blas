/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "csyr_fill_ext.h"

// 17-column CSV schema:
//   case_name, description, uplo, n, alpha_real, alpha_imag, x, incx, a, lda,
//   x_len, handle, untouched_check, expect_result, mere_threshold, mare_multiplier, random_seed
// The 14-column csyr_test.csv is a subset: the three new
// columns fall back to their defaults when absent (ReadMap default).
// Shared by both arch trees: the arch22 ST consumes the scalar/token fields
// below, the arch35 ST consumes alpha (composed complex), alphaOnDevice and
// the BlasFillMode x/a views of the same x/a columns.
struct CsyrParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    aclblasComplex alpha{1.0f, 0.0f};
    bool alphaNull = false;
    bool alphaOnDevice = false;
    std::string xToken = "RANDOM_NORM_5_5";
    std::string aToken = "RANDOM_NORM_5_5";
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    BlasFillMode a = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    int lda = 0;
    int xLen = 0;               // 0 = default logical length n (K-dim mismatch cases override)
    bool handleValid = true;    // handle column: VALID (default) / NULLPTR
    bool untouchedCheck = false; // G3 snapshot assertion switch
    bool isPerf = false;        // formal performance case: warmup 10 + sample 60
    bool isPerfSmoke = false;   // smoke performance case: warmup 5 + sample 10, no gate
    bool isDeterminism = false; // repeat the same call 3 times and compare

    CsyrParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
        alphaNull = (alphaRealStr == "null" || alphaRealStr == "NULLPTR" || alphaRealStr == "NULL");
        if (!alphaNull) {
            alphaReal = parseFloat(alphaRealStr, 1.0f);
            alphaImag = parseFloat(alphaImagStr, 0.0f);
        }
        alpha.real = alphaReal;
        alpha.imag = alphaImag;
        alphaOnDevice = (ReadMap(map, "alpha_on_device", "false") == "true");
        xToken = ReadMap(map, "x", "RANDOM_NORM_5_5");
        aToken = ReadMap(map, "a", "RANDOM_NORM_5_5");
        // BlasFillMode views for the arch35 ST, filled only for tokens the stock
        // frame parser accepts (see csyr_fill_ext.h); csyr extension tokens
        // (RANDOM_GAUSS_* / VALUE_NEG_ZERO / INJECT_NAN_ONE_ELEMENT / SINGLE_POS_INF)
        // stay on the arch22-only fill path (fill.h parsing would crash).
        if (ClassifyCsyrFill(xToken) == CsyrFillKind::FRAME ||
            ClassifyCsyrFill(xToken) == CsyrFillKind::NULLPTR_TOKEN) {
            x = parseFill(xToken);
        }
        if (ClassifyCsyrFill(aToken) == CsyrFillKind::FRAME ||
            ClassifyCsyrFill(aToken) == CsyrFillKind::NULLPTR_TOKEN) {
            a = parseFill(aToken);
        }
        incx = parseInt(ReadMap(map, "incx", "1"));
        // Negative-direction values (lda=0, n<0, incx=0) must reach the operator
        // verbatim - no clamping to max(1, n).
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        xLen = parseInt(ReadMap(map, "x_len", "0"));
        std::string handleStr = ReadMap(map, "handle", "VALID");
        handleValid = (handleStr != "NULLPTR" && handleStr != "NULL");
        std::string untouchedStr = ReadMap(map, "untouched_check", "FALSE");
        untouchedCheck = (untouchedStr == "TRUE" || untouchedStr == "true" || untouchedStr == "1");
        // Scenario classification: the L0/L1/L2 case tables use generic case_name
        // (L0_012...) and carry the scenario tag in the description column
        // (pf_smoke_* / pf_* / det_*); csyr_test.csv uses TC_PF_* case names.
        const auto startsWith = [](const std::string& s, const char* prefix) {
            return s.rfind(prefix, 0) == 0;
        };
        isPerfSmoke = startsWith(caseName, "pf_smoke") || startsWith(description, "pf_smoke");
        isPerf = isPerfSmoke || startsWith(caseName, "pf_") || startsWith(description, "pf_") ||
                 startsWith(caseName, "TC_PF") || startsWith(description, "TC_PF");
        isDeterminism = startsWith(caseName, "det_") || startsWith(description, "det_");
    }

    // Logical length of x (x_len overrides n for the K-dim mismatch scenarios).
    int LogicalXLen() const { return (xLen > 0) ? xLen : n; }
    // Physical x buffer length in complex elements: 1 + (logicalLen - 1) * |incx|.
    int PhysicalXLen() const
    {
        const int logical = LogicalXLen();
        if (logical <= 0) {
            return 0;
        }
        return 1 + (logical - 1) * std::abs(incx);
    }
    // Physical A storage length in complex elements: lda * n.
    size_t PhysicalALen() const
    {
        if (n <= 0 || lda <= 0) {
            return 0;
        }
        return static_cast<size_t>(lda) * static_cast<size_t>(n);
    }
};
