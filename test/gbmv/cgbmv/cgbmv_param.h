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

#include <cctype>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// trans accepts the enum names used by parseOpTrans, or a raw integer (e.g. "999") for invalid-enum cases.
inline aclblasOperation_t ParseCgbmvTrans(const std::string& s)
{
    if (!s.empty() && (std::isdigit(static_cast<unsigned char>(s[0])) != 0 || s[0] == '-')) {
        return static_cast<aclblasOperation_t>(parseInt(s, 0xFF));
    }
    return parseOpTrans(s);
}

inline bool IsNullScalar(const std::string& s) { return s == "null" || s == "nullptr" || s == "NULLPTR"; }

// Fill spec of A/x/y: any BlasFillMode string (e.g. RANDOM_NORM_5_5 = uniform on [-5, 5]) or GAUSS_<mu>_<sigma>
// for the normal distribution N(mu, sigma^2). Real and imaginary parts are sampled independently.
struct CgbmvFill {
    BlasFillMode mode = parseFill("RANDOM_NORM_5_5");
    bool gauss = false;
    float mu = 0.0f;
    float sigma = 1.0f;
};

inline CgbmvFill ParseCgbmvFill(const std::string& s)
{
    CgbmvFill f;
    const std::string prefix = "GAUSS_";
    if (s.rfind(prefix, 0) != 0) {
        f.mode = parseFill(s);
        return f;
    }
    std::string rest = s.substr(prefix.size());
    size_t sep = rest.find('_');
    f.gauss = true;
    f.mu = parseFloat(rest.substr(0, sep), 0.0f);
    f.sigma = (sep == std::string::npos) ? 1.0f : parseFloat(rest.substr(sep + 1), 1.0f);
    return f;
}

inline bool IsNullFill(const CgbmvFill& f) { return !f.gauss && f.mode.method == BlasFillMode::M_NULLPTR; }

struct CgbmvParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int kl = 0;
    int ku = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    bool nullAlpha = false;
    CgbmvFill aFill;
    int lda = 1;
    CgbmvFill xFill;
    int incx = 1;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    bool nullBeta = false;
    CgbmvFill yFill;
    int incy = 1;
    int aAlignOffset = 0; // complex elements between the device allocation start and A / x / y
    int xAlignOffset = 0;
    int yAlignOffset = 0;

    CgbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = ParseCgbmvTrans(ReadMap(map, "trans", "ACLBLAS_OP_N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        kl = parseInt(ReadMap(map, "kl", "0"));
        ku = parseInt(ReadMap(map, "ku", "0"));

        std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        nullAlpha = IsNullScalar(alphaRealStr);
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(ReadMap(map, "alpha_imag", "0.0"), 0.0f);

        aFill = ParseCgbmvFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(kl + ku + 1)));
        xFill = ParseCgbmvFill(ReadMap(map, "x_fill", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));

        std::string betaRealStr = ReadMap(map, "beta_real", "0.0");
        nullBeta = IsNullScalar(betaRealStr);
        betaReal = nullBeta ? 0.0f : parseFloat(betaRealStr, 0.0f);
        betaImag = nullBeta ? 0.0f : parseFloat(ReadMap(map, "beta_imag", "0.0"), 0.0f);

        yFill = ParseCgbmvFill(ReadMap(map, "y_fill", "RANDOM_NORM_5_5"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        aAlignOffset = parseInt(ReadMap(map, "a_align_offset", "0"));
        xAlignOffset = parseInt(ReadMap(map, "x_align_offset", "0"));
        yAlignOffset = parseInt(ReadMap(map, "y_align_offset", "0"));
    }
};
