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
#include <cctype>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

inline bool parseCsyr2Bool(const std::string& value)
{
    return value == "1" || value == "true" || value == "TRUE";
}

inline aclblasComplex parseCsyr2Complex(const csv_map& map, const std::string& realKey, const std::string& imagKey)
{
    return {parseFloat(ReadMap(map, realKey, "1.0")), parseFloat(ReadMap(map, imagKey, "0.0"))};
}

struct Csyr2Param : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    bool handleNull = false;
    bool alphaNull = false;
    BlasFillMode xRealFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode xImagFill = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    BlasFillMode yRealFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode yImagFill = parseFill("RANDOM_NORM_5_5");
    int incy = 1;
    BlasFillMode aRealFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode aImagFill = parseFill("RANDOM_NORM_5_5");
    int lda = 1;
    std::string distribution = "FIXED";
    float distributionMean = 0.0f;
    float distributionStddev = 1.0f;

    explicit Csyr2Param(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseCsyr2Complex(map, "alpha_real", "alpha_imag");
        handleNull = parseCsyr2Bool(ReadMap(map, "handle_null", "0"));
        alphaNull = parseCsyr2Bool(ReadMap(map, "alpha_null", "0"));

        xRealFill = parseFill(ReadMap(map, "x_fill", "RANDOM_NORM_5_5"));
        xImagFill = parseFill(ReadMap(map, "x_imag_fill", ReadMap(map, "x_fill", "RANDOM_NORM_5_5")));
        incx = parseInt(ReadMap(map, "incx", "1"));
        yRealFill = parseFill(ReadMap(map, "y_fill", "RANDOM_NORM_5_5"));
        yImagFill = parseFill(ReadMap(map, "y_imag_fill", ReadMap(map, "y_fill", "RANDOM_NORM_5_5")));
        incy = parseInt(ReadMap(map, "incy", "1"));
        aRealFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        aImagFill = parseFill(ReadMap(map, "a_imag_fill", ReadMap(map, "a_fill", "RANDOM_NORM_5_5")));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        distribution = ReadMap(map, "distribution", "FIXED");
        distributionMean = parseFloat(ReadMap(map, "distribution_mean", "0.0"));
        distributionStddev = parseFloat(ReadMap(map, "distribution_stddev", "1.0"));
    }
};

namespace csyr2_arch22_test {

inline bool Csyr2CsvIsNullToken(const std::string& s)
{
    std::string t = s;
    for (char& c : t) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return t == "null" || t == "nullptr" || t == "none";
}

struct Csyr2Param : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    bool nullAlpha = false;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;
    BlasFillMode a = parseFill("RANDOM");
    int lda = 0;

    explicit Csyr2Param(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        n = parseInt(ReadMap(map, "n", "0"));
        const std::string ar = ReadMap(map, "alpha_real", ReadMap(map, "alpha", "1.0"));
        const std::string ai = ReadMap(map, "alpha_imag", "0.0");
        nullAlpha = Csyr2CsvIsNullToken(ar) || Csyr2CsvIsNullToken(ai);
        if (!nullAlpha) {
            alpha.real = parseFloat(ar, 1.0f);
            alpha.imag = parseFloat(ai, 0.0f);
        }
        x = parseFill(ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM")));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y_fill", ReadMap(map, "y", "RANDOM")));
        incy = parseInt(ReadMap(map, "incy", "1"));
        a = parseFill(ReadMap(map, "a_fill", ReadMap(map, "a", "RANDOM")));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
    }

    aclblasStatus_t ExpectedStatus() const { return expectResult; }
};

}  // namespace csyr2_arch22_test
