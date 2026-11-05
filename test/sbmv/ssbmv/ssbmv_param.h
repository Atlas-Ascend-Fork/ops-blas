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
#include <climits>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

/*!
 * \brief Reads the first of two spellings of the same column, falling back to a default.
 *
 * The fill columns of A, x and y are spelled "a"/"x"/"y" in the arch35 CSV and
 * "a_fill"/"x_fill"/"y_fill" in the arch22 CSV; both are accepted here so that one parameter
 * struct serves both architectures.
 */
inline std::string ReadFillColumn(const csv_map& map, const char* primary, const char* alias, const char* defaultValue)
{
    auto it = map.find(primary);
    if (it != map.end()) {
        return it->second;
    }
    return ReadMap(map, alias, std::string(defaultValue));
}

/*!
 * \brief True when a scalar column holds the "no pointer" marker.
 *
 * The arch22 CSV expresses a null scalar as the literal token "null" in the alpha/beta value
 * column, while the arch35 CSV keeps a value there and marks the pointer with the separate
 * alpha_fill/beta_fill column. Both conventions have to select the null pointer at the call site.
 */
inline bool IsNullScalarToken(const csv_map& map, const char* key)
{
    auto it = map.find(key);
    if (it == map.end()) {
        return false;
    }
    std::string token = it->second;
    std::transform(
        token.begin(), token.end(), token.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return token == "null" || token == "nullptr";
}

/*!
 * \brief Parses the uplo column, accepting the fully qualified enum spelling.
 *
 * The arch22 CSV spells the value out as "ACLBLAS_FILL_MODE_UPPER" / "ACLBLAS_FILL_MODE_LOWER"
 * (and uses "999" for the invalid-enum case), which parseFillMode does not recognise on its own.
 */
inline aclblasFillMode_t ParseUplo(const csv_map& map)
{
    const std::string prefix = "ACLBLAS_FILL_MODE_";
    std::string token = ReadMap(map, "uplo", "UPPER");
    if (token.compare(0, prefix.size(), prefix) == 0) {
        token = token.substr(prefix.size());
    }
    return parseFillMode(token);
}

struct SsbmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    BlasFillMode alphaFill = parseFill("RANDOM");
    BlasFillMode a = parseFill("RANDOM");
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    float beta = 0.0f;
    BlasFillMode betaFill = parseFill("RANDOM");
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;

    SsbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = ParseUplo(map);
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        alphaFill = parseFill(ReadMap(map, "alpha_fill", "RANDOM"));
        if (map.count("alpha_fill") == 0 && IsNullScalarToken(map, "alpha")) {
            alphaFill = parseFill("NULLPTR");
        }
        a = parseFill(ReadFillColumn(map, "a", "a_fill", "RANDOM"));
        /* k == INT_MAX 时 k + 1 在 int 域溢出，默认 lda 直接钳到 INT_MAX（与 stbsv 同口径）。 */
        lda = parseInt(ReadMap(map, "lda", std::to_string(k >= INT_MAX ? INT_MAX : std::max(1, k + 1))));
        x = parseFill(ReadFillColumn(map, "x", "x_fill", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        betaFill = parseFill(ReadMap(map, "beta_fill", "RANDOM"));
        if (map.count("beta_fill") == 0 && IsNullScalarToken(map, "beta")) {
            betaFill = parseFill("NULLPTR");
        }
        y = parseFill(ReadFillColumn(map, "y", "y_fill", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};
