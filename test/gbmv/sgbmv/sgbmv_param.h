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

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

inline bool IsNullString(const std::string& str) { return str == "null" || str == "nullptr" || str == "NULLPTR"; }

inline float ParseOptionalFloat(const std::string& str, float defaultVal, bool& isNull)
{
    isNull = IsNullString(str);
    return isNull ? 0.0f : parseFloat(str, defaultVal);
}

inline BlasFillMode ParseOptionalFill(const std::string& str, bool& isNull)
{
    isNull = IsNullString(str);
    return isNull ? parseFill("RANDOM") : parseFill(str);
}

struct GbmvParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int kl = 0;
    int ku = 0;
    float alpha = 1.0f;
    bool nullAlpha = false;
    BlasFillMode a = parseFill("RANDOM");
    bool nullA = false;
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM");
    bool nullX = false;
    int incx = 1;
    float beta = 0.0f;
    bool nullBeta = false;
    BlasFillMode y = parseFill("RANDOM");
    bool nullY = false;
    int incy = 1;
    // Extended CSV columns: handle=nullptr carrier for the null-handle group,
    // and the alpha=0 exact-semantics switch
    bool nullHandle = false;
    bool expectExact = false;

    GbmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        kl = parseInt(ReadMap(map, "kl", "0"));
        ku = parseInt(ReadMap(map, "ku", "0"));
        alpha = ParseOptionalFloat(ReadMap(map, "alpha", "1.0"), 1.0f, nullAlpha);
        a = ParseOptionalFill(ReadMap(map, "a_fill", ReadMap(map, "a", "RANDOM")), nullA);
        lda = parseInt(ReadMap(map, "lda", std::to_string(kl + ku + 1)));
        x = ParseOptionalFill(ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM")), nullX);
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = ParseOptionalFloat(ReadMap(map, "beta", "0.0"), 0.0f, nullBeta);
        y = ParseOptionalFill(ReadMap(map, "y_fill", ReadMap(map, "y", "RANDOM")), nullY);
        incy = parseInt(ReadMap(map, "incy", "1"));
        nullHandle = IsNullString(ReadMap(map, "handle", "VALID"));
        expectExact = (ReadMap(map, "expect_exact", "FALSE") == "TRUE");
    }
};
