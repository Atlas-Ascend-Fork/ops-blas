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
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// CSV "null"/"NULL"/"nullptr" markers for the c / s scalar columns.
// The operator treats c and s independently, so either may be null.
inline bool CrotIsNullMarker(const std::string& s)
{
    std::string lower;
    lower.reserve(s.size());
    for (char ch : s) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return lower == "null" || lower == "nullptr" || lower == "none";
}

// Parameter struct for aclblasCrot.
// Field order matches the aclblasCrot API signature.
//   aclblasCrot(handle, n, x, incx, y, incy, c, s)
// s is complex64, so the CSV splits it into s_real / s_imag.
// mere_threshold / mare_multiplier come from BlasTestParamBase (CSV columns).
struct CrotParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_1");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_1");
    int incy = 1;
    float c = 0.6f;
    float sReal = 0.0f;
    float sImag = 0.0f;
    bool cIsNull = false;
    bool sIsNull = false;
    std::string csPtrMode = "host";

    CrotParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        x = parseFill(ReadMap(m, "x", "RANDOM_NORM_1"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        y = parseFill(ReadMap(m, "y", "RANDOM_NORM_1"));
        incy = parseInt(ReadMap(m, "incy", "1"));

        const std::string cStr = ReadMap(m, "c", "0.6");
        if (CrotIsNullMarker(cStr)) {
            cIsNull = true;
        } else {
            c = parseFloat(cStr, 0.6f);
        }

        const std::string srStr = ReadMap(m, "s_real", "0");
        const std::string siStr = ReadMap(m, "s_imag", "0");
        if (CrotIsNullMarker(srStr) || CrotIsNullMarker(siStr)) {
            sIsNull = true;
        } else {
            sReal = parseFloat(srStr, 0.0f);
            sImag = parseFloat(siStr, 0.0f);
        }

        csPtrMode = ReadMap(m, "cs_ptr_mode", "host");
    }
};
