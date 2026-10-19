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

#include <cstdint>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "types.h"

inline PrecisionMode parsePrecisionMode(const std::string& s)
{
    static const std::unordered_map<std::string, PrecisionMode> t = {
        {"ABS", PrecisionMode::ABS}, {"REL", PrecisionMode::REL},
        {"COMBINED", PrecisionMode::COMBINED}, {"MERE_MARE", PrecisionMode::MERE_MARE},
        {"EXACT", PrecisionMode::EXACT}, {"INTEGER", PrecisionMode::INTEGER}};
    return parseEnum(s, t, PrecisionMode::ABS);
}

// crotg CSV column format (see gen_csv.py):
//   case_name,description,a_real,a_imag,b_real,b_imag,c,s,iters,
//   expect_result,mere_threshold,mare_multiplier,random_seed
// a_real/a_imag/b_real/b_imag: float literal (incl. 1e30/1e-38/inf/nan) or "null" (nullptr negative case)
// c/s: "out" (valid output pointer) or "null" (nullptr negative case)
// iters: 1 for accuracy cases; number of consecutive in-stream calls for performance cases
struct CrotgParam : public BlasTestParamBase {
    // Flat float view: consumed directly by arch22 cases via a_real/a_imag/b_real/b_imag
    float a_real = 0.0f, a_imag = 0.0f;
    float b_real = 0.0f, b_imag = 0.0f;
    PrecisionMode verifyMode = PrecisionMode::ABS;

    // Complex / null-pointer / iteration-count view: consumed by arch35 cases via a/b + IsNull flags + iters
    aclblasComplex a{};
    aclblasComplex b{};
    bool aIsNull = false;
    bool bIsNull = false;
    bool cIsNull = false;
    bool sIsNull = false;
    int iters = 1;

    CrotgParam(const csv_map& m) : BlasTestParamBase(m)
    {
        std::string ar = ReadMap(m, "a_real", "0");
        std::string ai = ReadMap(m, "a_imag", "0");
        std::string br = ReadMap(m, "b_real", "0");
        std::string bi = ReadMap(m, "b_imag", "0");

        if (ar == "null" || ai == "null") {
            aIsNull = true;
        } else {
            a_real = parseFloat(ar);
            a_imag = parseFloat(ai);
            a.real = a_real;
            a.imag = a_imag;
        }
        if (br == "null" || bi == "null") {
            bIsNull = true;
        } else {
            b_real = parseFloat(br);
            b_imag = parseFloat(bi);
            b.real = b_real;
            b.imag = b_imag;
        }

        std::string cStr = ReadMap(m, "c", "out");
        std::string sStr = ReadMap(m, "s", "out");
        cIsNull = (cStr == "null");
        sIsNull = (sStr == "null");

        iters = parseInt(ReadMap(m, "iters", "1"), 1);
        if (iters < 1) {
            iters = 1;
        }

        verifyMode = parsePrecisionMode(ReadMap(m, "verify_mode", "ABS"));
    }
};
