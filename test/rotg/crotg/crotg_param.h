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

struct CrotgParam : public BlasTestParamBase {
    float a_real = 0.0f, a_imag = 0.0f;
    float b_real = 0.0f, b_imag = 0.0f;
    PrecisionMode verifyMode = PrecisionMode::ABS;

    CrotgParam(const csv_map& m) : BlasTestParamBase(m)
    {
        a_real = parseFloat(ReadMap(m, "a_real", "0"));
        a_imag = parseFloat(ReadMap(m, "a_imag", "0"));
        b_real = parseFloat(ReadMap(m, "b_real", "0"));
        b_imag = parseFloat(ReadMap(m, "b_imag", "0"));
        verifyMode = parsePrecisionMode(ReadMap(m, "verify_mode", "ABS"));
    }
};
