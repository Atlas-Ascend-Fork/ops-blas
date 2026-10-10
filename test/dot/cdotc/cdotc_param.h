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

#include "csv_loader.h"

inline BlasFillMode ParseCdotcFill(const std::string& value)
{
    // The task CSV uses RANDOM_NORM_5_5 for a uniform [-5, 5] fill.
    // The shared parser spells the same range as RANDOM_NORM_N5_5.
    return parseFill(value == "RANDOM_NORM_5_5" ? "RANDOM_NORM_N5_5" : value);
}

struct CdotcParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_N5_5");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_N5_5");
    int incy = 1;
    BlasFillMode resultFill = parseFill("VALUE_NORM_0");
    std::string distribution = "uniform";

    explicit CdotcParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        x = ParseCdotcFill(ReadMap(map, "x", "RANDOM_NORM_N5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = ParseCdotcFill(ReadMap(map, "y", "RANDOM_NORM_N5_5"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        resultFill = parseFill(ReadMap(map, "result_fill", "VALUE_NORM_0"));
        distribution = ReadMap(map, "distribution", "uniform");
    }
};
