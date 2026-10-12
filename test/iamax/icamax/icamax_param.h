/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "csv_loader.h"

struct IcamaxParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    bool resultIsNull = false;

    explicit IcamaxParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        x = parseFill(ReadMap(map, "x", "RANDOM_NORM_5_5"));
        resultIsNull = ReadMap(map, "result", "") == "NULLPTR";
    }
};
