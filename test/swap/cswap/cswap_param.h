/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSWAP_PARAM_H
#define CSWAP_PARAM_H

#include "csv_loader.h"

struct CswapParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    int incy = 1;
    BlasFillMode x = BlasFillMode("INDEX");
    BlasFillMode y = BlasFillMode("VALUE_NORM_N999");
    int xAlignOffset = 0;
    int yAlignOffset = 0;

    explicit CswapParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        x = BlasFillMode(ReadMap(map, "x_fill", "INDEX"));
        y = BlasFillMode(ReadMap(map, "y_fill", "VALUE_NORM_N999"));
        xAlignOffset = parseInt(ReadMap(map, "x_align_offset", "0"));
        yAlignOffset = parseInt(ReadMap(map, "y_align_offset", "0"));
    }
};

#endif // CSWAP_PARAM_H
