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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "csv_loader.h"

struct CdotuParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_5_5");
    int incy = 1;
    bool resultIsNull = false;

    CdotuParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        x = parseFill(ReadMap(map, "x", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y", "RANDOM_NORM_5_5"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        BlasFillMode resultFill = parseFill(ReadMap(map, "result_fill", "VALUE_NORM_0"));
        resultIsNull = (resultFill.method == BlasFillMode::M_NULLPTR);
    }
};