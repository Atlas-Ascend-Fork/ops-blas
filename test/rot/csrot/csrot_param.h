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
#include "csv_loader.h"

// Parameter struct for aclblasCsrot.
// Field order matches the aclblasCsrot API signature:
//   aclblasCsrot(handle, n, x, incx, y, incy, c, s)
// CSV columns: case_name,description,n,x,incx,y,incy,c,s,cs_ptr_mode,
//              expect_result,mere_threshold,mare_multiplier,random_seed
// c / s are float scalar columns; "null" means a null pointer (negative case) and
// must be checked BEFORE parseFloat (parseFloat("null") would throw).
// cs_ptr_mode in {host, device, mixed} controls where the c / s scalar pointers
// are materialized.
struct CsrotParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_5_5");
    int incy = 1;
    float c = 0.6f;
    float s = 0.8f;
    bool cIsNull = false;
    bool sIsNull = false;
    std::string csPtrMode = "host";

    CsrotParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        x = parseFill(ReadMap(m, "x", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        y = parseFill(ReadMap(m, "y", "RANDOM_NORM_5_5"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        cIsNull = (ReadMap(m, "c", "0.6") == "null");
        sIsNull = (ReadMap(m, "s", "0.8") == "null");
        if (!cIsNull) {
            c = parseFloat(ReadMap(m, "c", "0.6"));
        }
        if (!sIsNull) {
            s = parseFloat(ReadMap(m, "s", "0.8"));
        }
        csPtrMode = ReadMap(m, "cs_ptr_mode", "host");
    }
};
