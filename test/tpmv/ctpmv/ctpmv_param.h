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

#include "cann_ops_blas.h"
#include "csv_loader.h"

struct CtpmvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int n;
    int incx;
    BlasFillMode ap;
    BlasFillMode x;
    double mereThreshold;
    double mareMultiplier;

    explicit CtpmvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        diag = parseDiagType(ReadMap(map, "diag", "NON_UNIT"));
        n = parseInt(ReadMap(map, "n", "0"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        ap = parseFill(ReadMap(map, "aPacked", "INDEX"));
        x = parseFill(ReadMap(map, "x", "INDEX"));
        mereThreshold = parseDouble(ReadMap(map, "mere_threshold", "0"));
        mareMultiplier = parseDouble(ReadMap(map, "mare_multiplier", "0"));
    }
};
