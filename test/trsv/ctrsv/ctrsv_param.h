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
#include <algorithm>
#include "cann_ops_blas.h"
#include "csv_loader.h"
struct CtrsvParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int64_t n = 0, lda = 0, incx = 1;
    BlasFillMode A = parseFill("RANDOM"), x = parseFill("RANDOM");
    bool normal = false;
    bool smallWorkspace = false;
    CtrsvParam(const csv_map& m) : BlasTestParamBase(m)
    {
        uplo = parseFillMode(ReadMap(m, "uplo", "LOWER"));
        trans = parseOpTrans(ReadMap(m, "trans", "N"));
        diag = parseDiagType(ReadMap(m, "diag", "NON_UNIT"));
        n = parseInt(ReadMap(m, "n", "0"));
        lda = parseInt(ReadMap(m, "lda", std::to_string(std::max(int64_t(1), n))));
        incx = parseInt(ReadMap(m, "incx", "1"));
        A = parseFill(ReadMap(m, "A", "RANDOM"));
        x = parseFill(ReadMap(m, "x", "RANDOM"));
    }
};
