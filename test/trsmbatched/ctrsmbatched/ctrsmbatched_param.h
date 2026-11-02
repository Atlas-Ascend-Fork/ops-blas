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
#include "csv_loader.h"

struct CtrsmBatchedParam : BlasTestParamBase {
    aclblasSideMode_t side;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int m, n, lda, ldb, batchCount;
    aclblasComplex alpha;
    bool nullAlpha;
    std::string aFill, bFill;
    explicit CtrsmBatchedParam(const csv_map& row) : BlasTestParamBase(row)
    {
        side = parseSideMode(ReadMap(row, "side"));
        uplo = parseFillMode(ReadMap(row, "uplo"));
        trans = parseOpTrans(ReadMap(row, "trans"));
        diag = parseDiagType(ReadMap(row, "diag"));
        m = ParseIntColumn(row, "m");
        n = ParseIntColumn(row, "n");
        lda = ReadMap(row, "lda").empty() ? std::max(1, side == ACLBLAS_SIDE_LEFT ? m : n) : ParseIntColumn(row, "lda");
        ldb = ReadMap(row, "ldb").empty() ? std::max(1, m) : ParseIntColumn(row, "ldb");
        batchCount = ParseIntColumn(row, "batchCount");
        nullAlpha = ReadMap(row, "alpha_real") == "null";
        // alpha_real=="null" skips both components: an empty alpha_imag must not
        // terminate construction for rows that never use alpha.
        alpha = nullAlpha ? aclblasComplex{0.0f, 0.0f} :
                            aclblasComplex{ParseFloatColumn(row, "alpha_real"), ParseFloatColumn(row, "alpha_imag")};
        aFill = ReadMap(row, "a");
        bFill = ReadMap(row, "b");
    }
};
