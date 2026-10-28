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

// Required numeric columns must fail with the column name in the message
// (issue #643) instead of a bare std::invalid_argument from std::stoi("").
inline int ParseIntColumn(const csv_map& row, const char* key, const std::string& def = std::string())
{
    const std::string raw = ReadMap(row, key, def);
    try {
        return std::stoi(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("column '") + key + "' is missing, empty or non-numeric: '" + raw + "'");
    }
}

inline float ParseFloatColumn(const csv_map& row, const char* key, const std::string& def = std::string())
{
    const std::string raw = ReadMap(row, key, def);
    try {
        return std::stof(raw);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("column '") + key + "' is missing, empty or non-numeric: '" + raw + "'");
    }
}

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
        lda = ParseIntColumn(row, "lda", std::to_string(std::max(1, side == ACLBLAS_SIDE_LEFT ? m : n)));
        ldb = ParseIntColumn(row, "ldb", std::to_string(std::max(1, m)));
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
