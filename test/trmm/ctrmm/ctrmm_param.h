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
#include <algorithm>
#include <cstdlib>
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// Negative enum encoding used by the official CSV (e.g. TC_ED_155..158):
// side/uplo/trans/diag columns carry the literal "999". The framework's
// parse* helpers fall back to a default for unknown strings, which would
// silently turn 999 into a legal enum and break the INVALID_VALUE cases —
// so the raw "999" is preserved via a direct enum cast here.
static inline aclblasSideMode_t CtrmmParseSide(const std::string& s)
{
    if (s == "999") {
        return static_cast<aclblasSideMode_t>(999);
    }
    return parseSideMode(s);
}

static inline aclblasFillMode_t CtrmmParseUplo(const std::string& s)
{
    if (s == "999") {
        return static_cast<aclblasFillMode_t>(999);
    }
    return parseFillMode(s);
}

static inline aclblasOperation_t CtrmmParseTrans(const std::string& s)
{
    if (s == "999") {
        return static_cast<aclblasOperation_t>(999);
    }
    return parseOpTrans(s);
}

static inline aclblasDiagType_t CtrmmParseDiag(const std::string& s)
{
    if (s == "999") {
        return static_cast<aclblasDiagType_t>(999);
    }
    return parseDiagType(s);
}

// Integer parse that treats an existing-but-empty CSV cell as "use the
// default" (the official TC_PF rows leave lda/ldb/ldc empty to denote the
// compact leading dimensions).
static inline int CtrmmParseInt(const std::string& s, int def)
{
    if (s.empty()) {
        return def;
    }
    try {
        return std::stoi(s);
    } catch (const std::exception&) {
        return def;
    }
}

// Parameter struct for aclblasCtrmm (complex64 triangular matrix-matrix
// multiply, out-of-place C). CSV contract (22 columns, task package
// ctrmm_test.csv): case_name, description, side, uplo, trans, diag, m, n,
// alpha_real, alpha_imag, nullA, nullB, nullC, a_fill, lda, b_fill, ldb,
// ldc, expect_result, mere_threshold, mare_multiplier, random_seed.
struct CtrmmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    aclblasDiagType_t diag = ACLBLAS_NON_UNIT;
    int m = 0;
    int n = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    bool nullAlpha = false;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode bFill = parseFill("RANDOM_NORM_5_5");
    double mereThreshold = 1.0 / 8192.0;
    double mareMultiplier = 10.0;

    bool AlphaIsZero() const { return !nullAlpha && alphaReal == 0.0f && alphaImag == 0.0f; }

    explicit CtrmmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side = CtrmmParseSide(ReadMap(map, "side", "LEFT"));
        uplo = CtrmmParseUplo(ReadMap(map, "uplo", "UPPER"));
        trans = CtrmmParseTrans(ReadMap(map, "trans", "N"));
        diag = CtrmmParseDiag(ReadMap(map, "diag", "NON_UNIT"));
        m = std::stoi(ReadMap(map, "m", "0"));
        n = std::stoi(ReadMap(map, "n", "0"));

        std::string alphaReStr = ReadMap(map, "alpha_real", "1");
        std::string alphaImStr = ReadMap(map, "alpha_imag", "0");
        nullAlpha =
            (alphaReStr == "null" || alphaReStr == "nullptr" || alphaImStr == "null" || alphaImStr == "nullptr");
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaReStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImStr, 0.0f);

        nullA = (ReadMap(map, "nullA", "0") == "1");
        nullB = (ReadMap(map, "nullB", "0") == "1");
        nullC = (ReadMap(map, "nullC", "0") == "1");

        // Leading dimensions: an empty cell denotes the compact dimension
        // (cherk_param.h precedent): A is m-square (LEFT) or n-square (RIGHT),
        // B/C are m x n.
        const int defLda = (side == ACLBLAS_SIDE_LEFT) ? std::max(1, m) : std::max(1, n);
        const int defLdbc = std::max(1, m);
        lda = CtrmmParseInt(ReadMap(map, "lda", std::to_string(defLda)), defLda);
        ldb = CtrmmParseInt(ReadMap(map, "ldb", std::to_string(defLdbc)), defLdbc);
        ldc = CtrmmParseInt(ReadMap(map, "ldc", std::to_string(defLdbc)), defLdbc);
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));

        mereThreshold = parseDouble(ReadMap(map, "mere_threshold", "0.00012207031"));
        mareMultiplier = parseDouble(ReadMap(map, "mare_multiplier", "10.0"));
    }
};
