/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHERKX_PARAM_H
#define CHERKX_PARAM_H

#include <string>
#include <algorithm>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// CSV boolean column parser: "1" / "true" / "yes" (case-insensitive on the literal)
// are true, everything else (including the empty/absent column) is false.
// Used by the alpha_on_device / beta_on_device pointer-mode columns and by the
// legacy nullA / nullB / nullC / nullAlpha / nullBeta / a_eq_b flag columns.
static inline bool CherkxParseCsvFlag(const std::string& s)
{
    return s == "1" || s == "true" || s == "TRUE" || s == "True" || s == "yes";
}

// "null" literal detector for scalar columns. Per 测试方案 §5.0 the alpha_real
// column may carry the `null` literal, which is OR-equivalent to the nullAlpha
// flag column (either one makes the test pass nullptr to the API).
static inline bool CherkxIsNullLiteral(const std::string& s)
{
    return s == "null" || s == "nullptr";
}

// Parameter struct for aclblasCherkx (complex<float> Hermitian rank-kx update).
// Mirrors aclblasSsyrkxParam (dual-matrix A/B) with the cuBLAS-alignment additions
// (测试方案 §2.1 步骤 6 / §5.0 CSV 表头迁移):
//   - A/B/C and alpha are aclblasComplex; beta stays float (real scalar, HERKX spec).
//     alpha is therefore split into the two CSV columns alpha_real / alpha_imag
//     (仓内 cher2k 先例：test/her2k/cher2k/cher2k_param.h).
//   - nullAlpha/nullBeta: pass nullptr to the API and expect INVALID_VALUE. The
//     `null` literal in alpha_real / beta is OR-ed into these flags (§5.0 迁移规则).
//   - alphaOnDevice/betaOnDevice: scalar pointer mode (测试方案 §3.1 指针模式因子,
//     对齐 ssymm 终态 wrapper 参数命名). Default false = Host mode; both must
//     be set together — a mixed (D,H)/(H,D) combination is rejected by the operator
//     before dereferencing (TC_AL_18/19).
//   - aEqB: when true the test generates B with the SAME seed as A (no +1 offset) so
//     A and B carry identical data, exercising the A==B degenerate scenario
//     (C = alpha*A*A^H + beta*C, equivalent to aclblasCherk).
struct CherkxParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_1");
    int lda = 0;
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_1");
    int ldb = 0;
    float beta = 0.0f;
    BlasFillMode cFill = BlasFillMode("VALUE_NORM_0");
    int ldc = 0;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool aEqB = false;
    bool alphaOnDevice = false;
    bool betaOnDevice = false;

    explicit CherkxParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo  = parseFillMode(ReadMap(map, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n     = parseInt(ReadMap(map, "n", "0"));
        k     = parseInt(ReadMap(map, "k", "0"));

        // Complex alpha: `null` in alpha_real encodes a null scalar pointer
        // (cher2k / chemm convention); alpha_imag is ignored in that case.
        const std::string alphaRealStr = ReadMap(map, "alpha_real", "1.0");
        const std::string alphaImagStr = ReadMap(map, "alpha_imag", "0.0");
        nullAlpha = CherkxIsNullLiteral(alphaRealStr) || CherkxParseCsvFlag(ReadMap(map, "nullAlpha", "0"));
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(alphaImagStr, 0.0f);

        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_NORM_1"));

        // Default lda/ldb (column-major):
        //   trans=N -> A/B are (n x k),  lda/ldb >= max(1, n)
        //   trans=C -> A/B are (k x n),  lda/ldb >= max(1, k)
        int defaultLdab = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        lda = parseInt(ReadMap(map, "lda", std::to_string(defaultLdab)));

        bFill = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_NORM_1"));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(defaultLdab)));

        // Real beta: the `null` literal is OR-ed into the nullBeta flag column.
        const std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = CherkxIsNullLiteral(betaStr) || CherkxParseCsvFlag(ReadMap(map, "nullBeta", "0"));
        beta = nullBeta ? 0.0f : parseFloat(betaStr, 0.0f);

        cFill = BlasFillMode(ReadMap(map, "c_fill", "VALUE_NORM_0"));
        ldc   = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));

        nullA = CherkxParseCsvFlag(ReadMap(map, "nullA", "0"));
        nullB = CherkxParseCsvFlag(ReadMap(map, "nullB", "0"));
        nullC = CherkxParseCsvFlag(ReadMap(map, "nullC", "0"));
        aEqB  = CherkxParseCsvFlag(ReadMap(map, "a_eq_b", "0"));

        // Scalar pointer mode (缺省 Host，测试方案 §5.0 迁移规则)
        alphaOnDevice = CherkxParseCsvFlag(ReadMap(map, "alpha_on_device", "0"));
        betaOnDevice  = CherkxParseCsvFlag(ReadMap(map, "beta_on_device", "0"));
    }
};

#endif // CHERKX_PARAM_H
