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
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "dtype_cast.h"
#include "hgemm_param.h"

// Golden reference per task spec sec. 3.1/sec. 3.2/sec. 3.5: compute C = alpha*op(A)*op(B)
// + beta*C in FLOAT32 (Netlib cblas_sgemm) on FLOAT16-quantized inputs, then
// round the result to FLOAT16 (round-to-nearest-even) element by element.
// Inputs are FLOAT32 arrays that already hold FLOAT16-representable values.
inline aclblasStatus_t aclblasHgemm_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, float alpha,
    const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (m < 0 || n < 0 || k < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    cblas_sgemm(CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}

// Mixed-tolerance verification per task spec sec. 3.2 (FLOAT16 tier). Finite
// elements pass when |actual-golden| <= atol + rtol*|golden| with
// matched_ratio >= 0.99 and max_abs_error <= max(1e-1, 32*ULP). Non-finite
// elements follow sec. 3.2 clause 5 and are tracked separately: both NaN,
// same-sign Inf, or one side NaN and the other +/-Inf are equivalent, any
// other pairing fails.
struct HgemmVerifyResult {
    bool pass = false;
    size_t finiteCount = 0;
    size_t finiteFailCount = 0;
    size_t nonFiniteCount = 0;
    size_t nonFinitePassCount = 0;
    double maxAbsError = 0.0;
    double matchedRatio = 1.0;
};

// ULP(x) = 2^(floor(log2|x|) - mantissaBits), ULP(0) = 2^(emin - mantissaBits)
inline double HgemmUlpAt(double goldAbs, int mantissaBits, int emin)
{
    if (goldAbs == 0.0) {
        return std::pow(2.0, static_cast<double>(emin - mantissaBits));
    }
    int exp = 0;
    std::frexp(goldAbs, &exp);
    return std::ldexp(1.0, std::max(exp - 1, emin) - mantissaBits);
}

// Task spec sec. 3.2 clause 5 special-value rule: both NaN, same-sign Inf, or
// one side NaN and the other +/-Inf are equivalent (pass); any other pairing
// between a finite and a non-finite value fails.
inline bool HgemmNonFiniteEquivalent(float outVal, float goldVal)
{
    bool bothNan = std::isnan(outVal) && std::isnan(goldVal);
    bool sameSignInf = std::isinf(outVal) && std::isinf(goldVal) && ((outVal > 0) == (goldVal > 0));
    bool nanInfPair = (std::isnan(outVal) && std::isinf(goldVal)) || (std::isinf(outVal) && std::isnan(goldVal));
    return bothNan || sameSignInf || nanInfPair;
}

inline HgemmVerifyResult VerifyHgemmElements(const float* actual, const float* golden, size_t count)
{
    constexpr double kRtol = 0.0009765625; // 2^-10
    constexpr double kAtol = 0.0009765625; // 2^-10
    constexpr double kRequiredMatchedRatio = 0.99;
    constexpr double kFixedAbsLimit = 1e-1;
    constexpr int kMantissaBits = 10;
    constexpr int kEmin = -14;

    HgemmVerifyResult res;
    bool absLimitOk = true;
    for (size_t i = 0; i < count; i++) {
        float outVal = actual[i];
        float goldVal = golden[i];
        if (!std::isfinite(outVal) || !std::isfinite(goldVal)) {
            res.nonFiniteCount++;
            if (HgemmNonFiniteEquivalent(outVal, goldVal)) {
                res.nonFinitePassCount++;
            }
            continue;
        }
        res.finiteCount++;
        double diff = std::abs(static_cast<double>(outVal) - static_cast<double>(goldVal));
        if (diff > res.maxAbsError) {
            res.maxAbsError = diff;
        }
        double goldAbs = std::abs(static_cast<double>(goldVal));
        if (diff > kAtol + kRtol * goldAbs) {
            res.finiteFailCount++;
        }
        if (diff > std::max(kFixedAbsLimit, 32.0 * HgemmUlpAt(goldAbs, kMantissaBits, kEmin))) {
            absLimitOk = false;
        }
    }
    res.matchedRatio = res.finiteCount > 0 ?
                           1.0 - static_cast<double>(res.finiteFailCount) / static_cast<double>(res.finiteCount) :
                           1.0;
    bool finiteOk = (res.matchedRatio >= kRequiredMatchedRatio) && absLimitOk;
    bool nonFiniteOk = (res.nonFiniteCount == res.nonFinitePassCount);
    res.pass = finiteOk && nonFiniteOk;
    return res;
}
