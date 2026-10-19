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

#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {

// golden uses double precision + std::hypot for intermediate values
// to avoid float overflow/underflow, serving as a high-precision reference.
// The algorithm mirrors LAPACK/Netlib crotg and computes rnorm via std::hypot.
inline void CrotgGoldenCompute(aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    const float ar = a->real;
    const float ai = a->imag;
    const float br = b->real;
    const float bi = b->imag;

    // Compute entirely in double + std::hypot to avoid float large-value square overflow
    const double dar = ar;
    const double dai = ai;
    const double dbr = br;
    const double dbi = bi;
    const double danorm = std::hypot(dar, dai);
    const double dbnorm = std::hypot(dbr, dbi);

    if (dbnorm == 0.0) {
        // b = 0: c = 1, s = 0, r = a
        *c = 1.0f;
        s->real = 0.0f;
        s->imag = 0.0f;
        // a unchanged (r = a)
        return;
    }

    if (danorm == 0.0) {
        // a = 0, b != 0: c = 0, s = conjg(b)/bnorm, r = bnorm
        *c = 0.0f;
        s->real = static_cast<float>(dbr / dbnorm);
        s->imag = static_cast<float>(-dbi / dbnorm);
        a->real = static_cast<float>(dbnorm);
        a->imag = 0.0f;
        return;
    }

    // General case: rnorm via double-precision hypot. std::hypot is internally
    // scaled, so it neither overflows nor underflows even when anorm/bnorm are
    // near FLT_MAX, matching the operator's hypotf-based implementation.
    const double drnorm = std::hypot(danorm, dbnorm);
    if (drnorm == 0.0) {
        // Defensive guard against division by zero (G.EXP.22-CPP).
        // Unreachable here: danorm and dbnorm are both non-zero at this point,
        // so hypot(danorm, dbnorm) is always positive.
        return;
    }

    // alpha = a / anorm (sgn)
    const double dalR = dar / danorm;
    const double dalI = dai / danorm;

    // r = alpha * rnorm
    a->real = static_cast<float>(dalR * drnorm);
    a->imag = static_cast<float>(dalI * drnorm);
    // c = anorm / rnorm
    *c = static_cast<float>(danorm / drnorm);
    // s = alpha * conjg(b) / rnorm, conjg(b) = (br, -bi)
    // alpha * conjg(b) = (dalR*br + dalI*bi, dalI*br - dalR*bi)
    s->real = static_cast<float>((dalR * dbr + dalI * dbi) / drnorm);
    s->imag = static_cast<float>((dalI * dbr - dalR * dbi) / drnorm);
}

} // namespace

inline aclblasStatus_t aclblasCrotg_cpu(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    CrotgGoldenCompute(a, b, c, s);
    return ACLBLAS_STATUS_SUCCESS;
}
