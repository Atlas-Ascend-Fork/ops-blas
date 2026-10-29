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

#include <complex>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "cgemm_ex_param.h"

inline void RealToComplex(const float* src, std::vector<std::complex<float>>& dst, size_t count)
{
    dst.resize(count);
    for (size_t i = 0; i < count; i++) {
        dst[i] = std::complex<float>(src[i], 0.0f);
    }
}

inline aclblasStatus_t aclblasCgemmEx_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    std::complex<float> alpha, const std::complex<float>* A, int lda, const std::complex<float>* B, int ldb,
    std::complex<float> beta, std::complex<float>* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m < 0 || n < 0 || k < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    cblas_cgemm(CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}
