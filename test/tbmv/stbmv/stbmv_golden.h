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
#include <cstdlib>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasStbmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* a, int lda, float* x, int incx)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (k < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (k > std::numeric_limits<int>::max() - 1) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_ENUM;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_ENUM;
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) return ACLBLAS_STATUS_INVALID_ENUM;
    if (lda < k + 1) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (a == nullptr || x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    cblas_stbmv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), n, k, a, lda, x, incx);
    return ACLBLAS_STATUS_SUCCESS;
}
