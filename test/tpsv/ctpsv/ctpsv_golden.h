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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline size_t CtpsvPackedUpperIdxCpu(int i, int j) { return static_cast<size_t>(i + j * (j + 1) / 2); }

inline size_t CtpsvPackedLowerIdxCpu(int i, int j, int n) { return static_cast<size_t>(i + (2 * n - j - 1) * j / 2); }

inline aclblasStatus_t aclblasCtpsv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* ap, aclblasComplex* x, int incx)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (ap == nullptr || x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    cblas_ctpsv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), n, ap, x, incx);
    return ACLBLAS_STATUS_SUCCESS;
}
