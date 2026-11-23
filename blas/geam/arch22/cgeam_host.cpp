/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgeam_host.cpp
 * \brief Host-side implementation for aclblasCgeam (arch22 / dav-2201).
 */

#include <algorithm>
#include <cstdint>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "cgeam_kernel.h"
#include "cgeam_tiling_data.h"
#include "geam_host_common.h"

namespace {

constexpr const char* OP_TAG = "aclblasCgeam";

} // namespace

aclblasStatus_t aclblasCgeam(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
{
    return ExecuteGeamHost<aclblasComplex, CgeamTilingData>(
        OP_TAG, handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc,
        [alpha, beta](CgeamTilingData& tiling) {
            tiling.alphaR = alpha->real;
            tiling.alphaI = alpha->imag;
            tiling.alphaIsZero = IsGeamScalarZero(alpha) ? 1U : 0U;
            tiling.betaR = beta->real;
            tiling.betaI = beta->imag;
            tiling.betaIsZero = IsGeamScalarZero(beta) ? 1U : 0U;
        },
        cgeam_kernel_do);
}
