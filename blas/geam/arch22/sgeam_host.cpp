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
 * \file sgeam_host.cpp
 * \brief Host-side implementation for aclblasSgeam (arch22 / dav-2201).
 */

#include <algorithm>
#include <cstdint>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "geam_host_common.h"
#include "sgeam_kernel.h"
#include "sgeam_tiling_data.h"

namespace {

constexpr const char* OP_TAG = "aclblasSgeam";

} // namespace

aclblasStatus_t aclblasSgeam(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const float* alpha,
    const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
{
    return ExecuteGeamHost<float, SgeamTilingData>(
        OP_TAG, handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc,
        [alpha, beta](SgeamTilingData& tiling) {
            tiling.alpha = *alpha;
            tiling.alphaIsZero = (*alpha == 0.0F) ? 1U : 0U;
            tiling.beta = *beta;
            tiling.betaIsZero = (*beta == 0.0F) ? 1U : 0U;
        },
        sgeam_kernel_do);
}
