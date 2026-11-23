/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "ctpttr_kernel.h"
#include "common/helper/host_utils.h"
#include "log/log.h"

static_assert(sizeof(aclblasComplex) == 8U, "complex64 ABI mismatch");

extern "C" aclblasStatus_t aclblasCtpttr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* AP, aclblasComplex* A,
    const int lda)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCtpttr", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCtpttr", "n=%d, expected >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCtpttr", "uplo=%d, expected 121(UPPER) or 122(LOWER)", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCtpttr", "lda=%d, expected >= max(1, n=%d)", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (AP == nullptr) {
        OP_LOGE("aclblasCtpttr", "AP must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasCtpttr", "A must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Non-overlap is a caller precondition, not an in-place operation.
    aclrtStream stream = nullptr;
    const auto status = aclblasGetStream(handle, &stream);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCtpttr", "aclblasGetStream failed, status=%d", static_cast<int>(status));
        return status;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        OP_LOGE("aclblasCtpttr", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const auto tiling =
        MakeCtpttrTiling(static_cast<uint32_t>(n), static_cast<uint32_t>(lda), uplo == ACLBLAS_UPPER, cores);
    // issue #1181: a failed launch must not be reported as SUCCESS with A stale.
    const aclError launch =
        ctpttr_kernel_do(reinterpret_cast<const uint8_t*>(AP), reinterpret_cast<uint8_t*>(A), tiling, stream);
    CHECK_RET(launch == ACL_SUCCESS, OP_LOGE("aclblasCtpttr", "ctpttr_kernel_do failed: %d", launch);
              return ACLBLAS_STATUS_EXECUTION_FAILED);
    return ACLBLAS_STATUS_SUCCESS;
}
