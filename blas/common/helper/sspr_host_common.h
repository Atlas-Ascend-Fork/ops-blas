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

#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// Host-side entry checks shared by the arch22 and arch35 aclblasSspr implementations
// (!831 codecheck: the arch pairs duplicated the handle/n/quick-return preamble and the
// post-alpha pointer/core-count guards verbatim; the per-arch parameter validators stay
// in the arch files because their contracts intentionally differ, see below).

// Parameter validator contract: arch22 rejects an illegal uplo with INVALID_ENUM and checks
// uplo first; arch35 rejects it with INVALID_VALUE and checks alpha first. The order and the
// error code are per-arch semantics, so the validator is passed in and invoked here to keep
// the shared order: handle -> n < 0 -> n == 0 quick-return -> per-arch parameter validation.
typedef aclblasStatus_t (*SsprParamsValidator)(aclblasFillMode_t uplo, int incx, const float* alpha);

// Runs the shared entry sequence. Returns false when the caller must return ctx->st
// immediately: a validation error, or the n == 0 no-op quick-return (BLAS quick-return
// happens before the parameter validation and before any alpha dereference). Returns true
// when the caller should continue with alpha resolution, tiling and launch.
struct SsprEntryCtx {
    _aclblas_handle* h;
    aclblasStatus_t st;
};

static inline bool SsprEntryProceed(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int incx, const float* alpha,
    SsprParamsValidator validateParams, SsprEntryCtx* ctx)
{
    ctx->h = static_cast<_aclblas_handle*>(handle);
    ctx->st = ACLBLAS_STATUS_SUCCESS;
    CHECK_RET(ctx->h != nullptr, OP_LOGE("aclblasSspr", "handle is nullptr");
              ctx->st = ACLBLAS_STATUS_HANDLE_IS_NULLPTR; return false);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSspr", "n must be >= 0, got %d", n); ctx->st = ACLBLAS_STATUS_INVALID_VALUE;
              return false);
    if (n == 0) {
        return false;
    }
    ctx->st = validateParams(uplo, incx, alpha);
    return ctx->st == ACLBLAS_STATUS_SUCCESS;
}

// Runs the shared post-alpha-resolution guard sequence. Returns false when the caller must
// return *stOut immediately: the alpha == 0 no-op quick-return (x/ap are not referenced,
// issue #781), an x/ap nullptr error, or the vector core count guard. Returns true when the
// caller should continue with tiling and launch.
static inline bool SsprPostAlphaChecks(
    float alphaVal, const float* x, const float* ap, aclblasStatus_t* stOut, uint32_t* aivCoreNumOut)
{
    if (alphaVal == 0.0f) {
        *stOut = ACLBLAS_STATUS_SUCCESS;
        return false;
    }
    CHECK_RET(x != nullptr, OP_LOGE("aclblasSspr", "x must not be nullptr"); *stOut = ACLBLAS_STATUS_INVALID_VALUE;
              return false);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasSspr", "ap must not be nullptr"); *stOut = ACLBLAS_STATUS_INVALID_VALUE;
              return false);
    *aivCoreNumOut = GetAivCoreCount();
    CHECK_RET(*aivCoreNumOut != 0, OP_LOGE("aclblasSspr", "vector core count is 0");
              *stOut = ACLBLAS_STATUS_EXECUTION_FAILED; return false);
    return true;
}
