/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Independently written baseline; API / integration references in docs/SOURCES.md.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "cscal_plan.h"

void cscal_kernel_do(uint8_t* x, const CscalTilingData& t, void* stream);

static_assert(std::is_standard_layout<aclblasComplex>::value, "complex must be standard-layout");
static_assert(
    sizeof(aclblasComplex) == 8 && offsetof(aclblasComplex, real) == 0 && offsetof(aclblasComplex, imag) == 4,
    "expected interleaved float32 complex");

aclblasStatus_t aclblasCscal(aclblasHandle_t handle, int n, const aclblasComplex* alpha, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCscal", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // Deliberate priority: valid handle -> legal no-op -> pointer checks -> alpha==1.
    if (n <= 0 || incx <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (alpha == nullptr || x == nullptr) {
        OP_LOGE("aclblasCscal", "alpha or x is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const float ar = alpha->real, ai = alpha->imag; // HOST pointer, task contract.
    if (ar == 1.0f && ai == 0.0f)
        return ACLBLAS_STATUS_SUCCESS;
    uint64_t bytes;
    if (!cscal950::StorageBytes(n, incx, bytes)) {
        OP_LOGE("aclblasCscal", "storage size overflows, n=%d, incx=%d", n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(x);
    if (base > UINTPTR_MAX - (bytes - 1)) {
        OP_LOGE("aclblasCscal", "x address range overflows, n=%d, incx=%d", n, incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CscalTilingData t{};
    if (!cscal950::MakePlan(n, incx, ar, ai, GetAivCoreCount(), t)) {
        OP_LOGE("aclblasCscal", "failed to create tiling plan, n=%d, incx=%d", n, incx);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // No GM workspace, Host index table, allocation, memcpy or stream synchronization.
    // Like the surrounding direct-invoke operators, asynchronous device failures are
    // observed by the caller at stream synchronization, NOT by this SUCCESS return.
    cscal_kernel_do(reinterpret_cast<uint8_t*>(x), t, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
