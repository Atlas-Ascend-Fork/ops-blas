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
#include "cann_ops_blas.h"

// Data operands are already on device; alpha remains a host scalar.
// Exactly one public API invocation per wrapper call. Allocation, warmup and
// measurement loops belong to the test, outside the timed kernel.
inline aclblasStatus_t Csyr2Npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    return aclblasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
}
