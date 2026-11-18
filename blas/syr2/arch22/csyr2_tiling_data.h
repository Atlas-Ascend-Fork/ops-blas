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
struct Csyr2TilingData {
    uint32_t n;    /**< matrix order (rows/columns of A) */
    uint32_t uplo; /**< triangle selector: ACLBLAS_UPPER (121) / ACLBLAS_LOWER (122), see aclblasFillMode_t */
    uint32_t lda;  /**< leading dimension of A, lda >= n */
    float ar;      /**< scalar alpha, real part */
    float ai;      /**< scalar alpha, imaginary part */
    int64_t incx;  /**< x stride (signed; negative uses the Netlib physical-base convention) */
    int64_t incy;  /**< y stride (signed; negative uses the Netlib physical-base convention) */
};
