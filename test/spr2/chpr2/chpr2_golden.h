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

#include <vector>
#include <cblas.h>
#include "chpr2_param.h"

inline void Chpr2Golden(
    aclblasFillMode_t uplo, int n, const Complex& alpha, const std::vector<Complex>& x, int incx,
    const std::vector<Complex>& y, int incy, std::vector<Complex>& ap)
{
    cblas_chpr2(
        CblasColMajor, uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, n, &alpha, x.data(), incx, y.data(), incy,
        ap.data());
}
