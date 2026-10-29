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
#include <cblas.h>
#include "cher2_param.h"

inline void Cher2Golden(const Cher2Param& p, const aclblasComplex* x, const aclblasComplex* y, aclblasComplex* a)
{
    cblas_cher2(
        CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, &p.alpha, x, p.incx, y, p.incy, a,
        p.lda);
}
