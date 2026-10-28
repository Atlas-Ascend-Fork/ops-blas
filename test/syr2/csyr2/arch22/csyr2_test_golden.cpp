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
 * \file csyr2_test_golden.cpp
 * \brief Netlib ssyr2 列循环的复数自实现。不跳过 x[j]=y[j]=0 列，Inf/NaN 按 IEEE 乘加传播。
 */

#include "csyr2_golden.h"

#include "csyr2_tiling.h"

namespace csyr2_arch22_test {

void Csyr2Golden(
    aclblasFillMode_t uplo,
    int n,
    aclblasComplex alpha,
    const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda)
{
    if (n == 0 || (alpha.real == 0.0f && alpha.imag == 0.0f)) {
        return;
    }
    if (x == nullptr || y == nullptr || A == nullptr || incx == 0 || incy == 0) {
        return;
    }

    const int kx0 = Csyr2Kx0(n, incx);
    const int ky0 = Csyr2Ky0(n, incy);

    for (int j = 0; j < n; ++j) {
        const aclblasComplex yj = y[ky0 + j * incy];
        const aclblasComplex xj = x[kx0 + j * incx];
        /* temp1 = alpha * y(j)；temp2 = alpha * x(j) */
        const aclblasComplex temp1 = {
            alpha.real * yj.real - alpha.imag * yj.imag,
            alpha.real * yj.imag + alpha.imag * yj.real};
        const aclblasComplex temp2 = {
            alpha.real * xj.real - alpha.imag * xj.imag,
            alpha.real * xj.imag + alpha.imag * xj.real};

        const int iBegin = (uplo == ACLBLAS_UPPER) ? 0 : j;
        const int iEnd = (uplo == ACLBLAS_UPPER) ? (j + 1) : n;
        for (int i = iBegin; i < iEnd; ++i) {
            const aclblasComplex xi = x[kx0 + i * incx];
            const aclblasComplex yi = y[ky0 + i * incy];
            /* A(i,j) += x(i)*temp1 + y(i)*temp2，对角虚部普通累加 */
            const float zr1 = xi.real * temp1.real - xi.imag * temp1.imag;
            const float zi1 = xi.real * temp1.imag + xi.imag * temp1.real;
            const float zr2 = yi.real * temp2.real - yi.imag * temp2.imag;
            const float zi2 = yi.real * temp2.imag + yi.imag * temp2.real;
            A[i + j * lda].real += zr1 + zr2;
            A[i + j * lda].imag += zi1 + zi2;
        }
    }
}

}  // namespace csyr2_arch22_test
