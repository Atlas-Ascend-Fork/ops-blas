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

#include <complex>
#include <cstddef>
#include "cann_ops_blas.h"

// todo: 优化 cmake 让 family/dtype/arch/test.cpp 可以直接找到 family/golden.h

// std::conj(float) 返回 complex<float> ， complex 不能自动转换成 float
// 这里定义一个直接返回 float 的 conj
template <class T>
constexpr T conjugate(T x)
{
    return x;
}
template <class T>
constexpr std::complex<T> conjugate(std::complex<T> x)
{
    return std::conj(x);
}

static_assert(conjugate(1.0f) == 1.0f);
// std::conj is constexpr since c++20
// using namespace std::complex_literals;
// add this code if upgrade to c++20 `static_assert(conjugate(1.0if) == -1.0if);`

template <class T>
inline void aclblasGeam_cpu(
    aclblasOperation_t transa, aclblasOperation_t transb, std::size_t m, std::size_t n, T alpha, const T* A,
    std::size_t lda, T beta, const T* B, std::size_t ldb, T* C, std::size_t ldc)
{
    for (std::size_t j = 0; j < n; j++) {
        for (std::size_t i = 0; i < m; i++) {
            // maybe A == C or B == C
            T cij = 0;
            if (alpha != 0.0f) {
                // ignore nan in A, when alpha == 0
                T aij;
                if (transa == ACLBLAS_OP_N) {
                    aij = A[i + j * lda];
                } else {
                    aij = A[j + i * lda];
                }
                if (transa == ACLBLAS_OP_C) {
                    aij = conjugate(aij);
                }
                cij += alpha * aij;
            }
            if (beta != 0.0f) {
                T bij;
                if (transb == ACLBLAS_OP_N) {
                    bij = B[i + j * ldb];
                } else {
                    bij = B[j + i * ldb];
                }
                if (transb == ACLBLAS_OP_C) {
                    bij = conjugate(bij);
                }
                cij += beta * bij;
            }
            C[i + j * ldc] = cij;
        }
    }
}
