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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t ValidateGetrfParams(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, int* PivotArray, int* infoArray,
    int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0 || batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (Aarray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (PivotArray != nullptr && infoArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool ComplexIsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }

inline float ComplexAbs1(const aclblasComplex& value) { return std::abs(value.real) + std::abs(value.imag); }

inline aclblasComplex ComplexDivide(const aclblasComplex& numerator, const aclblasComplex& denominator)
{
    const float denom = denominator.real * denominator.real + denominator.imag * denominator.imag;
    return {
        (numerator.real * denominator.real + numerator.imag * denominator.imag) / denom,
        (numerator.imag * denominator.real - numerator.real * denominator.imag) / denom};
}

inline aclblasComplex ComplexMultiply(const aclblasComplex& lhs, const aclblasComplex& rhs)
{
    return {lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real};
}

inline void FactorizeSingleMatrix(aclblasComplex* A, int n, int lda, int* pivotArray, int& info, bool usePivot)
{
    info = 0;
    for (int k = 0; k < n; k++) {
        int pivotRow = k;
        if (usePivot) {
            float pivotAbs = 0.0f;
            for (int i = k; i < n; i++) {
                const float candidateAbs = ComplexAbs1(A[i + k * lda]);
                if (candidateAbs > pivotAbs) {
                    pivotAbs = candidateAbs;
                    pivotRow = i;
                }
            }
            pivotArray[k] = pivotRow + 1;
        }

        aclblasComplex pivot = A[pivotRow + k * lda];
        if (ComplexIsZero(pivot)) {
            if (info == 0) {
                info = k + 1;
            }
            continue;
        }

        if (pivotRow != k) {
            for (int j = 0; j < n; j++) {
                std::swap(A[k + j * lda], A[pivotRow + j * lda]);
            }
        }

        const aclblasComplex diagonal = A[k + k * lda];
        for (int i = k + 1; i < n; i++) {
            A[i + k * lda] = ComplexDivide(A[i + k * lda], diagonal);
        }

        for (int j = k + 1; j < n; j++) {
            for (int i = k + 1; i < n; i++) {
                const aclblasComplex multiplier = A[i + k * lda];
                const aclblasComplex product = ComplexMultiply(multiplier, A[k + j * lda]);
                A[i + j * lda].real -= product.real;
                A[i + j * lda].imag -= product.imag;
            }
        }
    }
}

inline aclblasStatus_t aclblasCgetrfBatched_cpu(
    aclblasHandle_t handle, int n, aclblasComplex* const Aarray[], int lda, int* PivotArray, int* infoArray,
    int batchSize)
{
    aclblasStatus_t validRet = ValidateGetrfParams(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool usePivot = (PivotArray != nullptr);
    for (int b = 0; b < batchSize; b++) {
        int info = 0;
        int* piv = usePivot ? PivotArray + b * n : nullptr;
        FactorizeSingleMatrix(Aarray[b], n, lda, piv, info, usePivot);
        if (infoArray != nullptr) {
            infoArray[b] = info;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
