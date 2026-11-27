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
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// LAPACK 3.12.0 ctrttp_ (CBLAS has no cblas_ctrttp; the routine is a LAPACK format
// conversion). Declared here exactly like the real-precision strttp_ golden next door.
extern "C" {
void ctrttp_(const char* uplo, const int* n, const aclblasComplex* a, const int* lda, aclblasComplex* ap, int* info);
}

inline aclblasStatus_t aclblasCtrttp_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* A, int lda, aclblasComplex* AP)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr || AP == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    char uploChar = ToLapackUplo(uplo);
    int info = 0;
    ctrttp_(&uploChar, &n, A, &lda, AP, &info);
    return (info == 0) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// Byte-wise comparison of complex64 elements, real and imaginary parts judged separately as
// the task book requires. Verifier's EXACT mode treats NaN==NaN and +0==-0 as equal, which is
// weaker than "bit-exact, NaN payload and Inf sign included", so each 4-byte half is compared
// with memcmp instead.
inline bool CtrttpVerifyBitExact(
    const aclblasComplex* actual, const aclblasComplex* golden, uint64_t count, const std::string& caseId,
    const char* label, uint64_t* realMismatchOut = nullptr, uint64_t* imagMismatchOut = nullptr)
{
    uint64_t realMismatch = 0;
    uint64_t imagMismatch = 0;
    uint64_t firstReal = 0;
    uint64_t firstImag = 0;
    for (uint64_t i = 0; i < count; i++) {
        if (std::memcmp(&actual[i].real, &golden[i].real, sizeof(float)) != 0) {
            if (realMismatch == 0) {
                firstReal = i;
            }
            realMismatch++;
        }
        if (std::memcmp(&actual[i].imag, &golden[i].imag, sizeof(float)) != 0) {
            if (imagMismatch == 0) {
                firstImag = i;
            }
            imagMismatch++;
        }
    }

    std::cout << "[" << caseId << "] " << label << " " << ((realMismatch + imagMismatch) == 0 ? "PASSED" : "FAILED")
              << " (bit-exact, real " << realMismatch << "/" << count << ", imag " << imagMismatch << "/" << count
              << ")";
    if (realMismatch != 0) {
        std::cout << std::setprecision(9) << " first real mismatch at " << firstReal << " actual="
                  << actual[firstReal].real << " golden=" << golden[firstReal].real;
    }
    if (imagMismatch != 0) {
        std::cout << std::setprecision(9) << " first imag mismatch at " << firstImag << " actual="
                  << actual[firstImag].imag << " golden=" << golden[firstImag].imag;
    }
    std::cout << std::endl;
    if (realMismatchOut != nullptr) {
        *realMismatchOut = realMismatch;
    }
    if (imagMismatchOut != nullptr) {
        *imagMismatchOut = imagMismatch;
    }
    return (realMismatch + imagMismatch) == 0;
}

