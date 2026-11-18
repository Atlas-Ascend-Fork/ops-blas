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

#include <climits>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <cblas.h>
#include <gtest/gtest.h>
#include "cann_ops_blas.h"
#if defined(__linux__)
#include <dlfcn.h>
#include "chemv_reference_config.h"
#endif

namespace chemv_test {
using ChemvReferenceFunction = decltype(&cblas_chemv);
// Keep a link-time dependency on CBLAS even with --as-needed. Linux setup
// replaces this address with the dlsym result only after verifying its owner.
inline ChemvReferenceFunction chemvReference = &cblas_chemv;

#if defined(__linux__)
inline bool VerifyReferenceSymbol(
    const char* symbol, const char* environment, const char* configuredPath, void*& address, std::string& actualPath)
{
    const char* expected = configuredPath;
    // Environment variables may confirm the pinned library, never replace it.
    const char* requested = std::getenv(environment);
    if (requested != nullptr && *requested != '\0') {
        char* requestedCanonical = realpath(requested, nullptr);
        char* configuredCanonical = realpath(configuredPath, nullptr);
        const bool matches = requestedCanonical != nullptr && configuredCanonical != nullptr &&
                             std::strcmp(requestedCanonical, configuredCanonical) == 0;
        std::free(requestedCanonical);
        std::free(configuredCanonical);
        if (!matches) {
            ADD_FAILURE() << "Reference override conflicts with pinned configuration: " << environment << "="
                          << requested << " configured=" << configuredPath;
            return false;
        }
    }
    if (expected == nullptr || *expected == '\0') {
        ADD_FAILURE() << "Missing expected reference library for " << symbol;
        return false;
    }
    address = dlsym(RTLD_DEFAULT, symbol);
    Dl_info owner{};
    if (address == nullptr || dladdr(address, &owner) == 0 || owner.dli_fname == nullptr) {
        ADD_FAILURE() << "Cannot resolve actual reference library for " << symbol;
        return false;
    }
    char* actualCanonical = realpath(owner.dli_fname, nullptr);
    char* expectedCanonical = realpath(expected, nullptr);
    if (actualCanonical == nullptr || expectedCanonical == nullptr) {
        std::free(actualCanonical);
        std::free(expectedCanonical);
        ADD_FAILURE() << "Cannot canonicalize reference paths for " << symbol << ": actual=" << owner.dli_fname
                      << " expected=" << expected;
        return false;
    }
    actualPath = actualCanonical;
    const std::string expectedPath = expectedCanonical;
    std::free(actualCanonical);
    std::free(expectedCanonical);
    if (actualPath != expectedPath) {
        ADD_FAILURE() << "Reference binding mismatch for " << symbol << ": actual=" << actualPath
                      << " expected=" << expectedPath;
        return false;
    }
    return true;
}
#endif

inline bool VerifyReferenceBinding()
{
#if defined(__linux__)
    void* cblasAddress = nullptr;
    void* blasAddress = nullptr;
    std::string cblasPath;
    std::string blasPath;
    const bool cblasVerified = VerifyReferenceSymbol(
        "cblas_chemv", "CHEMV_CBLAS_LIBRARY", CHEMV_EXPECTED_CBLAS_LIBRARY, cblasAddress, cblasPath);
    if (!cblasVerified ||
        !VerifyReferenceSymbol("chemv_", "CHEMV_BLAS_LIBRARY", CHEMV_EXPECTED_BLAS_LIBRARY, blasAddress, blasPath)) {
        chemvReference = nullptr;
        return false;
    }
    // POSIX dlsym returns the callable address. All golden calls use this exact
    // verified address, not an unverified direct/PLT call to cblas_chemv.
    chemvReference = reinterpret_cast<ChemvReferenceFunction>(cblasAddress);
    std::cout << "[REFERENCE] {\"cblas\":" << std::quoted(cblasPath) << ",\"blas\":" << std::quoted(blasPath)
              << ",\"verified\":true}" << std::endl;
    return true;
#elif defined(_WIN32)
    // Local CPU validation links a clearly labelled CBLAS test double. This is
    // never reported as Linux binding verification or Netlib 3.10.0 acceptance.
    chemvReference = &cblas_chemv;
    std::cout << "[REFERENCE] {\"mode\":\"cpu_test_double\",\"verified\":false}" << std::endl;
    return true;
#else
    ADD_FAILURE() << "Reference binding verification is supported only on Linux; local CPU doubles use Windows";
    return false;
#endif
}

inline aclblasStatus_t ValidateCpuAttributes(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < n || incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN || alpha == nullptr ||
        beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasChemv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const auto status = ValidateCpuAttributes(uplo, n, lda, incx, incy, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool alphaZero = alpha->real == 0.0f && alpha->imag == 0.0f;
    // Match the no-op API contract: no device data pointers are required.
    if (alphaZero && beta->real == 1.0f && beta->imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (y == nullptr || (!alphaZero && (A == nullptr || x == nullptr))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (chemvReference == nullptr) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // CBLAS receives valid dummy addresses for unused A/x when alpha=0.
    const aclblasComplex dummy{};
    chemvReference(
        CblasColMajor, uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, n, alpha, A == nullptr ? &dummy : A, lda,
        x == nullptr ? &dummy : x, incx, beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace chemv_test
