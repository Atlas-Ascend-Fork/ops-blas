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
#include <string>
#include "acl/acl_rt.h"
#include "cann_ops_blas_common.h"
#include "op_common/log/log.h"

inline bool IsGeamScalarZero(const float* p)
{
    return (p == nullptr) || (*p == 0.0F);
}

inline bool IsGeamScalarZero(const aclblasComplex* p)
{
    return (p == nullptr) || (p->real == 0.0F && p->imag == 0.0F);
}

inline aclblasStatus_t ValidateGeamTranspose(
    aclblasOperation_t transa, aclblasOperation_t transb, const std::string& opName)
{
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE(opName, "Invalid transa=%d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE(opName, "Invalid transb=%d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateGeamLd(
    int m, int n, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    const std::string& opName)
{
    const int minLda = (transa == ACLBLAS_OP_N) ? std::max(1, m) : std::max(1, n);
    if (lda < minLda) {
        OP_LOGE(opName, "lda=%d must be >= %d", lda, minLda);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLdb = (transb == ACLBLAS_OP_N) ? std::max(1, m) : std::max(1, n);
    if (ldb < minLdb) {
        OP_LOGE(opName, "ldb=%d must be >= %d", ldb, minLdb);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE(opName, "ldc=%d must be >= max(1, m=%d)", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamPointers(
    const T* alpha, const T* a, const T* beta, const T* b, T* c, const std::string& opName)
{
    if (alpha == nullptr || beta == nullptr || c == nullptr) {
        OP_LOGE(opName, "alpha/beta/C pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsGeamScalarZero(alpha) && a == nullptr) {
        OP_LOGE(opName, "A pointer is nullptr when alpha != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsGeamScalarZero(beta) && b == nullptr) {
        OP_LOGE(opName, "B pointer is nullptr when beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamInplace(
    const T* a, const T* b, T* c, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    const std::string& opName)
{
    if (a != nullptr && c == a && (transa != ACLBLAS_OP_N || lda != ldc)) {
        OP_LOGE(opName, "Invalid in-place C==A");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (b != nullptr && c == b && (transb != ACLBLAS_OP_N || ldb != ldc)) {
        OP_LOGE(opName, "Invalid in-place C==B");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamParams(
    const std::string& opName, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const T* alpha,
    const T* a, int lda, const T* beta, const T* b, int ldb, T* c, int ldc)
{
    aclblasStatus_t status = ValidateGeamTranspose(transa, transb, opName);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m < 0 || n < 0) {
        OP_LOGE(opName, "m=%d and n=%d must be >= 0", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    status = ValidateGeamPointers<T>(alpha, a, beta, b, c, opName);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateGeamLd(m, n, lda, ldb, ldc, transa, transb, opName);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateGeamInplace<T>(a, b, c, lda, ldb, ldc, transa, transb, opName);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t InitGeamCoreTiling(
    const std::string& opName, int n, uint32_t& uN, uint32_t& usedCoreNum, uint32_t& perCoreN, uint32_t& remainder)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(opName, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uN = static_cast<uint32_t>(n);
    usedCoreNum = std::max(1U, std::min(uN, aivCoreNum));
    if (usedCoreNum == 0U) {
        OP_LOGE(opName, "usedCoreNum is zero");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    perCoreN = uN / usedCoreNum;
    remainder = uN % usedCoreNum;
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t PrepareGeamHost(
    const std::string& opName, aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m,
    int n, const T* alpha, const T* a, int lda, const T* beta, const T* b, int ldb, T* c, int ldc, uint32_t& uN,
    uint32_t& usedCoreNum, uint32_t& perCoreN, uint32_t& remainder, bool& earlyReturn)
{
    earlyReturn = false;
    if (handle == nullptr) {
        OP_LOGE(opName, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const aclblasStatus_t status =
        ValidateGeamParams(opName, transa, transb, m, n, alpha, a, lda, beta, b, ldb, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        earlyReturn = true;
        return status;
    }
    return InitGeamCoreTiling(opName, n, uN, usedCoreNum, perCoreN, remainder);
}

template <class T, class TilingT, class FillFn, class LaunchFn>
inline aclblasStatus_t ExecuteGeamHost(
    const std::string& opName, aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m,
    int n, const T* alpha, const T* A, int lda, const T* beta, const T* B, int ldb, T* C, int ldc, FillFn fillFn,
    LaunchFn launchFn)
{
    uint32_t uN = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t perCoreN = 0U;
    uint32_t remainder = 0U;
    bool earlyReturn = false;
    const aclblasStatus_t prepSt = PrepareGeamHost(
        opName, handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc, uN, usedCoreNum, perCoreN,
        remainder, earlyReturn);
    if (prepSt != ACLBLAS_STATUS_SUCCESS || earlyReturn) {
        return prepSt;
    }

    TilingT tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = uN;
    tiling.opA = transa;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.opB = transb;
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.usedCoreNum = usedCoreNum;
    tiling.perCoreN = perCoreN;
    tiling.remainder = remainder;
    fillFn(tiling);

    uint8_t* gmC = reinterpret_cast<uint8_t*>(C);
    uint8_t* gmA = (A != nullptr) ? reinterpret_cast<uint8_t*>(const_cast<T*>(A)) : gmC;
    uint8_t* gmB = (B != nullptr) ? reinterpret_cast<uint8_t*>(const_cast<T*>(B)) : gmC;
    launchFn(gmA, gmB, gmC, tiling, usedCoreNum, handle->stream);
    if (aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) != ACL_SUCCESS) {
        OP_LOGE(opName, "Kernel launch failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
