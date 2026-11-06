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
#include <iterator>
#include "log/log.h"
#include "cann_ops_blas.h"

namespace gemm_ex {
// Dimensions have already passed nonnegative validation. The two-dimensional
// product fits uint64_t for int inputs; divide before comparing the third.
inline bool VolumeAtMost(int m, int n, int k, uint64_t limit)
{
    return k == 0 || static_cast<uint64_t>(m) * n <= limit / static_cast<uint64_t>(k);
}

inline bool IsKnownType(aclDataType type)
{
    // aclDataType is sparse; neither a range check nor a default FP32 cast is valid.
    constexpr aclDataType known[] = {
        ACL_FLOAT,       ACL_FLOAT16,     ACL_INT8,        ACL_INT32,         ACL_UINT8,       ACL_INT16,
        ACL_UINT16,      ACL_UINT32,      ACL_INT64,       ACL_UINT64,        ACL_DOUBLE,      ACL_BOOL,
        ACL_STRING,      ACL_COMPLEX64,   ACL_COMPLEX128,  ACL_BF16,          ACL_INT4,        ACL_UINT1,
        ACL_COMPLEX32,   ACL_HIFLOAT8,    ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT6_E3M2,
        ACL_FLOAT6_E2M3, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2};
    return std::find(std::begin(known), std::end(known), type) != std::end(known);
}
inline bool IsOperation(aclblasOperation_t op)
{
    return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C;
}
inline bool IsZero(float value) { return value == 0.0f; }
inline bool IsZero(aclblasComplex value) { return value.real == 0.0f && value.imag == 0.0f; }

inline bool ValidDimensions(
    aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, int lda, int ldb, int ldc)
{
    return m >= 0 && n >= 0 && k >= 0 && lda >= std::max(1, ta == ACLBLAS_OP_N ? m : k) &&
           ldb >= std::max(1, tb == ACLBLAS_OP_N ? k : n) && ldc >= std::max(1, m);
}
inline aclblasStatus_t ValidateTypes(aclDataType at, aclDataType bt, aclDataType ct, aclDataType supported)
{
    if (!IsKnownType(at) || !IsKnownType(bt) || !IsKnownType(ct)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return at == supported && bt == supported && ct == supported ? ACLBLAS_STATUS_SUCCESS :
                                                                   ACLBLAS_STATUS_NOT_SUPPORTED;
}
template <class Scalar>
inline bool ValidPointers(int k, Scalar alpha, Scalar beta, const void* a, const void* b, const void* c)
{
    return !(k > 0 && !IsZero(alpha) && (!a || !b)) && !(!IsZero(beta) && !c);
}

template <class Scalar>
inline aclblasStatus_t Validate(
    aclblasHandle_t handle, aclblasOperation_t ta, aclblasOperation_t tb, int m, int n, int k, const Scalar* alpha,
    const void* a, aclDataType at, int lda, const void* b, aclDataType bt, int ldb, const Scalar* beta, void* c,
    aclDataType ct, int ldc, aclDataType supported, const char* api)
{
    if (!handle) {
        OP_LOGE(api, "%s: handle is nullptr", api);
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!ValidDimensions(ta, tb, m, n, k, lda, ldb, ldc)) {
        OP_LOGE(api, "%s: invalid dimensions m=%d n=%d k=%d lda=%d ldb=%d ldc=%d", api, m, n, k, lda, ldb, ldc);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!alpha || !beta) {
        OP_LOGE(api, "%s: alpha or beta is nullptr", api);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto typeStatus = ValidateTypes(at, bt, ct, supported);
    if (typeStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            api, "%s: unsupported or invalid dtype A=%d B=%d C=%d, status=%d", api, static_cast<int>(at),
            static_cast<int>(bt), static_cast<int>(ct), static_cast<int>(typeStatus));
        return typeStatus;
    }
    if (!IsOperation(ta) || !IsOperation(tb)) {
        OP_LOGE(api, "%s: invalid operation transA=%d transB=%d", api, static_cast<int>(ta), static_cast<int>(tb));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (!ValidPointers(k, *alpha, *beta, a, b, c)) {
        OP_LOGE(api, "%s: required matrix pointer is nullptr", api);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace gemm_ex
