/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_EX_GOLDEN_H
#define GEMM_GROUPED_BATCHED_EX_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cblas.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "dtype_cast.h"

inline uint16_t FloatToHalf(float val) { return blas_common::FloatToHalf(val); }

inline float HalfToFloat(uint16_t h) { return blas_common::HalfToFloat(h); }

inline uint16_t FloatToBfloat(float val) { return blas_common::FloatToBf16(val); }

inline float BfloatToFloat(uint16_t b) { return blas_common::Bf16ToFloat(b); }

inline float GroupedFp8ToFloat(uint8_t raw, bool e5m2)
{
    uint32_t sign = static_cast<uint32_t>(raw & 0x80u) << 24;
    uint32_t exponent = e5m2 ? ((raw >> 2) & 0x1fu) : ((raw >> 3) & 0x0fu);
    uint32_t mantissa = e5m2 ? (raw & 0x3u) : (raw & 0x7u);
    if (exponent == 0) {
        float value = static_cast<float>(mantissa) * (e5m2 ? 0.0000152587890625f : 0.001953125f);
        return (raw & 0x80u) != 0 ? -value : value;
    }
    uint32_t bits;
    if (e5m2) {
        if (exponent == 0x1fu) {
            bits = sign | 0x7f800000u | (mantissa << 21);
        } else {
            bits = sign | ((exponent + 112u) << 23) | (mantissa << 21);
        }
    } else {
        if (exponent == 0x0fu && mantissa == 0x7u) {
            bits = sign | 0x7fc00000u;
        } else {
            bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
        }
    }
    return blas_common::BitCast<float>(bits);
}

inline float DtypeElementToFloat(const void* data, aclDataType dtype, int64_t idx)
{
    if (dtype == ACL_FLOAT16) {
        return HalfToFloat(reinterpret_cast<const uint16_t*>(data)[idx]);
    }
    if (dtype == ACL_BF16) {
        return BfloatToFloat(reinterpret_cast<const uint16_t*>(data)[idx]);
    }
    if (dtype == ACL_FLOAT8_E4M3FN) {
        return GroupedFp8ToFloat(reinterpret_cast<const uint8_t*>(data)[idx], false);
    }
    if (dtype == ACL_FLOAT8_E5M2) {
        return GroupedFp8ToFloat(reinterpret_cast<const uint8_t*>(data)[idx], true);
    }
    return reinterpret_cast<const float*>(data)[idx];
}

inline float RoundTripOutput(float val, aclDataType Ctype)
{
    if (Ctype == ACL_FLOAT16)
        return HalfToFloat(FloatToHalf(val));
    if (Ctype == ACL_BF16)
        return BfloatToFloat(FloatToBfloat(val));
    return val;
}

inline bool IsTransposed(aclblasOperation_t op) { return op == ACLBLAS_OP_T || op == ACLBLAS_OP_C; }

inline void ScaleGroupedColumns(int m, int n, float beta, float* c, int ldc)
{
    if (beta == 1.0f)
        return;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const auto index = static_cast<size_t>(col) * ldc + row;
            c[index] = beta == 0.0f ? 0.0f : beta * c[index];
        }
    }
}

inline void GemmGroupedBatchedCpuOne(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, float alpha, float beta, const float* A,
    int lda, const float* B, int ldb, float* C, int ldc, aclDataType Ctype)
{
    if (Ctype == ACL_FLOAT) {
        if (m == 0 || n == 0)
            return;
        if (k == 0 || alpha == 0.0f) {
            ScaleGroupedColumns(m, n, beta, C, ldc);
        } else {
            cblas_sgemm(
                CblasColMajor, IsTransposed(transA) ? CblasTrans : CblasNoTrans,
                IsTransposed(transB) ? CblasTrans : CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return;
    }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                const int64_t aRow = IsTransposed(transA) ? p : i;
                const int64_t aCol = IsTransposed(transA) ? i : p;
                const int64_t bRow = IsTransposed(transB) ? j : p;
                const int64_t bCol = IsTransposed(transB) ? p : j;
                sum += A[aRow + aCol * lda] * B[bRow + bCol * ldb];
            }
            const int64_t cIdx = i + static_cast<int64_t>(j) * ldc;
            C[cIdx] = RoundTripOutput(alpha * sum + beta * C[cIdx], Ctype);
        }
    }
}

inline bool IsValidCpuOperation(aclblasOperation_t op)
{
    return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C;
}

inline aclblasStatus_t ValidateCpuGroup(
    int group, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[], const int mArray[],
    const int nArray[], const int kArray[], const int ldaArray[], const int ldbArray[], const int ldcArray[],
    const int groupSize[])
{
    if (!IsValidCpuOperation(transaArray[group]) || !IsValidCpuOperation(transbArray[group])) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (mArray[group] < 0 || nArray[group] < 0 || kArray[group] < 0 || groupSize[group] < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int minLda = IsTransposed(transaArray[group]) ? std::max(1, kArray[group]) : std::max(1, mArray[group]);
    int minLdb = IsTransposed(transbArray[group]) ? std::max(1, nArray[group]) : std::max(1, kArray[group]);
    if (ldaArray[group] < minLda || ldbArray[group] < minLdb || ldcArray[group] < std::max(1, mArray[group])) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline int GroupInstanceStart(int group, const int groupSize[])
{
    int start = 0;
    for (int i = 0; i < group; ++i) {
        start += groupSize[i];
    }
    return start;
}

inline void RunCpuGroup(
    int group, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[], const int mArray[],
    const int nArray[], const int kArray[], const float alphaArray[], const std::vector<std::vector<uint8_t>>& aRawData,
    aclDataType Atype, const int ldaArray[], const std::vector<std::vector<uint8_t>>& bRawData, aclDataType Btype,
    const int ldbArray[], const float betaArray[], std::vector<std::vector<float>>& cGoldenOut, aclDataType Ctype,
    const int ldcArray[], const int groupSize[])
{
    int m = mArray[group];
    int n = nArray[group];
    int k = kArray[group];
    if (m == 0 || n == 0) {
        return;
    }
    int colsA = IsTransposed(transaArray[group]) ? m : k;
    int colsB = IsTransposed(transbArray[group]) ? k : n;
    int start = GroupInstanceStart(group, groupSize);
    for (int inst = 0; inst < groupSize[group]; ++inst) {
        int idx = start + inst;
        int64_t aSize = static_cast<int64_t>(ldaArray[group]) * colsA;
        int64_t bSize = static_cast<int64_t>(ldbArray[group]) * colsB;
        int64_t cSize = static_cast<int64_t>(ldcArray[group]) * n;
        std::vector<float> aFP32(static_cast<size_t>(aSize));
        std::vector<float> bFP32(static_cast<size_t>(bSize));
        std::vector<float> cFP32(static_cast<size_t>(cSize));
        for (int64_t i = 0; i < aSize; ++i) {
            aFP32[static_cast<size_t>(i)] = DtypeElementToFloat(aRawData[idx].data(), Atype, i);
        }
        for (int64_t i = 0; i < bSize; ++i) {
            bFP32[static_cast<size_t>(i)] = DtypeElementToFloat(bRawData[idx].data(), Btype, i);
        }
        for (int64_t i = 0; i < cSize; ++i) {
            cFP32[static_cast<size_t>(i)] = RoundTripOutput(cGoldenOut[idx][static_cast<size_t>(i)], Ctype);
        }
        GemmGroupedBatchedCpuOne(
            transaArray[group], transbArray[group], m, n, k, alphaArray[group], betaArray[group], aFP32.data(),
            ldaArray[group], bFP32.data(), ldbArray[group], cFP32.data(), ldcArray[group], Ctype);
        cGoldenOut[idx] = cFP32;
    }
}

inline aclblasStatus_t aclblasGemmGroupedBatchedEx_cpu(
    aclblasHandle_t handle, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[],
    const int mArray[], const int nArray[], const int kArray[], const float* alphaArray,
    const std::vector<std::vector<uint8_t>>& aRawData, aclDataType Atype, const int ldaArray[],
    const std::vector<std::vector<uint8_t>>& bRawData, aclDataType Btype, const int ldbArray[], const float* betaArray,
    std::vector<std::vector<float>>& cGoldenOut, aclDataType Ctype, const int ldcArray[], int groupCount,
    const int groupSize[], aclblasComputeType_t /*computeType*/)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (transaArray == nullptr || transbArray == nullptr || mArray == nullptr || nArray == nullptr ||
        kArray == nullptr || alphaArray == nullptr || betaArray == nullptr || ldaArray == nullptr ||
        ldbArray == nullptr || ldcArray == nullptr || groupSize == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (groupCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    for (int g = 0; g < groupCount; g++) {
        aclblasStatus_t status = ValidateCpuGroup(
            g, transaArray, transbArray, mArray, nArray, kArray, ldaArray, ldbArray, ldcArray, groupSize);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    if (groupCount == 0)
        return ACLBLAS_STATUS_SUCCESS;

    for (int g = 0; g < groupCount; g++) {
        RunCpuGroup(
            g, transaArray, transbArray, mArray, nArray, kArray, alphaArray, aRawData, Atype, ldaArray, bRawData, Btype,
            ldbArray, betaArray, cGoldenOut, Ctype, ldcArray, groupSize);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GEMM_GROUPED_BATCHED_EX_GOLDEN_H
