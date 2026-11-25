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

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "blasLtMatmul_param.h"
#include "dtype_cast.h"

// ── MXFP8 dequantize ──

inline float dequantMxfp8E4m3(uint8_t fp8Val, float scaleVal)
{
    uint32_t sign = (fp8Val >> 7) & 1;
    uint32_t exp  = (fp8Val >> 3) & 0xF;
    uint32_t mant = fp8Val & 0x7;
    float value;
    if (exp == 0 && mant == 0) {
        value = 0.0f;
    } else if (exp == 15) {
        value = 0.0f;
    } else if (exp == 0) {
        value = static_cast<float>(mant) / 8.0f * std::pow(2.0f, -6);
    } else {
        value = (1.0f + static_cast<float>(mant) / 8.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f);
    }
    if (sign) value = -value;
    return value * scaleVal;
}

// ── MXFP4 dequantize ──

inline void dequantMxfp4E2m1(uint8_t packedByte, float scaleVal, float& low, float& high)
{
    auto dequantSingle = [](uint8_t nibble, float scale) -> float {
        uint32_t sign = (nibble >> 3) & 1;
        uint32_t exp  = (nibble >> 1) & 0x3;
        uint32_t mant = nibble & 0x1;
        float value;
        // OCP MXFP4 E2M1 (bias=1): exp=0,m=0->0; exp=0,m=1->0.5; exp>=1 -> (1+m/2)*2^(exp-1)
        if (exp == 0) {
            value = (mant == 0) ? 0.0f : static_cast<float>(mant) * 0.5f;
        } else {
            value = (1.0f + static_cast<float>(mant) * 0.5f) * std::pow(2.0f, static_cast<float>(exp) - 1.0f);
        }
        if (sign) value = -value;
        return value * scale;
    };
    // Packed layout matches gen_data.py: first fp4 in low nibble, second in high nibble.
    low  = dequantSingle(packedByte & 0xF, scaleVal);
    high = dequantSingle((packedByte >> 4) & 0xF, scaleVal);
}

inline float dequantMxfp8E8m0(uint8_t fp8Val)
{
    uint32_t exp = fp8Val;
    if (exp == 0) return 0.0f;
    if (exp == 255) return INFINITY;
    return std::pow(2.0f, static_cast<float>(exp) - 127.0f);
}

// ── Dequantize MXFP8 matrix to FP32 ──

inline float lookupMxfp8Scale(
    const float* scale, int idxMinor, int idxK, int kDim, int minorDim, bool scaleKMajor)
{
    const int block = idxK / 64;
    const int sub = (idxK % 64) / 32;
    if (scaleKMajor) {
        // Layout [ceil(k/64), minorDim, 2]: stride between K-blocks is minorDim * 2
        return scale[block * minorDim * 2 + idxMinor * 2 + sub];
    }
    const int stride = mxScaleStrideAlongK(kDim);
    return scale[idxMinor * stride + block * 2 + sub];
}

inline void dequantMxfp8MatrixToFP32(
    const uint8_t* data, const float* scale,
    int rows, int cols, int ld,
    aclDataType dtype, bool scaleKMajor, int kDimForScale,
    std::vector<float>& out)
{
    out.resize(static_cast<size_t>(rows) * cols, 0.0f);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uint8_t rawVal = data[i * ld + j];
            const int idxMinor = scaleKMajor ? j : i;
            const int idxK = scaleKMajor ? i : j;
            const int minorDim = scaleKMajor ? cols : rows;
            float scaleVal = lookupMxfp8Scale(scale, idxMinor, idxK, kDimForScale, minorDim, scaleKMajor);
            float fp32Val = dequantMxfp8E4m3(rawVal, scaleVal);
            out[static_cast<size_t>(i) * cols + j] = fp32Val;
        }
    }
}

// ── Dequantize MXFP4 matrix to FP32 ──

inline void dequantMxfp4MatrixToFP32(
    const uint8_t* data, const float* scale,
    int rows, int logicalCols, int ldBytes,
    aclDataType dtype, bool scaleKMajor, int kDimForScale,
    std::vector<float>& out)
{
    out.resize(static_cast<size_t>(rows) * logicalCols, 0.0f);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < logicalCols; j++) {
            int byteIdx = j / 2;
            int nibbleIdx = j % 2;
            uint8_t packedByte = data[i * ldBytes + byteIdx];
            const int idxMinor = scaleKMajor ? j : i;
            const int idxK = scaleKMajor ? i : j;
            const int minorDim = scaleKMajor ? logicalCols : rows;
            float scaleVal = lookupMxfp8Scale(scale, idxMinor, idxK, kDimForScale, minorDim, scaleKMajor);
            float low, high;
            dequantMxfp4E2m1(packedByte, scaleVal, low, high);
            float fp32Val = (nibbleIdx == 0) ? low : high;
            out[static_cast<size_t>(i) * logicalCols + j] = fp32Val;
        }
    }
}

// ── BF16 ↔ FP32 conversion ──

inline float bf16ToFP32(uint16_t bf16Val)
{
    uint32_t fp32Bits = static_cast<uint32_t>(bf16Val) << 16;
    float result;
    memcpy(&result, &fp32Bits, sizeof(float));
    return result;
}

inline uint16_t fp32ToBF16(float fp32Val)
{
    return blas_common::FloatToBf16(fp32Val);
}

// ── CPU Golden ──

inline aclblasStatus_t aclblasLtMatmul_cpu(
    aclblasLtHandle_t lightHandle,
    aclblasLtMatmulDesc_t computeDesc,
    const float* alpha,
    const void* A_host,
    aclblasLtMatrixLayout_t Adesc,
    const void* B_host,
    aclblasLtMatrixLayout_t Bdesc,
    const float* beta,
    const float* C_host,
    aclblasLtMatrixLayout_t Cdesc,
    float* D_host,
    aclblasLtMatrixLayout_t Ddesc,
    const void* scaleA_host,
    const void* scaleB_host)
{
    if (lightHandle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (computeDesc == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (Adesc == nullptr || Bdesc == nullptr || Cdesc == nullptr || Ddesc == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;

    // Read descriptor attributes
    int32_t transAVal = ACLBLAS_OP_N, transBVal = ACLBLAS_OP_N;
    size_t sizeWritten = 0;
    aclblasLtMatmulDescGetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSA, &transAVal, sizeof(int32_t), &sizeWritten);
    aclblasLtMatmulDescGetAttribute(computeDesc, ACLBLASLT_MATMUL_DESC_TRANSB, &transBVal, sizeof(int32_t), &sizeWritten);
    aclblasOperation_t transA = static_cast<aclblasOperation_t>(transAVal);
    aclblasOperation_t transB = static_cast<aclblasOperation_t>(transBVal);

    uint64_t rowsA = 0, colsA = 0; int64_t ldA = 0; uint32_t dtypeAVal = 0;
    aclblasLtMatrixLayoutGetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_ROWS, &rowsA, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_COLS, &colsA, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_LD, &ldA, sizeof(int64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Adesc, ACLBLASLT_MATRIX_LAYOUT_TYPE, &dtypeAVal, sizeof(uint32_t), &sizeWritten);
    aclDataType dtypeA = static_cast<aclDataType>(dtypeAVal);

    uint64_t rowsB = 0, colsB = 0; int64_t ldB = 0; uint32_t dtypeBVal = 0;
    aclblasLtMatrixLayoutGetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_ROWS, &rowsB, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_COLS, &colsB, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_LD, &ldB, sizeof(int64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Bdesc, ACLBLASLT_MATRIX_LAYOUT_TYPE, &dtypeBVal, sizeof(uint32_t), &sizeWritten);
    aclDataType dtypeB = static_cast<aclDataType>(dtypeBVal);

    uint64_t rowsD = 0, colsD = 0; int64_t ldD = 0; uint32_t dtypeDVal = 0;
    aclblasLtMatrixLayoutGetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_ROWS, &rowsD, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_COLS, &colsD, sizeof(uint64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_LD, &ldD, sizeof(int64_t), &sizeWritten);
    aclblasLtMatrixLayoutGetAttribute(Ddesc, ACLBLASLT_MATRIX_LAYOUT_TYPE, &dtypeDVal, sizeof(uint32_t), &sizeWritten);
    aclDataType dtypeD = static_cast<aclDataType>(dtypeDVal);

    int M = static_cast<int>(rowsD);
    int N = static_cast<int>(colsD);

    if (M < 0 || N < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (M == 0 || N == 0) return ACLBLAS_STATUS_SUCCESS;
    if (A_host == nullptr || B_host == nullptr || D_host == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    int physRowsA = static_cast<int>(rowsA);
    int physColsA = static_cast<int>(colsA);
    int physRowsB = static_cast<int>(rowsB);
    int physColsB = static_cast<int>(colsB);
    const int K = (transA == ACLBLAS_OP_N) ? physColsA : physRowsA;

    // Dequantize A
    std::vector<float> A_fp32;
    if (dtypeA == ACL_FLOAT) {
        A_fp32.assign(static_cast<const float*>(A_host),
                      static_cast<const float*>(A_host) + static_cast<size_t>(physRowsA) * ldA);
    } else if (dtypeA == ACL_FLOAT8_E4M3FN) {
        const uint8_t* scaleA_raw = static_cast<const uint8_t*>(scaleA_host);
        const size_t scaleABytes = mxScaleBufferBytesA(M, K, transA);
        std::vector<float> scaleA_fp32(scaleABytes);
        for (size_t i = 0; i < scaleABytes; i++) {
            scaleA_fp32[i] = dequantMxfp8E8m0(scaleA_raw[i]);
        }
        dequantMxfp8MatrixToFP32(static_cast<const uint8_t*>(A_host), scaleA_fp32.data(), physRowsA, physColsA,
                                  static_cast<int>(ldA), dtypeA, transA != ACLBLAS_OP_N, K, A_fp32);
    } else if (dtypeA == ACL_FLOAT4_E2M1) {
        const uint8_t* scaleA_raw = static_cast<const uint8_t*>(scaleA_host);
        const size_t scaleABytes = mxScaleBufferBytesA(M, K, transA);
        std::vector<float> scaleA_fp32(scaleABytes);
        for (size_t i = 0; i < scaleABytes; i++) {
            scaleA_fp32[i] = dequantMxfp8E8m0(scaleA_raw[i]);
        }
        dequantMxfp4MatrixToFP32(static_cast<const uint8_t*>(A_host), scaleA_fp32.data(), physRowsA, physColsA,
                                  static_cast<int>(ldA), dtypeA, transA != ACLBLAS_OP_N, K, A_fp32);
    }

    // Dequantize B
    std::vector<float> B_fp32;
    if (dtypeB == ACL_FLOAT) {
        B_fp32.assign(static_cast<const float*>(B_host),
                      static_cast<const float*>(B_host) + static_cast<size_t>(physRowsB) * ldB);
    } else if (dtypeB == ACL_FLOAT8_E4M3FN) {
        const uint8_t* scaleB_raw = static_cast<const uint8_t*>(scaleB_host);
        const size_t scaleBBytes = mxScaleBufferBytesB(N, K, transB);
        std::vector<float> scaleB_fp32(scaleBBytes);
        for (size_t i = 0; i < scaleBBytes; i++) {
            scaleB_fp32[i] = dequantMxfp8E8m0(scaleB_raw[i]);
        }
        dequantMxfp8MatrixToFP32(static_cast<const uint8_t*>(B_host), scaleB_fp32.data(), physRowsB, physColsB,
                                  static_cast<int>(ldB), dtypeB, transB == ACLBLAS_OP_N, K, B_fp32);
    } else if (dtypeB == ACL_FLOAT4_E2M1) {
        const uint8_t* scaleB_raw = static_cast<const uint8_t*>(scaleB_host);
        const size_t scaleBBytes = mxScaleBufferBytesB(N, K, transB);
        std::vector<float> scaleB_fp32(scaleBBytes);
        for (size_t i = 0; i < scaleBBytes; i++) {
            scaleB_fp32[i] = dequantMxfp8E8m0(scaleB_raw[i]);
        }
        dequantMxfp4MatrixToFP32(static_cast<const uint8_t*>(B_host), scaleB_fp32.data(), physRowsB, physColsB,
                                  static_cast<int>(ldB), dtypeB, transB == ACLBLAS_OP_N, K, B_fp32);
    }

    // Build op(A) and op(B)
    int kDim = K;
    std::vector<float> opA_fp32(static_cast<size_t>(M) * kDim);
    std::vector<float> opB_fp32(static_cast<size_t>(kDim) * N);

    if (transA == ACLBLAS_OP_N) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < kDim; j++)
                opA_fp32[static_cast<size_t>(i) * kDim + j] = A_fp32[static_cast<size_t>(i) * physColsA + j];
    } else {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < kDim; j++)
                opA_fp32[static_cast<size_t>(i) * kDim + j] = A_fp32[static_cast<size_t>(j) * M + i];
    }

    if (transB == ACLBLAS_OP_N) {
        for (int i = 0; i < kDim; i++)
            for (int j = 0; j < N; j++)
                opB_fp32[static_cast<size_t>(i) * N + j] = B_fp32[static_cast<size_t>(i) * physColsB + j];
    } else {
        for (int i = 0; i < kDim; i++)
            for (int j = 0; j < N; j++)
                opB_fp32[static_cast<size_t>(i) * N + j] = B_fp32[static_cast<size_t>(j) * kDim + i];
    }

    // Compute D = alpha * op(A) * op(B) + beta * C
    std::vector<float> D_fp32(static_cast<size_t>(M) * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int p = 0; p < kDim; p++) {
                sum += static_cast<double>(opA_fp32[static_cast<size_t>(i) * kDim + p]) *
                       static_cast<double>(opB_fp32[static_cast<size_t>(p) * N + j]);
            }
            double cVal = 0.0;
            if (C_host != nullptr && *beta != 0.0f) {
                cVal = static_cast<double>(C_host[static_cast<size_t>(i) * N + j]) * static_cast<double>(*beta);
            }
            D_fp32[static_cast<size_t>(i) * N + j] = static_cast<float>(static_cast<double>(*alpha) * sum + cVal);
        }
    }

    // Convert output to target dtype
    if (dtypeD == ACL_FLOAT) {
        memcpy(D_host, D_fp32.data(), D_fp32.size() * sizeof(float));
    } else if (dtypeD == ACL_BF16) {
        for (size_t i = 0; i < D_fp32.size(); i++) {
            uint16_t bf16 = fp32ToBF16(D_fp32[i]);
            D_host[i] = bf16ToFP32(bf16);  // round-trip for fair comparison
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

