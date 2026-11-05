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

#include <string>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "csv_loader.h"

// ── Dtype string → aclDataType ──
inline aclDataType parseAclDataType(const std::string& s)
{
    if (s == "FP32" || s == "ACL_FLOAT" || s == "0")
        return ACL_FLOAT;
    if (s == "BF16" || s == "ACL_BF16" || s == "27")
        return ACL_BF16;
    if (s == "MXFP8_E4M3FN" || s == "ACL_FLOAT8_E4M3FN" || s == "36")
        return ACL_FLOAT8_E4M3FN;
    // issue #802: E5M2 negative-case support (rejected by the validation chain with
    // NOT_SUPPORTED; the parse must land on the real enum, not the ACL_FLOAT fallback).
    if (s == "MXFP8_E5M2" || s == "ACL_FLOAT8_E5M2" || s == "35")
        return ACL_FLOAT8_E5M2;
    if (s == "MXFP4_E2M1" || s == "ACL_FLOAT4_E2M1" || s == "40")
        return ACL_FLOAT4_E2M1;
    return ACL_FLOAT; // fallback
}

// ── Helper: is MXFP type ──
inline bool isMxfpType(aclDataType dt)
{
    return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2 || dt == ACL_FLOAT4_E2M1;
}

// ── Helper: dtype element size in bytes ──
inline size_t dtypeElementSize(aclDataType dt)
{
    if (dt == ACL_FLOAT)
        return sizeof(float); // 4
    if (dt == ACL_BF16)
        return 2;
    if (dt == ACL_FLOAT8_E4M3FN)
        return 1;
    if (dt == ACL_FLOAT8_E5M2)
        return 1;
    if (dt == ACL_FLOAT4_E2M1)
        return 1; // packed: 2 elements per byte
    return 4;     // fallback
}

// ── Helper: compute type for matmulDesc ──
inline aclblasComputeType_t getComputeType(aclDataType dtypeA, aclDataType dtypeD)
{
    return ACLBLAS_COMPUTE_32F; // all paths use 32F compute
}

// ── Compute matrix dimensions based on transposition ──
inline int getPhysicalRowsA(int M, int K, aclblasOperation_t transA) { return (transA == ACLBLAS_OP_N) ? M : K; }
inline int getPhysicalColsA(int M, int K, aclblasOperation_t transA) { return (transA == ACLBLAS_OP_N) ? K : M; }
inline int getPhysicalRowsB(int K, int N, aclblasOperation_t transB) { return (transB == ACLBLAS_OP_N) ? K : N; }
inline int getPhysicalColsB(int K, int N, aclblasOperation_t transB) { return (transB == ACLBLAS_OP_N) ? N : K; }

// MXFP4 CSV lda/ldb: logical element leading dim (same convention as MXFP8).
inline int64_t mxfp4PackedLd(int64_t logicalLd) { return (logicalLd + 1) / 2; }
// MX scale layout aligned with quant_matmul_mx_swat (CeilDiv(k,64)*2 per K-tile)
inline int mxScaleStrideAlongK(int kDim) { return ((kDim + 63) / 64) * 2; }

inline size_t mxScaleBufferBytesA(int M, int K, aclblasOperation_t transA)
{
    if (transA == ACLBLAS_OP_N) {
        return static_cast<size_t>(M) * static_cast<size_t>(mxScaleStrideAlongK(K));
    }
    return static_cast<size_t>(K) * static_cast<size_t>(mxScaleStrideAlongK(M));
}

inline size_t mxScaleBufferBytesB(int N, int K, aclblasOperation_t transB)
{
    if (transB == ACLBLAS_OP_N) {
        return static_cast<size_t>(mxScaleStrideAlongK(K)) * static_cast<size_t>(N);
    }
    return static_cast<size_t>(N) * static_cast<size_t>(mxScaleStrideAlongK(K));
}

struct LtMatmulParam : public BlasTestParamBase {
    aclDataType dtypeA = ACL_FLOAT;
    aclDataType dtypeB = ACL_FLOAT;
    aclDataType dtypeC = ACL_FLOAT; // accumulate dtype (C matrix dtype)
    aclDataType dtypeD = ACL_FLOAT; // output dtype (D matrix dtype)
    int M = 0;
    int N = 0;
    int K = 0;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int ldd = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    std::string algoMode = "default"; // "default" or "nullptr"
    bool CIsNull = false;
    bool handleNull = false;      // pass nullptr ltHandle (TC_L0_25)
    bool computeDescNull = false; // pass nullptr computeDesc (TC_L0_26)
    bool alphaNull = false;       // pass nullptr alpha (TC_L0_27)
    bool Anull = false;           // pass nullptr A device ptr (TC_L0_28)
    BlasFillMode scaleAFill = parseFill("RANDOM");
    BlasFillMode scaleBFill = parseFill("RANDOM");

    LtMatmulParam(const csv_map& m) : BlasTestParamBase(m)
    {
        dtypeA = parseAclDataType(ReadMap(m, "dtypeA", "FP32"));
        dtypeB = parseAclDataType(ReadMap(m, "dtypeB", "FP32"));
        dtypeC = parseAclDataType(ReadMap(m, "dtypeC", "FP32"));
        dtypeD = parseAclDataType(ReadMap(m, "dtypeD", "FP32"));
        M = parseInt(ReadMap(m, "M", "0"));
        N = parseInt(ReadMap(m, "N", "0"));
        K = parseInt(ReadMap(m, "K", "0"));
        transA = parseOpTrans(ReadMap(m, "transA", "N"));
        transB = parseOpTrans(ReadMap(m, "transB", "N"));
        lda = parseInt(ReadMap(m, "lda", "0"));
        ldb = parseInt(ReadMap(m, "ldb", "0"));
        ldc = parseInt(ReadMap(m, "ldc", "0"));
        ldd = parseInt(ReadMap(m, "ldd", "0"));
        alpha = parseFloat(ReadMap(m, "alpha", "1.0"));
        beta = parseFloat(ReadMap(m, "beta", "0.0"));
        algoMode = ReadMap(m, "algo_mode", "default");
        CIsNull = (ReadMap(m, "C_is_null", "false") == "true");
        handleNull = (ReadMap(m, "handle_null", "false") == "true");
        computeDescNull = (ReadMap(m, "computeDesc_null", "false") == "true");
        alphaNull = (ReadMap(m, "alpha_null", "false") == "true");
        Anull = (ReadMap(m, "A_null", "false") == "true");

        // Scale factor fill strategy (only meaningful for MXFP types)
        std::string sfA = ReadMap(m, "scaleA_fill", "RANDOM");
        std::string sfB = ReadMap(m, "scaleB_fill", "RANDOM");
        if (sfA == "none")
            scaleAFill = parseFill("VALUE_NORM_1"); // not applicable for FP32
        else
            scaleAFill = parseFill(sfA);
        if (sfB == "none")
            scaleBFill = parseFill("VALUE_NORM_1");
        else
            scaleBFill = parseFill(sfB);

        // Default ld values (row-major: ld >= physical cols)
        int physColsA = getPhysicalColsA(M, K, transA);
        int physColsB = getPhysicalColsB(K, N, transB);
        if (lda == 0)
            lda = std::max(1, physColsA);
        if (ldb == 0)
            ldb = std::max(1, physColsB);
        if (ldc == 0)
            ldc = std::max(1, M);
        if (ldd == 0)
            ldd = std::max(1, M);
    }
};
