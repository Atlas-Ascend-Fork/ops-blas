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

#include <cstdint>

struct CgemmSbTiling {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t paddedM;
    uint32_t paddedN;
    uint32_t paddedK;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t transA;
    uint32_t transB;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    uint32_t batchCount;
    int64_t strideA;
    int64_t strideB;
    int64_t strideC;
    uint64_t workspaceStride;
};

// Compact unit-coefficient square kernels only need these shape parameters.
struct CgemmSbCompactTiling {
    uint32_t batchCount;
    uint16_t m;
    uint8_t transA;
    uint8_t transB;
};
static_assert(sizeof(CgemmSbCompactTiling) == 8U, "Compact kernel parameters must fit in 8 bytes");

inline bool CgemmSbFastShape(const CgemmSbTiling& t)
{
    return t.alphaReal == 1.0f && t.alphaImag == 0.0f && t.betaReal == 0.0f && t.betaImag == 0.0f && t.m % 256U == 0U &&
           t.k % 256U == 0U && t.n % 64U == 0U;
}

inline bool CgemmSbAugmentedShape(const CgemmSbTiling& t)
{
    return t.transA == ACLBLAS_OP_N && t.transB == ACLBLAS_OP_N && t.alphaReal == 1.0f && t.alphaImag == 0.0f &&
           t.betaReal == 0.0f && t.betaImag == 0.0f && t.m <= 128U && t.n <= 128U && t.k <= 128U && t.m % 8U == 0U &&
           t.n % 8U == 0U && t.k % 8U == 0U && t.ldc % 8U == 0U && t.strideC >= static_cast<int64_t>(t.ldc) * t.n &&
           t.strideC % 8 == 0;
}

inline bool CgemmSbSmallShape(const CgemmSbTiling& t)
{
    return t.alphaReal == 1.0f && t.alphaImag == 0.0f && t.betaReal == 0.0f && t.betaImag == 0.0f && t.m <= 16U &&
           t.m == t.n && t.m == t.k && (t.m * t.n) % 8U == 0U && t.lda == t.m && t.ldb == t.k && t.ldc == t.m &&
           t.strideA == static_cast<int64_t>(t.m) * t.k && t.strideB == static_cast<int64_t>(t.k) * t.n &&
           t.strideC == static_cast<int64_t>(t.m) * t.n;
}

// One batch contains four input planes and four real GEMM products.
inline uint64_t CgemmSbWorkspaceFloats(uint32_t m, uint32_t n, uint32_t k)
{
    return 2ULL * m * k + 2ULL * k * n + 4ULL * m * n;
}
