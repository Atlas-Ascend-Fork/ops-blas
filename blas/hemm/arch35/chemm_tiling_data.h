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
struct ChemmTiling {
    int64_t m;
    int64_t n;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    float ar;
    float ai;
    float br;
    float bi;
    uint32_t left;
    uint32_t upper;
};

struct ChemmCubeGemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t tempRowStride;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t splitK;
    uint32_t products;
};

// The combine kernel uses a small, named set of numerical-risk policies.  The
// value is carried through the host launch plan so shape handling does not
// depend on output-coordinate tables.
enum class ChemmRiskProfile : uint32_t {
    Default = 0,
    DeepKConservative = 1,
};
