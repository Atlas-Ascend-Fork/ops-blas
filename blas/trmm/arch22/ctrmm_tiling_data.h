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

// All matrix strides below count complex elements at the public boundary and
// real float elements inside the padded, row-major workspace.
struct CtrmmTiling {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t q;
    uint32_t r;
    uint32_t qp;
    uint32_t rp;
    uint32_t right;
    uint32_t transpose;
    uint32_t conjugate;
    uint32_t upper;
    uint32_t unit;
    uint32_t originalTrans;
    uint32_t vectorCores;
    uint32_t cubeCores;
    uint32_t useCube;
    float alphaReal;
    float alphaImag;
    float fastInputLimit;
    uint64_t aPlane;
    uint64_t xPlane;
    uint64_t flagsOffset;
};

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif
void ctrmm_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const CtrmmTiling& tiling, void* stream);
