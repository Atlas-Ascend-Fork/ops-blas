/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chbmv_tiling_data.h
 * \brief Tiling struct and kernel launch declaration for aclblasChbmv (arch35).
 */

#pragma once

#include <cstdint>

struct ChbmvTilingData {
    uint32_t numThreads = 0;
    uint32_t numBlocks = 0;
    uint32_t rowsPerBlock = 0;
    uint32_t useUb = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t uplo = 0;
    float alphaReal = 0.0f;
    float alphaImag = 0.0f;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    int64_t incx = 0;
    int64_t incy = 0;
};

void chbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const ChbmvTilingData& tiling,
    void* stream);
