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

constexpr uint32_t SGEMV_MODE_NO_TRANS = 0;
constexpr uint32_t SGEMV_MODE_NO_TRANS_SPLIT_N = 1;
constexpr uint32_t SGEMV_MODE_TRANS = 2;

struct SgemvTilingData {
    uint32_t kernelMode;
    uint32_t usedCoreNum;
    uint32_t workPerCore;
    int32_t m;
    int32_t n;
    int32_t lda;
    int32_t incx;
    int32_t incy;
    float alpha;
    float beta;
};

void aclblasSgemv_do(uint32_t blockDim, void *stream, uint8_t *a, uint8_t *x, uint8_t *y, SgemvTilingData *tiling);
