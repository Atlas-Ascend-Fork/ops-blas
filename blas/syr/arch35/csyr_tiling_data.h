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

// SIMT launch policy: small matrices use fewer threads; larger matrices use
// the full thread block. These are tuning thresholds, not API size limits.
constexpr int CSYR_SMALL_MATRIX_MAX_N = 512;
constexpr uint32_t CSYR_SMALL_THREADS = 512;
constexpr uint32_t CSYR_LARGE_THREADS = 1024;

struct CsyrTilingData {
    uint32_t numThreads;
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    float alphaReal;
    float alphaImag;
    uint32_t alphaIsDevice;
    int64_t incx;
};
