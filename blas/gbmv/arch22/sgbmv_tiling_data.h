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

struct SgbmvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t kl;
    uint32_t ku;
    uint32_t lda;
    uint32_t trans; // 0 = ACLBLAS_OP_N, 1 = ACLBLAS_OP_T / C
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
    uint32_t useCoreNum;
    uint32_t colSplit; // 1 = N-domain column-split path (compact band storage)
};
