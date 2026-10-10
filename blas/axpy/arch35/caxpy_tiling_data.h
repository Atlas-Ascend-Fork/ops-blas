/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAXPY_TILING_DATA_H
#define CAXPY_TILING_DATA_H

#include <cstdint>

struct CaxpyTilingData {
    uint32_t totalN;
    int32_t incx;
    int32_t incy;
    uint32_t threadCount;
    uint32_t numBlocks;
    float alphaReal;
    float alphaImag;
};

#endif // CAXPY_TILING_DATA_H
