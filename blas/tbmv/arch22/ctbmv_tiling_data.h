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

enum class CtbmvPath : uint8_t {
    Generic = 0U,
    CompactPanel = 3U,
    Row = 4U,
    SmallScalar = 7U,
    WideColumn = 13U,
    Transpose = 14U,
    TinyVector = 15U,
    WideGroup = 16U,
    SmallPanel = 17U,
};

struct CtbmvTilingData {
    int64_t incx;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t rowsPerCore;
    uint16_t useCoreNum;
    uint8_t uplo;
    uint8_t trans;
    uint8_t diag;
    CtbmvPath path;
};

static_assert(sizeof(CtbmvTilingData) == 32U, "CtbmvTilingData must stay compact");
