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

// 单次 DataCopyPad（DataCopyParams 版）搬入的最大 float 数：blockLen(uint16 字节) 上限
// 65535B，16376 * 4B = 65504B 且为 8 的倍数。
constexpr uint32_t SCASUM_PAD_CHUNK_FLOATS = 16376;

struct ScasumTilingData {
    uint32_t n;
    int32_t incx;
    uint32_t useCoreNum;
    uint32_t perCoreN;
    uint32_t remainder;
};
