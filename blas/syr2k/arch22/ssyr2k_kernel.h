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

#include "ssyr2k_tiling_data.h"

void ssyr2k_gemm_launch(
    uint32_t blockDim, void* stream, uint8_t* a, uint8_t* b, uint8_t* temp, const Ssyr2kGemmTilingData& tiling,
    bool isTrans);

void ssyr2k_combine_launch(
    uint32_t blockDim, void* stream, uint8_t* temp, uint8_t* c, const Ssyr2kCombineTilingData& tiling);
