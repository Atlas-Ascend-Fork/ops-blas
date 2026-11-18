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

#include "sgemv_tiling_data.h"

void sgemv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream);
void sgemv_init_kernel_do(uint8_t* y, const SgemvTilingData& tiling, void* stream);
void sgemv_small_m_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream);
void sgemv_cube_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, const SgemvTilingData& tiling, void* stream);
void sgemv_split_reduce_kernel_do(
    uint8_t* partial, uint8_t* y, const SgemvTilingData& tiling, uint32_t numBlocks, void* stream);
