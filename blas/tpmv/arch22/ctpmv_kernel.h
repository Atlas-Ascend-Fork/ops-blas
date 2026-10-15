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
#include "ctpmv_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void ctpmv_kernel_do(
    GM_ADDR ap, GM_ADDR x, GM_ADDR workspace, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);

void ctpmv_batched_reference_kernel_do(
    GM_ADDR ap, GM_ADDR x, const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream);
