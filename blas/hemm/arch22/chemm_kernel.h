/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License in the root of the software repository for the full text of the License.
 */

// chemm (arch22) kernel launch declarations shared between host and device.

#pragma once

#include <cstdint>

#include "chemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void chemm_mixed_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const ChemmMixedTilingData& tiling, bool right, bool lower,
    void* stream);
void chemm_aiv_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const ChemmAivTilingData& tiling, bool right, bool lower, void* stream);
void chemm_ordered_aiv_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, const ChemmAivTilingData& tiling, bool right, bool lower, void* stream);
