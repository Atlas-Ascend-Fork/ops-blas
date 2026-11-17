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
#include "csyr2_tiling_data.h"

void csyr2_unit_tiled_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_upper_atomic_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_small_upper_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_alpha_tiled_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);

void csyr2_kernel_do(uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&, uint32_t, void*);

void csyr2_direct_vector_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);

void csyr2_lower_vector_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_lower_columns_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_lower_general_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);

void csyr2_banded_vector_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_lower_contiguous_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
void csyr2_lower_halfwrite_do(
    uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&, uint32_t, uint32_t);
void csyr2_banded_vector_manual_do(uint32_t, void*, uint8_t*, uint8_t*, uint8_t*, const Csyr2TilingData&);
