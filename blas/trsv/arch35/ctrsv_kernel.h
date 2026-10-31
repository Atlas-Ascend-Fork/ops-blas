/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "ctrsv_tiling_data.h"

// Cached x/scales plus 64 packed 64x64 complex panels and classification flags
// need at most 2,162,944 bytes. Reserve 3 MiB including alignment headroom.
constexpr uint64_t CTRSV_WORKSPACE_BYTES = 3ULL * 1024 * 1024;

void ctrsv_kernel_do(uint8_t* a, uint8_t* x, CtrsvTilingData tiling, void* stream, void* workspace);
