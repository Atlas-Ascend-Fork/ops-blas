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

#include "ctrttp_tiling_data.h"

// Launches the aclblasCtrttp AIV kernel: every block packs the columns j = blockIdx,
// blockIdx + useCoreNum, ... of the triangular matrix.
// The launch is asynchronous; the caller must synchronize the bound stream before
// reading AP back.
void ctrttp_kernel_do(uint8_t* A, uint8_t* AP, const CtrttpTilingData& tiling, uint32_t numBlocks, void* stream);

