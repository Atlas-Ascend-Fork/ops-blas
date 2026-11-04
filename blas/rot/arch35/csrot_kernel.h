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
#include "csrot_tiling_data.h"

// csrot_kernel_do: host-side kernel launcher.
// x/y point at the complex64 device buffers, viewed as 2*n interleaved float
// lanes (real at even lanes, imag at odd lanes) inside the kernels.
// cPtr/sPtr are non-null device pointers when the corresponding scalar lives on
// the device, and nullptr when it was resolved to a host scalar (the kernel then
// reads tiling.cosValue / tiling.sinValue instead).
void csrot_kernel_do(
    float* x, float* y, float* cPtr, float* sPtr, uint32_t numBlocks, const CsrotTilingData& tiling, void* stream);
