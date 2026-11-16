/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// chemm (arch22) left/one-panel/upper direct-mixed variant: one MIX kernel instantiation per translation
// unit keeps this object's .cb (L0B) static allocation within the 64 KB
// toolchain budget.

#include "chemm_kernel_common.h"
#include "chemm_kernel_aic.h"
#include "chemm_kernel_mix_aiv.h"
#include "chemm_kernel_aiv.h"
#include "chemm_kernel_ordered.h"
#include "chemm_kernel_entries.h"

void LaunchDirectMixedLeftPanelUpper(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream)
{
    chemm_direct_mixed_kernel<false, true, false><<<tiling.blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}
