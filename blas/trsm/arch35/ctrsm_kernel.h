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

#include "ctrsm_tiling_data.h"

void ctrsm_kernel_do(uint8_t* a, uint8_t* b, const CtrsmTilingData& tiling, uint32_t numBlocks, void* stream);
void ctrsm_zero_kernel_do(uint8_t* b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream);
void ctrsm_mix_block_kernel_do(
    uint8_t* aPacked, uint8_t* bCombined, uint8_t* xCombined, uint8_t* systemWorkspace, const CtrsmTilingData& tiling,
    const void* cubeTilingData, uint32_t numBlocks, void* stream);
void ctrsm_block_prepare_dma_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* aPacked, uint8_t* bPacked, const CtrsmTilingData& tiling, uint32_t blocks,
    void* stream);
void ctrsm_block_finish_dma_kernel_do(
    uint8_t* a, uint8_t* aPacked, uint8_t* bPacked, uint8_t* b, const CtrsmTilingData& tiling, uint32_t blocks,
    void* stream);
