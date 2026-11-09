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

#include "cgemm_tiling_data.h"
#include "gemm_kernel.h"

void gemm_cgemm_deinterleave_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi,
    GM_ADDR auxiliaryA, GM_ADDR auxiliaryB, const CgemmDeinterleaveTilingData& tilingData);

void gemm_cgemm_products_do(
    uint32_t numBlocks, void* stream, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR auxiliaryA,
    GM_ADDR auxiliaryB, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, const GemmTilingData& tilingData);

void gemm_cgemm_epilogue_do(
    uint32_t numBlocks, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR cInOut,
    const CgemmEpilogueTilingData& tilingData, GM_ADDR originalA = nullptr, GM_ADDR originalB = nullptr);

void gemm_cgemm_fused_do(
    uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, const GemmTilingData&);

void gemm_cgemm_onchip_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, const CgemmOnchipTilingData&);
