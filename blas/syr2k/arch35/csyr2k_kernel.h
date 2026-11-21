/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyr2k_kernel.h
 * \brief Kernel launcher declarations for aclblasCsyr2k (arch35).
 */

#pragma once

#include <cstdint>
#include "csyr2k_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void csyr2k_strict_narrow_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c, const Csyr2kStrictNarrowTilingData& tiling,
    uint32_t numBlocks, void* stream);

void csyr2k_half_prepare_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR alpha, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow, GM_ADDR asHalf,
    GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf, GM_ADDR bsLow,
    GM_ADDR fastFlags, const Csyr2kFastPrepareTilingData& tiling, uint32_t numBlocks, void* stream);

void csyr2k_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR fastFlags,
    const Csyr2kDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream);

void csyr2k_gemm_dispatch_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf, GM_ADDR adLow,
    GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow, GM_ADDR bsHalf,
    GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags,
    const Csyr2kGemmTilingData& tiling, uint32_t numBlocks, void* stream);

void csyr2k_partial_accumulate_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR t5, GM_ADDR t6, GM_ADDR t7, GM_ADDR t8,
    const Csyr2kPartialAccumulateTilingData& tiling, uint32_t numBlocks, void* stream);

void csyr2k_gemm_mix_epilogue_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR ar, GM_ADDR ai, GM_ADDR br, GM_ADDR bi, GM_ADDR arHalf, GM_ADDR arLow, GM_ADDR adHalf,
    GM_ADDR adLow, GM_ADDR asHalf, GM_ADDR asLow, GM_ADDR brHalf, GM_ADDR brLow, GM_ADDR biHalf, GM_ADDR biLow,
    GM_ADDR bsHalf, GM_ADDR bsLow, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha,
    GM_ADDR beta, GM_ADDR c, const Csyr2kGemmTilingData& gemmTiling,
    const Csyr2kDeinterleaveTilingData& deinterleaveTiling, const Csyr2kCombineTilingData& combineTiling,
    uint32_t numBlocks, void* stream);

void csyr2k_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR fastFlags, GM_ADDR alpha, GM_ADDR beta, GM_ADDR c,
    const Csyr2kCombineTilingData& tiling, uint32_t numBlocks, void* stream);
