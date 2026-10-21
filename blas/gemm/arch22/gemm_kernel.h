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
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
struct GemmArch22Tiling {
    int32_t m, n, k;
    int32_t computeK;
    int32_t pm, pn, pk;
    int32_t mBase, nBase, kBase;
    int32_t firstK;
    int32_t rawPack;
    int32_t lda, ldb, ldc;
    int32_t transA, transB;
    float alpha, beta;
    uint32_t cubeBlocks;
    uint64_t packAOffset, packBOffset, productOffset, accumOffset;
};
void sgemm_arch22_pack_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const GemmArch22Tiling& tiling);
void sgemm_arch22_matmul_rows_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling);
void sgemm_arch22_matmul_rows_right_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling);
void sgemm_arch22_matmul_rows_left_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling);
void sgemm_arch22_matmul_rows_both_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling);
void sgemm_arch22_accumulate_do(uint32_t blocks, void* stream, GM_ADDR workspace, const GemmArch22Tiling& tiling);
void sgemm_arch22_epilogue_do(uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmArch22Tiling& tiling);
void sgemm_arch22_zero_do(uint32_t blocks, void* stream, GM_ADDR c, const GemmArch22Tiling& tiling);
void sgemm_arch22_matmul_tiles_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmArch22Tiling& tiling);
void sgemm_arch22_scatter_tiles_do(uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmArch22Tiling& tiling);
void sgemm_arch22_wholegrid_nosplit_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b,
                                       GM_ADDR c, const GemmArch22Tiling& tiling);
void sgemm_arch22_wholegrid_pack_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b,
                                    GM_ADDR workspace, const GemmArch22Tiling& tiling);
void sgemm_arch22_wholegrid_epilogue_do(uint32_t blocks, void* stream, GM_ADDR workspace,
                                        GM_ADDR c, const GemmArch22Tiling& tiling);
