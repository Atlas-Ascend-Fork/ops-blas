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
struct GemmExTiling {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t computeK;
    int32_t splitParts;
    int32_t productTileM;
    int32_t productTileN;
    uint64_t productPlane;
    int32_t pm;
    int32_t pn;
    int32_t pk;
    int32_t mBase;
    int32_t nBase;
    int32_t kBase;
    int32_t firstK;
    int32_t rawPack;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t transA;
    int32_t transB;
    float alpha;
    float beta;
    uint32_t cubeBlocks;
    uint64_t packAOffset;
    uint64_t packBOffset;
    uint64_t productOffset;
    uint64_t accumOffset;
};
namespace gemm_ex {
struct NativeArgs {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t transA;
    int32_t transB;
    uint32_t gridAndColumns;
    uint32_t reciprocalColumns;
};
static_assert(sizeof(NativeArgs) == 40);
struct ShortArgs {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t columns;
    uint32_t reciprocal;
};
template <uint32_t TM, uint32_t TN>
inline NativeArgs MakeNativeArgs(const GemmExTiling& t)
{
    const uint32_t columns = (static_cast<uint32_t>(t.m) + TN - 1) / TN;
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + TM - 1) / TM) * columns;
    const uint32_t reciprocal = columns == 1 ? 0 : static_cast<uint32_t>(((1ULL << 32) + columns - 1) / columns);
    return {t.m, t.n, t.k, t.lda, t.ldb, t.ldc, t.transA, t.transB, grid | (columns << 16), reciprocal};
}
inline ShortArgs MakeShortArgs(const GemmExTiling& t, uint32_t ldc, uint32_t columns)
{
    return {
        static_cast<uint32_t>(t.m),
        static_cast<uint32_t>(t.n),
        static_cast<uint32_t>(t.k),
        static_cast<uint32_t>(t.lda),
        static_cast<uint32_t>(t.ldb),
        ldc,
        columns,
        columns == 1 ? 0U : static_cast<uint32_t>(((1ULL << 32) + columns - 1) / columns)};
}
inline NativeArgs MakeSplitArgs(const GemmExTiling& t, uint32_t groups)
{
    NativeArgs args{t.m, t.n, t.k, t.lda, t.ldb, t.ldc, 0, 1, groups, 0};
    if (groups == 0 || groups == 1) {
        return args;
    }
    args.reciprocalColumns = static_cast<uint32_t>(((1ULL << 32) + groups - 1) / groups);
    return args;
}
template <uint32_t TM, uint32_t TN>
inline uint32_t NativeBlockCount(uint32_t blocks, const GemmExTiling& t)
{
    const uint32_t grid = ((static_cast<uint32_t>(t.n) + TM - 1) / TM) * ((static_cast<uint32_t>(t.m) + TN - 1) / TN);
    return blocks < grid ? blocks : grid;
}
template <typename Args>
inline Args MakePackArgs(uint32_t blocks, const GemmExTiling& t, const GemmExTiling& padded)
{
    if (blocks == 0 || padded.lda <= 0 || padded.ldb <= 0) {
        return {};
    }
    const uint32_t rowsA = t.transA ? t.m : t.k;
    const uint32_t widthA = t.transA ? t.k : t.m;
    const uint32_t rowsB = t.transB ? t.k : t.n;
    const uint32_t widthB = t.transB ? t.n : t.k;
    const uint32_t perA = (rowsA + blocks - 1) / blocks;
    const uint32_t perB = (rowsB + blocks - 1) / blocks;
    const uint32_t batchA = 32768 / padded.lda;
    const uint32_t batchB = 32768 / padded.ldb;
    return {
        {rowsA, widthA, static_cast<uint32_t>(padded.lda), perA | (batchA << 16)},
        {rowsB, widthB, static_cast<uint32_t>(padded.ldb), perB | (batchB << 16)}};
}
} // namespace gemm_ex

void gemmex_fallback_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const GemmExTiling& tiling);
void gemmex_fallback_matmul_rows_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling);
void gemmex_fallback_matmul_rows_right_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling);
void gemmex_fallback_matmul_rows_left_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling);
void gemmex_fallback_matmul_rows_both_trans_do(
    void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling);
void gemmex_fallback_accumulate_do(uint32_t blocks, void* stream, GM_ADDR workspace, const GemmExTiling& tiling);
void gemmex_fallback_epilogue_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmExTiling& tiling);
void gemmex_fallback_zero_do(uint32_t blocks, void* stream, GM_ADDR c, const GemmExTiling& tiling);
void gemmex_fallback_matmul_tiles_do(void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling);
void gemmex_fallback_scatter_tiles_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmExTiling& tiling);
void gemmex_fallback_wholegrid_nosplit_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmExTiling& tiling);
void gemmex_fallback_wholegrid_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR workspace, const GemmExTiling& tiling);
void gemmex_fallback_wholegrid_epilogue_do(
    uint32_t blocks, void* stream, GM_ADDR workspace, GM_ADDR c, const GemmExTiling& tiling);

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_short_native_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t, uint32_t ldc);
void gemmex_fallback_catlass_do(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmExTiling t);
void gemmex_fallback_catlass_copy_do(
    uint32_t blocks, void* stream, GM_ADDR input, GM_ADDR output, GemmExTiling t, uint32_t sourceLd);
void gemmex_fallback_catlass_pack_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GemmExTiling t, GemmExTiling padded);
#endif

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_short_nz_native_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
void gemmex_fallback_moderate_nz_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void gemmex_fallback_nz_fourrow_copy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
void gemmex_fallback_nz_bitcopy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
#endif

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_dma_pack_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, GemmExTiling);
#endif

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_tt_short_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
#endif

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_column_owner_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void gemmex_fallback_nz_loop_copy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
#endif

#ifdef GEMMEX_CATLASS_ENABLED
void gemmex_fallback_split_nt_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void gemmex_fallback_split_sum_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t, uint32_t);
#endif

void gemmex_fallback_tt_stagger159_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);

void gemmex_fallback_split232_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);

void gemmex_fallback_singleload174_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);

// Additional task-private launchers; no public BLAS ABI is introduced here.
void sgemm_ex_arch22_catlass_copy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
void sgemm_ex_arch22_catlass_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_catlass_pack_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, GemmExTiling);
void sgemm_ex_arch22_column_owner_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_dma_pack_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, GemmExTiling);
void sgemm_ex_arch22_moderate_nz_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_nz_fourrow_copy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_nz_loop_copy_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_short_native_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
void sgemm_ex_arch22_split_nt_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemm_ex_arch22_split_sum_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t, uint32_t);
void cgemm_pad305_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t);
void cgemm_long301_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, const GemmExTiling&);
void cgemm_short347_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void sgemmex_nn333_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
void sgemmex_nn333_output_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
void SgemmExSmallTt259Do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling);
void gemmex_nn263_do(uint32_t, void*, GM_ADDR, GM_ADDR, GM_ADDR, GemmExTiling, uint32_t);
void gemmex_nn263_output_do(uint32_t, void*, GM_ADDR, GM_ADDR, GemmExTiling);
