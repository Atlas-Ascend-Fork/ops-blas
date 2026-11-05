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
 * \file ssbmv_tiling_data.h
 * \brief Tiling data shared by the Ssbmv host and kernel (arch22).
 *
 * Band storage convention (the one cblas_ssbmv uses, verified against the reference):
 *   UPPER: A(t, j) = M(j - k + t, j),   t = 0..k, diagonal of column j at t = k.
 *   LOWER: A(t, j) = M(j + t, j),       t = 0..k, diagonal of column j at t = 0.
 * Only one triangle is stored; the other half of the product follows from A = A^T.
 *
 * Decomposition: the output rows are split into contiguous row blocks, one per AIV core.
 * Core b owns rows [r0, r1) with r1 = min(n, r0 + blockRows). A stored entry A(t, j) is used
 * twice, and both uses are computed by the core that owns the output row they feed:
 *   - the column's own scatter, y[j - k + t] += A(t, j) * x[j] (UPPER) / y[j + t] (LOWER):
 *     for a fixed column the stored band is *contiguous* in GM, so a column contribution is
 *     one contiguous segment, shifted onto the aligned accumulator with DataCopyPad's
 *     leftPadding so every vector instruction keeps a 32B-aligned base;
 *   - the mirrored use, y[j] += sum_t A(t, j) * x[j - k + t] (UPPER) / x[j + t] (LOWER),
 *     which for a fixed t is a strided gather down the stored column (stride lda) paired
 *     with a contiguous slice of x: one vector FMA per (t, row block), no reduction.
 *
 * A row is written by exactly one core, so no atomics and no workspace are needed, and the
 * amount of A read is at most twice the amount of A stored.
 */

#pragma once

#include <cstdint>

/** \brief Upper bound of AIV cores used for grid sizing. */
constexpr uint32_t SSBMV_MAX_CORE_NUM = 64;

/*!< Rows one core is asked to own, as a floor on the grid size.
 *
 * Every block costs a fixed dispatch time. That dispatch is what the grid pays; what it buys is
 * fewer rows per core, and the per-core work only starts growing once a vector instruction's
 * 64 lanes are exceeded. Keeping a core's row block at or above one vector width therefore
 * removes the dispatch cost without adding work. */
constexpr uint32_t SSBMV_ROWS_PER_CORE = 64;

/** \brief Vector alignment granularity in float32 elements (32B / sizeof(float)). */
constexpr uint32_t SSBMV_ALIGN_ELEMS = 8;

/*!< Floats reserved for the staging run of a strided y. Only allocated when incy != 1, so it
 *   costs the vector cases nothing; it caps how many rows a strided store handles at once. */
constexpr uint32_t SSBMV_Y_STAGE_FLOATS = 8192;

/** \brief Row-block height above which a core splits its block into UB-sized chunks. */
constexpr uint32_t SSBMV_CHUNK_ROWS_CAP = 2048;

/*!< Upper bound, in float32 elements, of one band segment the wide-band column mirror stages at a
 *   time. The mirror splits a column of the band into pieces of at most this length, so a bandwidth
 *   taller than UB is reduced in several passes instead of being staged whole. Host and kernel
 *   share the value: the host charges three buffers of this size, the kernel allocates exactly
 *   those. 8192 leaves room for the row buffers and the scatter queue inside the 192KB UB. */
constexpr uint32_t SSBMV_MIRROR_TILE = 8192;

/** \brief Bytes of one float32 element. */
constexpr uint32_t SSBMV_BYTES_PER_FLOAT32 = 4;

/*!< Slack in floats appended to every UB buffer: the tail of a vector op or of a DataCopyPad may
 *   occupy a whole 32B block even when only part of it carries payload. */
constexpr uint32_t SSBMV_UB_SLACK = 16;

/*!< Row sized UB buffers the kernel allocates: the accumulator, the beta*y staging, the gathered
 *   band vector, the gathered x slice, the two gather offset tables, plus two for the staged
 *   column segment. The host has to budget for exactly this count, so it lives here instead of in
 *   either translation unit. The segment buffers scale with segBatch * (k + 8) rather than with a
 *    row and are charged separately, so they are not part of this count. */
constexpr uint32_t SSBMV_ROW_BUFFER_COUNT = 6;

/*!< Columns the scatter stages per queue round. A queue round costs four flag operations, and those
 *   flags - not the bytes moved - are what a column costs, so staging several columns per round
 *   divides that cost. The host lowers it when the slots do not fit in UB. */
constexpr uint32_t SSBMV_SEG_BATCH_MAX = 16;

/*!< Floats one staged column slot needs: the widest segment is k + 1 elements behind a head of up
 *   to SSBMV_ALIGN_ELEMS - 1, rounded up to the alignment. Host side only - the kernel reads the
 *   value from the tiling instead of calling this, because an aicore function cannot call it. */
inline uint32_t SsbmvSegSlotFloats(uint32_t k)
{
    return (k + 2U * SSBMV_ALIGN_ELEMS - 1U) / SSBMV_ALIGN_ELEMS * SSBMV_ALIGN_ELEMS;
}

/** \brief 相位副本个数 = 向量对齐宽度：把 x 窗口按 0..7 的错位各存一份，
 *   使任意错位的连续切片都能以 32B 对齐的下标被向量指令直接读取（免掉每 pass 一次 Gather）。 */
constexpr uint32_t SSBMV_PHASE_COUNT = SSBMV_ALIGN_ELEMS;

struct SsbmvTilingData {
    uint32_t n;          /**< order of A / logical length of x and y */
    uint32_t k;          /**< half bandwidth (number of super/sub-diagonals) */
    uint32_t lda;        /**< leading dimension of A, lda >= k + 1 */
    uint32_t uplo;       /**< ACLBLAS_UPPER (121) or ACLBLAS_LOWER (122) */
    uint32_t useCoreNum; /**< number of AIV cores the host decided to use, also the grid size */
    uint32_t blockRows;  /**< row-block height, always a multiple of SSBMV_ALIGN_ELEMS */
    uint32_t chunkRows;  /**< rows a core handles per UB-resident chunk, <= blockRows */
    uint32_t yStageRows; /**< rows per strided-y staging pass, >= 1 (only used when incy != 1) */
    uint32_t segBatch;   /**< columns staged per queue round in the scatter, >= 1 */
    uint32_t segSlot;    /**< floats reserved per staged column, >= k + 1 + alignment */
    float alpha;         /**< scalar multiplier of A * x */
    float beta;          /**< scalar multiplier of the incoming y */
    int64_t incx;        /**< stride of x, may be negative, never zero */
    int64_t incy;        /**< stride of y, may be negative, never zero */
    /*!< 1: the mirror walks the band by column and reduces each column's dot product, instead of
     *   one FMA per diagonal. Only worth it when the band is much taller than the row block (k >> rows),
     *   where the diagonal form needs ~3(k+1) calls per chunk to cover `rows` lanes each. */
    uint32_t bandForm;
    /*!< 1: the chunk's band span also covers the columns the scatter reads, so the scatter can
     *   gather its segments out of UB instead of issuing one GM copy per column. 0 when that span
     *   would not fit in UB (wide bands), which keeps the per-column GM path. */
    uint32_t spanExt;
    /*!< 1: 行-对角形式。整条带按 2k+1 条错位对角线累加：对每个偏移 e ∈ [-k, k]，
     *   y[r] += A[r * lda + C_e] * x[r + e]，其中 C_e 只依赖 e、lda、k（见 kernel）。
     *   每条对角线的 A 元素在带存储里等步长 lda，一次 Gather 即可取出，x 侧是连续切片，
     *   因此整个算子不再需要 scatter（逐列暂存/逐列搬运/偏移表标量循环），
     *   每 chunk 只有 3 条向量指令 × (2k+1)，且 pass 循环内没有任何跨流水同步。
     *   只有当整段带（lda * (chunkRows + k) 个元素）能放进 UB 时可用。 */
    uint32_t diagForm;
    /*!< 1: x 侧走「相位副本」。每 chunk 把 x 窗口按 0..7 错位各存一份（8 份），
     *   于是每条对角线的 x 操作数都是对齐的连续切片，直接用向量指令读，省掉每 pass 一次
     *   Gather（Gather 的开销显著高于普通向量指令）。副本的头部（x 下标为负的部分）与尾部
     *   预清零，无效 lane 因此读到 0，与 A 侧清零相乘仍是 0，不会把 NaN 带进结果。 */
    uint32_t phaseForm;
    /*!< 1: 本 tiling 的每个 UB 项都按 host 的精确公式验证过，能装进设备 UB；0: 装不下。
     *   kernel 不使用该字段；host 在它之上直接拒绝下发，避免把越界的 UB 分配带进核里。 */
    uint32_t ubFit;
};
