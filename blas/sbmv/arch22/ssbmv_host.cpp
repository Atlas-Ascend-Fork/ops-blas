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
 * \file ssbmv_host.cpp
 * \brief Host side of aclblasSsbmv for arch22: validation, tiling and a single kernel launch.
 *
 * On the common path the tiling is passed to the kernel by value and the workspace (only ever needed
 * when x and y overlap) is the one the handle already owns, so nothing is allocated and the stream is
 * not synchronized. The only exception is an in-place call whose x does not fit in that workspace:
 * it is grown in place (which synchronizes the stream once), or the call fails.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "ssbmv_tiling_data.h"
#include "ssbmv_kernel.h"

namespace {
/*!
 * \brief 行-对角形式的核数：按「每 block 固定开销 + 每核迭代成本 + 带宽」的成本模型取最优。
 *
 * 核数 c 的估计成本为
 *   c × 每 block 固定开销 + (2k+1) × ceil(ceil(n/c)/64) × 每迭代成本 + 总字节 / 带宽(c)。
 * 工作量按 (2k+1) × ceil(行/64) 个迭代计：c 太小时每核迭代数线性增长，c 太大时每 block 的
 * 固定开销线性堆叠，中间存在最优；窄带大 n 时最优远小于「每核 64 行」的经验规则。
 */
uint32_t CalSsbmvDiagCoreNum(uint32_t n, uint32_t k, uint32_t lda)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        return 0;
    }
    if (aivCoreNum > SSBMV_MAX_CORE_NUM) {
        aivCoreNum = SSBMV_MAX_CORE_NUM;
    }
    constexpr double CORE_FIXED_US = 0.40;
    constexpr double PASS_ITER_US = 0.067;
    constexpr double BW_PER_CORE = 74.0e9; /* bytes/s */
    constexpr double BW_CAP = 2367.0e9;
    const uint32_t cap = std::min(aivCoreNum, CeilDiv<uint32_t>(n, SSBMV_ALIGN_ELEMS));
    const double bytes = static_cast<double>(n) * lda * 4.0 + static_cast<double>(n) * 8.0;
    double best = -1.0;
    uint32_t bestCore = 1;
    for (uint32_t cores = 1; cores <= cap; cores++) {
        const uint32_t rowsPerCore = CeilDiv<uint32_t>(n, cores);
        const uint32_t iters = CeilDiv<uint32_t>(rowsPerCore, 64U);
        /* 带宽随核数线性增长、超过 BW_CAP 后封顶；cores >= 1 ⇒ bw >= 74e9 恒非零。静态检查
         * 不识别 std::min/max 的值域（用 max 取「下界」表达非零对它是不可见的），按 G.EXP.22-CPP
         * 在除数上显式判零，写法同 CalSsbmvYStageRows。 */
        double bw = std::min(BW_PER_CORE * cores, BW_CAP);
        if (bw == 0.0) {
            bw = BW_PER_CORE; /* 不可达；兜底为单核带宽 */
        }
        const double cost = CORE_FIXED_US * cores + PASS_ITER_US * (2.0 * k + 1.0) * iters + bytes / bw * 1.0e6;
        if (best < 0.0 || cost < best) {
            best = cost;
            bestCore = cores;
        }
    }
    return std::min(n, bestCore);
}

uint32_t CalSsbmvCoreNum(uint32_t n)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        return 0;
    }
    if (aivCoreNum > SSBMV_MAX_CORE_NUM) {
        aivCoreNum = SSBMV_MAX_CORE_NUM;
    }
    /* 每个 block 的派发成本是固定的，而每核的工作量在行数不超过一个向量宽度（64 lane）时
     * 并不随行数增长，所以核数不该按设备上限取。取「每核至少 SSBMV_ROWS_PER_CORE 行」与
     * 设备上限的较小者，n 很大时退化为原策略。 */
    uint32_t byRows = CeilDiv<uint32_t>(n, SSBMV_ROWS_PER_CORE);
    if (byRows < 1U) {
        byRows = 1U;
    }
    return std::min(n, std::min(aivCoreNum, byRows));
}

/*!
 * \brief Rows a core may keep in UB at once.
 *
 * Every buffer the kernel allocates has to be charged here, so the chunk is derived from the UB
 * size this device reports instead of a fixed constant:
 *   - SSBMV_ROW_BUFFER_COUNT buffers holding one row each. Each is rounded up to a multiple of the
 *     vector alignment and carries SSBMV_UB_SLACK floats of slack, which is charged as the
 *     constant below;
 *   - the A span the mirror gathers from, lda floats per row plus the k entries of the last
 *     column: by far the largest term once k grows, and the reason the chunk shrinks with k;
 *   - the x window both halves read through, chunkRows + 2k floats: the chunk's own columns plus
 *     the k the band reaches back over the scatter and forward over the mirror.
 *
 * The window term has two regimes and each bound below is solved for one of them; the tighter of
 * the two is safe in both. The result is at least one row, so the kernel stays correct (if slow)
 * for a band so wide that a single row fills UB.
 */
uint32_t CalSsbmvYStageRows(uint32_t absIncy)
{
    if (absIncy == 0U) {
        return 1U;
    }
    /* spanLen = (rows - 1) * absIncy + 1 has to stay inside the reserved budget. */
    const uint64_t rows = ((SSBMV_Y_STAGE_FLOATS - 1U) / absIncy) + 1U;
    return static_cast<uint32_t>(rows);
}

/*!
 * \brief 一次队列往返暂存的列数。
 *
 * 一次队列往返要付四次旗标操作，列的成本几乎全在这四次旗标上，搬运字节数的占比很小。
 * 因此让一轮服务多列可以直接摊薄。这里给它 UB 的 1/8 额度，放不下就退回 2 列、1 列；
 * 极端大 k 时退到 1，行为与优化前一致。
 */
uint32_t CalSsbmvSegBatch(uint32_t k, uint64_t budgetFloats)
{
    const uint32_t slot = SsbmvSegSlotFloats(k);
    for (uint32_t batch = SSBMV_SEG_BATCH_MAX; batch > 1U; batch >>= 1U) {
        if (2ULL * batch * slot <= budgetFloats / 8U) {
            return batch;
        }
    }
    return 1U;
}

/*!
 * \brief 通用/对角-带 span 形式的 UB 占用（float 数）。列形式有自己的公式，见 SsbmvBandUbFloats。
 */
uint64_t SsbmvUbFloats(
    uint32_t chunkRows, uint32_t k, uint32_t lda, uint32_t spanExtraCols, int64_t incy, uint64_t segFloats,
    uint32_t yStageRows)
{
    const uint64_t acc = CeilAlign<uint64_t>(chunkRows, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    const uint64_t xb = CeilAlign<uint64_t>(chunkRows + 2U * k, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    const uint64_t spanCols = static_cast<uint64_t>(chunkRows) + spanExtraCols;
    const uint64_t span =
        CeilAlign<uint64_t>(static_cast<uint64_t>(lda) * (spanCols - 1U) + k + 1U, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    /* segQueue 是双缓冲（kernel 按 SSBMV_BUFFER_NUM=2 开队列），这里把两槽都算上。 */
    uint64_t total = acc * SSBMV_ROW_BUFFER_COUNT + xb + span + 2U * segFloats;
    if (incy != 1) {
        const uint32_t absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);
        total += CeilAlign<uint64_t>((static_cast<uint64_t>(yStageRows) - 1U) * absIncy + 1U, SSBMV_ALIGN_ELEMS) +
                 SSBMV_UB_SLACK;
    }
    return total;
}

/*!
 * \brief 列形式（bandForm）的 UB 占用（float 数）。
 *
 * **必须与 kernel InitBuffers 的 bandForm 分支逐项一致**，否则 host 少算的 UB 会在 kernel 里溢出：
 *   6 个行大小缓冲 + x/span 两个最小占位 + seg 队列双缓冲（kernel 固定单列一轮）
 *   + A/x/归约临时三块 tile 缓冲 + red/off8/redAcc，incy != 1 时再加 strided-y 暂存。
 * 这里不含 2k 的 x 窗口、也不含 segBatch * (k+8) 的槽位 —— 列形式两半都从 GM 按段读，
 * 这两项原本是宽带撑爆 UB 的来源。
 */
uint64_t SsbmvBandUbFloats(uint32_t chunkRows, uint32_t lda, int64_t incy, uint32_t yStageRows)
{
    const uint64_t acc = CeilAlign<uint64_t>(chunkRows, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    const uint64_t tile = (lda < SSBMV_MIRROR_TILE) ? lda : SSBMV_MIRROR_TILE;
    const uint64_t slab = CeilAlign<uint64_t>(tile, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    const uint64_t segSlot = CeilAlign<uint64_t>(chunkRows + SSBMV_ALIGN_ELEMS, SSBMV_ALIGN_ELEMS);
    uint64_t total = acc * SSBMV_ROW_BUFFER_COUNT;
    total += 2U * (SSBMV_ALIGN_ELEMS + SSBMV_UB_SLACK); /* xBuf / spanBuf 占位 */
    total += 2U * segSlot;                              /* segQueue 双缓冲 */
    total += slab * 3U + 64U + 8U + SSBMV_ALIGN_ELEMS;  /* prod / xSl / tmp + red + off8 + redAcc */
    if (incy != 1) {
        const uint32_t absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);
        total += CeilAlign<uint64_t>((static_cast<uint64_t>(yStageRows) - 1U) * absIncy + 1U, SSBMV_ALIGN_ELEMS) +
                 SSBMV_UB_SLACK;
    }
    return total;
}

/*!
 * \brief 列形式的 chunk 行数：在 [1, min(blockRows, CAP)] 上二分找能装进 UB 的最大值，返回 0 表示
 *   连一行都装不下。谓词随行数单调（行缓冲、seg 槽、y 暂存都随行数增长，tile 缓冲是常数）。
 */
uint32_t CalSsbmvBandChunkRows(uint32_t blockRows, uint32_t lda, int64_t incy, uint32_t yStageRows, uint64_t ubFloats)
{
    auto fits = [&](uint32_t rows) {
        const uint32_t stage = (yStageRows < rows) ? yStageRows : rows;
        return SsbmvBandUbFloats(rows, lda, incy, stage) <= ubFloats;
    };
    uint64_t hi = (blockRows < SSBMV_CHUNK_ROWS_CAP) ? blockRows : SSBMV_CHUNK_ROWS_CAP;
    if (hi < 1U || !fits(1U)) {
        return 0U;
    }
    uint64_t lo = 1U;
    while (lo < hi) {
        const uint64_t mid = (lo + hi + 1U) / 2U;
        if (fits(static_cast<uint32_t>(mid))) {
            lo = mid;
        } else {
            hi = mid - 1U;
        }
    }
    return static_cast<uint32_t>(lo);
}

/*!
 * \brief 行-对角形式的 band span 元素个数。**必须与 kernel 的 DiagSpanFloats() 逐字一致**，
 * 否则 host 少算的 UB 会在 kernel 里溢出。
 */
uint64_t SsbmvDiagSpanFloats(uint32_t chunkRows, uint32_t k, uint32_t lda)
{
    const uint64_t spanMax = static_cast<uint64_t>(lda) * (chunkRows + k) + k + 1U;
    const uint64_t guard = static_cast<uint64_t>(SSBMV_ALIGN_ELEMS) * lda;
    return guard + CeilAlign<uint64_t>(spanMax, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
}

/*!
 * \brief 行-对角形式的 UB 占用（float 数）：4 个行大小缓冲（y/beta-y/aVec/xv）、两张
 *   uint32 偏移表、x 窗口和 band span。
 */
/*! \brief phaseForm 的相位副本占用的 float 数（8 份窗口副本）。 */
uint64_t SsbmvDiagPhaseFloats(uint32_t chunkRows, uint32_t k)
{
    const uint64_t xCnt = CeilAlign<uint64_t>(chunkRows + 2U * k, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    return static_cast<uint64_t>(SSBMV_PHASE_COUNT) * CeilAlign<uint64_t>(xCnt, SSBMV_ALIGN_ELEMS);
}

uint64_t SsbmvDiagUbFloats(uint32_t chunkRows, uint32_t k, uint32_t lda, int64_t incy, uint32_t yStageRows)
{
    const uint64_t acc = CeilAlign<uint64_t>(chunkRows, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    const uint64_t xb = CeilAlign<uint64_t>(chunkRows + 2U * k, SSBMV_ALIGN_ELEMS) + SSBMV_UB_SLACK;
    /* 两张偏移表是 uint32，各占 accRows 个元素（4 字节 = 1 个 float 的额度） */
    uint64_t total = acc * 6U + xb + SsbmvDiagSpanFloats(chunkRows, k, lda);
    if (incy != 1) {
        const uint32_t absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);
        total += CeilAlign<uint64_t>((static_cast<uint64_t>(yStageRows) - 1U) * absIncy + 1U, SSBMV_ALIGN_ELEMS) +
                 SSBMV_UB_SLACK;
    }
    return total;
}

/*!
 * \brief 行-对角形式下每核一次能处理的行数；放不下返回 0。
 *
 * 先按「每行 lda + 7 个 float、其余是常数项」求解，再用精确公式回验，避免估算偏差导致 UB 溢出。
 */
uint32_t CalSsbmvDiagChunkRows(
    uint32_t blockRows, uint32_t k, uint32_t lda, int64_t incy, uint32_t yStageRows, uint64_t ubFloats)
{
    const uint64_t perRow = 7U + lda;
    uint64_t constTerm = 4U * 32U + 2U * k + 32U + static_cast<uint64_t>(lda) * k + k + 1U + 32U;
    if (incy != 1) {
        const uint32_t absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);
        const uint64_t stageRows = static_cast<uint64_t>(yStageRows) * absIncy + 1U;
        const uint64_t staged = (stageRows < SSBMV_Y_STAGE_FLOATS) ? stageRows : SSBMV_Y_STAGE_FLOATS;
        constTerm += staged + 32U;
    }
    uint64_t rows = (ubFloats > constTerm) ? ((ubFloats - constTerm) / perRow) : 0U;
    if (rows > blockRows) {
        rows = blockRows;
    }
    if (rows > SSBMV_CHUNK_ROWS_CAP) {
        rows = SSBMV_CHUNK_ROWS_CAP;
    }
    while (rows >= 1U && SsbmvDiagUbFloats(static_cast<uint32_t>(rows), k, lda, incy, yStageRows) > ubFloats) {
        rows--;
    }
    return static_cast<uint32_t>(rows);
}

uint32_t CalSsbmvChunkRows(
    uint32_t n, uint32_t k, uint32_t lda, uint32_t blockRows, int64_t incy, uint64_t segFloats, uint32_t spanExtraCols)
{
    constexpr uint64_t SAFETY_MARGIN = 1024;
    constexpr uint64_t ROW_BUFFER_COUNT = SSBMV_ROW_BUFFER_COUNT;
    constexpr uint64_t OVERHEAD = SSBMV_ALIGN_ELEMS + SSBMV_UB_SLACK;
    uint64_t ubSize = 192U * 1024U;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    uint64_t budget = ((ubSize > SAFETY_MARGIN) ? (ubSize - SAFETY_MARGIN) : 0) / SSBMV_BYTES_PER_FLOAT32;
    /* 段缓冲按 segBatch * slot 计，替换掉原先「两行」的固定项 */
    budget = (budget > (2U * segFloats)) ? (budget - 2U * segFloats + 2U * OVERHEAD) : 0;
    if (incy != 1) {
        /* The strided y store stages a run of y through UB; the kernel sizes it from the same
         * constant (and allocates it under the same condition), so the chunk has to leave room
         * for it here. */
        const uint64_t staged = SSBMV_Y_STAGE_FLOATS + OVERHEAD;
        budget = (budget > staged) ? (budget - staged) : 0;
    }
    /* Floats left for the chunk-dependent terms once the fixed overhead of every buffer is paid. */
    const uint64_t base = (budget > (ROW_BUFFER_COUNT * OVERHEAD + 2 * OVERHEAD + 1)) ?
                              (budget - ROW_BUFFER_COUNT * OVERHEAD - 2 * OVERHEAD - 1) :
                              0;
    /* Per chunk the budget pays chunkRows * (row buffers + 1) for the buffers that scale with the
     * rows, lda * chunkRows for the span and 3k - lda for the two windows (a k that the span
     * reaches past the last column plus the 2k of the x window). */
    /* span 项：对角形式按 lda 每行 + 常量 (lda*(extra-1) + k + 1)。 */
    const uint64_t spanConst = static_cast<uint64_t>(lda) * spanExtraCols;
    const uint64_t perRow = lda + 1U;
    const uint64_t constTerm = 3U * k + spanConst;
    const uint64_t byRows =
        ((base + perRow) > constTerm) ? ((base + perRow - constTerm) / (ROW_BUFFER_COUNT + perRow)) : 0;
    uint64_t chunkRows = byRows;
    if (chunkRows > SSBMV_CHUNK_ROWS_CAP) {
        chunkRows = SSBMV_CHUNK_ROWS_CAP;
    }
    if (chunkRows > blockRows) {
        chunkRows = blockRows;
    }
    if (chunkRows < 1U) {
        chunkRows = 1U;
    }
    return static_cast<uint32_t>(chunkRows);
}

/*!
 * \brief 行-对角形式（diagForm）的选取：整段带（lda * (chunkRows + k) 个元素）能放进 UB 且
 * chunk 覆盖量达标时启用；它把 scatter 那一整块机制（逐列暂存、逐列搬运、偏移表标量循环）换成
 * 2k+1 条对角线的纯向量累加，pass 之间没有任何跨流水同步，窄带下每核调用数从 O(行数) 降到 O(k)。
 *
 * 准入条件：一次 chunk 至少要覆盖一个向量宽度**或整个行块**。行-对角形式的 span 是
 * lda*(chunkRows+k)，k 大时 chunkRows 会被压得很小，于是每个 chunk 都要重付一遍搬运与同步，
 * pass 总数按 chunk 数翻倍，性能反而明显劣于通用路径。行数不足一个向量宽度说明这条路径
 * 不划算，退回通用路径。
 * 「或整个行块」这一支是为小 n 开的：blockRows < 64 时（n <= 127）行块本身就不到一个向量
 * 宽度，此时 diagRows == blockRows 说明 chunk 没有被 UB 压小、每个核只做一个 chunk，
 * 正是行-对角形式最省的形态；旧条件会把这些形状全部踢回通用 kernel，小 n 下固定开销
 * 占比很高。 */
void SelectSsbmvDiagForm(
    SsbmvTilingData& tilingData, uint32_t k, uint32_t lda, int64_t incy, uint32_t stageRows, uint64_t ubBytes)
{
    const uint32_t diagRows =
        CalSsbmvDiagChunkRows(tilingData.blockRows, k, lda, incy, stageRows, ubBytes / SSBMV_BYTES_PER_FLOAT32);
    const uint32_t diagMinRows = std::min(SSBMV_ROWS_PER_CORE, tilingData.blockRows);
    if (diagRows >= diagMinRows) {
        tilingData.diagForm = 1U;
        tilingData.chunkRows = diagRows;
    }
}

/*!
 * \brief 相位副本（phaseForm）的选取：x 侧每条对角线一次 Gather 换成「8 份错位副本 + 对齐读」。
 * 只有当省下的 Gather 等效开销能被抵掉、且 UB 装得下时才开（否则退回逐 pass Gather）。
 *
 * 8 次相位 DMA 是**固定**成本，而每条对角线省下的 Gather 与对角线数成正比，故对角线越多
 * 越划算；对角线少时省下的部分抵不过副本成本。
 * 判据 = 「对角线数下限」+「省下的 Gather 等效 lane 数 > 副本 lane 数 + 固定开销」。 */
void SelectSsbmvPhaseForm(
    SsbmvTilingData& tilingData, uint32_t k, uint32_t lda, int64_t incy, uint32_t stageRows, uint64_t ubBytes)
{
    if (tilingData.diagForm == 0U || tilingData.incy != 1) {
        return;
    }
    constexpr uint32_t PHASE_MIN_DIAGS = 25U;      /* k >= 12 为盈亏分界 */
    constexpr uint64_t GATHER_COST_X = 5U;         /* Gather 相对普通向量指令的成本倍率 */
    constexpr uint64_t PHASE_FIXED_LANES = 12000U; /* 8 次相位 DMA 的固定开销，折算成 lane 当量 */
    const uint64_t savedEquiv = GATHER_COST_X * (2ULL * k + 1ULL) * tilingData.chunkRows;
    const uint64_t copyCost = SsbmvDiagPhaseFloats(tilingData.chunkRows, k);
    const uint64_t totalFloats = SsbmvDiagUbFloats(tilingData.chunkRows, k, lda, incy, stageRows) + copyCost;
    if ((2U * k + 1U) >= PHASE_MIN_DIAGS && savedEquiv >= (copyCost + PHASE_FIXED_LANES) &&
        totalFloats * SSBMV_BYTES_PER_FLOAT32 <= ubBytes) {
        tilingData.phaseForm = 1U;
    }
}

/*!
 * \brief 通用形式（非行-对角）的 chunk 选取：宽带优先「逐列点积」，否则尽量给 scatter 留 UB span。
 *
 * The scatter reads the k columns past a row block, so the chunk's span has to cover them for it
 * to be able to gather its segments out of UB instead of copying column by column from GM. That
 * span is affordable for narrow bands (the failing shapes) and hopeless for wide ones, so pick it
 * where it fits and keep the per-column GM path otherwise. */
void SelectSsbmvGeneralForm(
    SsbmvTilingData& tilingData, uint32_t n, uint32_t k, uint32_t lda, int64_t incy, uint64_t segFloats,
    uint32_t stageRows, uint64_t ubBytes)
{
    const uint32_t diagRows = CalSsbmvChunkRows(n, k, lda, tilingData.blockRows, incy, segFloats, 0U);
    /* 列形式的成本是每列 4 次调用；对角形式是每 chunk 3(k+1) 次。宽带（k 远大于每块行数）时
     * 后者会顶到 3(k+1)×chunks 而每次只覆盖 chunkRows 个 lane，改用列形式。
     * 列形式不建 x 窗口、seg 槽按行数开、段缓冲按 tile 分块，因此对 k 与 incx 都没有额外
     * 限制：宽带 + 任意步长也可用（x 段按物理下标逐 lane 填充）。 */
    const bool wideWins = (3U * (static_cast<uint64_t>(k) + 1U)) > (4U * static_cast<uint64_t>(diagRows));
    if (wideWins) {
        const uint32_t rows = CalSsbmvBandChunkRows(tilingData.blockRows, lda, incy, stageRows, ubBytes / 4U);
        if (rows > 0U) {
            tilingData.bandForm = 1U;
            tilingData.chunkRows = rows;
        }
    }
    if (tilingData.bandForm == 0U) {
        const uint32_t probe = CalSsbmvChunkRows(n, k, lda, tilingData.blockRows, incy, segFloats, k);
        const uint32_t stage = (stageRows > probe) ? probe : stageRows;
        if (SsbmvUbFloats(probe, k, lda, k, incy, segFloats, stage) * 4U <= ubBytes) {
            tilingData.spanExt = 1U;
            tilingData.chunkRows = probe;
        } else {
            /* 通用形式（逐列 span、无 spanExt）的估计可能偏乐观：从估计值往下验到能装进 UB 为止。
             * 一行都装不下时留 0，由 CalSsbmvTilingData 的 ubFit 统一拒绝。 */
            uint32_t rows = diagRows;
            while (rows >= 1U) {
                const uint32_t stageRowsUsed = (stageRows > rows) ? rows : stageRows;
                if (SsbmvUbFloats(rows, k, lda, 0U, incy, segFloats, stageRowsUsed) * 4U <= ubBytes) {
                    break;
                }
                rows--;
            }
            tilingData.chunkRows = rows;
        }
    }
}

/*!
 * \brief 按最终 tiling 的形态精确计算 UB 占用并与预算比较。
 *
 * diag/band 的选择本身用精确公式准入；通用回退（含 MaybeRetile 重算后的形态）由这里兜底。
 * 返回 false 时 aclblasSsbmv 直接返回 NOT_SUPPORTED，绝不把越界的 UB 分配下发到核里。
 */
bool SsbmvTilingFitsUb(
    const SsbmvTilingData& tilingData, uint32_t k, int64_t incy, uint64_t segFloats, uint64_t ubBytes)
{
    if (tilingData.chunkRows < 1U) {
        return false;
    }
    uint64_t needFloats = 0U;
    if (tilingData.diagForm != 0U) {
        needFloats = SsbmvDiagUbFloats(tilingData.chunkRows, k, tilingData.lda, incy, tilingData.yStageRows);
        if (tilingData.phaseForm != 0U) {
            needFloats += SsbmvDiagPhaseFloats(tilingData.chunkRows, k);
        }
    } else if (tilingData.bandForm != 0U) {
        needFloats = SsbmvBandUbFloats(tilingData.chunkRows, tilingData.lda, incy, tilingData.yStageRows);
    } else {
        const uint32_t extra = (tilingData.spanExt != 0U) ? k : 0U;
        needFloats =
            SsbmvUbFloats(tilingData.chunkRows, k, tilingData.lda, extra, incy, segFloats, tilingData.yStageRows);
    }
    return needFloats <= (ubBytes / SSBMV_BYTES_PER_FLOAT32);
}

SsbmvTilingData CalSsbmvTilingData(
    uint32_t useCoreNum, uint32_t n, uint32_t k, uint32_t lda, uint32_t uplo, float alpha, float beta, int64_t incx,
    int64_t incy)
{
    const uint32_t absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);
    SsbmvTilingData tilingData{};
    tilingData.n = n;
    tilingData.k = k;
    tilingData.lda = lda;
    tilingData.uplo = uplo;
    tilingData.alpha = alpha;
    tilingData.beta = beta;
    tilingData.incx = incx;
    tilingData.incy = incy;
    tilingData.useCoreNum = useCoreNum;
    tilingData.blockRows = CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, useCoreNum), SSBMV_ALIGN_ELEMS);
    uint64_t ubBudget = 192U * 1024U;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBudget);
    }
    const uint64_t budgetFloats = ubBudget / SSBMV_BYTES_PER_FLOAT32;
    tilingData.segSlot = SsbmvSegSlotFloats(k);
    tilingData.segBatch = CalSsbmvSegBatch(k, budgetFloats);
    const uint32_t slotFloats = tilingData.segSlot;
    const uint64_t segFloats = static_cast<uint64_t>(tilingData.segBatch) * slotFloats;
    uint32_t stageRows = CalSsbmvYStageRows(absIncy);
    tilingData.spanExt = 0U;
    tilingData.bandForm = 0U;
    tilingData.chunkRows = 0U;
    tilingData.diagForm = 0U;
    tilingData.phaseForm = 0U;
    constexpr uint64_t SAFETY = 4096U; /* bytes */
    const uint64_t ubBytes = (ubBudget > SAFETY) ? (ubBudget - SAFETY) : 0U;
    if (incx == 1) {
        SelectSsbmvDiagForm(tilingData, k, lda, incy, stageRows, ubBytes);
        SelectSsbmvPhaseForm(tilingData, k, lda, incy, stageRows, ubBytes);
    }
    if (tilingData.diagForm == 0U) {
        SelectSsbmvGeneralForm(tilingData, n, k, lda, incy, segFloats, stageRows, ubBytes);
    }
    /* A single staging pass covers a whole chunk when it can, so the run never has to be sized
     * larger than the chunk it serves. */
    if (stageRows > tilingData.chunkRows) {
        stageRows = tilingData.chunkRows;
    }
    tilingData.yStageRows = (stageRows < 1U) ? 1U : stageRows;
    tilingData.ubFit = SsbmvTilingFitsUb(tilingData, k, incy, segFloats, ubBytes) ? 1U : 0U;
    return tilingData;
}

/*!
 * \brief 行-对角形式下按成本模型重选核数并重算 tiling；准入不过则保持原核数。
 *
 * 老规则（每核至少 64 行）在窄带大 n 上会给出过多 block，而每个 block 的固定开销会被逐核
 * 重复支付。核数变了 blockRows 就变，tiling 要重算；万一新核数下 chunk 反而不满足行-对角
 * 形式的准入（UB 装不下 span），就退回原来的核数，避免把形状掉进通用路径。
 *
 * 调用方不得在「就地（x == y）且没有私有副本」时进入本函数：多核写回会覆盖别的核还要当 x 读的
 * 元素（见 aclblasSsbmv 入口处的兜底判断）。 */
void MaybeRetileForDiagForm(
    SsbmvTilingData& tiling, uint32_t& useCoreNum, uint32_t n, uint32_t k, uint32_t lda, uint32_t uplo, float alpha,
    float beta, int64_t incx, int64_t incy)
{
    if (tiling.diagForm == 0U || tiling.alpha == 0.0f || tiling.incy != 1) {
        return;
    }
    const uint32_t optCore = CalSsbmvDiagCoreNum(n, k, lda);
    if (optCore < 1U || optCore == useCoreNum) {
        return;
    }
    const SsbmvTilingData retiled = CalSsbmvTilingData(optCore, n, k, lda, uplo, alpha, beta, incx, incy);
    if (retiled.diagForm != 0U && retiled.ubFit != 0U) {
        useCoreNum = optCore;
        tiling = retiled;
    }
}

/*!
 * \brief |inc| in 64-bit. Negating INT_MIN as an int is undefined behaviour; widening first is not.
 */
uint64_t AbsInc(int64_t inc) { return static_cast<uint64_t>(inc > 0 ? inc : -inc); }

/*!
 * \brief 校验 aclblasSsbmv 的标量维度与指针参数。
 *
 * 拆成独立函数是因为这些检查本身就接近代码规范允许的函数长度上限（nbnc 行 <= 50）。
 * 顺序与 BLAS 惯例一致：标量维度在前、指针在后；n == 0 的提前返回不在这里，
 * 它必须先于 k/lda 的校验发生（任务书要求 n == 0 时即使 k 非法也返回 SUCCESS）。
 */
aclblasStatus_t ValidateSsbmvArgs(
    aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* a, int lda, const float* x, int incx,
    const float* beta, float* y, int incy)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE("aclblasSsbmv", "invalid uplo=%d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(k >= 0, OP_LOGE("aclblasSsbmv", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasSsbmv", "alpha must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    /* 写成 lda > k 而不是 lda >= k + 1：k == INT_MAX 时 k + 1 在 int 域溢出（UB）；int 的 lda
     * 不可能大于 INT_MAX，该比较等价地拒绝这种不存在合法 lda 的带宽。 */
    CHECK_RET(lda > k, OP_LOGE("aclblasSsbmv", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE("aclblasSsbmv", "beta must not be nullptr");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(y != nullptr, OP_LOGE("aclblasSsbmv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSsbmv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSsbmv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    /* alpha == 0 leaves A * x unread, so A and x are only required for a non-zero alpha. */
    if (*alpha != 0.0f) {
        CHECK_RET(a != nullptr, OP_LOGE("aclblasSsbmv", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSsbmv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    /* The kernel forms the physical index of a strided x/y as logical * |inc| in uint32
     * (XPhysicalPos/YPhysicalPos), while the buffers themselves are sized in uint64. Once
     * (n-1)*|inc| passes UINT32_MAX the product wraps to a small index that is still inside the
     * buffer, so the call silently reads and writes the wrong elements instead of failing. */
    const uint64_t xSpanElems = static_cast<uint64_t>(n - 1) * AbsInc(incx);
    if (xSpanElems > UINT32_MAX) {
        OP_LOGE(
            "aclblasSsbmv", "incx=%d is too large: (n-1)*|incx|=%llu exceeds UINT32_MAX %u", incx,
            static_cast<unsigned long long>(xSpanElems), UINT32_MAX);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t ySpanElems = static_cast<uint64_t>(n - 1) * AbsInc(incy);
    if (ySpanElems > UINT32_MAX) {
        OP_LOGE(
            "aclblasSsbmv", "incy=%d is too large: (n-1)*|incy|=%llu exceeds UINT32_MAX %u", incy,
            static_cast<unsigned long long>(ySpanElems), UINT32_MAX);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

/*!
 * \brief 就地调用（x == y）时给 x 准备一份私有副本。
 *
 * 就地调用必须整段先备份，单核也不例外：kernel 按 chunk 分块，chunk c 写回的 y 覆盖的正是
 * chunk c+1（或别的核）还要当 x 读的那几行。handle 自带的 workspace 够大就直接复用（不在每次
 * 调用里申请/释放）；不够就走 EnsureDefaultWorkspace 扩容（库自有的 workspace 原地增长，用户
 * 注入的库无权回收）；连扩都扩不出来时只能报错——没有备份没有正确解，绝不降级硬算。
 *
 * \param[out] xUsed   备份成功（copied 为 true）时指向 workspace 里的副本，否则仍为 x
 * \param[out] copied  就地调用成功备份时为 true；为 false 时保证 x 与 y 不相交
 */
aclblasStatus_t CopyInPlaceX(
    aclblasHandle_t handle, const float* x, const float* y, float alpha, uint64_t xSpan, aclrtStream stream,
    const float*& xUsed, bool& copied)
{
    copied = false;
    if (x != y || alpha == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const size_t xBytes = static_cast<size_t>(xSpan) * sizeof(float);
    if (GetEffectiveWorkspace(handle) == nullptr || GetEffectiveWorkspaceSize(handle) < xBytes) {
        const aclblasStatus_t resizeStatus = EnsureDefaultWorkspace(handle, xBytes);
        if (resizeStatus != ACLBLAS_STATUS_SUCCESS) {
            OP_LOGE(
                "aclblasSsbmv", "in-place x needs %zu bytes of workspace, expansion failed: %d", xBytes,
                static_cast<int>(resizeStatus));
            return resizeStatus;
        }
    }
    void* workspace = GetEffectiveWorkspace(handle);
    if (workspace == nullptr || GetEffectiveWorkspaceSize(handle) < xBytes) {
        /* 上面成功即保证容量，这里只是不让「没有副本」漏过去。 */
        OP_LOGE(
            "aclblasSsbmv", "in-place x needs %zu bytes of workspace, only %zu available", xBytes,
            GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const aclError copyRet = aclrtMemcpyAsync(workspace, xBytes, x, xBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (copyRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSsbmv", "in-place copy of x failed, ret=%d", static_cast<int>(copyRet));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    xUsed = static_cast<const float*>(workspace);
    copied = true;
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

/*!
 * \brief 拒绝任何 UB 预算装不下的 tiling。
 *
 * CalSsbmvTilingData 已对每个 UB 项按精确公式回验过；这里只是下发前的最后一道闸：把越界的
 * LocalTensor 分配带进核里会触发 VEC UB 越界异常（aivec error），而不是一个可诊断的返回码。
 */
aclblasStatus_t CheckSsbmvTilingFitsUb(const SsbmvTilingData& tiling, int n, int k, int lda, int64_t incx, int64_t incy)
{
    if (tiling.ubFit != 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    OP_LOGE(
        "aclblasSsbmv", "no tiling fits the device UB: n=%d k=%d lda=%d incx=%d incy=%d", n, k, lda,
        static_cast<int>(incx), static_cast<int>(incy));
    return ACLBLAS_STATUS_NOT_SUPPORTED;
}

aclblasStatus_t aclblasSsbmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasSsbmv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSsbmv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    /* n == 0 is a legal no-op and has to be answered before k, lda and the pointers are looked at. */
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t status = ValidateSsbmvArgs(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSsbmv", "argument validation failed, status=%d", static_cast<int>(status));
        return status;
    }

    aclrtStream useStream = handle->stream;

    uint32_t useCoreNum = CalSsbmvCoreNum(static_cast<uint32_t>(n));
    if (useCoreNum == 0) {
        OP_LOGE("aclblasSsbmv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const uint64_t xSpan = static_cast<uint64_t>(n - 1) * AbsInc(incx) + 1U;
    const float* xUsed = x;
    bool copied = false;
    const aclblasStatus_t copyStatus = CopyInPlaceX(handle, x, y, *alpha, xSpan, useStream, xUsed, copied);
    if (copyStatus != ACLBLAS_STATUS_SUCCESS) {
        /* CopyInPlaceX 已在失败点打印具体原因（workspace 大小 / 扩容 / memcpy 返回码）。 */
        return copyStatus;
    }

    SsbmvTilingData tiling = CalSsbmvTilingData(
        useCoreNum, static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(lda),
        static_cast<uint32_t>(uplo), *alpha, *beta, static_cast<int64_t>(incx), static_cast<int64_t>(incy));
    /* 就地且没有副本时禁止重选核数：多核下每个核写回的 y 都可能覆盖别的核还要当 x 读的元素。
     * CopyInPlaceX 在备份不出来时已经直接返回错误，所以这一支恒不成立，是结构性兜底。 */
    const bool inPlaceWithoutCopy = (x == y) && (*alpha != 0.0f) && !copied;
    if (!inPlaceWithoutCopy) {
        MaybeRetileForDiagForm(
            tiling, useCoreNum, static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(lda),
            static_cast<uint32_t>(uplo), *alpha, *beta, static_cast<int64_t>(incx), static_cast<int64_t>(incy));
    }
    const aclblasStatus_t fitStatus = CheckSsbmvTilingFitsUb(tiling, n, k, lda, incx, incy);
    if (fitStatus != ACLBLAS_STATUS_SUCCESS) {
        return fitStatus;
    }
    uint32_t numBlocks = CeilDiv<uint32_t>(static_cast<uint32_t>(n), tiling.blockRows);

    OP_LOGD(
        "aclblasSsbmv", "tiling: n=%u k=%u lda=%u uplo=%u coreNum=%u blockRows=%u chunkRows=%u blocks=%u", tiling.n,
        tiling.k, tiling.lda, tiling.uplo, tiling.useCoreNum, tiling.blockRows, tiling.chunkRows, numBlocks);

    ssbmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(const_cast<float*>(xUsed)),
        reinterpret_cast<uint8_t*>(y), nullptr, numBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
