/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <vector>

#include "ssymm_common_types.h"

#define SSYMM_LEFT_IS_FULL_WIDTH(colCount, fullWidthColCount) ((colCount) == (fullWidthColCount))
#define SSYMM_LEFT_HAS_DIRECT_CONTIG_PANEL_BUILD 1
#define SSYMM_LEFT_HAS_MIXED_SPLIT_PANEL_BUILD 1
#define SSYMM_LEFT_HAS_RUNTIME_STRATEGY 1
#define SSYMM_LEFT_HAS_RUNTIME_SUMMARY 1
#define SSYMM_LEFT_HAS_RUNTIME_TILE_DISPATCHER 1
#define SSYMM_LEFT_HAS_FORMAL_FAMILY_ROLE 1
#define SSYMM_LEFT_HAS_RECTANGLE_ROLE 1

enum class SsymmLeftRowGroupKind : uint32_t {
    SINGLE_ROW = 0,
    ROW_PAIR = 1,
    ROW_QUAD = 2,
};

enum class SsymmLeftPanelUplo : uint32_t {
    LOWER = 0,
    UPPER = 1,
};

enum class SsymmLeftPanelBuildKind : uint32_t {
    DIRECT_CONTIG = 0,
    MIRROR_GATHER = 1,
    MIXED_SPLIT = 2,
};

enum class SsymmLeftAPanelStrategy : uint32_t {
    DIRECT_CONTIG = 0,
    MIRROR_GATHER = 1,
    MIXED_SPLIT = 2,
};

#define SSYMM_LEFT_SELECT_ROW_GROUP_KIND(rowBlockCount) \
    (((rowBlockCount) >= 4) ?                           \
         SsymmLeftRowGroupKind::ROW_QUAD :              \
         (((rowBlockCount) >= 2) ? SsymmLeftRowGroupKind::ROW_PAIR : SsymmLeftRowGroupKind::SINGLE_ROW))

#define SSYMM_LEFT_PANEL_CLASSIFY_KIND(uplo, row, kBase, kCount)                                                     \
    (((uplo) == SsymmLeftPanelUplo::LOWER) ?                                                                         \
         (((row) >= ((kBase) + (kCount) - 1)) ?                                                                      \
              SsymmLeftPanelBuildKind::DIRECT_CONTIG :                                                               \
              (((row) < (kBase)) ? SsymmLeftPanelBuildKind::MIRROR_GATHER : SsymmLeftPanelBuildKind::MIXED_SPLIT)) : \
         (((row) <= (kBase)) ? SsymmLeftPanelBuildKind::DIRECT_CONTIG :                                              \
                               (((row) > ((kBase) + (kCount) - 1)) ? SsymmLeftPanelBuildKind::MIRROR_GATHER :        \
                                                                     SsymmLeftPanelBuildKind::MIXED_SPLIT)))

#define SSYMM_LEFT_PANEL_SELECT_STRATEGY(kind)                                                          \
    (((kind) == SsymmLeftPanelBuildKind::DIRECT_CONTIG) ?                                               \
         SsymmLeftAPanelStrategy::DIRECT_CONTIG :                                                       \
         (((kind) == SsymmLeftPanelBuildKind::MIRROR_GATHER) ? SsymmLeftAPanelStrategy::MIRROR_GATHER : \
                                                               SsymmLeftAPanelStrategy::MIXED_SPLIT))

#define SSYMM_LEFT_PANEL_DIRECT_OFFSET(uplo, row, kBase, kCount) \
    (((uplo) == SsymmLeftPanelUplo::LOWER) ?                     \
         0U :                                                    \
         (((row) <= (kBase)) ? 0U : (((row) > ((kBase) + (kCount) - 1)) ? (kCount) : ((row) - (kBase)))))

#define SSYMM_LEFT_PANEL_DIRECT_COUNT(uplo, row, kBase, kCount)                                                \
    (((uplo) == SsymmLeftPanelUplo::LOWER) ?                                                                   \
         (((row) >= ((kBase) + (kCount) - 1)) ? (kCount) : (((row) < (kBase)) ? 0U : ((row) - (kBase) + 1))) : \
         (((row) <= (kBase)) ? (kCount) : (((row) > ((kBase) + (kCount) - 1)) ? 0U : (((kBase) + (kCount)) - (row)))))

enum class RightCubeUplo : uint32_t {
    LOWER = 0,
    UPPER = 1,
};

enum class RightCubeChunkKind : uint32_t {
    DIRECT_FULL = 0,
    MIRROR_FULL = 1,
    MIXED_DIAG = 2,
    TAIL = 3,
};

enum class RightCubeExecuteChunkKind : uint32_t {
    DIRECT_FULL = 0,
    MIRROR_FULL = 1,
    MIXED_DIAG = 2,
    TAIL = 3,
    MIXED_DIAG_FULL_TILE = 4,
};

enum class RightCubeExecuteGeometry : uint32_t {
    DIRECT = 0,
    MIRROR = 1,
    MIXED_DIAG = 2,
};

enum class RightCubeExecuteExtent : uint32_t {
    FULL_TILE = 0,
    COL_TAIL = 1,
    K_TAIL = 2,
    COLK_TAIL = 3,
};

enum class RightCubePackMode : uint32_t {
    FAST_DIRECT = 0,
    FAST_MIRROR = 1,
    FAST_MIXED_FULL_TILE = 2,
    GENERIC_DIRECT = 3,
    GENERIC_MIRROR = 4,
    GENERIC_SYMMETRIC = 5,
};

enum class RightCubeLoadKind : uint32_t {
    DIRECT = 0,
    MIRROR_TRANSPOSED = 1,
    MIXED = 2,
};

enum class RightCubeStrategy : uint32_t {
    FALLBACK_TO_SCRATCH = 0,
    PARTIAL_CUBE = 1,
};

struct RightCubeTileShape {
    uint32_t tm;
    uint32_t tn;
    uint32_t tk;
};

struct RightCubeChunkPlanItem {
    uint32_t colBase;
    uint32_t colCount;
    uint32_t kBase;
    uint32_t kCount;
    RightCubeChunkKind kind;
};

struct RightCubeChunkLocalPolicy {
    RightCubeStrategy strategy;
    uint32_t tm;
    uint32_t tn;
    uint32_t tk;
    uint32_t cubeChunkMask;
    uint32_t fallbackChunkMask;
};

struct RightCubeExecuteChunkPlanItem {
    RightCubeChunkPlanItem chunk;
    RightCubeExecuteChunkKind executeKind;
};

struct RightCubeExecuteRuntimePlan {
    uint32_t cubeChunkMask = 0;
    uint32_t fallbackChunkMask = 0;
    std::vector<RightCubeExecuteChunkPlanItem> chunkItems;
};

struct SsymmRightCubeUnifiedConfig {
    uint32_t uplo;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t tm;
    uint32_t tn;
    uint32_t tk;
    uint32_t strategy;
    uint32_t cubeChunkMask;
    uint32_t fallbackChunkMask;
    uint32_t chunkColBase;
    uint32_t chunkColCount;
    uint32_t chunkKBase;
    uint32_t chunkKCount;
    float alpha;
    float beta;
};

struct SsymmLeftCubeConfig {
    uint32_t uplo;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t rowBase;
    uint32_t rowCount;
    uint32_t colBase;
    uint32_t colCount;
    uint32_t kBase;
    uint32_t kCount;
    float alpha;
    float beta;
};

enum class RightLowerChunkKind : uint32_t {
    DIRECT_FULL = 0,
    MIRROR_FULL = 1,
    MIXED_DIAG = 2,
    TAIL = 3,
};

struct RightLowerCubeTileShape {
    uint32_t tn;
    uint32_t tk;
};

struct RightLowerCubeChunkSummary {
    uint32_t directFull = 0;
    uint32_t mirrorFull = 0;
    uint32_t mixedDiag = 0;
    uint32_t tail = 0;
};

#define SSYMM_RIGHT_CUBE_CLASSIFY_CHUNK_KIND(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape) \
    (((rowCount) < (tileShape).tm || (colCount) < (tileShape).tn || (kCount) < (tileShape).tk) ?          \
         RightCubeChunkKind::TAIL :                                                                       \
         ((uplo) == RightCubeUplo::LOWER ?                                                                \
              (((kBase) >= ((colBase) + (colCount) - 1)) ?                                                \
                   RightCubeChunkKind::DIRECT_FULL :                                                      \
                   ((((kBase) + (kCount) - 1) < (colBase)) ? RightCubeChunkKind::MIRROR_FULL :            \
                                                             RightCubeChunkKind::MIXED_DIAG)) :           \
              (((((kBase) + (kCount) - 1) <= (colBase)) ?                                                 \
                    RightCubeChunkKind::DIRECT_FULL :                                                     \
                    (((kBase) > ((colBase) + (colCount) - 1)) ? RightCubeChunkKind::MIRROR_FULL :         \
                                                                RightCubeChunkKind::MIXED_DIAG)))))

#define SSYMM_RIGHT_CUBE_LOAD_KIND_FROM_CHUNK(kind)                                            \
    (((kind) == RightCubeChunkKind::DIRECT_FULL) ?                                             \
         RightCubeLoadKind::DIRECT :                                                           \
         (((kind) == RightCubeChunkKind::MIRROR_FULL) ? RightCubeLoadKind::MIRROR_TRANSPOSED : \
                                                        RightCubeLoadKind::MIXED))

#define SSYMM_RIGHT_CUBE_CHUNK_BIT(kind) (1u << static_cast<uint32_t>(kind))
#define SSYMM_RIGHT_CUBE_MASK_HAS(mask, kind) (((mask) & SSYMM_RIGHT_CUBE_CHUNK_BIT(kind)) != 0)
#define SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(kind) (1u << static_cast<uint32_t>(kind))
#define SSYMM_RIGHT_CUBE_EXEC_MASK_HAS(mask, kind) (((mask) & SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(kind)) != 0)

enum class SsymmTileKind : uint32_t {
    DirectFull = 0,
    MirrorFull = 1,
    MixedDiag = 2,
    Tail = 3,
};

struct SsymmTileClassifyShape {
    uint32_t tm = 0;
    uint32_t tn = 0;
    uint32_t tk = 0;
};

#define SSYMM_CLASSIFY_RIGHT_TILE_KIND(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape)                   \
    (((rowCount) < (tileShape).tm || (colCount) < (tileShape).tn || (kCount) < (tileShape).tk) ?                      \
         SsymmTileKind::Tail :                                                                                        \
         (((uplo) == ACLBLAS_LOWER) ?                                                                                 \
              (((kBase) >= ((colBase) + (colCount) - 1)) ?                                                            \
                   SsymmTileKind::DirectFull :                                                                        \
                   ((((kBase) + (kCount) - 1) < (colBase)) ? SsymmTileKind::MirrorFull : SsymmTileKind::MixedDiag)) : \
              (((((kBase) + (kCount) - 1) <= (colBase)) ?                                                             \
                    SsymmTileKind::DirectFull :                                                                       \
                    (((kBase) > ((colBase) + (colCount) - 1)) ? SsymmTileKind::MirrorFull :                           \
                                                                SsymmTileKind::MixedDiag)))))

#define SSYMM_CLASSIFY_LEFT_TILE_KIND(uplo, rowCount, rowBase, colCount, kBase, kCount, tileShape)                  \
    (((rowCount) < (tileShape).tm || (colCount) < (tileShape).tn || (kCount) < (tileShape).tk) ?                    \
         SsymmTileKind::Tail :                                                                                      \
         (((uplo) == ACLBLAS_LOWER) ? (((rowBase) >= ((kBase) + (kCount) - 1)) ?                                    \
                                           SsymmTileKind::DirectFull :                                              \
                                           ((((rowBase) + (rowCount) - 1) < (kBase)) ? SsymmTileKind::MirrorFull :  \
                                                                                       SsymmTileKind::MixedDiag)) : \
                                      (((((rowBase) + (rowCount) - 1) <= (kBase)) ?                                 \
                                            SsymmTileKind::DirectFull :                                             \
                                            (((rowBase) > ((kBase) + (kCount) - 1)) ? SsymmTileKind::MirrorFull :   \
                                                                                      SsymmTileKind::MixedDiag)))))

#define SSYMM_CLASSIFY_TILE_KIND(side, uplo, rowCount, colBase, colCount, kBase, kCount, tileShape)    \
    (((side) == ACLBLAS_SIDE_RIGHT) ?                                                                  \
         SSYMM_CLASSIFY_RIGHT_TILE_KIND(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape) : \
         SSYMM_CLASSIFY_LEFT_TILE_KIND(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape))

inline constexpr uint32_t SSYMM_RIGHT_CUBE_PARTIAL_MAX_M = 256;

// 按 RIGHT lower 的半边存储关系分类一个旧版 chunk。
// direct/mirror 完整块可以走连续搬运，mixed 对角块和尾块需要慢路径处理。
inline RightLowerChunkKind ClassifyRightLowerChunk(
    uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount, const RightLowerCubeTileShape& tileShape)
{
    if (colCount < tileShape.tn || kCount < tileShape.tk) {
        return RightLowerChunkKind::TAIL;
    }

    const uint32_t colEnd = colBase + colCount - 1;
    if (kBase >= colEnd) {
        return RightLowerChunkKind::DIRECT_FULL;
    }

    const uint32_t kEnd = kBase + kCount - 1;
    if (kEnd < colBase) {
        return RightLowerChunkKind::MIRROR_FULL;
    }
    return RightLowerChunkKind::MIXED_DIAG;
}

// 在 host 规划阶段按通用 chunk 类型分类 RIGHT tile。
// 这里复用宏规则，保证 host/device 对 direct、mirror、mixed、tail 的判断一致。
inline RightCubeChunkKind ClassifyRightCubeChunkHost(
    RightCubeUplo uplo, uint32_t rowCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    const RightCubeTileShape& tileShape)
{
    return SSYMM_RIGHT_CUBE_CLASSIFY_CHUNK_KIND(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape);
}

// 将 RIGHT tile 分类为 chunk-local 调度使用的执行类型。
// 单独区分完整对角块，方便 cube mask 和 fallback mask 拆分不同工作。
inline RightCubeExecuteChunkKind ClassifyRightCubeExecuteChunkKindHost(
    RightCubeUplo uplo, uint32_t rowCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    const RightCubeTileShape& tileShape)
{
    (void)rowCount;
    if (colCount < tileShape.tn || kCount < tileShape.tk) {
        return RightCubeExecuteChunkKind::TAIL;
    }

    RightCubeExecuteGeometry geometry = RightCubeExecuteGeometry::MIXED_DIAG;
    if (uplo == RightCubeUplo::LOWER) {
        if (kBase >= colBase + colCount - 1) {
            geometry = RightCubeExecuteGeometry::DIRECT;
        } else if (kBase + kCount - 1 < colBase) {
            geometry = RightCubeExecuteGeometry::MIRROR;
        }
    } else {
        if (kBase + kCount - 1 <= colBase) {
            geometry = RightCubeExecuteGeometry::DIRECT;
        } else if (kBase > colBase + colCount - 1) {
            geometry = RightCubeExecuteGeometry::MIRROR;
        }
    }
    if (geometry == RightCubeExecuteGeometry::DIRECT) {
        return RightCubeExecuteChunkKind::DIRECT_FULL;
    }
    if (geometry == RightCubeExecuteGeometry::MIRROR) {
        return RightCubeExecuteChunkKind::MIRROR_FULL;
    }
    if (colBase == kBase) {
        return RightCubeExecuteChunkKind::MIXED_DIAG_FULL_TILE;
    }
    return RightCubeExecuteChunkKind::MIXED_DIAG;
}

// 为一个 RIGHT tile 构造 host 侧 chunk 描述。
// 描述中保存 tile 边界和 direct/mirror/mixed/tail 基础分类。
inline RightCubeChunkPlanItem BuildRightCubeChunkPlanItem(
    RightCubeUplo uplo, uint32_t rowCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    const RightCubeTileShape& tileShape)
{
    return RightCubeChunkPlanItem{
        colBase,
        colCount,
        kBase,
        kCount,
        ClassifyRightCubeChunkHost(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape),
    };
}

// 为一个 RIGHT tile 构造执行期 chunk 描述。
// 它在基础边界信息上补充更细的 execute kind，供 runtime mask 使用。
inline RightCubeExecuteChunkPlanItem BuildRightCubeExecuteChunkPlanItem(
    RightCubeUplo uplo, uint32_t rowCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    const RightCubeTileShape& tileShape)
{
    return RightCubeExecuteChunkPlanItem{
        BuildRightCubeChunkPlanItem(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape),
        ClassifyRightCubeExecuteChunkKindHost(uplo, rowCount, colBase, colCount, kBase, kCount, tileShape),
    };
}

// 构造 RIGHT 路径中应走 cube 的 chunk 列表。
// direct、mirror 和完整对角块进入列表，普通 mixed 和 tail 留给 fallback mask。
inline RightCubeExecuteRuntimePlan BuildRightCubeExecuteRuntimePlan(
    RightCubeUplo uplo, const SsymmRightCubeUnifiedConfig& baseConfig)
{
    RightCubeExecuteRuntimePlan plan{};
    plan.cubeChunkMask = SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::DIRECT_FULL) |
                         SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIRROR_FULL) |
                         SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG_FULL_TILE);
    plan.fallbackChunkMask = SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG) |
                             SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::TAIL);

    RightCubeTileShape tileShape{baseConfig.tm, baseConfig.tn, baseConfig.tk};
    for (uint32_t colBase = 0; colBase < baseConfig.n; colBase += baseConfig.tn) {
        const uint32_t colCount = (colBase + baseConfig.tn <= baseConfig.n) ? baseConfig.tn : (baseConfig.n - colBase);
        for (uint32_t kBase = 0; kBase < baseConfig.n; kBase += baseConfig.tk) {
            const uint32_t kCount = (kBase + baseConfig.tk <= baseConfig.n) ? baseConfig.tk : (baseConfig.n - kBase);
            RightCubeExecuteChunkPlanItem item =
                BuildRightCubeExecuteChunkPlanItem(uplo, baseConfig.tm, colBase, colCount, kBase, kCount, tileShape);
            if (SSYMM_RIGHT_CUBE_EXEC_MASK_HAS(plan.cubeChunkMask, item.executeKind)) {
                plan.chunkItems.push_back(item);
            }
        }
    }
    return plan;
}

// 按 LOWER 语义构造 RIGHT runtime plan。
// 返回值只包含 chunk 元数据，真正下发 device 的 config 会在后续生成。
inline RightCubeExecuteRuntimePlan BuildRightCubeLowerRuntimePlan(const SsymmRightCubeUnifiedConfig& baseConfig)
{
    return BuildRightCubeExecuteRuntimePlan(RightCubeUplo::LOWER, baseConfig);
}

// 按 UPPER 语义构造 RIGHT runtime plan。
// 返回值只包含 chunk 元数据，真正下发 device 的 config 会在后续生成。
inline RightCubeExecuteRuntimePlan BuildRightCubeUpperRuntimePlan(const SsymmRightCubeUnifiedConfig& baseConfig)
{
    return BuildRightCubeExecuteRuntimePlan(RightCubeUplo::UPPER, baseConfig);
}

// 计算一次 RIGHT chunk-local 执行所需的临时 workspace 字节数。
// workspace 中依次放置 packed A、packed B 和 chunk 输出。
inline size_t CalcRightCubeChunkWorkspaceBytes(const SsymmRightCubeUnifiedConfig& config)
{
    const size_t aChunkBytes = static_cast<size_t>(config.tk) * static_cast<size_t>(config.tn) * sizeof(float);
    const size_t bChunkBytes = static_cast<size_t>(config.m) * static_cast<size_t>(config.tk) * sizeof(float);
    const size_t outChunkBytes = static_cast<size_t>(config.m) * static_cast<size_t>(config.tn) * sizeof(float);
    return aChunkBytes + bChunkBytes + outChunkBytes;
}

// 根据基础 RIGHT config 和 host 侧 plan item 生成单个 chunk 的 device config。
// runtime mask 会一并写入，使每次 launch 知道哪些工作属于 cube 或 fallback。
inline SsymmRightCubeUnifiedConfig BuildRightCubeChunkRuntimeConfig(
    const SsymmRightCubeUnifiedConfig& baseConfig, const RightCubeChunkPlanItem& item, uint32_t cubeChunkMask,
    uint32_t fallbackChunkMask)
{
    SsymmRightCubeUnifiedConfig config = baseConfig;
    config.cubeChunkMask = cubeChunkMask;
    config.fallbackChunkMask = fallbackChunkMask;
    config.chunkColBase = item.colBase;
    config.chunkColCount = item.colCount;
    config.chunkKBase = item.kBase;
    config.chunkKCount = item.kCount;
    return config;
}

// 根据当前 shape 和 leading dimension 选择 RIGHT backend 策略。
// 已支持的规则 shape 走 chunk-local cube，其余合法 shape 通过 scratch 回退。
inline RightCubeStrategy SelectRightCubeStrategy(
    RightCubeUplo uplo, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    (void)uplo;
    (void)m;
    (void)n;
    (void)lda;
    (void)ldb;
    (void)ldc;
    // FIXME(issue #863): PARTIAL_CUBE 的 chunk-local cube 路径（pack/dense/accum）计算结果错误，
    // 在修复前统一回退到 FALLBACK_TO_SCRATCH（已验证全 shape 正确）。跟踪与修复计划见 issue #863；
    // 修复完成前请勿将该回退常量替换为 chunk-local 路径或复用其配置常量视为已验证行为。
    return RightCubeStrategy::FALLBACK_TO_SCRATCH;
}

// 构造 host 生成 RIGHT config 时使用的 chunk-local 策略。
// 这里集中设置 tile 大小、策略选择以及默认 cube/fallback chunk mask。
inline RightCubeChunkLocalPolicy BuildRightCubeChunkLocalPolicy(
    RightCubeUplo uplo, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc)
{
    RightCubeChunkLocalPolicy policy{};
    policy.strategy = SelectRightCubeStrategy(uplo, m, n, lda, ldb, ldc);
    policy.tm = 8;
    policy.tn = 64;
    policy.tk = 64;
    policy.cubeChunkMask = SSYMM_RIGHT_CUBE_CHUNK_BIT(RightCubeChunkKind::DIRECT_FULL) |
                           SSYMM_RIGHT_CUBE_CHUNK_BIT(RightCubeChunkKind::MIRROR_FULL);
    policy.fallbackChunkMask = SSYMM_RIGHT_CUBE_CHUNK_BIT(RightCubeChunkKind::MIXED_DIAG) |
                               SSYMM_RIGHT_CUBE_CHUNK_BIT(RightCubeChunkKind::TAIL);
    return policy;
}
