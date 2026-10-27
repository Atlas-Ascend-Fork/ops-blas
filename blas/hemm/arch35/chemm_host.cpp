/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <cmath>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "chemm_tiling_data.h"
void chemm_simt_do(const void*, const void*, void*, void*, ChemmTiling, uint32_t, void*);

void chemm_ordered_pack_do(const void*, const void*, void*, ChemmTiling, uint32_t, void*);
void chemm_pack_do(const void*, const void*, void*, ChemmTiling, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*);
void chemm_pack3_fast_do(const void*, const void*, void*, ChemmTiling, uint32_t, uint32_t, uint32_t, uint32_t, void*);
void chemm_bound_reset_do(void*, uint32_t, void*);
void chemm_combined_reset_do(void*, void*, uint64_t, uint32_t, uint32_t, void*);
void chemm_combine_do(
    const void*, const void*, void*, void*, ChemmTiling, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*,
    uint64_t);
void chemm_combine_norisk_do(
    const void*, const void*, void*, void*, ChemmTiling, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*,
    uint64_t, void*, uint32_t);
void chemm_combine_narrow_output_do(const void*, void*, ChemmTiling, uint32_t, uint32_t, uint32_t, void*);
void chemm_cube_kernel_do(uint8_t*, const uint8_t*, uint8_t*, const ChemmCubeGemmTilingData&, uint32_t, void*);
void chemm_queue_reset_do(void*, uint64_t, uint32_t, void*);
void chemm_queue_fallback_do(
    const void*, const void*, void*, void*, uint64_t, ChemmTiling, uint32_t, void*, void*, const void*);
// Workload taxonomy.  Dimensions, aspect ratios and alignment properties
// select a policy family; launch execution remains independent of classification.
enum class ChemmWorkloadFamily {
    Generic,
    SmallLaunchBoundMicro,
    SmallLaunchBound,
    RightVeryNarrow,
    RightNearDeepN,
    RightModerateSquare,
    RightBalancedWide,
    LeftPrecisionRiskNarrow,
    RightRelaxedRiskBalanced,
    DeepMShortK,
    DeepKRiskManaged,
    RightShortKSquare
};

enum class ChemmPackMode { Checked, FastThreePlane };
enum class ChemmResetMode { Bounded, Combined };
enum class ChemmCombineMode { RiskChecked, RiskCheckedInline, BoundedNoRisk };

struct ChemmWorkloadClass {
    ChemmWorkloadFamily family;
    int64_t k;
    bool left;
    bool right;
    bool upper;
    bool lower;
    bool square;
    bool rightNearSquareSplit1;
    bool rightNearConservative;
    bool rightTallMidK;
    bool rightMidSquareSplit2;
    bool rightTallTransition;
    bool rightNarrowWideM;
    bool rightCompactHalf;
    bool leftWideMidK;
    bool leftNearWideN;
    bool leftSquareTransition;
    bool leftDeepKModalN;
    bool leftNearVeryNarrow;
    bool leftBalancedNearBoundary;
    bool rightDeepCompactLower;
};

static constexpr bool ChemmBetween(int64_t value, int64_t lower, int64_t upper)
{
    return value >= lower && value <= upper;
}

static constexpr bool ChemmIsPowerOfTwo(uint32_t value) { return value != 0U && (value & (value - 1U)) == 0U; }

struct ChemmShape {
    bool left;
    bool right;
    bool upper;
    bool lower;
    int64_t m;
    int64_t n;
    int64_t k;
    bool square;
    bool boundedDimensions;
};

static constexpr ChemmShape MakeChemmShape(const ChemmTiling& t)
{
    const bool left = t.left != 0U;
    const bool upper = t.upper != 0U;
    return {left, !left, upper, !upper, t.m, t.n, left ? t.m : t.n, t.m == t.n, t.m <= 16384 && t.n <= 16384};
}

static constexpr bool IsAlignedPowerOfTwoSquare(const ChemmShape& s)
{
    return s.boundedDimensions && s.left && s.upper && s.square && ChemmIsPowerOfTwo(static_cast<uint32_t>(s.m));
}

static constexpr bool IsRightNearDeepFamily(const ChemmShape& s)
{
    const bool lowerDeep = s.right && s.lower && s.n > 2850 && s.n <= 3075 && ChemmBetween(s.m, 1024, 1535);
    const bool upperDeep = s.right && s.upper && s.n > 2500 && s.n < 2700 && ChemmBetween(s.m, 256, 799);
    return lowerDeep || upperDeep;
}

static constexpr bool IsRightModerateFamily(const ChemmShape& s)
{
    return s.right && s.square &&
           ((s.upper && ChemmBetween(s.m, 1480, 2020)) || (s.lower && ChemmBetween(s.m, 1560, 1940)));
}

static constexpr bool IsRightRelaxedFamily(const ChemmShape& s)
{
    return s.right && s.upper && ChemmBetween(s.m, 3076, 3399) && ChemmBetween(s.n, 3076, 3399) &&
           (s.m > s.n ? s.m - s.n : s.n - s.m) <= (s.m > s.n ? s.n : s.m) / 10;
}

static constexpr bool IsDeepKRecovered(const ChemmShape& s)
{
    const bool envelope = ChemmBetween(s.m, 512, 4096) && ChemmBetween(s.n, 512, 4096) && s.k > 3075;
    return envelope && (s.left || s.lower || s.m <= 2048 || s.m >= 3072);
}

static constexpr ChemmWorkloadFamily SelectChemmLaunchFamily(const ChemmShape& s)
{
    const bool alignedSquare = IsAlignedPowerOfTwoSquare(s);
    if (alignedSquare && ChemmBetween(s.m, 48, 95))
        return ChemmWorkloadFamily::SmallLaunchBoundMicro;
    if (alignedSquare && ChemmBetween(s.m, 384, 640))
        return ChemmWorkloadFamily::SmallLaunchBound;
    return ChemmWorkloadFamily::Generic;
}

static constexpr ChemmWorkloadFamily SelectRightFamilyBeforeLeftRisk(const ChemmShape& s)
{
    if (s.right && s.upper && s.n < 64 && s.m > 2300)
        return ChemmWorkloadFamily::RightVeryNarrow;
    if (IsRightNearDeepFamily(s))
        return ChemmWorkloadFamily::RightNearDeepN;
    if (IsRightModerateFamily(s))
        return ChemmWorkloadFamily::RightModerateSquare;
    if (s.right && s.upper && ChemmBetween(s.m, 2100, 2350) && ChemmBetween(s.n, 1500, 1750))
        return ChemmWorkloadFamily::RightBalancedWide;
    return ChemmWorkloadFamily::Generic;
}

static constexpr ChemmWorkloadFamily SelectFamilyAfterLeftRisk(const ChemmShape& s)
{
    if (s.left && s.upper && s.m > 2800 && ChemmBetween(s.n, 64, 256))
        return ChemmWorkloadFamily::LeftPrecisionRiskNarrow;
    if (IsRightRelaxedFamily(s))
        return ChemmWorkloadFamily::RightRelaxedRiskBalanced;
    if (s.right && s.lower && s.m > 2048 && ChemmBetween(s.n, 128, 512))
        return ChemmWorkloadFamily::DeepMShortK;
    if (IsDeepKRecovered(s))
        return ChemmWorkloadFamily::DeepKRiskManaged;
    if (s.right && s.square && s.k <= 2048)
        return ChemmWorkloadFamily::RightShortKSquare;
    return ChemmWorkloadFamily::Generic;
}

static constexpr ChemmWorkloadFamily SelectWorkloadFamily(const ChemmShape& s)
{
    const ChemmWorkloadFamily launchFamily = SelectChemmLaunchFamily(s);
    if (launchFamily != ChemmWorkloadFamily::Generic)
        return launchFamily;
    const ChemmWorkloadFamily rightFamily = SelectRightFamilyBeforeLeftRisk(s);
    if (rightFamily != ChemmWorkloadFamily::Generic)
        return rightFamily;
    return SelectFamilyAfterLeftRisk(s);
}

static constexpr ChemmWorkloadClass MakeChemmWorkloadClass(const ChemmShape& s, ChemmWorkloadFamily family)
{
    const bool nearSquareRisk = (s.upper && s.k >= 2800) || (s.lower && ((s.k > 2400 && s.k < 2640) || s.k >= 3000));
    const bool nearK = s.right && s.k > 2048 && s.k <= 3075;
    const bool rightNearSquareSplit1 = nearK && s.square && !nearSquareRisk;
    const bool rightNearConservative = nearK && (!s.square || nearSquareRisk);
    const bool rightTallMidK = s.right && s.k >= 1536 && s.k < 2048 && s.m >= 2048;
    const bool rightMidSquareSplit2 = s.right && s.square && s.m >= 1500 && s.m < 2048;
    const bool rightTallTransition = s.right && s.k >= 1520 && s.k <= 1570 && s.m >= 2200;
    const bool rightNarrowWideM = s.right && s.n >= 128 && s.n <= 1251 && s.m >= 1200;
    const bool rightCompactHalf = s.boundedDimensions && s.right && s.lower &&
                                  ChemmIsPowerOfTwo(static_cast<uint32_t>(s.m)) &&
                                  ChemmIsPowerOfTwo(static_cast<uint32_t>(s.n)) && s.n * 10 >= s.m * 4 &&
                                  s.n * 10 <= s.m * 6 && s.m >= 768 && s.m < 1200;
    const bool leftWideMidK = s.left && ChemmBetween(s.m, 200, 1000) && s.n >= 800;
    const bool leftNearWideN = s.left && ChemmBetween(s.k, 1500, 3075) && s.n >= s.m;
    const bool leftSquareTransition = s.left && s.lower && s.square && ChemmBetween(s.m, 1660, 1720);
    const bool leftDeepKModalN = s.left && ChemmBetween(s.k, 2100, 2150) && ChemmBetween(s.n, 500, 600);
    const bool leftNearVeryNarrow = s.left && s.k > 2048 && s.k <= 3075 && s.n <= 128;
    const bool leftBalancedNearBoundary = s.left && s.k > 2048 && s.k <= 3075 && s.n < s.m && s.n >= s.m - s.m / 5;
    const bool rightDeepCompactLower = s.right && s.lower && s.n > 3075 && s.m < 768 && s.n < 3450;
    return {
        family,
        s.k,
        s.left,
        s.right,
        s.upper,
        s.lower,
        s.square,
        rightNearSquareSplit1,
        rightNearConservative,
        rightTallMidK,
        rightMidSquareSplit2,
        rightTallTransition,
        rightNarrowWideM,
        rightCompactHalf,
        leftWideMidK,
        leftNearWideN,
        leftSquareTransition,
        leftDeepKModalN,
        leftNearVeryNarrow,
        leftBalancedNearBoundary,
        rightDeepCompactLower};
}

static constexpr ChemmWorkloadClass ClassifyChemmWorkload(const ChemmTiling& t)
{
    const ChemmShape shape = MakeChemmShape(t);
    return MakeChemmWorkloadClass(shape, SelectWorkloadFamily(shape));
}

struct ChemmCubePolicy {
    uint32_t paddedK;
    uint32_t products;
    uint32_t tileM;
    uint32_t tileN;
    ChemmPackMode packMode;
    ChemmResetMode resetMode;
    ChemmCombineMode combineMode;
    ChemmRiskProfile riskProfile;
    bool resetQueue;
    bool queueFallback;
    uint32_t splits;
};

struct ChemmModePolicy {
    ChemmPackMode packMode = ChemmPackMode::Checked;
    ChemmResetMode resetMode = ChemmResetMode::Bounded;
    ChemmCombineMode combineMode = ChemmCombineMode::RiskChecked;
    bool skipQueue = false;
};

static void SelectRightChemmSplits(const ChemmWorkloadClass& workload, bool scalarConservative, uint32_t& splits)
{
    if (scalarConservative &&
        (workload.rightNearSquareSplit1 || workload.family == ChemmWorkloadFamily::RightVeryNarrow))
        splits = 1;
    if (scalarConservative && workload.rightNearConservative)
        splits = 2;
    if (workload.rightTallTransition)
        splits = 1;
    else if (scalarConservative && (workload.rightTallMidK || workload.rightMidSquareSplit2))
        splits = 2;
    if (scalarConservative && (workload.rightNarrowWideM || workload.rightCompactHalf))
        splits = 1;
}

static void SelectLeftChemmSplits(const ChemmWorkloadClass& workload, bool scalarConservative, uint32_t& splits)
{
    if (scalarConservative && workload.leftWideMidK)
        splits = 1;
    if (scalarConservative && workload.leftNearWideN)
        splits = 2;
    if (workload.leftSquareTransition)
        splits = 1;
    if (scalarConservative && workload.leftDeepKModalN)
        splits = 1;
    if (scalarConservative && workload.leftNearVeryNarrow)
        splits = 2;
    if (scalarConservative && (workload.leftBalancedNearBoundary || workload.rightDeepCompactLower))
        splits = 2;
}

static uint32_t SelectChemmSplits(const ChemmWorkloadClass& workload, uint32_t conservativeSplits)
{
    uint32_t splits = conservativeSplits;
    const bool scalarConservative = splits < 8U;
    SelectRightChemmSplits(workload, scalarConservative, splits);
    SelectLeftChemmSplits(workload, scalarConservative, splits);
    return splits;
}

static ChemmModePolicy SelectChemmModes(const ChemmWorkloadClass& workload)
{
    ChemmModePolicy mode;
    if (workload.family == ChemmWorkloadFamily::DeepMShortK) {
        mode.combineMode = ChemmCombineMode::RiskCheckedInline;
        mode.skipQueue = true;
    }
    if (workload.family == ChemmWorkloadFamily::SmallLaunchBoundMicro) {
        mode.resetMode = ChemmResetMode::Combined;
        mode.combineMode = ChemmCombineMode::BoundedNoRisk;
        mode.skipQueue = true;
    } else if (
        workload.family == ChemmWorkloadFamily::SmallLaunchBound ||
        workload.family == ChemmWorkloadFamily::RightVeryNarrow) {
        mode.combineMode = ChemmCombineMode::BoundedNoRisk;
        mode.skipQueue = true;
    } else if (workload.family == ChemmWorkloadFamily::LeftPrecisionRiskNarrow) {
        mode.packMode = ChemmPackMode::FastThreePlane;
        mode.combineMode = ChemmCombineMode::BoundedNoRisk;
    } else if (
        workload.family == ChemmWorkloadFamily::RightRelaxedRiskBalanced ||
        workload.family == ChemmWorkloadFamily::RightModerateSquare ||
        workload.family == ChemmWorkloadFamily::RightBalancedWide ||
        workload.family == ChemmWorkloadFamily::RightShortKSquare) {
        mode.combineMode = ChemmCombineMode::BoundedNoRisk;
    } else if (workload.family == ChemmWorkloadFamily::DeepKRiskManaged) {
        mode.combineMode = ChemmCombineMode::BoundedNoRisk;
        mode.skipQueue = false;
    }
    return mode;
}

static ChemmCubePolicy SelectChemmCubePolicy(
    const ChemmTiling& t, const ChemmWorkloadClass& workload, uint32_t conservativeSplits, uint32_t genericProducts)
{
    const bool acceleratedProducts = workload.family == ChemmWorkloadFamily::RightNearDeepN ||
                                     workload.family == ChemmWorkloadFamily::RightRelaxedRiskBalanced ||
                                     workload.family == ChemmWorkloadFamily::LeftPrecisionRiskNarrow ||
                                     workload.family == ChemmWorkloadFamily::DeepKRiskManaged;
    const bool smallMWideN = workload.right && ChemmBetween(t.m, 128, 192) && ChemmBetween(t.n, 1088, 1300);
    const bool tileM64 = workload.family == ChemmWorkloadFamily::LeftPrecisionRiskNarrow || (smallMWideN && t.n < 1200);
    const bool tileN64 = workload.family == ChemmWorkloadFamily::LeftPrecisionRiskNarrow ||
                         (smallMWideN && t.n >= 1200) ||
                         (workload.family == ChemmWorkloadFamily::RightNearDeepN && workload.upper);
    const ChemmModePolicy mode = SelectChemmModes(workload);
    const ChemmRiskProfile riskProfile =
        workload.k > 3075 ? ChemmRiskProfile::DeepKConservative : ChemmRiskProfile::Default;

    constexpr uint32_t narrowKGranularity = 32U;
    const uint32_t rightVeryNarrowPaddedK =
        (static_cast<uint32_t>(workload.k) + narrowKGranularity - 1) / narrowKGranularity * narrowKGranularity;
    const uint32_t paddedK = workload.family == ChemmWorkloadFamily::RightVeryNarrow ?
                                 rightVeryNarrowPaddedK :
                                 static_cast<uint32_t>((workload.k + 127) / 128 * 128);
    return {
        paddedK,
        acceleratedProducts ? 3U : genericProducts,
        tileM64 ? 64U : 128U,
        tileN64 ? 64U : 128U,
        mode.packMode,
        mode.resetMode,
        mode.combineMode,
        riskProfile,
        !mode.skipQueue,
        !mode.skipQueue,
        SelectChemmSplits(workload, conservativeSplits)};
}

struct ChemmCubeExecution {
    uint32_t cubeCores;
    ChemmTiling t;
    ChemmWorkloadClass workload;
    ChemmCubePolicy policy;
    uint32_t mp;
    uint32_t np;
    uint32_t splits;
    uint32_t outputCores;
    uint64_t capacity;
    uint8_t* w;
    uint8_t* right;
    uint8_t* out;
    uint8_t* bounded;
    uint8_t* queue;
    bool scalarBounded;
    bool compactUnitScalarNarrowOutput;
};

static aclblasStatus_t PrepareChemmCubeExecution(
    aclblasHandle_t h, const void* a, const void* b, void* c, ChemmTiling t, uint32_t cores,
    ChemmCubeExecution& execution)
{
    execution.workload = ClassifyChemmWorkload(t);
    execution.t = t;
    execution.policy = SelectChemmCubePolicy(
        t, execution.workload, (t.ai != 0 || std::abs(t.ar) > 1 || std::abs(t.br) > 1 || std::abs(t.bi) > 1) ? 8U : 4U,
        (t.left != 0U ? t.m : t.n) > 2048 ? 4U : 3U);
    execution.mp = static_cast<uint32_t>((t.m + 15) / 16 * 16);
    execution.np = static_cast<uint32_t>((t.n + 15) / 16 * 16);
    execution.splits = execution.policy.splits;
    execution.outputCores = reinterpret_cast<uintptr_t>(c) % 512 == 0 ? cores : 1;

    const uint64_t lsize = uint64_t(execution.np) * execution.policy.paddedK;
    const uint64_t rsize = uint64_t(execution.policy.paddedK) * execution.mp;
    const uint64_t outsize = uint64_t(execution.mp) * execution.np;
    const uint64_t bytes =
        (3 * lsize + 3 * rsize + execution.policy.products * execution.splits * outsize) * sizeof(float);
    const uint64_t total = uint64_t(t.ldc) * t.n;
    execution.capacity =
        total <= 0x3fffffff ?
            ((total + uint64_t(execution.outputCores) * 512 - 1) / (uint64_t(execution.outputCores) * 512)) * 512 :
            0;
    const uint64_t queueBytes =
        execution.capacity ? uint64_t(execution.outputCores) * (execution.capacity + 64) * sizeof(uint64_t) : 0;
    auto ret = EnsureDefaultWorkspace(h, bytes + queueBytes + 512);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;
    execution.cubeCores = GetAicCoreCount();
    if (!execution.cubeCores)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    execution.w = (uint8_t*)GetEffectiveWorkspace(h);
    execution.right = execution.w + 3 * lsize * sizeof(float);
    execution.out = execution.right + 3 * rsize * sizeof(float);
    execution.bounded = execution.w + bytes + queueBytes;
    execution.queue = execution.w + bytes;
    execution.scalarBounded = std::isfinite(t.ar) && std::isfinite(t.ai) && std::abs(t.ar) <= 1 &&
                              std::abs(t.ai) <= 1 && uint64_t(t.lda) * execution.workload.k <= 0x3fffffff &&
                              uint64_t(t.ldb) * t.n <= 0x3fffffff && uint64_t(t.ldc) * t.n <= 0x3fffffff;
    execution.compactUnitScalarNarrowOutput =
        execution.workload.family == ChemmWorkloadFamily::RightVeryNarrow && t.ldc == t.m &&
        execution.outputCores > 1 && t.ar == 1.f && __builtin_bit_cast(uint32_t, t.ai) == 0 &&
        __builtin_bit_cast(uint32_t, t.br) == 0 && __builtin_bit_cast(uint32_t, t.bi) == 0;
    return ACLBLAS_STATUS_SUCCESS;
}

static void ResetChemmCubeState(aclblasHandle_t h, const ChemmCubeExecution& execution, uint32_t cores)
{
    if (execution.compactUnitScalarNarrowOutput)
        return;
    if (execution.policy.resetMode == ChemmResetMode::Combined)
        chemm_combined_reset_do(
            execution.bounded, execution.queue, execution.capacity, execution.scalarBounded ? 1 : 0, cores, h->stream);
    else if (execution.policy.resetMode == ChemmResetMode::Bounded)
        chemm_bound_reset_do(execution.bounded, execution.scalarBounded ? 1 : 0, h->stream);
}

static void PackChemmCubeProducts(
    aclblasHandle_t h, const void* a, const void* b, const ChemmCubeExecution& execution, uint32_t cores)
{
    if (execution.compactUnitScalarNarrowOutput || execution.policy.packMode == ChemmPackMode::FastThreePlane)
        chemm_pack3_fast_do(
            a, b, execution.w, execution.t, execution.np, execution.mp, execution.policy.paddedK, cores, h->stream);
    else
        chemm_pack_do(
            a, b, execution.w, execution.t, execution.np, execution.mp, execution.policy.paddedK, cores, h->stream,
            execution.bounded);
}

static void CombineChemmCubeProducts(
    aclblasHandle_t h, const void* a, const void* b, void* c, const ChemmCubeExecution& execution)
{
    if (execution.compactUnitScalarNarrowOutput) {
        chemm_combine_narrow_output_do(
            execution.out, c, execution.t, execution.mp, execution.np, execution.outputCores, h->stream);
    } else if (execution.policy.combineMode == ChemmCombineMode::BoundedNoRisk) {
        chemm_combine_norisk_do(
            a, b, execution.out, c, execution.t, execution.mp, execution.np, execution.splits,
            execution.policy.products, execution.outputCores, h->stream, execution.queue,
            execution.policy.queueFallback ? execution.capacity : 0, execution.bounded,
            static_cast<uint32_t>(execution.policy.riskProfile));
    } else if (execution.policy.combineMode == ChemmCombineMode::RiskCheckedInline) {
        chemm_combine_do(
            a, b, execution.out, c, execution.t, execution.mp, execution.np, execution.splits,
            execution.policy.products, execution.outputCores, h->stream, nullptr, 0);
    } else {
        chemm_combine_do(
            a, b, execution.out, c, execution.t, execution.mp, execution.np, execution.splits,
            execution.policy.products, execution.outputCores, h->stream, execution.queue, execution.capacity);
    }
}

static aclblasStatus_t ChemmCubeLaunch(
    aclblasHandle_t h, const void* a, const void* b, void* c, ChemmTiling t, uint32_t cores)
{
    ChemmCubeExecution execution;
    auto ret = PrepareChemmCubeExecution(h, a, b, c, t, cores, execution);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;
    ChemmCubeGemmTilingData g{execution.np,
                              execution.mp,
                              execution.policy.paddedK,
                              execution.policy.paddedK,
                              execution.mp,
                              execution.mp,
                              execution.mp,
                              execution.policy.tileM,
                              execution.policy.tileN,
                              execution.policy.tileM,
                              execution.policy.tileN,
                              256,
                              execution.splits,
                              execution.policy.products};

    ResetChemmCubeState(h, execution, cores);
    PackChemmCubeProducts(h, a, b, execution, cores);
    chemm_cube_kernel_do(execution.w, execution.right, execution.out, g, execution.cubeCores, h->stream);
    if (execution.capacity && execution.policy.resetQueue)
        chemm_queue_reset_do(execution.queue, execution.capacity, execution.outputCores, h->stream);
    CombineChemmCubeProducts(h, a, b, c, execution);
    if (execution.capacity && execution.policy.queueFallback)
        chemm_queue_fallback_do(
            a, b, c, execution.queue, execution.capacity, execution.t, execution.outputCores, h->stream,
            execution.bounded, execution.w);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmMode(aclblasSideMode_t side, aclblasFillMode_t uplo)
{
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasChemm", "side must be LEFT or RIGHT, got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasChemm", "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmShape(int64_t m, int64_t n, bool& quickSuccess)
{
    if (m < 0) {
        OP_LOGE("aclblasChemm", "m must be >= 0, got %lld", static_cast<long long>(m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasChemm", "n must be >= 0, got %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    quickSuccess = m == 0 || n == 0;
    return ACLBLAS_STATUS_SUCCESS;
}

static bool ChemmExceedsMaxElements(int64_t leadDimension, int64_t extent, int64_t maxElements)
{
    int64_t elements = 0;
    if (__builtin_mul_overflow(leadDimension, extent, &elements))
        return true;
    return elements > maxElements;
}

static aclblasStatus_t ValidateChemmRequiredPointers(
    const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B, const aclblasComplex* beta,
    const aclblasComplex* C)
{
    if (alpha == nullptr) {
        OP_LOGE("aclblasChemm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasChemm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE("aclblasChemm", "A must not be nullptr when m and n are positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (B == nullptr) {
        OP_LOGE("aclblasChemm", "B must not be nullptr when m and n are positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (C == nullptr) {
        OP_LOGE("aclblasChemm", "C must not be nullptr when m and n are positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmLeadingDimensions(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
    const int64_t k = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (lda < k) {
        OP_LOGE(
            "aclblasChemm", "lda must be >= k, got lda=%lld, k=%lld", static_cast<long long>(lda),
            static_cast<long long>(k));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < m) {
        OP_LOGE(
            "aclblasChemm", "ldb must be >= m, got ldb=%lld, m=%lld", static_cast<long long>(ldb),
            static_cast<long long>(m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < m) {
        OP_LOGE(
            "aclblasChemm", "ldc must be >= m, got ldc=%lld, m=%lld", static_cast<long long>(ldc),
            static_cast<long long>(m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmMatrixSizes(
    aclblasSideMode_t side, int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
    constexpr int64_t maxElements = std::numeric_limits<int64_t>::max() / sizeof(aclblasComplex);
    const int64_t k = side == ACLBLAS_SIDE_LEFT ? m : n;
    if (ChemmExceedsMaxElements(lda, k, maxElements)) {
        OP_LOGE(
            "aclblasChemm", "A element count exceeds limit, lda=%lld, k=%lld", static_cast<long long>(lda),
            static_cast<long long>(k));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ChemmExceedsMaxElements(ldb, n, maxElements)) {
        OP_LOGE(
            "aclblasChemm", "B element count exceeds limit, ldb=%lld, n=%lld", static_cast<long long>(ldb),
            static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ChemmExceedsMaxElements(ldc, n, maxElements)) {
        OP_LOGE(
            "aclblasChemm", "C element count exceeds limit, ldc=%lld, n=%lld", static_cast<long long>(ldc),
            static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateChemmBuffers(
    aclblasSideMode_t side, int64_t m, int64_t n, const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda,
    const aclblasComplex* B, int64_t ldb, const aclblasComplex* beta, const aclblasComplex* C, int64_t ldc,
    bool& quickSuccess)
{
    auto status = ValidateChemmRequiredPointers(alpha, A, B, beta, C);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ValidateChemmLeadingDimensions(side, m, n, lda, ldb, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ValidateChemmMatrixSizes(side, m, n, lda, ldb, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    quickSuccess = alpha->real == 0 && alpha->imag == 0 && beta->real == 1 && beta->imag == 0;
    return ACLBLAS_STATUS_SUCCESS;
}

// Validate public inputs before workload classification and kernel dispatch.
static aclblasStatus_t ValidateChemmArguments(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, const aclblasComplex* C, int64_t ldc, bool& quickSuccess)
{
    if (!handle) {
        OP_LOGE("aclblasChemm", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    auto status = ValidateChemmMode(side, uplo);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = ValidateChemmShape(m, n, quickSuccess);
    if (status != ACLBLAS_STATUS_SUCCESS || quickSuccess)
        return status;
    return ValidateChemmBuffers(side, m, n, alpha, A, lda, B, ldb, beta, C, ldc, quickSuccess);
}

static bool ShouldUseChemmCube(const ChemmTiling& t, const ChemmWorkloadClass& workload, bool orderedScalar)
{
    const bool veryNarrowForceCube = workload.family == ChemmWorkloadFamily::RightVeryNarrow && t.ar == 1.f &&
                                     __builtin_bit_cast(uint32_t, t.ai) == 0 &&
                                     __builtin_bit_cast(uint32_t, t.br) == 0 && __builtin_bit_cast(uint32_t, t.bi) == 0;
    return !orderedScalar && workload.k <= 4096 && t.m >= 64 && (t.n >= 64 || veryNarrowForceCube) && t.m <= 16384 &&
           t.n <= 16384 && (t.ar != 0 || t.ai != 0) && (t.left || t.ai == 0);
}

static aclblasStatus_t RunOrderedChemmFallback(
    aclblasHandle_t handle, const void* A, const void* B, void* C, ChemmTiling t, uint32_t cores, bool orderedScalar)
{
    const int64_t k = t.left != 0U ? t.m : t.n;
    const uint32_t outputCores = reinterpret_cast<uintptr_t>(C) % 512 == 0 ? cores : 1;
    void* orderedWorkspace = nullptr;
    if ((t.ar != 0 || t.ai != 0) && (k > 2048 || orderedScalar) && k <= 4096 && t.m <= 16384 && t.n <= 16384 &&
        t.m * t.n >= 4096 && t.lda * k <= 0x3fffffff && t.ldb * t.n <= 0x3fffffff && t.ldc * t.n <= 0x3fffffff) {
        const uint64_t bytes = (uint64_t(k) * k + (t.left ? uint64_t(t.m) * t.n : 0)) * sizeof(aclblasComplex);
        auto ret = EnsureDefaultWorkspace(handle, bytes);
        if (ret != ACLBLAS_STATUS_SUCCESS)
            return ret;
        orderedWorkspace = GetEffectiveWorkspace(handle);
        chemm_ordered_pack_do(A, B, orderedWorkspace, t, cores, handle->stream);
    }
    chemm_simt_do(A, B, C, orderedWorkspace, t, outputCores, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasChemm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, int64_t lda, const aclblasComplex* B, int64_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, int64_t ldc)
{
    bool quickSuccess = false;
    auto status = ValidateChemmArguments(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, quickSuccess);
    if (status != ACLBLAS_STATUS_SUCCESS || quickSuccess)
        return status;
    const uint32_t cores = GetAivCoreCount();
    if (!cores)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    ChemmTiling t{
        m,
        n,
        lda,
        ldb,
        ldc,
        alpha->real,
        alpha->imag,
        beta->real,
        beta->imag,
        uint32_t(side == ACLBLAS_SIDE_LEFT),
        uint32_t(uplo == ACLBLAS_UPPER)};
    const ChemmWorkloadClass workload = ClassifyChemmWorkload(t);
    const bool orderedScalar = t.ai != 0 || std::abs(t.ar) > 1 || std::abs(t.br) > 1 || std::abs(t.bi) > 1;
    if (ShouldUseChemmCube(t, workload, orderedScalar))
        return ChemmCubeLaunch(handle, A, B, C, t, cores);
    return RunOrderedChemmFallback(handle, A, B, C, t, cores, orderedScalar);
}
