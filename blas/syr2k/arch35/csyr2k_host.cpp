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
 * \file csyr2k_host.cpp
 * \brief CSYR2K Host implementation for ascend950 (DAV_3510).
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "csyr2k_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

namespace {

constexpr const char* OP_TAG = "aclblasCsyr2k";

struct Csyr2kWorkspace {
    uint8_t* ar = nullptr;
    uint8_t* ai = nullptr;
    uint8_t* br = nullptr;
    uint8_t* bi = nullptr;
    uint8_t* arHalf = nullptr;
    uint8_t* arLow = nullptr;
    uint8_t* adHalf = nullptr;
    uint8_t* adLow = nullptr;
    uint8_t* asHalf = nullptr;
    uint8_t* asLow = nullptr;
    uint8_t* brHalf = nullptr;
    uint8_t* brLow = nullptr;
    uint8_t* biHalf = nullptr;
    uint8_t* biLow = nullptr;
    uint8_t* bsHalf = nullptr;
    uint8_t* bsLow = nullptr;
    uint8_t* t1 = nullptr;
    uint8_t* t2 = nullptr;
    uint8_t* t3 = nullptr;
    uint8_t* t4 = nullptr;
    uint8_t* t5 = nullptr;
    uint8_t* t6 = nullptr;
    uint8_t* t7 = nullptr;
    uint8_t* t8 = nullptr;
    uint8_t* fastFlags = nullptr;
    uint8_t* fastAggregateFlag = nullptr;
    size_t matrixPlaneBytes = 0;
    size_t halfPlaneBytes = 0;
    size_t tempPlaneBytes = 0;
    size_t fastFlagBytes = 0;
    size_t totalBytes = 0;
};

bool CheckedMul(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedAlign(size_t value, size_t alignment, size_t& result)
{
    if (alignment == 0) {
        return false;
    }
    size_t withPadding = 0;
    if (!CheckedAdd(value, alignment - 1, withPadding)) {
        return false;
    }
    result = withPadding / alignment * alignment;
    return true;
}

uint8_t* OffsetPlane(uint8_t* base, uint64_t elementOffset, size_t elementSize)
{
    if (base == nullptr) {
        return nullptr;
    }
    return base + static_cast<size_t>(elementOffset) * elementSize;
}

uint32_t GetAivMatrixBlockCount(uint32_t rows, uint32_t cols, uint32_t coreCount)
{
    uint64_t rowTiles = CeilDiv<uint32_t>(rows, CSYR2K_ARCH35_AIV_BLOCK);
    uint64_t colTiles = CeilDiv<uint32_t>(cols, CSYR2K_ARCH35_AIV_BLOCK);
    uint64_t needed = std::max<uint64_t>(rowTiles * colTiles, 1);
    return static_cast<uint32_t>(std::min<uint64_t>(needed, coreCount));
}

uint32_t GetAivTriangleBlockCount(uint32_t n, uint32_t coreCount)
{
    uint64_t tiles = CeilDiv<uint32_t>(n, CSYR2K_ARCH35_AIV_BLOCK);
    uint64_t needed = std::max<uint64_t>(tiles * (tiles + 1) / 2, 1);
    return static_cast<uint32_t>(std::min<uint64_t>(needed, coreCount));
}

aclblasStatus_t ValidateCsyr2kParams(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const aclblasComplex* beta, aclblasComplex* c,
    int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
              OP_LOGE(OP_TAG, "trans must be OP_N(111), OP_T(112), or OP_C(113), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);

    int minLda = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    int minLdb = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(lda >= minLda, OP_LOGE(OP_TAG, "lda must be >= %d, got %d", minLda, lda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= minLdb, OP_LOGE(OP_TAG, "ldb must be >= %d, got %d", minLdb, ldb);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= %d, got %d", std::max(1, n), ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(c != nullptr, OP_LOGE(OP_TAG, "C must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(a != nullptr || k == 0, OP_LOGE(OP_TAG, "A must not be nullptr when k > 0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(b != nullptr || k == 0, OP_LOGE(OP_TAG, "B must not be nullptr when k > 0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

struct Csyr2kGemmPartition {
    uint32_t mBlocks = 1;
    uint32_t nBlocks = 1;
};

Csyr2kGemmPartition ChooseGemmPartition(uint32_t n, uint32_t usedAicCoreNum)
{
    uint32_t maxMBlocks = std::max<uint32_t>(CeilDiv<uint32_t>(n, CSYR2K_ARCH35_DEFAULT_TILE_M), 1);
    uint32_t maxNBlocks = std::max<uint32_t>(CeilDiv<uint32_t>(n, CSYR2K_ARCH35_DEFAULT_TILE_N), 1);
    Csyr2kGemmPartition best{};
    uint32_t bestUtilization = 1;
    uint32_t bestAspectGap = std::numeric_limits<uint32_t>::max();
    for (uint32_t mBlocks = 1; mBlocks <= std::min<uint32_t>(maxMBlocks, usedAicCoreNum); ++mBlocks) {
        uint32_t nBlocks = std::min<uint32_t>(maxNBlocks, usedAicCoreNum / mBlocks);
        if (nBlocks == 0) {
            continue;
        }
        uint32_t utilization = mBlocks * nBlocks;
        uint32_t aspectGap = (mBlocks > nBlocks) ? (mBlocks - nBlocks) : (nBlocks - mBlocks);
        if (utilization > bestUtilization || (utilization == bestUtilization && aspectGap < bestAspectGap)) {
            best = {mBlocks, nBlocks};
            bestUtilization = utilization;
            bestAspectGap = aspectGap;
        }
    }
    return best;
}

uint32_t GetGemmTileExtent(uint32_t n, uint32_t singleCore, uint32_t defaultTile, uint32_t alignment)
{
    uint32_t tile = std::min<uint32_t>(defaultTile, singleCore);
    return n < defaultTile ? std::min<uint32_t>(singleCore, CeilAlign<uint32_t>(n, alignment)) : tile;
}

uint32_t GetGemmKChunk(uint32_t tileM, uint32_t tileN)
{
    uint32_t l1Budget =
        CSYR2K_ARCH35_L1_SIZE_BYTES * CSYR2K_ARCH35_L1_USAGE_RATIO_NUM / CSYR2K_ARCH35_L1_USAGE_RATIO_DEN;
    uint32_t denominator =
        CSYR2K_ARCH35_L1_BUF_NUM * CSYR2K_ARCH35_FP32_SIZE *
        (CeilAlign<uint32_t>(tileM, CSYR2K_ARCH35_BASE_M) + CeilAlign<uint32_t>(tileN, CSYR2K_ARCH35_BASE_N));
    uint32_t maxAlignedK = (denominator > 0) ? (l1Budget / denominator) : 0;
    uint32_t maxKChunk = (maxAlignedK / CSYR2K_ARCH35_BASE_K) * CSYR2K_ARCH35_BASE_K;
    return std::min<uint32_t>(CSYR2K_ARCH35_DEFAULT_TILE_K_CHUNK, std::max<uint32_t>(maxKChunk, CSYR2K_ARCH35_BASE_K));
}

Csyr2kGemmTilingData MakeGemmTiling(
    uint32_t n, uint32_t k, uint32_t leftLd, uint32_t rightLd, uint32_t tempLdc, aclblasOperation_t trans,
    uint32_t usedAicCoreNum)
{
    Csyr2kGemmPartition partition = ChooseGemmPartition(n, usedAicCoreNum);
    Csyr2kGemmTilingData tiling{};
    tiling.n = n;
    tiling.k = k;
    tiling.leftLd = leftLd;
    tiling.rightLd = rightLd;
    tiling.isTransN = (trans == ACLBLAS_OP_N) ? 1 : 0;
    tiling.singleCoreM = CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, partition.mBlocks), CSYR2K_ARCH35_BASE_M);
    tiling.singleCoreN = CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, partition.nBlocks), CSYR2K_ARCH35_BASE_N);
    tiling.tileM = GetGemmTileExtent(n, tiling.singleCoreM, CSYR2K_ARCH35_DEFAULT_TILE_M, CSYR2K_ARCH35_BASE_M);
    tiling.tileN = GetGemmTileExtent(n, tiling.singleCoreN, CSYR2K_ARCH35_DEFAULT_TILE_N, CSYR2K_ARCH35_BASE_N);
    tiling.tileKChunk = GetGemmKChunk(tiling.tileM, tiling.tileN);
    tiling.tempRowStride = tempLdc;
    return tiling;
}

struct Csyr2kWorkspaceAreas {
    size_t matrixArea = 0;
    size_t halfArea = 0;
    size_t tempArea = 0;
    size_t partialArea = 0;
};

bool CalculateWorkspacePlaneSizes(
    uint32_t n, uint32_t matrixRows, uint32_t matrixCols, Csyr2kWorkspace& workspace, size_t& matrixElements)
{
    size_t matrixBytes = 0;
    size_t halfBytes = 0;
    size_t tempElements = 0;
    size_t tempBytes = 0;
    uint32_t tempLdc = CeilAlign<uint32_t>(n, CSYR2K_ARCH35_FIXPIPE_N_ALIGN);
    return CheckedMul(static_cast<size_t>(matrixRows), static_cast<size_t>(matrixCols), matrixElements) &&
           CheckedMul(matrixElements, sizeof(float), matrixBytes) &&
           CheckedAlign(matrixBytes, CSYR2K_ARCH35_GM_ALIGN, workspace.matrixPlaneBytes) &&
           CheckedMul(matrixElements, CSYR2K_ARCH35_FP16_SIZE, halfBytes) &&
           CheckedAlign(halfBytes, CSYR2K_ARCH35_GM_ALIGN, workspace.halfPlaneBytes) &&
           CheckedMul(static_cast<size_t>(tempLdc), static_cast<size_t>(n), tempElements) &&
           CheckedMul(tempElements, sizeof(float), tempBytes) &&
           CheckedAlign(tempBytes, CSYR2K_ARCH35_GM_ALIGN, workspace.tempPlaneBytes);
}

bool CalculateWorkspaceAreas(
    uint32_t k, bool enableExactFast, uint32_t fastFlagCount, Csyr2kWorkspace& workspace, Csyr2kWorkspaceAreas& areas)
{
    size_t fastFlagElements = 0;
    return CheckedMul(workspace.matrixPlaneBytes, 4, areas.matrixArea) &&
           CheckedMul(workspace.halfPlaneBytes, enableExactFast ? 12 : 0, areas.halfArea) &&
           CheckedMul(workspace.tempPlaneBytes, 4, areas.tempArea) &&
           CheckedMul(workspace.tempPlaneBytes, k > CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K ? 4 : 0, areas.partialArea) &&
           CheckedAdd(static_cast<size_t>(fastFlagCount), fastFlagCount == 0 ? 0U : 1U, fastFlagElements) &&
           CheckedMul(fastFlagElements, sizeof(uint32_t), workspace.fastFlagBytes) &&
           CheckedAlign(workspace.fastFlagBytes, CSYR2K_ARCH35_GM_ALIGN, workspace.fastFlagBytes);
}

bool CalculateWorkspaceSize(
    uint32_t n, uint32_t k, uint32_t matrixRows, uint32_t matrixCols, bool enableExactFast, uint32_t fastFlagCount,
    Csyr2kWorkspace& workspace, size_t& inputArea, size_t& total)
{
    size_t matrixElements = 0;
    Csyr2kWorkspaceAreas areas;
    if (!CalculateWorkspacePlaneSizes(n, matrixRows, matrixCols, workspace, matrixElements) ||
        !CalculateWorkspaceAreas(k, enableExactFast, fastFlagCount, workspace, areas)) {
        return false;
    }
    size_t dataArea = 0;
    inputArea = std::max(areas.matrixArea, areas.halfArea);
    return CheckedAdd(inputArea, areas.tempArea, dataArea) && CheckedAdd(dataArea, areas.partialArea, dataArea) &&
           CheckedAdd(dataArea, workspace.fastFlagBytes, total);
}

void AssignExactFastPlanes(Csyr2kWorkspace& workspace, uint8_t* base)
{
    workspace.arHalf = base;
    workspace.arLow = workspace.arHalf + workspace.halfPlaneBytes;
    workspace.adHalf = workspace.arLow + workspace.halfPlaneBytes;
    workspace.adLow = workspace.adHalf + workspace.halfPlaneBytes;
    workspace.asHalf = workspace.adLow + workspace.halfPlaneBytes;
    workspace.asLow = workspace.asHalf + workspace.halfPlaneBytes;
    workspace.brHalf = workspace.asLow + workspace.halfPlaneBytes;
    workspace.brLow = workspace.brHalf + workspace.halfPlaneBytes;
    workspace.biHalf = workspace.brLow + workspace.halfPlaneBytes;
    workspace.biLow = workspace.biHalf + workspace.halfPlaneBytes;
    workspace.bsHalf = workspace.biLow + workspace.halfPlaneBytes;
    workspace.bsLow = workspace.bsHalf + workspace.halfPlaneBytes;
}

void AssignWorkspacePlanes(
    Csyr2kWorkspace& workspace, uint8_t* base, size_t inputArea, uint32_t k, uint32_t fastFlagCount,
    bool enableExactFast)
{
    workspace.ar = base;
    workspace.ai = workspace.ar + workspace.matrixPlaneBytes;
    workspace.br = workspace.ai + workspace.matrixPlaneBytes;
    workspace.bi = workspace.br + workspace.matrixPlaneBytes;
    if (enableExactFast) {
        AssignExactFastPlanes(workspace, base);
    }
    workspace.t1 = base + inputArea;
    workspace.t2 = workspace.t1 + workspace.tempPlaneBytes;
    workspace.t3 = workspace.t2 + workspace.tempPlaneBytes;
    workspace.t4 = workspace.t3 + workspace.tempPlaneBytes;
    uint8_t* afterFinalTemps = workspace.t4 + workspace.tempPlaneBytes;
    if (k > CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K) {
        workspace.t5 = afterFinalTemps;
        workspace.t6 = workspace.t5 + workspace.tempPlaneBytes;
        workspace.t7 = workspace.t6 + workspace.tempPlaneBytes;
        workspace.t8 = workspace.t7 + workspace.tempPlaneBytes;
    }
    uint8_t* afterTemps = workspace.t8 == nullptr ? afterFinalTemps : workspace.t8 + workspace.tempPlaneBytes;
    workspace.fastFlags = fastFlagCount == 0 ? nullptr : afterTemps;
    workspace.fastAggregateFlag = workspace.fastFlags;
}

aclblasStatus_t PrepareWorkspace(
    _aclblas_handle* handle, uint32_t n, uint32_t k, uint32_t matrixRows, uint32_t matrixCols, bool enableExactFast,
    uint32_t fastFlagCount, Csyr2kWorkspace& workspace)
{
    size_t inputArea = 0;
    size_t total = 0;
    if (!CalculateWorkspaceSize(
            n, k, matrixRows, matrixCols, enableExactFast, fastFlagCount, workspace, inputArea, total)) {
        OP_LOGE(OP_TAG, "workspace size overflow: n=%u k=%u", n, k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    workspace.totalBytes = total;
    aclblasStatus_t status = EnsureDefaultWorkspace(handle, total);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%zu, ret=%d", total, status);
        return status;
    }

    AssignWorkspacePlanes(
        workspace, static_cast<uint8_t*>(GetEffectiveWorkspace(handle)), inputArea, k, fastFlagCount, enableExactFast);
    return ACLBLAS_STATUS_SUCCESS;
}

struct Csyr2kLaunchArgs {
    _aclblas_handle* handle;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasOperation_t normalizedTrans;
    uint32_t n;
    uint32_t k;
    const aclblasComplex* alpha;
    const aclblasComplex* a;
    uint32_t lda;
    const aclblasComplex* b;
    uint32_t ldb;
    const aclblasComplex* beta;
    aclblasComplex* c;
    uint32_t ldc;
};

struct Csyr2kLaunchState {
    Csyr2kWorkspace workspace{};
    uint32_t vectorCoreCount = 0;
    uint32_t usedAicCoreNum = 0;
    uint32_t matrixRows = 0;
    uint32_t matrixCols = 0;
    uint32_t deinterleaveBlocks = 0;
    uint32_t tempLdc = 0;
    uint32_t fastFlagCount = 0;
    uint32_t routeFlagCount = 0;
    uint32_t directTriangleTile = 0;
    bool enableExactFast = false;
    bool useMixEpilogue = false;
};

bool LaunchStrictNarrowIfNeeded(const Csyr2kLaunchArgs& args)
{
    if (args.n > CSYR2K_ARCH35_STRICT_NARROW_MAX_N || args.k < CSYR2K_ARCH35_STRICT_NARROW_MIN_K ||
        args.normalizedTrans != ACLBLAS_OP_T) {
        return false;
    }
    Csyr2kStrictNarrowTilingData tiling{args.n, args.k, args.lda, args.ldb, args.ldc, static_cast<uint8_t>(args.uplo)};
    OP_LOGI(OP_TAG, "launching ordered narrow reduction: n=%u k=%u", args.n, args.k);
    csyr2k_strict_narrow_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.b)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.alpha)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.beta)), reinterpret_cast<GM_ADDR>(args.c), tiling, 1,
        args.handle->stream);
    return true;
}

aclblasStatus_t LaunchExactFastPrepare(const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state)
{
    if (!state.enableExactFast) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError memsetRet = aclrtMemsetAsync(
        state.workspace.fastFlags, state.workspace.fastFlagBytes, 0, (state.fastFlagCount + 1U) * sizeof(uint32_t),
        args.handle->stream);
    if (memsetRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "fast-route memset failed, ret=%d", memsetRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    Csyr2kFastPrepareTilingData tiling{state.matrixRows, state.matrixCols, args.lda, args.ldb};
    csyr2k_half_prepare_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.b)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.alpha)), state.workspace.arHalf,
        state.workspace.arLow, state.workspace.adHalf, state.workspace.adLow, state.workspace.asHalf,
        state.workspace.asLow, state.workspace.brHalf, state.workspace.brLow, state.workspace.biHalf,
        state.workspace.biLow, state.workspace.bsHalf, state.workspace.bsLow, state.workspace.fastFlags, tiling,
        state.deinterleaveBlocks, args.handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

void LaunchDeinterleave(const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state)
{
    if (state.useMixEpilogue) {
        return;
    }
    Csyr2kDeinterleaveTilingData tiling{state.matrixRows, state.matrixCols,    args.lda,
                                        args.ldb,         state.fastFlagCount, state.deinterleaveBlocks};
    OP_LOGI(OP_TAG, "launching SIMD deinterleave kernel: blocks=%u", state.deinterleaveBlocks);
    csyr2k_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.b)), state.workspace.ar, state.workspace.ai,
        state.workspace.br, state.workspace.bi, state.workspace.fastFlags, tiling, state.deinterleaveBlocks,
        args.handle->stream);
}

aclblasStatus_t PrepareCsyr2kLaunch(const Csyr2kLaunchArgs& args, Csyr2kLaunchState& state)
{
    state.vectorCoreCount = GetAivCoreCount();
    if (state.vectorCoreCount == 0) {
        OP_LOGE(OP_TAG, "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    state.tempLdc = CeilAlign<uint32_t>(args.n, CSYR2K_ARCH35_FIXPIPE_N_ALIGN);
    if (args.k == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    state.usedAicCoreNum = GetUsedAicCoreNum(args.n, CSYR2K_ARCH35_BASE_M, OP_TAG);
    if (state.usedAicCoreNum == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    state.matrixRows = args.trans == ACLBLAS_OP_N ? args.n : args.k;
    state.matrixCols = args.trans == ACLBLAS_OP_N ? args.k : args.n;
    state.deinterleaveBlocks = GetAivMatrixBlockCount(state.matrixRows, state.matrixCols, state.vectorCoreCount);
    state.enableExactFast =
        ((args.n >= CSYR2K_ARCH35_EXACT_FAST_MIN_N && args.k >= CSYR2K_ARCH35_EXACT_FAST_MIN_K) ||
         (args.n >= CSYR2K_ARCH35_EXACT_FAST_LARGE_N && args.k >= CSYR2K_ARCH35_EXACT_FAST_LARGE_N_MIN_K)) &&
                            args.k <= CSYR2K_ARCH35_EXACT_FAST_MAX_K;
    state.fastFlagCount = state.enableExactFast ? 1U : 0U;
    state.routeFlagCount = state.fastFlagCount;
    state.directTriangleTile = std::min<uint32_t>(args.n, CSYR2K_ARCH35_DIRECT_TRIANGLE_TILE);
    state.useMixEpilogue =
        state.enableExactFast && args.n == CSYR2K_ARCH35_MIX_Q8_MATRIX_SIZE &&
        args.k == CSYR2K_ARCH35_MIX_Q8_MATRIX_SIZE && args.uplo == ACLBLAS_UPPER &&
        args.normalizedTrans == ACLBLAS_OP_N && state.directTriangleTile == CSYR2K_ARCH35_MIX_Q8_MACRO_TILE &&
        state.usedAicCoreNum == CSYR2K_ARCH35_MIX_Q8_AIC_CORE_COUNT &&
        state.deinterleaveBlocks == CSYR2K_ARCH35_MIX_Q8_DEINTERLEAVE_BLOCK_COUNT;
    aclblasStatus_t status = PrepareWorkspace(
        args.handle, args.n, args.k, state.matrixRows, state.matrixCols, state.enableExactFast, state.fastFlagCount,
        state.workspace);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = LaunchExactFastPrepare(args, state);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        LaunchDeinterleave(args, state);
    }
    return status;
}

Csyr2kGemmTilingData MakeLaunchGemmTiling(const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state)
{
    Csyr2kGemmTilingData tiling = MakeGemmTiling(
        args.n, args.k, state.matrixRows, state.matrixRows, state.tempLdc, args.normalizedTrans, state.usedAicCoreNum);
    tiling.fastFlagCount = state.routeFlagCount;
    tiling.uploMode = static_cast<uint8_t>(args.uplo);
    tiling.directTriangleTile = state.directTriangleTile;
    tiling.isResidual4M = static_cast<uint8_t>(args.k > CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K ? 1 : 0);
    return tiling;
}

void LaunchMixEpilogue(
    const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state, const Csyr2kGemmTilingData& gemmTiling)
{
    const Csyr2kWorkspace& ws = state.workspace;
    Csyr2kDeinterleaveTilingData deinterleaveTiling{state.matrixRows, state.matrixCols,    args.lda,
                                                    args.ldb,         state.fastFlagCount, state.deinterleaveBlocks};
    Csyr2kCombineTilingData combineTiling{};
    combineTiling.n = args.n;
    combineTiling.ldc = args.ldc;
    combineTiling.tempLdc = state.tempLdc;
    combineTiling.fastFlagCount = state.routeFlagCount;
    combineTiling.directTriangleTile = state.directTriangleTile;
    combineTiling.uploMode = static_cast<uint8_t>(args.uplo);
    combineTiling.mixOffdiagEnabled = 1;
    csyr2k_gemm_mix_epilogue_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.a)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.b)), ws.ar, ws.ai, ws.br, ws.bi, ws.arHalf, ws.arLow,
        ws.adHalf, ws.adLow, ws.asHalf, ws.asLow, ws.brHalf, ws.brLow, ws.biHalf, ws.biLow, ws.bsHalf, ws.bsLow, ws.t1,
        ws.t2, ws.t3, ws.t4, ws.fastAggregateFlag, reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.alpha)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.beta)), reinterpret_cast<GM_ADDR>(args.c),
        gemmTiling, deinterleaveTiling, combineTiling, state.usedAicCoreNum, args.handle->stream);
}

void LaunchGemmDispatch(
    const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state, const Csyr2kGemmTilingData& tiling,
    uint64_t leftOffset, uint64_t rightOffset, uint8_t* out1, uint8_t* out2, uint8_t* out3, uint8_t* out4)
{
    const Csyr2kWorkspace& ws = state.workspace;
    csyr2k_gemm_dispatch_kernel_do(
        OffsetPlane(ws.ar, leftOffset, sizeof(float)), OffsetPlane(ws.ai, leftOffset, sizeof(float)),
        OffsetPlane(ws.br, rightOffset, sizeof(float)), OffsetPlane(ws.bi, rightOffset, sizeof(float)),
        OffsetPlane(ws.arHalf, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.arLow, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.adHalf, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.adLow, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.asHalf, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.asLow, leftOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.brHalf, rightOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.brLow, rightOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.biHalf, rightOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.biLow, rightOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.bsHalf, rightOffset, CSYR2K_ARCH35_FP16_SIZE),
        OffsetPlane(ws.bsLow, rightOffset, CSYR2K_ARCH35_FP16_SIZE), out1, out2, out3, out4, ws.fastAggregateFlag,
        tiling, state.usedAicCoreNum, args.handle->stream);
}

void LaunchSegmentedGemm(
    const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state, const Csyr2kGemmTilingData& gemmTiling)
{
    const Csyr2kWorkspace& ws = state.workspace;
    uint32_t segmentK = (args.k / CSYR2K_ARCH35_HIGH_K_PART_COUNT / CSYR2K_ARCH35_BASE_K) * CSYR2K_ARCH35_BASE_K;
    Csyr2kPartialAccumulateTilingData partialTiling{static_cast<uint64_t>(args.n) * state.tempLdc};
    for (uint32_t segment = 0; segment < CSYR2K_ARCH35_HIGH_K_PART_COUNT; ++segment) {
        uint32_t segmentStart = segment * segmentK;
        Csyr2kGemmTilingData tiling = gemmTiling;
        tiling.k = segment + 1 == CSYR2K_ARCH35_HIGH_K_PART_COUNT ? args.k - segmentStart : segmentK;
        tiling.isResidual4M = 1;
        uint64_t leftOffset = tiling.isTransN != 0 ? static_cast<uint64_t>(segmentStart) * tiling.leftLd : segmentStart;
        uint64_t rightOffset =
            tiling.isTransN != 0 ? static_cast<uint64_t>(segmentStart) * tiling.rightLd : segmentStart;
        uint8_t* out1 = segment == 0 ? ws.t1 : ws.t5;
        uint8_t* out2 = segment == 0 ? ws.t2 : ws.t6;
        uint8_t* out3 = segment == 0 ? ws.t3 : ws.t7;
        uint8_t* out4 = segment == 0 ? ws.t4 : ws.t8;
        LaunchGemmDispatch(args, state, tiling, leftOffset, rightOffset, out1, out2, out3, out4);
        if (segment != 0) {
            csyr2k_partial_accumulate_kernel_do(
                ws.t1, ws.t2, ws.t3, ws.t4, ws.t5, ws.t6, ws.t7, ws.t8, partialTiling, state.vectorCoreCount,
                args.handle->stream);
        }
    }
}

void LaunchGemm(const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state)
{
    Csyr2kGemmTilingData tiling = MakeLaunchGemmTiling(args, state);
    OP_LOGD(
        OP_TAG,
        "gemm tiling: n=%u k=%u matrixLd=%u tempLdc=%u cores=%u transN=%u "
        "singleCoreM=%u singleCoreN=%u tileM=%u tileN=%u tileKChunk=%u",
        args.n, args.k, state.matrixRows, state.tempLdc, state.usedAicCoreNum, tiling.isTransN, tiling.singleCoreM,
        tiling.singleCoreN, tiling.tileM, tiling.tileN, tiling.tileKChunk);
    if (state.useMixEpilogue) {
        LaunchMixEpilogue(args, state, tiling);
    } else if (args.k <= CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K) {
        const Csyr2kWorkspace& ws = state.workspace;
        LaunchGemmDispatch(args, state, tiling, 0, 0, ws.t1, ws.t2, ws.t3, ws.t4);
    } else {
        LaunchSegmentedGemm(args, state, tiling);
    }
}

void LaunchCombine(const Csyr2kLaunchArgs& args, const Csyr2kLaunchState& state)
{
    Csyr2kCombineTilingData tiling{};
    tiling.n = args.n;
    tiling.ldc = args.ldc;
    tiling.tempLdc = state.tempLdc;
    tiling.fastFlagCount = state.routeFlagCount;
    tiling.directTriangleTile = state.directTriangleTile;
    tiling.uploMode = static_cast<uint8_t>(args.uplo);
    tiling.skipTemp = static_cast<uint8_t>(args.k == 0 ? 1 : 0);
    tiling.isResidual4M =
        static_cast<uint8_t>(state.routeFlagCount != 0 && args.k > CSYR2K_ARCH35_RESIDUAL_3TERM_MAX_K);
    tiling.mixOffdiagEnabled = static_cast<uint8_t>(state.useMixEpilogue ? 1 : 0);
    uint32_t blocks = GetAivTriangleBlockCount(args.n, state.vectorCoreCount);
    OP_LOGI(
        OP_TAG, "launching SIMD combine kernel: blocks=%u kZero=%d workspace=%zu", blocks, args.k == 0,
        state.workspace.totalBytes);
    csyr2k_combine_kernel_do(
        state.workspace.t1, state.workspace.t2, state.workspace.t3, state.workspace.t4,
        state.workspace.fastAggregateFlag, reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.alpha)),
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(args.beta)), reinterpret_cast<GM_ADDR>(args.c), tiling,
        blocks, args.handle->stream);
}

aclblasStatus_t LaunchCsyr2kKernel(
    _aclblas_handle* handle, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* a, uint32_t lda, const aclblasComplex* b, uint32_t ldb,
    const aclblasComplex* beta, aclblasComplex* c, uint32_t ldc)
{
    bool skipTemp = (k == 0);
    aclblasOperation_t normalizedTrans = (trans == ACLBLAS_OP_C) ? ACLBLAS_OP_T : trans;
    Csyr2kLaunchArgs args{handle, uplo, trans, normalizedTrans, n, k, alpha, a, lda, b, ldb, beta, c, ldc};
    if (LaunchStrictNarrowIfNeeded(args)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    Csyr2kLaunchState state{};
    aclblasStatus_t status = PrepareCsyr2kLaunch(args, state);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (!skipTemp) {
        LaunchGemm(args, state);
    }
    LaunchCombine(args, state);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCsyr2k(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const aclblasComplex* beta, aclblasComplex* c,
    int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t status = ValidateCsyr2kParams(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    return LaunchCsyr2kKernel(
        static_cast<_aclblas_handle*>(handle), uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha,
        a, static_cast<uint32_t>(lda), b, static_cast<uint32_t>(ldb), beta, c, static_cast<uint32_t>(ldc));
}
