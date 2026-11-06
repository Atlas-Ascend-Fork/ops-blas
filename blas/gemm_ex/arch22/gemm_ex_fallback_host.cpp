/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include <algorithm>
#include <cstdint>
#include <limits>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_ex_host_helpers.h"

namespace {
constexpr int32_t MATMUL_TILE_M = 1024;
constexpr int32_t MATMUL_TILE_N = 1024;
constexpr int32_t MATMUL_TILE_K = 1024;
constexpr int32_t DIRECT_TRANSPOSE_EXPERIMENT_MAX = 1024;

constexpr uint64_t ALIGNMENT_ELEMENTS = 8; // 32 B for float GM buffers
// 2D-tiled candidate path: 20 packed tiles of 256x208 floats.
constexpr uint64_t TILES_WORKSPACE_FLOATS = 20 * 256 * 208;
constexpr int32_t WHOLE_GRID_COLS = 256;
constexpr int32_t WHOLE_GRID_ROWS = 256;
// The static Matmul baseK is 64. Keeping an even number of baseK iterations
// avoids the right-trans odd-tail pipeline penalty measured at K=4160.
constexpr int32_t WHOLE_GRID_REDUCTION_ALIGNMENT = 128;
constexpr int32_t WHOLE_GRID_MIN_DIM = 2048;
constexpr int32_t WHOLE_GRID_MAX_DIM = 8192;

uint32_t ClampBlocks(uint64_t tasks, uint32_t cores)
{
    if (tasks == 0 || cores == 0) {
        return 0;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(tasks, cores));
}

bool IsTranspose(aclblasOperation_t op) { return op == ACLBLAS_OP_T || op == ACLBLAS_OP_C; }

bool ValidSgemmLeadingDims(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc)
{
    const int minLda = IsTranspose(transa) ? std::max(1, k) : std::max(1, m);
    const int minLdb = IsTranspose(transb) ? std::max(1, n) : std::max(1, k);
    return lda >= minLda && ldb >= minLdb && ldc >= std::max(1, m);
}

bool ValidSgemmOperands(int k, float alpha, const float* a, const float* b, const float* c)
{
    if (k > 0 && alpha != 0.0f && (a == nullptr || b == nullptr)) {
        return false;
    }
    return c != nullptr;
}

uint64_t AlignElements(uint64_t elements)
{
    if (elements > std::numeric_limits<uint64_t>::max() - (ALIGNMENT_ELEMENTS - 1)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (elements + ALIGNMENT_ELEMENTS - 1) / ALIGNMENT_ELEMENTS * ALIGNMENT_ELEMENTS;
}

int32_t AlignDim(int32_t dim, int32_t multiple)
{
    if (multiple <= 0) {
        return dim;
    }
    const int64_t safe = multiple;
    const int64_t aligned = (static_cast<int64_t>(dim) + safe - 1) / safe * safe;
    if (aligned > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(aligned);
}

bool CanUseDirectTransposeMatmul(const GemmExTiling& tiling)
{
    // Matmul GM tensors have no leading-dimension parameter. The raw
    // column-major operand is direct only if compact. Limit this experiment
    // to one complete public problem; a host submatrix otherwise has a
    // different physical row stride than its logical K/M width.
    return tiling.m <= DIRECT_TRANSPOSE_EXPERIMENT_MAX && tiling.n <= DIRECT_TRANSPOSE_EXPERIMENT_MAX &&
           tiling.k <= DIRECT_TRANSPOSE_EXPERIMENT_MAX &&
           ((tiling.m % 16 == 0 && tiling.n % 16 == 0 && tiling.k % 8 == 0) ||
            (tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.ldc == tiling.m && tiling.m >= 16 &&
             tiling.n >= 16 && tiling.k >= 16)) &&
           tiling.lda == (tiling.transA ? tiling.k : tiling.m) && tiling.ldb == (tiling.transB ? tiling.n : tiling.k);
}

bool CanUseWholeProblemPack(const GemmExTiling& tiling)
{
    // Pack each public operand once, then run one whole-problem multi-Cube
    // launch. NN uses the DMA column path; transpose modes use the generic
    // address mapping in the pack kernel.
    return tiling.m <= DIRECT_TRANSPOSE_EXPERIMENT_MAX && tiling.n <= DIRECT_TRANSPOSE_EXPERIMENT_MAX &&
           tiling.k <= DIRECT_TRANSPOSE_EXPERIMENT_MAX;
}

bool CanUseRawTransposePack(const GemmExTiling& tiling)
{
    // Preserve each padded operand's raw physical orientation and let the
    // matching rows Matmul specialization interpret its transpose. Every raw
    // row is then one aligned DMA instead of scalar logical materialization.
    return (tiling.transA != 0 || tiling.transB != 0);
}

bool CanUseWholeGridSquare(const GemmExTiling& tiling)
{
    if (tiling.transA == 0 && tiling.m >= 448 && tiling.n >= 448 && std::max({tiling.m, tiling.n, tiling.k}) <= 1024) {
        return false;
    }
    const bool compact =
        tiling.lda == (tiling.transA ? tiling.k : tiling.m) && tiling.ldb == (tiling.transB ? tiling.n : tiling.k);
    const bool aligned = tiling.m % 16 == 0 && tiling.n % 16 == 0 && tiling.k % 32 == 0;
    return (compact || aligned) && tiling.m >= 16 && tiling.n >= 16 && tiling.k >= 16 &&
           std::max({tiling.m, tiling.n, tiling.k}) > 128 && tiling.m <= WHOLE_GRID_MAX_DIM &&
           tiling.n <= WHOLE_GRID_MAX_DIM && tiling.k <= WHOLE_GRID_MAX_DIM;
}
bool WholeGridInputsAreDirect(const GemmExTiling& tiling)
{
    return tiling.lda == (tiling.transA ? tiling.k : tiling.m) && tiling.ldb == (tiling.transB ? tiling.n : tiling.k) &&
           tiling.computeK == tiling.k && tiling.pm == tiling.m && tiling.pn == tiling.n;
}

bool WholeGridOutputIsDirect(const GemmExTiling& tiling)
{
    return tiling.splitParts <= 1 && tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.ldc == tiling.m &&
           !(tiling.k <= 256 && static_cast<uint64_t>(tiling.m) * tiling.n >= 65536 && tiling.m % 8 != 0);
}

bool TryWholeGridWorkspace(GemmExTiling* tiling, bool packInputs, bool product, size_t* bytes)
{
    uint64_t end = 0;
    const auto addAligned = [&end](uint64_t value) {
        const uint64_t aligned = AlignElements(value);
        if (aligned == std::numeric_limits<uint64_t>::max() || end > std::numeric_limits<uint64_t>::max() - aligned) {
            return false;
        }
        end += aligned;
        return true;
    };
    tiling->packAOffset = 0;
    tiling->packBOffset = 0;
    tiling->productOffset = 0;
    tiling->accumOffset = 0;
    if (packInputs) {
        const uint64_t packA = static_cast<uint64_t>(tiling->pm) * tiling->computeK;
        const uint64_t packB = static_cast<uint64_t>(tiling->pn) * tiling->computeK;
        tiling->packAOffset = end;
        if (!addAligned(packA)) {
            return false;
        }
        tiling->packBOffset = end;
        if (!addAligned(packB)) {
            return false;
        }
    }
    if (product) {
        tiling->productOffset = end;
        tiling->accumOffset = end;
        if (!addAligned(tiling->productPlane * std::max(1, tiling->splitParts))) {
            return false;
        }
    }
    if (end > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    *bytes = static_cast<size_t>(end * sizeof(float));
    return true;
}

void LaunchWholeGridSquare(
    _aclblas_handle* h, const GemmExTiling& tiling, const float* a, const float* b, float* c, uint8_t* workspace,
    bool packInputs, bool productOutput, uint32_t packBlocks, uint32_t epilogueBlocks, uint32_t cubeBlocks)
{
    auto* matmulA = reinterpret_cast<uint8_t*>(const_cast<float*>(a));
    auto* matmulB = reinterpret_cast<uint8_t*>(const_cast<float*>(b));
    if (packInputs) {
        gemmex_fallback_wholegrid_pack_do(packBlocks, h->stream, matmulA, matmulB, workspace, tiling);
        matmulA = workspace + tiling.packAOffset * sizeof(float);
        matmulB = workspace + tiling.packBOffset * sizeof(float);
    }
    auto* matmulOutput =
        productOutput ? workspace + tiling.productOffset * sizeof(float) : reinterpret_cast<uint8_t*>(c);
    gemmex_fallback_wholegrid_nosplit_do(cubeBlocks, h->stream, matmulA, matmulB, matmulOutput, tiling);
    if (productOutput) {
        gemmex_fallback_wholegrid_epilogue_do(
            epilogueBlocks, h->stream, workspace, reinterpret_cast<uint8_t*>(c), tiling);
    }
}

void ConfigureWholeGrid(GemmExTiling& tiling)
{
    // The installed right-transpose GM path becomes MTE2-bound on a partial
    // baseK. An odd count of complete baseK iterations is also slower. Pack
    // both reduction operands to the next pair of baseK iterations and fill
    // only the added reduction coordinates with zero.
    // Cube matmul consumes complete base-K groups. NN also needs a padded
    // reduction when K is not 32-aligned; the pack adapter clears the added
    // rows before launch.
    tiling.pm = AlignDim(tiling.m, 16);
    tiling.pn = AlignDim(tiling.n, 16);
    tiling.computeK = AlignDim(tiling.k, tiling.transA && tiling.k > 96 ? 128 : 32);
    const bool rawSmall = std::max(tiling.m, tiling.n) <= 256 && tiling.k <= 128 &&
                          tiling.lda == (tiling.transA ? tiling.k : tiling.m) &&
                          tiling.ldb == (tiling.transB ? tiling.n : tiling.k);
    if (rawSmall) {
        tiling.pm = tiling.m;
        tiling.pn = tiling.n;
        tiling.computeK = tiling.k;
    }
    tiling.splitParts = tiling.computeK >= 4096 && tiling.computeK % 128 == 0 ? 4 : 1;
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if (area < 32768 && tiling.k >= 512) {
        tiling.computeK = AlignDim(tiling.k, 512);
        tiling.splitParts = 4;
    }
}

struct WholeGridTile {
    int32_t columns;
    int32_t rows;
};
bool WideAlignedTile(const GemmExTiling& t, uint64_t area)
{
    return area >= 262144 && area < 4194304 && t.computeK <= 128 && t.transB == 0 && t.m % 16 == 0 && t.n % 16 == 0;
}
bool WideMediumTile(const GemmExTiling& t, uint64_t area)
{
    return area >= 65536 && area < 262144 && t.computeK > 128 && t.computeK <= 1024 && t.transA == 0 && t.transB != 0 &&
           t.m % 16 == 0 && t.n % 16 == 0;
}
WholeGridTile ChooseWholeGridTile(const GemmExTiling& tiling)
{
    const uint64_t area = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    const bool wideAlignedTn = WideAlignedTile(tiling, area);
    const bool wideMediumNt = WideMediumTile(tiling, area);
    const bool wideRectNn = wideAlignedTn && tiling.transA == 0 &&
                            ((tiling.pm + 383) / 384) * ((tiling.pn + 111) / 112) <= tiling.cubeBlocks;
    const int32_t tileSize =
        wideRectNn ? 384 : ((wideAlignedTn || wideMediumNt) ? 256 : (area < 65536 ? 64 : (area < 4194304 ? 128 : 256)));
    const int32_t rowTile = wideRectNn ? 112 : (wideMediumNt ? 64 : (tileSize == 128 && area < 262144 ? 64 : tileSize));
    return {tileSize, rowTile};
}

void ConfigureWholeGridProduct(
    GemmExTiling& tiling, bool productOutput, int32_t tileSize, int32_t rowTile, uint64_t blocks)
{
    tiling.productPlane = static_cast<uint64_t>(tiling.pm) * tiling.pn;
    if (productOutput && tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.splitParts == 1 && tileSize <= 128 &&
        tiling.k <= 256) {
        tiling.productTileM = tileSize;
        tiling.productTileN = rowTile;
        tiling.productPlane = blocks * tileSize * rowTile;
    }
}

aclblasStatus_t RunWholeGridSquare(_aclblas_handle* h, GemmExTiling tiling, const float* a, const float* b, float* c)
{
    ConfigureWholeGrid(tiling);
    const auto tile = ChooseWholeGridTile(tiling);
    const int32_t tileSize = tile.columns;
    const int32_t rowTile = tile.rows;
    const uint64_t blocks = tiling.splitParts * static_cast<uint64_t>((tiling.pm + tileSize - 1) / tileSize) *
                            static_cast<uint64_t>((tiling.pn + rowTile - 1) / rowTile);
    if (blocks == 0 || blocks > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_VALUE", __func__);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t aicCores = GetAicCoreCount();
    const uint32_t aivCores = GetAivCoreCount();
    if (aicCores == 0) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    // The whole-grid Cube kernel maps one block directly to one output tile
    // (it has no block-stride loop), so every tile must be launched.  The
    // runtime core count is still used by the row-partitioned kernels below.
    const uint32_t cubeBlocks = static_cast<uint32_t>(blocks);

    const bool packInputs = !WholeGridInputsAreDirect(tiling);
    const bool productOutput = !WholeGridOutputIsDirect(tiling);
    ConfigureWholeGridProduct(tiling, productOutput, tileSize, rowTile, blocks);
    if ((packInputs || productOutput) && aivCores == 0) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint64_t epilogueTasks = productOutput ? static_cast<uint64_t>(tiling.n) * ((tiling.m + 1023) / 1024) : 0;
    const uint32_t packBlocks = packInputs ? aivCores : 0;
    const uint32_t epilogueBlocks = ClampBlocks(epilogueTasks, aivCores);
    if ((packInputs && packBlocks == 0) || (productOutput && epilogueBlocks == 0) || cubeBlocks == 0) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    size_t workspaceBytes = 0;
    if (!TryWholeGridWorkspace(&tiling, packInputs, productOutput, &workspaceBytes)) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_VALUE", __func__);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (workspaceBytes != 0) {
        const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    auto* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    LaunchWholeGridSquare(
        h, tiling, a, b, c, workspace, packInputs, productOutput, packBlocks, epilogueBlocks, cubeBlocks);
    return ACLBLAS_STATUS_SUCCESS;
}

bool TryWorkspaceLayout(GemmExTiling* tiling, size_t* bytes, uint64_t minProductFloats = 0)
{
    const uint64_t packA = static_cast<uint64_t>(tiling->pm) * tiling->pk;
    const uint64_t packB = static_cast<uint64_t>(tiling->pn) * tiling->pk;
    uint64_t product = static_cast<uint64_t>(tiling->pm) * tiling->pn;
    if (product < minProductFloats) {
        product = minProductFloats;
    }
    const uint64_t aEnd = AlignElements(packA);
    const uint64_t bSize = AlignElements(packB);
    const uint64_t productSize = AlignElements(product);
    if (aEnd == std::numeric_limits<uint64_t>::max() || bSize == std::numeric_limits<uint64_t>::max() ||
        productSize == std::numeric_limits<uint64_t>::max() || aEnd > std::numeric_limits<uint64_t>::max() - bSize) {
        return false;
    }
    const uint64_t bEnd = aEnd + bSize;
    if (bEnd > std::numeric_limits<uint64_t>::max() - productSize) {
        return false;
    }
    const uint64_t productEnd = bEnd + productSize;
    if (productEnd > std::numeric_limits<uint64_t>::max() - productSize) {
        return false;
    }
    const uint64_t total = productEnd + productSize;
    if (total > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    tiling->packAOffset = 0;
    tiling->packBOffset = aEnd;
    tiling->productOffset = bEnd;
    tiling->accumOffset = productEnd;
    *bytes = static_cast<size_t>(total * sizeof(float));
    return true;
}

aclblasStatus_t ZeroLogicalC(_aclblas_handle* h, int m, int n, int ldc, float* c)
{
    GemmExTiling tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldc = ldc;
    const uint32_t aivCores = GetAivCoreCount();
    const uint32_t blocks = ClampBlocks(static_cast<uint64_t>(m) * n, aivCores);
    if (blocks == 0) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    gemmex_fallback_zero_do(blocks, h->stream, reinterpret_cast<uint8_t*>(c), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_HANDLE_IS_NULLPTR", __func__);
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) ||
        (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C)) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_ENUM", __func__);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_VALUE", __func__);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!ValidSgemmLeadingDims(transa, transb, m, n, k, lda, ldb, ldc)) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_VALUE", __func__);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ValidSgemmOperands(k, *alpha, a, b, c) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INVALID_VALUE;
}

// Selects the row-owner matmul kernel from the raw transpose combination of
// the public operands, so the dispatch does not re-express this table inline.
void RunRowsMatmul(
    void* stream, int32_t transA, int32_t transB, GM_ADDR a, GM_ADDR b, GM_ADDR output, const GemmExTiling& tiling)
{
    if (transB == 0) {
        if (transA == 0) {
            gemmex_fallback_matmul_rows_do(stream, a, b, output, tiling);
        } else {
            gemmex_fallback_matmul_rows_right_trans_do(stream, a, b, output, tiling);
        }
    } else if (transA == 0) {
        gemmex_fallback_matmul_rows_left_trans_do(stream, a, b, output, tiling);
    } else {
        gemmex_fallback_matmul_rows_both_trans_do(stream, a, b, output, tiling);
    }
}

// Runs one K contribution of a single host tile. The caller supplies the
// already-bounded (m/n/k) tiling descriptor; this fills the local pack contract
// and dispatches to the direct, raw-orientation, or generic packed path.
void RunDirectKContribution(
    _aclblas_handle* h, const GemmExTiling& tiling, const GemmExTiling& kTiling, bool directOutputToC, bool tiled2dNn,
    const float* a, const float* b, uint8_t* workspace, float* c)
{
    GemmExTiling matmulTiling = kTiling;
    if (directOutputToC) {
        matmulTiling.pm = tiling.m;
        matmulTiling.pn = tiling.n;
        matmulTiling.pk = tiling.k;
    }
    auto* matmulOutput = workspace;
    if (tiled2dNn) {
        gemmex_fallback_matmul_tiles_do(
            h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), workspace, matmulTiling);
    } else {
        if (directOutputToC) {
            matmulOutput = reinterpret_cast<uint8_t*>(c);
            matmulTiling.productOffset = 0;
        }
        RunRowsMatmul(
            h->stream, tiling.transA, tiling.transB, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), matmulOutput, matmulTiling);
    }
}

void RunKTileContribution(
    _aclblas_handle* h, GemmExTiling& tiling, const GemmExTiling& wsLayout, bool directWholeProblem,
    bool directOutputToC, bool tiled2dNn, bool oneKContribution, const float* a, const float* b, uint8_t* workspace,
    float* c)
{
    tiling.rawPack = !directWholeProblem && CanUseRawTransposePack(tiling) ? 1 : 0;
    GemmExTiling kTiling = tiling;
    if (!oneKContribution && tiling.firstK != 0) {
        kTiling.productOffset = wsLayout.accumOffset;
    }
    if (directWholeProblem) {
        RunDirectKContribution(h, tiling, kTiling, directOutputToC, tiled2dNn, a, b, workspace, c);
    } else if (tiling.rawPack != 0) {
        gemmex_fallback_pack_do(
            ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pk, GetAivCoreCount()), h->stream,
            reinterpret_cast<uint8_t*>(const_cast<float*>(a)), reinterpret_cast<uint8_t*>(const_cast<float*>(b)),
            workspace, tiling);
        auto* packedA = workspace + tiling.packAOffset * sizeof(float);
        auto* packedB = workspace + tiling.packBOffset * sizeof(float);
        RunRowsMatmul(h->stream, tiling.transA, tiling.transB, packedA, packedB, workspace, kTiling);
    } else {
        gemmex_fallback_pack_do(
            ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pk, GetAivCoreCount()), h->stream,
            reinterpret_cast<uint8_t*>(const_cast<float*>(a)), reinterpret_cast<uint8_t*>(const_cast<float*>(b)),
            workspace, tiling);
        GemmExTiling packedTiling = kTiling;
        packedTiling.transA = 0;
        packedTiling.transB = 0;
        RunRowsMatmul(
            h->stream, 0, 0, workspace + tiling.packAOffset * sizeof(float),
            workspace + tiling.packBOffset * sizeof(float), workspace, packedTiling);
    }
    if (!oneKContribution && tiling.firstK == 0) {
        gemmex_fallback_accumulate_do(
            ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace, tiling);
    }
}

// Runs the full K-loop for one (m,n) host tile and applies the epilogue.
void RunHostTile(
    _aclblas_handle* h, GemmExTiling& tiling, int32_t hostTileK, int64_t kTotal, const GemmExTiling& wsLayout,
    bool directWholeProblem, bool directOutputToC, bool tiled2dNn, bool oneKContribution, const float* a,
    const float* b, uint8_t* workspace, float* c)
{
    for (int64_t kBase = 0; kBase < kTotal; kBase += hostTileK) {
        tiling.kBase = static_cast<int32_t>(kBase);
        tiling.k = static_cast<int32_t>(std::min<int64_t>(hostTileK, kTotal - kBase));
        tiling.pk = AlignDim(tiling.k, 8);
        tiling.firstK = (kBase == 0) ? 1 : 0;
        RunKTileContribution(
            h, tiling, wsLayout, directWholeProblem, directOutputToC, tiled2dNn, oneKContribution, a, b, workspace, c);
    }
    if (tiled2dNn) {
        gemmex_fallback_scatter_tiles_do(
            ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace,
            reinterpret_cast<uint8_t*>(c), tiling);
    } else if (!directOutputToC) {
        gemmex_fallback_epilogue_do(
            ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace,
            reinterpret_cast<uint8_t*>(c), tiling);
    }
}

// Loop-invariant decisions that select the gemm execution path.
struct SgemmDispatchFlags {
    bool directWholeProblem;
    bool wholeProblemPack;
    bool wholeProblem;
    bool directOutputToC;
    bool tiled2dNn;
    bool oneKContribution;
    int32_t hostTileM;
    int32_t hostTileN;
    int32_t hostTileK;
};

SgemmDispatchFlags ComputeSgemmDispatch(const GemmExTiling& tiling, int m, int n, int k)
{
    SgemmDispatchFlags flags{};
    flags.directWholeProblem = CanUseDirectTransposeMatmul(tiling);
    flags.wholeProblemPack = !flags.directWholeProblem && CanUseWholeProblemPack(tiling);
    flags.wholeProblem = flags.directWholeProblem || flags.wholeProblemPack;
    // Row-only ownership underfills Cube when the internal row count (public
    // N) is small. Split the other output axis only for sufficiently deep,
    // strongly rectangular problems; square and shallow problems retain the
    // direct row kernel because tile scatter costs more than the extra cores
    // save there.
    flags.tiled2dNn = flags.directWholeProblem && tiling.transA == 0 && tiling.transB == 0 && n <= 128 && m >= 512 &&
                      k >= 512 && m % 16 == 0 && n % 16 == 0 && k % 8 == 0;
    flags.directOutputToC =
        flags.directWholeProblem && tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.ldc == tiling.m;
    flags.hostTileM = flags.wholeProblem ? m : MATMUL_TILE_M;
    flags.hostTileN = flags.wholeProblem ? n : MATMUL_TILE_N;
    flags.hostTileK = flags.wholeProblem ? k : MATMUL_TILE_K;
    flags.oneKContribution = k <= flags.hostTileK;
    return flags;
}

// Precompute the workspace layout, allocate the default workspace buffer, and
// return the effective workspace address used by the host-tile loop.
aclblasStatus_t PrepareSgemmWorkspace(
    _aclblas_handle* h, GemmExTiling& tiling, const SgemmDispatchFlags& flags, GemmExTiling& workspaceLayout,
    size_t& workspaceBytes, uint8_t*& workspace)
{
    workspace = nullptr;
    workspaceLayout = tiling;
    workspaceLayout.pm = AlignDim(flags.hostTileM, 16);
    workspaceLayout.pn = AlignDim(flags.hostTileN, 16);
    workspaceLayout.pk = AlignDim(flags.hostTileK, 8);
    if (!TryWorkspaceLayout(&workspaceLayout, &workspaceBytes, flags.tiled2dNn ? TILES_WORKSPACE_FLOATS : 0)) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INVALID_VALUE", __func__);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.packAOffset = workspaceLayout.packAOffset;
    tiling.packBOffset = workspaceLayout.packBOffset;
    tiling.productOffset = workspaceLayout.productOffset;
    // With one host K tile, product is already the complete internal C^T.
    // Point the epilogue at it and avoid a product-to-accumulator pass.
    tiling.accumOffset = flags.oneKContribution ? workspaceLayout.productOffset : workspaceLayout.accumOffset;
    const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    return workspace == nullptr ? ACLBLAS_STATUS_ALLOC_FAILED : ACLBLAS_STATUS_SUCCESS;
}
void RunHostTiles(
    _aclblas_handle* h, GemmExTiling tiling, const SgemmDispatchFlags& flags, const GemmExTiling& workspaceLayout,
    int m, int n, int k, const float* a, const float* b, uint8_t* workspace, float* c)
{
    for (int64_t nBase = 0; nBase < n; nBase += flags.hostTileN) {
        for (int64_t mBase = 0; mBase < m; mBase += flags.hostTileM) {
            tiling.nBase = static_cast<int32_t>(nBase);
            tiling.mBase = static_cast<int32_t>(mBase);
            tiling.n = static_cast<int32_t>(std::min<int64_t>(flags.hostTileN, static_cast<int64_t>(n) - nBase));
            tiling.m = static_cast<int32_t>(std::min<int64_t>(flags.hostTileM, static_cast<int64_t>(m) - mBase));
            tiling.pn = AlignDim(tiling.n, 16);
            tiling.pm = AlignDim(tiling.m, 16);
            RunHostTile(
                h, tiling, flags.hostTileK, k, workspaceLayout, flags.directWholeProblem, flags.directOutputToC,
                flags.tiled2dNn, flags.oneKContribution, a, b, workspace, c);
        }
    }
}
class PrivateRealGemmRun {
public:
    PrivateRealGemmRun(
        const RealBackendKernels& calls, _aclblas_handle* handle, const GemmExTiling& parameters, const float* left,
        const float* right, float* output)
        : kernels(calls), h(handle), tiling(parameters), a(left), b(right), c(output)
    {}

    aclblasStatus_t Run()
    {
#ifdef GEMMEX_CATLASS_ENABLED
        if (IsNativeCandidate()) {
            vectors = GetAivCoreCount();
            if (!vectors) {
                OP_LOGE(
                    "GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
            return RunNative();
        }
#endif
        if (CanUseWholeGridSquare(tiling)) {
            return RunWholeGridSquare(h, tiling, a, b, c);
        }
        return RunTiled();
    }

private:
    const RealBackendKernels& kernels;
    _aclblas_handle* h;
    GemmExTiling tiling;
    const float* a;
    const float* b;
    float* c;
    uint32_t vectors = 0;

#ifdef GEMMEX_CATLASS_ENABLED
    bool NativeShapeRange() const
    {
        return tiling.m >= 16 && tiling.n >= 16 && tiling.k >= 16 && tiling.k <= 2048 && tiling.m <= 2048 &&
               tiling.n <= 2048 && !(std::max(tiling.m, tiling.n) <= 256 && tiling.k <= 128);
    }
    bool IsNativeCandidate() const
    {
        return NativeShapeRange() && (tiling.m % 16 != 0 || tiling.n % 16 != 0 || tiling.k % 32 != 0) &&
               tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.ldc == tiling.m &&
               tiling.lda == (tiling.transA ? tiling.k : tiling.m) &&
               tiling.ldb == (tiling.transB ? tiling.n : tiling.k);
    }
    bool ModerateNn(uint32_t grid) const
    {
        return tiling.transA == 0 && tiling.transB == 0 && tiling.k > 32 && tiling.k <= 128 && tiling.m > 1536 &&
               tiling.n > 128 && static_cast<uint32_t>(tiling.n) <= vectors * 4 && grid <= tiling.cubeBlocks;
    }
    aclblasStatus_t RunNative()
    {
        if (tiling.m <= 64 && tiling.k <= 256 && tiling.n >= 512) {
            kernels.columnOwner(
                tiling.cubeBlocks, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<float*>(b)), reinterpret_cast<uint8_t*>(c), tiling);
            return ACLBLAS_STATUS_SUCCESS;
        }
        constexpr uint32_t tn = 368;
        const uint32_t residentGrid = static_cast<uint32_t>((tiling.n + 63) / 64) * ((tiling.m + tn - 1) / tn);
        if (tiling.transA == 0 && tiling.transB == 0 && tiling.k <= 32 && residentGrid <= tiling.cubeBlocks) {
            return RunResidentShort(residentGrid);
        }
        const uint32_t moderateGrid = static_cast<uint32_t>((tiling.n + 79) / 80) * ((tiling.m + 255) / 256);
        if (ModerateNn(moderateGrid)) {
            return RunModerate(moderateGrid);
        }
        if (tiling.transA == 0 && tiling.transB != 0 && tiling.m <= 128 && tiling.n <= 256 && tiling.k > 256) {
            return RunSplitNt();
        }
        return RunPacked();
    }

    aclblasStatus_t RunResidentShort(uint32_t residentGrid)
    {
        if (static_cast<uint32_t>((tiling.n + 15) / 16) <= vectors && AlignDim(tiling.m, 16) <= 1520) {
            auto status = EnsureDefaultWorkspace(h, static_cast<size_t>(residentGrid) * 64 * 368 * sizeof(float));
            if (status != ACLBLAS_STATUS_SUCCESS) {
                return status;
            }
            auto* product = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
            kernels.shortNzNative(
                residentGrid, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<float*>(b)), product, tiling, 0);
            kernels.nzBitcopy(vectors, h->stream, product, reinterpret_cast<uint8_t*>(c), tiling);
            return ACLBLAS_STATUS_SUCCESS;
        }
        const uint32_t outputLd = AlignDim(tiling.m, 128);
        auto status = EnsureDefaultWorkspace(h, static_cast<size_t>(outputLd) * tiling.n * sizeof(float));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        auto* product = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        kernels.shortNative(
            residentGrid, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), product, tiling, outputLd);
        kernels.catlassCopy(vectors, h->stream, product, reinterpret_cast<uint8_t*>(c), tiling, outputLd);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunModerate(uint32_t moderateGrid)
    {
        auto status = EnsureDefaultWorkspace(
            h, static_cast<size_t>(AlignDim(tiling.m, 16)) * AlignDim(tiling.n, 16) * sizeof(float));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        auto* product = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        kernels.moderateNz(
            moderateGrid, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), product, tiling);
        kernels.nzFourrowCopy(vectors, h->stream, product, reinterpret_cast<uint8_t*>(c), tiling);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunSplitNt()
    {
        constexpr uint32_t splits = 8;
        GemmExTiling padded = tiling;
        padded.lda = AlignDim(tiling.m, 128);
        padded.ldb = AlignDim(tiling.n, 128);
        padded.ldc = AlignDim(tiling.m, 128);
        const uint64_t ae = static_cast<uint64_t>(tiling.k) * padded.lda;
        const uint64_t be = static_cast<uint64_t>(tiling.k) * padded.ldb;
        const uint64_t ce = static_cast<uint64_t>(tiling.n) * padded.ldc * splits;
        auto status = EnsureDefaultWorkspace(h, static_cast<size_t>(ae + be + ce) * sizeof(float));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        auto* pa = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        auto* pb = pa + ae * sizeof(float);
        auto* partial = pb + be * sizeof(float);
        kernels.catlassPack(
            vectors, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), pa, pb, tiling, padded);
        kernels.splitNt(tiling.cubeBlocks, h->stream, pa, pb, partial, padded);
        kernels.splitSum(vectors, h->stream, partial, reinterpret_cast<uint8_t*>(c), tiling, padded.ldc, splits);
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t RunPacked()
    {
        GemmExTiling padded = tiling;
        padded.ldc = AlignDim(tiling.m, 128);
        padded.lda = AlignDim(tiling.transA ? tiling.k : tiling.m, 128);
        padded.ldb = AlignDim(tiling.transB ? tiling.n : tiling.k, 128);
        const uint64_t aElements = static_cast<uint64_t>(tiling.transA ? tiling.m : tiling.k) * padded.lda;
        const uint64_t bElements = static_cast<uint64_t>(tiling.transB ? tiling.k : tiling.n) * padded.ldb;
        const uint64_t cElements = std::max(
            static_cast<uint64_t>(padded.ldc) * tiling.n,
            static_cast<uint64_t>(AlignDim(tiling.m, 16)) * AlignDim(tiling.n, 16));
        auto status = EnsureDefaultWorkspace(h, static_cast<size_t>(aElements + bElements + cElements) * sizeof(float));
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
        auto* pa = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        auto* pb = pa + aElements * sizeof(float);
        auto* product = pb + bElements * sizeof(float);
        if (tiling.transA != 0 && tiling.transB != 0 && tiling.k <= 32 && AlignDim(tiling.m, 16) <= 1536) {
            kernels.dmaPack(
                vectors, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<float*>(b)), pa, pb, tiling, padded);
            kernels.ttShort(tiling.cubeBlocks, h->stream, pa, pb, product, padded);
            kernels.nzLoopCopy(vectors, h->stream, product, reinterpret_cast<uint8_t*>(c), tiling);
            return ACLBLAS_STATUS_SUCCESS;
        }
        kernels.catlassPack(
            vectors, h->stream, reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), pa, pb, tiling, padded);
        if (tiling.m <= 32 && tiling.k > 256) {
            padded.ldc = tiling.ldc;
            kernels.columnOwner(tiling.cubeBlocks, h->stream, pa, pb, reinterpret_cast<uint8_t*>(c), padded);
            return ACLBLAS_STATUS_SUCCESS;
        }
        kernels.catlass(tiling.cubeBlocks, h->stream, pa, pb, product, padded);
        kernels.catlassCopy(vectors, h->stream, product, reinterpret_cast<uint8_t*>(c), tiling, padded.ldc);
        return ACLBLAS_STATUS_SUCCESS;
    }

#endif
    aclblasStatus_t RunTiled()
    {
        const SgemmDispatchFlags flags = ComputeSgemmDispatch(tiling, tiling.m, tiling.n, tiling.k);
        const uint32_t aivCores = GetAivCoreCount();
        const bool needsVector = !flags.directOutputToC || flags.tiled2dNn || !flags.directWholeProblem;
        if (needsVector && aivCores == 0) {
            OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        GemmExTiling workspaceLayout = tiling;
        size_t workspaceBytes = 0;
        uint8_t* workspace = nullptr;
        auto status = PrepareSgemmWorkspace(h, tiling, flags, workspaceLayout, workspaceBytes, workspace);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }

        RunHostTiles(h, tiling, flags, workspaceLayout, tiling.m, tiling.n, tiling.k, a, b, workspace, c);
        return ACLBLAS_STATUS_SUCCESS;
    }
};

aclblasStatus_t ScalePlanarOutput(aclblasHandle_t handle, int m, int n, const float* beta, float* c, int ldc)
{
    if (*beta == 0.0f) {
        return ZeroLogicalC(handle, m, n, ldc, c);
    }
    for (int j = 0; j < n; ++j) {
        const auto status = aclblasSscal(handle, m, beta, c + static_cast<size_t>(j) * ldc, 1);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

aclblasStatus_t RunGemmExRealBackend(
    const RealBackendKernels& kernels, aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta,
    float* c, int ldc)
{
    aclblasStatus_t status = ValidateSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) {
        return status;
    }
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    if (k == 0 || *alpha == 0.0f) {
        return ScalePlanarOutput(handle, m, n, beta, c, ldc);
    }

    GemmExTiling tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = ldc;
    tiling.transA = IsTranspose(transa) ? 1 : 0;
    tiling.transB = IsTranspose(transb) ? 1 : 0;
    tiling.alpha = *alpha;
    tiling.beta = *beta;
    tiling.cubeBlocks = GetAicCoreCount();
    if (tiling.cubeBlocks == 0) {
        OP_LOGE("GemmExRealBackend", "GemmExRealBackend: %s failed with ACLBLAS_STATUS_INTERNAL_ERROR", __func__);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return PrivateRealGemmRun(kernels, h, tiling, a, b, c).Run();
}

extern "C" aclblasStatus_t GemmExPlanarBackend(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const float* alpha, const float* a, int lda, const float* b, int ldb, const float* beta, float* c, int ldc)
{
    static const RealBackendKernels kernels{
        gemmex_fallback_catlass_copy_do, gemmex_fallback_catlass_do,         gemmex_fallback_catlass_pack_do,
        gemmex_fallback_column_owner_do, gemmex_fallback_dma_pack_do,        gemmex_fallback_moderate_nz_do,
        gemmex_fallback_nz_bitcopy_do,   gemmex_fallback_nz_fourrow_copy_do, gemmex_fallback_nz_loop_copy_do,
        gemmex_fallback_short_native_do, gemmex_fallback_short_nz_native_do, gemmex_fallback_split_nt_do,
        gemmex_fallback_split_sum_do,    gemmex_fallback_tt_short_do};
    return RunGemmExRealBackend(kernels, handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
