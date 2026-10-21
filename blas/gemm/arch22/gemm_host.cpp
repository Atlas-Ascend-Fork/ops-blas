/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include <limits>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_kernel.h"

namespace {
constexpr int32_t kMatmulTileM = 1024;
constexpr int32_t kMatmulTileN = 1024;
constexpr int32_t kMatmulTileK = 1024;
constexpr int32_t kDirectTransposeExperimentMax = 1024;

constexpr uint64_t kAlignmentElements = 8; // 32 B for float GM buffers
// 2D-tiled candidate path: 20 packed tiles of 256x208 floats.
constexpr uint64_t kTilesWorkspaceFloats = 20 * 256 * 208;
constexpr int32_t kWholeGridCols = 256;
constexpr int32_t kWholeGridRows = 256;
// The static Matmul baseK is 64. Keeping an even number of baseK iterations
// avoids the right-trans odd-tail pipeline penalty measured at K=4160.
constexpr int32_t kWholeGridReductionAlignment = 128;
constexpr int32_t kWholeGridMinDim = 2048;
constexpr int32_t kWholeGridMaxDim = 8192;

uint32_t ClampBlocks(uint64_t tasks, uint32_t cores)
{
    if (tasks == 0 || cores == 0) return 0;
    return static_cast<uint32_t>(std::min<uint64_t>(tasks, cores));
}

bool IsTranspose(aclblasOperation_t op)
{
    return op == ACLBLAS_OP_T || op == ACLBLAS_OP_C;
}

bool ValidSgemmLeadingDims(aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
                           int lda, int ldb, int ldc)
{
    const int minLda = IsTranspose(transa) ? std::max(1, k) : std::max(1, m);
    const int minLdb = IsTranspose(transb) ? std::max(1, n) : std::max(1, k);
    return lda >= minLda && ldb >= minLdb && ldc >= std::max(1, m);
}

bool ValidSgemmOperands(int k, float alpha, const float* a, const float* b, const float* c)
{
    if (k > 0 && alpha != 0.0f && (a == nullptr || b == nullptr))
        return false;
    return c != nullptr;
}

uint64_t AlignElements(uint64_t elements)
{
    if (elements > std::numeric_limits<uint64_t>::max() - (kAlignmentElements - 1)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (elements + kAlignmentElements - 1) / kAlignmentElements * kAlignmentElements;
}

int32_t AlignDim(int32_t dim, int32_t multiple)
{
    if (multiple <= 0) return dim;
    const int64_t safe = multiple;
    const int64_t aligned = (static_cast<int64_t>(dim) + safe - 1) / safe * safe;
    if (aligned > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(aligned);
}

bool CanUseDirectTransposeMatmul(const GemmArch22Tiling& tiling)
{
    // Matmul GM tensors have no leading-dimension parameter. The raw
    // column-major operand is direct only if compact. Limit this experiment
    // to one complete public problem; a host submatrix otherwise has a
    // different physical row stride than its logical K/M width.
    return tiling.m <= kDirectTransposeExperimentMax && tiling.n <= kDirectTransposeExperimentMax &&
           tiling.k <= kDirectTransposeExperimentMax &&
           tiling.m % 16 == 0 && tiling.n % 16 == 0 && tiling.k % 8 == 0 &&
           tiling.lda == (tiling.transA ? tiling.k : tiling.m) &&
           tiling.ldb == (tiling.transB ? tiling.n : tiling.k);
}

bool CanUseWholeProblemPack(const GemmArch22Tiling& tiling)
{
    // Pack each public operand once, then run one whole-problem multi-Cube
    // launch. NN uses the DMA column path; transpose modes use the generic
    // address mapping in the pack kernel.
    return tiling.m <= kDirectTransposeExperimentMax &&
           tiling.n <= kDirectTransposeExperimentMax &&
           tiling.k <= kDirectTransposeExperimentMax;
}

bool CanUseRawTransposePack(const GemmArch22Tiling& tiling)
{
    // Preserve each padded operand's raw physical orientation and let the
    // matching rows Matmul specialization interpret its transpose. Every raw
    // row is then one aligned DMA instead of scalar logical materialization.
    return (tiling.transA != 0 || tiling.transB != 0) &&
           tiling.m % 16 == 0 && tiling.n % 16 == 0 && tiling.k % 8 == 0;
}

bool CanUseWholeGridSquare(const GemmArch22Tiling& tiling)
{
    // Layout, transpose and alpha/beta are adapter concerns. The square AIC
    // backend requires only the proven full-K shape/alignment envelope.
    return tiling.m >= kWholeGridMinDim && tiling.n >= kWholeGridMinDim &&
           tiling.k >= kWholeGridMinDim &&
           tiling.m <= kWholeGridMaxDim && tiling.n <= kWholeGridMaxDim &&
           tiling.k <= kWholeGridMaxDim &&
           tiling.m % 16 == 0 && tiling.n % 16 == 0 && tiling.k % 32 == 0;
}

bool WholeGridInputsAreDirect(const GemmArch22Tiling& tiling)
{
    return tiling.transA == 0 && tiling.transB == 0 &&
           tiling.lda == tiling.m && tiling.ldb == tiling.k &&
           tiling.computeK == tiling.k;
}

bool WholeGridOutputIsDirect(const GemmArch22Tiling& tiling)
{
    return tiling.alpha == 1.0f && tiling.beta == 0.0f && tiling.ldc == tiling.m;
}

bool TryWholeGridWorkspace(GemmArch22Tiling* tiling, bool packInputs, bool product,
                           size_t* bytes)
{
    uint64_t end = 0;
    const auto addAligned = [&end](uint64_t value) {
        const uint64_t aligned = AlignElements(value);
        if (aligned == std::numeric_limits<uint64_t>::max() ||
            end > std::numeric_limits<uint64_t>::max() - aligned) return false;
        end += aligned;
        return true;
    };
    tiling->packAOffset = 0;
    tiling->packBOffset = 0;
    tiling->productOffset = 0;
    tiling->accumOffset = 0;
    if (packInputs) {
        const uint64_t packA = static_cast<uint64_t>(tiling->m) * tiling->computeK;
        const uint64_t packB = static_cast<uint64_t>(tiling->n) * tiling->computeK;
        tiling->packAOffset = end;
        if (!addAligned(packA)) return false;
        tiling->packBOffset = end;
        if (!addAligned(packB)) return false;
    }
    if (product) {
        tiling->productOffset = end;
        tiling->accumOffset = end;
        if (!addAligned(static_cast<uint64_t>(tiling->m) * tiling->n)) return false;
    }
    if (end > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    *bytes = static_cast<size_t>(end * sizeof(float));
    return true;
}

void LaunchWholeGridSquare(_aclblas_handle* h, const GemmArch22Tiling& tiling,
                           const float* a, const float* b, float* c,
                           uint8_t* workspace, bool packInputs, bool productOutput,
                           uint32_t packBlocks, uint32_t epilogueBlocks, uint32_t cubeBlocks)
{
    auto* matmulA = reinterpret_cast<uint8_t*>(const_cast<float*>(a));
    auto* matmulB = reinterpret_cast<uint8_t*>(const_cast<float*>(b));
    if (packInputs) {
        sgemm_arch22_wholegrid_pack_do(packBlocks, h->stream, matmulA, matmulB,
                                      workspace, tiling);
        matmulA = workspace + tiling.packAOffset * sizeof(float);
        matmulB = workspace + tiling.packBOffset * sizeof(float);
    }
    auto* matmulOutput = productOutput
        ? workspace + tiling.productOffset * sizeof(float)
        : reinterpret_cast<uint8_t*>(c);
    sgemm_arch22_wholegrid_nosplit_do(cubeBlocks, h->stream,
                                     matmulA, matmulB, matmulOutput, tiling);
    if (productOutput) {
        sgemm_arch22_wholegrid_epilogue_do(epilogueBlocks, h->stream, workspace,
                                          reinterpret_cast<uint8_t*>(c), tiling);
    }
}

aclblasStatus_t RunWholeGridSquare(_aclblas_handle* h, GemmArch22Tiling tiling,
                                   const float* a, const float* b, float* c)
{
    // The installed right-transpose GM path becomes MTE2-bound on a partial
    // baseK. An odd count of complete baseK iterations is also slower. Pack
    // both reduction operands to the next pair of baseK iterations and fill
    // only the added reduction coordinates with zero.
    // Cube matmul consumes complete base-K groups. NN also needs a padded
    // reduction when K is not 32-aligned; the pack adapter clears the added
    // rows before launch.
    tiling.computeK = AlignDim(tiling.k, tiling.transA != 0 ? kWholeGridReductionAlignment : 32);
    const uint64_t blocks = static_cast<uint64_t>((tiling.m + kWholeGridCols - 1) / kWholeGridCols) *
                            static_cast<uint64_t>((tiling.n + kWholeGridRows - 1) / kWholeGridRows);
    if (blocks == 0 || blocks > std::numeric_limits<uint32_t>::max()) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t aicCores = GetAicCoreCount();
    const uint32_t aivCores = GetAivCoreCount();
    if (aicCores == 0) return ACLBLAS_STATUS_INTERNAL_ERROR;
    // The whole-grid Cube kernel maps one block directly to one output tile
    // (it has no block-stride loop), so every tile must be launched.  The
    // runtime core count is still used by the row-partitioned kernels below.
    const uint32_t cubeBlocks = static_cast<uint32_t>(blocks);

    const bool packInputs = !WholeGridInputsAreDirect(tiling);
    const bool productOutput = !WholeGridOutputIsDirect(tiling);
    if ((packInputs || productOutput) && aivCores == 0) return ACLBLAS_STATUS_INTERNAL_ERROR;
    const uint64_t packTasks = packInputs
        ? std::max<uint64_t>((tiling.m + 1023) / 1024, (tiling.n + 1023) / 1024) : 0;
    const uint64_t epilogueTasks = productOutput
        ? static_cast<uint64_t>(tiling.n) * ((tiling.m + 1023) / 1024) : 0;
    const uint32_t packBlocks = ClampBlocks(packTasks, aivCores);
    const uint32_t epilogueBlocks = ClampBlocks(epilogueTasks, aivCores);
    if ((packInputs && packBlocks == 0) || (productOutput && epilogueBlocks == 0) || cubeBlocks == 0) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    size_t workspaceBytes = 0;
    if (!TryWholeGridWorkspace(&tiling, packInputs, productOutput, &workspaceBytes)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (workspaceBytes != 0) {
        const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
        if (status != ACLBLAS_STATUS_SUCCESS) return status;
    }
    auto* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    LaunchWholeGridSquare(h, tiling, a, b, c, workspace, packInputs, productOutput,
                          packBlocks, epilogueBlocks, cubeBlocks);
    return ACLBLAS_STATUS_SUCCESS;
}

bool TryWorkspaceLayout(GemmArch22Tiling* tiling, size_t* bytes, uint64_t minProductFloats = 0)
{
    const uint64_t packA = static_cast<uint64_t>(tiling->pm) * tiling->pk;
    const uint64_t packB = static_cast<uint64_t>(tiling->pn) * tiling->pk;
    uint64_t product = static_cast<uint64_t>(tiling->pm) * tiling->pn;
    if (product < minProductFloats) product = minProductFloats;
    const uint64_t aEnd = AlignElements(packA);
    const uint64_t bSize = AlignElements(packB);
    const uint64_t productSize = AlignElements(product);
    if (aEnd == std::numeric_limits<uint64_t>::max() || bSize == std::numeric_limits<uint64_t>::max() ||
        productSize == std::numeric_limits<uint64_t>::max() ||
        aEnd > std::numeric_limits<uint64_t>::max() - bSize) return false;
    const uint64_t bEnd = aEnd + bSize;
    if (bEnd > std::numeric_limits<uint64_t>::max() - productSize) return false;
    const uint64_t productEnd = bEnd + productSize;
    if (productEnd > std::numeric_limits<uint64_t>::max() - productSize) return false;
    const uint64_t total = productEnd + productSize;
    if (total > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    tiling->packAOffset = 0;
    tiling->packBOffset = aEnd;
    tiling->productOffset = bEnd;
    tiling->accumOffset = productEnd;
    *bytes = static_cast<size_t>(total * sizeof(float));
    return true;
}

aclblasStatus_t ZeroLogicalC(_aclblas_handle* h, int m, int n, int ldc, float* c)
{
    GemmArch22Tiling tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.ldc = ldc;
    const uint32_t aivCores = GetAivCoreCount();
    const uint32_t blocks = ClampBlocks(static_cast<uint64_t>(m) * n, aivCores);
    if (blocks == 0) return ACLBLAS_STATUS_INTERNAL_ERROR;
    sgemm_arch22_zero_do(blocks, h->stream, reinterpret_cast<uint8_t*>(c), tiling);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateSgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
                              int m, int n, int k, const float* alpha, const float* a, int lda,
                              const float* b, int ldb, const float* beta, float* c, int ldc)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if ((transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) ||
        (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C)) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (!ValidSgemmLeadingDims(transa, transb, m, n, k, lda, ldb, ldc)) return ACLBLAS_STATUS_INVALID_VALUE;
    return ValidSgemmOperands(k, *alpha, a, b, c) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INVALID_VALUE;
}

// Selects the row-owner matmul kernel from the raw transpose combination of
// the public operands, so the dispatch does not re-express this table inline.
void RunRowsMatmul(void* stream, int32_t transA, int32_t transB,
                   GM_ADDR a, GM_ADDR b, GM_ADDR output,
                   const GemmArch22Tiling& tiling)
{
    if (transB == 0) {
        if (transA == 0) {
            sgemm_arch22_matmul_rows_do(stream, a, b, output, tiling);
        } else {
            sgemm_arch22_matmul_rows_right_trans_do(stream, a, b, output, tiling);
        }
    } else if (transA == 0) {
        sgemm_arch22_matmul_rows_left_trans_do(stream, a, b, output, tiling);
    } else {
        sgemm_arch22_matmul_rows_both_trans_do(stream, a, b, output, tiling);
    }
}

// Runs one K contribution of a single host tile. The caller supplies the
// already-bounded (m/n/k) tiling descriptor; this fills the local pack contract
// and dispatches to the direct, raw-orientation, or generic packed path.
void RunKTileContribution(_aclblas_handle* h, GemmArch22Tiling& tiling,
                          const GemmArch22Tiling& wsLayout, bool directWholeProblem,
                          bool directOutputToC, bool tiled2dNn, bool oneKContribution,
                          const float* a, const float* b, uint8_t* workspace, float* c)
{
    tiling.rawPack = !directWholeProblem && CanUseRawTransposePack(tiling) ? 1 : 0;
    GemmArch22Tiling kTiling = tiling;
    if (!oneKContribution && tiling.firstK != 0) {
        kTiling.productOffset = wsLayout.accumOffset;
    }
    if (directWholeProblem) {
        GemmArch22Tiling matmulTiling = kTiling;
        auto* matmulOutput = workspace;
        if (tiled2dNn) {
            sgemm_arch22_matmul_tiles_do(h->stream,
                reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
                reinterpret_cast<uint8_t*>(const_cast<float*>(b)), workspace, matmulTiling);
        } else {
            if (directOutputToC) {
                matmulOutput = reinterpret_cast<uint8_t*>(c);
                matmulTiling.productOffset = 0;
            }
            RunRowsMatmul(h->stream, tiling.transA, tiling.transB,
                          reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
                          reinterpret_cast<uint8_t*>(const_cast<float*>(b)),
                          matmulOutput, matmulTiling);
        }
    } else if (tiling.rawPack != 0) {
        sgemm_arch22_pack_do(ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pk, GetAivCoreCount()), h->stream,
            reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), workspace, tiling);
        auto* packedA = workspace + tiling.packAOffset * sizeof(float);
        auto* packedB = workspace + tiling.packBOffset * sizeof(float);
        RunRowsMatmul(h->stream, tiling.transA, tiling.transB, packedA, packedB,
                      workspace, kTiling);
    } else {
        sgemm_arch22_pack_do(ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pk, GetAivCoreCount()), h->stream,
            reinterpret_cast<uint8_t*>(const_cast<float*>(a)),
            reinterpret_cast<uint8_t*>(const_cast<float*>(b)), workspace, tiling);
        GemmArch22Tiling packedTiling = kTiling;
        packedTiling.transA = 0;
        packedTiling.transB = 0;
        RunRowsMatmul(h->stream, 0, 0,
                      workspace + tiling.packAOffset * sizeof(float),
                      workspace + tiling.packBOffset * sizeof(float),
                      workspace, packedTiling);
    }
    if (!oneKContribution && tiling.firstK == 0) {
        sgemm_arch22_accumulate_do(ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace, tiling);
    }
}

// Runs the full K-loop for one (m,n) host tile and applies the epilogue.
void RunHostTile(_aclblas_handle* h, GemmArch22Tiling& tiling,
                 int32_t hostTileK, int64_t kTotal,
                 const GemmArch22Tiling& wsLayout, bool directWholeProblem,
                 bool directOutputToC, bool tiled2dNn, bool oneKContribution,
                 const float* a, const float* b, uint8_t* workspace, float* c)
{
    for (int64_t kBase = 0; kBase < kTotal; kBase += hostTileK) {
        tiling.kBase = static_cast<int32_t>(kBase);
        tiling.k = static_cast<int32_t>(std::min<int64_t>(hostTileK, kTotal - kBase));
        tiling.pk = AlignDim(tiling.k, 8);
        tiling.firstK = (kBase == 0) ? 1 : 0;
        RunKTileContribution(h, tiling, wsLayout, directWholeProblem,
                             directOutputToC, tiled2dNn, oneKContribution,
                             a, b, workspace, c);
    }
    if (tiled2dNn) {
        sgemm_arch22_scatter_tiles_do(ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace,
                                      reinterpret_cast<uint8_t*>(c), tiling);
    } else if (!directOutputToC) {
        sgemm_arch22_epilogue_do(ClampBlocks(static_cast<uint64_t>(tiling.pm) * tiling.pn, GetAivCoreCount()), h->stream, workspace,
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

SgemmDispatchFlags ComputeSgemmDispatch(const GemmArch22Tiling& tiling, int m, int n, int k)
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
    flags.tiled2dNn = flags.directWholeProblem && tiling.transA == 0 && tiling.transB == 0 &&
                      n <= 128 && m >= 512 && k >= 512;
    flags.directOutputToC = flags.directWholeProblem && tiling.alpha == 1.0f &&
                            tiling.beta == 0.0f && tiling.ldc == tiling.m;
    flags.hostTileM = flags.wholeProblem ? m : kMatmulTileM;
    flags.hostTileN = flags.wholeProblem ? n : kMatmulTileN;
    flags.hostTileK = flags.wholeProblem ? k : kMatmulTileK;
    flags.oneKContribution = k <= flags.hostTileK;
    return flags;
}

// Precompute the workspace layout, allocate the default workspace buffer, and
// return the effective workspace address used by the host-tile loop.
aclblasStatus_t PrepareSgemmWorkspace(_aclblas_handle* h, GemmArch22Tiling& tiling,
                                      const SgemmDispatchFlags& flags, GemmArch22Tiling& workspaceLayout,
                                      size_t& workspaceBytes, uint8_t*& workspace)
{
    workspace = nullptr;
    workspaceLayout = tiling;
    workspaceLayout.pm = AlignDim(flags.hostTileM, 16);
    workspaceLayout.pn = AlignDim(flags.hostTileN, 16);
    workspaceLayout.pk = AlignDim(flags.hostTileK, 8);
    if (!TryWorkspaceLayout(&workspaceLayout, &workspaceBytes,
                            flags.tiled2dNn ? kTilesWorkspaceFloats : 0)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.packAOffset = workspaceLayout.packAOffset;
    tiling.packBOffset = workspaceLayout.packBOffset;
    tiling.productOffset = workspaceLayout.productOffset;
    // With one host K tile, product is already the complete internal C^T.
    // Point the epilogue at it and avoid a product-to-accumulator pass.
    tiling.accumOffset = flags.oneKContribution ? workspaceLayout.productOffset : workspaceLayout.accumOffset;
    const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    return workspace == nullptr ? ACLBLAS_STATUS_ALLOC_FAILED : ACLBLAS_STATUS_SUCCESS;
}
} // namespace

extern "C" aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* a, int lda,
    const float* b, int ldb, const float* beta, float* c, int ldc)
{
    aclblasStatus_t status = ValidateSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS || m == 0 || n == 0) return status;
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    if (k == 0 || *alpha == 0.0f) {
        if (*beta == 0.0f) return ZeroLogicalC(h, m, n, ldc, c);
        for (int j = 0; j < n; ++j) {
            status = aclblasSscal(handle, m, beta, c + static_cast<size_t>(j) * ldc, 1);
            if (status != ACLBLAS_STATUS_SUCCESS) return status;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    GemmArch22Tiling tiling{};
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
    if (tiling.cubeBlocks == 0) return ACLBLAS_STATUS_INTERNAL_ERROR;
    if (CanUseWholeGridSquare(tiling)) return RunWholeGridSquare(h, tiling, a, b, c);

    const SgemmDispatchFlags flags = ComputeSgemmDispatch(tiling, m, n, k);
    const uint32_t aivCores = GetAivCoreCount();
    const bool needsVector = !flags.directOutputToC || flags.tiled2dNn || !flags.directWholeProblem;
    if (needsVector && aivCores == 0) return ACLBLAS_STATUS_INTERNAL_ERROR;
    GemmArch22Tiling workspaceLayout = tiling;
    size_t workspaceBytes = 0;
    uint8_t* workspace = nullptr;
    status = PrepareSgemmWorkspace(h, tiling, flags, workspaceLayout, workspaceBytes, workspace);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    for (int64_t nBase = 0; nBase < n; nBase += flags.hostTileN) {
        for (int64_t mBase = 0; mBase < m; mBase += flags.hostTileM) {
            tiling.nBase = static_cast<int32_t>(nBase);
            tiling.mBase = static_cast<int32_t>(mBase);
            tiling.n = static_cast<int32_t>(std::min<int64_t>(flags.hostTileN, static_cast<int64_t>(n) - nBase));
            tiling.m = static_cast<int32_t>(std::min<int64_t>(flags.hostTileM, static_cast<int64_t>(m) - mBase));
            tiling.pn = AlignDim(tiling.n, 16);
            tiling.pm = AlignDim(tiling.m, 16);
            RunHostTile(h, tiling, flags.hostTileK, k, workspaceLayout, flags.directWholeProblem,
                        flags.directOutputToC, flags.tiled2dNn, flags.oneKContribution, a, b, workspace, c);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
