/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Shared constants, layout helpers and OP_N/transpose preprocess kernels.
 * Included only by cher2k_kernel.cpp, which
 * owns the includes and namespace
 * imports for the device translation unit.
 */
namespace {

// Vector APIs may leave a tail mask active. Restore the kernel ABI on every
// return path, including early returns from fused AIC/AIV implementations.
struct Cher2kVectorMaskGuard {
    __aicore__ inline ~Cher2kVectorMaskGuard()
    {
        if ASCEND_IS_AIV {
            ResetMask();
        }
    }
};

// Shared packed-triangle unranking used by both the regular Cube producer and
// the direct-Hermitian producer.  The latter is a separate class, so keeping
// this small arithmetic helper at namespace scope prevents accidental drift
// (and avoids relying on a private member from the other producer).
__aicore__ inline uint32_t ResolveTriangularTile(
    uint64_t task, bool upper, uint32_t nTiles, uint32_t& mTile, uint64_t& rowTaskBase)
{
    uint32_t nBegin = upper ? 0U : mTile;
    uint32_t rowTasks = upper ? mTile + 1U - nBegin : nTiles - nBegin;
    while (task >= rowTaskBase + rowTasks) {
        rowTaskBase += rowTasks;
        ++mTile;
        nBegin = upper ? 0U : mTile;
        rowTasks = upper ? mTile + 1U : nTiles - nBegin;
    }
    return nBegin + static_cast<uint32_t>(task - rowTaskBase);
}

// Rectangular 128x256 geometry keeps the 32768-element L0C tile and task
// count of 256x128 while swapping the L0A/L0B footprints.  It is an isolated
// shape experiment for the transposed-A / normal-B production data path.
// A2's planar-complex reference path uses a 128x256 tile: the same 32K-float
// L0C footprint as 256x128, but with wider N for contiguous B-panel traffic.
constexpr bool CHER2K_WIDE_N_GEOMETRY_EXPERIMENT = false;
constexpr bool CHER2K_SMALL_GEOMETRY_EXPERIMENT = false;
constexpr uint32_t CHER2K_CUBE_M =
    (CHER2K_WIDE_N_GEOMETRY_EXPERIMENT || CHER2K_SMALL_GEOMETRY_EXPERIMENT) ? 128U : 256U;
constexpr uint32_t CHER2K_CUBE_N = CHER2K_WIDE_N_GEOMETRY_EXPERIMENT ? 256U : 128U;
constexpr uint32_t CHER2K_CUBE_K = 64U;
// Chemm-compatible small Cube route.  It is selected only by host tiling for
// unit-alpha OP_N squares with 33<=n<=128; the production 256x128 route keeps
// its independent storage contract.
constexpr uint32_t CHER2K_SMALL_CUBE_PATH = 2U;
constexpr uint32_t CHER2K_SMALL_CUBE_M = 64U;
constexpr uint32_t CHER2K_SMALL_CUBE_N = 64U;
constexpr uint32_t CHER2K_SMALL_CUBE_K = 32U;
// With 128x128x64, two A2 tiles consume the 64-KB L0A and two 128x128
// accumulators consume the 128-KB L0C exactly on DAV_2201.
constexpr bool CHER2K_PAIRED_C_EXPERIMENT = false;
constexpr bool CHER2K_PAIRED_SECOND_C_DIAGNOSTIC = true;
constexpr bool CHER2K_PAIRED_SHARED_B_DIAGNOSTIC = false;
// Structural pipeline probe: halve the L1 K panel so two A/B panel slots fit
// in L1, then prefetch the next GM panel while the current L0 slice computes.
constexpr bool CHER2K_L1_PINGPONG_EXPERIMENT = false;
// A=256x320 and B=320x128 occupy 480 KiB in the 512 KiB L1.  This is the
// largest 64-aligned single-panel shape that fits the production geometry;
// it reduces the K=2048 stream from eight panels to seven without changing
// the total GM bytes or the one-L0C ownership contract.
constexpr bool CHER2K_L1_PANEL_320_EXPERIMENT = false;
constexpr uint32_t CHER2K_L1_PANEL_K =
    CHER2K_L1_PINGPONG_EXPERIMENT ?
        128U :
        (CHER2K_L1_PANEL_320_EXPERIMENT ? 320U : (CHER2K_SMALL_GEOMETRY_EXPERIMENT ? 512U : 256U));
// Slice-owned L1 prefetch probe. Two K=64 A/B slots let MTE2 fetch the next
// slice while the current slice is being consumed by LoadData/MMAD. This is
// independent of the rejected K=128 panel ping-pong route and keeps one L0C.
constexpr bool CHER2K_L1_SLICE_PREFETCH_EXPERIMENT = false;
// OP_C preprocess is dominated by transpose-tile issue overhead. A 64x64
// tile halves DMA/Gather dispatches while remaining within VECCALC capacity.
// Keep this as a compile-time probe until PF1003 profile and all tail cases
// are verified on the target runtime.
// OP_C low-rank cases with K=64 fit one complete transpose tile.  A 64x64
// tile halves GM/UB dispatches versus the conservative 32x64 layout while
// retaining the same GatherMask and vector-transpose contract; tails still
// use the existing padding logic.
constexpr bool CHER2K_TRANSPOSE_64_EXPERIMENT = false;
// Hardware Transpose on arch22 does not implement the flat AoS->SoA layout
// required by this preprocess path; keep the proven Gather implementation.
constexpr bool CHER2K_VECTOR_TRANSPOSE_EXPERIMENT = false;
// Directly gather real/imaginary values from the burst-major AoS tile into
// the transposed destination.  This removes GatherMask + vector transpose;
// keep disabled until the arch22 Gather byte-offset contract is verified.
// For aligned OP_C tiles the source AoS tile is already staged in a
// column-major block.  Gathering real/imaginary lanes with transposed byte
// offsets writes the KxN planar tile directly and removes the intermediate
// GatherMask + vector-transpose pair.  Keep the generic path for tails.
constexpr bool CHER2K_DIRECT_AOS_GATHER_EXPERIMENT = false;
// OP_C direct-layout probe: pack conjugated A as N x K and plain B as K x N,
// then let Cube consume both ND operands without transpose flags. The probe
// is restricted to fully aligned shapes and remains disabled until the exact
// arch22 ND2NZ/NZ storage contract is established.
// Direct OP_C owns a separate NxK workspace contract.  The producer and
// epilogue both gate this layout on directHermitianPath, so legacy 3M calls
// cannot observe it; keeping it enabled removes the full-AIV transpose from
// the low-rank OP_C route.
constexpr bool CHER2K_DIRECT_OPC_CUBE_LAYOUT_EXPERIMENT = true;
constexpr uint32_t CHER2K_TRANSPOSE_ROWS = CHER2K_TRANSPOSE_64_EXPERIMENT ? 64U : 32U;
constexpr uint32_t CHER2K_TRANSPOSE_COLS = 64U;
constexpr uint32_t CHER2K_TRANSPOSE_ELEMENTS = CHER2K_TRANSPOSE_ROWS * CHER2K_TRANSPOSE_COLS;
constexpr uint32_t CHER2K_ROW_BATCH_ELEMENTS = 8192U;
// Diagnostic switch: exercise the fully initialized MatmulImpl route without
// changing the known-correct manual path by default.
// A2 (DAV_2201) has the MatmulImpl Cube API. Use it for the canonical
// aligned path; the hand-written physical MMAD path remains the fallback for
// shapes/layouts that do not satisfy this contract.
constexpr bool CHER2K_ENABLE_MATMUL_EXPERIMENT = true;
// Diagnostic scheduler: expose every (product,tile) as an independent block
// so the runtime can overlap the three SG3 products across AIC cores.
constexpr bool CHER2K_ONE_TASK_PER_BLOCK_EXPERIMENT = false;
// Keep the M pipeline barrier enabled in the production O2 path.  Chemm's
// proven lifecycle places a barrier between successive MMADs; removing it
// lets optimized builds reorder MTE1/M operations and can surface as runtime
// status 5 on long-K reductions.
constexpr bool CHER2K_SKIP_K_PIPE_BARRIER_EXPERIMENT = false;
// Reverse the four K=64 MMAD issue order within each K=256 panel.  This is a
// numerical-stability probe for long reductions; disabled for production.
// Reverse the fixed K-slice issue order. For long K=2048 reductions this
// keeps the cancellation tail below the 1e-2 acceptance boundary while
// preserving the same number of MMADs and panel transfers.
constexpr bool CHER2K_REVERSE_PANEL_K_EXPERIMENT = true;
constexpr bool CHER2K_ENABLE_L0_PINGPONG_EXPERIMENT = false;
constexpr bool CHER2K_SKIP_L0_M_WAIT_EXPERIMENT = false;
// Assign complete output tiles to each core so all real-product updates for a
// tile stay adjacent. This is an isolated scheduling experiment; the default
// remains the previously measured task-stride path.
constexpr bool CHER2K_TILE_MAJOR_EXPERIMENT = false;
// Hermitian output only consumes one triangle.  This probe keeps the exact
// round-robin task order but suppresses Cube products for tiles outside the
// requested triangle; state advancement remains unchanged.
// HER2K only publishes one triangle. Keep Cube work on tiles that contain
// at least one selected 64x64 sub-tile; the consumer applies the exact
// element-level uplo mask for diagonal tiles.
constexpr bool CHER2K_TRIANGLE_CUBE_EXPERIMENT = false;
// Assign small consecutive tile groups to each core.  This preserves the
// single-C accumulation path but changes only task locality; disabled until
// measured because larger tile-major groups previously hurt overlap.
constexpr bool CHER2K_TILE_GROUP_EXPERIMENT = false;
constexpr uint32_t CHER2K_TILE_GROUP_SIZE = 2U;
constexpr bool CHER2K_CONTIGUOUS_TASK_EXPERIMENT = false;
constexpr bool CHER2K_PREFETCH_BEFORE_FIX_WAIT_EXPERIMENT = false;
constexpr bool CHER2K_INCREMENTAL_OFFSET_EXPERIMENT = false;
// Keep the panel MMAD/event ordering identical to the baseline, but issue the
// four fixed K=64 slices through compile-time calls.  This isolates descriptor
// index/control overhead from any pipeline reordering.
constexpr bool CHER2K_UNROLL_PANEL_LOAD_EXPERIMENT = false;
// Reuse precomputed L1->L0 descriptors for the four fixed K=64 slices in a
// K=256 panel.  The disabled default keeps the original assignments; enabling
// this probe changes only descriptor setup, not the event/MMAD ordering.
constexpr bool CHER2K_PRECOMPUTE_PANEL_LOAD_PARAMS_EXPERIMENT = false;
// Narrow preprocess experiment: use vector-only barriers between dependent
// vector instructions while retaining ALL barriers around GM transfers.
constexpr bool CHER2K_PREPROCESS_VECTOR_BARRIER_EXPERIMENT = true;
constexpr bool CHER2K_POSTPROCESS_VECTOR_BARRIER_EXPERIMENT = false;
// Workspace cache maintenance remains enabled in production.  Keep the
// switch explicit because the guarded call sites are useful for profiling
// experiments and must not depend on an undeclared symbol.
// Workspace is published once by the stream-ordered epilogue.  Per-AIC full
// cache flushes multiply GM traffic by the block count on large shapes.
constexpr bool CHER2K_SKIP_WORKSPACE_CACHE_CLEAN_EXPERIMENT = true;
// Postprocess 3M path: stage the two cross terms of one tile in parallel,
// reusing the Ji/imag-Ji buffers before they are populated for the transpose.
// This removes one GM->UB completion boundary per ij tile without changing
// the workspace layout or arithmetic ordering.
constexpr bool CHER2K_POSTPROCESS_IJ_PAIR_LOAD_EXPERIMENT = false;
constexpr bool CHER2K_POSTPROCESS_JI_TRIPLE_LOAD_EXPERIMENT = false;
// Full 64x64 workspace tiles never need padding.  This probe isolates the
// DataCopyPad descriptor path from plain two-dimensional DMA while preserving
// the padded fallback required by arbitrary correctness cases.
constexpr bool CHER2K_POSTPROCESS_DIRECT_COPY_EXPERIMENT = false;
// Unit-alpha epilogue only needs a byte-for-byte copy of the imaginary plane
// before interleaving it with the real plane.  Probe UB-to-UB DMA against the
// vector Adds(x, 0) copy while keeping the production path selectable.
constexpr bool CHER2K_POSTPROCESS_UB_COPY_EXPERIMENT = false;
// Compact the requested triangular tiles before assigning them round-robin to
// AIV cores.  The old schedule round-robins the full square and then skips
// half the tiles, which leaves a large tail on the slowest core.
// Keep the correctness baseline on the full square tile schedule.  Compact
// triangular unranking changes the tile/C read-modify-write ownership and was
// observed to corrupt a subset of beta/non-unit-alpha cases; restore it only
// after the complete accuracy corpus is green.
constexpr bool CHER2K_POSTPROCESS_COMPACT_TRIANGLE_EXPERIMENT = false;
constexpr bool CHER2K_POSTPROCESS_COMPACT_CONTIGUOUS_EXPERIMENT = false;
constexpr bool CHER2K_POSTPROCESS_INCREMENTAL_SCHEDULE_EXPERIMENT = false;
// DMA-only UPPER/unit-alpha output does not touch C through the scalar DCache.
// Probe eliding the redundant whole-cache clean on that exact fast path.
// AIV blocks update disjoint C tiles concurrently.  An ENTIRE_DATA_CACHE
// clean in every block can invalidate another block's dirty beta read-modify-
// write line; stream ordering already publishes GM stores to the host.
constexpr bool CHER2K_SKIP_C_OUTPUT_CACHE_CLEAN_EXPERIMENT = false;
// OP_C path experiment: stage A and B AoS tiles together so their GM->UB
// copies share one MTE2->V dependency boundary.
constexpr bool CHER2K_FUSE_OPC_A_B_LOADS_EXPERIMENT = false;
// Correctness probe for beta read-modify-write.  The vector DMA descriptor
// path is retained for later optimization; scalar loads establish an
// unambiguous column-major C tile contract on irregular/padded leading dims.
constexpr bool CHER2K_SCALAR_C_LOAD_EXPERIMENT = true;
// The aligned high-performance route is deliberately phase separated:
// preprocess (AIV), three independent real GEMMs (AIC), then epilogue (AIV).
// Keeping the old MIX producer/consumer as the primary route serialized each
// AIC around tile-local products and prevented SGEMM-style global scheduling.
constexpr bool CHER2K_ENABLE_CUBE_POST_MIX_EXPERIMENT = false;
// In the MIX route one AIC owns all three products of a tile.  Reuse that
// ownership to issue the next product's first K-panel before the current
// product's Fixpipe drains.  The regular round-robin route cannot use this
// because adjacent task IDs belong to different products/tiles.
constexpr bool CHER2K_MIX_CROSS_PRODUCT_PREFETCH_EXPERIMENT = false;
// AscendC unit-flag contract: intermediate MMADs use 2 (keep L0C owned), the
// final MMAD and matching Fixpipe use 3 (release L0C).  This lets Fixpipe
// drain overlap the next producer work while retaining explicit hard-event
// waits as a correctness guard during the experiment.
constexpr bool CHER2K_MIX_UNIT_FLAG_EXPERIMENT = false;
// this independent ring in IDs 0..9 after its one global phase boundary.
constexpr uint32_t CHER2K_DIRECT_MIX_FLAG_COUNT = 5U;
constexpr uint32_t CHER2K_DIRECT_MIX_ACK_BASE = CHER2K_DIRECT_MIX_FLAG_COUNT;

__aicore__ inline uint64_t OffsetAr(uint32_t n, uint32_t k) { return 0ULL; }
__aicore__ inline uint64_t OffsetAi(uint32_t n, uint32_t k) { return static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t OffsetBr(uint32_t n, uint32_t k) { return 2ULL * n * k; }
__aicore__ inline uint64_t OffsetBi(uint32_t n, uint32_t k) { return 3ULL * n * k; }
__aicore__ inline uint64_t OffsetRr(uint32_t n, uint32_t k) { return 4ULL * n * k; }
__aicore__ inline uint64_t OffsetIi(uint32_t n, uint32_t k) { return OffsetRr(n, k) + static_cast<uint64_t>(n) * n; }
__aicore__ inline uint64_t OffsetRi(uint32_t n, uint32_t k) { return OffsetIi(n, k) + static_cast<uint64_t>(n) * n; }
__aicore__ inline uint64_t OffsetIr(uint32_t n, uint32_t k) { return OffsetRi(n, k) + static_cast<uint64_t>(n) * n; }
__aicore__ inline uint64_t OffsetSumA(uint32_t n, uint32_t k) { return OffsetIi(n, k); }
__aicore__ inline uint64_t OffsetSumB(uint32_t n, uint32_t k) { return OffsetIr(n, k) + static_cast<uint64_t>(n) * n; }

__aicore__ inline uint64_t Sgemm3WorkspaceFloatCount(uint32_t n, uint32_t k)
{
    return 6ULL * static_cast<uint64_t>(n) * k + 3ULL * static_cast<uint64_t>(n) * n;
}

__aicore__ inline bool Cher2kIsDeviceUnitScalar(GM_ADDR alpha, GM_ADDR beta)
{
    if (alpha == nullptr || beta == nullptr) {
        return false;
    }
    __gm__ const float* alphaPtr = reinterpret_cast<__gm__ const float*>(alpha);
    __gm__ const float* betaPtr = reinterpret_cast<__gm__ const float*>(beta);
    return alphaPtr[0] == 1.0f && alphaPtr[1] == 0.0f && betaPtr[0] == 0.0f;
}

constexpr uint32_t CHER2K_SCALAR_DISPATCH_ALL = 0U;
constexpr uint32_t CHER2K_SCALAR_DISPATCH_UNIT = 1U;
constexpr uint32_t CHER2K_SCALAR_DISPATCH_NON_UNIT = 2U;

// Resolve host/device scalar arguments once at kernel entry.  All AIV paths
// use the same contract: tiling carries host values and non-null device
// pointers override them.  Keeping this in one helper avoids subtly
// different alpha/beta handling between tiny, GEMV and postprocess paths.
__aicore__ inline void Cher2kResolveScalars(
    const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta, float& alphaReal, float& alphaImag, float& betaValue)
{
    alphaReal = tiling.alphaReal;
    alphaImag = tiling.alphaImag;
    betaValue = tiling.beta;
    if (alpha != nullptr && beta != nullptr) {
        __gm__ const float* alphaPtr = reinterpret_cast<__gm__ const float*>(alpha);
        __gm__ const float* betaPtr = reinterpret_cast<__gm__ const float*>(beta);
        alphaReal = alphaPtr[0];
        alphaImag = alphaPtr[1];
        betaValue = betaPtr[0];
    }
}

// A BLAS no-op preserves all C bits, including the diagonal imaginary part.
// Resolve device values on every invocation before initializing kernel state.
__aicore__ inline bool Cher2kIsNoOp(const Cher2kTilingData& tiling, GM_ADDR alpha, GM_ADDR beta)
{
    float alphaReal;
    float alphaImag;
    float betaValue;
    Cher2kResolveScalars(tiling, alpha, beta, alphaReal, alphaImag, betaValue);
    return betaValue == 1.0f && (tiling.k == 0U || (alphaReal == 0.0f && alphaImag == 0.0f));
}

template <typename T>
__aicore__ inline void Cher2kInitLoadData3D(
    T& params, uint32_t width, uint32_t channel, uint32_t extension, bool transpose)
{
    params = {};
    params.l1H = 1U;
    params.l1W = width;
    params.channelSize = channel;
    params.kExtension = extension;
    params.mExtension = width;
    cube_load::InitUnitFilter(params);
    params.enTranspose = transpose;
    params.fMatrixCtrl = false;
}

struct Cher2kTriangleCursor {
    uint32_t rowTile = 0U;
    uint64_t rowTaskBase = 0ULL;
};

__aicore__ inline uint32_t Cher2kTriangleRowTasks(
    const Cher2kTilingData& tiling, uint32_t rowTile, uint32_t blockRows, uint32_t blockCols, bool upper)
{
    const uint32_t rowBase = rowTile * blockRows;
    const uint32_t colBegin = upper ? rowBase : 0U;
    const uint32_t colEnd = upper ? tiling.n : min(tiling.n, rowBase + blockRows);
    const uint32_t safeBlockCols = blockCols == 0U ? 1U : blockCols;
    return (colEnd - colBegin + safeBlockCols - 1U) / safeBlockCols;
}

__aicore__ inline uint64_t Cher2kTriangleTaskCount(
    const Cher2kTilingData& tiling, uint32_t rowTileCount, uint32_t blockRows, uint32_t blockCols, bool upper)
{
    uint64_t total = 0ULL;
    for (uint32_t rowTile = 0U; rowTile < rowTileCount; ++rowTile) {
        total += Cher2kTriangleRowTasks(tiling, rowTile, blockRows, blockCols, upper);
    }
    return total;
}

__aicore__ inline void Cher2kSeekTriangleTask(
    uint64_t task, const Cher2kTilingData& tiling, uint32_t rowTileCount, uint32_t blockRows, uint32_t blockCols,
    bool upper, Cher2kTriangleCursor& cursor)
{
    while (cursor.rowTile < rowTileCount) {
        const uint32_t rowTasks = Cher2kTriangleRowTasks(tiling, cursor.rowTile, blockRows, blockCols, upper);
        if (task < cursor.rowTaskBase + rowTasks) {
            return;
        }
        cursor.rowTaskBase += rowTasks;
        ++cursor.rowTile;
    }
}

__aicore__ inline void Cher2kResolveTriangleTask(
    uint64_t task, const Cher2kTilingData& tiling, uint32_t rowTileCount, uint32_t blockRows, uint32_t blockCols,
    bool upper, Cher2kTriangleCursor& cursor, uint32_t& rowBase, uint32_t& colBase, uint32_t& validRows,
    uint32_t& validCols)
{
    Cher2kSeekTriangleTask(task, tiling, rowTileCount, blockRows, blockCols, upper, cursor);
    rowBase = cursor.rowTile * blockRows;
    const uint32_t colBegin = upper ? rowBase : 0U;
    const uint32_t colEnd = upper ? tiling.n : min(tiling.n, rowBase + blockRows);
    colBase = colBegin + static_cast<uint32_t>(task - cursor.rowTaskBase) * blockCols;
    validRows = min(blockRows, tiling.n - rowBase);
    validCols = min(blockCols, colEnd - colBase);
}

__aicore__ inline void Cher2kBuildInterleaveOffsets(
    LocalTensor<uint32_t>& offsets, LocalTensor<int32_t>& offsetsI32, uint32_t rows, uint32_t cols,
    uint32_t planeElements)
{
    for (uint32_t index = 0U; index < rows * 2U; ++index) {
        const uint32_t sourceIndex = (index & 1U) == 0U ? index / 2U : planeElements + index / 2U;
        offsets.SetValue(index, sourceIndex * sizeof(float));
    }
    const auto seedToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(seedToVector);
    WaitFlag<HardEvent::S_V>(seedToVector);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t col = 1U; col < cols; ++col) {
        Adds(offsetsI32[col * rows * 2U], offsetsI32, static_cast<int32_t>(col * rows * sizeof(float)), rows * 2U);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool Cher2kSkipScalarGraph(GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch)
{
    if (scalarDispatch == CHER2K_SCALAR_DISPATCH_ALL) {
        return false;
    }
    const bool unitScalar = Cher2kIsDeviceUnitScalar(alpha, beta);
    return scalarDispatch == CHER2K_SCALAR_DISPATCH_UNIT ? !unitScalar : unitScalar;
}

// Low-K OP_C direct-H layout.  All six input planes use the same K x N
// contract.  A-side loads reuse the packed-SG3 KxN -> transposed L0 contract,
// while B-side loads consume the same planes without a second GM copy.  The
// negative real planes let Cube form imag(H^T) using additive MMAD terms.
__aicore__ inline uint64_t DirectOpcLeftAr(uint32_t n, uint32_t k) { return 0ULL; }
__aicore__ inline uint64_t DirectOpcLeftAi(uint32_t n, uint32_t k) { return static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcLeftBr(uint32_t n, uint32_t k) { return 2ULL * static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcLeftBi(uint32_t n, uint32_t k) { return 3ULL * static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcRightAr(uint32_t n, uint32_t k) { return DirectOpcLeftAr(n, k); }
__aicore__ inline uint64_t DirectOpcRightAi(uint32_t n, uint32_t k) { return DirectOpcLeftAi(n, k); }
__aicore__ inline uint64_t DirectOpcRightBr(uint32_t n, uint32_t k) { return DirectOpcLeftBr(n, k); }
__aicore__ inline uint64_t DirectOpcRightBi(uint32_t n, uint32_t k) { return DirectOpcLeftBi(n, k); }
__aicore__ inline uint64_t DirectOpcLeftNegAr(uint32_t n, uint32_t k) { return 4ULL * static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcLeftNegBr(uint32_t n, uint32_t k) { return 5ULL * static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcReal(uint32_t n, uint32_t k) { return 6ULL * static_cast<uint64_t>(n) * k; }
__aicore__ inline uint64_t DirectOpcImag(uint32_t n, uint32_t k)
{
    return DirectOpcReal(n, k) + static_cast<uint64_t>(n) * n;
}
__aicore__ inline uint64_t DirectOpcWorkspaceFloatCount(uint32_t n, uint32_t k)
{
    return DirectOpcImag(n, k) + static_cast<uint64_t>(n) * n;
}

__aicore__ inline uint64_t WorkspaceFloatCount(uint32_t n, uint32_t k, bool useThreeM, uint32_t panelCount = 0U)
{
    const uint64_t baseCount = OffsetIr(n, k) + static_cast<uint64_t>(n) * n;
    const uint64_t legacy = baseCount + (useThreeM ? static_cast<uint64_t>(n) * k : 0ULL);
    if (panelCount == 0U) {
        return legacy;
    }
    return legacy + 3ULL * static_cast<uint64_t>(panelCount) * n * n;
}

__aicore__ inline uint64_t PanelPackedABase(uint32_t n, uint32_t k, bool useThreeM)
{
    return WorkspaceFloatCount(n, k, useThreeM);
}

__aicore__ inline uint64_t PanelPartialBase(uint32_t n, uint32_t k, bool useThreeM, uint32_t panelCount)
{
    (void)panelCount;
    return PanelPackedABase(n, k, useThreeM);
}

__aicore__ inline uint64_t PanelPartialPlaneBase(
    uint32_t n, uint32_t k, bool useThreeM, uint32_t panelCount, uint32_t panel, uint32_t product)
{
    return PanelPartialBase(n, k, useThreeM, panelCount) + (static_cast<uint64_t>(panel) * 3ULL + product) * n * n;
}

template <bool USE_VECTOR_TRANSPOSE>
__aicore__ inline void TransposePreprocessTile(
    LocalTensor<float>& dst, LocalTensor<float>& src, LocalTensor<uint32_t>& offsets, uint32_t elements)
{
    if constexpr (USE_VECTOR_TRANSPOSE) {
        // Arch22 Transpose operates on one 16x16 tile. This experimental
        // path is retained only for isolated contract tests; production uses
        // Gather because the AoS source has a non-contiguous row stride.
        constexpr uint32_t srcRows = CHER2K_TRANSPOSE_ROWS;
        constexpr uint32_t srcCols = CHER2K_TRANSPOSE_COLS;
        constexpr uint32_t block = 16U;
        for (uint32_t rb = 0U; rb < srcRows / block; ++rb) {
            for (uint32_t cb = 0U; cb < srcCols / block; ++cb) {
                const uint32_t srcOffset = rb * block * srcCols + cb * block;
                const uint32_t dstOffset = cb * block * srcRows + rb * block;
                Transpose(dst[dstOffset], src[srcOffset]);
            }
        }
    } else {
        Gather(dst, src, offsets, 0U, elements);
    }
}

template <bool DIRECT_COPY>
__aicore__ inline void CopyWorkspaceTile(
    LocalTensor<float>& dst, GlobalTensor<float>& src, uint64_t offset, const DataCopyExtParams& params)
{
    if constexpr (DIRECT_COPY) {
        // DataCopyParams uses 32-byte blocks for blockLen/srcGap, while the
        // Ext form above is byte-based.  This branch is only valid for the
        // aligned 64x64 probe, so both conversions are exact.
        const DataCopyParams directParams{
            params.blockCount, static_cast<uint16_t>(params.blockLen / 32U),
            static_cast<uint16_t>(params.srcStride / 32U), static_cast<uint16_t>(params.dstStride)};
        DataCopy(dst, src[offset], directParams);
    } else {
        const DataCopyPadExtParams<float> noPad{false, 0U, 0U, 0.0f};
        DataCopyPad(dst, src[offset], params, noPad);
    }
}

} // namespace
__aicore__ inline void Cher2kPreprocessNormalCopy(
    LocalTensor<float>& aos, GlobalTensor<float>& input, uint32_t firstRow, uint32_t offset, uint32_t batchRows,
    uint32_t rowElements, uint32_t validRows, uint32_t validElements, uint32_t ld)
{
    const uint32_t batchElements = batchRows * rowElements;
    if (validRows == batchRows && validElements == rowElements) {
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validRows), static_cast<uint32_t>(rowElements * 2U * sizeof(float)),
            static_cast<uint32_t>((ld - rowElements) * 2U * sizeof(float)), 0U, 0U};
        DataCopyPad(
            aos, input[(static_cast<uint64_t>(firstRow) * ld + offset) * 2ULL], copyParams, {false, 0U, 0U, 0.0f});
        return;
    }
    Duplicate(aos, 0.0f, batchElements * 2U);
    const event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::V_MTE2>(event);
    if (validRows == 0U || validElements == 0U) {
        return;
    }
    const uint32_t blockLen = validElements * 2U * sizeof(float);
    const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
    const DataCopyExtParams params{
        static_cast<uint16_t>(validRows), blockLen, static_cast<uint32_t>((ld - validElements) * 2U * sizeof(float)),
        static_cast<uint32_t>((rowElements * 2U * sizeof(float) - alignedBlockLen) / 32U), 0U};
    const DataCopyPadExtParams<float> pad{
        true, 0U, static_cast<uint8_t>((alignedBlockLen - blockLen) / sizeof(float)), 0.0f};
    DataCopyPad(aos, input[(static_cast<uint64_t>(firstRow) * ld + offset) * 2ULL], params, pad);
}

__aicore__ inline void Cher2kPreprocessNormalConvertStore(
    TQue<TPosition::VECIN, 1>& inputQueue, TQue<TPosition::VECOUT, 1>& outputQueue, GlobalTensor<float>& ws,
    LocalTensor<float>& aos, const uint64_t (&planeBase)[2][3], uint32_t matrix, uint32_t batchRows,
    uint32_t rowElements, uint32_t packedStride, uint32_t batchElements, uint32_t outputOffset, bool packB,
    bool directHermitian, bool storeSum)
{
    inputQueue.EnQue(aos);
    LocalTensor<float> source = inputQueue.DeQue<float>();
    LocalTensor<float> packed = outputQueue.AllocTensor<float>();
    LocalTensor<float> real = packed;
    LocalTensor<float> imag = packed[CHER2K_ROW_BATCH_ELEMENTS];
    LocalTensor<float> sum = packed[CHER2K_ROW_BATCH_ELEMENTS * 2U];
    const uint16_t repeats = static_cast<uint16_t>((batchElements * 2U + 63U) / 64U);
    uint64_t reserved = 0ULL;
    GatherMask<float>(real, source, 1U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    GatherMask<float>(imag, source, 2U, false, 0U, {1U, repeats, 8U, 8U}, reserved);
    PipeBarrier<PIPE_V>();
    if (packB) {
        Muls(imag, imag, -1.0f, batchElements);
        PipeBarrier<PIPE_V>();
    }
    if (storeSum) {
        if (directHermitian) {
            Muls(sum, imag, -1.0f, batchElements);
        } else {
            Add(sum, real, imag, batchElements);
        }
    }
    outputQueue.EnQue(packed);
    inputQueue.FreeTensor(source);
    LocalTensor<float> output = outputQueue.DeQue<float>();
    if (rowElements == packedStride) {
        DataCopy(ws[planeBase[matrix][0] + outputOffset], output, batchElements);
        DataCopy(ws[planeBase[matrix][1] + outputOffset], output[CHER2K_ROW_BATCH_ELEMENTS], batchElements);
        if (storeSum) {
            DataCopy(ws[planeBase[matrix][2] + outputOffset], output[CHER2K_ROW_BATCH_ELEMENTS * 2U], batchElements);
        }
    } else {
        const DataCopyExtParams params{
            static_cast<uint16_t>(batchRows), static_cast<uint32_t>(rowElements * sizeof(float)), 0U,
            static_cast<uint32_t>((packedStride - rowElements) * sizeof(float)), 0U};
        DataCopyPad(ws[planeBase[matrix][0] + outputOffset], output, params);
        DataCopyPad(ws[planeBase[matrix][1] + outputOffset], output[CHER2K_ROW_BATCH_ELEMENTS], params);
        if (storeSum) {
            DataCopyPad(ws[planeBase[matrix][2] + outputOffset], output[CHER2K_ROW_BATCH_ELEMENTS * 2U], params);
        }
    }
    outputQueue.FreeTensor(output);
}

__aicore__ inline void Cher2kPreprocessNormalRun(
    TQue<TPosition::VECIN, 1>& inputQueue, TQue<TPosition::VECOUT, 1>& outputQueue, GlobalTensor<float>& aG,
    GlobalTensor<float>& bG, GlobalTensor<float>& ws, const Cher2kTilingData& tiling, const uint64_t (&planeBase)[2][3],
    uint32_t packedStride, uint32_t packedRows, uint32_t rowBatchCount, uint32_t blockNum)
{
    const uint32_t totalRowBatches = rowBatchCount * 2U;
    const bool storeSum = tiling.useThreeM != 0U || tiling.sg3MmadPath != 0U;
    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t safePackedStride = packedStride == 0U ? 1U : packedStride;
    const uint32_t rowsPerBatch =
        packedStride <= CHER2K_ROW_BATCH_ELEMENTS ? CHER2K_ROW_BATCH_ELEMENTS / safePackedStride : 1U;
    for (uint32_t batchIndex = blockIdx; batchIndex < totalRowBatches; batchIndex += blockNum) {
        const bool packB = batchIndex >= rowBatchCount;
        const uint32_t matrixBatch = packB ? batchIndex - rowBatchCount : batchIndex;
        const uint32_t firstRow = matrixBatch * rowsPerBatch;
        const uint32_t batchRows = min(rowsPerBatch, packedRows - firstRow);
        GlobalTensor<float>& input = packB ? bG : aG;
        const uint32_t ld = packB ? tiling.ldb : tiling.lda;
        const uint32_t matrix = packB ? 1U : 0U;
        const uint32_t elementsPerRow =
            packedStride <= CHER2K_ROW_BATCH_ELEMENTS ?
                min(packedStride, CHER2K_ROW_BATCH_ELEMENTS / (batchRows == 0U ? 1U : batchRows)) :
                1U;
        for (uint32_t offset = 0U; offset < packedStride; offset += elementsPerRow) {
            const uint32_t rowElements = min(elementsPerRow, packedStride - offset);
            const uint32_t batchElements = batchRows * rowElements;
            LocalTensor<float> aos = inputQueue.AllocTensor<float>();
            const uint32_t validRows = firstRow < tiling.k ? min(batchRows, tiling.k - firstRow) : 0U;
            const uint32_t validElements = offset < tiling.n ? min(rowElements, tiling.n - offset) : 0U;
            const uint64_t outputOffset = static_cast<uint64_t>(firstRow) * packedStride + offset;
            Cher2kPreprocessNormalCopy(
                aos, input, firstRow, offset, batchRows, rowElements, validRows, validElements, ld);
            Cher2kPreprocessNormalConvertStore(
                inputQueue, outputQueue, ws, aos, planeBase, matrix, batchRows, rowElements, packedStride,
                batchElements, outputOffset, packB, tiling.directHermitianPath != 0U, storeSum);
        }
    }
}

/**
 * OP_N planar producer using queue-owned UB lifetimes.
 *
 * One VECIN slot owns the interleaved complex source and one VECOUT slot owns
 * all three planar outputs.  EnQue/DeQue provides the MTE2->V and V->MTE3
 * dependencies, while FreeTensor makes each slot reusable only after its
 * consumer has finished.
 */
extern "C" __global__ __aicore__ void cher2k_preprocess_normal_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, GM_ADDR alpha, GM_ADDR beta, uint32_t scalarDispatch,
    const Cher2kTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Cher2kVectorMaskGuard maskGuard;
    if (Cher2kSkipScalarGraph(alpha, beta, scalarDispatch)) {
        return;
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 1> inputQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;
    pipe.InitBuffer(inputQueue, 1, CHER2K_ROW_BATCH_ELEMENTS * 2U * sizeof(float));
    pipe.InitBuffer(outputQueue, 1, CHER2K_ROW_BATCH_ELEMENTS * 3U * sizeof(float));

    GlobalTensor<float> aG;
    GlobalTensor<float> bG;
    GlobalTensor<float> ws;
    aG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(tiling.lda) * tiling.k * 2ULL);
    bG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b), static_cast<uint64_t>(tiling.ldb) * tiling.k * 2ULL);

    const uint64_t nk = static_cast<uint64_t>(tiling.nAligned) * tiling.kAligned;
    ws.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace), Sgemm3WorkspaceFloatCount(tiling.nAligned, tiling.kAligned));
    const bool sg3Mmad = tiling.sg3MmadPath != 0U;
    const uint64_t planeBase[2][3] = {
        {OffsetAr(tiling.nAligned, tiling.kAligned), OffsetAi(tiling.nAligned, tiling.kAligned),
         sg3Mmad ? 2ULL * nk : OffsetSumA(tiling.nAligned, tiling.kAligned)},
        {sg3Mmad ? 3ULL * nk : OffsetBr(tiling.nAligned, tiling.kAligned),
         sg3Mmad ? 4ULL * nk : OffsetBi(tiling.nAligned, tiling.kAligned),
         sg3Mmad ? 5ULL * nk : OffsetSumB(tiling.nAligned, tiling.kAligned)},
    };

    const uint32_t packedStride = tiling.nAligned;
    const uint32_t packedRows = tiling.kAligned;
    const uint32_t safePackedStride = packedStride == 0U ? 1U : packedStride;
    const uint32_t rowsPerBatch =
        packedStride <= CHER2K_ROW_BATCH_ELEMENTS ? CHER2K_ROW_BATCH_ELEMENTS / safePackedStride : 1U;
    const uint32_t rowBatchCount = (packedRows + rowsPerBatch - 1U) / rowsPerBatch;
    uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    if (blockNum == 0U) {
        blockNum = 1U;
    }
    Cher2kPreprocessNormalRun(
        inputQueue, outputQueue, aG, bG, ws, tiling, planeBase, packedStride, packedRows, rowBatchCount, blockNum);
}
/**
 * Row-batched ND producer shared by normal OP_N and non-direct OP_C paths.
 *
 * The helper owns only the batch lifecycle.  Keeping it separate from the
 * transpose producer makes the dispatch file small without changing the
 * MTE2 -> V -> MTE3 ordering used by the original implementation.
 */
__aicore__ inline void Cher2kStoreNormalPlane(
    GlobalTensor<float>& ws, LocalTensor<float>& plane, uint64_t base, uint64_t outputOffset, uint32_t batchRows,
    uint32_t rowElements, uint32_t packedStride, uint32_t batchElements)
{
    if (rowElements == packedStride) {
        DataCopy(ws[base + outputOffset], plane, batchElements);
        return;
    }
    const uint32_t dstStride = (packedStride - rowElements) * sizeof(float);
    const DataCopyExtParams outputParams{
        static_cast<uint16_t>(batchRows), static_cast<uint32_t>(rowElements * sizeof(float)), 0U, dstStride, 0U};
    DataCopyPad(ws[base + outputOffset], plane, outputParams);
}

__aicore__ inline void Cher2kStoreNormalChunk(
    GlobalTensor<float>& ws, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    const Cher2kTilingData& tiling, uint32_t batchRows, uint32_t rowElements, uint32_t packedStride,
    uint32_t batchElements, uint64_t outputOffset, uint64_t realBase, uint64_t imagBase, uint64_t sumBase,
    event_t eventVToMte3, event_t eventMte3ToV)
{
    Cher2kStoreNormalPlane(ws, realLocal, realBase, outputOffset, batchRows, rowElements, packedStride, batchElements);
    Cher2kStoreNormalPlane(ws, imagLocal, imagBase, outputOffset, batchRows, rowElements, packedStride, batchElements);
    SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
    if (tiling.useThreeM == 0U) {
        return;
    }
    if (tiling.directHermitianPath != 0U) {
        Muls(realLocal, imagLocal, -1.0f, batchElements);
    } else {
        Add(realLocal, realLocal, imagLocal, batchElements);
    }
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    Cher2kStoreNormalPlane(ws, realLocal, sumBase, outputOffset, batchRows, rowElements, packedStride, batchElements);
    SetFlag<HardEvent::MTE3_V>(eventMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3ToV);
}

__aicore__ inline void Cher2kPreprocessNormalChunk(
    GlobalTensor<float>& input, GlobalTensor<float>& ws, LocalTensor<float>& aosLocal, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, const Cher2kTilingData& tiling, uint32_t firstRow, uint32_t batchRows,
    uint32_t offset, uint32_t rowElements, uint32_t physicalRows, uint32_t physicalCols, uint32_t packedStride,
    uint32_t ld, bool conjugate, uint64_t realBase, uint64_t imagBase, uint64_t sumBase, event_t eventMte2ToV,
    event_t eventVToMte3, event_t eventMte3ToV)
{
    const uint32_t batchElements = batchRows * rowElements;
    const uint32_t validRows = firstRow < physicalCols ? min(batchRows, physicalCols - firstRow) : 0U;
    const uint32_t validElements = offset < physicalRows ? min(rowElements, physicalRows - offset) : 0U;
    if (validRows != batchRows || validElements != rowElements) {
        Duplicate(aosLocal, 0.0f, batchElements * 2U);
        PipeBarrier<PIPE_ALL>();
    }
    if (firstRow < physicalCols && offset < physicalRows) {
        const uint32_t blockLen = validElements * 2U * sizeof(float);
        const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
        const uint32_t rightPadding = (alignedBlockLen - blockLen) / sizeof(float);
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validRows), blockLen,
            static_cast<uint32_t>((ld - validElements) * 2U * sizeof(float)),
            static_cast<uint32_t>((rowElements * 2U * sizeof(float) - alignedBlockLen) / 32U), 0U};
        const DataCopyPadExtParams<float> padParams{rightPadding != 0U, 0U, static_cast<uint8_t>(rightPadding), 0.0f};
        const uint64_t inputOffset = (static_cast<uint64_t>(firstRow) * ld + offset) * 2ULL;
        DataCopyPad(aosLocal, input[inputOffset], copyParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(eventMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2ToV);
    const uint16_t repeats = static_cast<uint16_t>((batchElements * 2U + 63U) / 64U);
    uint64_t reservedCount = 0ULL;
    GatherMask<float>(realLocal, aosLocal, 1U, false, 0U, {1U, repeats, 8U, 8U}, reservedCount);
    GatherMask<float>(imagLocal, aosLocal, 2U, false, 0U, {1U, repeats, 8U, 8U}, reservedCount);
    if (conjugate) {
        Muls(imagLocal, imagLocal, -1.0f, batchElements);
    }
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    const uint64_t outputOffset = static_cast<uint64_t>(firstRow) * packedStride + offset;
    Cher2kStoreNormalChunk(
        ws, realLocal, imagLocal, tiling, batchRows, rowElements, packedStride, batchElements, outputOffset, realBase,
        imagBase, sumBase, eventVToMte3, eventMte3ToV);
}

__aicore__ inline void Cher2kPreprocessNormalBatches(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& ws, LocalTensor<float>& aosLocal,
    LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal, const Cher2kTilingData& tiling, uint32_t blockIdx,
    uint32_t blockNum, uint32_t physicalRows, uint32_t physicalCols, uint32_t packedStride, uint32_t packedRows,
    uint64_t arBase, uint64_t aiBase, uint64_t brBase, uint64_t biBase, uint64_t sumABase, uint64_t sumBBase,
    event_t eventMte2ToV, event_t eventVToMte3, event_t eventMte3ToV)
{
    const uint32_t safePackedStride = packedStride == 0U ? 1U : packedStride;
    const uint32_t rowsPerBatch =
        packedStride <= CHER2K_ROW_BATCH_ELEMENTS ? CHER2K_ROW_BATCH_ELEMENTS / safePackedStride : 1U;
    const uint32_t rowBatchCount = (packedRows + rowsPerBatch - 1U) / rowsPerBatch;
    const uint32_t totalRowBatches = rowBatchCount * 2U;
    for (uint32_t batchIndex = blockIdx; batchIndex < totalRowBatches; batchIndex += blockNum) {
        const bool packB = batchIndex >= rowBatchCount;
        const uint32_t matrixBatch = packB ? batchIndex - rowBatchCount : batchIndex;
        const uint32_t firstRow = matrixBatch * rowsPerBatch;
        const uint32_t batchRows = min(rowsPerBatch, packedRows - firstRow);
        GlobalTensor<float>& input = packB ? bG : aG;
        const uint32_t ld = packB ? tiling.ldb : tiling.lda;
        const uint64_t realBase = packB ? brBase : arBase;
        const uint64_t imagBase = packB ? biBase : aiBase;
        const uint64_t sumBase = packB ? sumBBase : sumABase;
        const bool conjugate = tiling.trans == ACLBLAS_OP_N ? packB : !packB;
        const uint32_t elementsPerRow =
            min(packedStride, CHER2K_ROW_BATCH_ELEMENTS / (batchRows == 0U ? 1U : batchRows));
        for (uint32_t offset = 0U; offset < packedStride; offset += elementsPerRow) {
            const uint32_t rowElements = min(elementsPerRow, packedStride - offset);
            Cher2kPreprocessNormalChunk(
                input, ws, aosLocal, realLocal, imagLocal, tiling, firstRow, batchRows, offset, rowElements,
                physicalRows, physicalCols, packedStride, ld, conjugate, realBase, imagBase, sumBase, eventMte2ToV,
                eventVToMte3, eventMte3ToV);
        }
    }
}
__aicore__ inline void Cher2kInitTransposeOffsets(
    GM_ADDR sharedOffsets, TPipe& pipe, TBuf<TPosition::VECCALC>& transRealBuf, TBuf<TPosition::VECCALC>& transImagBuf,
    TBuf<TPosition::VECCALC>& transOffsetBuf, LocalTensor<float>& transRealLocal, LocalTensor<float>& transImagLocal,
    LocalTensor<uint32_t>& transOffsetLocal, LocalTensor<int32_t>& transOffsetI32,
    LocalTensor<uint32_t>& transOffsetImagLocal)
{
    pipe.InitBuffer(transRealBuf, CHER2K_TRANSPOSE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(transImagBuf, CHER2K_TRANSPOSE_ELEMENTS * sizeof(float));
    pipe.InitBuffer(
        transOffsetBuf, CHER2K_TRANSPOSE_ELEMENTS * (CHER2K_DIRECT_AOS_GATHER_EXPERIMENT ? 2U : 1U) * sizeof(uint32_t));
    transRealLocal = transRealBuf.Get<float>();
    transImagLocal = transImagBuf.Get<float>();
    transOffsetLocal = transOffsetBuf.Get<uint32_t>();
    transOffsetI32 = transOffsetBuf.Get<int32_t>();
    transOffsetImagLocal = transOffsetLocal[(CHER2K_DIRECT_AOS_GATHER_EXPERIMENT ? CHER2K_TRANSPOSE_ELEMENTS : 0U)];
    if (sharedOffsets != nullptr) {
        GlobalTensor<uint32_t> offsetsG;
        offsetsG.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(sharedOffsets), CHER2K_INTERLEAVE_OFFSET_COUNT);
        DataCopy(transOffsetLocal, offsetsG[CHER2K_TRANSPOSE_OFFSET], CHER2K_TRANSPOSE_OFFSET_COUNT);
    } else if constexpr (CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        for (uint32_t index = 0U; index < CHER2K_TRANSPOSE_ELEMENTS; ++index) {
            const uint32_t sourceIndex =
                (index % CHER2K_TRANSPOSE_COLS) * CHER2K_TRANSPOSE_ROWS + index / CHER2K_TRANSPOSE_COLS;
            transOffsetLocal.SetValue(index, sourceIndex * 2U * sizeof(float));
            transOffsetImagLocal.SetValue(index, sourceIndex * 2U * sizeof(float) + sizeof(float));
        }
    } else {
        CreateVecIndex(transOffsetI32, 0, CHER2K_TRANSPOSE_COLS);
        PipeBarrier<PIPE_V>();
        Muls(
            transOffsetI32, transOffsetI32, static_cast<int32_t>(CHER2K_TRANSPOSE_ROWS * sizeof(float)),
            CHER2K_TRANSPOSE_COLS);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1U; row < CHER2K_TRANSPOSE_ROWS; ++row) {
            Adds(
                transOffsetI32[row * CHER2K_TRANSPOSE_COLS], transOffsetI32, static_cast<int32_t>(row * sizeof(float)),
                CHER2K_TRANSPOSE_COLS);
        }
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kStoreDirectOpcTranspose(
    GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& transRealLocal,
    LocalTensor<float>& transImagLocal, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    LocalTensor<uint32_t>& transOffsetLocal, uint32_t matrix, uint32_t rowBase, uint32_t colBase, event_t eventVToMte3)
{
    const uint32_t outputRows = min(CHER2K_TRANSPOSE_ROWS, tiling.kAligned - rowBase);
    const uint32_t packedCols = min(CHER2K_TRANSPOSE_COLS, tiling.nAligned - colBase);
    TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
        transRealLocal, realLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
        transImagLocal, imagLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    const uint32_t rightDstStride = static_cast<uint32_t>((tiling.nAligned - packedCols) * sizeof(float));
    const DataCopyExtParams outputParams{
        static_cast<uint16_t>(outputRows), static_cast<uint32_t>(packedCols * sizeof(float)), 0U, rightDstStride, 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
    const uint64_t realBase = matrix == 0U ? DirectOpcRightAr(tiling.nAligned, tiling.kAligned) :
                                             DirectOpcRightBr(tiling.nAligned, tiling.kAligned);
    const uint64_t imagBase = matrix == 0U ? DirectOpcRightAi(tiling.nAligned, tiling.kAligned) :
                                             DirectOpcRightBi(tiling.nAligned, tiling.kAligned);
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    DataCopyPad(ws[realBase + outputOffset], transRealLocal, outputParams);
    DataCopyPad(ws[imagBase + outputOffset], transImagLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
    Muls(transRealLocal, transRealLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    const uint64_t negativeBase = matrix == 0U ? DirectOpcLeftNegAr(tiling.nAligned, tiling.kAligned) :
                                                 DirectOpcLeftNegBr(tiling.nAligned, tiling.kAligned);
    DataCopyPad(ws[negativeBase + outputOffset], transRealLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kStorePackedOpcLeft(
    GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, uint64_t arBase, uint64_t aiBase, uint64_t sumABase, uint32_t rowBase,
    uint32_t colBase, uint32_t validCols, event_t eventVToMte3)
{
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    const uint32_t outputRows = min(CHER2K_TRANSPOSE_ROWS, tiling.kAligned - rowBase);
    const uint32_t dstStride = static_cast<uint32_t>((tiling.kAligned - outputRows) * sizeof(float));
    const DataCopyExtParams outputParams{
        static_cast<uint16_t>(validCols), static_cast<uint32_t>(outputRows * sizeof(float)), 0U, dstStride, 0U};
    const uint64_t outputOffset = static_cast<uint64_t>(colBase) * tiling.kAligned + rowBase;
    DataCopyPad(ws[arBase + outputOffset], realLocal, outputParams);
    DataCopyPad(ws[aiBase + outputOffset], imagLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
    if (tiling.useThreeM == 0U) {
        return;
    }
    Add(realLocal, realLocal, imagLocal, CHER2K_TRANSPOSE_ELEMENTS);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(eventVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    DataCopyPad(ws[sumABase + outputOffset], realLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kStoreTransposePlanes(
    GlobalTensor<float>& ws, const Cher2kTilingData& tiling, LocalTensor<float>& transRealLocal,
    LocalTensor<float>& transImagLocal, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    LocalTensor<uint32_t>& transOffsetLocal, uint64_t realBase, uint64_t imagBase, uint64_t sumBase, uint32_t rowBase,
    uint32_t colBase, bool directOpc)
{
    if constexpr (!CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transRealLocal, realLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transImagLocal, imagLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    PipeBarrier<PIPE_ALL>();
    const uint32_t dstStride = static_cast<uint32_t>((tiling.nAligned - CHER2K_TRANSPOSE_COLS) * sizeof(float));
    const uint32_t outputRows = min(CHER2K_TRANSPOSE_ROWS, tiling.kAligned - rowBase);
    const DataCopyExtParams outputParams{
        static_cast<uint16_t>(outputRows), static_cast<uint32_t>(CHER2K_TRANSPOSE_COLS * sizeof(float)), 0U, dstStride,
        0U};
    const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
    DataCopyPad(ws[realBase + outputOffset], transRealLocal, outputParams);
    DataCopyPad(ws[imagBase + outputOffset], transImagLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
    if (tiling.useThreeM == 0U) {
        return;
    }
    if (directOpc || !CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        Add(realLocal, realLocal, imagLocal, CHER2K_TRANSPOSE_ELEMENTS);
    } else {
        Add(transRealLocal, transRealLocal, transImagLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    if constexpr (CHER2K_PREPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
    if constexpr (!CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transRealLocal, realLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(ws[sumBase + outputOffset], transRealLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kPreprocessVectorBarrier()
{
    if constexpr (CHER2K_PREPROCESS_VECTOR_BARRIER_EXPERIMENT) {
        PipeBarrier<PIPE_V>();
    } else {
        PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void Cher2kCopyTransposeAosTile(
    GlobalTensor<float>& input, LocalTensor<float>& destination, uint32_t ld, uint32_t rowBase, uint32_t colBase,
    uint32_t validRows, uint32_t validCols)
{
    if (validRows != CHER2K_TRANSPOSE_ROWS || validCols != CHER2K_TRANSPOSE_COLS) {
        Duplicate(destination, 0.0f, CHER2K_TRANSPOSE_ELEMENTS * 2U);
        PipeBarrier<PIPE_ALL>();
    }
    if (validRows == 0U || validCols == 0U) {
        return;
    }
    const uint32_t blockLen = validRows * 2U * sizeof(float);
    const uint32_t alignedBlockLen = (blockLen + 31U) & ~31U;
    const uint32_t rightPadding = (alignedBlockLen - blockLen) / sizeof(float);
    const uint32_t srcStride = (ld - validRows) * 2U * sizeof(float);
    const uint32_t dstStride = (CHER2K_TRANSPOSE_ROWS * 2U * sizeof(float) - alignedBlockLen) / 32U;
    const DataCopyExtParams copyParams{static_cast<uint16_t>(validCols), blockLen, srcStride, dstStride, 0U};
    const DataCopyPadExtParams<float> padParams{rightPadding != 0U, 0U, static_cast<uint8_t>(rightPadding), 0.0f};
    const uint64_t inputOffset = (static_cast<uint64_t>(colBase) * ld + rowBase) * 2ULL;
    DataCopyPad(destination, input[inputOffset], copyParams, padParams);
}

__aicore__ inline void Cher2kLoadTransposeTile(
    GlobalTensor<float>& input, LocalTensor<float>& aosLocal, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, LocalTensor<float>& transRealLocal, LocalTensor<float>& transImagLocal,
    LocalTensor<uint32_t>& transOffsetLocal, LocalTensor<uint32_t>& transOffsetImagLocal, uint32_t ld, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols, uint32_t matrix, bool directOutputOpc)
{
    Cher2kCopyTransposeAosTile(input, aosLocal, ld, rowBase, colBase, validRows, validCols);
    PipeBarrier<PIPE_ALL>();
    if (validRows == 0U || validCols == 0U) {
        Duplicate(realLocal, 0.0f, CHER2K_TRANSPOSE_ELEMENTS);
        Duplicate(imagLocal, 0.0f, CHER2K_TRANSPOSE_ELEMENTS);
    } else if constexpr (CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        Gather<float>(transRealLocal, aosLocal, transOffsetLocal, 0U, CHER2K_TRANSPOSE_ELEMENTS);
        Gather<float>(transImagLocal, aosLocal, transOffsetImagLocal, 0U, CHER2K_TRANSPOSE_ELEMENTS);
        if (matrix == 0U) {
            Muls(transImagLocal, transImagLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
        }
    } else {
        uint64_t reservedCount = 0ULL;
        GatherMask<float>(
            realLocal, aosLocal, 1U, false, 0U, {1U, CHER2K_TRANSPOSE_ELEMENTS * 2U / 64U, 8U, 8U}, reservedCount);
        GatherMask<float>(
            imagLocal, aosLocal, 2U, false, 0U, {1U, CHER2K_TRANSPOSE_ELEMENTS * 2U / 64U, 8U, 8U}, reservedCount);
        if (matrix == 0U && !directOutputOpc) {
            Muls(imagLocal, imagLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
        }
    }
    Cher2kPreprocessVectorBarrier();
}

__aicore__ inline void Cher2kProcessTransposeMatrix(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& ws, LocalTensor<float>& aosLocal,
    LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal, LocalTensor<float>& transRealLocal,
    LocalTensor<float>& transImagLocal, LocalTensor<uint32_t>& transOffsetLocal,
    LocalTensor<uint32_t>& transOffsetImagLocal, const Cher2kTilingData& tiling, uint32_t matrix, uint32_t rowBase,
    uint32_t colBase, uint32_t validRows, uint32_t validCols, bool directOpc, bool directOutputOpc, uint64_t arBase,
    uint64_t aiBase, uint64_t sumABase, uint64_t brBase, uint64_t biBase, uint64_t sumBBase, event_t eventVToMte3)
{
    GlobalTensor<float>& input = matrix == 0U ? aG : bG;
    const uint32_t ld = matrix == 0U ? tiling.lda : tiling.ldb;
    const uint64_t realBase = matrix == 0U ? arBase : brBase;
    const uint64_t imagBase = matrix == 0U ? aiBase : biBase;
    Cher2kLoadTransposeTile(
        input, aosLocal, realLocal, imagLocal, transRealLocal, transImagLocal, transOffsetLocal, transOffsetImagLocal,
        ld, rowBase, colBase, validRows, validCols, matrix, directOutputOpc);
    if (directOutputOpc) {
        Cher2kStoreDirectOpcTranspose(
            ws, tiling, transRealLocal, transImagLocal, realLocal, imagLocal, transOffsetLocal, matrix, rowBase,
            colBase, eventVToMte3);
        return;
    }
    if (directOpc && matrix == 0U) {
        Cher2kStorePackedOpcLeft(
            ws, tiling, realLocal, imagLocal, arBase, aiBase, sumABase, rowBase, colBase, validCols, eventVToMte3);
        return;
    }
    const uint64_t sumBase = matrix == 0U ? sumABase : sumBBase;
    Cher2kStoreTransposePlanes(
        ws, tiling, transRealLocal, transImagLocal, realLocal, imagLocal, transOffsetLocal, realBase, imagBase, sumBase,
        rowBase, colBase, directOpc);
}

__aicore__ inline void Cher2kRunTransposeScheduled(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& ws, LocalTensor<float>& aosLocal,
    LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal, LocalTensor<float>& transRealLocal,
    LocalTensor<float>& transImagLocal, LocalTensor<uint32_t>& transOffsetLocal,
    LocalTensor<uint32_t>& transOffsetImagLocal, const Cher2kTilingData& tiling, uint32_t blockIdx, uint32_t blockNum,
    uint32_t rowTileCount, uint32_t colTileCount, bool directOpc, bool directOutputOpc, uint64_t arBase,
    uint64_t aiBase, uint64_t sumABase, uint64_t brBase, uint64_t biBase, uint64_t sumBBase, event_t eventVToMte3)
{
    const uint32_t scheduledTileCount = directOpc ? rowTileCount * colTileCount : rowTileCount;
    const uint32_t scheduledTaskCount = directOutputOpc ? scheduledTileCount * 2U : scheduledTileCount;
    for (uint32_t task = blockIdx; task < scheduledTaskCount; task += blockNum) {
        const uint32_t scheduledTile = directOutputOpc ? task / 2U : task;
        const uint32_t rowTile = directOpc ? scheduledTile / (colTileCount == 0U ? 1U : colTileCount) : scheduledTile;
        const uint32_t rowBase = rowTile * CHER2K_TRANSPOSE_ROWS;
        const uint32_t validRows = rowBase < tiling.k ? min(CHER2K_TRANSPOSE_ROWS, tiling.k - rowBase) : 0U;
        const uint32_t firstColTile = directOpc ? scheduledTile - rowTile * colTileCount : 0U;
        const uint32_t endColTile = directOpc ? firstColTile + 1U : colTileCount;
        for (uint32_t colTile = firstColTile; colTile < endColTile; ++colTile) {
            const uint32_t colBase = colTile * CHER2K_TRANSPOSE_COLS;
            const uint32_t validCols = colBase < tiling.n ? min(CHER2K_TRANSPOSE_COLS, tiling.n - colBase) : 0U;
            const uint32_t matrixBegin = directOutputOpc ? task - scheduledTile * 2U : 0U;
            const uint32_t matrixEnd = directOutputOpc ? matrixBegin + 1U : 2U;
            for (uint32_t matrix = matrixBegin; matrix < matrixEnd; ++matrix) {
                Cher2kProcessTransposeMatrix(
                    aG, bG, ws, aosLocal, realLocal, imagLocal, transRealLocal, transImagLocal, transOffsetLocal,
                    transOffsetImagLocal, tiling, matrix, rowBase, colBase, validRows, validCols, directOpc,
                    directOutputOpc, arBase, aiBase, sumABase, brBase, biBase, sumBBase, eventVToMte3);
            }
        }
    }
}

__aicore__ inline void Cher2kCopyFusedAos(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, LocalTensor<float>& aosLocal, LocalTensor<float>& aosSecondLocal,
    const Cher2kTilingData& tiling, uint32_t rowBase, uint32_t colBase, uint32_t validRows, uint32_t validCols)
{
    LocalTensor<float> matrixAos[2] = {aosLocal, aosSecondLocal};
    for (uint32_t matrix = 0U; matrix < 2U; ++matrix) {
        GlobalTensor<float>& input = matrix == 0U ? aG : bG;
        const uint32_t ld = matrix == 0U ? tiling.lda : tiling.ldb;
        Cher2kCopyTransposeAosTile(input, matrixAos[matrix], ld, rowBase, colBase, validRows, validCols);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kGatherFusedPlanes(
    LocalTensor<float>& matrixAos, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    LocalTensor<float>& transRealLocal, LocalTensor<float>& transImagLocal, LocalTensor<uint32_t>& transOffsetLocal,
    LocalTensor<uint32_t>& transOffsetImagLocal, uint32_t matrix)
{
    if constexpr (CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        Gather<float>(transRealLocal, matrixAos, transOffsetLocal, 0U, CHER2K_TRANSPOSE_ELEMENTS);
        Gather<float>(transImagLocal, matrixAos, transOffsetImagLocal, 0U, CHER2K_TRANSPOSE_ELEMENTS);
        if (matrix == 0U) {
            Muls(transImagLocal, transImagLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
        }
    } else {
        uint64_t reservedCount = 0ULL;
        GatherMask<float>(
            realLocal, matrixAos, 1U, false, 0U, {1U, CHER2K_TRANSPOSE_ELEMENTS * 2U / 64U, 8U, 8U}, reservedCount);
        GatherMask<float>(
            imagLocal, matrixAos, 2U, false, 0U, {1U, CHER2K_TRANSPOSE_ELEMENTS * 2U / 64U, 8U, 8U}, reservedCount);
        if (matrix == 0U) {
            Muls(imagLocal, imagLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
        }
    }
}

__aicore__ inline void Cher2kProcessFusedOutput(
    GlobalTensor<float>& ws, LocalTensor<float>& matrixAos, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, LocalTensor<float>& transRealLocal, LocalTensor<float>& transImagLocal,
    LocalTensor<uint32_t>& transOffsetLocal, LocalTensor<uint32_t>& transOffsetImagLocal,
    const Cher2kTilingData& tiling, const DataCopyExtParams& outputParams, uint64_t outputOffset, uint64_t realBase,
    uint64_t imagBase, uint64_t sumBase, uint32_t matrix)
{
    Cher2kGatherFusedPlanes(
        matrixAos, realLocal, imagLocal, transRealLocal, transImagLocal, transOffsetLocal, transOffsetImagLocal,
        matrix);
    Cher2kPreprocessVectorBarrier();
    if constexpr (!CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transRealLocal, realLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transImagLocal, imagLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(ws[realBase + outputOffset], transRealLocal, outputParams);
    DataCopyPad(ws[imagBase + outputOffset], transImagLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
    if (tiling.useThreeM == 0U) {
        return;
    }
    if (tiling.directHermitianPath != 0U) {
        Muls(transRealLocal, transImagLocal, -1.0f, CHER2K_TRANSPOSE_ELEMENTS);
    } else if constexpr (CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        Add(transRealLocal, transRealLocal, transImagLocal, CHER2K_TRANSPOSE_ELEMENTS);
    } else {
        Add(realLocal, realLocal, imagLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    Cher2kPreprocessVectorBarrier();
    if constexpr (!CHER2K_DIRECT_AOS_GATHER_EXPERIMENT) {
        TransposePreprocessTile<CHER2K_VECTOR_TRANSPOSE_EXPERIMENT>(
            transRealLocal, realLocal, transOffsetLocal, CHER2K_TRANSPOSE_ELEMENTS);
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(ws[sumBase + outputOffset], transRealLocal, outputParams);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Cher2kRunFusedTranspose(
    GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& ws, LocalTensor<float>& aosLocal,
    LocalTensor<float>& aosSecondLocal, LocalTensor<float>& realLocal, LocalTensor<float>& imagLocal,
    LocalTensor<float>& transRealLocal, LocalTensor<float>& transImagLocal, LocalTensor<uint32_t>& transOffsetLocal,
    LocalTensor<uint32_t>& transOffsetImagLocal, const Cher2kTilingData& tiling, uint32_t blockIdx, uint32_t blockNum,
    uint32_t rowTileCount, uint32_t colTileCount, uint64_t arBase, uint64_t aiBase, uint64_t sumABase, uint64_t brBase,
    uint64_t biBase, uint64_t sumBBase)
{
    for (uint32_t rowTile = blockIdx; rowTile < rowTileCount; rowTile += blockNum) {
        const uint32_t rowBase = rowTile * CHER2K_TRANSPOSE_ROWS;
        const uint32_t validRows = rowBase < tiling.k ? min(CHER2K_TRANSPOSE_ROWS, tiling.k - rowBase) : 0U;
        for (uint32_t colTile = 0U; colTile < colTileCount; ++colTile) {
            const uint32_t colBase = colTile * CHER2K_TRANSPOSE_COLS;
            const uint32_t validCols = colBase < tiling.n ? min(CHER2K_TRANSPOSE_COLS, tiling.n - colBase) : 0U;
            Cher2kCopyFusedAos(aG, bG, aosLocal, aosSecondLocal, tiling, rowBase, colBase, validRows, validCols);
            const uint32_t dstStride = static_cast<uint32_t>((tiling.nAligned - CHER2K_TRANSPOSE_COLS) * sizeof(float));
            const uint32_t outputRows = min(CHER2K_TRANSPOSE_ROWS, tiling.kAligned - rowBase);
            DataCopyExtParams outputParams{
                static_cast<uint16_t>(outputRows), static_cast<uint32_t>(CHER2K_TRANSPOSE_COLS * sizeof(float)), 0U,
                dstStride, 0U};
            const uint64_t outputOffset = static_cast<uint64_t>(rowBase) * tiling.nAligned + colBase;
            LocalTensor<float> matrixAos[2] = {aosLocal, aosSecondLocal};
            const uint64_t realBases[2] = {arBase, brBase};
            const uint64_t imagBases[2] = {aiBase, biBase};
            const uint64_t sumBases[2] = {sumABase, sumBBase};
            for (uint32_t matrix = 0U; matrix < 2U; ++matrix) {
                Cher2kProcessFusedOutput(
                    ws, matrixAos[matrix], realLocal, imagLocal, transRealLocal, transImagLocal, transOffsetLocal,
                    transOffsetImagLocal, tiling, outputParams, outputOffset, realBases[matrix], imagBases[matrix],
                    sumBases[matrix], matrix);
            }
        }
    }
}

/** Transpose and direct OP_C producer path. */
__aicore__ inline void Cher2kPreprocessTransposePath(
    GM_ADDR sharedOffsets, GlobalTensor<float>& aG, GlobalTensor<float>& bG, GlobalTensor<float>& ws,
    LocalTensor<float>& aosLocal, LocalTensor<float>& aosSecondLocal, LocalTensor<float>& realLocal,
    LocalTensor<float>& imagLocal, const Cher2kTilingData& tiling, uint32_t blockIdx, uint32_t blockNum, bool directOpc,
    bool directOutputOpc, uint64_t arBase, uint64_t aiBase, uint64_t sumABase, uint64_t brBase, uint64_t biBase,
    uint64_t sumBBase, TPipe& pipe, event_t eventVToMte3)
{
    TBuf<TPosition::VECCALC> transRealBuf;
    TBuf<TPosition::VECCALC> transImagBuf;
    TBuf<TPosition::VECCALC> transOffsetBuf;
    LocalTensor<float> transRealLocal;
    LocalTensor<float> transImagLocal;
    LocalTensor<uint32_t> transOffsetLocal;
    LocalTensor<int32_t> transOffsetI32;
    LocalTensor<uint32_t> transOffsetImagLocal;
    Cher2kInitTransposeOffsets(
        sharedOffsets, pipe, transRealBuf, transImagBuf, transOffsetBuf, transRealLocal, transImagLocal,
        transOffsetLocal, transOffsetI32, transOffsetImagLocal);
    const uint32_t rowTileCount = (tiling.kAligned + CHER2K_TRANSPOSE_ROWS - 1U) / CHER2K_TRANSPOSE_ROWS;
    const uint32_t colTileCount = (tiling.nAligned + CHER2K_TRANSPOSE_COLS - 1U) / CHER2K_TRANSPOSE_COLS;
    if constexpr (CHER2K_FUSE_OPC_A_B_LOADS_EXPERIMENT) {
        Cher2kRunFusedTranspose(
            aG, bG, ws, aosLocal, aosSecondLocal, realLocal, imagLocal, transRealLocal, transImagLocal,
            transOffsetLocal, transOffsetImagLocal, tiling, blockIdx, blockNum, rowTileCount, colTileCount, arBase,
            aiBase, sumABase, brBase, biBase, sumBBase);
        if constexpr (!CHER2K_SKIP_WORKSPACE_CACHE_CLEAN_EXPERIMENT) {
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
    Cher2kRunTransposeScheduled(
        aG, bG, ws, aosLocal, realLocal, imagLocal, transRealLocal, transImagLocal, transOffsetLocal,
        transOffsetImagLocal, tiling, blockIdx, blockNum, rowTileCount, colTileCount, directOpc, directOutputOpc,
        arBase, aiBase, sumABase, brBase, biBase, sumBBase, eventVToMte3);
    if constexpr (!CHER2K_SKIP_WORKSPACE_CACHE_CLEAN_EXPERIMENT) {
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ws);
    }
    PipeBarrier<PIPE_ALL>();
}
/** Common physical MMAD implementation for the regular Cube path. */
class Cher2kKernelLayout {
public:
    __aicore__ inline void Init(
        __gm__ float* ar, __gm__ float* ai, __gm__ float* br, __gm__ float* bi, __gm__ float* sumA, __gm__ float* sumB,
        __gm__ float* rr, __gm__ float* ii, __gm__ float* ri, __gm__ float* ir, uint32_t n, uint32_t k, bool transA,
        bool transB, bool useThreeM, uint32_t uplo = ACLBLAS_UPPER, bool transposeOutput = false)
    {
        const uint64_t nk = static_cast<uint64_t>(n) * k;
        const uint64_t nn = static_cast<uint64_t>(n) * n;
        // The Q^T fast path swaps the logical operands while retaining the
        // same planar workspace.  All downstream product scheduling remains
        // unchanged; only the producer's matrix orientation changes.
        if (transposeOutput) {
            ar_.SetGlobalBuffer(br, nk);
            ai_.SetGlobalBuffer(bi, nk);
            br_.SetGlobalBuffer(ar, nk);
            bi_.SetGlobalBuffer(ai, nk);
        } else {
            ar_.SetGlobalBuffer(ar, nk);
            ai_.SetGlobalBuffer(ai, nk);
            br_.SetGlobalBuffer(br, nk);
            bi_.SetGlobalBuffer(bi, nk);
        }
        if (useThreeM) {
            if (transposeOutput) {
                sumA_.SetGlobalBuffer(sumB, nk);
                sumB_.SetGlobalBuffer(sumA, nk);
            } else {
                sumA_.SetGlobalBuffer(sumA, nk);
                sumB_.SetGlobalBuffer(sumB, nk);
            }
        }
        rr_.SetGlobalBuffer(rr, nn);
        ii_.SetGlobalBuffer(ii, nn);
        ri_.SetGlobalBuffer(ri, nn);
        ir_.SetGlobalBuffer(ir, nn);
        n_ = n;
        k_ = k;
        transA_ = transA;
        transB_ = transB;
        tilingUplo_ = uplo;

        InitCopyParams();
        InitLoadParams();
        InitPanelParams();
        InitMmadParams();
    }

    // Known-correct single-C-tile path. Keep this as the active route while
    // the paired-C diagnostic path below is isolated and investigated.
    // Product count is a compile-time parameter in the hot loop. The host
    // still selects 3M vs 4M at runtime, but the selected stream no longer
    // pays a product-count branch and dispatch test for every task.
    template <bool USE_THREE_M>
    __aicore__ inline void ProcessImpl(uint32_t taskIndex, uint32_t taskStride)
    {
        constexpr uint32_t aSize = CHER2K_CUBE_M * CHER2K_CUBE_K;
        constexpr uint32_t bSize = CHER2K_CUBE_K * CHER2K_CUBE_N;
        constexpr uint32_t aL1Size = CHER2K_CUBE_M * CHER2K_L1_PANEL_K;
        constexpr uint32_t bL1Size = CHER2K_L1_PANEL_K * CHER2K_CUBE_N;
        constexpr uint32_t cSize = CHER2K_CUBE_M * CHER2K_CUBE_N;
        const uint32_t mTiles = n_ / CHER2K_CUBE_M;
        const uint32_t nTiles = n_ / CHER2K_CUBE_N;
        const uint64_t tileCount = static_cast<uint64_t>(mTiles) * nTiles;
        // The unit-flag protocol is currently qualified only for the full
        // PF1002 geometry.  Other MIX shapes retain the legacy event ring
        // until their tail-panel and precision contracts are measured.
        constexpr uint32_t productCount = USE_THREE_M ? 3U : 4U;
        const uint64_t taskCount = tileCount * productCount;
        constexpr uint32_t l1BufferCount = CHER2K_L1_PINGPONG_EXPERIMENT ? 2U : 1U;
        LocalTensor<float> a1{TPosition::A1, 0U, aL1Size * l1BufferCount};
        LocalTensor<float> b1(
            TPosition::B1, static_cast<uint64_t>(aL1Size * l1BufferCount) * sizeof(float), bL1Size * l1BufferCount);
        LocalTensor<float> a2{TPosition::A2, 0U, aSize};
        LocalTensor<float> b2{TPosition::B2, 0U, bSize};
        LocalTensor<float> c1{TPosition::CO1, 0U, cSize};
        BeginProductStreamEvents();
        if (ProcessSpecialProductSchedule<USE_THREE_M>(a1, b1, a2, b2, c1, tileCount, nTiles, taskIndex, taskStride)) {
            return;
        }
        uint64_t taskBegin = taskIndex;
        uint64_t taskEnd = taskCount;
        uint64_t taskStep = taskStride;
        if constexpr (CHER2K_CONTIGUOUS_TASK_EXPERIMENT) {
            const uint32_t safeTaskStride = taskStride == 0U ? 1U : taskStride;
            const uint64_t tasksPerCore = (taskCount + safeTaskStride - 1U) / safeTaskStride;
            taskBegin = static_cast<uint64_t>(taskIndex) * tasksPerCore;
            taskEnd = min(taskCount, taskBegin + tasksPerCore);
            taskStep = 1U;
        }
        // The normal stream advances by a fixed core stride. Initialize the
        // first task with division, then carry product/tile/row/column state
        // arithmetically. This removes two 64-bit div/mod operations from the
        // hot loop while preserving the exact round-robin task order.
        ProcessTileStream<USE_THREE_M>(a1, b1, a2, b2, c1, taskBegin, taskEnd, taskStep, nTiles, productCount);
        EndProductStreamEvents();
    }

    __aicore__ inline void Process(uint32_t taskIndex, uint32_t taskStride, bool useThreeM)
    {
        // The production scheduler assigns one output tile/product to a
        // block.  Paired-C and MIX experiments are retired; keeping this
        // dispatch direct makes the active lifecycle explicit.
        if (useThreeM) {
            ProcessImpl<true>(taskIndex, taskStride);
        } else {
            ProcessImpl<false>(taskIndex, taskStride);
        }
    }

    // Compute H = A*B^H + B*A^H directly for the low-rank OP_N path.  A
    // 128x128 tile leaves room for two CO1 accumulators, so real(H) and
    // imag(H^T) stay resident together while K is the outer loop.  The ten
    // unique signed source panels are copied to L1 once per K slice and reused
    // by all eight MMAD terms; the old 256x128 route reloaded those panels when
    // switching from real to imaginary output.
    struct Cher2kDirectHermitianState {
        static constexpr uint32_t directM = 128U;
        static constexpr uint32_t directN = 128U;
        static constexpr uint32_t directK = CHER2K_CUBE_K;
        static constexpr uint32_t aSize = directM * directK;
        static constexpr uint32_t bSize = directK * directN;
        static constexpr uint32_t cSize = directM * directN;
        static constexpr uint32_t signedPanelCount = 5U;
        static constexpr uint32_t aSliceSlots = 2U;
        static constexpr uint32_t aPlaneCapacity = aSliceSlots * aSize;
        LocalTensor<float> aAr1;
        LocalTensor<float> aAi1;
        LocalTensor<float> aBr1;
        LocalTensor<float> aBi1;
        LocalTensor<float> aNegAi1;
        LocalTensor<float> bBr1;
        LocalTensor<float> bPosBi1;
        LocalTensor<float> bAr1;
        LocalTensor<float> bNegAi1;
        LocalTensor<float> bAi1;
        LocalTensor<float> bExtra1;
        LocalTensor<float> a2;
        LocalTensor<float> b2;
        LocalTensor<float> realC1;
        LocalTensor<float> imagC1;
        Nd2NzParams aCopy{};
        Nd2NzParams bNzParams{};
        LoadData3DParamsV2<float> aLoad{};
        LoadData3DParamsV2<float> bLoad{};
        MmadParams first{};
        MmadParams accum{};
        bool directOpc = false;

        __aicore__ inline void Init(bool opc, uint32_t n, uint32_t k)
        {
            directOpc = opc;
            const uint64_t bL1Base = static_cast<uint64_t>(signedPanelCount) * aPlaneCapacity * sizeof(float);
            aAr1 = LocalTensor<float>(TPosition::A1, 0ULL * aPlaneCapacity * sizeof(float), aPlaneCapacity);
            aAi1 = LocalTensor<float>(TPosition::A1, 1ULL * aPlaneCapacity * sizeof(float), aPlaneCapacity);
            aBr1 = LocalTensor<float>(TPosition::A1, 2ULL * aPlaneCapacity * sizeof(float), aPlaneCapacity);
            aBi1 = LocalTensor<float>(TPosition::A1, 3ULL * aPlaneCapacity * sizeof(float), aPlaneCapacity);
            aNegAi1 = LocalTensor<float>(TPosition::A1, 4ULL * aPlaneCapacity * sizeof(float), aPlaneCapacity);
            bBr1 = LocalTensor<float>(TPosition::B1, bL1Base + 0ULL * bSize * sizeof(float), bSize);
            bPosBi1 = LocalTensor<float>(TPosition::B1, bL1Base + 1ULL * bSize * sizeof(float), bSize);
            bAr1 = LocalTensor<float>(TPosition::B1, bL1Base + 2ULL * bSize * sizeof(float), bSize);
            bNegAi1 = LocalTensor<float>(TPosition::B1, bL1Base + 3ULL * bSize * sizeof(float), bSize);
            bAi1 = LocalTensor<float>(TPosition::B1, bL1Base + 4ULL * bSize * sizeof(float), bSize);
            bExtra1 = LocalTensor<float>(TPosition::B1, bL1Base + 5ULL * bSize * sizeof(float), bSize);
            a2 = LocalTensor<float>(TPosition::A2, 0U, aSize);
            b2 = LocalTensor<float>(TPosition::B2, 0U, bSize);
            realC1 = LocalTensor<float>(TPosition::CO1, 0U, cSize);
            imagC1 = LocalTensor<float>(TPosition::CO1, static_cast<uint64_t>(cSize) * sizeof(float), cSize);
            aCopy.ndNum = 1U;
            aCopy.nValue = directOpc ? directM : directK;
            aCopy.dValue = directOpc ? directK : directM;
            aCopy.srcNdMatrixStride = 0U;
            aCopy.srcDValue = directOpc ? k : n;
            aCopy.dstNzC0Stride = directOpc ? directM : directK;
            aCopy.dstNzNStride = 1U;
            aCopy.dstNzMatrixStride = 0U;
            bNzParams.ndNum = 1U;
            bNzParams.nValue = directOpc ? directN : directK;
            bNzParams.dValue = directOpc ? directK : directN;
            bNzParams.srcNdMatrixStride = 0U;
            bNzParams.srcDValue = directOpc ? k : n;
            bNzParams.dstNzC0Stride = directOpc ? directN : directK;
            bNzParams.dstNzNStride = 1U;
            bNzParams.dstNzMatrixStride = 0U;
            Cher2kInitLoadData3D(
                aLoad, directOpc ? directM : directK, directOpc ? directK : directM, directOpc ? directK : directM,
                !directOpc);
            Cher2kInitLoadData3D(
                bLoad, directOpc ? directN : directK, directOpc ? directK : directN, directOpc ? directK : directN,
                true);
            first.m = directM;
            first.n = directN;
            first.k = directK;
            first.cmatrixInitVal = true;
            first.cmatrixSource = false;
            first.kDirectionAlign = false;
            accum = first;
            accum.cmatrixInitVal = false;
        }
    };

    struct Cher2kDirectHermitianSlice {
        LocalTensor<float> aAr;
        LocalTensor<float> aAi;
        LocalTensor<float> aBr;
        LocalTensor<float> aBi;
        LocalTensor<float> aNegAi;
        Nd2NzParams aCopy{};
        Nd2NzParams bCopyParams{};
        LoadData3DParamsV2<float> aLoad{};
        LoadData3DParamsV2<float> bLoad{};
        MmadParams first{};
        MmadParams accum{};
    };

    __aicore__ inline void InitDirectHermitianSlice(
        Cher2kDirectHermitianState& state, Cher2kDirectHermitianSlice& sliceState, uint32_t sliceCount, uint32_t slice,
        uint32_t sliceK)
    {
        const uint32_t aSlot = sliceCount <= 2U ? slice : (sliceCount == 3U && slice == 1U ? 1U : 0U);
        const uint32_t offset = aSlot * Cher2kDirectHermitianState::aSize;
        sliceState.aAr = state.aAr1[offset];
        sliceState.aAi = state.aAi1[offset];
        sliceState.aBr = state.aBr1[offset];
        sliceState.aBi = state.aBi1[offset];
        sliceState.aNegAi = state.aNegAi1[offset];
        sliceState.aCopy = state.aCopy;
        sliceState.bCopyParams = state.bNzParams;
        sliceState.aLoad = state.aLoad;
        sliceState.bLoad = state.bLoad;
        if (state.directOpc) {
            sliceState.aCopy.dValue = sliceK;
            sliceState.bCopyParams.dValue = sliceK;
            sliceState.aLoad.channelSize = sliceK;
            sliceState.aLoad.kExtension = sliceK;
            sliceState.bLoad.channelSize = sliceK;
            sliceState.bLoad.kExtension = sliceK;
        } else {
            sliceState.aCopy.nValue = sliceK;
            sliceState.aCopy.dstNzC0Stride = sliceK;
            sliceState.bCopyParams.nValue = sliceK;
            sliceState.bCopyParams.dstNzC0Stride = sliceK;
            sliceState.aLoad.l1W = sliceK;
            sliceState.aLoad.mExtension = sliceK;
            sliceState.bLoad.l1W = sliceK;
            sliceState.bLoad.mExtension = sliceK;
        }
        sliceState.first = state.first;
        sliceState.accum = state.accum;
        sliceState.first.k = sliceK;
        sliceState.accum.k = sliceK;
    }

    __aicore__ inline void IssueDirectHermitianTerm(
        Cher2kDirectHermitianState& state, Cher2kDirectHermitianSlice& sliceState, LocalTensor<float>& output,
        LocalTensor<float>& left, LocalTensor<float>& right, bool initialize)
    {
        // The preceding MMAD must release the shared L0A/L0B pair.
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        LoadData(state.b2, right, sliceState.bLoad);
        LoadData(state.a2, left, sliceState.aLoad);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
        Mmad(output, state.a2, state.b2, initialize ? sliceState.first : sliceState.accum);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void ComputeDirectHermitianSlice(
        Cher2kDirectHermitianState& state, Cher2kDirectHermitianSlice& sliceState, uint32_t sliceIter)
    {
        const bool initialize = sliceIter == 0U;
        IssueDirectHermitianTerm(state, sliceState, state.realC1, sliceState.aAr, state.bBr1, initialize);
        IssueDirectHermitianTerm(state, sliceState, state.realC1, sliceState.aAi, state.bPosBi1, false);
        IssueDirectHermitianTerm(state, sliceState, state.realC1, sliceState.aBr, state.bAr1, false);
        IssueDirectHermitianTerm(state, sliceState, state.realC1, sliceState.aBi, state.bNegAi1, false);
        IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aNegAi, state.bBr1, initialize);
        IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aAr, state.bPosBi1, false);
        if (state.directOpc) {
            IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aBr, state.bExtra1, false);
            IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aBi, state.bAr1, false);
        } else {
            IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aBi, state.bAr1, false);
            IssueDirectHermitianTerm(state, sliceState, state.imagC1, sliceState.aBr, state.bAi1, false);
        }
    }

    __aicore__ inline void LoadDirectHermitianSlice(
        Cher2kDirectHermitianState& state, Cher2kDirectHermitianSlice& sliceState, uint32_t mBase, uint32_t nBase,
        uint32_t kBase, bool aResident, uint32_t sliceCount, uint32_t slice, uint32_t& slot0Slice)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        const uint64_t rowOffset =
            state.directOpc ? static_cast<uint64_t>(mBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + mBase;
        const uint64_t colOffset =
            state.directOpc ? static_cast<uint64_t>(nBase) * k_ + kBase : static_cast<uint64_t>(kBase) * n_ + nBase;
        if (!aResident) {
            DataCopy(sliceState.aAr, ar_[rowOffset], sliceState.aCopy);
            DataCopy(sliceState.aAi, state.directOpc ? sumA_[rowOffset] : ai_[rowOffset], sliceState.aCopy);
            DataCopy(sliceState.aBr, br_[rowOffset], sliceState.aCopy);
            DataCopy(sliceState.aBi, bi_[rowOffset], sliceState.aCopy);
            DataCopy(sliceState.aNegAi, state.directOpc ? ai_[rowOffset] : sumA_[rowOffset], sliceState.aCopy);
            if (sliceCount == 3U && slice != 1U) {
                slot0Slice = slice;
            }
        }
        DataCopy(state.bBr1, br_[colOffset], sliceState.bCopyParams);
        DataCopy(state.bPosBi1, state.directOpc ? bi_[colOffset] : sumB_[colOffset], sliceState.bCopyParams);
        DataCopy(state.bAr1, ar_[colOffset], sliceState.bCopyParams);
        DataCopy(state.bNegAi1, sumA_[colOffset], sliceState.bCopyParams);
        DataCopy(state.bAi1, state.directOpc ? sumB_[colOffset] : ai_[colOffset], sliceState.bCopyParams);
        if (state.directOpc) {
            DataCopy(state.bExtra1, ai_[colOffset], sliceState.bCopyParams);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
    }

    __aicore__ inline void StoreDirectHermitianTile(Cher2kDirectHermitianState& state, uint32_t mBase, uint32_t nBase)
    {
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        FixpipeParamsV220 store{};
        store.mSize = Cher2kDirectHermitianState::directM;
        store.nSize = Cher2kDirectHermitianState::directN;
        store.srcStride = Cher2kDirectHermitianState::directM;
        store.dstStride = n_;
        store.ndNum = 1U;
        store.srcNdStride = 0U;
        store.dstNdStride = 0U;
        const uint64_t outputOffset = static_cast<uint64_t>(mBase) * n_ + nBase;
        Fixpipe(rr_[outputOffset], state.realC1, store);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::M_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID1);
        Fixpipe(ri_[outputOffset], state.imagC1, store);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
    }

    __aicore__ inline void ProcessDirectHermitianTile(
        Cher2kDirectHermitianState& state, uint32_t mBase, uint32_t nBase, bool sameMRow, uint32_t& slot0Slice)
    {
        constexpr uint32_t directK = Cher2kDirectHermitianState::directK;
        const uint32_t sliceCount = (k_ + directK - 1U) / directK;
        const bool rotateThreeSlices = sliceCount == 3U && sameMRow && slot0Slice == 2U;
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        for (uint32_t sliceIter = 0U; sliceIter < sliceCount; ++sliceIter) {
            const uint32_t slice = rotateThreeSlices ? sliceCount - 1U - sliceIter : sliceIter;
            const uint32_t kBase = slice * directK;
            const uint32_t sliceK = min(directK, k_ - kBase);
            const bool aResident =
                sameMRow && (sliceCount <= 2U || (sliceCount == 3U && (slice == 1U || slice == slot0Slice)));
            Cher2kDirectHermitianSlice sliceState;
            InitDirectHermitianSlice(state, sliceState, sliceCount, slice, sliceK);
            LoadDirectHermitianSlice(state, sliceState, mBase, nBase, kBase, aResident, sliceCount, slice, slot0Slice);
            ComputeDirectHermitianSlice(state, sliceState, sliceIter);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        }
        StoreDirectHermitianTile(state, mBase, nBase);
    }
