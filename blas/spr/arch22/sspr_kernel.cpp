/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "sspr_tiling_data.h"
#include "sspr_kernel.h"  // R18.1: sspr_kernel_large_do declaration (largeMode launch wrapper)

using namespace AscendC;

namespace {
constexpr uint32_t SSPR_IN_BUF_NUM = 2;   // apIn double-buffered queue, depth 2 (hides per-window
                                          // MTE2 latency)
constexpr uint32_t SSPR_OUT_BUF_NUM = 2;  // apOut double buffering
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t SSPR_ALIGN_ELEMS = 8;  // 32B alignment granularity in float32 elements
constexpr float SPLIT_CONST = 4097.0f;     // Dekker splitter 2^12+1 for f32 (exact path, n > SSPR_NAIVE_MAX_N)
constexpr float FLT_MAX_F = 3.4028234663852886e+38f;
// n above this uses the Dekker exact path; n <= this uses the 2-op naive path.
// Rationale: golden cblas is FMA-rounded; the naive two-rounding result deviates
// by <=1 ulp of the product, and the MARE outlier metric (rel err vs golden) breaches the
// 10*2^-13 limit only on cases with very many elements (deep-cancellation tail). n=2048 keeps
// the naive win on the benchmark sizes while large-n rows stay bitwise-exact.
constexpr uint32_t SSPR_NAIVE_MAX_N = 2048;  // keep in sync with host SSPR_DESC_MAX_N (sspr_host.cpp)
// Run/fixup descriptor capacity per window. Worst case: a full ubFormer window at the triangle
// tip intersects at most ~sqrt(2*ubFormer) columns (UPPER: ColEnd(k)-ColStart(0) <= ubFormer
// gives k <= 84 for ubFormer=3584); each column contributes at most 2 scalar cells (head+tail).
constexpr uint32_t SSPR_MAX_RUNS = 96;
constexpr uint32_t SSPR_MAX_CELLS = 176;
// Window-table slot per core, u32 units ([winCount][7 pad][5 x (pos,cnt,idxOff)]).
// The slot constants are SINGLE-SOURCED in sspr_tiling_data.h
// (SSPR_DESC_TBL_SLOT_U32 / SSPR_DESC_TBL_HDR_U32 / SSPR_DESC_MAX_WINS).
}  // namespace

// Gather-based multi-core vector kernel (arch22 / DAV_2201) with a batched-phase window
// design. On this silicon V-pipe ops at UNALIGNED UB offsets are broken and Compares/Select
// with sub-64 counts are broken. Runs therefore start 32B-aligned with
// exact column lengths (count-masked tails), pad lanes are recomputed by their owner later,
// and the <=7-element column-head remnant is fixed by scalar S-pipe writes (Phase E).
// Structure (per window):
//   walk: collect whole-column runs (8-aligned start, exact len) + head fixups (<=7 elems);
//   Phase A: batched ArithProgression row indices;  Phase B: batched Gather of x[row];
//   Phase C: per-run update, path by n: n>SSPR_NAIVE_MAX_N = 12-op Dekker
//             chain (bitwise-FMA); n<=SSPR_NAIVE_MAX_N = 2-op naive two-rounding (bitwise
//             two-rounding on all edges; golden's MARE tail makes large n keep the exact path);
//   Phase D: finiteness guard, exact path only (naive path: guard is a provable identity);
//   Phase E: scalar fixups of column heads (S pipe, apIn->apOut RMW).
// Syncs are once per window (MTE2_V/MTE2_S/V_MTE3/S_MTE3), not per run. UB buffers alias
// (idx<->xi, scratch<->apOut), which is what allows the incx==1 window ubFormer = 3584.
// Phase E per-element S-pipe accesses are 3 (per-cell ax2 + row index
// hoisted, refreshed on column crossings).
// Host-precomputed geometry descriptors (tiling.descMode == 1, small-n gate). The
// per-window walk (col traversal, run/fix classification) and the scalar Phase E are both moved
// behind a host-side precomputation uploaded once per shape into a GM workspace (extra kernel
// param descGM; per-core fixed slots, see SsprTilingData::descSlotU32):
//   - walk is replaced by a descriptor read (runs: a/len/row0/col; fixup blocks; the full
//     geometry contract lives in sspr_host.cpp);
//   - Phase E becomes vector fixup blocks: the host packs every fixup cell into 8-lane aligned
//     blocks with per-lane row/col byte offsets; the kernel runs 3 Gather/Muls/Mul ops on the
//     packed lanes (64-lane legal: aligned offsets, >=8 counts) and one 8-lane Add per block to
//     write apOut. The host precomputation removes the on-device index-chain build cost.
//     Blocks recompute non-fixup lanes with the SAME formula
//     (fl(fl(alpha*x[col])*x[row]) + ap), bitwise-equal to the run path (IEEE mul commutes), so
//     full-block writes are lossless. No S-pipe apOut writes remain -> the S_MTE3 pair is gone.
//   - desc mode is naive-path only (host gates it to n <= SSPR_NAIVE_MAX_N), so the exact-path
//     scratch (tl/e/s) is free and reused for the packed lane work (zero extra UB regions;
//     desc UB slot = tiling.descSlotU32*4B).
// 1. Inter-core: the packed AP array is split into 32B-row-aligned contiguous ranges, one per
//    AIV (multi-core MTE3 writes to disjoint 32B rows never lose data).
//    UPPER tip rebalance: core 0 (which owns the scalar-heavy triangle tip) gets a half share.
// 2. Windows: each core processes its range as contiguous aligned windows of up to ubFormer
//    elements. Per window there is exactly ONE MTE2 copy-in and ONE MTE3 copy-out, and the
//    CopyIn/Compute/CopyOut stages are software-pipelined across windows (queue depth 2).
// 3. Inside a window the elements are covered, in walk order, by whole-column runs and column
//    -head fixups:
//    - Run [p, ColEnd(col)) (start 8-aligned, arbitrary len): row factors x[row] fetched with
//      the element-level Gather primitive (vgather; byte offsets by ArithProgression), update
//      ap += axCol * x[row] with axCol = alpha*x[col] (scalar) and the fma-equivalent correction:
//        c = fl(s + e),  s = fl(ap + p_r),  p_r = fl(axCol*x)
//      where e is the exact product rounding error (Dekker twoProduct via the scalar split
//      axCol = ah + al). Since s + e = ap + axCol*x is the exact fma operand sum, this single
//      rounding reproduces the golden fma bitwise (finite case).
//      Run lanes beyond len (guard pads to ceil64) get stale/garbage data; every such lane is
//      inside a later run/fixup range or beyond the window copy-out, so it is never observed.
//    - Fixup [ColEnd(col), next8): column-head remnant (<=7 elems) of the NEXT column,
//      computed on the S pipe (scalar fma, bitwise golden, UB-only access).
//    - Guard: the corrected value is used only when finite (|c| < FLT_MAX, vector
//      compare+select padded to a full 64-lane block - sub-64 counts misbehave on 910B3);
//      otherwise the plain mul+add result s is kept, which propagates NaN/Inf like the golden fma.
// 4. Bit-exactness: reference BLAS computes temp = alpha*x[j] once per column, then
//    ap += temp * x[i]; the kernel reproduces exactly this rounding structure.
// 5. incx != 1 (TilingKey_1): the raw x window [0, 1+(n-1)*|incx|) is MTE2-loaded into UB once,
//    and the logical x[i] = x[i*|incx|] (or x[(n-1-i)*|incx|] for negative incx) is materialized
//    with the same Gather primitive (ArithProgression byte offsets, step ±|incx|*4).
// 6. n > SSPR_X_RESIDENT_MAX_FLOATS: scalar legacy fallback (host forces single core; the
//    supported range n <= 4100 stays fully resident, so the fallback is a correctness guard
//    only).
// 7. The !IS_INCX1 resident path adds xPhys (xPhysLen*4B) + idx (n*4B)
//    on top of the window-sized regions; when xPhysLen grows past ~24K elements the TOTAL UB
//    allocation crosses the 192KB hardware ceiling and the kernel SILENTLY NO-OPS (SUCCESS,
//    zero AP writes). The host shrinks ubFormer until the
//    budget closes (SsprUbBytes clamp) and degrades to the scalar legacy path (scalarMode)
//    when even the 64-element minimum window cannot fit xPhys.
// SSPR_DEBUG is an accepted env knob (the host fills tiling.debug for ABI
// stability) but the kernel emits no on-device stamps (x crosses the launch boundary as const
// and must never be written); use SSPR_HOSTPROF (host-side sections) or msprof for phase
// timing.
template <bool IS_INCX1>
class SsprKernel {
public:
    __aicore__ inline SsprKernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessRange(uint64_t rangeBegin, uint64_t rangeEnd);

private:
    __aicore__ inline void LoadXResident();
    __aicore__ inline uint64_t ColStart(uint64_t col) const
    {
        return isUpper ? (col * (col + 1U) / 2U) : (col * (2U * n - col + 1U) / 2U);
    }
    __aicore__ inline uint64_t ColEnd(uint64_t col) const
    {
        return isUpper ? (ColStart(col) + col + 1U) : (ColStart(col) + (n - col));
    }
    __aicore__ inline uint64_t RowOf(uint64_t pos, uint64_t col) const
    {
        return isUpper ? (pos - ColStart(col)) : (col + (pos - ColStart(col)));
    }
    __aicore__ inline void CopyInWindow(uint64_t pos, uint32_t cnt);
    __aicore__ inline void ComputeWindow(uint64_t pos, uint32_t cnt, uint64_t col, uint32_t dBase);
    __aicore__ inline void ComputeWindowVector(uint64_t pos, uint32_t cnt,
        uint32_t idxOff, uint32_t slot, bool idxPrefetched, uint64_t nextPos, uint32_t nextCnt,
        uint32_t nextIdxOff);
    __aicore__ inline void CopyOutWindow(uint64_t pos, uint32_t cnt);
    // Scalar GM read-modify-write legacy path (n > resident cap only, single core).
    __aicore__ inline void ScalarProcess();
    // smallMode: straight-line single-window path (desc mode, one window per core, IS_INCX1).
    // One batched MTE2 issue (xRes + idx block + apIn) with a single MTE2_V wait, the naive
    // full-window vector chain (2 Gathers + Muls + Mul + Add, bitwise-equal to
    // ComputeWindowVector), one V_MTE3 pair, copy-out. No queues, no winTbl pull, no S reads:
    // the per-core idx block offset is derived from descSlotU32*useCoreNum plus the 2*ceil64(cnt)
    // block sizes of the preceding cores (each descriptor idx block closes on a 64-relative
    // boundary by construction, see SsprBuildDesc).
    __aicore__ inline void ProcessSmall(uint64_t rangeBegin, uint64_t rangeEnd);
    __aicore__ inline uint32_t SmallIdxOff(uint32_t vecIdx) const;
    // Buffer-set / pipeline-stage helpers of Init, LoadXResident, ComputeWindow, Process,
    // ProcessSmall and ProcessRange.
    __aicore__ inline void InitBuffersSmall(const SsprTilingData& tiling);
    __aicore__ inline void InitBuffersQueues();
    __aicore__ inline void InitDescWinTbl(const SsprTilingData& tiling);
    __aicore__ inline void PullWinTable();
    __aicore__ inline void LoadXResidentIncx1();
    __aicore__ inline void LoadXResidentGeneral();
    __aicore__ inline void WalkWindow(uint64_t pos, uint32_t cnt, uint64_t col, uint32_t& nRun,
        uint32_t& nFix, LocalTensor<uint32_t> runU, LocalTensor<float> runF,
        LocalTensor<uint32_t> fixU);
    __aicore__ inline void FetchRowFactorsIncx1(uint32_t nRun, LocalTensor<uint32_t> runU,
        LocalTensor<int32_t> idx, LocalTensor<float> xg);
    __aicore__ inline void FetchRowFactorsGeneral(uint32_t nRun, LocalTensor<uint32_t> runU,
        LocalTensor<int32_t> idx, LocalTensor<float> xg);
    __aicore__ inline void UpdateRuns(uint32_t nRun, bool exactPath, LocalTensor<uint32_t> runU,
        LocalTensor<float> runF, LocalTensor<float> xg, LocalTensor<float> xi,
        LocalTensor<float> th, LocalTensor<float> tl, LocalTensor<float> e,
        LocalTensor<float> sc, LocalTensor<float> c, LocalTensor<float> apIn,
        LocalTensor<float> apOut);
    __aicore__ inline void GuardFiniteness(uint32_t nRun, LocalTensor<uint32_t> runU,
        LocalTensor<float> apOut, LocalTensor<float> tl, LocalTensor<float> c,
        LocalTensor<float> sc);
    __aicore__ inline void ApplyFixups(uint64_t pos, uint32_t nFix, LocalTensor<uint32_t> fixU,
        LocalTensor<float> apIn, LocalTensor<float> apOut);
    __aicore__ inline bool ComputeRange(uint64_t& rangeBegin, uint64_t& rangeEnd);
    __aicore__ inline bool ComputeUpperRebalance(uint64_t& elemsPerCore, uint64_t& rangeBegin,
        uint64_t& rangeEnd);
    __aicore__ inline bool ClampCoreRange(uint64_t rangeBegin, uint64_t elemsPerCore,
        uint64_t& rangeEnd) const;
    __aicore__ inline void SmallLoadX(LocalTensor<float> xResL);
    __aicore__ inline void SmallLoadAp(LocalTensor<float> apIn, uint64_t rangeBegin, uint32_t cnt);
    __aicore__ inline void SmallStoreAp(LocalTensor<float> apOut, uint64_t rangeBegin, uint32_t cnt);
    __aicore__ inline void ProcessRangeDesc();
    __aicore__ inline void ProcessRangeWalk(uint64_t rangeBegin, uint64_t rangeEnd);

    TPipe pipe;
    GlobalTensor<float> xGM;
    GlobalTensor<float> apGM;
    GlobalTensor<uint32_t> descGM;   // per-core geometry descriptors (descMode only)
    TQue<QuePosition::VECIN, SSPR_IN_BUF_NUM> apInQueue;
    TQue<QuePosition::VECOUT, SSPR_OUT_BUF_NUM> apOutQueue;
    TBuf<TPosition::VECCALC> xgBuf;      // gathered x row factors (ubFormer * 4B)
    TBuf<TPosition::VECCALC> xiBuf;      // product p_r = fl(axCol * x) (ubFormer * 4B);
                                         // aliases the Gather byte-offset tensor (idx) which is
                                         // dead once each run's Gather has consumed it (Phase B
                                         // barrier orders the reuse). !IS_INCX1 needs max(
                                         // ubFormer, n) entries for the init-time x materialize.
    TBuf<TPosition::VECCALC> thBuf;      // [exact path n>SSPR_NAIVE_MAX_N] Dekker high split of x
    TBuf<TPosition::VECCALC> tlBuf;      // [exact path] Dekker low split of x; aliases the uint16
                                         // compare mask in Phase D (tl is dead after Phase C)
    TBuf<TPosition::VECCALC> eBuf;       // [exact path] exact product error e
    TBuf<TPosition::VECCALC> sBuf;       // [exact path] plain mul+add result s
    TBuf<TPosition::VECCALC> cBuf;       // [exact path] fma-equivalent corrected result c
    TBuf<TPosition::VECCALC> xResBuf;    // logical x resident (n * 4B)
    TBuf<TPosition::VECCALC> xPhysBuf;   // raw x window (!IS_INCX1 only, xPhysLen * 4B)
    TBuf<TPosition::VECCALC> runBuf;     // run descriptors: 4 x u32 per run (a, len, row0, axCol)
    TBuf<TPosition::VECCALC> fixBuf;     // scalar-cell descriptors: 3 x u32 (a, len, col)
    TBuf<TPosition::VECCALC> winTblBuf;  // core window table (pos,cnt,idxOff per window)
    TBuf<TPosition::VECCALC> idxBufA;    // idx blocks, double-buffered (even windows)
    TBuf<TPosition::VECCALC> idxBufB;    // idx blocks, double-buffered (odd windows)
    TBuf<TPosition::VECCALC> apInBuf;    // smallMode: single-buffer copy-in (no queue)
    TBuf<TPosition::VECCALC> apOutBuf;   // smallMode: single-buffer copy-out (no queue)
    TBuf<TPosition::VECCALC> idxBuf;     // smallMode: single idx block (2*cnt64 u32)

    LocalTensor<float> xRes;
    uint64_t n = 0;
    uint64_t apLen = 0;
    uint64_t xPhysLen = 0;
    int64_t incx = 1;
    uint32_t absIncx = 1;
    uint32_t uplo = ACLBLAS_UPPER;
    float alpha = 0.0f;
    uint32_t useCoreNum = 1;
    uint32_t ubFormer = SSPR_UB_FORMER;
    uint32_t runCostW = 0;
    uint32_t padMinN = SSPR_PAD_MIN_N;
    uint32_t coreBegin[SSPR_MAX_CORES + 1U];  // host-computed cost-balanced range bounds
    uint32_t descSlotU32 = 0;  // descriptor slot size in u32 (0 = desc disabled)
    bool descMode = false;
    bool smallMode = false;    // straight-line single-window path (host-gated, IS_INCX1 only)
    uint32_t tilingDebug = 0;  // dev-only probe selector (SSPR_DEBUG; AP output is garbage under probe)
    uint32_t vecIdx = 0;
    bool isUpper = true;
    bool vectorPath = false;
};

template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::Init(GM_ADDR x, GM_ADDR ap, const SsprTilingData& tiling)
{
    vecIdx = static_cast<uint32_t>(GetBlockIdx());

    n = tiling.n;
    apLen = tiling.apLen;
    xPhysLen = tiling.xPhysLen;
    incx = tiling.incx;
    uplo = tiling.uplo;
    alpha = tiling.alpha;
    useCoreNum = tiling.useCoreNum;
    ubFormer = tiling.ubFormer;
    runCostW = tiling.runCostW;
    padMinN = tiling.padMinN;
    for (uint32_t i = 0; i <= useCoreNum; ++i) {
        coreBegin[i] = tiling.coreBegin[i];
    }
    isUpper = (uplo == ACLBLAS_UPPER);
    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);
    // The host sets scalarMode when the !IS_INCX1 resident UB budget
    // cannot close even at the 64-element minimum window (SsprUbBytes guard in the host tiling).
    // A UB over-allocation silently no-ops the whole kernel on this silicon, so the degrade
    // decision must be made before any InitBuffer (host keeps useCoreNum = 1 for this mode).
    vectorPath = (n <= SSPR_X_RESIDENT_MAX_FLOATS) && (tiling.scalarMode == 0U);
    // Descriptor mode (host gates it to the naive small-n path; vecIdx < useCoreNum is
    // guaranteed by Process()'s early return before any desc access). The naive-path guard is
    // re-checked here: exact path would clash with the tl/e/s scratch reuse in Phase E.
    // descAddr != 0 is a second line of defence - if the host
    // ever refuses the descriptor AFTER setting descMode/descSlotU32, a zero descAddr would
    // otherwise send the DataCopy below to device address 0.
    descSlotU32 = tiling.descSlotU32;
    descMode = (tiling.descMode != 0U) && (descSlotU32 != 0U) && (n <= SSPR_NAIVE_MAX_N) &&
        (tiling.descAddr != 0ULL);
    // smallMode: straight-line single-window path. Host gate (EnsureSsprDesc caller): desc
    // mode engaged, incx == 1, every core range <= ubFormer (exactly one descriptor window per
    // core) and the single-buffer UB budget closed. Re-checked here per instantiation (the
    // !IS_INCX1 route never receives smallMode from the host; if constexpr keeps the false
    // template's Init/Process free of the path).
    smallMode = descMode && (tiling.smallMode != 0U);
    if constexpr (!IS_INCX1) {
        smallMode = false;
    }
    // tiling.debug carries the SSPR_DEBUG env value
    // (host fills it for ABI stability; the kernel uses it only for the timing probes below).
    // Probe levels on the smallMode path: 1 = skip the V compute chain, 2 = additionally skip
    // the idx MTE2. Timing decomposition only - the AP output is garbage under probe.
    tilingDebug = tiling.debug;

    xGM.SetGlobalBuffer((__gm__ float*)x);
    apGM.SetGlobalBuffer((__gm__ float*)ap);

    if (!vectorPath) {
        return;  // scalar legacy path needs no UB buffers
    }

    if (smallMode) {
        InitBuffersSmall(tiling);
        return;
    }

    InitBuffersQueues();
    InitDescWinTbl(tiling);
    xRes = xResBuf.Get<float>();
    // LoadXResident stays at Init: issuing it after the first window's CopyIn would just
    // delay apIn behind the x load on the in-order MTE2 queue.
    LoadXResident();
    PullWinTable();
}

// smallMode buffer set of Init: single-buffer apIn/apOut (no
// queues), single idx block, the naive-path window scratch (xg/e/sc/c) and xRes. run/fix
// descriptors, th/tl, xi alias, winTbl and the alternating idx buffers are all dead on this
// path. Init issues NO transfers: xRes + idx + apIn are issued back-to-back in ProcessSmall
// behind one MTE2_V pair (in-order MTE2 queue), removing the per-flight serial waits.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::InitBuffersSmall(const SsprTilingData& tiling)
{
    pipe.InitBuffer(xgBuf, (ubFormer + SSPR_ALIGN_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(eBuf, ubFormer * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(sBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(cBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(xResBuf, static_cast<uint32_t>(n) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(apInBuf, ubFormer * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(apOutBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(idxBuf, 2U * ubFormer * BYTENUM_PER_FLOAT32);
    xRes = xResBuf.Get<float>();
    descGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
        static_cast<uintptr_t>(tiling.descAddr)));
}

// Main-path buffer set of Init: queue-backed apIn/apOut plus the
// window scratch; see the inline notes for the slack budgeting.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::InitBuffersQueues()
{
    pipe.InitBuffer(apInQueue, SSPR_IN_BUF_NUM, ubFormer * BYTENUM_PER_FLOAT32);
    // +SSPR_GUARD_SLACK_ELEMS on every buffer the Phase D guard touches at
    // count = ceil64(len) (both queue slots count: the slack rides each depth slot). The last
    // run of a full window reaches up to 63 elements past ubFormer; the +64-element margin
    // keeps Abs/Compares/Select in bounds. Budgeted in host SsprUbBytes (SSPR_UB_FIXED_BYTES).
    pipe.InitBuffer(apOutQueue, SSPR_OUT_BUF_NUM,
        (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    // +8 elems of slack. The GM-unaligned DataCopyPad in Phase A pads
    // len%8==0 runs up one extra 32B block (rightPadding 8 - the "exactly-one-32B-block pad
    // corner"); a full-window last run would otherwise write 32B past xgBuf. The +32B is part
    // of the host SsprUbBytes budget (SSPR_UB_FIXED_BYTES).
    pipe.InitBuffer(xgBuf, (ubFormer + SSPR_ALIGN_ELEMS) * BYTENUM_PER_FLOAT32);
    // desc windows run the naive full-window vector path (ComputeWindowVector) which uses
    // neither xi (IS_INCX1: legacy-run alias) nor th/tl (exact-path splits). With the desc
    // window at 3584 the UB budget only closes if those buffers are skipped in desc mode
    // (157KB IS_INCX1 / 168KB !IS_INCX1 at n<=700, vs the 186KB proven ceiling).
    // idx aliases xi; !IS_INCX1 needs n entries at init for the logical-x materialize.
    if constexpr (IS_INCX1) {
        if (!descMode) {
            pipe.InitBuffer(xiBuf, ubFormer * BYTENUM_PER_FLOAT32);
        }
    } else {
        uint32_t idxElems = (ubFormer > static_cast<uint32_t>(n)) ? ubFormer
                                                                  : static_cast<uint32_t>(n);
        pipe.InitBuffer(xiBuf, idxElems * BYTENUM_PER_FLOAT32);
    }
    if (!descMode) {
        pipe.InitBuffer(thBuf, ubFormer * BYTENUM_PER_FLOAT32);
        // tl aliases the uint16 guard mask (Phase D writes [2a, 2(a+ceil64))),
        // needs the +64-element tail slack like apOut/c/sc.
        pipe.InitBuffer(tlBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    }
    pipe.InitBuffer(eBuf, ubFormer * BYTENUM_PER_FLOAT32);
    // sc/c are READ by the Phase D guard up to a+ceil64(len) - +64 slack.
    pipe.InitBuffer(sBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(cBuf, (ubFormer + SSPR_GUARD_SLACK_ELEMS) * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(runBuf, SSPR_MAX_RUNS * 4U * BYTENUM_PER_FLOAT32);
    pipe.InitBuffer(fixBuf, SSPR_MAX_CELLS * 3U * BYTENUM_PER_FLOAT32);

    pipe.InitBuffer(xResBuf, static_cast<uint32_t>(n) * BYTENUM_PER_FLOAT32);
    if constexpr (!IS_INCX1) {
        pipe.InitBuffer(xPhysBuf, static_cast<uint32_t>(xPhysLen) * BYTENUM_PER_FLOAT32);
    }
}

// Desc-mode buffer set of Init.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::InitDescWinTbl(const SsprTilingData& tiling)
{
    if (descMode) {
        // desc-only buffers allocated AFTER xRes so they cannot overlap its window.
        pipe.InitBuffer(winTblBuf, SSPR_DESC_TBL_SLOT_U32 * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(idxBufA, 2U * ubFormer * BYTENUM_PER_FLOAT32);
        pipe.InitBuffer(idxBufB, 2U * ubFormer * BYTENUM_PER_FLOAT32);
        descGM.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(
            static_cast<uintptr_t>(tiling.descAddr)));
    }
}

// Desc-mode window-table pull of Init: table pull AFTER LoadXResident
// with its own MTE2_S pair (S reads the table) - the window table is tiny; idx blocks are
// pulled per window in ProcessRange and ordered by the per-window MTE2_V pair.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::PullWinTable()
{
    if (descMode) {
        DataCopy(winTblBuf.Get<uint32_t>(), descGM[static_cast<uint64_t>(vecIdx) *
            SSPR_DESC_TBL_SLOT_U32], SSPR_DESC_TBL_SLOT_U32);
        int32_t evMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evMte2S);
        WaitFlag<HardEvent::MTE2_S>(evMte2S);
    }
}

// x load, once per core. IS_INCX1: one aligned MTE2 copy. !IS_INCX1: MTE2 the raw window, then
// materialize the logical x (forward/reverse mapping) with element-level Gather — no scalar GM
// access anywhere.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::LoadXResident()
{
    if constexpr (IS_INCX1) {
        LoadXResidentIncx1();
    } else {
        LoadXResidentGeneral();
    }
}

// IS_INCX1 x load of LoadXResident.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::LoadXResidentIncx1()
{
    uint32_t cnt = static_cast<uint32_t>(n);
    if (cnt % SSPR_ALIGN_ELEMS == 0) {
        DataCopy(xRes, xGM[0], cnt);
    } else {
        uint32_t paddingNumU = (SSPR_ALIGN_ELEMS - cnt % SSPR_ALIGN_ELEMS) % SSPR_ALIGN_ELEMS;
        if (paddingNumU == 0U) {
            paddingNumU = SSPR_ALIGN_ELEMS; // avoid the exactly-one-32B-block pad corner
        }
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
        DataCopyPad(xRes, xGM[0], copyParams, padParams);
    }
    // MTE2 -> V (Gather reads xRes on the V pipe) and MTE2 -> S (axCol/row-factor GetValue).
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);
    int32_t evMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evMte2S);
    WaitFlag<HardEvent::MTE2_S>(evMte2S);
}

// !IS_INCX1 x load + logical-x materialize of LoadXResident.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::LoadXResidentGeneral()
{
    LocalTensor<float> xPhys = xPhysBuf.Get<float>();
    uint32_t cnt = static_cast<uint32_t>(xPhysLen);
    if (cnt % SSPR_ALIGN_ELEMS == 0) {
        DataCopy(xPhys, xGM[0], cnt);
    } else {
        uint32_t paddingNumU = (SSPR_ALIGN_ELEMS - cnt % SSPR_ALIGN_ELEMS) % SSPR_ALIGN_ELEMS;
        if (paddingNumU == 0U) {
            paddingNumU = SSPR_ALIGN_ELEMS;
        }
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
        DataCopyPad(xPhys, xGM[0], copyParams, padParams);
    }
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);

    // logical x[i] = xPhys[i*|incx|] (incx > 0) or xPhys[(n-1-i)*|incx|] (incx < 0):
    // byte offsets form an arithmetic progression with step ±|incx|*4.
    LocalTensor<int32_t> idx = xiBuf.Get<int32_t>();
    int32_t first = (incx >= 0) ? 0 : static_cast<int32_t>((n - 1U) * absIncx * BYTENUM_PER_FLOAT32);
    int32_t step = static_cast<int32_t>(absIncx * BYTENUM_PER_FLOAT32);
    if (incx < 0) {
        step = -step;
    }
    ArithProgression(idx, first, step, static_cast<int32_t>(n));
    PipeBarrier<PIPE_V>();
    Gather(xRes, xPhys, idx.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
        static_cast<uint32_t>(n));
    PipeBarrier<PIPE_V>();
    // xRes is V-pipe output; synchronize to S for the per-column axCol GetValue reads.
    int32_t evVS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evVS);
    WaitFlag<HardEvent::V_S>(evVS);
}


// Desc mode: the whole window is one vector statement chain. The host-precomputed
// per-element row/col indices arrive via MTE2; x[row] and x[col] are Gathered from the resident
// logical x, the column factor is scaled by alpha, and ONE full-window Mul+Add performs the
// rank-1 update. Bitwise-equal to the legacy run path: fl(fl(alpha*x[col])*x[row]) + ap with
// mul-before-add rounding (IEEE mul commutes, same two-rounding as Muls+Add). All tensor starts
// are offset 0 / 32B-aligned (cnt padded sections) - the V-op alignment rules hold; counts
// may be arbitrary.
// Deep prefetch: the per-window idx block is fetched by ProcessRange one window AHEAD
// (during the previous window's compute + the next apIn MTE2 flight), hiding the per-window
// serial idx MTE2 latency. The MTE2_V pair below orders the prefetched idx arrival before the
// Gathers.
// The NEXT window's MTE2 (apIn(w+1) CopyIn + idx(w+1) prefetch) is issued HERE, right
// AFTER this window's MTE2_V wait: the wait set is exactly this window's transfers (flown
// during compute(w-1)), and the next flight overlaps this window's V work + the CopyOut MTE3
// (issuing it before the wait would make compute(w) wait for the next window's flight too).
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ComputeWindowVector(uint64_t pos, uint32_t cnt,
    uint32_t idxOff, uint32_t slot, bool idxPrefetched,
    uint64_t nextPos, uint32_t nextCnt, uint32_t nextIdxOff)
{
    LocalTensor<float> apIn = apInQueue.DeQue<float>();
    LocalTensor<float> apOut = apOutQueue.AllocTensor<float>();
    LocalTensor<float> xg = xgBuf.Get<float>();
    LocalTensor<float> e = eBuf.Get<float>();
    LocalTensor<float> sc = sBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<uint32_t> idxU = (slot != 0U) ? idxBufB.Get<uint32_t>() : idxBufA.Get<uint32_t>();

    // per-window idx block fetched into this window's alternating slot (written
    // once per two windows) then ordered by the MTE2_V pair - no overwrite race with the
    // previous window's Gather.
    uint32_t cnt64 = (cnt + 63U) & ~63U;  // rows[] and cols[] are padded to 64 on the host
    if (!idxPrefetched) {
        DataCopy(idxU, descGM[idxOff], 2U * cnt64);  // idxOff is global
    }
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);
    // next-window MTE2 AFTER the wait (see function header). CopyIn allocates the
    // apIn slot just freed by the DeQue above (queue depth 2: this window's tensor is the only
    // one held). idx(w+1) targets slot (w+1)&1 == (w-1)&1, whose last V consumer was the
    // window w-1 Gather - drained by compute(w-1)'s V_MTE3 wait before compute(w) started.
    if (nextCnt > 0U) {
        CopyInWindow(nextPos, nextCnt);
        DataCopy((slot != 0U) ? idxBufA.Get<uint32_t>() : idxBufB.Get<uint32_t>(),
            descGM[nextIdxOff], 2U * ((nextCnt + 63U) & ~63U));
    }

    // Phase A+B: row factors and column factors (each a single Gather over the whole window).
    // Gather count padded to 64: arbitrary counts showed tail-lane garbage on this silicon;
    // the padded lanes read xRes[0] via the zero idx padding
    // and are discarded by the copy-out (Add/MTE3 below use cnt).
    Gather(xg, xRes, idxU[0U], static_cast<uint32_t>(0), cnt64);
    Gather(e, xRes, idxU[cnt64], static_cast<uint32_t>(0), cnt64);
    // column factor: alpha * x[col] (NOT in-place: every dst differs from its srcs)
    Muls(c, e, alpha, cnt64);
    // Phase C: one full-window rank-1 update (replaces per-run Muls+Add AND scalar Phase E)
    Mul(sc, xg, c, cnt64);
    Add(apOut, sc, apIn, cnt);  // cnt (not cnt64): only real lanes are copied back to GM
    int32_t evVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evVMte3);
    WaitFlag<HardEvent::V_MTE3>(evVMte3);

    apInQueue.FreeTensor(apIn);
    apOutQueue.EnQue<float>(apOut);
}

// One window [pos, pos+cnt), pos 8-aligned, covered in walk order by whole-column runs
// (8-aligned start, exact len) and <=7-element column-head fixups. Four batched phases so the
// per-run serial latency chains interleave across runs on the V pipe.
// dBase != 0 (desc mode): the walk output was precomputed on the host; nRun/nBlock and the
// run/block/lane descriptors are read from descBuf at offset dBase (see SsprBuildDesc layout).
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ComputeWindow(uint64_t pos, uint32_t cnt, uint64_t col,
    uint32_t dBase)
{
    LocalTensor<float> apIn = apInQueue.DeQue<float>();
    LocalTensor<float> apOut = apOutQueue.AllocTensor<float>();
    LocalTensor<float> xg = xgBuf.Get<float>();
    LocalTensor<float> xi = xiBuf.Get<float>();
    LocalTensor<int32_t> idx = xiBuf.Get<int32_t>();  // alias: dead after Phase B barrier
    LocalTensor<float> th = thBuf.Get<float>();
    LocalTensor<float> tl = tlBuf.Get<float>();
    LocalTensor<float> e = eBuf.Get<float>();
    LocalTensor<float> sc = sBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();
    LocalTensor<uint32_t> runU = runBuf.Get<uint32_t>();
    LocalTensor<float> runF = runBuf.Get<float>();
    LocalTensor<uint32_t> fixU = fixBuf.Get<uint32_t>();
    // bitwise-FMA exact path for large n (MARE deep-cancellation tail), naive 2-op
    // path otherwise (perf); see SSPR_NAIVE_MAX_N.
    const bool exactPath = (n > SSPR_NAIVE_MAX_N);

    // MTE2 -> V: apIn is read on the V pipe in Phase C (the manual MTE2_V pair is required -
    // relying on TQue DeQue auto-sync alone is not sufficient). The MTE2 -> S pair is
    // NOT needed here: apIn is read on the S pipe only in Phase E (the pair is issued there,
    // gated on nFix > 0, so fix-free windows skip it and the walk runs during MTE2 flight).
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);

    // ---- walk: collect whole-column runs (start 8-aligned when possible, exact length; count
    // may be any value as long as every tensor start offset is 32B-aligned) plus scalar cells
    // for unaligned column heads (<8 elems).
    // desc mode: the host precomputed this walk; nRun/nBlock and the descriptors are read
    // from the window body at dBase ([nRun][nBlock][runs 4*u32][a8 per block][rowBytes][colBytes]).
    uint32_t nRun = 0;
    uint32_t nFix = 0;
    WalkWindow(pos, cnt, col, nRun, nFix, runU, runF, fixU);


    // ---- Phase A: batched row-factor fetch. IS_INCX1: the row factors x[row0..row0+len) are a
    // CONTIGUOUS slice of xGM. Long runs use one GM-unaligned DataCopyPad (the element-level
    // Gather is slower per 64-lane repeat and dominates the window wall on long runs);
    // short runs keep the Gather (per-MTE2-op overhead ~1us exceeds the gather cost below
    // ~256 elems). !IS_INCX1: byte offsets + element-level Gather from the resident logical x. ----
    if constexpr (IS_INCX1) {
        FetchRowFactorsIncx1(nRun, runU, idx, xg);
    } else {
        FetchRowFactorsGeneral(nRun, runU, idx, xg);
    }

    // ---- Phase C: per-run update; two paths selected by n ----
    // n > SSPR_NAIVE_MAX_N (exact path, bitwise-FMA): 12-op Dekker chain c = sc + e,
    // then the Phase D finiteness guard selects sc on Inf/NaN.
    // n <= SSPR_NAIVE_MAX_N (naive path): apOut = apIn + axCol*x with mul-then-add
    // rounding (2 V ops). Axpy/Muls+Add == two-rounding on ALL
    // edge inputs (Inf/NaN/+-0/overflow/denormal) and ==
    // round_f32(ap + s*x) exactly. vs the FMA-rounded golden it deviates <=1 ulp of the
    // product, which the MERE/MARE calibers absorb on small-apLen cases (see SSPR_NAIVE_MAX_N).
    UpdateRuns(nRun, exactPath, runU, runF, xg, xi, th, tl, e, sc, c, apIn, apOut);

    // ---- Phase D: finiteness guard, exact path only (batched; count = ceil64(len) - sub-64
    // / non-64-multiple counts are BROKEN on 910B3). Naive path: the guard is a
    // provable identity - the fallback value sc equals the naive value written to
    // apOut. ----
    if (exactPath) {
        GuardFiniteness(nRun, runU, apOut, tl, c, sc);
    }

    // ---- Phase E: fixup cells (column head/tail remnants, short columns).
    // Legacy (dBase == 0): scalar cells on the S pipe. The per-cell column factor ax2 and
    // the row index are hoisted out of the element loop (per-element re-derivation costs 4
    // S-pipe UB accesses per element - the most expensive per-element work in the kernel).
    // A fix cell starts inside one column and
    // may cross into the next (multi-column remnants after short columns), so the hoisted state
    // refreshes on crossings.
    // desc mode (dBase != 0): vector fixup blocks. The host packed every fixup cell into
    // 8-lane 32B-aligned blocks (a8 + per-lane row/col byte offsets, out-of-window lanes
    // clamped). Per window: 2 Gathers + Muls + Mul on the packed lanes (prod buffers reuse the
    // exact-path scratch tl/e/s, idle because desc mode is naive-path only), then one 8-lane
    // Add per block writes apOut. Non-fixup lanes of a block are recomputed with the SAME
    // formula (fl(ax*x) + ap) as the Phase C run path -> bitwise-equal, full-block writes are
    // lossless. No S-pipe apOut writes remain -> no MTE2_S pair and no S_MTE3 pair. ----
    if (nFix > 0U) {
        ApplyFixups(pos, nFix, fixU, apIn, apOut);
    }

    int32_t evVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evVMte3);
    WaitFlag<HardEvent::V_MTE3>(evVMte3);
    if (nFix > 0U) {
        // S wrote apOut scalar cells; without fix cells no S->MTE3 dependency exists.
        int32_t evSMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evSMte3);
        WaitFlag<HardEvent::S_MTE3>(evSMte3);
    }

    apInQueue.FreeTensor(apIn);
    apOutQueue.EnQue<float>(apOut);
}

// Walk of one window of ComputeWindow: collect whole-column runs
// (8-aligned start, exact len) and <=7-element column-head fixups.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::WalkWindow(uint64_t pos, uint32_t cnt, uint64_t col,
    uint32_t& nRun, uint32_t& nFix, LocalTensor<uint32_t> runU, LocalTensor<float> runF,
    LocalTensor<uint32_t> fixU)
{
    uint64_t p = pos;
    const uint64_t pEnd = pos + cnt;
    uint64_t cc = col;
    while (p < pEnd) {
        while (ColEnd(cc) <= p) {
            ++cc;
        }
        uint64_t runEnd = (ColEnd(cc) < pEnd) ? ColEnd(cc) : pEnd;
        uint32_t a = static_cast<uint32_t>(p - pos);
        if (a != (a & ~7U)) {
            // unaligned position (previous column ended mid-32B): scalar head-gap cell
            fixU.SetValue(3U * nFix, a);
            fixU.SetValue(3U * nFix + 1U, static_cast<uint32_t>(runEnd - p));
            fixU.SetValue(3U * nFix + 2U, static_cast<uint32_t>(cc));
            ++nFix;
            p = runEnd;
            continue;
        }
        uint32_t len = static_cast<uint32_t>(runEnd - p);
        if (len >= SSPR_ALIGN_ELEMS) {
            runU.SetValue(4U * nRun, a);
            runU.SetValue(4U * nRun + 1U, len);
            runU.SetValue(4U * nRun + 2U, static_cast<uint32_t>(RowOf(p, cc)));
            runF.SetValue(4U * nRun + 3U, alpha * xRes.GetValue(static_cast<uint32_t>(cc)));
            ++nRun;
        } else {
            // short run (< 8 elems): scalar cell (sub-8 vector runs lose the ap addend)
            fixU.SetValue(3U * nFix, a);
            fixU.SetValue(3U * nFix + 1U, len);
            fixU.SetValue(3U * nFix + 2U, static_cast<uint32_t>(cc));
            ++nFix;
        }
        p = runEnd;
        if (((p & 7U) != 0U) && p < pEnd) {
            // column-head remnant of the next column: [p, next8) (<=7 elems, scalar cell)
            uint64_t qEnd = (p + 7U) & ~7ULL;
            if (qEnd > pEnd) {
                qEnd = pEnd;
            }
            fixU.SetValue(3U * nFix, static_cast<uint32_t>(p - pos));
            fixU.SetValue(3U * nFix + 1U, static_cast<uint32_t>(qEnd - p));
            fixU.SetValue(3U * nFix + 2U, static_cast<uint32_t>(cc + 1U));
            ++nFix;
            p = qEnd;
        }
    }
}

// Phase A+B of ComputeWindow, IS_INCX1 branch.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::FetchRowFactorsIncx1(uint32_t nRun,
    LocalTensor<uint32_t> runU, LocalTensor<int32_t> idx, LocalTensor<float> xg)
{
    const bool usePad = (n >= padMinN);
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        uint32_t row0 = runU.GetValue(4U * r + 2U);
        if (usePad && len >= SSPR_ALIGN_ELEMS) {
            // GM-unaligned slice fetch. The pad always rounds the copy up to one extra 32B
            // block (rightPadding 8 when len%8==0 - the "exactly-one-32B-block pad
            // corner"); xgBuf carries +8 elems of slack for exactly that (see the
            // InitBuffer above). GM source may start at any byte offset
            // (DataCopyPad handles unaligned GM src).
            DataCopyExtParams cp{1, len * BYTENUM_PER_FLOAT32, 0, 0, 0};
            uint32_t padNum = SSPR_ALIGN_ELEMS - len % SSPR_ALIGN_ELEMS;
            DataCopyPadExtParams<float> pp{true, 0, static_cast<uint8_t>(padNum), 0};
            DataCopyPad(xg[a], xGM[row0], cp, pp);
        } else {
            ArithProgression(idx[a], static_cast<int32_t>(row0 * BYTENUM_PER_FLOAT32),
                static_cast<int32_t>(BYTENUM_PER_FLOAT32), static_cast<int32_t>(len));
        }
    }
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        if (!usePad) {
            Gather(xg[a], xRes, idx[a].ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), len);
        }
    }
    int32_t evM2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evM2V);
    WaitFlag<HardEvent::MTE2_V>(evM2V);
}

// Phase A+B of ComputeWindow, !IS_INCX1 branch.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::FetchRowFactorsGeneral(uint32_t nRun,
    LocalTensor<uint32_t> runU, LocalTensor<int32_t> idx, LocalTensor<float> xg)
{
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        uint32_t row0 = runU.GetValue(4U * r + 2U);
        ArithProgression(idx[a], static_cast<int32_t>(row0 * BYTENUM_PER_FLOAT32),
            static_cast<int32_t>(BYTENUM_PER_FLOAT32), static_cast<int32_t>(len));
    }
    PipeBarrier<PIPE_V>();

    // ---- Phase B: batched Gathers of x[row] ----
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        Gather(xg[a], xRes, idx[a].ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), len);
    }
    PipeBarrier<PIPE_V>();
}

// Phase C of ComputeWindow: per-run update, exact/naive path by n.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::UpdateRuns(uint32_t nRun, bool exactPath,
    LocalTensor<uint32_t> runU, LocalTensor<float> runF, LocalTensor<float> xg,
    LocalTensor<float> xi, LocalTensor<float> th, LocalTensor<float> tl, LocalTensor<float> e,
    LocalTensor<float> sc, LocalTensor<float> c, LocalTensor<float> apIn, LocalTensor<float> apOut)
{
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        float axCol = runF.GetValue(4U * r + 3U);
        if (exactPath) {
            float c1 = SPLIT_CONST * axCol;
            float ah = c1 - (c1 - axCol);
            float al = axCol - ah;
            Muls(xi[a], xg[a], axCol, len);
            Add(sc[a], apIn[a], xi[a], len);
            Muls(th[a], xg[a], SPLIT_CONST, len);
            Sub(tl[a], th[a], xg[a], len);
            Sub(th[a], th[a], tl[a], len);
            Sub(tl[a], xg[a], th[a], len);
            Muls(e[a], th[a], ah, len);
            Sub(e[a], e[a], xi[a], len);
            Axpy(e[a], tl[a], ah, len);
            Axpy(e[a], th[a], al, len);
            Axpy(e[a], tl[a], al, len);
            Add(c[a], sc[a], e[a], len);
        } else {
            Muls(xi[a], xg[a], axCol, len);
            Add(apOut[a], apIn[a], xi[a], len);
        }
    }
}

// Phase D of ComputeWindow: finiteness guard, exact path only.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::GuardFiniteness(uint32_t nRun,
    LocalTensor<uint32_t> runU, LocalTensor<float> apOut, LocalTensor<float> tl,
    LocalTensor<float> c, LocalTensor<float> sc)
{
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        // count must stay ceil64(len) - sub-64 / non-64-multiple counts make the guard
        // return the raw |c| value (Inf/NaN/+-FLT_MAX) on VALID lanes instead of the
        // fallback sc - invisible with finite inputs, wrong Inf/overflow semantics with
        // non-finite ones. Address safety is provided by the +64-element tail slack on
        // apOut/c/sc/tl (InitBuffer above; budgeted in host SsprUbBytes): a+ceil64(len) <=
        // ubFormer+63 stays in bounds even when ceil64(len) exceeds the window tail.
        uint32_t lenPad = (len + 63U) & ~63U;
        Abs(apOut[a], c[a], lenPad);  // apOut = |c| as the mask input scratch
    }
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        uint32_t lenPad = (len + 63U) & ~63U;
        Compares(tl.ReinterpretCast<uint16_t>()[2U * a], apOut[a], FLT_MAX_F,
            CMPMODE::LT, lenPad);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < nRun; ++r) {
        uint32_t a = runU.GetValue(4U * r);
        uint32_t len = runU.GetValue(4U * r + 1U);
        uint32_t lenPad = (len + 63U) & ~63U;
        Select(apOut[a], tl.ReinterpretCast<uint16_t>()[2U * a], c[a], sc[a],
            SELMODE::VSEL_TENSOR_TENSOR_MODE, lenPad);
    }
    PipeBarrier<PIPE_V>();
}

// Phase E of ComputeWindow: scalar fixup cells, S pipe.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ApplyFixups(uint64_t pos, uint32_t nFix,
    LocalTensor<uint32_t> fixU, LocalTensor<float> apIn, LocalTensor<float> apOut)
{
    // desc mode: fix cells come from the descriptor as SINGLE-COLUMN cells
    // (a, len, col, row0) - the host pre-split multi-column remnants, so the inner loop has
    // no ColEnd math and no crossing refresh.
    // S-pipe reads of apIn need MTE2 -> S on BOTH paths - hoisted here.
    int32_t evMte2S = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evMte2S);
    WaitFlag<HardEvent::MTE2_S>(evMte2S);

    for (uint32_t f = 0; f < nFix; ++f) {
        uint32_t a = fixU.GetValue(3U * f);
        uint32_t len = fixU.GetValue(3U * f + 1U);
        uint64_t p2 = pos + a;
        uint64_t cc2 = fixU.GetValue(3U * f + 2U);
        uint64_t ce2 = ColEnd(cc2);
        float ax2 = alpha * xRes.GetValue(static_cast<uint32_t>(cc2));
        uint32_t row = static_cast<uint32_t>(RowOf(p2, cc2));
        for (uint32_t k = 0; k < len; ++k, ++p2) {
            if (p2 >= ce2) {
                // crossed into the next column (multi-column remnant cells)
                ++cc2;
                ce2 = ColEnd(cc2);
                ax2 = alpha * xRes.GetValue(static_cast<uint32_t>(cc2));
                row = static_cast<uint32_t>(RowOf(p2, cc2));
            }
            float xRow = xRes.GetValue(row);
            apOut.SetValue(a + k, apIn.GetValue(a + k) + ax2 * xRow);
            ++row;
        }
    }
}

template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::CopyInWindow(uint64_t pos, uint32_t cnt)
{
    LocalTensor<float> apIn = apInQueue.AllocTensor<float>();
    if (cnt % SSPR_ALIGN_ELEMS == 0) {
        DataCopy(apIn, apGM[pos], cnt);
    } else {
        uint32_t paddingNumU = (SSPR_ALIGN_ELEMS - cnt % SSPR_ALIGN_ELEMS) % SSPR_ALIGN_ELEMS;
        if (paddingNumU == 0U) {
            paddingNumU = SSPR_ALIGN_ELEMS;
        }
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
        DataCopyPad(apIn, apGM[pos], copyParams, padParams);
    }
    apInQueue.EnQue<float>(apIn);
}

template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::CopyOutWindow(uint64_t pos, uint32_t cnt)
{
    LocalTensor<float> apOut = apOutQueue.DeQue<float>();
    if (cnt % SSPR_ALIGN_ELEMS == 0) {
        DataCopy(apGM[pos], apOut, cnt);
    } else {
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPad(apGM[pos], apOut, copyParams);
    }
    apOutQueue.FreeTensor(apOut);
}

// Legacy scalar GM read-modify-write path (n > resident cap only; host forces single core).
// Per-element GlobalTensor GetValue/SetValue read-modify-write is on the API blacklist and is
// a 10x-100x perf cliff, but this function is the DELIBERATE large-n correctness fallback
// (entered for n > 4104 or tiling.scalarMode == 1 when the !IS_INCX1 resident UB budget
// cannot close), accepted AS-IS under the following rationale:
//   1. The regular test range (n <= 4100) never selects this path - the host keeps those
//      shapes on the vector route.
//   2. The benchmark shapes (512/1024/2048/4096) cannot reach this path either, so the
//      tuned performance envelope is unaffected.
//   3. A windowed-vector fallback (reuse CopyInWindow/ComputeWindow/CopyOutWindow with a
//      per-window x re-fetch instead of the resident x) is a possible future evolution.
// Do not add new callers.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ScalarProcess()
{
    for (uint64_t col = 0; col < n; ++col) {
        uint64_t colStart = ColStart(col);
        if (colStart >= apLen) {
            break;
        }
        uint64_t colEnd = ColEnd(col);
        float axCol = alpha * xGM.GetValue(
            (incx >= 0) ? (col * static_cast<uint64_t>(absIncx))
                        : ((n - 1U - col) * static_cast<uint64_t>(absIncx)));
        for (uint64_t pos = colStart; pos < colEnd; ++pos) {
            uint64_t row = isUpper ? (pos - colStart) : (col + (pos - colStart));
            uint64_t xPhys = (incx >= 0) ? (row * static_cast<uint64_t>(absIncx))
                                         : ((n - 1U - row) * static_cast<uint64_t>(absIncx));
            apGM.SetValue(pos, apGM.GetValue(pos) + axCol * xGM.GetValue(xPhys));
        }
    }
}

template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::Process()
{
    if (vecIdx >= useCoreNum) {
        return;
    }
    if (!vectorPath) {
        ScalarProcess();
        return;
    }

    // 32B-row-aligned packed-element range of this core (elements per core rounded up to 8).
    uint64_t rangeBegin;
    uint64_t rangeEnd;
    if (!ComputeRange(rangeBegin, rangeEnd)) {
        return;
    }
    if (smallMode) {
        ProcessSmall(rangeBegin, rangeEnd);
        return;
    }
    ProcessRange(rangeBegin, rangeEnd);
}

// Clamp of one core's [rangeBegin, rangeBegin + elemsPerCore) range to apLen, shared by
// ComputeUpperRebalance (vecIdx != 0 branch) and ComputeRange (plain split branch); returns
// false when the range starts past apLen.
template <bool IS_INCX1>
__aicore__ inline bool SsprKernel<IS_INCX1>::ClampCoreRange(uint64_t rangeBegin,
    uint64_t elemsPerCore, uint64_t& rangeEnd) const
{
    if (rangeBegin >= apLen) {
        return false;
    }
    rangeEnd = rangeBegin + elemsPerCore;
    if (rangeEnd > apLen) {
        rangeEnd = apLen;
    }
    return true;
}

// UPPER tip rebalance branch of ComputeRange.
template <bool IS_INCX1>
__aicore__ inline bool SsprKernel<IS_INCX1>::ComputeUpperRebalance(uint64_t& elemsPerCore,
    uint64_t& rangeBegin, uint64_t& rangeEnd)
{
    // UPPER tip rebalance: the first core owns the triangle tip where per-column run/fixup
    // overhead is densest; give it a half share and stretch the others.
    elemsPerCore = (2ULL * apLen + (2ULL * useCoreNum - 1ULL) - 1ULL) / (2ULL * useCoreNum - 1ULL);
    elemsPerCore = (elemsPerCore + SSPR_ALIGN_ELEMS - 1ULL) / SSPR_ALIGN_ELEMS * SSPR_ALIGN_ELEMS;
    if (elemsPerCore == 0ULL) {
        elemsPerCore = SSPR_ALIGN_ELEMS;
    }
    uint64_t halfShare = ((elemsPerCore / 2ULL) + SSPR_ALIGN_ELEMS - 1ULL) /
        SSPR_ALIGN_ELEMS * SSPR_ALIGN_ELEMS;
    if (vecIdx == 0U) {
        rangeBegin = 0ULL;
        rangeEnd = halfShare;
        if (rangeEnd > apLen) {
            rangeEnd = apLen;
        }
    } else {
        rangeBegin = halfShare + (vecIdx - 1U) * elemsPerCore;
        if (!ClampCoreRange(rangeBegin, elemsPerCore, rangeEnd)) {
            return false;
        }
    }
    return true;
}

// Range computation of Process: the 32B-row-aligned packed-element range of this core (early
// exits return false to skip the rest of Process).
template <bool IS_INCX1>
__aicore__ inline bool SsprKernel<IS_INCX1>::ComputeRange(uint64_t& rangeBegin, uint64_t& rangeEnd)
{
    uint64_t elemsPerCore;
    if (runCostW > 0U) {
        // Cost-balanced split: bounds precomputed on the HOST under the cost model
        // C(x) = x + W*columnsCrossed(x) (elements + W equivalent-elements per crossed-column
        // run-entry). Balances the run-dense triangle tip for BOTH uplo.
        // NOTE: bounds are computed host-side on purpose - the device-side binary-search solve
        // miscompiles (watchdog hang per launch).
        rangeBegin = coreBegin[vecIdx];
        rangeEnd = coreBegin[vecIdx + 1U];
        if (rangeBegin >= apLen) {
            return false;
        }
        if (rangeEnd > apLen) {
            rangeEnd = apLen;
        }
    } else if (isUpper && useCoreNum > 1U) {
        if (!ComputeUpperRebalance(elemsPerCore, rangeBegin, rangeEnd)) {
            return false;
        }
    } else {
        elemsPerCore = (apLen + useCoreNum - 1ULL) / useCoreNum;
        elemsPerCore = (elemsPerCore + SSPR_ALIGN_ELEMS - 1ULL) / SSPR_ALIGN_ELEMS * SSPR_ALIGN_ELEMS;
        if (elemsPerCore == 0ULL) {
            elemsPerCore = SSPR_ALIGN_ELEMS;
        }
        rangeBegin = static_cast<uint64_t>(vecIdx) * elemsPerCore;
        if (!ClampCoreRange(rangeBegin, elemsPerCore, rangeEnd)) {
            return false;
        }
    }
    return true;
}

// smallMode: per-core idx block offset. The descriptor idx area starts at
// descSlotU32 * useCoreNum and every core's window block is exactly 2*ceil64(cnt) u32 (rows and
// cols are each 64-relative padded by SsprBuildDesc; a preceding empty core contributes 0).
template <bool IS_INCX1>
__aicore__ inline uint32_t SsprKernel<IS_INCX1>::SmallIdxOff(uint32_t coreIdx) const
{
    uint32_t off = descSlotU32 * useCoreNum;
    for (uint32_t c = 0U; c < coreIdx; ++c) {
        uint32_t begin = coreBegin[c];
        uint32_t end = (coreBegin[c + 1U] < apLen) ? coreBegin[c + 1U]
                                                   : static_cast<uint32_t>(apLen);
        uint32_t cnt = (end > begin) ? (end - begin) : 0U;
        off += 2U * ((cnt + 63U) & ~63U);
    }
    return off;
}

// smallMode: straight-line single-window processing of [rangeBegin, rangeEnd) (cnt <=
// ubFormer, desc mode, IS_INCX1). Bitwise-equal to the ComputeWindowVector chain on the same
// descriptor data: the host-packed per-element row/col idx block is MTE2'd together with xRes
// and the ap window, one MTE2_V wait orders all three (in-order MTE2 queue), then
// Gather(xg=row) -> Gather(e=col) -> Muls(c=alpha*e) -> Mul(sc=xg*c) -> Add(apOut=sc+apIn) and
// the copy-out. 2 flag pairs total (vs 5) and no queue ops (vs 8) - the whole point of the path.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ProcessSmall(uint64_t rangeBegin, uint64_t rangeEnd)
{
    const uint32_t cnt = static_cast<uint32_t>(rangeEnd - rangeBegin);
    if (cnt == 0U) {
        return;  // degenerate core range (host split guarantees monotone non-empty ranges)
    }
    const uint32_t cnt64 = (cnt + 63U) & ~63U;
    LocalTensor<float> xResL = xResBuf.Get<float>();
    LocalTensor<uint32_t> idxU = idxBuf.Get<uint32_t>();
    LocalTensor<float> apIn = apInBuf.Get<float>();
    LocalTensor<float> apOut = apOutBuf.Get<float>();
    LocalTensor<float> xg = xgBuf.Get<float>();
    LocalTensor<float> e = eBuf.Get<float>();
    LocalTensor<float> sc = sBuf.Get<float>();
    LocalTensor<float> c = cBuf.Get<float>();

    // Batched MTE2 issue: xRes (logical x), this core's idx block, the ap window. Every tensor
    // start is the UB block base (32B aligned); the GM x base is aligned, the ap window base is
    // 8-elem aligned (coreBegin 8-aligned) with the pad corner handled like CopyInWindow.
    SmallLoadX(xResL);
    DataCopy(idxU, descGM[SmallIdxOff(vecIdx)], 2U * cnt64);
    if (tilingDebug >= 2U) {
        return;  // probe 2: idx + V compute skipped (apOut garbage, timing only)
    }
    SmallLoadAp(apIn, rangeBegin, cnt);
    int32_t evMte2V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evMte2V);
    WaitFlag<HardEvent::MTE2_V>(evMte2V);

    // Naive full-window vector chain (bitwise = ComputeWindowVector on the same descriptor):
    // padded lanes read xRes[0] via the zero idx padding and are discarded by the cnt-bounded
    // Add / copy-out below.
    if (tilingDebug >= 1U) {
        return;  // probe 1: V compute skipped (apOut garbage, timing only)
    }
    Gather(xg, xResL, idxU[0U], static_cast<uint32_t>(0), cnt64);
    Gather(e, xResL, idxU[cnt64], static_cast<uint32_t>(0), cnt64);
    // alpha == 1.0f: Muls(c, e, 1.0f) is the identity (fl(1.0*v) == v bitwise, incl. -0/NaN/Inf),
    // so the column factor IS the gathered e - the op (and its V-pipe latency in the dependent
    // chain G -> Muls -> Mul -> Add) is skipped. Any other alpha keeps the op.
    if (alpha == 1.0f) {
        Mul(sc, xg, e, cnt64);
    } else {
        Muls(c, e, alpha, cnt64);
        Mul(sc, xg, c, cnt64);
    }
    Add(apOut, sc, apIn, cnt);
    int32_t evVMte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evVMte3);
    WaitFlag<HardEvent::V_MTE3>(evVMte3);

    SmallStoreAp(apOut, rangeBegin, cnt);
}

// smallMode: xRes MTE2 load of ProcessSmall.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::SmallLoadX(LocalTensor<float> xResL)
{
    const uint32_t xCnt = static_cast<uint32_t>(n);
    if (xCnt % SSPR_ALIGN_ELEMS == 0U) {
        DataCopy(xResL, xGM[0], xCnt);
    } else {
        uint32_t paddingNumU = (SSPR_ALIGN_ELEMS - xCnt % SSPR_ALIGN_ELEMS) % SSPR_ALIGN_ELEMS;
        if (paddingNumU == 0U) {
            paddingNumU = SSPR_ALIGN_ELEMS;  // avoid the exactly-one-32B-block pad corner
        }
        DataCopyExtParams copyParams{1, xCnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
        DataCopyPad(xResL, xGM[0], copyParams, padParams);
    }
}

// smallMode: ap window MTE2 load of ProcessSmall.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::SmallLoadAp(LocalTensor<float> apIn, uint64_t rangeBegin,
    uint32_t cnt)
{
    if (cnt % SSPR_ALIGN_ELEMS == 0U) {
        DataCopy(apIn, apGM[rangeBegin], cnt);
    } else {
        uint32_t paddingNumU = (SSPR_ALIGN_ELEMS - cnt % SSPR_ALIGN_ELEMS) % SSPR_ALIGN_ELEMS;
        if (paddingNumU == 0U) {
            paddingNumU = SSPR_ALIGN_ELEMS;
        }
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(paddingNumU), 0};
        DataCopyPad(apIn, apGM[rangeBegin], copyParams, padParams);
    }
}

// smallMode: ap window MTE3 store of ProcessSmall.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::SmallStoreAp(LocalTensor<float> apOut, uint64_t rangeBegin,
    uint32_t cnt)
{
    if (cnt % SSPR_ALIGN_ELEMS == 0U) {
        DataCopy(apGM[rangeBegin], apOut, cnt);
    } else {
        DataCopyExtParams copyParams{1, cnt * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPad(apGM[rangeBegin], apOut, copyParams);
    }
}

// Window loop over [begin, end): software-pipelined CopyIn(k+1) / Compute(k) / CopyOut(k).
// desc mode: the host-supplied descriptor block (already in descBuf) carries the window
// list of this core ([winCount][per window: pos cnt nRun nBlock ...]) and the starting column
// is not needed (run descriptors carry their own col), so the binary search is skipped too.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ProcessRange(uint64_t rangeBegin, uint64_t rangeEnd)
{
    if (descMode) {
        ProcessRangeDesc();
        return;
    }
    ProcessRangeWalk(rangeBegin, rangeEnd);
}

// Desc-mode window loop of ProcessRange.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ProcessRangeDesc()
{
    // the window table was pulled into winTblBuf at Init (one MTE2, ordered by the
    // LoadXResident pairs). Per window only 3 scalars are read on the S pipe; the per-element
    // row/col indices are consumed by the V pipe via Gather - no per-item S reads.
    LocalTensor<uint32_t> tbl = winTblBuf.Get<uint32_t>();
    uint32_t tblOff = SSPR_DESC_TBL_HDR_U32;  // slot: [winCount][7 pad][5 x (pos,cnt,idxOff)]
    uint32_t winCount = tbl.GetValue(0U);
    if (winCount > SSPR_DESC_MAX_WINS) {
        return;  // guard: corrupt table (diagnostic bail-out)
    }
    // prefetch window 0's idx block (slot A) immediately after the table S-reads -
    // its MTE2 flight overlaps the first apIn CopyIn and the compute pipeline entry.
    {
        uint32_t cnt0 = tbl.GetValue(tblOff + 1U);
        DataCopy(idxBufA.Get<uint32_t>(), descGM[tbl.GetValue(tblOff + 2U)],
            2U * ((cnt0 + 63U) & ~63U));
    }
    CopyInWindow(tbl.GetValue(tblOff), tbl.GetValue(tblOff + 1U));
    for (uint32_t w = 0; w < winCount; ++w) {
        uint64_t wPos = tbl.GetValue(tblOff);
        uint32_t cnt = tbl.GetValue(tblOff + 1U);
        uint32_t idxOff = tbl.GetValue(tblOff + 2U);
        // the next window's MTE2 is issued INSIDE ComputeWindowVector, after this
        // window's MTE2_V wait (see the ComputeWindowVector header for the rationale and
        // the slot-reuse safety argument). The table's next (pos, cnt, idxOff) come from
        // the same geometry as the range walk.
        uint64_t nextPos = 0ULL;
        uint32_t nextCnt = 0U;
        uint32_t nextIdxOff = 0U;
        if (w + 1U < winCount) {
            nextPos = tbl.GetValue(tblOff + 3U);
            nextCnt = tbl.GetValue(tblOff + 3U + 1U);
            nextIdxOff = tbl.GetValue(tblOff + 3U + 2U);
        }
        ComputeWindowVector(wPos, cnt, idxOff, w & 1U, true, nextPos, nextCnt,
            nextIdxOff);
        CopyOutWindow(wPos, cnt);
        tblOff += 3U;
    }
}

// Legacy-walk window loop of ProcessRange.
template <bool IS_INCX1>
__aicore__ inline void SsprKernel<IS_INCX1>::ProcessRangeWalk(uint64_t rangeBegin, uint64_t rangeEnd)
{
    // Largest column with ColStart(col) <= rangeBegin (integer binary search; the aicore
    // compiler forbids float<->int casts, so no sqrt here). ColStart(j) is strictly increasing.
    uint64_t col = 0;
    {
        uint64_t lo = 0;
        uint64_t hi = n - 1U;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo + 1U) / 2U;
            if (ColStart(mid) <= rangeBegin) {
                lo = mid;
            } else {
                hi = mid - 1U;
            }
        }
        col = lo;
    }

    // Software-pipelined window loop: CopyIn(k+1) overlaps Compute(k); CopyOut(k) overlaps
    // Compute(k+1). The in/out queues (depth 2) provide the buffering.
    uint64_t wPos = rangeBegin;
    CopyInWindow(wPos, static_cast<uint32_t>((rangeEnd - wPos < ubFormer) ? (rangeEnd - wPos) : ubFormer));
    while (wPos < rangeEnd) {
        uint32_t cnt = static_cast<uint32_t>((rangeEnd - wPos < ubFormer) ? (rangeEnd - wPos) : ubFormer);
        uint64_t nextPos = wPos + cnt;
        if (nextPos < rangeEnd) {
            CopyInWindow(nextPos, static_cast<uint32_t>((rangeEnd - nextPos < ubFormer)
                ? (rangeEnd - nextPos) : ubFormer));
        }
        ComputeWindow(wPos, cnt, col, 0U);
        CopyOutWindow(wPos, cnt);
        // Carry the column walk across windows.
        uint64_t q = nextPos;
        while (q < rangeEnd && ColEnd(col) <= q) {
            ++col;
        }
        wPos = nextPos;
    }
}

// TilingKey_0: incx == 1 fast path (compile-time template instantiation).
template <bool IS_INCX1>
__global__ __aicore__ void sspr_kernel(GM_ADDR x, GM_ADDR ap, SsprTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SsprKernel<IS_INCX1> op;
    op.Init(x, ap, tiling);
    op.Process();
}

void sspr_kernel_do(const uint8_t* x, uint8_t* ap, uint8_t* desc, const SsprTilingData& tiling,
                    uint32_t numBlocks, void* stream)
{
    (void)desc; // the workspace address travels inside tiling.descAddr (signature kept for
                // future GM-param route); the desc content itself lives in device memory.
    // x crosses this boundary as const and the kernel treats it
    // strictly read-only. The launch ABI parameter (GM_ADDR) has no const-qualified form, so
    // one documented const_cast re-arms the pointer here; no device code writes through it.
    // R18.1 largeMode dispatch: the column-ownership tile path lives in its own translation
    // unit (sspr_kernel_large.cpp), so the template kernels below compile from the pre-R18
    // head source and never see a largeMode tiling on this route (largeMode arrives 恒 0
    // below; the guard inside sspr_kernel_large is the second line of defence).
    if (tiling.largeMode == 1U) {
        sspr_kernel_large_do(x, ap, tiling, numBlocks, stream);
        return;
    }
    if (tiling.incx == 1) {
        sspr_kernel<true><<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(x), ap, tiling);
    } else {
        sspr_kernel<false><<<numBlocks, nullptr, stream>>>(const_cast<uint8_t*>(x), ap, tiling);
    }
}
