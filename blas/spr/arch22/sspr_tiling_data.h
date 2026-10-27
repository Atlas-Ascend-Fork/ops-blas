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

#include <cstddef>
#include <cstdint>

// UB tile size for one AP block, in float32 elements.
// The incx==1 path raises the window to 3584 (56x64) — fewer windows = fewer per-window
// MTE dispatches/flag round trips; UB budget with the buffer aliasing (11 regions: apIn/apOut
// depth-2 queues + xg/xi(=idx)/th/tl(=mask)/e/s/c) = 11*3584*4B = 157KB + xRes (n<=4100: 16.4KB)
// = ~174KB <= 192KB (UB hardware ceiling).
// The incx!=1 path keeps 2048: extra xPhys (3*(n-1)*4B <= 49.2KB) + oversized idx (n*4B) fit:
// 11*2048*4B + idx-extra 8.2KB + xRes 16.4KB + xPhys 49.2KB ~ 162KB.
// xPhys is UNBOUNDED in |incx| - the host clamps ubFormer (64-elem
// steps, host SsprUbBytes guard) whenever the total crosses the 192KB ceiling, and degrades
// to the scalar legacy path (tiling.scalarMode) when even the 64-elem minimum cannot fit.
constexpr uint32_t SSPR_UB_FORMER = 2048;
constexpr uint32_t SSPR_UB_FORMER_INCX1 = 3584;

// The Phase D finiteness guard (Abs/Compares/Select) must run at count = ceil64(len) - sub-64 /
// non-64-multiple counts are BROKEN on 910B3 (at count = len the
// guard returns the raw |c| value on valid lanes instead of the Inf/NaN fallback sc). For the
// last run of a full window (a + len == cnt == ubFormer, len % 64 != 0) ceil64(len) reaches up
// to 63 elements past ubFormer, so the apOut (both queue slots) / c / sc / tl buffers carry
// +SSPR_GUARD_SLACK_ELEMS of slack to keep every guard access in bounds. SINGLE-SOURCED here:
// consumed by the kernel InitBuffer and by the host SsprUbBytes budget (5 slack units x 4B =
// 1280B in SSPR_UB_FIXED_BYTES; desc mode skips tl, so its share is conservative by 256B).
constexpr uint32_t SSPR_GUARD_SLACK_ELEMS = 64U;

// x resident segment cap in float32 elements. 4104 keeps the supported range
// (n <= 4100) fully resident in UB; larger n falls back to the single-core scalar legacy
// path (host forces useCoreNum = 1). UB budget at cap with incx=3 (worst): xPhys 48KB +
// xRes 16.4KB + idx 16.4KB + apIn/apOut double-buffered 2*(2*8KB) + tmp 8KB ~ 121KB.
constexpr uint32_t SSPR_X_RESIDENT_MAX_FLOATS = 4104;

// Per-core minimum data volume in bytes: MIN_TILING_BITS = 32768 bits = 4096 bytes.
// AP total bytes = n(n+1)/2 * 4B = 2n(n+1); cores are shrunk so each core's share >= this value.
// Degenerate band n <= 44 (2n(n+1) < 4096) converges to the shrink floor: single-core fallback.
constexpr uint64_t SSPR_MIN_CORE_BYTES = 4096;

// Max AIV cores for the host-computed cost-balanced split bounds array (910B3 = 40 AIV).
constexpr uint32_t SSPR_MAX_CORES = 40;

// Row-factor fetch strategy threshold: for large n the per-run GM-unaligned DataCopyPad wins
// (n=4096: 497us vs 588us gather); for small n the MTE2 per-op overhead of many short runs
// loses to the element-level Gather (n=512: 36.6us gather vs 81us pad). Runtime-tunable
// (env SSPR_PAD_MIN) and re-evaluated under the cost-balanced split.
constexpr uint32_t SSPR_PAD_MIN_N = 4000;

// Per-core geometry-descriptor table constants, SINGLE-SOURCED here. Layout of one core's
// table slot:
//   [0] winCount, [1..SSPR_DESC_TBL_HDR_U32-1] pad,
//   then up to SSPR_DESC_MAX_WINS windows x (pos, cnt, idxOff) - 3 u32 per window.
// Host guard: a shape whose worst core block needs more than SSPR_DESC_MAX_WINS windows falls
// back to the kernel-side walk path (sspr_host.cpp SsprBuildDesc).
constexpr uint32_t SSPR_DESC_TBL_SLOT_U32 = 24U;  // 1 + 7 pad + 5 x 3 = 23, slot-padded to 24
constexpr uint32_t SSPR_DESC_TBL_HDR_U32 = 8U;    // window entries start at slot offset 8
constexpr uint32_t SSPR_DESC_MAX_WINS = 5U;       // max windows per core in one descriptor slot

// R18/R19 largeMode (column-ownership tile path) constants, SINGLE-SOURCED here; consumed
// by the kernel (sspr_kernel_large.cpp Init/ProcessLargeColumns) and by the host
// (SsprUbBytesLarge budget mirror, SsprLargeGateEligible gate). Gate (R19): incx == 1 &&
// BOTH fill modes && SSPR_LARGE_MIN_N_UPPER < n <= SSPR_LARGE_MAX_N && desc workspace
// unavailable (guaranteed in that band by the desc volume gate). Column ownership: each core
// owns a weighted range of COLUMNS (kernel-side WeightedColumnBoundary, ported from the !333
// UnifiedContiguousColumns skeleton), so cross-core MTE3 writes may share 32B GM blocks at
// column boundaries (unlike the walk path's 32B-row-aligned element ranges) - the P0
// cross-core partial-row test guards exactly this.
// R19 note: a th/tile-hoist variant (T=8192, 2 tiles for n>8192) was REJECTED - the 2-tile
// reload showed rare intermittent nondeterminism at 16384U (see sspr_kernel_large.cpp header);
// R19 keeps the R18.1 single-tile skeleton, T = 16384 covers the whole gate band.
constexpr uint32_t SSPR_LARGE_TILE_FLOATS = 16384U;      // x row-factor tile (xTile); n <= this
                                                         // stays in one tile per core
constexpr uint32_t SSPR_LARGE_SEG_FLOATS = 2048U;        // segment length, exact band (n > 2048)
constexpr uint32_t SSPR_LARGE_SEG_FLOATS_NAIVE = 4096U;  // segment length, naive band (n <= 2048)
constexpr uint32_t SSPR_LARGE_MIN_N_UPPER = 1300U;       // crossover (BOTH fill modes since
                                                         // R19): below this the walk path still
                                                         // wins (1270U +1% lead)
constexpr uint32_t SSPR_LARGE_MAX_N = 16384U;            // gate cap; tiles cover n at T rows each

// TilingData for sspr (arch22 / DAV_2201, AIV vector route).
struct SsprTilingData {
    uint64_t n;              // matrix order = number of packed columns (host guarantees n > 0 at launch)
    int64_t incx;            // x stride (signed; negative uses x[(n-1-i)*|incx|] reverse mapping)
    uint32_t uplo;           // 121 = ACLBLAS_UPPER / 122 = ACLBLAS_LOWER (column offset & length formulas)
    float alpha;             // scalar multiplier (dereferenced from the Host pointer into tiling)
    uint64_t xPhysLen;       // x physical length = 1 + (n-1)*|incx|
    uint64_t apLen;          // AP packed length = n*(n+1)/2
    uint32_t useCoreNum;     // AIV cores actually used (host: min(GetAivCoreCount(), n, 4KB-share cores))
    uint32_t columnsPerCore; // base columns per core = CeilDiv(n, useCoreNum)
    uint32_t ubFormer;       // UB tile size in float32 elements (64-elem multiple = 256B aligned)
    uint32_t ubLoopMax;      // worst-case UB loop bound = CeilDiv(maxColLen, ubFormer); maxColLen = n
    uint32_t ubTailMax;      // worst-case tail bound = min(ubFormer, maxColLen - (ubLoopMax-1)*ubFormer)
    uint32_t debug;          // debug knob (env SSPR_DEBUG). No on-device stamps are
                             // emitted - x crosses the launch boundary as const and is never
                             // written. The field stays for ABI stability; the kernel
                             // currently ignores it.
    uint32_t runCostW;       // per-crossed-column run-entry cost in equivalent elements for the
                             // cost-balanced core split (0 = legacy element split + UPPER half-share)
    uint32_t padMinN;        // min n for the GM-unaligned DataCopyPad row-factor fetch (env-tunable)
    uint32_t coreBegin[SSPR_MAX_CORES + 1U];  // range start of each core; [useCoreNum] = apLen
    uint32_t descMode;       // 1 = per-core geometry descriptors in the descGM workspace
                             // (host-precomputed walk: kills the kernel walk + scalar Phase E);
                             // 0 = legacy kernel-side walk (large n, desc build failure, env off)
    uint32_t descSlotU32;    // descriptor slot size in u32 units (32B-aligned); valid iff
                             // descMode == 1; core i block starts at descGM[i * descSlotU32]
    uint64_t descAddr;       // device address of the descriptor workspace (valid iff descMode
                             // == 1); passed via tiling so the kernel signature stays 3-arg
    uint32_t scalarMode;     // 1 = degrade to the single-core scalar legacy
                             // path - the !IS_INCX1 resident UB budget cannot close even at the
                             // 64-element minimum window (host SsprUbBytes guard); 0 = vector path.
                             // Must sit at the struct TAIL - fields are append-only; a mid-struct
                             // insertion would shift coreBegin[] and everything after it by 4B,
                             // an ABI break under mixed host/kernel deploys.
    uint32_t smallMode;      // 1 = single-window-per-core straight-line path (ProcessSmall):
                             // desc mode engaged, every core range <= SSPR_DESC_UBF (exactly one
                             // descriptor window per core) and the single-buffer UB budget closes.
                             // The straight-line chain (one batched MTE2 wait, two flag pairs, no
                             // queues / table pull / S reads) removes the per-call fixed overhead
                             // that dominated small-n wall time. append-only tail; the
                             // per-core idx block offset is NOT carried here - the kernel derives
                             // it from descSlotU32*useCoreNum + sum(2*ceil64(core range)) because
                             // every descriptor idx block is exactly 2*ceil64(cnt) u32 (64-relative
                             // padding closes at the block boundary by construction).
    uint32_t largeMode;      // 1 = R18/R19 column-ownership tile path (large-n incx==1, BOTH
                             // fill modes, desc unavailable): per-core weighted COLUMN ranges +
                             // resident xTile/thTile/tlTile (SSPR_LARGE_TILE_FLOATS each, R19
                             // hoists the Dekker split) processed in SSPR_LARGE_SEG_FLOATS-sized
                             // segments; arithmetic chain statement-identical to the walk path
                             // (naive 2-op / Dekker + guard by SSPR_NAIVE_MAX_N band).
                             // append-only tail, after smallMode.
};

// Tail append rule: fields are appended at the struct tail only, so scalarMode and smallMode
// keep their offsets and the total size is unchanged at 272B incl. the 4B tail pad.
static_assert(offsetof(SsprTilingData, smallMode) ==
                  offsetof(SsprTilingData, scalarMode) + sizeof(uint32_t),
              "SsprTilingData ABI drifted (smallMode must follow scalarMode)");

// ABI layout guards. sspr_host.cpp and sspr_kernel.cpp compile from this single
// header, so the layout is consistent by construction; these asserts turn any
// accidental mid-struct insertion (or pad drift) into a build failure instead of silent
// cross-compilation-unit field reinterpretation.
static_assert(offsetof(SsprTilingData, coreBegin) == 72U, "SsprTilingData ABI drifted (coreBegin)");
static_assert(offsetof(SsprTilingData, descMode) ==
                  offsetof(SsprTilingData, coreBegin) + (SSPR_MAX_CORES + 1U) * sizeof(uint32_t),
              "SsprTilingData ABI drifted (fields after coreBegin are append-only)");
static_assert(offsetof(SsprTilingData, descAddr) == 248U,
              "SsprTilingData ABI drifted (descAddr; 4B pad after descSlotU32 for 8B alignment)");
static_assert(offsetof(SsprTilingData, scalarMode) ==
                  offsetof(SsprTilingData, descAddr) + sizeof(uint64_t),
              "SsprTilingData ABI drifted (fields after descAddr are append-only)");
static_assert(offsetof(SsprTilingData, largeMode) ==
                  offsetof(SsprTilingData, smallMode) + sizeof(uint32_t),
              "SsprTilingData ABI drifted (largeMode must follow smallMode)");
static_assert(sizeof(SsprTilingData) == 272U, "SsprTilingData ABI drifted (total size)");
