
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "caxpy_tiling_data.h"
#include "caxpy_dispatch_policy.h"

#if __DAV_C220_VEC__
// Unified Buffer capacity on 910B3 (192 KiB). All *_ub_layout() functions below
// must fit their working set within this budget; the static_asserts next to
// each layout enforce it at compile time instead of relying on manual review.
constexpr uint32_t CAXPY_UB_BUDGET_BYTES = 192 * 1024;

// --- Dense (contiguous incx=1, incy=1) kernel UB layout -------------------
// Two ping-pong slots for the load buffer (ubX) and the output buffer (ubOut),
// each sized to one full tile (38 KiB), followed by a full-tile offset table
// (ubOffset) and a small 8-float coefficient table (ubCoefficient).
constexpr uint32_t CAXPY_DENSE_TILE_FLOATS = 38 * 1024 / sizeof(float);
constexpr uint32_t CAXPY_DENSE_TILE_BYTES = 38 * 1024;
constexpr uint32_t CAXPY_DENSE_UB_OUT0_BYTES = 0 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_OUT1_BYTES = 1 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_X0_BYTES = 2 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_X1_BYTES = 3 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_OFFSET_BYTES = 4 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_COEFFICIENT_BYTES = 5 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_DENSE_UB_COEFFICIENT_COUNT = 8;
// Offset table region: one uint32 pair-swap offset per float in a full tile.
// The trailing terms mirror the strided-kernel GM mask layout below this
// function (CAXPY_X_OFFSET_COUNT and 2*CAXPY_SPAN_OFFSET_COUNT, both defined
// later in this file for the strided kernels' own UB tables): the host lays
// out gmAug as [compute offsets][x-normalize offsets][store offsets]
// [y-gather offsets][dense swap offsets], so this dense-kernel offset into
// gmAug must skip past the strided sections that precede it in that layout,
// even though the dense and strided kernels are otherwise independent code
// paths. See caxpy_host.cpp's DENSE_SWAP_OFFSET_START for the authoritative
// host-side computation this mirrors.
constexpr uint32_t CAXPY_DENSE_SWAP_SKIP_X_OFFSET_WORDS = 512 * 8 + 64;          // == CAXPY_X_OFFSET_COUNT
constexpr uint32_t CAXPY_DENSE_SWAP_SKIP_SPAN_OFFSET_WORDS = 2 * (512 * 2 + 64); // == 2 * CAXPY_SPAN_OFFSET_COUNT
constexpr uint32_t CAXPY_DENSE_SWAP_OFFSET_START =
    CAXPY_DENSE_TILE_FLOATS + CAXPY_DENSE_SWAP_SKIP_X_OFFSET_WORDS + CAXPY_DENSE_SWAP_SKIP_SPAN_OFFSET_WORDS;
static_assert(
    CAXPY_DENSE_UB_COEFFICIENT_BYTES + CAXPY_DENSE_UB_COEFFICIENT_COUNT * sizeof(float) <= CAXPY_UB_BUDGET_BYTES,
    "CAXPY dense UB layout exceeds the 910B3 Unified Buffer budget");

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_uint32(
    __ubuf__ uint32_t* dst, __gm__ uint32_t* src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(uint32_t);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    __ubuf__ float* dst, __gm__ float* src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(
    __gm__ float* dst, __ubuf__ float* src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) uint32_t abs_increment(int64_t increment)
{
    return increment > 0 ? static_cast<uint32_t>(increment) : static_cast<uint32_t>(-increment);
}

__aicore__ __inline__ __attribute__((always_inline)) bool copy_complex_gm2ub_signed(
    __ubuf__ float* dst, __ubuf__ float* denseTemp, __gm__ float* source, uint32_t logicalOffset,
    uint32_t totalComplexCount, uint32_t complexCount, int64_t signedIncrement)
{
    constexpr uint32_t maxBurstCount = 4088;
    const uint32_t increment = abs_increment(signedIncrement);
    const uint32_t physicalOffset = signedIncrement > 0 ?
                                        logicalOffset * increment :
                                        (totalComplexCount - logicalOffset - complexCount) * increment;
    auto src = source + physicalOffset * 2;
    if (increment == 1) {
        copy_gm_to_ubuf_align_b32(denseTemp, src, 0, 1, complexCount * 2 * sizeof(float), 0, 0, 0, 0);
        return true;
    }
    const uint32_t gapBytes = (increment - 1) * 2 * sizeof(float);
    uint32_t copied = 0;
    while (copied < complexCount) {
        const uint32_t batch = (complexCount - copied) > maxBurstCount ? maxBurstCount : complexCount - copied;
        asc_copy_gm2ub_align(
            dst + copied * 8, src + copied * increment * 2, static_cast<uint16_t>(batch), 2 * sizeof(float), 0, 0,
            gapBytes, 0);
        copied += batch;
    }
    return false;
}

__aicore__ __inline__ __attribute__((always_inline)) void normalize_loaded_complexes(
    __ubuf__ float* dst, __ubuf__ float* denseTemp, uint32_t complexCount, bool denseLoaded, bool reverse)
{
    if (denseLoaded) {
        for (uint32_t i = 0; i < complexCount; ++i) {
            const uint32_t sourceIndex = reverse ? complexCount - 1 - i : i;
            dst[i * 8] = denseTemp[sourceIndex * 2];
            dst[i * 8 + 1] = denseTemp[sourceIndex * 2 + 1];
        }
        return;
    }
    if (reverse) {
        for (uint32_t left = 0; left < complexCount / 2; ++left) {
            const uint32_t right = complexCount - 1 - left;
            const float leftReal = dst[left * 8];
            const float leftImag = dst[left * 8 + 1];
            dst[left * 8] = dst[right * 8];
            dst[left * 8 + 1] = dst[right * 8 + 1];
            dst[right * 8] = leftReal;
            dst[right * 8 + 1] = leftImag;
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_complex_ub2gm_signed(
    __gm__ float* destination, __ubuf__ float* src, uint32_t logicalOffset, uint32_t totalComplexCount,
    uint32_t complexCount, int64_t signedIncrement)
{
    constexpr uint32_t maxBurstCount = 4088;
    const uint32_t increment = abs_increment(signedIncrement);
    const uint32_t physicalOffset = signedIncrement > 0 ?
                                        logicalOffset * increment :
                                        (totalComplexCount - logicalOffset - complexCount) * increment;
    auto dst = destination + physicalOffset * 2;
    if (increment == 1) {
        copy_ubuf_to_gm_align_b32(dst, src, 0, 1, complexCount * 2 * sizeof(float), 0, 0, 0, 0);
        return;
    }
    const uint32_t gapBytes = (increment - 1) * 2 * sizeof(float);
    uint32_t copied = 0;
    while (copied < complexCount) {
        const uint32_t batch = (complexCount - copied) > maxBurstCount ? maxBurstCount : complexCount - copied;
        asc_copy_ub2gm_align(
            dst + copied * increment * 2, src + copied * 8, static_cast<uint16_t>(batch), 2 * sizeof(float), 0, 0, 0,
            gapBytes);
        copied += batch;
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_compute_planar(
    __ubuf__ float* resultReal, __ubuf__ float* resultImag, __ubuf__ float* sourceReal, __ubuf__ float* sourceImag,
    float alphaReal, float alphaImag, uint32_t repeat)
{
    // Keep every vector address 32-byte aligned. An interleaved implementation using
    // source + 1 is mathematically equivalent, but faults on C220 vector instructions.
    vmuls(resultReal, sourceReal, alphaReal, repeat, 1, 1, 8, 8);
    vmuls(resultImag, sourceImag, alphaReal, repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vaxpy(resultReal, sourceImag, -alphaImag, repeat, 1, 1, 8, 8);
    vaxpy(resultImag, sourceReal, alphaImag, repeat, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_load(
    __gm__ float* gmX, __ubuf__ float* ubX, uint32_t copyLength, uint32_t eventId)
{
    copy_vec_gm2ub(ubX, gmX, copyLength);
    set_flag(PIPE_MTE2, PIPE_V, eventId);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_compute(
    __ubuf__ float* ubX, __ubuf__ float* ubOut, __ubuf__ uint32_t* ubOffset, __ubuf__ float* ubCoefficient,
    float alphaReal, uint32_t computeLength, uint32_t eventId)
{
    const uint32_t repeat = (computeLength + 63) / 64;

    wait_flag(PIPE_MTE2, PIPE_V, eventId);
    // Turn [real, imag] into [imag, real]. The alternating coefficient then
    // applies [-alphaImag, +alphaImag] without splitting into planar buffers.
    vgather(reinterpret_cast<__ubuf__ uint32_t*>(ubOut), ubOffset, (uintptr_t)ubX, 8, repeat);
    pipe_barrier(PIPE_V);
    vmuls(ubX, ubX, alphaReal, repeat, 1, 1, 8, 8);
    vmul(ubOut, ubOut, ubCoefficient, repeat, 1, 1, 0, 8, 8, 0);
    pipe_barrier(PIPE_V);
    vadd(ubOut, ubX, ubOut, repeat, 1, 1, 1, 8, 8, 8);
    set_flag(PIPE_V, PIPE_MTE3, eventId);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_store(
    __gm__ float* gmY, __ubuf__ float* ubOut, uint32_t copyLength, uint32_t eventId)
{
    wait_flag(PIPE_V, PIPE_MTE3, eventId);
    copy_vec_ub2gm(gmY, ubOut, copyLength);
    set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_prepare(
    __gm__ uint32_t* gmAug, __ubuf__ uint32_t*& ubOffset, __ubuf__ float*& ubCoefficient, float alphaImag)
{
    constexpr uint32_t tileFloats = CAXPY_DENSE_TILE_FLOATS;
    ubOffset = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_DENSE_UB_OFFSET_BYTES);
    ubCoefficient = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_DENSE_UB_COEFFICIENT_BYTES);
    copy_vec_gm2ub_uint32(ubOffset, gmAug + CAXPY_DENSE_SWAP_OFFSET_START, tileFloats);
    for (uint32_t index = 0; index < CAXPY_DENSE_UB_COEFFICIENT_COUNT; ++index)
        ubCoefficient[index] = (index & 1) == 0 ? -alphaImag : alphaImag;
    set_flag(PIPE_S, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID2);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_load_next_if_any(
    __gm__ float* gmX, __ubuf__ float* ubX[2], uint32_t offset, uint32_t calNum, uint32_t tileFloats, uint32_t tile,
    uint32_t tileCount, uint32_t current)
{
    if (tile + 1 >= tileCount)
        return;
    const uint32_t next = 1 - current;
    const uint32_t nextEvent = next == 0 ? EVENT_ID0 : EVENT_ID1;
    const uint32_t nextOffset = (tile + 1) * tileFloats;
    const uint32_t nextCount = calNum - nextOffset > tileFloats ? tileFloats : calNum - nextOffset;
    wait_flag(PIPE_MTE3, PIPE_MTE2, nextEvent);
    caxpy_dense_load(gmX + offset + nextOffset, ubX[next], nextCount, nextEvent);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_dense_run(
    __gm__ float* gmX, __gm__ uint32_t* gmAug, __gm__ float* gmY, float alphaReal, float alphaImag, uint32_t offset,
    uint32_t calNum)
{
    constexpr uint32_t tileFloats = CAXPY_DENSE_TILE_FLOATS;
    __ubuf__ float* ubOut[2] = {
        reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_DENSE_UB_OUT0_BYTES),
        reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_DENSE_UB_OUT1_BYTES)};
    __ubuf__ float* ubX[2] = {
        reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_DENSE_UB_X0_BYTES),
        reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_DENSE_UB_X1_BYTES)};
    __ubuf__ uint32_t* ubOffset = nullptr;
    __ubuf__ float* ubCoefficient = nullptr;
    caxpy_dense_prepare(gmAug, ubOffset, ubCoefficient, alphaImag);

    set_atomic_add();
    set_atomic_f32();
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    const uint32_t tileCount = (calNum + tileFloats - 1) / tileFloats;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    const uint32_t firstCount = calNum > tileFloats ? tileFloats : calNum;
    caxpy_dense_load(gmX + offset, ubX[0], firstCount, EVENT_ID0);

    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t current = tile & 1;
        const uint32_t currentEvent = current == 0 ? EVENT_ID0 : EVENT_ID1;
        const uint32_t currentOffset = tile * tileFloats;
        const uint32_t currentCount = calNum - currentOffset > tileFloats ? tileFloats : calNum - currentOffset;

        caxpy_dense_load_next_if_any(gmX, ubX, offset, calNum, tileFloats, tile, tileCount, current);

        caxpy_dense_compute(ubX[current], ubOut[current], ubOffset, ubCoefficient, alphaReal, tileFloats, currentEvent);
        caxpy_dense_store(gmY + offset + currentOffset, ubOut[current], currentCount, currentEvent);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
}

constexpr uint32_t CAXPY_MAX_DATA_COUNT = 38 * 1024 / sizeof(float);
constexpr uint32_t CAXPY_UB_COMPLEX_CAPACITY = CAXPY_MAX_DATA_COUNT / 2;
// CAXPY_STRIDED_TILE_COUNT is defined once in caxpy_dispatch_policy.h (shared
// with the host's shared-tile-capacity computation); this file must not
// redefine it, or the two could silently drift apart the way the host's
// GenXNormalizeOffsets 16-element compute-offset table drifted from the
// device's copy size before that was fixed.
static_assert(CAXPY_STRIDED_TILE_COUNT == 512, "CAXPY_STRIDED_TILE_COUNT definition moved or changed unexpectedly");
constexpr uint32_t CAXPY_INTERLEAVE_COUNT = 16;
constexpr uint32_t CAXPY_BLOCKED_FLOATS = 8;
constexpr uint32_t CAXPY_GATHER_PADDING = 64;
constexpr uint32_t CAXPY_X_OFFSET_COUNT = CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS + CAXPY_GATHER_PADDING;
constexpr uint32_t CAXPY_SPAN_OFFSET_COUNT = CAXPY_STRIDED_TILE_COUNT * 2 + CAXPY_GATHER_PADDING;
static_assert(
    CAXPY_DENSE_SWAP_SKIP_X_OFFSET_WORDS == CAXPY_X_OFFSET_COUNT,
    "CAXPY_DENSE_SWAP_OFFSET_START's skip term no longer matches CAXPY_X_OFFSET_COUNT");
static_assert(
    CAXPY_DENSE_SWAP_SKIP_SPAN_OFFSET_WORDS == 2 * CAXPY_SPAN_OFFSET_COUNT,
    "CAXPY_DENSE_SWAP_OFFSET_START's skip term no longer matches CAXPY_SPAN_OFFSET_COUNT");
constexpr uint32_t CAXPY_STORE_OFFSET_START = CAXPY_MAX_DATA_COUNT + CAXPY_X_OFFSET_COUNT;
constexpr uint32_t CAXPY_Y_GATHER_OFFSET_START = CAXPY_STORE_OFFSET_START + CAXPY_SPAN_OFFSET_COUNT;

// --- Strided kernel UB byte layout -----------------------------------------
// Named byte offsets for the "shared" ping-pong buffer pair (strided_ub_layout,
// used directly by run_shared_pipeline slot 0, and by disjoint/shared variants
// via the overrides below). Each of x/work/y/lowScratch gets a 38 KiB region
// (CAXPY_DENSE_TILE_BYTES, reused here rather than duplicated: the strided
// blocked layout only needs CAXPY_STRIDED_TILE_COUNT*CAXPY_BLOCKED_FLOATS=16 KiB
// per region, but the historical layout budgets a full 38 KiB slot per region
// to match the dense kernel's tile size, leaving headroom after each xDense/
// workDense sub-offset). The offset-table region follows after 4 such slots.
constexpr uint32_t CAXPY_STRIDED_UB_X_BYTES = 0 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_STRIDED_UB_WORK_BYTES = 1 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_STRIDED_UB_Y_BYTES = 2 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_STRIDED_UB_LOW_SCRATCH_BYTES = 3 * CAXPY_DENSE_TILE_BYTES;
constexpr uint32_t CAXPY_STRIDED_UB_OFFSET_TABLES_BYTES = 4 * CAXPY_DENSE_TILE_BYTES;
// Offset table sub-regions, laid out consecutively starting at
// CAXPY_STRIDED_UB_OFFSET_TABLES_BYTES: xNormalizeOffset, then computeOffset
// (sized for one CAXPY_INTERLEAVE_COUNT sub-chunk: convertRepeat<=2, one
// offset pair per element -> 2*8=16 words, rounded up to the 2*64 historical
// allocation for headroom), then yGatherOffset, then storeOffset.
constexpr uint32_t CAXPY_COMPUTE_OFFSET_COUNT = 2 * 64;
constexpr uint32_t CAXPY_STRIDED_UB_COMPUTE_OFFSET_BYTES =
    CAXPY_STRIDED_UB_OFFSET_TABLES_BYTES + CAXPY_X_OFFSET_COUNT * sizeof(uint32_t);
constexpr uint32_t CAXPY_STRIDED_UB_Y_GATHER_OFFSET_BYTES =
    CAXPY_STRIDED_UB_COMPUTE_OFFSET_BYTES + CAXPY_COMPUTE_OFFSET_COUNT * sizeof(uint32_t);
constexpr uint32_t CAXPY_STRIDED_UB_STORE_OFFSET_BYTES =
    CAXPY_STRIDED_UB_Y_GATHER_OFFSET_BYTES + CAXPY_SPAN_OFFSET_COUNT * sizeof(uint32_t);
constexpr uint32_t CAXPY_STRIDED_UB_Y_COMPACT_BYTES =
    CAXPY_STRIDED_UB_STORE_OFFSET_BYTES + CAXPY_SPAN_OFFSET_COUNT * sizeof(uint32_t);
static_assert(
    CAXPY_STRIDED_UB_OFFSET_TABLES_BYTES == 152 * 1024,
    "CAXPY strided offset-table region start moved; update disjoint/shared_ub_layout overrides");
static_assert(
    CAXPY_STRIDED_UB_Y_COMPACT_BYTES + CAXPY_STRIDED_TILE_COUNT * 2 * sizeof(float) <= CAXPY_UB_BUDGET_BYTES,
    "CAXPY strided UB layout exceeds the 910B3 Unified Buffer budget");

struct CaxpyStridedUb {
    __ubuf__ float* x;
    __ubuf__ float* work;
    __ubuf__ float* y;
    __ubuf__ uint32_t* lowScratch;
    __ubuf__ float* xDense;
    __ubuf__ float* yDense;
    __ubuf__ float* workDense;
    __ubuf__ uint32_t* xNormalizeOffset;
    __ubuf__ uint32_t* computeOffset;
    __ubuf__ uint32_t* yGatherOffset;
    __ubuf__ uint32_t* storeOffset;
    __ubuf__ float* yCompact;
};

__aicore__ __inline__ __attribute__((always_inline)) CaxpyStridedUb strided_ub_layout()
{
    CaxpyStridedUb ub;
    ub.x = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_STRIDED_UB_X_BYTES);
    ub.work = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_STRIDED_UB_WORK_BYTES);
    ub.y = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_STRIDED_UB_Y_BYTES);
    ub.lowScratch = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_STRIDED_UB_LOW_SCRATCH_BYTES);
    ub.xDense = ub.x + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    ub.yDense = ub.y + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    ub.workDense = ub.work + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    ub.xNormalizeOffset = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_STRIDED_UB_OFFSET_TABLES_BYTES);
    ub.computeOffset = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_STRIDED_UB_COMPUTE_OFFSET_BYTES);
    ub.yGatherOffset = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_STRIDED_UB_Y_GATHER_OFFSET_BYTES);
    ub.storeOffset = reinterpret_cast<__ubuf__ uint32_t*>((uintptr_t)CAXPY_STRIDED_UB_STORE_OFFSET_BYTES);
    ub.yCompact = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_STRIDED_UB_Y_COMPACT_BYTES);
    return ub;
}

__aicore__ __inline__ __attribute__((always_inline)) void strided_preload_offsets(
    CaxpyStridedUb& ub, __gm__ uint32_t* gmAug, bool useXSpanGather, bool useYSpanGather)
{
    copy_vec_gm2ub_uint32(ub.computeOffset, gmAug, CAXPY_COMPUTE_OFFSET_COUNT);
    if (useXSpanGather) {
        copy_vec_gm2ub_uint32(ub.xNormalizeOffset, gmAug + CAXPY_MAX_DATA_COUNT, CAXPY_X_OFFSET_COUNT);
    }
    if (useYSpanGather) {
        copy_vec_gm2ub_uint32(ub.yGatherOffset, gmAug + CAXPY_Y_GATHER_OFFSET_START, CAXPY_SPAN_OFFSET_COUNT);
        copy_vec_gm2ub_uint32(ub.storeOffset, gmAug + CAXPY_STORE_OFFSET_START, CAXPY_SPAN_OFFSET_COUNT);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) bool strided_load_x_span(
    CaxpyStridedUb& ub, __gm__ float* gmX, uint32_t logicalOffset, uint32_t totalCount, uint32_t count, int64_t incx,
    bool useSpanGather)
{
    if (!useSpanGather) {
        return copy_complex_gm2ub_signed(ub.x, ub.xDense, gmX, logicalOffset, totalCount, count, incx);
    }
    const uint32_t increment = abs_increment(incx);
    const uint32_t physicalOffset =
        incx > 0 ? logicalOffset * increment : (totalCount - logicalOffset - count) * increment;
    copy_vec_gm2ub(
        reinterpret_cast<__ubuf__ float*>(ub.lowScratch), gmX + physicalOffset * 2, (count - 1) * increment * 2 + 2);
    return false;
}

__aicore__ __inline__ __attribute__((always_inline)) void strided_normalize_inputs(
    CaxpyStridedUb& ub, uint32_t count, int64_t incx, int64_t incy, bool xDense, bool yDense, bool useXSpanGather,
    bool useYSpanGather)
{
    if (useXSpanGather) {
        vgather(
            reinterpret_cast<__ubuf__ uint32_t*>(ub.x), ub.xNormalizeOffset, (uintptr_t)ub.lowScratch, 8,
            (count + 7) / 8);
    }
    if (useYSpanGather) {
        vgather(
            reinterpret_cast<__ubuf__ uint32_t*>(ub.yCompact), ub.yGatherOffset, (uintptr_t)ub.y, 8,
            (count * 2 + 63) / 64);
    }
    const bool normalizeX = xDense || incx < 0;
    const bool normalizeY = incy != 1 && !useYSpanGather && (yDense || incy < 0);
    if (!normalizeX && !normalizeY)
        return;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    if (normalizeX)
        normalize_loaded_complexes(ub.x, ub.xDense, count, xDense, incx < 0);
    if (normalizeY)
        normalize_loaded_complexes(ub.y, ub.yDense, count, yDense, incy < 0);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

__aicore__ __inline__ __attribute__((always_inline)) void strided_compute_tile(
    CaxpyStridedUb& ub, float real, float imag, uint32_t count, int64_t incy, bool useYSpanGather)
{
    auto workReal = ub.workDense;
    auto workImag = workReal + CAXPY_UB_COMPLEX_CAPACITY;
    for (uint32_t converted = 0; converted < count; converted += CAXPY_INTERLEAVE_COUNT) {
        const uint32_t convertCount =
            count - converted > CAXPY_INTERLEAVE_COUNT ? CAXPY_INTERLEAVE_COUNT : count - converted;
        const uint32_t convertRepeat = (convertCount + 7) / 8;
        const uint32_t computeRepeat = (convertCount + 63) / 64;
        const uint32_t blockedOffset = converted * CAXPY_BLOCKED_FLOATS;
        auto xReal = ub.x + blockedOffset;
        auto xImag = xReal + CAXPY_UB_COMPLEX_CAPACITY;
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(workReal), reinterpret_cast<__ubuf__ uint32_t*>(xReal), nullptr,
            convertRepeat, 1, 1, 8, 8);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(workImag), reinterpret_cast<__ubuf__ uint32_t*>(xReal), nullptr,
            convertRepeat, 1, 2, 8, 8);
        pipe_barrier(PIPE_V);
        caxpy_compute_planar(xReal, xImag, workReal, workImag, real, imag, computeRepeat);
        vgather(
            reinterpret_cast<__ubuf__ uint32_t*>(ub.work + blockedOffset), ub.computeOffset, (uintptr_t)xReal, 8,
            convertRepeat);
        pipe_barrier(PIPE_V);
        if (incy != 1 && !useYSpanGather) {
            vadd(
                ub.work + blockedOffset, ub.work + blockedOffset, ub.y + blockedOffset, convertRepeat, 1, 1, 1, 8, 8,
                8);
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) CaxpyStridedUb disjoint_ub_layout(uint32_t index)
{
    CaxpyStridedUb ub = strided_ub_layout();
    ub.x = reinterpret_cast<__ubuf__ float*>(
        (uintptr_t)(index == 0 ? CAXPY_STRIDED_UB_X_BYTES : CAXPY_STRIDED_UB_WORK_BYTES));
    ub.work = reinterpret_cast<__ubuf__ float*>(
        (uintptr_t)(index == 0 ? CAXPY_STRIDED_UB_Y_BYTES : CAXPY_STRIDED_UB_LOW_SCRATCH_BYTES));
    ub.xDense = ub.x + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    ub.workDense = ub.work + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    return ub;
}

__aicore__ __inline__ __attribute__((always_inline)) void disjoint_load_tile(
    CaxpyStridedUb& ub, __gm__ float* gmX, uint32_t logicalOffset, uint32_t totalCount, uint32_t count, int64_t incx,
    uint32_t eventId)
{
    copy_complex_gm2ub_signed(ub.x, ub.xDense, gmX, logicalOffset, totalCount, count, incx);
    set_flag(PIPE_MTE2, PIPE_V, eventId);
}

__aicore__ __inline__ __attribute__((always_inline)) void disjoint_compute_tile(
    CaxpyStridedUb& ub, float alphaReal, float alphaImag, uint32_t count, int64_t incx, uint32_t eventId)
{
    wait_flag(PIPE_MTE2, PIPE_V, eventId);
    strided_normalize_inputs(ub, count, incx, 1, false, false, false, false);
    strided_compute_tile(ub, alphaReal, alphaImag, count, 1, false);
}

__aicore__ __inline__ __attribute__((always_inline)) void strided_store_tile(
    CaxpyStridedUb& ub, __gm__ float* gmY, uint32_t logicalOffset, uint32_t totalCount, uint32_t count, int64_t incy,
    uint32_t eventId)
{
    __ubuf__ float* source = ub.work;
    if (incy == 1) {
        const uint32_t repeat = (count * 2 + 63) / 64;
        vgather(reinterpret_cast<__ubuf__ uint32_t*>(ub.workDense), ub.storeOffset, (uintptr_t)ub.work, 8, repeat);
        pipe_barrier(PIPE_V);
        source = ub.workDense;
    }
    set_flag(PIPE_V, PIPE_MTE3, eventId);
    wait_flag(PIPE_V, PIPE_MTE3, eventId);
    copy_complex_ub2gm_signed(gmY, source, logicalOffset, totalCount, count, incy);
    set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
}

__aicore__ __inline__ __attribute__((always_inline)) void disjoint_preload_offsets(
    CaxpyStridedUb& ping, __gm__ uint32_t* gmAug, int64_t incy)
{
    copy_vec_gm2ub_uint32(ping.computeOffset, gmAug, CAXPY_COMPUTE_OFFSET_COUNT);
    if (incy == 1)
        copy_vec_gm2ub_uint32(ping.storeOffset, gmAug + CAXPY_STORE_OFFSET_START, CAXPY_SPAN_OFFSET_COUNT);
}

__aicore__ __inline__ __attribute__((always_inline)) void disjoint_load_next_if_any(
    CaxpyStridedUb& ping, CaxpyStridedUb& pong, __gm__ float* gmX, uint32_t logicalOffset, uint32_t totalComplexCount,
    uint32_t complexCount, uint32_t tileCapacity, int64_t incx, uint32_t tile, uint32_t tileCount, uint32_t current)
{
    if (tile + 1 >= tileCount)
        return;
    const uint32_t next = 1 - current;
    const uint32_t nextEvent = next == 0 ? EVENT_ID0 : EVENT_ID1;
    CaxpyStridedUb& nextUb = next == 0 ? ping : pong;
    const uint32_t nextOffset = (tile + 1) * tileCapacity;
    const uint32_t nextCount = complexCount - nextOffset > tileCapacity ? tileCapacity : complexCount - nextOffset;
    wait_flag(PIPE_MTE3, PIPE_MTE2, nextEvent);
    disjoint_load_tile(nextUb, gmX, logicalOffset + nextOffset, totalComplexCount, nextCount, incx, nextEvent);
}

__aicore__ __inline__ __attribute__((always_inline)) void run_disjoint_pipeline(
    __gm__ float* gmX, __gm__ uint32_t* gmAug, __gm__ float* gmY, float alphaReal, float alphaImag,
    uint32_t logicalOffset, uint32_t totalComplexCount, uint32_t complexCount, int64_t incx, int64_t incy)
{
    CaxpyStridedUb ping = disjoint_ub_layout(0);
    CaxpyStridedUb pong = disjoint_ub_layout(1);
    disjoint_preload_offsets(ping, gmAug, incy);

    set_atomic_add();
    set_atomic_f32();
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    constexpr uint32_t tileCapacity = CAXPY_STRIDED_TILE_COUNT;
    const uint32_t tileCount = (complexCount + tileCapacity - 1) / tileCapacity;
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    const uint32_t firstCount = complexCount > tileCapacity ? tileCapacity : complexCount;
    disjoint_load_tile(ping, gmX, logicalOffset, totalComplexCount, firstCount, incx, EVENT_ID0);

    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t current = tile & 1;
        const uint32_t currentEvent = current == 0 ? EVENT_ID0 : EVENT_ID1;
        CaxpyStridedUb& currentUb = current == 0 ? ping : pong;
        const uint32_t currentOffset = tile * tileCapacity;
        const uint32_t currentCount =
            complexCount - currentOffset > tileCapacity ? tileCapacity : complexCount - currentOffset;

        disjoint_load_next_if_any(
            ping, pong, gmX, logicalOffset, totalComplexCount, complexCount, tileCapacity, incx, tile, tileCount,
            current);

        disjoint_compute_tile(currentUb, alphaReal, alphaImag, currentCount, incx, currentEvent);
        strided_store_tile(
            currentUb, gmY, logicalOffset + currentOffset, totalComplexCount, currentCount, incy, currentEvent);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
}

__aicore__ __inline__ __attribute__((always_inline)) CaxpyStridedUb shared_ub_layout(uint32_t index)
{
    CaxpyStridedUb ub = strided_ub_layout();
    if (index != 0) {
        ub.work = reinterpret_cast<__ubuf__ float*>((uintptr_t)CAXPY_STRIDED_UB_Y_BYTES);
        ub.workDense = ub.work + CAXPY_STRIDED_TILE_COUNT * CAXPY_BLOCKED_FLOATS;
    }
    return ub;
}

__aicore__ __inline__ __attribute__((always_inline)) void shared_load_tile(
    CaxpyStridedUb& ub, __gm__ float* gmX, uint32_t logicalOffset, uint32_t totalCount, uint32_t count, int64_t incx,
    uint32_t waitEvent, uint32_t readyEvent, bool waitForRaw)
{
    // MTE3 owns each work buffer after a store. Hand it back through MTE2,
    // which is the event direction used by the stable dense pipeline.
    wait_flag(PIPE_MTE3, PIPE_MTE2, readyEvent);
    if (waitForRaw)
        wait_flag(PIPE_V, PIPE_MTE2, waitEvent);
    strided_load_x_span(ub, gmX, logicalOffset, totalCount, count, incx, true);
    set_flag(PIPE_MTE2, PIPE_V, readyEvent);
}

__aicore__ __inline__ __attribute__((always_inline)) void shared_compute_tile(
    CaxpyStridedUb& ub, float alphaReal, float alphaImag, uint32_t count, int64_t incx, uint32_t eventId)
{
    wait_flag(PIPE_MTE2, PIPE_V, eventId);
    strided_normalize_inputs(ub, count, incx, 1, false, false, true, false);
    // raw span is dead immediately after gather/normalization. Release it to
    // MTE2 before the remaining complex arithmetic finishes on PIPE_V.
    set_flag(PIPE_V, PIPE_MTE2, eventId);
    strided_compute_tile(ub, alphaReal, alphaImag, count, 1, false);
}

__aicore__ __inline__ __attribute__((always_inline)) void shared_tile_bounds(
    uint32_t complexCount, uint32_t tileCount, uint32_t tile, uint32_t tileCapacity, uint32_t& tileOffset,
    uint32_t& tileLength)
{
    // A final tile shorter than the interleave width is mis-staged by the span-read
    // path, so avoid producing one: when the natural schedule would leave such a
    // remainder, split the last two tiles evenly instead. Coverage stays exact and
    // contiguous and no tile exceeds the capacity, so buffer sizing is unchanged.
    // tileCapacity is a runtime value (see run_shared_pipeline) that is always
    // >= CAXPY_SHARED_MIN_USABLE_TILE_CAPACITY (CAXPY_INTERLEAVE_COUNT + 1), so
    // this rebalancing always has room to produce two tiles of at least
    // CAXPY_INTERLEAVE_COUNT/2 elements each.
    const uint32_t remainder = complexCount - (tileCount - 1) * tileCapacity;
    if (tileCount >= 2 && remainder < CAXPY_INTERLEAVE_COUNT + 1) {
        const uint32_t splitBase = (tileCount - 2) * tileCapacity;
        if (tile < tileCount - 2) {
            tileOffset = tile * tileCapacity;
            tileLength = tileCapacity;
            return;
        }
        const uint32_t splitTotal = complexCount - splitBase;
        const uint32_t firstHalf = (splitTotal + 1) / 2;
        if (tile == tileCount - 2) {
            tileOffset = splitBase;
            tileLength = firstHalf;
        } else {
            tileOffset = splitBase + firstHalf;
            tileLength = splitTotal - firstHalf;
        }
        return;
    }
    tileOffset = tile * tileCapacity;
    tileLength = complexCount - tileOffset > tileCapacity ? tileCapacity : complexCount - tileOffset;
}

__aicore__ __inline__ __attribute__((always_inline)) void shared_preload_offsets(
    CaxpyStridedUb& ping, __gm__ uint32_t* gmAug, int64_t incy)
{
    strided_preload_offsets(ping, gmAug, true, false);
    if (incy == 1)
        copy_vec_gm2ub_uint32(ping.storeOffset, gmAug + CAXPY_STORE_OFFSET_START, CAXPY_SPAN_OFFSET_COUNT);
}

__aicore__ __inline__ __attribute__((always_inline)) void shared_load_next_if_any(
    CaxpyStridedUb& ping, CaxpyStridedUb& pong, __gm__ float* gmX, uint32_t logicalOffset, uint32_t totalComplexCount,
    uint32_t complexCount, uint32_t tileCapacity, int64_t incx, uint32_t tile, uint32_t tileCount, uint32_t current)
{
    if (tile + 1 >= tileCount)
        return;
    const uint32_t next = 1 - current;
    const uint32_t nextEvent = next == 0 ? EVENT_ID0 : EVENT_ID1;
    const uint32_t currentEvent = current == 0 ? EVENT_ID0 : EVENT_ID1;
    CaxpyStridedUb& nextUb = next == 0 ? ping : pong;
    uint32_t nextOffset = 0;
    uint32_t nextCount = 0;
    shared_tile_bounds(complexCount, tileCount, tile + 1, tileCapacity, nextOffset, nextCount);
    shared_load_tile(
        nextUb, gmX, logicalOffset + nextOffset, totalComplexCount, nextCount, incx, currentEvent, nextEvent, true);
}

__aicore__ __inline__ __attribute__((always_inline)) void run_shared_pipeline(
    __gm__ float* gmX, __gm__ uint32_t* gmAug, __gm__ float* gmY, float alphaReal, float alphaImag,
    uint32_t logicalOffset, uint32_t totalComplexCount, uint32_t complexCount, int64_t incx, int64_t incy,
    uint32_t tileCapacity)
{
    CaxpyStridedUb ping = shared_ub_layout(0);
    CaxpyStridedUb pong = shared_ub_layout(1);
    shared_preload_offsets(ping, gmAug, incy);

    set_atomic_add();
    set_atomic_f32();
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

    const uint32_t tileCount = (complexCount + tileCapacity - 1) / tileCapacity;
    uint32_t firstOffset = 0;
    uint32_t firstCount = 0;
    shared_tile_bounds(complexCount, tileCount, 0, tileCapacity, firstOffset, firstCount);
    shared_load_tile(
        ping, gmX, logicalOffset + firstOffset, totalComplexCount, firstCount, incx, EVENT_ID0, EVENT_ID0, false);

    for (uint32_t tile = 0; tile < tileCount; ++tile) {
        const uint32_t current = tile & 1;
        const uint32_t currentEvent = current == 0 ? EVENT_ID0 : EVENT_ID1;
        CaxpyStridedUb& currentUb = current == 0 ? ping : pong;
        uint32_t currentOffset = 0;
        uint32_t currentCount = 0;
        shared_tile_bounds(complexCount, tileCount, tile, tileCapacity, currentOffset, currentCount);

        shared_compute_tile(currentUb, alphaReal, alphaImag, currentCount, incx, currentEvent);

        shared_load_next_if_any(
            ping, pong, gmX, logicalOffset, totalComplexCount, complexCount, tileCapacity, incx, tile, tileCount,
            current);

        strided_store_tile(
            currentUb, gmY, logicalOffset + currentOffset, totalComplexCount, currentCount, incy, currentEvent);
    }

    const uint32_t lastRawEvent = ((tileCount - 1) & 1) == 0 ? EVENT_ID0 : EVENT_ID1;
    wait_flag(PIPE_V, PIPE_MTE2, lastRawEvent);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    set_atomic_none();
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_strided_shared_run(
    __gm__ float* gmX, __gm__ uint32_t* gmAug, __gm__ float* gmY, float alphaReal, float alphaImag,
    uint32_t logicalOffset, uint32_t totalComplexCount, uint32_t complexCount, int64_t incx, int64_t incy,
    uint32_t tileCapacity)
{
    run_shared_pipeline(
        gmX, gmAug, gmY, alphaReal, alphaImag, logicalOffset, totalComplexCount, complexCount, incx, incy,
        tileCapacity);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_strided_disjoint_run(
    __gm__ float* gmX, __gm__ uint32_t* gmAug, __gm__ float* gmY, float alphaReal, float alphaImag,
    uint32_t logicalOffset, uint32_t totalComplexCount, uint32_t complexCount, int64_t incx, int64_t incy,
    uint32_t tileCapacity)
{
    // The disjoint pipeline never span-reads, so it has no UB-buffer-fit
    // constraint to shrink its tile for; it always uses the fixed
    // CAXPY_STRIDED_TILE_COUNT capacity and ignores the runtime tileCapacity
    // value threaded through only so both kernel entries share one signature.
    (void)tileCapacity;
    run_disjoint_pipeline(
        gmX, gmAug, gmY, alphaReal, alphaImag, logicalOffset, totalComplexCount, complexCount, incx, incy);
}

#endif

extern "C" __global__ __aicore__ __vector__ void caxpy(
    __gm__ float* __restrict__ x, __gm__ uint32_t* __restrict__ aug, __gm__ float* __restrict__ y,
    __gm__ float* __restrict__ tiling_gm)
{
#if __DAV_C220_VEC__
    auto vec_idx = AscendC::GetBlockIdx();

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    auto tiling = reinterpret_cast<__gm__ CaxpyTilingData*>(tiling_gm);

    const uint32_t numBlocks = tiling->activeBlocks;
    const uint32_t totalN = tiling->totalN;
    if (numBlocks == 0 || vec_idx >= numBlocks) {
        return;
    }
    const uint32_t perBlock = totalN / numBlocks;
    const uint32_t remainder = totalN % numBlocks;
    const uint32_t complexCount = perBlock + (vec_idx < remainder ? 1 : 0);
    const uint32_t complexOffset = vec_idx * perBlock + (vec_idx < remainder ? vec_idx : remainder);
    caxpy_dense_run(x, aug, y, tiling->alphaReal, tiling->alphaImag, complexOffset * 2, complexCount * 2);
#endif
}

#if __DAV_C220_VEC__
#define DEFINE_CAXPY_STRIDED_KERNEL(KERNEL, PIPELINE)                                                                  \
    extern "C" __global__ __aicore__ __vector__ void KERNEL(                                                           \
        __gm__ float* __restrict__ x, __gm__ uint32_t* __restrict__ aug, __gm__ float* __restrict__ y,                 \
        __gm__ float* __restrict__ tilingGm)                                                                           \
    {                                                                                                                  \
        const uint32_t vecIdx = AscendC::GetBlockIdx();                                                                \
        set_mask_norm();                                                                                               \
        set_vector_mask((uint64_t) - 1, (uint64_t) - 1);                                                               \
        auto tiling = reinterpret_cast<__gm__ CaxpyTilingData*>(tilingGm);                                             \
        const uint32_t numBlocks = tiling->activeBlocks;                                                               \
        const float alphaReal = tiling->alphaReal;                                                                     \
        const float alphaImag = tiling->alphaImag;                                                                     \
        const uint32_t totalN = tiling->totalN;                                                                        \
        const int64_t incx = tiling->incx;                                                                             \
        const int64_t incy = tiling->incy;                                                                             \
        const uint32_t tileCapacity = tiling->sharedTileCapacity;                                                      \
        if (numBlocks == 0) {                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
        const uint32_t waveCapacity = numBlocks * CAXPY_STRIDED_MAX_PER_BLOCK_PER_WAVE;                                \
        for (uint32_t logicalBase = 0; logicalBase < totalN; logicalBase += waveCapacity) {                            \
            const uint32_t waveN = totalN - logicalBase > waveCapacity ? waveCapacity : totalN - logicalBase;          \
            const uint32_t waveBlocks = numBlocks < waveN ? numBlocks : waveN;                                         \
            if (vecIdx >= waveBlocks) {                                                                                \
                continue;                                                                                              \
            }                                                                                                          \
            const uint32_t rowNum = waveN / waveBlocks;                                                                \
            const uint32_t remain = waveN % waveBlocks;                                                                \
            const uint32_t myCount = rowNum + (vecIdx < remain ? 1 : 0);                                               \
            const uint32_t myStart = logicalBase + vecIdx * rowNum + (vecIdx < remain ? vecIdx : remain);              \
            if (incy > 0) {                                                                                            \
                PIPELINE(x, aug, y, alphaReal, alphaImag, myStart, totalN, myCount, incx, incy, tileCapacity);         \
            } else {                                                                                                   \
                const uint32_t mirroredStart = totalN - myStart - myCount;                                             \
                PIPELINE(x, aug, y, alphaReal, alphaImag, mirroredStart, totalN, myCount, -incx, -incy, tileCapacity); \
            }                                                                                                          \
        }                                                                                                              \
    }
#else
#define DEFINE_CAXPY_STRIDED_KERNEL(KERNEL, PIPELINE)                  \
    extern "C" __global__ __aicore__ __vector__ void KERNEL(           \
        __gm__ float*, __gm__ uint32_t*, __gm__ float*, __gm__ float*) \
    {}
#endif
DEFINE_CAXPY_STRIDED_KERNEL(caxpy_strided_shared, caxpy_strided_shared_run)
DEFINE_CAXPY_STRIDED_KERNEL(caxpy_strided_disjoint, caxpy_strided_disjoint_run)
#undef DEFINE_CAXPY_STRIDED_KERNEL

void caxpy_kernel_do(
    GM_ADDR x, GM_ADDR maskBuf, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    caxpy<<<numBlocks, nullptr, stream>>>((float*)x, (uint32_t*)maskBuf, (float*)y, (float*)tilingGm);
}

void caxpy_strided_shared_kernel_do(
    GM_ADDR x, GM_ADDR maskBuf, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    caxpy_strided_shared<<<numBlocks, nullptr, stream>>>((float*)x, (uint32_t*)maskBuf, (float*)y, (float*)tilingGm);
}

void caxpy_strided_disjoint_kernel_do(
    GM_ADDR x, GM_ADDR maskBuf, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    caxpy_strided_disjoint<<<numBlocks, nullptr, stream>>>((float*)x, (uint32_t*)maskBuf, (float*)y, (float*)tilingGm);
}

#if __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void scalar_copy_complex_in(__ubuf__ float* dst, __gm__ float* src)
{
    copy_gm_to_ubuf_align_b32(dst, src, 0, 1, 2 * sizeof(float), 0, 0, 0, 0);
}

__aicore__ __inline__ __attribute__((always_inline)) void scalar_copy_complex_out(
    __gm__ float* dst, __ubuf__ float* src)
{
    copy_ubuf_to_gm_align_b32(dst, src, 0, 1, 2 * sizeof(float), 0, 0, 0, 0);
}

__aicore__ __inline__ __attribute__((always_inline)) void scalar_compute_all(
    __gm__ float* gmX, __gm__ float* gmY, float alphaReal, float alphaImag, uint32_t logicalOffset,
    uint32_t totalComplexCount, uint32_t complexCount, int64_t incx, int64_t incy)
{
    const uint32_t absIncx = abs_increment(incx);
    const uint32_t absIncy = abs_increment(incy);
    auto ubX = reinterpret_cast<__ubuf__ float*>((uintptr_t)0);
    auto ubY = reinterpret_cast<__ubuf__ float*>((uintptr_t)32);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    for (uint32_t local = 0; local < complexCount; ++local) {
        const uint32_t logical = logicalOffset + local;
        const uint32_t xPhysical = incx > 0 ? logical * absIncx : (totalComplexCount - 1 - logical) * absIncx;
        const uint32_t yPhysical = incy > 0 ? logical * absIncy : (totalComplexCount - 1 - logical) * absIncy;

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        scalar_copy_complex_in(ubX, gmX + xPhysical * 2);
        scalar_copy_complex_in(ubY, gmY + yPhysical * 2);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        const float xReal = ubX[0];
        const float xImag = ubX[1];
        const float yReal = ubY[0];
        const float yImag = ubY[1];
        ubY[0] = yReal + alphaReal * xReal - alphaImag * xImag;
        ubY[1] = yImag + alphaReal * xImag + alphaImag * xReal;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        scalar_copy_complex_out(gmY + yPhysical * 2, ubY);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}
#endif

extern "C" __global__ __aicore__ __vector__ void caxpy_strided_scalar(
    __gm__ float* __restrict__ x, __gm__ float* __restrict__ y, __gm__ float* __restrict__ tilingGm)
{
#if __DAV_C220_VEC__
    const uint32_t vecIdx = AscendC::GetBlockIdx();
    auto tiling = reinterpret_cast<__gm__ CaxpyTilingData*>(tilingGm);
    const uint32_t numBlocks = tiling->activeBlocks;
    const float alphaReal = tiling->alphaReal;
    const float alphaImag = tiling->alphaImag;
    const uint32_t totalN = tiling->totalN;
    const int64_t incx = tiling->incx;
    const int64_t incy = tiling->incy;
    if (numBlocks == 0 || vecIdx >= numBlocks) {
        return;
    }

    const uint32_t rowNum = totalN / numBlocks;
    const uint32_t remain = totalN % numBlocks;
    const uint32_t count = rowNum + (vecIdx < remain ? 1 : 0);
    const uint32_t start = vecIdx * rowNum + (vecIdx < remain ? vecIdx : remain);
    scalar_compute_all(x, y, alphaReal, alphaImag, start, totalN, count, incx, incy);
#endif
}

void caxpy_strided_scalar_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm, uint32_t numBlocks, void* stream)
{
    caxpy_strided_scalar<<<numBlocks, nullptr, stream>>>((float*)x, (float*)y, (float*)tilingGm);
}
