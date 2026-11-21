/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"
#if ASC_DEVKIT_GE_9_1
#include "kernel_operator.h"
#include "cgemm_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"
using namespace AscendC;

// Selection and correction UB offsets are measured in float elements.
constexpr uint32_t CGEMM_SELECT_THREAD_COUNT = 256;
constexpr uint32_t CGEMM_SELECT_CHUNK_ELEMENTS = 2048;
constexpr uint32_t CGEMM_SELECT_ENERGY_OFFSET = 4096;
constexpr uint32_t CGEMM_SELECT_FLAGS_OFFSET = 12288;
constexpr uint32_t CGEMM_SELECT_PREFIX_OFFSET = 14336;
constexpr uint32_t CGEMM_SELECT_PATH_OFFSET = 14848;
constexpr uint32_t CGEMM_BRIDGE_SEGMENT_LENGTH = 128;
constexpr uint32_t CGEMM_TRANSPOSE_OUTPUT_OFFSET = 8192;
constexpr uint32_t CGEMM_PREFIX_B_OFFSET = 8448;
constexpr uint32_t CGEMM_PREFIX_SLOT_ELEMENTS = 16896;
constexpr uint32_t CGEMM_PREFIX_TERMS_OFFSET = 33792;
constexpr uint32_t CGEMM_PREFIX_RESULT_OFFSET = 42240;
constexpr uint32_t CGEMM_PERMUTE_ENERGY_OFFSET = 8192;
constexpr uint32_t CGEMM_PERMUTE_OUTPUT_OFFSET = 8448;
constexpr uint32_t CGEMM_PERMUTE_OUTPUT_HIGH_OFFSET = 8512;

// Screening heuristic: FP32 unit roundoff is 2^-24, so its squared scale is 2^-48.
// These parameters select work for recomputation; they are not output error tolerances.
constexpr float CGEMM_ROUNDING_VARIANCE_SCALE = 0x1p-48f;
constexpr float CGEMM_ROUNDING_RISK_THRESHOLD_SQUARED = 0.0001f;
constexpr float CGEMM_ROUNDING_BRIDGE_NORMALIZER = 18.0f;
// Sparse FP32 correction for contiguous square, unit-output GEMM.
// Selection uses only input energies and computed partial sums; no reference data.
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMM_SELECT_THREAD_COUNT) inline void SelectVf(
    __ubuf__ const float* c, __ubuf__ const float* energy, __gm__ uint32_t* list, __gm__ uint32_t* counts, uint32_t n,
    float confidence, uint32_t block, uint32_t blocks, uint32_t start, uint32_t size, uint32_t rowBase, uint32_t col,
    bool first, bool last, __ubuf__ uint32_t* flags, __ubuf__ uint32_t* prefix, __ubuf__ const float* path,
    float bridge)
{
    const uint32_t total = n * n;
    const uint32_t span = (total + blocks - 1) / blocks;
    const uint32_t begin = block * span;
    if (first && threadIdx.x == 0)
        prefix[CGEMM_SELECT_THREAD_COUNT] = 0;
    __syncthreads();
    uint32_t found = 0;
    for (uint32_t j = threadIdx.x; j < size; j += blockDim.x) {
        const float re = fabsf(c[2 * j]), im = fabsf(c[2 * j + 1]);
        const float magnitude = re > im ? re : im;
        const float ea = energy[rowBase + j], eb = energy[n + col];
        const float variance = CGEMM_ROUNDING_VARIANCE_SCALE * (path[j] + ea * eb * bridge);
        const bool selected = !isfinite(magnitude) || !isfinite(variance) ||
                              confidence * confidence * variance >= CGEMM_ROUNDING_RISK_THRESHOLD_SQUARED;
        flags[j] = selected;
        found += selected;
    }
    prefix[threadIdx.x] = found;
    __syncthreads();
    for (uint32_t distance = 1; distance < blockDim.x; distance *= 2) {
        const uint32_t source = threadIdx.x >= distance ? threadIdx.x - distance : 0;
        const uint32_t value = prefix[source];
        const uint32_t extra = threadIdx.x >= distance ? value : 0;
        __syncthreads();
        prefix[threadIdx.x] += extra;
        __syncthreads();
    }
    uint32_t offset = prefix[CGEMM_SELECT_THREAD_COUNT] + prefix[threadIdx.x] - found;
    for (uint32_t j = threadIdx.x; j < size; j += blockDim.x) {
        if (flags[j])
            list[begin + offset++] = start + j;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        prefix[CGEMM_SELECT_THREAD_COUNT] += prefix[blockDim.x - 1];
        if (last)
            counts[block] = prefix[CGEMM_SELECT_THREAD_COUNT];
    }
}

extern "C" __global__ __vector__ void cgemm3m_select_kernel(
    GM_ADDR c, GM_ADDR energy, GM_ADDR list, GM_ADDR counts, uint32_t n, float confidence, GM_ADDR path)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    LocalMemAllocator<Hardware::UB> allocator;
    auto data = allocator.Alloc<float>(CGEMM_SELECT_UB_ELEMENTS);
    auto storage = reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr());
    const uint32_t tail = n % CGEMM_BRIDGE_SEGMENT_LENGTH;
    const float bridge = (static_cast<float>(n / CGEMM_BRIDGE_SEGMENT_LENGTH) *
                              (CGEMM_BRIDGE_SEGMENT_LENGTH * CGEMM_BRIDGE_SEGMENT_LENGTH) +
                          tail * tail) /
                         (CGEMM_ROUNDING_BRIDGE_NORMALIZER * static_cast<float>(n) * n);
    // Energy contains n row energies followed by n column energies; copy length is in bytes.
    asc_copy_gm2ub_align(
        storage + CGEMM_SELECT_ENERGY_OFFSET, reinterpret_cast<__gm__ float*>(energy), 1, 2 * n * sizeof(float), 0, 0,
        false, 0, 0, 0);
    const uint32_t span = (n * n + GetBlockNum() - 1) / GetBlockNum();
    const uint32_t begin = GetBlockIdx() * span;
    const uint32_t end = begin + span < n * n ? begin + span : n * n;
    for (uint32_t start = begin; start < end;) {
        const uint32_t row = start % n, col = start / n;
        uint32_t size = end - start < CGEMM_SELECT_CHUNK_ELEMENTS ? end - start : CGEMM_SELECT_CHUNK_ELEMENTS;
        size = size < n - row ? size : n - row;
        asc_copy_gm2ub_align(
            storage, reinterpret_cast<__gm__ float*>(c) + start * 2, 1, size * 8, 0, 0, false, 0, 0, 0);
        asc_copy_gm2ub_align(
            storage + CGEMM_SELECT_PATH_OFFSET, reinterpret_cast<__gm__ float*>(path) + start, 1, size * sizeof(float),
            0, 0, false, 0, 0, 0);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        asc_vf_call<SelectVf>(
            dim3{CGEMM_SELECT_THREAD_COUNT, 1, 1}, storage, storage + CGEMM_SELECT_ENERGY_OFFSET,
            reinterpret_cast<__gm__ uint32_t*>(list), reinterpret_cast<__gm__ uint32_t*>(counts), n, confidence,
            GetBlockIdx(), GetBlockNum(), start, size, row, col, start == begin, start + size == end,
            reinterpret_cast<__ubuf__ uint32_t*>(storage + CGEMM_SELECT_FLAGS_OFFSET),
            reinterpret_cast<__ubuf__ uint32_t*>(storage + CGEMM_SELECT_PREFIX_OFFSET),
            storage + CGEMM_SELECT_PATH_OFFSET, bridge);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        start += size;
    }
}

__simd_vf__ inline void TransposeTile(__ubuf__ float* data, uint32_t rows)
{
    auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> lane, index;
    AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    AscendC::Reg::Muls(index, lane, 128u, all);
    for (uint32_t row = 0; row < rows; ++row) {
        AscendC::Reg::RegTensor<float> re, im, lo, hi;
        AscendC::Reg::Gather(re, data + row * 2, index, all);
        AscendC::Reg::Gather(im, data + row * 2 + 1, index, all);
        AscendC::Reg::Interleave(lo, hi, re, im);
        AscendC::Reg::StoreAlign(data + CGEMM_TRANSPOSE_OUTPUT_OFFSET + row * 128, lo, all);
        AscendC::Reg::StoreAlign(data + CGEMM_TRANSPOSE_OUTPUT_OFFSET + row * 128 + 64, hi, all);
    }
}

extern "C" __global__ __aicore__ void cgemm3m_pack_kernel(GM_ADDR a, GM_ADDR packed, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> ga, gp;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    gp.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packed));
    LocalTensor<float> data(TPosition::VECCALC, 0, CGEMM_PACK_UB_ELEMENTS);
    const uint32_t tiles = (n + 63) / 64;
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    for (uint32_t tile = GetBlockIdx(); tile < tiles * tiles; tile += GetBlockNum()) {
        const uint32_t row = tile % tiles * 64, col = tile / tiles * 64;
        const uint32_t rows = n - row < 64 ? n - row : 64;
        const uint32_t cols = n - col < 64 ? n - col : 64;
        // DataCopyPad: GM strides are bytes; UB strides are 32-byte blocks.
        // UB rounds each blockLen up to 32 bytes, giving a fixed 512-byte tile pitch.
        const uint32_t rowBytes = rows * 8;
        const uint32_t colBytes = cols * 8;
        const uint32_t gmPitchBytes = n * 8; // No rounding: n need not be a multiple of four.
        const uint32_t inputGapBlocks = (512 - (rowBytes + 31) / 32 * 32) / 32;
        const uint32_t outputGapBlocks = (512 - (colBytes + 31) / 32 * 32) / 32;
        DataCopyExtParams p{static_cast<uint16_t>(cols), rowBytes, gmPitchBytes - rowBytes, inputGapBlocks, 0};
        DataCopyPad(data, ga[2 * (static_cast<uint64_t>(col) * n + row)], p, pad);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        TransposeTile(reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr()), rows);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (rows == 64 && cols == 64) {
            DataCopyExtParams q{static_cast<uint16_t>(rows), colBytes, outputGapBlocks, gmPitchBytes - colBytes, 0};
            DataCopyPad(gp[2 * (static_cast<uint64_t>(row) * n + col)], data[CGEMM_TRANSPOSE_OUTPUT_OFFSET], q);
        } else {
            // TransposeTile produced rows output vectors, each with cols valid complex elements.
            // Address each tail row explicitly; never write the padded lanes beyond cols.
            for (uint32_t outputRow = 0; outputRow < rows; ++outputRow) {
                DataCopyPad(
                    gp[2 * (static_cast<uint64_t>(row + outputRow) * n + col)],
                    data[CGEMM_TRANSPOSE_OUTPUT_OFFSET + outputRow * 128], {1, colBytes, 0, 0, 0});
            }
        }
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

__simd_callee__ inline void ThreeProductTerms(
    __ubuf__ float* data, uint32_t k, AscendC::Reg::RegTensor<float>& real, AscendC::Reg::RegTensor<float>& imag,
    AscendC::Reg::MaskReg& all)
{
    AscendC::Reg::RegTensor<float> ar, ai, br, bi, p, q, r, x;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, data + k * 128);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
        br, bi, data + CGEMM_PREFIX_B_OFFSET + k * 128);
    AscendC::Reg::Add(x, ar, ai, all);
    AscendC::Reg::Mul(p, x, br, all);
    AscendC::Reg::Sub(x, bi, br, all);
    AscendC::Reg::Mul(q, ar, x, all);
    AscendC::Reg::Add(x, br, bi, all);
    AscendC::Reg::Mul(r, ai, x, all);
    AscendC::Reg::Sub(real, p, r, all);
    AscendC::Reg::Add(imag, p, q, all);
}

__simd_callee__ inline void SegmentExclusivePrefix(
    AscendC::Reg::RegTensor<float>& sumR, AscendC::Reg::RegTensor<float>& sumI,
    AscendC::Reg::RegTensor<float>& initialR, AscendC::Reg::RegTensor<float>& initialI,
    AscendC::Reg::RegTensor<uint32_t>& lane, AscendC::Reg::RegTensor<float>& zero, AscendC::Reg::MaskReg& all)
{
    for (uint32_t distance = 1; distance < 64; distance *= 2) {
        AscendC::Reg::RegTensor<uint32_t> shifted;
        AscendC::Reg::RegTensor<uint32_t> d;
        AscendC::Reg::RegTensor<float> r;
        AscendC::Reg::RegTensor<float> i;
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::Duplicate(d, distance, all);
        AscendC::Reg::Sub(shifted, lane, d, all);
        AscendC::Reg::Compare<uint32_t, CMPMODE::GE>(valid, lane, d, all);
        AscendC::Reg::Gather(r, sumR, shifted);
        AscendC::Reg::Gather(i, sumI, shifted);
        AscendC::Reg::Select(r, r, zero, valid);
        AscendC::Reg::Select(i, i, zero, valid);
        AscendC::Reg::Add(sumR, sumR, r, all);
        AscendC::Reg::Add(sumI, sumI, i, all);
    }
    AscendC::Reg::RegTensor<uint32_t> shifted;
    AscendC::Reg::RegTensor<uint32_t> one;
    AscendC::Reg::MaskReg nonzero;
    AscendC::Reg::Duplicate(one, 1u, all);
    AscendC::Reg::Sub(shifted, lane, one, all);
    AscendC::Reg::Compare<uint32_t, CMPMODE::GE>(nonzero, lane, one, all);
    AscendC::Reg::Gather(initialR, sumR, shifted);
    AscendC::Reg::Gather(initialI, sumI, shifted);
    AscendC::Reg::Select(initialR, initialR, zero, nonzero);
    AscendC::Reg::Select(initialI, initialI, zero, nonzero);
}

__simd_callee__ inline void SelectStableSegmentSum(
    AscendC::Reg::RegTensor<float>& bridged, AscendC::Reg::RegTensor<float>& direct,
    AscendC::Reg::RegTensor<float>& prefixes, AscendC::Reg::MaskReg& all)
{
    AscendC::Reg::RegTensor<float> magnitude;
    AscendC::Reg::RegTensor<float> maximum;
    AscendC::Reg::RegTensor<float> limit;
    AscendC::Reg::RegTensor<float> scale;
    AscendC::Reg::MaskReg cancellation;
    AscendC::Reg::Abs(magnitude, prefixes, all);
    AscendC::Reg::ReduceMax(maximum, magnitude, all);
    AscendC::Reg::Duplicate(maximum, maximum, all);
    AscendC::Reg::Abs(magnitude, direct, all);
    // Prefixes at least 32 times the result can lose five significant bits
    // during replay/subtraction. Avoid that bridge for strongly cancelling sums.
    AscendC::Reg::Duplicate(scale, 32.0f, all);
    AscendC::Reg::Mul(limit, magnitude, scale, all);
    AscendC::Reg::Compare<float, CMPMODE::GE>(cancellation, maximum, limit, all);
    AscendC::Reg::Select(bridged, direct, bridged, cancellation);
}

__simd_vf__ inline void FastSegmentDot(__ubuf__ float* data, __ubuf__ float* terms, __ubuf__ float* output, uint32_t n)
{
    auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    const uint32_t span = ((n + 63) / 64) | 1u;
    AscendC::Reg::RegTensor<uint32_t> lane;
    AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    AscendC::Reg::RegTensor<float> sumR, sumI, zero;
    AscendC::Reg::Duplicate(sumR, 0.0f, all);
    AscendC::Reg::Duplicate(sumI, 0.0f, all);
    AscendC::Reg::Duplicate(zero, 0.0f, all);
    for (uint32_t k = 0; k < span; ++k) {
        AscendC::Reg::RegTensor<float> x, y;
        ThreeProductTerms(data, k, x, y, all);
        AscendC::Reg::Add(sumR, sumR, x, all);
        AscendC::Reg::Add(sumI, sumI, y, all);
        AscendC::Reg::StoreAlign(terms + k * 128, x, all);
        AscendC::Reg::StoreAlign(terms + 64 + k * 128, y, all);
    }
    AscendC::Reg::RegTensor<float> directR;
    AscendC::Reg::RegTensor<float> directI;
    AscendC::Reg::Reduce(directR, sumR, all);
    AscendC::Reg::Reduce(directI, sumI, all);
    AscendC::Reg::RegTensor<float> initialR, initialI;
    // Seed each segment with preceding segments before replaying its terms.
    // FP32 rounding occurs during replay; subtracting the seed afterwards is
    // numerically different from accumulating each segment from zero.
    SegmentExclusivePrefix(sumR, sumI, initialR, initialI, lane, zero, all);
    AscendC::Reg::RegTensor<float> finalR, finalI;
    AscendC::Reg::Add(finalR, initialR, zero, all);
    AscendC::Reg::Add(finalI, initialI, zero, all);
    for (uint32_t k = 0; k < span; ++k) {
        AscendC::Reg::RegTensor<float> r, i;
        AscendC::Reg::LoadAlign(r, terms + k * 128);
        AscendC::Reg::LoadAlign(i, terms + 64 + k * 128);
        AscendC::Reg::Add(finalR, finalR, r, all);
        AscendC::Reg::Add(finalI, finalI, i, all);
    }
    AscendC::Reg::Sub(finalR, finalR, initialR, all);
    AscendC::Reg::Sub(finalI, finalI, initialI, all);
    AscendC::Reg::Reduce(finalR, finalR, all);
    AscendC::Reg::Reduce(finalI, finalI, all);
    SelectStableSegmentSum(finalR, directR, sumR, all);
    SelectStableSegmentSum(finalI, directI, sumI, all);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(output, finalR, all);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(output + 1, finalI, all);
}

__aicore__ inline void LoadPrefixInput(
    LocalTensor<float> data, GlobalTensor<float> ga, GlobalTensor<float> gb, uint32_t index, uint32_t n, uint32_t width,
    uint32_t& cachedCol)
{
    const uint32_t row = index % n;
    const uint32_t col = index / n;
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    DataCopyPad(data, ga[2 * static_cast<uint64_t>(row) * width], {1, width * 8, 0, 0, 0}, pad);
    if (col != cachedCol) {
        DataCopyPad(
            data[CGEMM_PREFIX_B_OFFSET], gb[2 * static_cast<uint64_t>(col) * width], {1, width * 8, 0, 0, 0}, pad);
        cachedCol = col;
    }
}

__aicore__ inline void LoadPrefixIndices(LocalTensor<uint32_t> indices, GM_ADDR list, uint32_t start, uint32_t count)
{
    auto dst = reinterpret_cast<__ubuf__ uint32_t*>(indices.GetPhyAddr());
    auto src = reinterpret_cast<__gm__ uint32_t*>(list) + start;
    asc_copy_gm2ub_align_sync(dst, src, count * sizeof(uint32_t));
}

__aicore__ inline uint32_t PrepareNextPrefixItem(
    LocalTensor<float> data, LocalTensor<uint32_t> indices, GlobalTensor<float> ga, GlobalTensor<float> gb,
    GM_ADDR list, uint32_t listBase, uint32_t count, uint32_t item, uint32_t n, uint32_t width, uint32_t& cachedCol)
{
    const uint32_t nextSlot = 1 - item % 2;
    if (item + 1 >= count) {
        if (item > 0)
            WaitFlag<HardEvent::V_MTE2>(nextSlot);
        return 0;
    }
    if ((item + 1) % CGEMM_PREFIX_INDEX_BATCH == 0) {
        const uint32_t remaining = count - item - 1;
        LoadPrefixIndices(
            indices, list, listBase + item + 1,
            remaining < CGEMM_PREFIX_INDEX_BATCH ? remaining : CGEMM_PREFIX_INDEX_BATCH);
    }
    const uint32_t nextIndex = indices.GetValue((item + 1) % CGEMM_PREFIX_INDEX_BATCH);
    if (item > 0)
        WaitFlag<HardEvent::V_MTE2>(nextSlot);
    LoadPrefixInput(data[nextSlot * CGEMM_PREFIX_SLOT_ELEMENTS], ga, gb, nextIndex, n, width, cachedCol);
    SetFlag<HardEvent::MTE2_V>(nextSlot);
    return nextIndex;
}

extern "C" __global__ __aicore__ void cgemm3m_prefix_recompute_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR list, GM_ADDR counts, uint32_t n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> ga, gb, gc;
    GlobalTensor<uint32_t> gl, gn;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    gc.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    gl.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(list));
    gn.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(counts));
    LocalTensor<float> data(TPosition::VECCALC, 0, CGEMM_PREFIX_UB_ELEMENTS);
    const uint32_t span = (n * n + GetBlockNum() - 1) / GetBlockNum();
    const uint32_t count = gn.GetValue(GetBlockIdx());
    if (count == 0)
        return;
    const uint32_t width = (((n + 63) / 64) | 1u) * 64;
    uint32_t cachedCol[2] = {n, n};
    const uint32_t listBase = GetBlockIdx() * span;
    auto indices = data[CGEMM_PREFIX_INDEX_OFFSET].ReinterpretCast<uint32_t>();
    LoadPrefixIndices(indices, list, listBase, count < CGEMM_PREFIX_INDEX_BATCH ? count : CGEMM_PREFIX_INDEX_BATCH);
    uint32_t index = indices.GetValue(0);
    LoadPrefixInput(data, ga, gb, index, n, width, cachedCol[0]);
    SetFlag<HardEvent::MTE2_V>(0);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t slot = item % 2, nextSlot = 1 - slot;
        const uint32_t nextIndex =
            PrepareNextPrefixItem(data, indices, ga, gb, list, listBase, count, item, n, width, cachedCol[nextSlot]);
        WaitFlag<HardEvent::MTE2_V>(slot);
        if (item >= 2)
            WaitFlag<HardEvent::MTE3_V>(slot);
        auto ptr = reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr());
        FastSegmentDot(
            ptr + slot * CGEMM_PREFIX_SLOT_ELEMENTS, ptr + CGEMM_PREFIX_TERMS_OFFSET,
            ptr + CGEMM_PREFIX_RESULT_OFFSET + slot * 8, n);
        SetFlag<HardEvent::V_MTE2>(slot);
        SetFlag<HardEvent::V_MTE3>(slot);
        WaitFlag<HardEvent::V_MTE3>(slot);
        DataCopyPad(gc[index * 2], data[CGEMM_PREFIX_RESULT_OFFSET + slot * 8], {1, 8, 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(slot);
        index = nextIndex;
    }
    WaitFlag<HardEvent::V_MTE2>((count - 1) % 2);
    WaitFlag<HardEvent::MTE3_V>(0);
    if (count > 1) {
        WaitFlag<HardEvent::MTE3_V>(1);
    }
}

__simd_vf__ inline void PermuteSegments(__ubuf__ float* data, uint32_t n, uint32_t span, float sign)
{
    auto all = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::RegTensor<uint32_t> lane, index, limit;
    AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(lane), int32_t(0));
    AscendC::Reg::Muls(index, lane, span * 2, all);
    AscendC::Reg::Duplicate(limit, n * 2, all);
    AscendC::Reg::RegTensor<float> zero, sumSquares;
    AscendC::Reg::Duplicate(zero, 0.0f, all);
    AscendC::Reg::Duplicate(sumSquares, 0.0f, all);
    for (uint32_t k = 0; k < span; ++k) {
        AscendC::Reg::RegTensor<float> re, im, lo, hi;
        AscendC::Reg::MaskReg valid;
        AscendC::Reg::Compare<uint32_t, CMPMODE::LT>(valid, index, limit, all);
        AscendC::Reg::Gather(re, data, index, valid);
        AscendC::Reg::Gather(im, data + 1, index, valid);
        AscendC::Reg::Select(re, re, zero, valid);
        AscendC::Reg::Select(im, im, zero, valid);
        AscendC::Reg::Muls(im, im, sign, all);
        AscendC::Reg::MulAddDst(sumSquares, re, re, all);
        AscendC::Reg::MulAddDst(sumSquares, im, im, all);
        AscendC::Reg::Interleave(lo, hi, re, im);
        AscendC::Reg::StoreAlign(data + CGEMM_PERMUTE_OUTPUT_OFFSET + k * 128, lo, all);
        AscendC::Reg::StoreAlign(data + CGEMM_PERMUTE_OUTPUT_HIGH_OFFSET + k * 128, hi, all);
        AscendC::Reg::Adds(index, index, 2u, all);
    }
    AscendC::Reg::Reduce(sumSquares, sumSquares, all);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        data + CGEMM_PERMUTE_ENERGY_OFFSET, sumSquares, all);
}

extern "C" __global__ __aicore__ void cgemm3m_permute_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR pa, GM_ADDR pb, GM_ADDR energy, uint32_t n, uint32_t width, float signA, float signB)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<float> ga, gb, gpa, gpb, ge;
    ga.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    gb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    gpa.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pa));
    gpb.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pb));
    ge.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(energy));
    LocalTensor<float> data(TPosition::VECCALC, 0, CGEMM_PERMUTE_UB_ELEMENTS);
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    for (uint32_t index = GetBlockIdx(); index < n * 2; index += GetBlockNum()) {
        const uint32_t row = index < n ? index : index - n;
        if (index < n)
            DataCopyPad(data, ga[2 * static_cast<uint64_t>(row) * n], {1, n * 8, 0, 0, 0}, pad);
        else
            DataCopyPad(data, gb[2 * static_cast<uint64_t>(row) * n], {1, n * 8, 0, 0, 0}, pad);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        PermuteSegments(reinterpret_cast<__ubuf__ float*>(data.GetPhyAddr()), n, width / 64, index < n ? signA : signB);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (index < n)
            DataCopyPad(
                gpa[2 * static_cast<uint64_t>(row) * width], data[CGEMM_PERMUTE_OUTPUT_OFFSET],
                {1, width * 8, 0, 0, 0});
        else
            DataCopyPad(
                gpb[2 * static_cast<uint64_t>(row) * width], data[CGEMM_PERMUTE_OUTPUT_OFFSET],
                {1, width * 8, 0, 0, 0});
        DataCopyPad(ge[index], data[CGEMM_PERMUTE_ENERGY_OFFSET], {1, 4, 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

void gemm_cgemm3m_rounding_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR energy, GM_ADDR list, GM_ADDR counts,
    uint32_t n, float confidence, GM_ADDR packed, GM_ADDR path, char ta, char tb)
{
    GM_ADDR aReady = a;
    GM_ADDR bReady = b;
    if (ta == 'N') {
        cgemm3m_pack_kernel<<<blocks, nullptr, stream>>>(a, packed, n);
        aReady = packed;
    }
    if (tb != 'N') {
        bReady = packed + CgemmRoundingMatrixBytes(n);
        cgemm3m_pack_kernel<<<blocks, nullptr, stream>>>(b, bReady, n);
    }
    const uint32_t width = (((n + 63) / 64) | 1u) * 64;
    GM_ADDR permutedA = packed + 2 * CgemmRoundingMatrixBytes(n);
    GM_ADDR permutedB = permutedA + static_cast<uint64_t>(n) * width * 8;
    cgemm3m_permute_kernel<<<blocks, nullptr, stream>>>(
        aReady, bReady, permutedA, permutedB, energy, n, width, ta == 'C' ? -1.0f : 1.0f, tb == 'C' ? -1.0f : 1.0f);
    // __vector__ hybrid launch uses <<<block_num, dyn_ub_size, stream>>>.
    const uint32_t dynUbSize = CGEMM_SELECT_UB_BYTES;
    cgemm3m_select_kernel<<<blocks, dynUbSize, stream>>>(c, energy, list, counts, n, confidence, path);
    cgemm3m_prefix_recompute_kernel<<<blocks, nullptr, stream>>>(permutedA, permutedB, c, list, counts, n);
}

#endif
