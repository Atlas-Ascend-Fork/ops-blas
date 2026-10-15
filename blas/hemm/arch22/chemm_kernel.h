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

#include "kernel_operator.h"
#include "chemm_tiling_data.h"

namespace {

constexpr uint32_t kMicro = CHEMM_MICRO_TILE;
constexpr uint32_t kMicroElements = kMicro * kMicro;
constexpr uint32_t kComplexTileFloats = 2U * kMicroElements;
constexpr uint32_t kComplexColumnFloats = 2U * kMicro;
constexpr uint32_t kFp32DataBlockElements = 8U;
constexpr uint8_t kVectorRepeats = 4U;
constexpr uint32_t kAivRowsPerVectorCore = 8U;
constexpr uint32_t kAivRowsPerTransposeRepeat = 4U;
constexpr uint32_t kAivYCachePlaneElements = CHEMM_AIV_CACHE_FEEDS * kMicroElements;
constexpr uint32_t kAivAccumulatorParts = 2U;
constexpr uint32_t kAivAccumulatorBanks = 16U;
constexpr uint64_t kFullFp32Mask = 0xffffffffffffffffULL;
constexpr uint64_t kMaskAddressVectorMask = 0xffffffffULL;

static __gm__ uint64_t g_packMasks[12] __attribute__((aligned(32))) = {
    0xfff8fffcfffeffffULL, 0xff80ffc0ffe0fff0ULL, 0xf800fc00fe00ff00ULL, 0x8000c000e000f000ULL,
    0x000f000700030001ULL, 0x00ff007f003f001fULL, 0x0fff07ff03ff01ffULL, 0xffff7fff3fff1fffULL,
    0x0008000400020001ULL, 0x0080004000200010ULL, 0x0800040002000100ULL, 0x8000400020001000ULL};

// Only layout primitives shared by both device translation units belong here.
// Private scheduling, packing and arithmetic implementations belong in CPP.
__aicore__ inline uint64_t ElementIndex(uint32_t row, uint32_t col, uint32_t ld)
{
    return static_cast<uint64_t>(col) * ld + row;
}

template <uint32_t kSrcStride>
__aicore__ inline void StoreComplexTail(
    __gm__ float* dst, __ubuf__ float* src, uint32_t cols, uint32_t rows, uint32_t ld)
{
    const uint32_t floats = 2U * rows;
    const uint32_t aligned = (floats + 7U) & ~7U;
    copy_ubuf_to_gm_align_b32(
        dst, src, 0U, cols, floats * sizeof(float), 0U, 0U, (kSrcStride - aligned) / 8U,
        (2U * ld - floats) * sizeof(float));
}

template <uint32_t kCols = kMicro>
__aicore__ inline void CInterleavedToPlanes(
    __ubuf__ float* complexTile, __ubuf__ float* real, __ubuf__ float* imag, uint32_t dstStride = kMicro,
    uint32_t srcStride = kComplexColumnFloats, uint8_t repeats = kMicro / kAivRowsPerTransposeRepeat)
{
    uint64_t src[16];
    uint64_t dst[16];
#pragma unroll
    for (uint32_t col = 0U; col < kMicro; ++col) {
        src[col] = reinterpret_cast<uint64_t>(complexTile + (col < kCols ? col * srcStride : 0U));
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        __ubuf__ float* plane = (lane & 1U) == 0U ? real : imag;
        dst[2U * lane] = reinterpret_cast<uint64_t>(plane + lane / 2U * dstStride);
        dst[2U * lane + 1U] = dst[2U * lane] + 8U * sizeof(float);
    }
    set_va_reg_sb(VA0, dst);
    set_va_reg_sb(VA1, dst + 8U);
    set_va_reg_sb(VA2, src);
    set_va_reg_sb(VA3, src + 8U);
    scatter_vnchwconv_b32(VA0, VA2, repeats, dstStride / 2U, 1U);
}

template <uint32_t kRowGroups = 2U, bool kSingleRepeat = false>
__aicore__ inline void PlanesToCInterleaved(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* complexTile, uint32_t dstStride, uint32_t rowShift = 0U)
{
    uint64_t src[16];
    uint64_t dst[16];
#pragma unroll
    for (uint32_t group = 0U; group < kRowGroups; ++group) {
#pragma unroll
        for (uint32_t row = 0U; row < 8U; ++row) {
            const uint32_t planeRow = (group * 8U + row + rowShift) & 15U;
            src[2U * row] = reinterpret_cast<uint64_t>(real + planeRow * kMicro);
            src[2U * row + 1U] = reinterpret_cast<uint64_t>(imag + planeRow * kMicro);
            dst[2U * row] = reinterpret_cast<uint64_t>(complexTile + row * dstStride + group * 16U);
            dst[2U * row + 1U] = dst[2U * row] + 8U * sizeof(float);
        }
        set_va_reg_sb(VA0, dst);
        set_va_reg_sb(VA1, dst + 8U);
        set_va_reg_sb(VA2, src);
        set_va_reg_sb(VA3, src + 8U);
        scatter_vnchwconv_b32(
            VA0, VA2, kSingleRepeat ? 1U : 2U, kSingleRepeat ? 0U : dstStride, kSingleRepeat ? 0U : 1U);
    }
}

__aicore__ inline void TransposePlane16(
    __ubuf__ float* srcTile, __ubuf__ float* dstTile, uint32_t srcStride = kMicro, uint32_t dstStride = kMicro)
{
    uint64_t src[16];
    uint64_t dst[16];
#pragma unroll
    for (uint32_t row = 0U; row < kMicro; ++row) {
        src[row] = reinterpret_cast<uint64_t>(srcTile + row * srcStride);
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < 8U; ++lane) {
        dst[2U * lane] = reinterpret_cast<uint64_t>(dstTile + lane * dstStride);
        dst[2U * lane + 1U] = dst[2U * lane] + 8U * sizeof(float);
    }
    set_va_reg_sb(VA0, dst);
    set_va_reg_sb(VA1, dst + 8U);
    set_va_reg_sb(VA2, src);
    set_va_reg_sb(VA3, src + 8U);
    scatter_vnchwconv_b32(VA0, VA2, 2U, dstStride, 1U);
}

__aicore__ inline void ApplyHermitianMasks(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* transReal, __ubuf__ float* transImag,
    __ubuf__ uint8_t* triangleMask, __ubuf__ uint8_t* diagonalMask)
{
    __ubuf__ uint32_t addresses[32] __attribute__((aligned(32)));
    set_vector_mask(0ULL, kMaskAddressVectorMask);
    vector_dup(addresses, static_cast<uint32_t>(reinterpret_cast<uint64_t>(triangleMask)), 1U, 1U, 1U, 8U, 0U);
    pipe_barrier(PIPE_V);
    set_cmpmask(addresses);
    pipe_barrier(PIPE_V);
    set_vector_mask(0ULL, kFullFp32Mask);
    vsel(real, real, transReal, 4U, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    vsel(imag, imag, transImag, 4U, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    pipe_barrier(PIPE_V);
    vector_dup(transReal, 0.0f, 4U, 1U, 1U, 8U, 0U);
    set_vector_mask(0ULL, kMaskAddressVectorMask);
    vector_dup(addresses, static_cast<uint32_t>(reinterpret_cast<uint64_t>(diagonalMask)), 1U, 1U, 1U, 8U, 0U);
    pipe_barrier(PIPE_V);
    set_cmpmask(addresses);
    pipe_barrier(PIPE_V);
    set_vector_mask(0ULL, kFullFp32Mask);
    vsel(imag, transReal, imag, 4U, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
    set_mask_norm();
    set_vector_mask(0ULL, kFullFp32Mask);
}

template <bool kLower>
__aicore__ inline void DecodeHermitianPlanes16(
    uint32_t row, uint32_t col, __ubuf__ float* raw, __ubuf__ float* real, __ubuf__ float* imag,
    __ubuf__ float* transReal, __ubuf__ float* transImag, __ubuf__ uint8_t* triangleMask,
    __ubuf__ uint8_t* diagonalMask)
{
    const bool reflected = kLower ? row < col : row > col;
    if (reflected) {
        CInterleavedToPlanes(raw, transReal, transImag);
        pipe_barrier(PIPE_V);
        TransposePlane16(transReal, real);
        TransposePlane16(transImag, imag);
        pipe_barrier(PIPE_V);
        vmuls(imag, imag, -1.0f, 4U, 1U, 1U, 8U, 8U);
        return;
    }
    CInterleavedToPlanes(raw, real, imag);
    if (row == col) {
        pipe_barrier(PIPE_V);
        TransposePlane16(real, transReal);
        TransposePlane16(imag, transImag);
        pipe_barrier(PIPE_V);
        vmuls(transImag, transImag, -1.0f, 4U, 1U, 1U, 8U, 8U);
        ApplyHermitianMasks(real, imag, transReal, transImag, triangleMask, diagonalMask);
    }
}

template <bool kFull>
__aicore__ inline void LoadOutputTile16(
    __gm__ float* source, uint32_t row, uint32_t col, uint32_t ld, uint32_t rows, uint32_t cols, __ubuf__ float* raw)
{
    const uint32_t validRows = rows - row < kMicro ? rows - row : kMicro;
    const uint32_t validCols = cols - col < kMicro ? cols - col : kMicro;
    const uint32_t floats = 2U * validRows;
    const uint32_t aligned = (floats + 7U) & ~7U;
    if constexpr (!kFull) {
        vector_dup(raw, 0.0f, 8U, 1U, 1U, 8U, 0U);
    }
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    copy_gm_to_ubuf_align_b32(
        raw, source + 2U * ElementIndex(row, col, ld), 0U, validCols, floats * sizeof(float), 0U, aligned - floats,
        (2U * ld - floats) * sizeof(float), (kComplexColumnFloats - aligned) / 8U);
}

} // namespace
