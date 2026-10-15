/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "chemm_kernel.h"

namespace {

constexpr uint32_t kInputBankElements = CHEMM_MAX_P_ELEMENTS / 2U + CHEMM_L0_ELEMENTS / 2U;
constexpr uint32_t kBulkElements = 2U * CHEMM_K_FEED * CHEMM_MAX_ROWS / 2U;
constexpr uint32_t kPlaneElements = CHEMM_MAX_ROWS / 2U * kMicro;
constexpr uint32_t kScratchElements =
    2U * kInputBankElements + kBulkElements + 2U * kPlaneElements + 4U * kMicroElements + 24U;
static_assert(kScratchElements * sizeof(float) <= 196352U, "MIX vector scratch exceeds usable UB");

struct PackBuffers {
    __ubuf__ float* input;
    __ubuf__ float* bulk;
    __ubuf__ float* raw;
    __ubuf__ float* real;
    __ubuf__ float* imag;
    __ubuf__ float* transReal;
    __ubuf__ float* transImag;
    __ubuf__ uint8_t* masks;
};

struct TileWork {
    uint32_t row;
    uint32_t col;
    uint32_t pFirst;
    uint32_t pRows;
    uint32_t qRows;
};

struct FeedWork {
    uint32_t row;
    uint32_t col;
    uint32_t rows;
    uint32_t job;
    uint32_t totalJobs;
    uint32_t feed;
    bool newColumn;
    bool releaseP;
};

// Immutable indices describe the eight-element-aligned UB pitches, not input shapes.
struct HermitianGather {
    uint32_t index[CHEMM_K_FEED / 8U][2][2U * CHEMM_K_FEED * CHEMM_K_FEED];
    uint64_t conjugate[CHEMM_K_FEED / 8U][2][2U * CHEMM_K_FEED * CHEMM_K_FEED / 64U];
};

constexpr HermitianGather MakeHermitianGather()
{
    HermitianGather indices{};
    for (uint32_t pitch = 8U; pitch <= CHEMM_K_FEED; pitch += 8U) {
        for (uint32_t lower = 0U; lower < 2U; ++lower) {
            for (uint32_t element = 0U; element < 2U * pitch * CHEMM_K_FEED; ++element) {
                const uint32_t col = element / (2U * pitch);
                const uint32_t row = element % pitch;
                const bool imag = element % (2U * pitch) >= pitch;
                const bool reflected = lower != 0U ? row < col : row > col;
                const uint32_t source = reflected ? row * pitch + col : col * pitch + row;
                indices.index[pitch / 8U - 1U][lower][element] =
                    (imag && row == col ? 2U * CHEMM_K_FEED * CHEMM_K_FEED : 2U * source + (imag ? 1U : 0U)) *
                    sizeof(float);
                indices.conjugate[pitch / 8U - 1U][lower][element / 64U] |= uint64_t(imag && reflected)
                                                                            << (element % 64U);
            }
        }
    }
    return indices;
}

static __gm__ HermitianGather g_hermitianGather __attribute__((aligned(32))) = MakeHermitianGather();

template <bool kWait = true, event_t kEvent = EVENT_ID3>
__aicore__ inline void LoadDenseRaw(
    __gm__ float* source, uint32_t ld, uint32_t rows, uint32_t cols, uint32_t stride, uint32_t elements, bool clear,
    __ubuf__ float* raw)
{
    if (clear) {
        vector_dup(raw, 0.0f, static_cast<uint8_t>(elements / 64U), 1U, 1U, 8U, 0U);
    }
    set_flag(PIPE_V, PIPE_MTE2, kEvent);
    wait_flag(PIPE_V, PIPE_MTE2, kEvent);
    if (rows != 0U && cols != 0U) {
        const uint32_t floats = 2U * rows;
        const uint32_t aligned = (floats + 7U) & ~7U;
        if (ld == rows && stride == floats && aligned == floats) {
            copy_gm_to_ubuf(raw, source, 0U, 1U, rows * cols / 4U, 0U, 0U);
        } else {
            copy_gm_to_ubuf_align_b32(
                raw, source, 0U, cols, floats * sizeof(float), 0U, aligned - floats, (2U * ld - floats) * sizeof(float),
                (stride - aligned) / 8U);
        }
    }
    set_flag(PIPE_MTE2, PIPE_V, kEvent);
    if constexpr (kWait) {
        wait_flag(PIPE_MTE2, PIPE_V, kEvent);
    }
}

__aicore__ inline void CopyPlaneRows(
    __ubuf__ float* dst, __ubuf__ float* src, uint32_t rows, uint32_t cols, uint32_t srcStride, uint32_t dstStride)
{
    copy_ubuf_to_ubuf(dst, src, 0U, rows, cols / 8U, (srcStride - cols) / 8U, (dstStride - cols) / 8U);
}

template <bool kLower, bool kConjugate = false>
__aicore__ inline void ReadHermitian(
    __gm__ float* a, const ChemmMixedTilingData& t, uint32_t row, uint32_t col, PackBuffers& ub)
{
    const bool reflected = kLower ? row < col : row > col;
    const uint32_t srcRow = reflected ? col : row;
    const uint32_t srcCol = reflected ? row : col;
    const uint32_t rows = static_cast<uint32_t>(t.dimA);
    const uint32_t cols = rows;
    const uint32_t ld = static_cast<uint32_t>(t.lda);
    const uint32_t validRows = srcRow < rows ? (rows - srcRow < kMicro ? rows - srcRow : kMicro) : 0U;
    const uint32_t validCols = srcCol < cols ? (cols - srcCol < kMicro ? cols - srcCol : kMicro) : 0U;
    LoadDenseRaw(
        a + 2U * ElementIndex(srcRow, srcCol, ld), ld, validRows, validCols, 2U * kMicro, kComplexTileFloats,
        validRows != kMicro || validCols != kMicro, ub.raw);
    DecodeHermitianPlanes16<kLower>(
        row, col, ub.raw, ub.real, ub.imag, ub.transReal, ub.transImag, ub.masks + (kLower ? 32U : 0U), ub.masks + 64U);
    pipe_barrier(PIPE_V);
    if constexpr (kConjugate) {
        vmuls(ub.imag, ub.imag, -1.0f, 4U, 1U, 1U, 8U, 8U);
        pipe_barrier(PIPE_V);
    }
}

template <bool kLower, bool kPrepare>
__aicore__ inline void PackResidentP(
    __gm__ float* a, const ChemmMixedTilingData& t, const TileWork& work, __ubuf__ float* p, __ubuf__ float* q,
    PackBuffers& ub)
{
    if (work.pRows == 0U) {
        return;
    }
    constexpr uint32_t indexElements = 2U * kMicro * CHEMM_K_FEED;
    constexpr uint32_t sourceElements = 2U * CHEMM_K_FEED * CHEMM_K_FEED;
    const uint32_t first = (work.col + work.pFirst) * t.expandedK;
    const uint32_t k = t.expandedK / 2U;
    const uint32_t pitch = k / 8U - 1U;
    const uint32_t elements = work.pRows * t.expandedK;
    const uint32_t repeats = elements / 64U;
    __ubuf__ uint32_t* index = reinterpret_cast<__ubuf__ uint32_t*>(q);
    __ubuf__ uint8_t* mask = reinterpret_cast<__ubuf__ uint8_t*>(q + indexElements);
    __ubuf__ uint32_t* addresses = reinterpret_cast<__ubuf__ uint32_t*>(q + indexElements + 32U);
    __ubuf__ float* source = q + indexElements + 64U;
    if constexpr (kPrepare) {
        copy_gm_to_ubuf(
            reinterpret_cast<__ubuf__ float*>(index),
            reinterpret_cast<__gm__ float*>(g_hermitianGather.index[pitch][kLower] + first), 0U, 1U, elements / 8U, 0U,
            0U);
        copy_gm_to_ubuf_align_b8(
            mask, reinterpret_cast<__gm__ uint8_t*>(g_hermitianGather.conjugate[pitch][kLower] + first / 64U), 0U, 1U,
            repeats * sizeof(uint64_t), 0U, 0U, 0U, 0U);
        // Gather the imaginary diagonal from zero, never from the ignored input diagonal.
        vector_dup(source + sourceElements, 0.0f, 1U, 1U, 1U, 8U, 0U);
        LoadDenseRaw<false>(a, t.lda, t.dimA, t.dimA, 2U * k, 2U * k * k, k != static_cast<uint32_t>(t.dimA), source);
        return;
    }
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    vgather(reinterpret_cast<__ubuf__ uint32_t*>(p), index, reinterpret_cast<uint64_t>(source), 8U, repeats);
    pipe_barrier(PIPE_V);
    vmuls(ub.real, p, -1.0f, repeats, 1U, 1U, 8U, 8U);
    set_vector_mask(0ULL, kMaskAddressVectorMask);
    vector_dup(addresses, static_cast<uint32_t>(reinterpret_cast<uint64_t>(mask)), 1U, 1U, 1U, 8U, 0U);
    pipe_barrier(PIPE_V);
    set_cmpmask(addresses);
    pipe_barrier(PIPE_V);
    set_vector_mask(0ULL, kFullFp32Mask);
    vsel(p, ub.real, p, repeats, 1U, 1U, 1U, 8U, 8U, 8U, 2U);
}

template <bool kPlanes>
__aicore__ inline void PrepareDenseOperand(
    __gm__ float* b, const ChemmMixedTilingData& t, const TileWork& work, uint32_t kBase, __ubuf__ float* target,
    PackBuffers& ub)
{
    const uint32_t k = t.expandedK / 2U;
    const uint32_t validK = static_cast<uint32_t>(t.dimA) - kBase < k ? static_cast<uint32_t>(t.dimA) - kBase : k;
    const uint32_t first = kPlanes ? work.col + work.pFirst : work.row;
    const uint32_t extent = kPlanes ? t.n : t.m;
    const uint32_t span = kPlanes ? work.pRows : work.qRows;
    const uint32_t validSpan = first < extent ? (extent - first < span ? extent - first : span) : 0U;
    LoadDenseRaw<kPlanes, kPlanes ? EVENT_ID3 : EVENT_ID4>(
        b + 2U * ElementIndex(kPlanes ? kBase : first, kPlanes ? first : kBase, t.ldb), t.ldb,
        kPlanes ? validK : validSpan, kPlanes ? validSpan : validK, 2U * (kPlanes ? k : span), 2U * span * k,
        validK != k, kPlanes ? ub.bulk : target);
    if constexpr (kPlanes) {
        const uint8_t repeats = static_cast<uint8_t>(work.pRows * k / 32U);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(ub.real), reinterpret_cast<__ubuf__ uint32_t*>(ub.bulk), nullptr,
            repeats, 1U, 1U, 8U, 8U);
        vreducev2(
            reinterpret_cast<__ubuf__ uint32_t*>(ub.imag), reinterpret_cast<__ubuf__ uint32_t*>(ub.bulk), nullptr,
            repeats, 1U, 2U, 8U, 8U);
        pipe_barrier(PIPE_V);
        CopyPlaneRows(target, ub.real, work.pRows, k, k, t.expandedK);
        CopyPlaneRows(target + k, ub.imag, work.pRows, k, k, t.expandedK);
    }
}

template <bool kRight, bool kLower>
__aicore__ inline void PackP(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& t, const TileWork& work, uint32_t kBase,
    __ubuf__ float* p, __ubuf__ float* q, PackBuffers& ub)
{
    if constexpr (!kRight) {
        if (work.pRows != 0U) {
            PrepareDenseOperand<true>(b, t, work, kBase, p, ub);
        }
        return;
    }
    const uint32_t k = t.expandedK / 2U;
    if (t.dimA <= static_cast<int32_t>(CHEMM_K_FEED)) {
        PackResidentP<kLower, true>(a, t, work, p, q, ub);
        PrepareDenseOperand<false>(b, t, work, kBase, ub.bulk, ub);
        PackResidentP<kLower, false>(a, t, work, p, q, ub);
        return;
    }
    PrepareDenseOperand<false>(b, t, work, kBase, ub.bulk, ub);
    for (uint32_t col = 0U; col < work.pRows; col += kMicro) {
        for (uint32_t localK = 0U; localK < k; localK += kMicro) {
            // A(k,j)=conj(A(j,k)); decoding in destination order avoids a second transpose.
            ReadHermitian<kLower, true>(a, t, work.col + work.pFirst + col, kBase + localK, ub);
            const uint32_t span = k - localK < kMicro ? k - localK : kMicro;
            CopyPlaneRows(p + col * t.expandedK + localK, ub.real, kMicro, span, kMicro, t.expandedK);
            CopyPlaneRows(p + col * t.expandedK + k + localK, ub.imag, kMicro, span, kMicro, t.expandedK);
        }
    }
}

__aicore__ inline void ExpandQ(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* q, uint32_t rows, uint32_t span, uint32_t k)
{
    const uint32_t stride = 4U * k;
    CopyPlaneRows(q, real, rows, span, kMicro, stride);
    CopyPlaneRows(q + 2U * k, imag, rows, span, kMicro, stride);
    CopyPlaneRows(q + 3U * k, real, rows, span, kMicro, stride);
    pipe_barrier(PIPE_V);
    vmuls(imag, imag, -1.0f, static_cast<uint8_t>(rows / 4U), 1U, 1U, 8U, 8U);
    pipe_barrier(PIPE_V);
    CopyPlaneRows(q + k, imag, rows, span, kMicro, stride);
}

__aicore__ inline void PackDenseQ(
    __gm__ float* b, const ChemmMixedTilingData& t, const TileWork& work, uint32_t kBase, bool preloaded,
    __ubuf__ float* q, PackBuffers& ub)
{
    const uint32_t k = t.expandedK / 2U;
    const uint32_t rawStride = 2U * work.qRows;
    if (!preloaded) {
        PrepareDenseOperand<false>(b, t, work, kBase, ub.bulk, ub);
    }
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
    for (uint32_t offset = t.rawK; offset < t.expandedK; offset += kMicro) {
        const uint32_t first = offset < k ? offset : offset - k;
        const uint32_t second = offset + 8U < k ? offset + 8U : offset + 8U - k;
        __ubuf__ float* packed = q + (offset - t.rawK) * rawStride;
        uint64_t src[16];
        uint64_t dst[16];
#pragma unroll
        for (uint32_t lane = 0U; lane < 8U; ++lane) {
            src[lane] = reinterpret_cast<uint64_t>(ub.bulk + (first + lane) * rawStride);
            src[8U + lane] = reinterpret_cast<uint64_t>(ub.bulk + (second + lane) * rawStride);
            dst[2U * lane] = reinterpret_cast<uint64_t>(packed + (offset < k ? lane : (lane ^ 1U)) * 8U);
            dst[2U * lane + 1U] =
                reinterpret_cast<uint64_t>(packed + 8U * rawStride + (offset + 8U < k ? lane : (lane ^ 1U)) * 8U);
        }
        set_va_reg_sb(VA0, dst);
        set_va_reg_sb(VA1, dst + 8U);
        set_va_reg_sb(VA2, src);
        set_va_reg_sb(VA3, src + 8U);
        scatter_vnchwconv_b32(VA0, VA2, work.qRows / 4U, 8U, 1U);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask(0ULL, 0x00ff00ff00ff00ffULL);
    vmuls(q + (k - t.rawK) * rawStride, q + (k - t.rawK) * rawStride, -1.0f, k * rawStride / 64U, 1U, 1U, 8U, 8U);
    set_vector_mask(0ULL, kFullFp32Mask);
}

template <bool kRight, bool kLower>
__aicore__ inline void PackQ(
    __gm__ float* a, __gm__ float* b, const ChemmMixedTilingData& t, const TileWork& work, uint32_t kBase,
    bool preloaded, __ubuf__ float* q, PackBuffers& ub)
{
    if constexpr (kRight) {
        PackDenseQ(b, t, work, kBase, preloaded, q, ub);
        return;
    }
    const uint32_t k = t.expandedK / 2U;
    for (uint32_t row = 0U; row < work.qRows; row += kMicro) {
        for (uint32_t localK = 0U; localK < k; localK += kMicro) {
            ReadHermitian<kLower>(a, t, work.row + row, kBase + localK, ub);
            const uint32_t span = k - localK < kMicro ? k - localK : kMicro;
            ExpandQ(ub.real, ub.imag, q + 2U * row * t.expandedK + localK, kMicro, span, k);
        }
    }
}

template <bool kSingleColumn = false>
__aicore__ inline TileWork WorkForTile(const ChemmMixedTilingData& t, uint32_t tile, uint32_t half)
{
    const uint32_t col = kSingleColumn ? 0U : tile / t.tileRows;
    const uint32_t first = t.panelM / (2U * kMicro) * kMicro;
    const uint32_t local = tile - col * t.tileRows;
    return {
        local * t.rowTile + half * (t.rowTile / 2U), col * t.panelM, half * first,
        half == 0U ? first : t.panelM - first, t.rowTile / 2U};
}

template <bool kRight, bool kStationary = false>
__aicore__ inline void PublishInput(
    __gm__ float* base, const ChemmMixedTilingData& t, const TileWork& work, uint32_t job, uint32_t feed, uint32_t half,
    __ubuf__ float* p, __ubuf__ float* q)
{
    const uint32_t slot = job % CHEMM_INPUT_RING;
    __gm__ float* input = base + slot * t.inputSlotStride;
    __gm__ float* pDst = base + t.pCacheOffset + feed * t.pElements;
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    if constexpr (kStationary) {
        if (work.pRows != 0U) {
            copy_ubuf_to_gm(pDst + work.pFirst * t.expandedK, p, 0U, 1U, work.pRows * t.expandedK / 8U, 0U, 0U);
        }
        if (t.rowOwned == 0U) {
            ffts_cross_core_sync(PIPE_MTE3, 1U | (2U << 4U) | (7U << 8U));
        }
        return;
    }
    if constexpr (kRight) {
        copy_ubuf_to_gm(
            input + half * work.qRows * 16U, q, 0U, (t.expandedK - t.rawK) / 8U, 2U * work.qRows, 0U, 2U * work.qRows);
    } else {
        copy_ubuf_to_gm(input + half * t.qElements / 2U, q, 0U, 1U, t.qElements / 16U, 0U, 0U);
    }
    set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(job & 1U));
    ffts_cross_core_sync(PIPE_MTE3, 1U | (2U << 4U) | (slot << 8U));
}

template <bool kRight, bool kLower>
__aicore__ inline void ProduceTiles(
    __gm__ float* a, __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, uint32_t begin, uint32_t end,
    uint32_t half, PackBuffers& ub)
{
    if (t.rowOwned == 0U) {
        copy_gm_to_ubuf_align_b8(ub.masks, reinterpret_cast<__gm__ uint8_t*>(g_packMasks), 0U, 1U, 96U, 0U, 0U, 0U, 0U);
    }
    uint32_t job = 0U;
    for (uint32_t tile = begin; tile < end; ++tile) {
        TileWork work = t.rowOwned != 0U ? WorkForTile<true>(t, tile, half) : WorkForTile(t, tile, half);
        const bool newColumn = tile == begin || tile % t.tileRows == 0U;
        if (newColumn && tile != begin && t.rowOwned == 0U) {
            wait_flag_dev(6U);
        }
        const bool packP = t.rowOwned != 0U ? tile == begin : newColumn;
        if (packP && t.rowOwned != 0U) {
            work.col = 0U;
            work.pFirst = half * kMicro;
            work.pRows = kMicro;
        }
        for (uint32_t feed = 0U; feed < t.kFeeds; ++feed, ++job) {
            if (job >= CHEMM_INPUT_RING) {
                wait_flag_dev(3U + job % CHEMM_INPUT_RING);
            }
            if (job >= 2U) {
                wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(job & 1U));
            }
            __ubuf__ float* p = ub.input + (job & 1U) * kInputBankElements;
            __ubuf__ float* q = p + CHEMM_MAX_P_ELEMENTS / 2U;
            if (packP) {
                PackP<kRight, kLower>(a, b, t, work, feed * CHEMM_K_FEED, p, q, ub);
                PublishInput<kRight, true>(base, t, work, job, feed, half, p, q);
            }
            PackQ<kRight, kLower>(a, b, t, work, feed * CHEMM_K_FEED, packP, q, ub);
            PublishInput<kRight>(base, t, work, job, feed, half, p, q);
        }
    }
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    if (job > 1U) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }
}

__aicore__ inline void RunProducers(
    __gm__ float* a, __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, uint32_t begin, uint32_t end,
    uint32_t half, __ubuf__ float* scratch)
{
    PackBuffers ub{
        scratch,
        scratch + 2U * kInputBankElements,
        scratch + 2U * kInputBankElements + kBulkElements,
        scratch + 2U * kInputBankElements + kBulkElements + kComplexTileFloats,
        scratch + 2U * kInputBankElements + kBulkElements + kComplexTileFloats + kPlaneElements,
        scratch + 2U * kInputBankElements + kBulkElements + kComplexTileFloats + 2U * kPlaneElements,
        scratch + 2U * kInputBankElements + kBulkElements + kComplexTileFloats + 2U * kPlaneElements + kMicroElements,
        reinterpret_cast<__ubuf__ uint8_t*>(scratch + kScratchElements - 24U)};
    set_atomic_none();
    set_mask_norm();
    set_vector_mask(0ULL, kFullFp32Mask);
    const bool right = (t.kernelFlags & CHEMM_FLAG_RIGHT) != 0U;
    const bool lower = (t.kernelFlags & CHEMM_FLAG_LOWER) != 0U;
    if (right && lower) {
        ProduceTiles<true, true>(a, b, base, t, begin, end, half, ub);
    } else if (right) {
        ProduceTiles<true, false>(a, b, base, t, begin, end, half, ub);
    } else if (lower) {
        ProduceTiles<false, true>(a, b, base, t, begin, end, half, ub);
    } else {
        ProduceTiles<false, false>(a, b, base, t, begin, end, half, ub);
    }
}

template <bool kResident>
__aicore__ inline void PrepareStationaryFeed(
    __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, const FeedWork& work, __cbuf__ float* pL1,
    __cbuf__ float* qL1, __cb__ float* qL0)
{
    if (!kResident && t.rawK != 0U) {
        __cbuf__ float* raw = qL1 + 2U * work.rows * (t.expandedK - t.rawK);
        const uint32_t rows = static_cast<uint32_t>(t.m) - work.row < work.rows ? t.m - work.row : work.rows;
        copy_gm_to_cbuf_multi_nd2nz_b32s(
            raw, b + 2U * ElementIndex(work.row, work.feed * CHEMM_K_FEED, static_cast<uint32_t>(t.ldb)), 0U, 1U,
            t.rawK, 2U * rows, 0U, 2U * static_cast<uint32_t>(t.ldb), t.rawK, 1U, 0U);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        load_cbuf_to_cb_transpose(qL0, raw, 0U, work.rows / 8U, 1U, 0U, false, work.rows / 8U - 1U);
    }
    if (!kResident && work.newColumn) {
        wait_flag_dev(7U);
    }
    if ((!kResident && t.residentP == 0U) || work.newColumn) {
        __gm__ float* source = base + t.pCacheOffset + work.feed * t.pElements;
        const uint32_t columns = kResident ? 2U * kMicro : t.panelM;
        copy_gm_to_cbuf_multi_nd2nz_b32s(pL1, source, 0U, 1U, columns, t.expandedK, 0U, t.expandedK, columns, 1U, 0U);
    }
}

template <bool kTail, bool kResident = false>
__aicore__ inline void LoadMatrixFeed(
    __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, const FeedWork& work, __cbuf__ float* pL1,
    __cbuf__ float* qL1, __cb__ float* qL0)
{
    const event_t event = static_cast<event_t>(work.job & 1U);
    if constexpr (!kTail) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, event);
        wait_flag(PIPE_M, PIPE_MTE1, static_cast<event_t>(kResident ? 0U : work.job & t.l0BankMask));
        if constexpr (kResident) {
            // Q-ready also covers the earlier P store on the same MTE3 pipe.
            wait_flag_dev(work.job % CHEMM_INPUT_RING);
        }
        PrepareStationaryFeed<kResident>(b, base, t, work, pL1, qL1, qL0);
        if constexpr (kResident) {
            copy_gm_to_cbuf(
                qL1, base + (work.job % CHEMM_INPUT_RING) * t.inputSlotStride, 0U, 1U, work.rows * t.expandedK / 4U, 0U,
                0U, PAD_NONE);
            if (work.job + CHEMM_INPUT_RING < work.totalJobs) {
                ffts_cross_core_sync(PIPE_MTE2, 1U | (2U << 4U) | ((3U + work.job % CHEMM_INPUT_RING) << 8U));
            }
        }
    } else {
        const uint32_t slot = work.job % CHEMM_INPUT_RING;
        wait_flag_dev(slot);
        __gm__ float* input = base + slot * t.inputSlotStride;
        if (kResident || (t.kernelFlags & CHEMM_FLAG_RIGHT) != 0U) {
            const uint32_t k = kResident ? t.expandedK : t.expandedK - t.rawK;
            copy_gm_to_cbuf(qL1, input, 0U, 1U, work.rows * k / 4U, 0U, 0U, PAD_NONE);
        } else {
            copy_gm_to_cbuf_multi_nd2nz_b32s(
                qL1, input, 0U, 1U, 2U * t.rowTile, t.expandedK, 0U, t.expandedK, 2U * t.rowTile, 1U, 0U);
        }
        if (work.job + CHEMM_INPUT_RING < work.totalJobs) {
            ffts_cross_core_sync(PIPE_MTE2, 1U | (2U << 4U) | ((3U + slot) << 8U));
        }
        if (work.releaseP && work.feed + 1U == t.kFeeds) {
            ffts_cross_core_sync(PIPE_MTE2, 1U | (2U << 4U) | (6U << 8U));
        }
    }
    set_flag(PIPE_MTE2, PIPE_MTE1, event);
}

template <bool kRaw>
__aicore__ inline void ComputeMatrixFeed(
    __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, const FeedWork& work, __cbuf__ float* pL1,
    __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* cL0)
{
    const event_t bank = static_cast<event_t>(work.job & t.l0BankMask);
    const event_t inputBank = static_cast<event_t>(work.job & 1U);
    wait_flag(PIPE_MTE2, PIPE_MTE1, inputBank);
    const uint32_t fractals = t.panelM / kMicro;
    constexpr uint32_t kPrefix = kRaw ? kMicro : 0U;
    const uint32_t tailK = t.expandedK - kPrefix;
    for (uint32_t row = 0U; row < fractals; ++row) {
        if constexpr (kRaw) {
            load_cbuf_to_ca(
                pL0 + row * kPrefix * kMicro, pL1 + row * 128U, 0U, kPrefix / 8U, fractals, 0U, 0U, false, inc);
        }
        load_cbuf_to_ca(
            pL0 + t.panelM * kPrefix + row * tailK * kMicro, pL1 + t.panelM * kPrefix + row * 128U, 0U, tailK / 8U,
            fractals, 0U, 0U, false, inc);
    }
    if constexpr (kRaw) {
        set_flag(PIPE_MTE1, PIPE_M, bank);
        wait_flag(PIPE_MTE1, PIPE_M, bank);
        mad(cL0, pL0, qL0, t.panelM, kPrefix, 2U * work.rows, 0U, false, false, work.feed == 0U);
        set_flag(PIPE_M, PIPE_MTE1, bank);
    }
    // The prefix is already running when Scalar waits for Q's producer.
    LoadMatrixFeed<true>(b, base, t, work, pL1, qL1, qL0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, inputBank);
    if constexpr (kRaw) {
        wait_flag(PIPE_M, PIPE_MTE1, bank);
    }
    // Prefix MMAD has released L0B; the tail reuses the same bank.
    load_cbuf_to_cb(qL0, qL1, 0U, work.rows * tailK / 64U, 1U, 0U, 0U, false, inc);
    set_flag(PIPE_MTE1, PIPE_MTE2, inputBank);
    set_flag(PIPE_MTE1, PIPE_M, bank);
    wait_flag(PIPE_MTE1, PIPE_M, bank);
    mad(cL0, pL0 + t.panelM * kPrefix, qL0, t.panelM, tailK, 2U * work.rows, 0U, false, false,
        !kRaw && work.feed == 0U);
    set_flag(PIPE_M, PIPE_MTE1, bank);
}

__aicore__ inline void StoreMatrixTile(
    __gm__ float* c, const ChemmMixedTilingData& t, uint32_t row, uint32_t col, uint32_t span, uint32_t bank,
    __cc__ float* result)
{
    const uint32_t rows = static_cast<uint32_t>(t.m) - row < span ? t.m - row : span;
    const uint32_t cols = static_cast<uint32_t>(t.n) - col < t.panelM ? static_cast<uint32_t>(t.n) - col : t.panelM;
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    copy_matrix_cc_to_gm(
        c + 2U * ElementIndex(row, col, static_cast<uint32_t>(t.ldc)), result, 0U, 2U * rows, cols,
        2U * static_cast<uint32_t>(t.ldc), t.panelM, 0U, QuantMode_t::NoQuant, 0U, false, true);
    set_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(bank));
}

__aicore__ inline void ComputeResidentTile(
    __gm__ float* b, __gm__ float* base, const ChemmMixedTilingData& t, const FeedWork& work, __cbuf__ float* pL1,
    __cbuf__ float* qL1, __ca__ float* pL0, __cb__ float* qL0, __cc__ float* result)
{
    const event_t inputBank = static_cast<event_t>(work.job & 1U);
    wait_flag(PIPE_MTE2, PIPE_MTE1, inputBank);
    load_cbuf_to_cb(qL0, qL1, 0U, work.rows * t.expandedK / 64U, 1U, 0U, 0U, false, inc);
    set_flag(PIPE_MTE1, PIPE_MTE2, inputBank);
    if (work.newColumn) {
        for (uint32_t row = 0U; row < t.panelM / kMicro; ++row) {
            load_cbuf_to_ca(
                pL0 + row * t.expandedK * kMicro, pL1 + row * 128U, 0U, t.expandedK / 8U, 2U, 0U, 0U, false, inc);
        }
    }
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    mad(result, pL0, qL0, t.panelM, t.expandedK, 2U * work.rows, 0U, false, false, true);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
}

template <bool kRaw, bool kResident = false>
__aicore__ inline void RunConsumer(
    __gm__ float* b, __gm__ float* c, __gm__ float* base, const ChemmMixedTilingData& t, uint32_t begin, uint32_t end,
    __cbuf__ float* l1, __ca__ float* p, __cb__ float* q, __cc__ float* result)
{
    const uint32_t feeds = kResident ? 1U : t.kFeeds;
    const uint32_t pSpan = kResident ? 2U * kMicro * t.expandedK : (t.residentP != 0U ? t.kFeeds : 2U) * t.pElements;
    const uint32_t totalJobs = (end - begin) * feeds;
    const uint32_t lastColumn = kResident ? 0U : (end - 1U) / t.tileRows;
    uint32_t job = 0U;
    for (uint32_t tile = begin, ordinal = 0U; tile < end; ++tile, ++ordinal) {
        const uint32_t cBank = ordinal & 1U;
        __cc__ float* output = result + cBank * t.panelM * 2U * t.rowTile;
        if (ordinal >= 2U) {
            wait_flag(PIPE_FIX, PIPE_M, static_cast<event_t>(cBank));
        }
        const bool newColumn = tile == begin || (!kResident && tile % t.tileRows == 0U);
        if (!kResident && t.residentP != 0U && newColumn && tile != begin) {
            // The stationary L1 panel has readers in both supply banks.
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        }
        const bool releaseP = !kResident && t.rowOwned == 0U && tile / t.tileRows != lastColumn &&
                              (t.residentP != 0U ? newColumn : (tile + 1U) % t.tileRows == 0U);
        const TileWork tileWork = WorkForTile<kResident>(t, tile, 0U);
        for (uint32_t feed = 0U; feed < feeds; ++feed, ++job) {
            const FeedWork work{tileWork.row, tileWork.col, 2U * tileWork.qRows, job,
                                totalJobs,    feed,         newColumn,           releaseP};
            __cbuf__ float* pL1 = l1 + (kResident ? 0U : t.residentP != 0U ? feed : (job & 1U)) * t.pElements;
            __cbuf__ float* qL1 = l1 + pSpan + (job & 1U) * t.qElements;
            const uint32_t bank = kResident ? 0U : job & t.l0BankMask;
            LoadMatrixFeed<false, kResident>(b, base, t, work, pL1, qL1, q + bank * t.qElements);
            if constexpr (kResident) {
                ComputeResidentTile(b, base, t, work, pL1, qL1, p, q, output);
            } else {
                ComputeMatrixFeed<kRaw>(
                    b, base, t, work, pL1, qL1, p + bank * t.pElements, q + bank * t.qElements, output);
            }
        }
        StoreMatrixTile(c, t, tileWork.row, tileWork.col, 2U * tileWork.qRows, cBank, output);
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    if (end - begin > 1U) {
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
}

} // namespace

__schedmode__(1) __global__ __mix__(1, 2) void chemm_mixed_kernel(
    __gm__ uint8_t* gmA, __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, __gm__ uint8_t* gmWorkspace,
    ChemmMixedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ float* a = reinterpret_cast<__gm__ float*>(gmA);
    __gm__ float* b = reinterpret_cast<__gm__ float*>(gmB);
    __gm__ float* c = reinterpret_cast<__gm__ float*>(gmC);
    // Raw block_idx is the MIX group; the C++ wrapper expands AIV indices.
    const uint32_t core = static_cast<uint32_t>(get_block_idx());
    const uint32_t extra = core < tiling.extraTiles ? core : tiling.extraTiles;
    const uint32_t begin = core * tiling.tilesPerCore + extra;
    const uint32_t limit = begin + tiling.tilesPerCore + (core < tiling.extraTiles ? 1U : 0U);
    const uint32_t end = limit < tiling.workTiles ? limit : tiling.workTiles;
    __gm__ float* base =
        reinterpret_cast<__gm__ float*>(gmWorkspace) + static_cast<uint64_t>(core) * tiling.inputCoreStride;
    if ASCEND_IS_AIC {
        set_ctrl(sbitset1(sbitset0(get_ctrl(), 46U), 51U));
        set_nd_para(1ULL);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        __cbuf__ float l1[CHEMM_L1_ELEMENTS] __attribute__((aligned(512)));
        __ca__ float p[CHEMM_L0_ELEMENTS] __attribute__((aligned(512)));
        __cb__ float q[CHEMM_L0_ELEMENTS] __attribute__((aligned(512)));
        __cc__ float result[2U * CHEMM_L0_ELEMENTS] __attribute__((aligned(512)));
        if (tiling.rowOwned != 0U) {
            RunConsumer<false, true>(b, c, base, tiling, begin, end, l1, p, q, result);
        } else if (tiling.rawK != 0U) {
            RunConsumer<true>(b, c, base, tiling, begin, end, l1, p, q, result);
        } else {
            RunConsumer<false>(b, c, base, tiling, begin, end, l1, p, q, result);
        }
    }
    if ASCEND_IS_AIV {
        __ubuf__ float scratch[kScratchElements] __attribute__((aligned(32)));
        RunProducers(a, b, base, tiling, begin, end, static_cast<uint32_t>(get_subblockid()), scratch);
    }
}

void chemm_mixed_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* workspace, const ChemmMixedTilingData& tiling, void* stream)
{
    chemm_mixed_kernel<<<tiling.blockDim, nullptr, stream>>>(a, b, c, workspace, tiling);
}
