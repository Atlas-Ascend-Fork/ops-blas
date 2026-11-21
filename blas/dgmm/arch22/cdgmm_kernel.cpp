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
 * \file cdgmm_kernel.cpp
 * \brief Device-side kernel for aclblasCdgmm (arch22).
 *        Row-major storage.  Complex elements are stored as interleaved
 *        float pairs (real, imag).
 *
 *        mode=LEFT: C[i,j] = x[i] * A[i,j]
 *            Each row is contiguous and multiplied by a single complex
 *            scalar x[i].  Reuses the original scalar-vector complex
 *            multiply path (GatherMask deinterleave, muls_v, Gather
 *            re-interleave) with ping-pong UB buffers.
 */

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#include "cdgmm_kernel.h"

// ==========================================================================
// UB copy helpers (kept from original colwise_mul, renamed)
// ==========================================================================

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_uint32(
    AscendC::LocalTensor<uint32_t> dst,
    AscendC::GlobalTensor<uint32_t> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(uint32_t);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, uint32_t>(dst, src,
                    0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(dst, src,
                    0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(
    AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(dst, src,
                   0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

// ==========================================================================
// Core compute: scalar complex multiply on a contiguous block.
// (Renamed from colwise_mul_compute_aiv, logic preserved.)
//
// Multiplies a contiguous block of interleaved complex floats by a complex
// scalar (s_real, s_imag):
//   out = a * (s_real + s_imag*i)
// ==========================================================================
__aicore__ __inline__ __attribute__((always_inline)) void cdgmm_compute_aiv(
    AscendC::GlobalTensor<float> gm_in,
    AscendC::GlobalTensor<float> gm_out,
    AscendC::LocalTensor<float> ub_in,
    AscendC::LocalTensor<float> ub_out,
    AscendC::LocalTensor<uint32_t> ub_offset,
    float s_real, float s_imag, uint32_t copy_len, uint32_t event_id)
{
    uint32_t len = copy_len;
    uint32_t repeatTime = (len + 63) / 64;
    uint32_t computeRepeat = (len / 2 + 63) / 64;

    uint32_t real_offset = 0;
    uint32_t imag_offset = 32 * 1024 / sizeof(float) / 2;

    AscendC::LocalTensor<float> ub_out_real = ub_out;
    AscendC::LocalTensor<float> ub_out_imag = ub_out[imag_offset];

    copy_vec_gm2ub(ub_in, gm_in, copy_len);

    SET_FLAG(MTE2, V, event_id);
    WAIT_FLAG(MTE2, V, event_id);

    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;

    AscendC::GatherMask<float>(ub_out_real, ub_in, 1, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(ub_out_imag, ub_in, 2, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    PIPE_BARRIER(V);

    // R * R
    muls_v<ArchType::ASCEND_V220, float>(ub_in, ub_out_real, s_real, computeRepeat, 1, 1, 8, 8);

    // R * I
    muls_v<ArchType::ASCEND_V220, float>(ub_in[imag_offset], ub_out_real, s_imag, computeRepeat, 1, 1, 8, 8);

    // I * I
    muls_v<ArchType::ASCEND_V220, float>(ub_out_real, ub_out_imag, s_imag, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * R - I * I
    sub_v<ArchType::ASCEND_V220, float>(ub_in, ub_in, ub_out_real, computeRepeat, 1, 1, 1, 8, 8, 8);

    // I * R
    muls_v<ArchType::ASCEND_V220, float>(ub_out_imag, ub_out_imag, s_real, computeRepeat, 1, 1, 8, 8);

    PIPE_BARRIER(V);
    // R * I + I * R
    add_v<ArchType::ASCEND_V220, float>(
        ub_in[imag_offset], ub_out_imag, ub_in[imag_offset], computeRepeat, 1, 1, 1, 8, 8, 8);

    PIPE_BARRIER(V);

    AscendC::Gather(ub_out, ub_in, ub_offset, (uint32_t)0, repeatTime * 64);
    PIPE_BARRIER(ALL);

    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    copy_vec_ub2gm(gm_out, ub_out, copy_len);
}

// ==========================================================================
// Pure computation: x index for given row with signed incx (Task 4)
// ==========================================================================
__aicore__ __inline__ __attribute__((always_inline)) int64_t CalcCdgmmXIndex(
    uint32_t row, uint32_t m, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(row) * static_cast<int64_t>(incx);
    }
    int64_t absIncx = -static_cast<int64_t>(incx);
    return (static_cast<int64_t>(m) - 1 - static_cast<int64_t>(row)) * absIncx;
}

// ==========================================================================
// UB buffer context for LEFT-mode ping-pong processing
// ==========================================================================
struct CdgmmLeftUbCtx {
    AscendC::LocalTensor<float> ubOutPing;
    AscendC::LocalTensor<float> ubOutPong;
    AscendC::LocalTensor<float> ubInPing;
    AscendC::LocalTensor<float> ubInPong;
    AscendC::LocalTensor<uint32_t> ubOffset;
    uint32_t pingFlag;
    uint32_t maxDataCount;
    uint32_t repeatTime;
    uint32_t remainNum;
};

// ==========================================================================
// Process full chunks for a single row (Task 4)
// ==========================================================================
__aicore__ __inline__ __attribute__((always_inline)) void ProcessCdgmmFullChunks(
    AscendC::GlobalTensor<float> gm_a,
    AscendC::GlobalTensor<float> gm_c,
    AscendC::GlobalTensor<uint32_t> gm_aug,
    float s_real, float s_imag,
    uint64_t aBase, uint64_t cBase,
    CdgmmLeftUbCtx& ctx)
{
    if (ctx.repeatTime == 0) {
        return;
    }

    uint64_t currOffset = 0;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    for (uint32_t i = 0; i < ctx.repeatTime; i++) {
        auto ubIn = ctx.pingFlag ? ctx.ubInPing : ctx.ubInPong;
        auto ubOut = ctx.pingFlag ? ctx.ubOutPing : ctx.ubOutPong;
        auto eventId = ctx.pingFlag ? EVENT_ID0 : EVENT_ID1;

        WAIT_FLAG(MTE3, MTE2, eventId);

        cdgmm_compute_aiv(gm_a[aBase + currOffset], gm_c[cBase + currOffset],
                          ubIn, ubOut, ctx.ubOffset, s_real, s_imag,
                          ctx.maxDataCount, eventId);

        SET_FLAG(MTE3, MTE2, eventId);
        currOffset += ctx.maxDataCount;
        ctx.pingFlag = 1 - ctx.pingFlag;
    }
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// ==========================================================================
// Process tail chunk for a single row (Task 4)
// ==========================================================================
__aicore__ __inline__ __attribute__((always_inline)) void ProcessCdgmmTailChunk(
    AscendC::GlobalTensor<float> gm_a,
    AscendC::GlobalTensor<float> gm_c,
    AscendC::GlobalTensor<uint32_t> gm_aug,
    float s_real, float s_imag,
    uint64_t aBase, uint64_t cBase, uint64_t prevOffset,
    CdgmmLeftUbCtx& ctx)
{
    if (ctx.remainNum == 0) {
        return;
    }

    uint64_t currOffset = prevOffset;
    SET_FLAG(MTE3, MTE2, EVENT_ID0);
    SET_FLAG(MTE3, MTE2, EVENT_ID1);

    auto ubIn = ctx.pingFlag ? ctx.ubInPing : ctx.ubInPong;
    auto ubOut = ctx.pingFlag ? ctx.ubOutPing : ctx.ubOutPong;
    auto eventId = ctx.pingFlag ? EVENT_ID0 : EVENT_ID1;
    WAIT_FLAG(MTE3, MTE2, eventId);

    cdgmm_compute_aiv(gm_a[aBase + currOffset], gm_c[cBase + currOffset],
                      ubIn, ubOut, ctx.ubOffset, s_real, s_imag,
                      ctx.remainNum, eventId);

    SET_FLAG(MTE3, MTE2, eventId);
    ctx.pingFlag = 1 - ctx.pingFlag;
    WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
    WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
}

// ==========================================================================
// LEFT mode row processing (ping-pong, migrated from colwise_mul_aiv)
//
// Row-major: row i starts at A_float[2*i*lda] and C_float[2*i*ldc].
// Each row has n complex elements = 2*n float elements.
// x[i] is read using incx stride: xIndex = (incx>0) ? i*incx : (m-1-i)*|incx|
// ==========================================================================
__aicore__ __inline__ __attribute__((always_inline)) void cdgmm_left_aiv(
    AscendC::GlobalTensor<float> gm_a,
    AscendC::GlobalTensor<float> gm_x,
    AscendC::GlobalTensor<uint32_t> gm_aug,
    AscendC::GlobalTensor<float> gm_c,
    uint32_t m, uint32_t n, int32_t incx,
    uint32_t lda, uint32_t ldc,
    uint32_t startRow, uint32_t rowCount)
{
    // ub 192kb
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    CdgmmLeftUbCtx ctx;
    ctx.ubOutPing = buf.GetBuffer<BufferType::ASCEND_UB, float>(0 * 1024);
    ctx.ubOutPong = buf.GetBuffer<BufferType::ASCEND_UB, float>(32 * 1024);
    ctx.ubInPing = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 1024);
    ctx.ubInPong = buf.GetBuffer<BufferType::ASCEND_UB, float>(96 * 1024);
    ctx.ubOffset = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(128 * 1024);
    ctx.pingFlag = 1;
    ctx.maxDataCount = 32 * 1024 / sizeof(float);

    uint32_t rowFloats = 2 * n;
    ctx.repeatTime = rowFloats / ctx.maxDataCount;
    ctx.remainNum = rowFloats % ctx.maxDataCount;

    // prepare offset
    copy_vec_gm2ub_uint32(ctx.ubOffset, gm_aug, ctx.maxDataCount);

    SET_FLAG(MTE2, V, EVENT_ID0);
    WAIT_FLAG(MTE2, V, EVENT_ID0);
    SET_FLAG(MTE2, V, EVENT_ID1);
    WAIT_FLAG(MTE2, V, EVENT_ID1);

    if (rowCount == 0) {
        return;
    }

    for (uint32_t localRow = 0; localRow < rowCount; localRow++) {
        uint32_t row = startRow + localRow;

        uint64_t aFloatOffset = 2ULL * static_cast<uint64_t>(row) * lda;
        uint64_t cFloatOffset = 2ULL * static_cast<uint64_t>(row) * ldc;

        int64_t xIndex = CalcCdgmmXIndex(row, m, incx);

        float s_real = gm_x.GetValue(2ULL * static_cast<uint64_t>(xIndex));
        float s_imag = gm_x.GetValue(2ULL * static_cast<uint64_t>(xIndex) + 1);

        SET_FLAG(S, V, EVENT_ID0);
        WAIT_FLAG(S, V, EVENT_ID0);
        SET_FLAG(S, V, EVENT_ID1);
        WAIT_FLAG(S, V, EVENT_ID1);

        ProcessCdgmmFullChunks(gm_a, gm_c, gm_aug, s_real, s_imag,
                               aFloatOffset, cFloatOffset, ctx);

        uint64_t prevOffset = static_cast<uint64_t>(ctx.repeatTime) * ctx.maxDataCount;
        ProcessCdgmmTailChunk(gm_a, gm_c, gm_aug, s_real, s_imag,
                              aFloatOffset, cFloatOffset, prevOffset, ctx);
    }
    PIPE_BARRIER(ALL);
}

// ==========================================================================
// Kernel entry point
// ==========================================================================
__global__ __aicore__ __vector__ void cdgmm(GM_ADDR A, GM_ADDR x,
                                            GM_ADDR C, GM_ADDR aug,
                                            GM_ADDR workSpace, GM_ADDR tilingGm)
{
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();

    auto core_idx = AscendC::GetBlockIdx();

    auto* tiling = reinterpret_cast<__gm__ CdgmmTilingData*>(tilingGm);

    uint32_t mode = tiling->mode;
    uint32_t m = tiling->m;
    uint32_t n = tiling->n;
    int32_t incx = tiling->incx;
    uint32_t lda = tiling->lda;
    uint32_t ldc = tiling->ldc;

    uint32_t startRow = tiling->startRow[core_idx];
    uint32_t rowCount = tiling->rowCount[core_idx];

    if (rowCount == 0) {
        return;
    }

    AscendC::GlobalTensor<float> a_tensor;
    AscendC::GlobalTensor<float> x_tensor;
    AscendC::GlobalTensor<uint32_t> aug_tensor;
    AscendC::GlobalTensor<float> c_tensor;

    a_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
    x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    aug_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(aug));
    c_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C));

    // Only LEFT is supported; RIGHT is rejected by Host before kernel launch.
    if (mode == CDGMM_MODE_LEFT) {
        cdgmm_left_aiv(a_tensor, x_tensor, aug_tensor, c_tensor,
                       m, n, incx, lda, ldc, startRow, rowCount);
    }
}

// Wrapper function for host to call
void cdgmm_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR C,
                     GM_ADDR aug, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void* stream)
{
    cdgmm<<<numBlocks, nullptr, stream>>>(A, x, C, aug, workSpace, tilingGm);
}
